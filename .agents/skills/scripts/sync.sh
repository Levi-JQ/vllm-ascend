#!/bin/bash
# sync.sh - 通过 rsync 同步本地代码到远程 itask 机器
#
# 用法:
#   scripts/sync.sh <itask-name>                                          # 默认模式：同步整个 workspace（含 codebases 主分支，纯部署用）
#   scripts/sync.sh <itask-name> --task <task-name>                       # 任务模式：同步共享顶层文件 + 指定任务的代码/文档
#   scripts/sync.sh <itask-name> --task <task-name> --force               # 任务模式 + 强制同步（--delete 仅作用于该任务目录，不影响其他任务/顶层文件）
#   scripts/sync.sh <itask-name> --task <task-name> --link-dest-task <t>  # 任务模式 + 指定远端内容复用的引用任务（默认自动选远端 code 最新的任务）
#   scripts/sync.sh <itask-name> --task <task-name> --no-verify          # 任务模式 + 跳过同步后校验
#   scripts/sync.sh <itask-name> --port 27890                             # 指定 SSH 隧道端口
#
# 需要先建立 SSH 隧道: scripts/ssh_tunnel.sh connect <itask-name> [port]
# rsync 通过 localhost:SSH_PORT 连接远程机器
# 远程工作目录: /a3_inference/itask/workdir/yjq02324703/workspace（与本地工作空间目录名一致）
#
# 传输量控制（复用远端共享盘已有内容，避免重复传输）:
#   1. 任务模式只同步共享顶层文件 + 当前任务目录，不再推送其他任务目录与 profiling/logs 归档
#      （这些只应 远端→本地 下载归档，反向推送纯属浪费；排除规则均锚定到根，不误伤仓库内嵌套同名目录，
#        如 vllm/vllm/v1/kv_offload/tiering/p2p/data、vllm/examples/features/profiling）
#   2. 任务代码三仓库按内容比对（rsync -c，与 mtime 无关）并用 --link-dest 硬链接复用远端已有任务代码
#      （A3 共享盘同 NFS 可硬链接；同内容文件 0 传输，实测新任务 vllm-ascend 47MB→2.8MB、vllm →3.1MB）
#      构建产物（.so/_cann_ops_custom/_build_info.py 等）均 gitignore 不在源里，永远不会被链接共享
#   3. 引用任务默认自动选远端 code 目录 mtime 最新的任务；内容校验保证引用不合适只是多传、不会传错
#   4. 同步后自动校验：取 git ls-files 文件清单对本地与远端做 md5 比对，确认源码一致
#      （--no-verify 跳过；只校验 git 追踪的源文件，构建产物不在范围）
#
# 两种模式:
#   默认模式（无 --task）: 同步整个 workspace 含 codebases 主分支代码。用于不改代码的纯部署测试。
#   任务模式（--task <name>）: 只同步共享文件 + task/<name>/（code 三仓库 + 文档），
#     远端目录结构 task/<name>/code/{vllm,vllm-ascend,vllm-ascend-tools}，start-service 用 TASK=<name> 启动。
#     worktree 的 .git 文件指向本地路径（远程无效），任务模式自动排除所有 .git。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PORT=27890
DEFAULT_REMOTE_BASE="/a3_inference/itask/workdir/yjq02324703"

TASK_NAME="${1:?Usage: sync.sh <itask-name> [--task <task-name>] [--force] [--port PORT] [--link-dest-task <task>]}"
shift

FORCE_FLAG=""
SSH_PORT="$DEFAULT_PORT"
TASK_CODE_NAME=""
LINK_DEST_TASK=""
NO_VERIFY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)           FORCE_FLAG="--delete"; shift ;;
        --port)            SSH_PORT="${2:?--port requires a value}"; shift 2 ;;
        --task)            TASK_CODE_NAME="${2:?--task requires a value}"; shift 2 ;;
        --link-dest-task)  LINK_DEST_TASK="${2:?--link-dest-task requires a value}"; shift 2 ;;
        --no-verify)      NO_VERIFY=true; shift ;;
        *)                 echo "Unknown option: $1"; exit 1 ;;
    esac
done

REMOTE_WORKDIR="${REMOTE_WORKDIR:-${DEFAULT_REMOTE_BASE}/workspace}"
SSH_CMD="ssh -o StrictHostKeyChecking=no -p ${SSH_PORT}"

# 确保 vllm-ascend 的 catlass submodule 已初始化（worktree 不共享 submodule 工作树）
ensure_submodule() {
    local ascend_dir="$1"
    if [[ -d "$ascend_dir/.git" ]]; then
        (cd "$ascend_dir" && git submodule update --init csrc/third_party/catlass 2>/dev/null || true)
    fi
}

# ==============================================================================
# 同步后校验：本地 vs 远端源码 md5 一致性
# 用 git ls-files 取本地追踪文件清单（天然排除 .so/__pycache__/build 产物），
# 对同一组文件在远端跑 md5sum，sorted diff 为空即一致。
# 只校验 git 追踪的源文件；构建产物不在范围。--no-verify 可跳过。
# ==============================================================================
verify_sync() {
    local local_dir="$1"
    local remote_dir="$2"
    local label="$3"

    if ! git -C "$local_dir" rev-parse --is-inside-work-tree &>/dev/null; then
        echo "    ~ $label: 非 git 仓库，跳过校验"
        return 0
    fi

    local files
    files=$(git -C "$local_dir" ls-files 2>/dev/null) || {
        echo "    ~ $label: git ls-files 失败，跳过校验"
        return 0
    }
    # 只校验 rsync 实际传输的普通文件：跟踪的 symlink 会被 rsync 跳过
    # （skipping non-regular file），计入清单会造成永久假阳性
    files=$(cd "$local_dir" && printf '%s\n' "$files" | while IFS= read -r f; do
        [ -f "$f" ] && [ ! -L "$f" ] && printf '%s\n' "$f"
    done)
    [[ -z "$files" ]] && { echo "    ~ $label: 无追踪文件，跳过校验"; return 0; }

    local count
    count=$(echo "$files" | wc -l | tr -d ' ')

    # 本地清单：md5 + 文件名，按文件名排序（LC_ALL=C 固定字节序，避免 macOS/Linux collation 差异假阳性）
    local local_manifest
    local_manifest=$(cd "$local_dir" && echo "$files" | tr '\n' '\0' | xargs -0 md5sum 2>/dev/null | LC_ALL=C sort -k2 || true)

    # 远端清单：同一组文件在远端跑 md5sum（经 SSH stdin 传文件列表）
    local remote_manifest
    remote_manifest=$(echo "$files" | $SSH_CMD root@localhost \
        "cd '$remote_dir' && tr '\n' '\0' | xargs -0 md5sum 2>/dev/null | LC_ALL=C sort -k2" 2>/dev/null || true)

    if [[ "$local_manifest" == "$remote_manifest" ]]; then
        echo "    ✓ $label: $count files verified"
        return 0
    else
        echo "    ✗ $label: MISMATCH — 本地与远端不一致的文件：" >&2
        diff <(echo "$local_manifest") <(echo "$remote_manifest") | head -30 >&2 || true
        return 1
    fi
}

# ==============================================================================
# 任务模式
# ==============================================================================
if [[ -n "$TASK_CODE_NAME" ]]; then
    TASK_CODE_DIR="$WORKSPACE_ROOT/task/$TASK_CODE_NAME/code"
    if [[ ! -d "$TASK_CODE_DIR" ]]; then
        echo "ERROR: task worktree not found: $TASK_CODE_DIR" >&2
        echo "       create it first: scripts/task_worktree.sh new $TASK_CODE_NAME" >&2
        exit 1
    fi
    REPOS=(vllm vllm-ascend vllm-ascend-tools)
    ensure_submodule "$TASK_CODE_DIR/vllm-ascend"

    echo "=== Syncing (task mode: $TASK_CODE_NAME) → itask: ${TASK_NAME} ==="
    echo "    code source: task/$TASK_CODE_NAME/code/{vllm,vllm-ascend,vllm-ascend-tools}"
    echo "    remote: root@localhost:${SSH_PORT}:${REMOTE_WORKDIR}/"

    # ---- 选择 link-dest 引用任务（远端已有代码，用于硬链接复用） ----
    if [[ -z "$LINK_DEST_TASK" ]]; then
        LINK_DEST_TASK=$($SSH_CMD root@localhost "
            ls -1dt ${REMOTE_WORKDIR}/task/*/code 2>/dev/null \
              | grep -v '/task/${TASK_CODE_NAME}/' | head -1 \
              | xargs -r dirname | xargs -r basename" 2>/dev/null || true)
    fi
    if [[ -n "$LINK_DEST_TASK" && "$LINK_DEST_TASK" != "$TASK_CODE_NAME" ]]; then
        echo "    link-dest: 复用远端 task/${LINK_DEST_TASK}/code 的同内容文件（硬链接，0 传输）"
    else
        LINK_DEST_TASK=""
        echo "    link-dest: 远端无可复用任务目录，全量传输"
    fi

    $SSH_CMD root@localhost "mkdir -p ${REMOTE_WORKDIR}/task/${TASK_CODE_NAME}/code"

    # ---- 1) 共享顶层文件（scripts/docs/prompts/connector 等，不含 task/ 与 codebases/） ----
    echo "--- [1/3] shared top-level files ---"
    rsync -rtP "$WORKSPACE_ROOT/" \
        -e "$SSH_CMD" \
        root@localhost:"${REMOTE_WORKDIR}/" \
        --exclude=/task \
        --exclude=/codebases \
        --exclude=/scratch \
        --exclude=/data \
        --exclude=/workspace \
        --exclude=/docs/others \
        --exclude=.git \
        --exclude=*.log \
        --exclude=*.pyc \
        --exclude=__pycache__ \
        --exclude=nohup.out

    # ---- 2) 任务文档（progress.md 等；code/logs/profiling 不同步——日志与 profiling 只从远端下载） ----
    echo "--- [2/3] task docs: task/${TASK_CODE_NAME} ---"
    rsync -rtP "$WORKSPACE_ROOT/task/$TASK_CODE_NAME/" \
        -e "$SSH_CMD" \
        root@localhost:"${REMOTE_WORKDIR}/task/$TASK_CODE_NAME/" \
        --exclude=/code \
        --exclude=/logs \
        --exclude=/profiling \
        --exclude=/profiles \
        --exclude=.git \
        --exclude=*.log \
        --exclude=*.pyc \
        --exclude=__pycache__ \
        --exclude=nohup.out \
        ${FORCE_FLAG}

    # ---- 3) 任务代码三仓库（内容比对 + link-dest 硬链接复用） ----
    for repo in "${REPOS[@]}"; do
        echo "--- [3/3] code: $repo ---"
        LINK_DEST_ARGS=()
        if [[ -n "$LINK_DEST_TASK" ]]; then
            if $SSH_CMD root@localhost "test -d ${REMOTE_WORKDIR}/task/${LINK_DEST_TASK}/code/${repo}" 2>/dev/null; then
                LINK_DEST_ARGS=(--link-dest="${REMOTE_WORKDIR}/task/${LINK_DEST_TASK}/code/${repo}")
            fi
        fi
        rsync -rcP "$TASK_CODE_DIR/$repo/" \
            -e "$SSH_CMD" \
            root@localhost:"${REMOTE_WORKDIR}/task/${TASK_CODE_NAME}/code/${repo}/" \
            --exclude=.git \
            --exclude=*.pyc \
            --exclude=__pycache__ \
            --exclude=nohup.out \
            ${LINK_DEST_ARGS[@]+"${LINK_DEST_ARGS[@]}"} \
            ${FORCE_FLAG}
    done

    # ---- 同步后校验 ----
    VERIFY_FAIL=0
    if [[ "$NO_VERIFY" != "true" ]]; then
        echo "--- verify ---"
        for repo in "${REPOS[@]}"; do
            verify_sync "$TASK_CODE_DIR/$repo" \
                "${REMOTE_WORKDIR}/task/${TASK_CODE_NAME}/code/${repo}" \
                "$repo" || VERIFY_FAIL=1
        done
        if [[ "$VERIFY_FAIL" -eq 1 ]]; then
            echo "ERROR: sync verification failed — 本地与远端代码不一致" >&2
            echo "       可能原因：并行会话覆盖远端代码、rsync 传输不完整" >&2
            exit 1
        fi
    fi

    VERIFY_STATUS="✓"
    [[ "$NO_VERIFY" == "true" ]] && VERIFY_STATUS="skipped"
    echo "=== Sync complete (task mode: $TASK_CODE_NAME, link-dest: ${LINK_DEST_TASK:-none}, verify: $VERIFY_STATUS) ==="
    exit 0
fi

# ==============================================================================
# 默认模式：同步整个 workspace 含 codebases 主分支（纯部署用）
# ==============================================================================
echo "=== Syncing workspace (default: codebases main) → itask: ${TASK_NAME} ==="
echo "    code source: codebases/{vllm,vllm-ascend,vllm-ascend-tools} (main worktree)"
ensure_submodule "$WORKSPACE_ROOT/codebases/vllm-ascend"
echo "    remote: root@localhost:${SSH_PORT}:${REMOTE_WORKDIR}/"

# 确保远程目录存在
$SSH_CMD root@localhost "mkdir -p ${REMOTE_WORKDIR}"

echo "--- rsync -> root@localhost:${SSH_PORT}:${REMOTE_WORKDIR}/ ---"
rsync -rtP "$WORKSPACE_ROOT/" \
    -e "$SSH_CMD" \
    root@localhost:"${REMOTE_WORKDIR}/" \
    --exclude=/scratch \
    --exclude=/data \
    --exclude=/workspace \
    --exclude=/docs/others \
    --exclude=/task/*/code \
    --exclude=/task/*/logs \
    --exclude=/task/*/profiling \
    --exclude=/task/*/profiles \
    --exclude=.git \
    --exclude=*.log \
    --exclude=*.pyc \
    --exclude=__pycache__ \
    --exclude=nohup.out

# ---- 同步后校验 ----
VERIFY_FAIL=0
if [[ "$NO_VERIFY" != "true" ]]; then
    echo "--- verify ---"
    for repo in vllm vllm-ascend vllm-ascend-tools; do
        verify_sync "$WORKSPACE_ROOT/codebases/$repo" \
            "${REMOTE_WORKDIR}/codebases/$repo" \
            "$repo" || VERIFY_FAIL=1
    done
    if [[ "$VERIFY_FAIL" -eq 1 ]]; then
        echo "ERROR: sync verification failed — 本地与远端代码不一致" >&2
        exit 1
    fi
fi

VERIFY_STATUS="✓"
[[ "$NO_VERIFY" == "true" ]] && VERIFY_STATUS="skipped"
echo "=== Sync complete (verify: $VERIFY_STATUS) ==="
