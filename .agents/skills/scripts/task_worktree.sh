#!/bin/bash
# task_worktree.sh - 管理任务 worktree（基于 codebases 三仓库）
#
# 核心思想：所有任务源代码只有 codebases/ 下一份（主工作树），每个"需要变更分支执行的任务"
# 在 task/<task-name>/code/ 下为 vllm/vllm-ascend/vllm-ascend-tools 各创建一个 worktree，
# 分支名统一为 <task-name>，基于 codebases 主工作树当前 HEAD 切出。多任务并行互不干扰。
#
# 用法:
#   scripts/task_worktree.sh new <task-name>              # 三仓库各创建 worktree（基于各自当前 HEAD）
#   scripts/task_worktree.sh new <task-name> --base <br>  # 三仓库都基于 <br> 分支切出（不存在则报错）
#   scripts/task_worktree.sh list                         # 列出所有任务 worktree
#   scripts/task_worktree.sh info <task-name>             # 查看某任务的分支/commit 状态
#   scripts/task_worktree.sh remove <task-name>           # 删除任务 worktree 目录 + 三仓库 <task-name> 分支（仅清理时用，默认保留）
#
# 受 worktree 机制管理的仓库：vllm、vllm-ascend、vllm-ascend-tools（其他 codebases 仓库不管理）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CODEBASES_DIR="$WORKSPACE_ROOT/codebases"
TASK_DIR="$WORKSPACE_ROOT/task"

# 受 worktree 机制管理的三个仓库
REPOS=(vllm vllm-ascend vllm-ascend-tools)

usage() {
    sed -n '3,15p' "$0"
}

# 检查某仓库的 <task-name> 分支是否已存在
branch_exists() {
    local repo_dir="$1" branch="$2"
    (cd "$repo_dir" && git rev-parse --verify "refs/heads/$branch" >/dev/null 2>&1)
}

cmd_new() {
    local task_name="${1:?task-name required (Usage: task_worktree.sh new <task-name> [--base <branch>])}"
    shift
    local base=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --base) base="${2:?--base requires a value}"; shift 2 ;;
            *) echo "Unknown option: $1" >&2; exit 1 ;;
        esac
    done

    local task_code_dir="$TASK_DIR/$task_name/code"
    if [[ -d "$task_code_dir" ]]; then
        echo "ERROR: task worktree already exists: $task_code_dir" >&2
        echo "       use 'scripts/task_worktree.sh info $task_name' to inspect" >&2
        exit 1
    fi

    mkdir -p "$task_code_dir"
    echo "=== Creating task worktree: $task_name ==="
    echo "    code dir: $task_code_dir"
    echo "    branch:   $task_name (all three repos)"
    [[ -n "$base" ]] && echo "    base:     $base" || echo "    base:     each repo's current HEAD"
    echo

    for repo in "${REPOS[@]}"; do
        local repo_dir="$CODEBASES_DIR/$repo"
        local wt_dir="$task_code_dir/$repo"
        if [[ ! -d "$repo_dir/.git" && ! -f "$repo_dir/.git" ]]; then
            echo "WARN: $repo_dir is not a git repo, skipped" >&2
            continue
        fi

        if branch_exists "$repo_dir" "$task_name"; then
            echo "--- $repo: reusing existing branch '$task_name' ---"
            (cd "$repo_dir" && git worktree add "$wt_dir" "$task_name")
        elif [[ -n "$base" ]]; then
            if ! (cd "$repo_dir" && git rev-parse --verify "refs/heads/$base" >/dev/null 2>&1); then
                echo "ERROR: $repo has no branch '$base' (--base)" >&2
                exit 1
            fi
            echo "--- $repo: creating branch '$task_name' from '$base' ---"
            (cd "$repo_dir" && git worktree add -b "$task_name" "$wt_dir" "$base")
        else
            echo "--- $repo: creating branch '$task_name' from current HEAD ---"
            (cd "$repo_dir" && git worktree add -b "$task_name" "$wt_dir" HEAD)
        fi

        # vllm-ascend 需要 catlass submodule（worktree 不共享 submodule 工作树）
        if [[ "$repo" == "vllm-ascend" ]]; then
            echo "--- $repo: initializing submodule catlass ---"
            (cd "$wt_dir" && git submodule update --init csrc/third_party/catlass 2>/dev/null || true)
        fi
        echo
    done

    echo "=== Done. Next steps ==="
    echo "  1. 编辑代码: cd $task_code_dir/{vllm,vllm-ascend,vllm-ascend-tools}"
    echo "  2. 同步远程: scripts/sync.sh <itask-name> --task $task_name --port <port>"
    echo "  3. 启动服务: /start-service (TASK=$task_name)"
}

cmd_list() {
    local task_name="${1:-}"
    if [[ -n "$task_name" ]]; then
        cmd_info "$task_name"
        return
    fi
    echo "=== Task worktrees (per repo) ==="
    for repo in "${REPOS[@]}"; do
        local repo_dir="$CODEBASES_DIR/$repo"
        [[ -d "$repo_dir/.git" || -f "$repo_dir/.git" ]] || continue
        echo "--- $repo ---"
        (cd "$repo_dir" && git worktree list)
        echo
    done
}

cmd_info() {
    local task_name="${1:?task-name required (Usage: task_worktree.sh info <task-name>)}"
    local task_code_dir="$TASK_DIR/$task_name/code"
    if [[ ! -d "$task_code_dir" ]]; then
        echo "ERROR: task worktree not found: $task_code_dir" >&2
        echo "       create it: scripts/task_worktree.sh new $task_name" >&2
        exit 1
    fi
    echo "=== Task: $task_name ==="
    echo "    code dir: $task_code_dir"
    echo
    for repo in "${REPOS[@]}"; do
        local wt_dir="$task_code_dir/$repo"
        if [[ -d "$wt_dir" ]]; then
            local branch commit dirty
            branch=$(cd "$wt_dir" && git branch --show-current 2>/dev/null || echo "?")
            commit=$(cd "$wt_dir" && git log --oneline -1 2>/dev/null || echo "?")
            dirty=$(cd "$wt_dir" && git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
            [[ "$dirty" != "0" ]] && commit="$commit  ($dirty uncommitted)"
            printf "  %-20s [%s] %s\n" "$repo" "$branch" "$commit"
        else
            printf "  %-20s (missing)\n" "$repo"
        fi
    done
}

cmd_remove() {
    local task_name="${1:?task-name required (Usage: task_worktree.sh remove <task-name>)}"
    local task_code_dir="$TASK_DIR/$task_name/code"
    if [[ ! -d "$task_code_dir" ]]; then
        echo "ERROR: task worktree not found: $task_code_dir" >&2
        exit 1
    fi

    echo "=== Removing task worktree: $task_name ==="
    for repo in "${REPOS[@]}"; do
        local repo_dir="$CODEBASES_DIR/$repo"
        local wt_dir="$task_code_dir/$repo"
        if [[ -d "$wt_dir" ]]; then
            echo "--- $repo: removing worktree ---"
            (cd "$repo_dir" && git worktree remove --force "$wt_dir" 2>/dev/null) \
                || { echo "    git worktree remove failed, force rm"; rm -rf "$wt_dir"; }
            if branch_exists "$repo_dir" "$task_name"; then
                (cd "$repo_dir" && git worktree prune 2>/dev/null || true)
                (cd "$repo_dir" && git branch -D "$task_name" 2>/dev/null) \
                    && echo "    deleted branch $task_name" \
                    || echo "    branch $task_name kept (may have unmerged commits)"
            fi
        fi
    done

    # 清理空的 code 目录和 task 目录
    rmdir "$task_code_dir" 2>/dev/null || true
    rmdir "$TASK_DIR/$task_name" 2>/dev/null || true
    echo "=== Removed: $task_name ==="
}

main() {
    local cmd="${1:-}"
    [[ -z "$cmd" ]] && { usage; exit 1; }
    shift
    case "$cmd" in
        new)        cmd_new "$@" ;;
        list)       cmd_list "$@" ;;
        info)       cmd_info "$@" ;;
        remove)     cmd_remove "$@" ;;
        -h|--help|help) usage ;;
        *) echo "Unknown command: $cmd" >&2; usage; exit 1 ;;
    esac
}

main "$@"
