#!/usr/bin/env bash
# 本地复刻 vllm-ascend PR CI 检查（pr_test.yaml 的 pre-commit job）。
# 用法: bash run_vllm_ascend_ci_checks.sh <vllm-ascend-repo-dir>
#   <vllm-ascend-repo-dir> = task/<t>/code/vllm-ascend（任务 worktree）或 codebases/vllm-ascend（主树）
# 幂等：首跑自动搭 venv / go / gitleaks / shellcheck / mypy 源码树（约 10 分钟），之后秒级复用。
# 退出码：0=全过；1=存在真实检查失败。环境搭建失败仅降级跳过对应 hook（CI 兜底）并告警。
set -euo pipefail

REPO_DIR="$(cd "${1:?usage: $0 <vllm-ascend-repo-dir>}" && pwd)"
GO_VER="${GO_VER:-1.27.1}"
FAIL=""

# ---------- 定位 workspace 根（含 scripts/task_worktree.sh 的目录） ----------
WS=""
d="$REPO_DIR"
while [ "$d" != "/" ]; do
  if [ -f "$d/scripts/task_worktree.sh" ]; then WS="$d"; break; fi
  d="$(dirname "$d")"
done
[ -n "$WS" ] || { echo "❌ 未找到 workspace 根（向上未发现 scripts/task_worktree.sh）"; exit 1; }
echo "▶ workspace: $WS"
echo "▶ repo:      $REPO_DIR"

VENV="$WS/scratch/lint-venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

# ---------- 1) lint venv（pre-commit + mypy + mypy 解析所需第三方包） ----------
MYPY_PKGS=(einops xgrammar safetensors openai msgspec pyzmq uvicorn fastapi
  pybase64 ray psutil types-psutil modelscope huggingface_hub Pillow types-Pillow
  scipy pytest pyyaml aiohttp uvloop sentence_transformers types-jsonschema
  soundfile pytest-mock torch numpy)

if [ -x "$VENV/bin/pre-commit" ] && [ -x "$VENV/bin/mypy" ]; then
  echo "▶ 复用 lint venv: $VENV"
else
  echo "▶ 创建 lint venv: $VENV"
  python3 -m venv "$VENV"
  "$PY" -m pip install -qU pip
  "$PIP" install -q -r "$REPO_DIR/requirements-lint.txt"
  "$PIP" install -q "${MYPY_PKGS[@]}"
fi
# venv 可能为旧建：仓库新引入依赖时按清单补装
if ! "$PY" -c "import msgspec, zmq, einops, xgrammar, scipy, aiohttp, uvloop, sentence_transformers, soundfile, pytest_mock, torch, numpy" >/dev/null 2>&1; then
  echo "▶ venv 缺依赖，补装中..."
  "$PIP" install -q "${MYPY_PKGS[@]}"
fi
export PATH="$VENV/bin:$PATH"

# ---------- 2) 本地工具：go / gitleaks / shellcheck ----------
SKIP_HOOKS=""
if [ -x /private/tmp/go-toolchain/go/bin/go ]; then
  export PATH="/private/tmp/go-toolchain/go/bin:$PATH"
elif command -v go >/dev/null 2>&1; then
  echo "▶ 使用系统 go: $(command -v go)"
else
  # go 仅 actionlint hook 需要；必须装在家目录外（pre-commit 忽略 $HOME 下可执行文件）
  case "$(uname -m)" in arm64) GOARCH=arm64 ;; x86_64) GOARCH=amd64 ;; *) GOARCH="" ;; esac
  if [ -n "$GOARCH" ] && "$PY" -c "
import urllib.request, sys
url = 'https://golang.google.cn/dl/go${GO_VER}.darwin-${GOARCH}.tar.gz'
with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'python-urllib'}), timeout=120) as r, open('/private/tmp/go.tar.gz', 'wb') as f:
    while c := r.read(1 << 20): f.write(c)
" 2>/dev/null; then
    mkdir -p /private/tmp/go-toolchain
    tar -xzf /private/tmp/go.tar.gz -C /private/tmp/go-toolchain && rm -f /private/tmp/go.tar.gz
    export PATH="/private/tmp/go-toolchain/go/bin:$PATH"
    echo "▶ go $GO_VER 就绪: $(command -v go)"
  else
    rm -f /private/tmp/go.tar.gz
    echo "⚠️ go 不可用，跳过 actionlint（只查 workflow YAML；CI 会兜底）"
    SKIP_HOOKS="${SKIP_HOOKS:+${SKIP_HOOKS},}actionlint"
  fi
fi

for t in gitleaks shellcheck; do
  if ! command -v "$t" >/dev/null 2>&1; then
    echo "▶ brew install $t"
    brew install -q "$t" >/dev/null 2>&1 || true
  fi
  if command -v "$t" >/dev/null 2>&1; then
    continue
  fi
  echo "⚠️ $t 不可用，跳过对应 hook（CI 会兜底）"
  case "$t" in
    gitleaks)   SKIP_HOOKS="${SKIP_HOOKS:+${SKIP_HOOKS},}gitleaks-offline-scan" ;;
    shellcheck) SKIP_HOOKS="${SKIP_HOOKS:+${SKIP_HOOKS},}shellcheck" ;;
  esac
done

# ---------- 3) pre-commit（与 CI 完全一致的命令） ----------
cd "$REPO_DIR"
export SHELLCHECK_OPTS="--exclude=SC2046,SC2006,SC2086"
export GOPROXY="${GOPROXY:-https://repo.huaweicloud.com/repository/goproxy/}"

echo "=== pre-commit run --all-files --hook-stage manual ${SKIP_HOOKS:+"(SKIP=$SKIP_HOOKS)"} ==="
if ! SKIP="${SKIP_HOOKS}" pre-commit run --all-files --hook-stage manual --show-diff-on-failure; then
  FAIL="${FAIL}pre-commit "
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "ℹ️ pre-commit 改写了工作区文件（多数为 ruff format 重排）：确认 diff 后提交为单独 [Misc] commit（git commit -s），然后重跑本脚本"
fi

# ---------- 4) mypy（tools/mypy.sh 1 <ver> 的 CI 等价） ----------
COMMIT_FILE="$REPO_DIR/.github/vllm-main-verified.commit"
VLLM_WT="$WS/scratch/vllm-main-mypy"
if [ ! -f "$COMMIT_FILE" ]; then
  echo "⚠️ 缺 $COMMIT_FILE，跳过 mypy（报告用户）"
  FAIL="${FAIL}mypy(env-missing-commit-file) "
else
  PINNED="$(tr -d '[:space:]' < "$COMMIT_FILE")"
  if [ ! -d "$VLLM_WT/vllm" ]; then
    echo "▶ 建立 vllm 源码 worktree @ $PINNED → $VLLM_WT"
    SRC="$WS/codebases/vllm"
    [ -d "$SRC/.git" ] || { echo "❌ 缺 $SRC，无法建 mypy 源码树（报告用户）"; FAIL="${FAIL}mypy(env-missing-vllm-repo) "; }
    if [ -d "$SRC/.git" ]; then
      git -C "$SRC" cat-file -e "$PINNED" 2>/dev/null || git -C "$SRC" fetch --quiet https://github.com/vllm-project/vllm.git "$PINNED"
      git -C "$SRC" worktree add --detach "$VLLM_WT" "$PINNED" >/dev/null
    fi
  fi
  if [ -d "$VLLM_WT/vllm" ]; then
    export PYTHONPATH="$VLLM_WT${PYTHONPATH:+:$PYTHONPATH}"
    for v in 3.10 3.11 3.12; do
      for scope in vllm_ascend examples tests; do
        echo "=== mypy python $v : $scope ==="
        # vllm_ascend.vllm_ascend_C = 本仓库编译扩展，本机必然缺失（CI 镜像有 .so），过滤之
        OUT="$("$VENV/bin/mypy" --follow-imports skip --check-untyped-defs --python-version "$v" --cache-dir=/dev/null "$scope" 2>&1 || true)"
        REAL="$(printf '%s\n' "$OUT" | grep 'error:' | grep -v 'vllm_ascend\.vllm_ascend_C' || true)"
        if [ -n "$REAL" ]; then
          FAIL="${FAIL}mypy($v/$scope) "
          printf '%s\n' "$OUT" | tail -25
        else
          printf '%s\n' "$OUT" | grep -E '(Success|Found)' | tail -1 || true
        fi
      done
    done
  fi
fi

# ---------- 5) 结论 ----------
echo "=================================="
if [ -z "$FAIL" ]; then
  echo "✅ 全部通过，可推送（记得逐 commit DCO 与 PR title 前缀，见 SKILL.md）"
  exit 0
else
  echo "❌ 未通过: $FAIL（修复进单独 [Misc] commit 后重跑）"
  exit 1
fi
