---
name: pr-precheck
description: vllm-ascend 社区 PR 提交必备检查 gate。只要用户要把分支推送到 jinqi 或 fny 远端（两者都是
  https://github.com/vllm-project/vllm-ascend 的 fork，推送即创建/更新社区 PR），或提到"提交 PR/更新 PR/提交社区/给官方提 PR"且涉及
  vllm-ascend 分支时必须触发；即使用户只说"push jinqi"、"推 fny"、"force push 到 jinqi"也要触发，推送前先跑本 gate。执行：DCO
  签名检查 → 本地复刻 CI（pre-commit 18 hooks + mypy×3 版本，脚本幂等）→ 修复单独 [Misc] commit → 推送 → 验证 GitHub check-runs。
---

# vllm-ascend fork 推送必备检查（PR precheck gate）

推分支到 `jinqi` / `fny` = 在 vllm-project/vllm-ascend 创建/更新 PR。社区 CI（`pr_test.yaml` 的
`pre-commit` job）会跑 lint + 类型检查 + DCO，**push 前在本地跑同等检查**，避免 CI 失败往返。

## 流程（按序执行，不可跳过）

### 0. 确定范围

确认当前分支、远端（jinqi/fny）、待推 commit 集：`git log --oneline jinqi/<branch>..HEAD`（分支远端不存在时为全部新 commit）。

### 1. DCO 签名检查（逐 commit）

```bash
git log --format='%h %(trailers:key=Signed-off-by)' jinqi/<branch>..HEAD
```

每个 commit 必须有 `Signed-off-by: <作者名> <邮箱>`，否则 DCO 检查卡 `action_required`。缺签的 commit 用
`git commit --amend -s`（HEAD）或 detach + `--amend -s --no-edit` + `git rebase --onto <新> <旧> <分支>`（非
HEAD，内容不变时零冲突）补签。

改写历史 + force push 需先向用户确认（已在预览分支等确认的场景，按 [[pr-fix-preview-branch-await-inspection]] 惯例处理）。

### 2. 本地复刻 CI 检查（重活，脚本幂等）

```bash
bash <workspace>/.zcode/skills/pr-precheck/scripts/run_vllm_ascend_ci_checks.sh <vllm-ascend-repo-dir>
```

`<vllm-ascend-repo-dir>` 即任务 worktree（`task/<t>/code/vllm-ascend`）或主树（`codebases/vllm-ascend`）。脚本自动搭建并复用环境，跑两项：

1. **pre-commit**：`pre-commit run --all-files --hook-stage manual`（与 CI 完全一致的 18 个 hook：ruff
   check/format、codespell、typos、clang-format、markdownlint、actionlint、gitleaks、shellcheck 及各本地检查 hook）
2. **mypy**：`--follow-imports skip --check-untyped-defs` × 3.10/3.11/3.12 × `vllm_ascend`/`examples`/`tests`，
   PYTHONPATH 指向 CI 锁定的 vllm main 源码 worktree（commit 读自 `.github/vllm-main-verified.commit`）

对失败项的处置：

- **ruff format/check、代码类失败** → 修复放**单独 `[Misc]` commit**，必须 `git commit -s`；不要混进功能 commit
- pre-commit hook 自动改写文件（如 `#TODO` → `# TODO`）时，脚本会提示；确认 diff 后提交重跑
- 环境性残留不算失败：mypy 输出中 `vllm_ascend.vllm_ascend_C` import-not-found 是本机无编译扩展所致（CI 镜像有 .so），脚本已过滤

### 3. PR title 前缀

正则（大小写不敏感）：`[BugFix|Performance|Test|CI|Feature|Doc|Misc|Community|Refactor]`，如 `[Feature] xxx`。
不合规先改 title（`gh` 未装时用 GitHub API 或让用户改）。

### 4. 推送

```bash
git push jinqi <branch>          # 普通
git push -f jinqi <branch>       # 历史已改写时
```

### 5. 推后验证 check-runs

**用 python urllib**（本机 curl 会被静默 SIGKILL）：

```bash
python3 -c "import urllib.request, json; d=json.load(urllib.request.urlopen(urllib.request.Request('https://api.github.com/repos/vllm-project/vllm-ascend/commits/<sha>/check-runs', headers={'Accept':'application/vnd.github+json','User-Agent':'python-urllib'}))); [print(r['status'], r.get('conclusion'), r['name']) for r in d['check_runs']]"
```

- `pre-commit`、`DCO` 必须转 success（几分钟内；可 sleep 后轮询）
- `ci-gate` failure 是**流程等待**：select-tests 判定有单测要跑时须等维护者加 `ready` 标签，作者侧无可做事项——向用户说明即可，不要试图绕过

## 环境说明

脚本幂等处理，首跑约 10 分钟（下载依赖），之后秒级复用：

- lint venv：`scratch/lint-venv`（pre-commit 4.0.1 + mypy 1.11.1 + mypy 解析所需第三方包，清单见脚本内 `MYPY_PKGS`）
- go 工具链：`/private/tmp/go-toolchain/`（golang.google.cn 镜像下载）。**必须在家目录外**——pre-commit 的
  `exe_exists` 会忽略 `$HOME` 下的可执行文件；go 仅 actionlint hook 需要，下载失败时脚本自动 SKIP 并告警
- gitleaks / shellcheck：brew 安装（gitleaks.sh 的 OBS wget 下载在本机会被 SIGKILL，勿走该路径）
- mypy 源码树：`scratch/vllm-main-mypy`（按 `.github/vllm-main-verified.commit` 锁定的 commit 建 detached worktree，勿动 `codebases/vllm` 主树分支）
- GOPROXY 固定华为云镜像（proxy.golang.org / go.dev 本机不可达）；mypy 固定 `--cache-dir=/dev/null`（1.11.1 写缓存会崩）

范围外：vllm / vllm-ascend-tools 仓库的 fork 推送不适用本 skill（上游检查体系不同）——遇到时报告用户，勿套用。
