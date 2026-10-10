---
name: build-vllm
description: |
  从源码 editable 安装开发版 vllm + vllm-ascend（pip install -e .），Python 即时生效，仅 C++ 扩展（.so）变更需重编。
  触发场景：编译/build/compile/源码安装 vllm 或 vllm-ascend 开发版；build_aclnn / C++ kernel 编译失败排错；
  卸载镜像预装 vllm/vllm-ascend 避免干扰；社区版 vllm-ascend 内网 patch；
  多机安装同一版本（p0 build 一次→传递产物→其他机 COMPILE_CUSTOM_KERNELS=0 快速安装）；
  编译成功 check（pip show Editable 指向 worktree + .so 存在 + import ok）；
  记录构建基线 marker 供 start-service 判定是否需要重编。
  用户提到"编译/build/compile/源码安装/开发版/editable"、"build_aclnn"、"多机安装 vllm/vllm-ascend"、"内网 patch"、"卸载 vllm/vllm-ascend"时使用。
compatibility: 依赖 itask SSH 隧道、scripts/ssh_tunnel.sh (仅 connect)、scripts/sync.sh；产出供 /start-service 启动使用
---

# Build vLLM Skill

把这个 skill 当作"如何从源码 editable 安装开发版 vllm + vllm-ascend 并用于部署"的执行手册。**只使用 editable install（`pip install -e .`）**——Python 代码即时生效，C++ 扩展变更需重编 + 重启；不使用 dist 目录输出，不做 dist→源码软链。本 SKILL.md 是导航中枢，详细流程下沉到 `reference/` 下各模块，按需加载。

> **与 /start-service 的分工**：本 skill 只负责**编译安装**（产出 editable 安装）。完成后回到 /start-service 启动服务——editable 已注册到 sys.path 指向源码目录，**无需设 PYTHONPATH 指向 dist**。反过来，/start-service 在启动前会用 git diff 判定"是否需要重编"，需要时调用本 skill；纯部署测试（不改代码、用源码 PYTHONPATH 或镜像默认）不触发本 skill。

## 变量定义

| 变量 | 值 | 说明 |
|------|-----|------|
| `WS` | `/a3_inference/itask/workdir/yjq02324703/workspace` | 远程工作空间根目录 |
| `TASK` | `<task-name>` | 任务模式必填，对应本地 `task/<task>/code/` worktree |
| `CODE_BASE` | 任务模式 `$WS/task/$TASK/code`；默认模式 `$WS/codebases` | 三仓库代码根目录 |
| `VLLM_DIR` | `$CODE_BASE/vllm` | vllm 源码目录（editable 安装指向此） |
| `ASCEND_DIR` | `$CODE_BASE/vllm-ascend` | vllm-ascend 源码目录（editable 安装指向此，.so 产于此） |

> **本地 vs 远程路径**：git diff、git log 等 git 操作在**本地 worktree** 执行（任务模式 `task/<task>/code/{vllm,vllm-ascend}`，默认模式 `codebases/{vllm,vllm-ascend}`）；编译/安装命令通过 SSH 隧道在**远程**执行。sync 把本地 worktree 同步到远程 `CODE_BASE`。

## Hard Rules

1. 编译前必须确认 SSH 隧道已建立，否则先运行 `scripts/ssh_tunnel.sh connect <task> <port>`
2. 编译前必须确认代码已同步：任务模式 `scripts/sync.sh <itask> --task <task> --port <port>`，默认模式 `scripts/sync.sh <itask> --port <port>`
3. **绝不擅自删除/复用 itask 机器**——必须问用户确认是复用已有还是新建
4. **编译前必须卸载镜像预装的 vllm + vllm-ascend**：`pip uninstall -y vllm vllm-ascend`，否则旧版残留干扰 editable 安装（import 优先级、版本混入、旧 .so）
5. vllm-ascend 编译需 CANN toolkit（`/usr/local/Ascend/ascend-toolkit/latest/`），编译前验证存在
6. 编译顺序：先 vllm（纯 Python，~1min）再 vllm-ascend（C++ kernel，5-25min）
7. **`build_aclnn.sh` 失败必须深度清理** `csrc/build csrc/output csrc/build_out`（与常规增量重试不同），改用两步编译法
8. **常规增量重试不要删 `csrc/build/`**（CMake 跳过已编 kernel）；只有 `build_aclnn.sh` 失败才深度清理
9. **只用 `pip install -e .`（editable）**，不用 `--target=<dist>`；editable 下 Python 即时生效，无需软链优化
10. vllm-ascend 编译后**必须重启 vllm 进程**才能加载新 .so（C++ 扩展变更除外，Python 即时生效）
11. **编译成功必须三重 check**：`pip show` 的 Editable 指向 worktree + `.so` 存在 + `import` ok（见 [reference/build.md](reference/build.md) Step 8）
12. 多机部署装同一版本：p0 build 一次→传递产物→其他机 `COMPILE_CUSTOM_KERNELS=0` 快速安装，详见 [reference/multi-node-install.md](reference/multi-node-install.md)
13. **清理进程必须使用 `scripts/kill_all.sh`**，不要手动 `kill -9`
14. 始终使用中文回答

## Self-refresh protocol

长对话 context 压缩后规则可能丢失。**每个阶段完成后重新调用本 skill** 刷新规则（尤其多机安装跨阶段时）。

## Module navigation

根据用户请求选择模块加载：

| 用户说 | 加载 |
| :--- | :--- |
| "编译/build/compile/源码安装/开发版/editable/卸载 vllm" | [reference/build.md](reference/build.md) |
| "build_aclnn 失败/编译错误/参数说明/源码缺失" | [reference/build-commands.md](reference/build-commands.md) |
| "内网 patch/gitcode 下载失败/社区版" | [reference/patches.md](reference/patches.md) |
| "多机安装/同一版本/传递产物" | [reference/multi-node-install.md](reference/multi-node-install.md) |

> 软链优化（dist→源码）已废弃——editable install 天然实现 Python 即时生效。"何时需要重编"决策表见 [reference/build.md](reference/build.md)。

## 收集信息

触发编译时，按顺序收集（用户未提供的必须询问）：

1. **itask 机器**：复用已有还是新建？（绝不擅自删除/复用）；a2 还是 a3？（两块独立 NFS 盘，a2 的安装不能给 a3 用，反之亦然）
2. **代码仓路径**：`VLLM_DIR` / `ASCEND_DIR`（任务模式 vs 默认模式）
3. **git 分支/commit**：打印 `git branch --show-current && git log --oneline -3`，问用户是否切换/pull/cherry-pick
4. **是否社区版 vllm-ascend**：`git remote -v` 指向 github → 可能需内网 patch（指向内部 git 已 patch 跳过）
5. **vllm 版本号**：`VLLM_VERSION_OVERRIDE` / `SETUPTOOLS_SCM_PRETEND_VERSION` 用值（避免 setuptools-scm 算错版本）

## 整体流程

```
Build Progress:
- [ ] 1. 确定 itask 机器（复用/新建 — 问用户）
- [ ] 2. 确认代码仓路径 + CANN toolkit
- [ ] 3. 打印 git 分支/log → 问用户确认
- [ ] 4. 加内网 patch（仅社区版 vllm-ascend）
- [ ] 5. 卸载镜像已有 vllm / vllm-ascend（pip uninstall -y）
- [ ] 6. 展示 editable 编译命令 → 问用户是否调整参数
- [ ] 7. 先装 vllm（empty）再装 vllm-ascend（COMPILE_CUSTOM_KERNELS=1 编 .so）→ 自动修错
- [ ] 8. 验证安装成功（pip show Editable→worktree + .so + import ok）
  ↻ Re-invoke build-vllm skill to refresh rules
- [ ] 9. 写入构建基线 marker（供 /start-service 判定重编）
- [ ] 10. 衔接 /start-service 启动（editable 已注册，无需 PYTHONPATH 指向 dist）
```

完整流程见 [reference/build.md](reference/build.md)。多机场景见 [reference/multi-node-install.md](reference/multi-node-install.md)。

## 构建基线 marker（供 /start-service 重编判定）

每次成功构建后，在**本地**记录当前 worktree 的 commit，供 /start-service 用 git diff 判定后续是否需要重编：

```bash
# 任务模式
git -C task/<task>/code/vllm-ascend rev-parse HEAD > scratch/.build-baseline/<task>.vllm-ascend
git -C task/<task>/code/vllm rev-parse HEAD > scratch/.build-baseline/<task>.vllm
# 默认模式（task 名用 default）
git -C codebases/vllm-ascend rev-parse HEAD > scratch/.build-baseline/default.vllm-ascend
git -C codebases/vllm rev-parse HEAD > scratch/.build-baseline/default.vllm
```

- marker 在本地 `scratch/.build-baseline/` 下，**不 sync、不纳入 git**（scratch/ 不跟踪）
- /start-service 读取 marker 后 `git diff <marker>..HEAD -- csrc/ setup.py pyproject.toml` 判定 C++/构建配置是否变更
- marker 缺失（首次/scratch 被清）时 /start-service 回退到 `git merge-base` 兜底（保守：本次任务触及 csrc 即建议重编）
