# 源码编译开发版（vLLM + vLLM-Ascend，editable 模式）

采用 **editable install（`pip install -e .`）**：vllm/vllm-ascend 注册到 sys.path 并指向源码目录，Python 代码修改即时生效（无需重装），C++ 扩展（.so）变更需重编 + 重启进程。**不使用 dist 目录输出，不做 dist→源码软链**——editable 已覆盖轻量迭代需求。

> **与启动流程的关系**：本模块产出 editable 安装。完成后回到 /start-service（[launch.md](../../start-service/reference/launch.md)）正常启动——editable 已注册，无需设 PYTHONPATH 指向 dist。

## Workflow checklist

```
Build Progress:
- [ ] 1. 确定 itask 机器（复用已有 / 新建 — 必须问用户，绝不擅自删除）
- [ ] 2. 确认代码仓路径（vllm + vllm-ascend 源码）+ CANN toolkit
- [ ] 3. 打印 git 分支/log → 问用户确认（按需切换分支 / pull）
- [ ] 4. 给社区版 vllm-ascend 加内网 patch（若 upstream 来自 github）
- [ ] 5. 卸载镜像已有 vllm / vllm-ascend（避免干扰 editable 安装）
- [ ] 6. 展示 editable 编译命令 → 问用户是否调整参数
- [ ] 7. 先装 vllm（empty）再装 vllm-ascend（COMPILE_CUSTOM_KERNELS=1 编 .so）→ 自动修错
- [ ] 8. 验证安装成功（pip show Editable 指向 worktree + .so 存在 + import ok）
  ↻ Re-invoke build-vllm skill to refresh rules
- [ ] 9. 写入构建基线 marker（供 /start-service 判定是否需要重编）
- [ ] 10. 衔接 /start-service 启动
```

## 约定：远程命令执行方式

本模块所有远程命令通过 SSH 隧道执行（与启动流程一致）：

```bash
ssh -o StrictHostKeyChecking=no -p <ssh-port> root@localhost "<bash 命令>"
```

`<ssh-port>` 是目标机器的 SSH 隧道端口（默认 27890）。如未建立隧道，先运行 `scripts/ssh_tunnel.sh connect <task> <port>`。

每条 `itask exec` 或 `ssh ... "<cmd>"` 都是独立的 shell，不保留环境变量——需要多步骤连续操作时，用 `bash -c '...'` 串成一条命令。

## Step 1: 确定 itask 机器

通过 `itask list --user <user>` 查看已有机器。**问用户**：

1. **复用已有机器还是新建？** — 绝不擅自删除或复用，必须用户确认
2. **a2 还是 a3 类型？** — 两者都有 NFS 共享盘（a2 在 `/a2_inference`，a3 在 `/a3_inference`），但**是两块独立的盘**——a2 上 editable 安装的 vllm-ascend（.so 在源码树）不能被 a3 部署机器直接用，反之亦然。中立陈述：

   - **a2 nocard**：NFS 在 `/a2_inference/...`；需确认镜像是否含 CANN toolkit（vllm-ascend 编译需要 ccec）
   - **a3 nocard**：NFS 在 `/a3_inference/...`；通常含 CANN toolkit
   - **16-card pod**：有 CANN + NPU（编译后可直接测试），但编译期间占用 16 卡
   - 编译是 CPU 任务（gcc/ccec 在 CPU 上跑），nocard 足够，不浪费 NPU 资源

## Step 2: 确认代码仓路径 + CANN toolkit

问用户：

- **vllm 源码** 路径：即 `VLLM_DIR`——任务模式 `.../workspace/task/<task>/code/vllm`，默认模式 `.../workspace/codebases/vllm`
- **vllm-ascend 源码** 路径：即 `ASCEND_DIR`——任务模式 `.../task/<task>/code/vllm-ascend`，默认模式 `.../codebases/vllm-ascend`

验证两个目录都存在且含 `setup.py` 或 `pyproject.toml`，并验证 CANN toolkit（vllm-ascend 编译必需）：

```bash
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "ls <vllm_path>/setup.py <vllm_path>/pyproject.toml <vllm_ascend_path>/setup.py 2>&1"
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "ls /usr/local/Ascend/ascend-toolkit/latest/"
```

## Step 3: 确认 git 分支

对 vllm + vllm-ascend 两个仓库，打印当前分支 + 最近 commit：

```bash
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "cd <vllm_path> && echo '=== vllm ===' && git branch --show-current && git log --oneline -3"
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "cd <vllm_ascend_path> && echo '=== vllm-ascend ===' && git branch --show-current && git log --oneline -3"
```

展示给用户，问："分支和 commit 确认正确吗？需要切换分支 / pull 最新 / cherry-pick 吗？"

如果用户要求 git 操作（checkout / pull / cherry-pick / merge），自动执行并重新打印结果供确认。

## Step 4: 加内网 patch（仅社区版 vllm-ascend）

如果 vllm-ascend 来自社区上游（github.com/vllm-project/vllm-ascend），可能需要内网兼容 patch（下载 URL、包索引等）。

检查 patch 是否已应用：

```bash
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "cd <vllm_ascend_path> && git diff --stat HEAD"
```

如果没有 patch，问用户要应用哪些。完整 patch 细节见 [patches.md](patches.md)。用户有自定义 patch 也可应用并记录下来。

## Step 5: 卸载镜像已有 vllm / vllm-ascend

镜像通常预装 vllm + vllm-ascend（在 site-packages），会干扰 editable 安装：旧版残留导致 import 优先级混乱、版本混入、旧 `.so` 未更新。**编译前必须卸载**：

```bash
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "pip uninstall -y vllm vllm-ascend 2>&1 | tail -5"
# 验证已卸载（pip show 应 not found）
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "pip show vllm vllm-ascend 2>&1 | grep -c 'not found'  # 期望 2"
```

> `pip uninstall -y` 对未安装的包只报 Skipping，安全。多机部署每台机器都要卸载（见 [multi-node-install.md](multi-node-install.md)）。

## Step 6: 展示 editable 编译命令 + 确认参数

展示默认编译命令，问是否需要调整：

**vllm**（纯 Python，无 NPU 编译，约 1 分钟）：

```bash
cd <vllm_path>
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=<ver> pip install -e . --no-build-isolation --no-deps
```

**vllm-ascend**（C++ kernel 编译，5-25 分钟）：

```bash
cd <vllm_ascend_path>
rm -rf csrc/build csrc/output csrc/build_out *.egg-info
MAX_JOBS=32 COMPILE_CUSTOM_KERNELS=1 pip install -e . --no-build-isolation --no-deps
```

问："编译参数是否需要调整？（如 COMPILE_CUSTOM_KERNELS、MAX_JOBS、CPLUS_INCLUDE_PATH 等）"

完整参数细节 + 故障排查见 [build-commands.md](build-commands.md)。

## Step 7: 编译 + 自动修错

先装 vllm（快，约 1 分钟），再装 vllm-ascend（慢，5-25 分钟）。vllm-ascend 编译耗时较长，建议用 `tee` 保存日志：

```bash
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "
bash -l -c '
  cd <vllm_ascend_path>
  rm -rf csrc/build csrc/output csrc/build_out *.egg-info
  MAX_JOBS=32 COMPILE_CUSTOM_KERNELS=1 pip install -e . --no-build-isolation --no-deps \
    2>&1 | tee /tmp/vllm-ascend-editable-compile.log
'
"
```

**`build_aclnn.sh` 失败时用两步编译法**（分两步完成，详见 [build-commands.md](build-commands.md) "方式二"）：

```bash
# Step 1: 深度清理 + 手动编译 CANN 算子
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "
bash -l -c '
  cd <vllm_ascend_path>
  rm -rf csrc/build csrc/output csrc/build_out *.egg-info
  bash csrc/build_aclnn.sh \$(pwd) ascend910_9392
'
"
# Step 2: 跳过算子编译，直接 editable 安装
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "
bash -l -c '
  cd <vllm_ascend_path>
  COMPILE_CUSTOM_KERNELS=0 pip install -e . --no-build-isolation --no-deps
'
"
```

关键要点：
- `build_aclnn.sh` 失败时必须 **深度清理** `csrc/build csrc/output csrc/build_out`（与常规增量重试不同）
- `COMPILE_CUSTOM_KERNELS=0` 跳过 pip 内部的 `build_aclnn.sh` 调用，直接用 Step 1 已编好的算子

**错误处理自主性**：

- **自动修**：缺 pip 包 → `pip install <pkg>`；C++ include 路径 → 调整 CPLUS_INCLUDE_PATH；numpy 版本冲突 → 重装正确版本
- **暂停上报**：vllm-ascend C++ kernel 编译错误（需领域知识）；git 冲突；磁盘空间不足

监控编译输出。出错时：读错误信息 → 已知可修问题（缺依赖、路径问题）自动修 + 重试 → 代码/编译器错误暂停 + 把错误上下文报给用户。

## Step 8: 验证安装成功

三重 check（全部通过才算安装成功）：

**1. editable 指向 worktree**（给定代码路径）：

```bash
# pip list -v 的 Location 列应等于 VLLM_DIR / ASCEND_DIR
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "pip list -v 2>/dev/null | grep -E '^(vllm|vllm-ascend) '"
# 或显式查 Editable 字段
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "pip show vllm vllm-ascend 2>/dev/null | grep -iE '^(Name|Version|Editable|Location):'"
```

> 关键判定：`pip list -v` 输出中 vllm / vllm-ascend 的 **Location 列 == 给定代码路径**（`VLLM_DIR` / `ASCEND_DIR`），即安装目录在代码路径下。若指向 site-packages 或 `/vllm-workspace` 等镜像副本，说明 editable 未生效或镜像版未卸载干净。

**2. C++ 产物 .so 存在**：

```bash
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "ls <vllm_ascend_path>/vllm_ascend/*.so"
# 期望: libvllm_ascend_kernels.so + vllm_ascend_C.cpython-*.so
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "ls <vllm_ascend_path>/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/ 2>/dev/null | head"
```

> **不要用 `find -name '*.so'` 通过 SSH 执行**——它会静默失败返回 0。直接用 `ls`。

**3. import ok**：

```bash
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "python -c 'import vllm; import vllm_ascend; from vllm_ascend import vllm_ascend_C; print(vllm.__version__, \"ok\")'"
```

> **部分 kernel 可能失败（如 36/668）但整体构建仍成功**——失败的 kernel 是非关键的。主 `.so` 文件（`libvllm_ascend_kernels.so`、`vllm_ascend_C.cpython-*.so`）无论 kernel 是否全过都会产出。

## 何时需要重新编译

| 变更类型 | 需要重编？ | 说明 |
| :--- | :--- | :--- |
| `vllm/vllm/` 下的 Python 代码 | ❌ 否 | editable 即时生效 |
| `vllm-ascend/vllm_ascend/` 下的 Python 代码 | ❌ 否 | editable 即时生效 |
| `vllm-ascend/csrc/` 下的 C++ kernel | ✅ 是 | 重编 vllm-ascend + **重启 vllm 进程** |
| `setup.py` / `pyproject.toml` 改动 | ✅ 是 | 重编 |
| vllm 版本变更 | ✅ 是 | 两者都重装 |

> /start-service 启动前用 git diff 检测 `csrc/`/`setup.py`/`pyproject.toml` 改动类型自动判定是否需要重编（见其"代码同步与重编检查"）。

## 编译完成后衔接部署

editable 已注册到 sys.path 指向源码目录，**无需设 PYTHONPATH 指向 dist**。回到 /start-service 正常启动即可（[launch.md](../../start-service/reference/launch.md) SSH 命令模板里的 `PYTHONPATH=$VLLM_DIR:$ASCEND_DIR` 保留亦等效，因 editable 与 PYTHONPATH 指向同一目录）。

如果部署机器与编译机器同类型（同为 a3 或同为 a2），editable 安装在共享 NFS 源码目录上时，部署机器直接 `import vllm` 即可用，无需额外操作。
