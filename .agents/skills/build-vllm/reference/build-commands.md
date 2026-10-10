# 编译命令参考（editable 模式）

## 前置：卸载镜像已有 vllm / vllm-ascend

镜像预装版会干扰 editable 安装（import 优先级、版本混入、旧 .so 残留），编译前必须卸载：

```bash
pip uninstall -y vllm vllm-ascend
# 验证（应 not found）
pip show vllm vllm-ascend 2>&1 | grep -c 'not found'  # 期望 2
```

> `pip uninstall -y` 对未安装的包只报 Skipping，安全。多机部署每台都要卸载。

## vllm（纯 Python，约 1 分钟）

```bash
cd <vllm_source>
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=<ver> pip install -e . --no-build-isolation --no-deps
```

- `VLLM_TARGET_DEVICE=empty`：setup.py `if _no_device(): ext_modules=[]`（不 build _C），跳过 GPU/CUDA 编译（vllm 纯 Python）
- `VLLM_VERSION_OVERRIDE=<ver>`：避免 setuptools-scm 无 .git 算错版本（worktree 有 .git 一般不需要，远程 sync 副本无 .git 时需要）
- `pip install -e .`：editable 模式，Python 代码修改后不需重装
- `--no-build-isolation`：使用当前环境而非隔离环境构建
- `--no-deps`：不装依赖（镜像里已有）
- `-i https://pypi.antfin-inc.com/simple/`：需要拉依赖时用内网 PyPI 镜像
- **需 setuptools_rust**（setup.py 顶层 `from setuptools_rust.build import build_rust`），缺则装：
  ```bash
  pip install setuptools_rust -i https://mirrors.aliyun.com/pypi/web/simple
  ```

## vllm-ascend（C++ kernel 编译，5-25 分钟）

### 方式一：标准 editable 编译（推荐）

```bash
cd <vllm_ascend_source>
rm -rf csrc/build csrc/output csrc/build_out *.egg-info
MAX_JOBS=32 COMPILE_CUSTOM_KERNELS=1 pip install -e . --no-build-isolation --no-deps
```

- `pip install -e .`：editable，Python 代码修改即时生效（C++ 扩展变更除外）
- `COMPILE_CUSTOM_KERNELS=1`：编译自定义 CANN 算子（`build_aclnn.sh`）
- `MAX_JOBS=32`：避免 opc 死锁（默认 `-j640` 死锁）
- 产物：`vllm_ascend/libvllm_ascend_kernels.so` + `vllm_ascend/vllm_ascend_C.cpython-*.so` + `vllm_ascend/_cann_ops_custom/`（~67M）+ `vllm_ascend/_build_info.py`

> **⚠️ `build_aclnn.sh` 在 `pip install -e .` 内部调用时经常因 CMake 缓存残留而失败**。如果遇到此问题，改用下方"方式二：两步编译法"。

### 方式二：两步编译法（build_aclnn.sh 失败时使用）

当方式一的 `pip install -e .` 因 `build_aclnn.sh` 失败而中断时，分两步完成：

**Step 1：手动编译 CANN 算子**

```bash
cd <vllm_ascend_source>
rm -rf csrc/build csrc/output csrc/build_out *.egg-info
bash csrc/build_aclnn.sh $(pwd) ascend910_9392
```

**Step 2：跳过算子编译，直接 editable 安装**

```bash
cd <vllm_ascend_source>
COMPILE_CUSTOM_KERNELS=0 pip install -e . --no-build-isolation --no-deps
```

- `COMPILE_CUSTOM_KERNELS=0`：跳过 `build_aclnn.sh`（Step 1 已手动完成），直接用已编好的算子，秒级注册 editable
- 其他机无 .git，用 `SETUPTOOLS_SCM_PRETEND_VERSION=<ver>` 假装版本（见 [multi-node-install.md](multi-node-install.md)）

**验证**：

```bash
python -c "import vllm; import vllm_ascend; from vllm_ascend import vllm_ascend_C; print(vllm.__version__); print('OK')"
```

## 编译参数说明

- `SOC_VERSION=ascend910_9392`：目标 NPU SOC（9392 = 910B3）
- `unset CPLUS_INCLUDE_PATH`：让 CMake 找到正确的 include 路径（CANN 工具链自带 GCC 7.3 头文件会与系统 GCC 12 冲突时）
  - **guian 镜像勿设 CPLUS_INCLUDE_PATH**：会泄漏 gcc12 std header 进 `build_aclnn` 的 Ascend 编译器，致 `c++config.h` 解析失败
- `MAX_JOBS=32`：opc 并行度，避免死锁
- `--no-deps`：不装依赖（镜像里已有）
- `--no-build-isolation`：用当前环境构建
- `-v`：详细输出，便于排错
- `-i https://pypi.antfin-inc.com/simple/`：内网 PyPI 镜像
- `2>&1 | tee compile.log`：捕获完整输出便于排错

**增量重试**：编译失败/卡住时，杀进程后重试，**不要删 `csrc/build/`**——CMake 会跳过已编译的 kernel，只重建失败/剩余的。只需 `rm -rf *.egg-info`（不删 `csrc/build/`）。

> **注意**：`build_aclnn.sh` 失败的情况例外——此时需要 `rm -rf csrc/build csrc/output csrc/build_out *.egg-info` 做深度清理，因为 aclnn 的 CMake 缓存会导致重试仍然失败。

## 常见编译错误

| 错误 | 原因 | 修复 |
| :--- | :--- | :--- |
| `Neither 'setup.py' nor 'pyproject.toml' found` | 目录不对（不在仓库根） | `cd` 到正确的仓库路径 |
| `CMake Error: CMakeCache.txt directory is different` | 上次在不同路径编译留下的陈旧 CMake 缓存 | 重编前 `rm -rf csrc/build csrc/CMakeCache.txt` |
| `FAILED: [code=1] ... Killed`（kernel 生成阶段） | OOM — nocard pod（32GB）不足以生成 kernel 二进制 | 改用 16-card pod（1600GB RAM） |
| `Parse error. Expected a command name, got "<<<<<<<"` | `git apply --3way` 留下的合并冲突标记 | `grep -rl '<<<<<<<' csrc/` → 清理冲突标记（见 [patches.md](patches.md)） |
| `Failed to download <package>` | 外部 URL（gitcode.com/gitee.com）被内网拦截 | 应用 patch（见 [patches.md](patches.md)）— 替换为内网 OSS 镜像 |
| `opc tool start working now` 后卡住 | opc tool 在特定 kernel 上死锁 | 杀 + 重试（增量，保留 `csrc/build/`）；或 `itask stop/start` 重启 pod |
| `disk full` | 构建产物过大 | 编译前清理旧产物 + `/tmp/*` |
| `numpy version conflict` | vllm 拉到不兼容的 numpy | `pip install numpy==1.26.4` |
| `build_aclnn.sh` 返回非零退出码 | CMake 缓存残留或增量构建状态不一致 | **深度清理**：`rm -rf csrc/build csrc/output csrc/build_out *.egg-info`；然后用方式二两步编译法 |
| 算子缺失 / `_C_ascend` 无属性 | 代码同步后未重编 vllm-ascend，C++ 算子未编译 | 重新 `pip install -e .` |
| `c++config.h` 解析失败（guian） | 误设 `CPLUS_INCLUDE_PATH` 泄漏 gcc12 std header | `unset CPLUS_INCLUDE_PATH` 后重编 |

## 验证安装成功

三重 check（全部通过才算成功）：

**1. editable 指向 worktree**（给定代码路径）：

```bash
# pip list -v 输出 3 列：Package Version Location —— Location 列应等于 VLLM_DIR / ASCEND_DIR
pip list -v 2>/dev/null | grep -E '^(vllm|vllm-ascend) '
# 或显式查 Editable 字段
pip show vllm vllm-ascend 2>/dev/null | grep -iE '^(Name|Version|Editable|Location):'
```

> 关键判定：vllm / vllm-ascend 的安装目录（Location / Editable project location）== 给定代码路径。若指向 site-packages 或镜像副本路径，说明 editable 未生效或镜像版未卸载干净。

**2. C++ 产物 .so 存在**：

```bash
# 检查 vllm-ascend 源码树下的 .so
ls <vllm_ascend_source>/vllm_ascend/*.so
# 应看到: libvllm_ascend_kernels.so + vllm_ascend_C.cpython-*.so
# 检查 _cann_ops_custom（编译好的 kernel 二进制）
ls <vllm_ascend_source>/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/ 2>/dev/null | head
```

> **不要用 `find -name '*.so'` 通过 SSH 执行**——它会静默失败返回 0。直接用 `ls`。

**3. import ok**：

```bash
python -c "import vllm; import vllm_ascend; from vllm_ascend import vllm_ascend_C; print(vllm.__version__, 'ok')"
```

**部分 kernel 可能失败（如 36/668）但整体构建仍成功**——失败的 kernel 是非关键的。主 `.so` 文件（`libvllm_ascend_kernels.so`、`vllm_ascend_C.cpython-*.so`）无论 kernel 是否全过都会产出。
