# 多机安装同一版本 vllm + vllm-ascend

多机部署（如 PD 分离 8P+8D）需所有机器装同一版本 vllm + vllm-ascend。**不要每台全量编译**——先在一台 build，传递产物，其他机器快速安装。**PYTHONPATH 覆盖 ≠ 安装**（镜像 editable install 仍在 site-packages，混入风险），必须 pip install editable。

## Workflow checklist

```
Multi-node Install:
- [ ] 1. 选 build 机（如 p0）+ 确认 CANN / setuptools_rust
- [ ] 2. p0 装 vllm（empty 纯Python）+ vllm-ascend（COMPILE_CUSTOM_KERNELS=1 全量build一次）
- [ ] 3. 传递产物到其他机（vllm代码 + vllm-ascend代码+.so+_cann_ops_custom+_build_info.py，不含csrc/build）
- [ ] 4. 其他机装 setuptools_rust → 装 vllm（empty）→ 装 vllm-ascend（COMPILE_CUSTOM_KERNELS=0 跳过build_aclnn，秒级）
- [ ] 5. check 报告：所有机 pip show + import + setup.py md5一致 + .so加载
```

## Step 1: build 机准备

确认 CANN toolkit + setuptools_rust（vllm setup.py 顶层 `from setuptools_rust.build import build_rust`，镜像可能缺），并卸载镜像预装 vllm/vllm-ascend 避免干扰 editable 安装：

```bash
ls /usr/local/Ascend/ascend-toolkit/latest/  # CANN
python -c "import setuptools_rust; print(setuptools_rust.__version__)"  # 缺则装
pip install setuptools_rust -i https://mirrors.aliyun.com/pypi/web/simple
pip uninstall -y vllm vllm-ascend  # 卸载镜像预装版，避免 editable 被旧版干扰
```

## Step 2: p0 安装（build 一次）

**vllm**（empty target，纯 Python，无需 rustc/cmake）：

```bash
cd <vllm_source>
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=<ver> pip install -e . --no-build-isolation --no-deps
```
- `VLLM_TARGET_DEVICE=empty`：setup.py `if _no_device(): ext_modules=[]`（不 build _C），rust `optional`（无 rustc 跳过）
- `VLLM_VERSION_OVERRIDE`：避免 setuptools-scm 无 .git 算错版本
- **需 setuptools_rust**（否则 setup.py import 失败）

**vllm-ascend**（全量 build .so 一次，~25min）：

```bash
cd <vllm_ascend_source>
MAX_JOBS=32 COMPILE_CUSTOM_KERNELS=1 pip install -e . --no-build-isolation --no-deps
```
- `MAX_JOBS=32`：避免 opc 死锁（默认 -j640 死锁）
- build 产物：`vllm_ascend/libvllm_ascend_kernels.so` + `vllm_ascend/vllm_ascend_C.cpython-312-*.so` + `vllm_ascend/_cann_ops_custom/`（~67M）+ `vllm_ascend/_build_info.py`（`__device_type__='A3'` for ascend910_9391）

## Step 3: 传递产物到其他机

```bash
rsync -az --delete \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='csrc/build' --exclude='*.egg-info' \
  -e "ssh <key/opts>" \
  <vllm_source>/ <peer>:<vllm_source>/
rsync -az --delete \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='csrc/build' --exclude='*.egg-info' \
  -e "ssh <key/opts>" \
  <vllm_ascend_source>/ <peer>:<vllm_ascend_source>/
```

**⚠️ rsync 陷阱**：
- `--exclude='build'` 误删源码 `csrc/cmake/third_party/build/`（含 protobuf-hide_absl_symbols.patch）→ build_aclnn abseil patch 失败。**用 `--exclude='csrc/build'`（精确路径）**。
- **不要 `--exclude='*.so'` / `--exclude='_cann_ops_custom'`**——这些是 build 产物，要传递！只排除 `csrc/build`（每次 build_aclnn 会 `rm -rf` 重建，传递无用）+ `.git` + `__pycache__`。
- `--delete` 会删 build 生成的 `_build_info.py` → 传递后若缺失，重建：`echo "__device_type__ = 'A3'" > vllm_ascend/_build_info.py`

## Step 4: 其他机快速安装（COMPILE_CUSTOM_KERNELS=0 跳过 build_aclnn）

**关键**：`build_aclnn.sh` 每次 pip install -e 都 `rm -rf csrc/build` 全量重建 opc（~25min）。传递 csrc/build 缓存**无用**（被删）。用 `COMPILE_CUSTOM_KERNELS=0`（setup.py:440 `if envs.COMPILE_CUSTOM_KERNELS: run_command("build_aclnn")`）跳过 build_aclnn，直接用已传递的 .so，秒级安装。

```bash
# 其他机
pip install setuptools_rust -i https://mirrors.aliyun.com/pypi/web/simple  # 镜像可能缺
pip uninstall -y vllm vllm-ascend  # 卸载镜像预装版，避免 editable 被旧版干扰

# vllm（empty，纯Python）
cd <vllm_source>
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=<ver> pip install -e . --no-build-isolation --no-deps

# vllm-ascend（跳过build_aclnn，用已有.so）
cd <vllm_ascend_source>
SETUPTOOLS_SCM_PRETEND_VERSION=<ver> COMPILE_CUSTOM_KERNELS=0 pip install -e . --no-build-isolation --no-deps
```
- `SETUPTOOLS_SCM_PRETEND_VERSION=<ver>`：其他机无 .git，setuptools-scm 会算成 0.0.0/0.1.dev0。用此环境变量假装版本（比 git init+tag 可靠，git add -A 大目录可能超时）。
- `COMPILE_CUSTOM_KERNELS=0`：跳过 build_aclnn，editable 注册指向 worktree（含已传递 .so），秒级。

## Step 5: check 报告（所有机必过）

```bash
WS=<worktree_code_root>
VV=$(pip show vllm|grep '^Version'|awk '{print $2}')       # 期望 <ver>
VL=$(pip show vllm|grep '^Editable'|awk '{print $NF}')      # 期望 <WS>/vllm 非 /vllm-workspace
AV=$(pip show vllm-ascend|grep '^Version'|awk '{print $2}')  # 期望 <ver>
AL=$(pip show vllm-ascend|grep '^Editable'|awk '{print $NF}')# 期望 <WS>/vllm-ascend 非 /vllm-workspace
IMP=$(python -c "import vllm;print(vllm.__version__)" 2>/dev/null|tail -1)  # 期望 <ver>
VMD=$(md5sum $WS/vllm/setup.py|awk '{print $1}')            # 所有机一致 = 同一commit
AMD=$(md5sum $WS/vllm-ascend/setup.py|awk '{print $1}')     # 所有机一致 = 同一commit
SO=$(python -c "from vllm_ascend import vllm_ascend_C;print('ok')" 2>/dev/null|tail -1)  # ok
```

check 报告要素：
- **安装成功**：所有机 pip show Version=<ver> + Editable→worktree（非 /vllm-workspace 镜像版）+ import=<ver> + .so=ok
- **代码同一 commit**：所有机 `setup.py md5` 一致（= 本地 worktree 基准 md5）。worktree 无 .git 无法 `git rev-parse`，用 md5 证明代码内容相同。记录本地基准 commit SHA（如 vllm-ascend 3e4325362）+ md5 比对。

## 长任务 detach（build 用）

build/编译用 `setsid`（nohup 在 itask exec 下不够，exec 会话关闭会杀进程）：

```bash
setsid bash -c "MAX_JOBS=32 pip install -e . --no-build-isolation --no-deps > /tmp/build.log 2>&1" < /dev/null > /dev/null 2>&1 & disown
```

## 切换特性/分支（csrc-clean PR）

PR 只改 Python（csrc 不变）时，.so 通用，切换只需 rsync Python 代码（editable 自动生效，不重 build）：

```bash
rsync -az --exclude='csrc/build' --exclude='*.so' --exclude='_cann_ops_custom' ...  # 只更新Python
```

若 PR 改 csrc（罕见），需 p0 重 build .so + 重新传递 + 其他机 COMPILE_CUSTOM_KERNELS=0 重装。
