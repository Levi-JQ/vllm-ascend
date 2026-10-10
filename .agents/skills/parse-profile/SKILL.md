---
name: parse-profile
description: |
  解析远端 NPU ascend_pt profiling 数据并下载到本地（不含采集）。
  当用户需要解析 ascend_pt、下载 profiling 结果、torch_npu profiler analyse、拉取 profile 时使用。
  触发场景：解析 ascend_pt、下载 profiling 结果、torch_npu profiler analyse、profiling 数据转 CSV、拉取 profile。
  核心规则：每份 profile 只解析拉取单张卡（默认 rank0；PD 分离 = p、d 各一份各一张卡，混布 = 一份一张卡），
  严禁批量解析/下载多卡；收到请求后全链路自动执行（定位 → 选单卡 → analyse → zip → 下载 → 解压），不中途询问。
  如果需要通过 API 采集 profiling（start_profile/stop_profile），先用 collect-profile skill。
compatibility: 需要远程机器 SSH 访问权限（itask exec 或 SSH 隧道）
---

# Parse Profile Skill

把本 skill 当作"如何解析远端 NPU profiling 数据并拉取到本地"的执行手册。

> 本 skill 负责已采集数据的解析和下载。如果需要通过 vLLM API 采集 profiling（start_profile → sleep → stop_profile），见 `collect-profile` skill。

## 核心原则（硬性规则）

### 1. 每份 profile 只处理一张卡

**"份"的定义**：
- **PD 分离**：P 节点的 profiling 目录算一份，D 节点的 profiling 目录算一份 → 共 2 份，每份各选 1 张卡，最终 p、d 各产出一个 zip
- **混布（PD 混部）**：整个 profiling 目录算一份 → 共 1 份，选 1 张卡，最终只产出 1 个 zip

**选卡规则**：默认选 **rank0**（目录名形如 `dp0_pp0_tp0_dcp0_ep0_rank0_<pid>_<timestamp>_ascend_pt`）；仅当用户明确指定 rank 时才选指定卡。算子级/耗时分析用一张卡的数据足够，多卡只会成倍拖慢解析（每张 3-10 分钟）与传输，没有收益。

**严禁批量**：绝不对多个 rank / 多张卡循环 analyse、zip 或下载。列出候选目录后只取一个，其余全部忽略。

### 2. 拉取即自动解析（全链路一次跑完）

用户要求"拉取 / 下载 / 解析 profile"时，不要只做其中一步，也不要中途询问确认，自动连续执行完整链路：

```
定位 ascend_pt 目录 → 每份只选 1 张卡（rank0）→ analyse 解析 → zip → 下载本地 → 解压
```

仅当找不到 ascend_pt 数据、或存在多个 scenario/机器/路径歧义无法自行判断时，才停下来向用户确认。

## 详细步骤

### Step 1: 确定 profiling 位置并选出单卡

profiling 数据在远程机器上，需要知道：
- **哪台机器**：通常通过 itask exec 或 SSH 隧道访问（如 k3-pd-p0-0807b / k3-pd-d0-0807b）。PD 分离时 p、d 数据通常分别在 p0、d0 两台机器上
- **机器路径**：启动服务时 `--profiler-config` 中 `torch_profiler_dir` 指定的目录

**最常见情况：用户直接给出具体到 ascend_pt 的完整路径**（照用即可，跳过定位）：
```
$WS/task/<task-name>/profiling/<scenario>/{p,d}/dp0_pp0_tp0_dcp0_ep0_rank0_<pid>_<timestamp>_ascend_pt
```

**用户只给了 task/scenario 或机器、需要自己定位时**：profiling 数据统一放在**远端 workspace** 下的 `task/<task-name>/profiling/<scenario>/{p,d}`：
```
$WS/task/<task-name>/profiling/<scenario>/{p,d}
```
`$WS` 是该机器上的 workspace 根目录，**典型值** `/a3_inference/itask/workdir/yjq02324703/workspace`（guian itask 机器）。不要硬编码绝对前缀——不同集群/机器的 workspace 根可能不同：
1. 先确认 workspace 根存在（`ls /a3_inference/itask/workdir/yjq02324703/workspace`）
2. 在其下定位 `task/<task-name>/profiling/<scenario>/`
3. 不确定 task/scenario 名时，直接搜 ascend_pt 目录：
   ```bash
   find $WS -maxdepth 7 -type d -name "*ascend_pt" 2>/dev/null
   ```

ascend_pt 子目录命名类似：
```
dp0_pp0_tp0_dcp0_ep0_rank0_<pid>_<timestamp>_ascend_pt
```

**选出单卡（每份只选一个目录）**：
```bash
# 列出该份下所有卡的 ascend_pt 目录（仅查看候选，不批量处理）
ls -d $ASCEND_PT_DIR/*_ascend_pt

# 只取 rank0 一个目录（用户指定 rank 时把 rank0 换成指定值）
ASCEND_PT_SUBDIR=$(ls -d $ASCEND_PT_DIR/dp*_pp*_tp*_dcp*_ep*_rank0_*_ascend_pt | head -1)
```

- **PD 分离**：对 `p/`、`d/` 两个目录各执行一次选卡（各选 1 张，通常都是 rank0），后续步骤对这两个目录各跑一遍——这是"两份"，不是"多张卡"
- **混布**：只有一个 profiling 目录，选 1 张卡，后续步骤只跑一遍
- 如果 rank0 目录不存在（个别 rank 未采集成功），取 `ls` 结果的第一个目录继续，不要试图解析多张卡来"补齐"

### Step 2: 解析 profiling 数据（只解析选定的单卡）

在远程机器上执行 Python 解析脚本。**注意：`cd` 进 ascend_pt 目录内执行 `analyse(path)`，且 path 参数传 ascend_pt 目录的绝对路径**（否则可能生成不完整的 ASCEND_PROFILER_OUTPUT）。

```python
import os
from torch_npu.profiler.profiler import analyse

# 选定的单卡 ascend_pt 目录的绝对路径（$WS 为该机器的 workspace 根）
ascend_pt_path = "$WS/task/xxx/profiling/scenario/p/dp0_pp0_tp0_dcp0_ep0_rank0_xxxxx_ascend_pt"
analyse(ascend_pt_path)
```

解析完成后，`ascend_pt_path` 目录下会生成 `ASCEND_PROFILER_OUTPUT` 子目录，包含：
- `op_statistic.csv` — 算子统计
- `kernel_details.csv` — kernel 详情
- `operator_details.csv` — 算子详情
- `trace_view.json` — trace view（最大文件）
- `communication.json` — 通信统计
- `step_trace_time.csv` — step 耗时
- `analysis.db` — 分析数据库

**注意事项：**
1. 解析需在 ascend_pt 目录**内**执行（`cd ascend_pt_path && python ...`），且 path 参数是绝对路径
2. 解析时间取决于数据量，通常 3-10 分钟（单卡解析一次即可，不要对多卡重复）
3. 解析用 Python 脚本文件更可靠（避免引号转义问题）：
   ```python
   # parse_prof.py
   import sys
   from torch_npu.profiler.profiler import analyse
   path = sys.argv[1]
   print(f"analysing {path}")
   analyse(path)
   print("analyse done")
   ```

### Step 3: 压缩 ASCEND_PROFILER_OUTPUT

解析完成后，将 `ASCEND_PROFILER_OUTPUT` 目录压缩为 zip：

```bash
# 在远程机器上执行
cd <ascend_pt_path>
zip -r /tmp/<task-name>_<scenario>[_<p|d>]_profile.zip ASCEND_PROFILER_OUTPUT/
```

**命名规则**：`<task-name>_<scenario>[_<p|d>]_profile.zip`
- **PD 分离**：P 节点产物带 `_p`，D 节点产物带 `_d`，示例 `feature_test_3_pfc1_dms_0819_p_profile.zip` / `feature_test_3_pfc1_dms_0819_d_profile.zip`
- **混布**：只有一份，**不带 p/d 后缀**，示例 `feature_test_3_pfc1_dms_0819_profile.zip`

**⚠️ 必须完整压缩整个 ASCEND_PROFILER_OUTPUT 目录，禁止排除任何文件**：
- **绝对不要**用 `-x` 排除 `trace_view.json` 或其他大文件。所有文件（含 trace_view.json、*.db 等大文件）都必须包含在 zip 中，确保下载到本地的是完整 profiling 数据。
- 不要自作主张为了"减小体积"而裁剪数据——完整数据是分析的前提，缺失文件会导致后续无法完整分析。
- 如果 zip 体积大，通过 OBS 传输（支持大文件），不要以体积为由排除文件。

### Step 4: 下载到本地

PD 分离时 p、d 两个 zip 都要下载（每份一个 zip、每个 zip 只含一张卡）；混布只有一个 zip。

#### 方式 A：通过 OBS 传输（推荐，跨集群/大文件首选）

华为 OBS 桶 `log`，endpoint `obs.cn-wulan.cloud.wulan-ai.iaas.antgroup-inc.cn`。

**skill 自带 OBS 工具**（`resources/` 目录，**不随本仓库分发**）：
- `obs-trans.sh` — 通用传输脚本（AK/SK 已内置）
- `obsutil-amd64` — x86_64 二进制（5.8.3，WSL/本地用）
- `obsutil-arm64` — aarch64 二进制（5.8.3，A3 NPU 节点用）

> **仓库版注意**：`resources/` 工具目录未随本仓库提交（obs-trans.sh 内置 OBS AK/SK 凭据，obsutil 为大体积二进制）。使用方式 A 前，需从原研究工作空间的 `.zcode/skills/parse-profile/resources/` 自取这些工具，并保持 obs-trans.sh 与所选 obsutil 二进制在同一目录。

**部署 OBS 工具到远程节点**（首次使用时）：
```bash
# 将 skill resources 目录的 obs-trans.sh + obsutil-arm64 上传到远程节点
# 本地路径 = <本地workspace>/.zcode/skills/parse-profile/resources/（本机即 /Users/yujinqi/workspace/.zcode/skills/...）
# 通过 SSH 隧道
scp -P 27890 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  <本地workspace>/.zcode/skills/parse-profile/resources/obs-trans.sh \
  <本地workspace>/.zcode/skills/parse-profile/resources/obsutil-arm64 \
  root@localhost:/tmp/obs-deploy/

# 或通过 inter-pod SSH 从 p0 分发到其他节点
scp -i /root/.ssh/pod_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  /tmp/obs-deploy/obs-trans.sh /tmp/obs-deploy/obsutil-arm64 \
  root@<target-ip>:/tmp/obs-deploy/
```

**上传（在远程机器上执行）**：
```bash
cd /tmp/obs-deploy  # obs-trans.sh + obsutil-arm64 所在目录
sh obs-trans.sh /tmp/<profile.zip>
# 输出: wget --no-check-certificate '<签名URL>' -O <profile.zip>
```

**下载（在本地执行）**：
```bash
# 复制上一步输出的 wget 命令，在本地执行
wget --no-check-certificate '<签名URL>' -O <本地路径>/<profile.zip>
```

**注意**：
- OBS 是只写不读（cp/sign 通，ls 被拒 403）
- 签名 URL 有效期 180 天
- 上传走 obsutil（需内网 endpoint），下载走 wget 签名 URL（任意环境）
- obs-trans.sh 按架构自动选择 obsutil 二进制（x86_64→obsutil-amd64, aarch64→obsutil-arm64）
- 如果远程节点已有 obs-trans（如 p0 的 `/root/obs-trans/`），可直接使用无需重新部署

#### 方式 B：通过 SSH 隧道 scp/rsync（隧道可用时）

如果已建立 SSH 隧道（如 `itask ssh-tunnel <task> --port 27890`）：

```bash
# scp 下载
scp -P 27890 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  root@localhost:/tmp/<profile.zip> <本地路径>/

# 或 rsync 下载（更可靠）
rsync -az -e "ssh -p 27890 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  root@localhost:/tmp/<profile.zip> <本地路径>/
```

#### 方式 C：通过 inter-pod SSH 转发（数据在其他节点时）

如果 profiling 数据在其他节点（如 d0），先通过 p0 的 inter-pod SSH 拉到 p0，再通过 OBS 或隧道下载：
```bash
# p0 上拉取 d0 的数据（inter-pod SSH）
scp -i /root/.ssh/pod_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  root@<d0-ip>:/tmp/<profile.zip> /tmp/<profile.zip>
```

**⚠️ 不要用 `cat | ssh` 方式传输 zip 文件**：之前测试过会导致文件损坏。

### Step 5: 本地保存与解压（自动完成）

本地保存在**本地 workspace** 下的同一相对结构：
```
<本地workspace>/task/<task-name>/profiling/<scenario>/
```
（本地 workspace 即本仓库工作区根，本机为 `/Users/yujinqi/workspace`。）

**下载完成后自动解压**到 zip 包所在目录，解压出的目录名 = zip 包名（去掉 `.zip` 后缀）：
```bash
cd <本地workspace>/task/<task-name>/profiling/<scenario>/
mkdir -p <name>_profile && unzip <name>_profile.zip -d <name>_profile/
# zip 顶层是 ASCEND_PROFILER_OUTPUT/，解压后：<name>_profile/ASCEND_PROFILER_OUTPUT/{op_statistic.csv, kernel_details.csv, trace_view.json, ...}
```

zip 与解压目录都保留（zip 供溯源/重传，解压目录供直接分析）。示例：

PD 分离（两份、每份一张卡）：
```
<本地workspace>/task/kimi-k3-mix-feature/profiling/feature_test_3_pfc1_dms_0819/
├── feature_test_3_pfc1_dms_0819_p_profile.zip
├── feature_test_3_pfc1_dms_0819_p_profile/ASCEND_PROFILER_OUTPUT/...
├── feature_test_3_pfc1_dms_0819_d_profile.zip
└── feature_test_3_pfc1_dms_0819_d_profile/ASCEND_PROFILER_OUTPUT/...
```

混布（一份、一张卡）：
```
<本地workspace>/task/<task-name>/profiling/<scenario>/
├── <scenario>_profile.zip
└── <scenario>_profile/ASCEND_PROFILER_OUTPUT/...
```

## 完整示例

```bash
# 1. 确定 profiling 路径（$WS = 远端 workspace 根，典型 /a3_inference/itask/workdir/yjq02324703/workspace）
# PD 分离示例：p、d 各一份；混布则只有一个目录。每份都只选 rank0 一张卡
ASCEND_PT_DIR="$WS/task/kimi-k3-mix-feature/profiling/feature_test_3_pfc1_dms_0819/p"
ASCEND_PT_SUBDIR=$(ls -d $ASCEND_PT_DIR/dp*_pp*_tp*_dcp*_ep*_rank0_*_ascend_pt | head -1)

# 2. 上传解析脚本
cat /tmp/parse_prof.py | ssh ... "cat > /tmp/parse_prof.py"

# 3. 解析（单卡一次）
ssh ... "cd $ASCEND_PT_SUBDIR && python /tmp/parse_prof.py $ASCEND_PT_SUBDIR"

# 4. 压缩
ssh ... "cd $ASCEND_PT_SUBDIR && zip -r /tmp/feature_test_3_pfc1_dms_0819_p_profile.zip ASCEND_PROFILER_OUTPUT/"

# 5. 下载（<本地workspace> 本机为 /Users/yujinqi/workspace）
scp -P 27890 ... root@localhost:/tmp/feature_test_3_pfc1_dms_0819_p_profile.zip \
  <本地workspace>/task/kimi-k3-mix-feature/profiling/feature_test_3_pfc1_dms_0819/

# 6. 解压到 zip 所在目录，目录名 = zip 包名
cd <本地workspace>/task/kimi-k3-mix-feature/profiling/feature_test_3_pfc1_dms_0819/
mkdir -p feature_test_3_pfc1_dms_0819_p_profile && unzip feature_test_3_pfc1_dms_0819_p_profile.zip -d feature_test_3_pfc1_dms_0819_p_profile/

# 7. PD 分离时对 d 侧重复 1-6（同样只选 rank0）；混布无此步
```

## 常见问题

1. **analyse 返回空 ASCEND_PROFILER_OUTPUT**：确保 `cd` 到 ascend_pt 目录内再执行 analyse，且 path 参数是绝对路径
2. **trace_view.json 过大**：可用 `zip -x "*/trace_view.json"` 排除，或先 trim（去空 cat kernel events）。但注意：默认必须完整打包（见 Step 3 的警告），仅当用户明确同意丢弃 trace 才可排除
3. **解析耗时**：大文件（>1GB ascend_pt）解析可能需 5-10 分钟，用 setsid 后台执行；正因为耗时，才更要坚持每份只解析一张卡
4. **多 rank 数据**：每个 rank 有独立的 ascend_pt 目录，**只解析、压缩、下载 rank0 这一张卡**（用户明确指定的 rank 除外）。绝不要循环处理所有 rank——即使"多拉几张也不费事"的想法也是禁止的，多卡解析/传输耗时和空间都是成倍增加的，且单卡数据已足够做算子级分析
5. **rsync 不可用**：部分容器无 rsync，改用 scp 或 OBS 传输
6. **不要用 cat 管道传输二进制文件**：`cat file | ssh` 会导致 zip 文件损坏，必须用 scp/rsync/OBS