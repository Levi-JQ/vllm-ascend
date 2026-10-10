# Skill: collect-profile

# Profile 采集流程

在 vLLM 推理服务运行期间，通过 API 触发 torch profiler 采集指定时间窗口的 profiling 数据。
采集完成后，解析和下载交给 `parse-profile` skill。

触发场景：用户需要采集 profiling、跑 bench 时抓 profile、start_profile/stop_profile、
profiling 采集、抓取算子耗时分析数据、采集 NPU profile、性能分析数据采集。
即使用户没明确说"请使用 skill"，只要需求本质是"在服务运行时采集一段 profiling 数据"，就应触发本 skill。

## 前置条件

1. **服务已启动**且 `--profiler-config` 已配置（部署脚本中含 `torch_profiler_dir`）
2. **SSH 隧道已建立**（`scripts/ssh_tunnel.sh connect <task> <port>`）
3. **服务端口已知**（部署脚本中的 `PORT`，默认 8000 或 1999）

## 核心流程

```
启动 bench（后台）→ 等待 decode 阶段 → POST /start_profile → sleep N 秒 → POST /stop_profile
```

**关键**：stop_profile 执行完成即意味着 profiling 数据已完整，不需要等 bench 结束。
采集完成后，直接按 `parse-profile` skill 自动解析下载（无需询问用户；每份 profile 只处理单张卡，默认 rank0）。

**PD 分离部署（硬性规则）**：P、D 必须分别独立采集——各自向**自己的 vllm 启动端口**发 start → 各自 sleep（**P 默认 10s、D 默认 3s**）→ 各自 stop，两侧**并行发起**。
**严禁把两侧的 stop 串在同一条命令里顺序执行**：`/stop_profile` 是阻塞调用（等全部 rank dump 完成才返回，16 卡要 2-3 分钟），排在后面的那侧 profiler 会在前一侧 dump 期间持续采集——实测 D 侧名义 2s 实际采了 ~3 分钟，单卡 ascend_pt 达 2.5GB。
**必须真并行，严禁任何串行顺序**：两条序列必须**同一时刻起跑**。agent 工具级"并行调用"可能被顺序执行（实测两条 itask exec 顺序跑：P 先完成 ~90s stop dump、D 才 start，D 侧解码已被排干——3.26s 真窗口仅录到 33 个 lifecycle 事件、零计算事件）。**一律用固化脚本 `scripts/pd_profile_capture.sh <p_node> <d_node> [p_win=10] [d_win=3] [port=8001] [prof_dir]`**——本地并行子进程保证同时起跑，内置采前 D 侧负载自检与采后真实窗口核验。
**采后验收门槛（硬性，不达标必须重采，不得以"图模式事件少"等理由放过）**：该侧 ASCEND_PROFILER_OUTPUT 必须含 `kernel_details.csv` / `op_statistic.csv` / `operator_details.csv` / `step_trace_time.csv` 且非空；缺任一 = 窗口没踩到负载或流程有误 → 删除该轮数据重新采集。

## 详细步骤

### Step 1: 确认服务就绪

```bash
ssh -p <ssh_port> root@localhost "grep -q 'Application startup complete' <log_file> && echo READY || echo NOT_READY"
```

服务未就绪时先等待（用 `scripts/check_ready.sh` 轮询）。

### Step 2: 启动 bench（后台）

```bash
ssh -p <ssh_port> root@localhost 'cd /tmp && nohup python -m vllm.entrypoints.cli.main bench serve \
    --backend vllm --trust-remote-code --model <model_name> \
    --tokenizer <tokenizer_path> --dataset-name random \
    --random-input-len <input_len> --random-output-len <output_len> \
    --ignore-eos --num-prompts <num> --max-concurrency 1 \
    --request-rate 1 --metric-percentiles "50,90,99" \
    --base-url http://localhost:<port> --temperature <temp> \
    > bench_res.txt 2>&1 & echo "Bench PID: $!"'
```

### Step 3: 等待 decode 阶段 + 采集 profile

**等待时间**：bench 启动后需等待服务进入 decode 阶段（通常 10-15s，含初始测试请求 + 首个 prefill）。
等待时间过长无妨——profiler 只在 /start_profile 到 /stop_profile 之间采集。

```bash
ssh -p <ssh_port> root@localhost "
sleep 15                          # 等待 decode 阶段
curl -s -X POST http://localhost:<port>/start_profile
sleep <capture_seconds>           # 采集 N 秒（用户指定，默认 2）
curl -s -X POST http://localhost:<port>/stop_profile
"
```

**参数说明**：
- `<capture_seconds>`：profiling 采集窗口。**单机/混部默认 2 秒**（decode 场景 2s 通常足够覆盖数十步 decode，
  每步包含全部 MoE 层的算子调用，足以做算子级耗时分析）；**PD 分离默认 P=10s、D=3s（见下）**。窗口值不得自行放大/缩小，变更须用户明确指定。
- `<port>`：服务监听端口（**PD 分离时 = 各节点自己的 vllm 启动端口，不是 proxy 端口**——start/stop 只对该节点进程生效）

**PD 分离采集（必须真并行，用固化脚本）**：P 默认 10s（prefill 算子大而稀疏，短窗代表性差），D 默认 3s（decode+dspark 每秒事件量巨大，3s 已覆盖大量步数且数据量可控）。**一律用固化脚本**（本地并行子进程保证两条序列同一时刻起跑、互不等待）：

```bash
# 首选：固化脚本（内置 D 侧负载自检 + 采后真实窗口核验）
scripts/pd_profile_capture.sh k3-p0 k3-d0        # P 10s / D 3s，端口 8001，profiling 目录自动探测
# 手工等价（仅脚本不可用；须两个真并行终端同时执行——agent 工具级"并行调用"实测可能被顺序执行）：
#   P 侧（p0）: curl -s -m 600 -X POST http://localhost:<p_port>/start_profile; sleep 10; curl -s -m 600 -X POST http://localhost:<p_port>/stop_profile
#   D 侧（d0）: curl -s -m 600 -X POST http://localhost:<d_port>/start_profile; sleep 3;  curl -s -m 600 -X POST http://localhost:<d_port>/stop_profile
```

> 时序原理：start 的 curl 在 profiler 初始化完成时返回（≈就绪）→ D 窗 [t0+~3s, t0+~6s]、P 窗 [t0+~8s, t0+~18s]；D 的 dump（秒级）落在 P 窗口内、P 的 dump（~1.5min）落在两窗之后——不存在"P dump 排空 D"的时间线。
> stop_profile 的 curl 会阻塞 1-2 分钟等 dump，正常；两侧并行时各自 dump 互不影响。
> **反例（严禁）**：①把 4 个调用串成一条命令 `start_p; start_d; sleep N; stop_p; stop_d`——stop_d 要等 stop_p 的 dump 完成才发出，D 实际窗口 = N + P 侧 dump 时长（实测名义 2s 实采 ~3min、单卡 2.5GB）；②**任何 P→D 串行顺序**——P 的 stop dump 阻塞 P 引擎 1-2min→无新 KV 下发→D 解码排空，D 窗口踩空转期（实测 3.26s 真窗口仅 33 个 lifecycle 事件、零计算事件、解析缺全部算子 CSV，当时 d0 `Running: 1 reqs / 2.0 tokens/s`）。
> **采集前负载自检**：看该节点日志最近的 `Running: N reqs` 行，N≥2 且 Avg generation throughput 明显>0 才 start（P 侧看 prompt/输入吞吐活跃）。
> **采集后验收**：解析产物必须含 kernel_details.csv/op_statistic.csv/operator_details.csv/step_trace_time.csv 且非空，缺任一即删除重采（详见核心流程的验收门槛）。

### Step 4: 确认数据采集完成

`/stop_profile` 返回非 error 响应即表示数据已完整。检查 profiling 目录有 ascend_pt 子目录生成：

```bash
ssh -p <ssh_port> root@localhost "ls <profiling_dir>/ | head -5"
```

ascend_pt 目录命名格式：`dp<N>_pp0_tp<N>_dcp0_ep<N>_rank<N>_<pid>_<timestamp>_ascend_pt`

### 后续：解析和下载 profiling 数据（自动执行，无需询问）

采集完成后，ascend_pt 数据在远端 `<profiling_dir>` 下。**不要询问用户是否需要解析下载**，直接按 `parse-profile` skill 全链路自动执行（定位 → 选单卡 → analyse → zip → 下载 → 解压）：

- **每份 profile 只处理一张卡**（默认 rank0）：PD 分离 = p、d 各一份，数据通常分别在 p0/d0 机器上；混布 = 整个目录一份
- **严禁批量解析/下载多张卡**——每份只取一个 ascend_pt 目录（rank0），其余 rank 全部忽略

## 常见问题

### PD 分离两侧 stop 串行 → 后一侧窗口爆炸（GB 级数据）

`/stop_profile` 是阻塞调用（等全部 rank dump 完成，16 卡约 2-3 分钟）。若把两侧 stop 串行执行，后一侧的 profiler 会在前一侧 dump 期间持续采集——名义 2-5s 实际 ~3 分钟，单卡 ascend_pt 达 2.5-3.6GB。预防：P/D 并行、各自独立 start/sleep/stop（见 Step 3）；已发生：删除该轮数据重采。

### /start_profile 返回空响应

curl 返回空字符串（HTTP 200 无 body）——**这是正常的**，profiler 已启动。
确认方法：检查 profiling 目录下是否开始生成 ascend_pt 子目录及数据文件。

### /stop_profile 返回 "Profiler must be initialized"

表示 `/start_profile` 未成功初始化 profiler。可能原因：
1. 服务未配置 `--profiler-config`（检查部署脚本是否含 `--profiler-config`）
2. `/start_profile` 调用时机太早（服务还在初始化，未进入推理阶段）
3. 调用了两次 `/start_profile`（第二次会 hang）

**解决**：确认 `--profiler-config` 已配置，确保只调用一次 `/start_profile`，
且在服务就绪 + bench 开始发送请求后调用。

### /start_profile 卡住（curl 无响应）

通常因重复调用 `/start_profile` 导致。**杀掉卡住的请求**，确认只调用一次。
若服务本身卡死，需 `kill_all.sh` 重启服务后重新采集。

### 需要手动停止 profiler（/stop_profile 不可用）

杀掉服务进程会触发 profiler flush——但这是 fallback，优先用 `/stop_profile` API：

```bash
ssh -p <ssh_port> root@localhost "bash /tmp/k.sh"  # kill_all.sh
```

kill 后 ascend_pt 数据会自动 finalize，可正常解析。

## 完整示例

```bash
# 参数
SSH_PORT=27927
SERVICE_PORT=1999
PROFILING_DIR=/a3_inference/itask/workdir/yjq02324703/workspace/task/<task>/profiling/<scenario>
CAPTURE_SECONDS=2

# 1. 启动 bench
ssh -p $SSH_PORT root@localhost 'cd /tmp && nohup python -m vllm.entrypoints.cli.main bench serve ... > bench_res.txt 2>&1 & echo "PID: $!"'

# 2. 等待 decode + 采集
ssh -p $SSH_PORT root@localhost "sleep 15; curl -s -X POST http://localhost:$SERVICE_PORT/start_profile; sleep $CAPTURE_SECONDS; curl -s -X POST http://localhost:$SERVICE_PORT/stop_profile"

# 3. 确认 ascend_pt 生成
ssh -p $SSH_PORT root@localhost "ls $PROFILING_DIR/ | head -5"

# 后续：解析 + 下载（自动执行，每份只取单卡 rank0）→ 见 parse-profile skill
```
