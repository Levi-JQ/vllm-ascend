---
name: start-service
description: |
  在远程 itask 机器上启动 vLLM 推理服务。启动前检查代码是否同步、是否需要重新编译安装开发版（git diff 检测改动类型），需要时调用 /build-vllm。
  触发场景：启动推理服务（prefill/decode/conductor）、PD 分离部署、混部部署、connector 启动；检测就绪、验证推理。
  用户提到"启动服务"、"start service"、"启动推理"、"start prefill"、"start decode"、"启动 conductor"、
  "检测就绪"、"验证推理"时使用。
  源码编译/安装（build/compile/源码安装/editable/内网 patch/多机安装）请用 /build-vllm。
compatibility: 依赖 itask exec（多机操作通道，cluster_ctl.sh 内部使用）、scripts/cluster_ctl.sh（多机部署唯一入口）、itask SSH 隧道 + scripts/ssh_tunnel.sh（sync/日志下载，仅 connect）、scripts/sync.sh、scripts/check_ready.sh（单机）、scripts/kill_all.sh、scripts/send_request.sh；重编时调用 /build-vllm
---

# Start Service Skill

把这个 skill 当作"如何在远程 itask 机器上启动 vLLM 推理服务"的执行手册。本 skill **只负责启动**：检查代码同步、判断是否需要重编（需要时委派 `/build-vllm`）、启动服务、检测就绪、验证推理。编译安装的细节全部在 `/build-vllm`。本 SKILL.md 是导航中枢，详细流程下沉到 `reference/` 下各模块，按需加载。

## 变量定义

| 变量 | 值 | 说明 |
|------|-----|------|
| `WS` | `/a3_inference/itask/workdir/yjq02324703/workspace` | 远程工作空间根目录 |
| `TASK` | `<task-name>` | 当前任务名（**任务模式必填**），对应本地 `task/<task>/code/` worktree |
| `CODE_BASE` | 任务模式 `$WS/task/$TASK/code`；默认模式 `$WS/codebases` | 三仓库代码根目录（见下方"代码来源"） |
| `VLLM_DIR` | `$CODE_BASE/vllm` | vllm 代码目录（启动前必须 cd 到此） |
| `ASCEND_DIR` | `$CODE_BASE/vllm-ascend` | vllm-ascend 目录 |
| `TOOLS_DIR` | `$CODE_BASE/vllm-ascend-tools` | 启动脚本目录 |
| `CONNECTOR_DIR` | `$WS/connector` | connector 目录 |

## 代码来源（两种模式）

`CODE_BASE` 决定远程启动用哪份代码，由 sync 阶段决定，`VLLM_DIR/ASCEND_DIR/TOOLS_DIR` 均基于 `CODE_BASE`：

| 模式 | 适用场景 | 创建 worktree | sync 命令 | `CODE_BASE` |
| :--- | :--- | :--- | :--- | :--- |
| **任务模式** | 需要变更分支、改代码的任务 | `scripts/task_worktree.sh new <task>` | `scripts/sync.sh <itask> --task <task> --port <port>` | `$WS/task/<task>/code` |
| **默认模式** | 不改代码的纯部署测试 | 不需要 | `scripts/sync.sh <itask> --port <port>` | `$WS/codebases` |

任务模式下，三仓库的 worktree（分支名统一 `<task>`，基于 codebases 当前 HEAD 切出）同步到远程 `task/<task>/code/{vllm,vllm-ascend,vllm-ascend-tools}`，远程目录结构与本地一致。worktree 生命周期（创建/查看/清理）详见 `/task-worktree` skill。

## 部署脚本 TASK 参数化规范（改到哪个脚本就按此模式改）

启动部署脚本（`vllm-ascend-tools` 各模型目录下的 `kimi-k3-deploy-*.sh` / `start_*.sh`）统一用 `TASK=<task>` 环境变量，把日志/profiling 自动挂到远端任务目录（与本地 `task/<task>/` 对应，不进 code/）：

- 日志：`$WS/task/<TASK>/logs/vllm_node<N>.log`（按节点编号命名）
- profiling：`$WS/task/<TASK>/profiling/`（`/start_profile` 采集、`/stop_profile` 导出 gzip；`--profiler-config` 常开不采集时零开销，可放心保留）

实现模式（加在脚本修改区，`mkdir -p "$LOG_DIR" "$PROFILING_DIR"`，vllm serve 参数加 `--profiler-config "$PROFILER_CONFIG"`）：
```bash
REMOTE_WS=/a3_inference/itask/workdir/yjq02324703/workspace
TASK="${TASK:-<task>}"    # 环境变量可传，默认任务名
LOG_DIR=$REMOTE_WS/task/$TASK/logs
PROFILING_DIR=$REMOTE_WS/task/$TASK/profiling
PROFILER_CONFIG='{"profiler": "torch", "torch_profiler_dir": "'"$PROFILING_DIR"'", "torch_profiler_with_stack": true}'
```

**现状**：多数存量脚本尚未参数化（`LOG_DIR=/tmp` 或日志落在脚本执行目录），**不预先批量改**——任务中需要改到某个脚本时（含只改部署脚本的任务）就地按此模式改，改完 sync 生效。已参数化的参照实现：`task/kimi-k3-prefill-opt/code/vllm-ascend-tools/kimi-k3/mixed/v026_256k/kimi-k3-deploy-4node.sh`。

**IP 自检**（与 TASK 参数化同哲学，改到哪个脚本就地补）：脚本解析出 RANK 对应 `LOCAL_IP` 后，校验本机 `/etc/hosts` 实际 IP 与之一致，不一致即退出——防"改了修改区没换 pod"/"IP 抄错"/误在别的机器以本 RANK 启动（pattern 见 `task/k3-pd-1m-main/code/vllm-ascend-tools/kimi-k3/pd/v026_1m/` 脚本）：

```bash
ACTUAL_IP=$(grep -E '^172\.' /etc/hosts 2>/dev/null | head -1 | awk '{print $1}')
if [ "$ACTUAL_IP" != "$LOCAL_IP" ]; then
    echo "❌ IP 自检失败: 本机实际 IP=${ACTUAL_IP:-未知} ≠ RANK${RANK} 修改区 IP=${LOCAL_IP}"
    echo "   修复（本地优先）: 本地改修改区 NODE IP → scripts/sync.sh 同步 → 重新启动"
    exit 1
fi
```

对 check_ready/日志路径的影响：脚本已参数化 → 日志在 `$WS/task/<TASK>/logs/`；未参数化 → 日志在 `$VLLM_DIR/`（或 `/tmp`），check_ready 相应传 `$VLLM_DIR/<log-file>`。

## 多机部署（cluster_ctl，PD 分离/多节点混部唯一入口）

多机启动/kill/检查**一律用 `scripts/cluster_ctl.sh`**，禁止手工逐台 SSH。职责边界：**agent 只写 plan 文件**（模型/脚本差异全部体现在命令串里），多机纪律（Pod-IP 校验、占用检查、杀干净、并行启动、就绪轮询）由脚本强制执行，agent 无法违反。

plan 文件 `task/<task>/cluster_plan.env`（参照实现 `task/k3-pd-1m-main/cluster_plan.env`）：

```bash
TASK=<task-name>
WS=/a3_inference/itask/workdir/yjq02324703/workspace
CODE_BASE=$WS/task/$TASK/code        # 默认模式: $WS/codebases
# 每项: "组名|机器列表|启动命令|日志模板"；{RANK} 按组内机器顺序替换
CLUSTER_GROUPS=(
  "P|k3-p0,k3-p1,k3-p2,k3-p3|cd $CODE_BASE/vllm && TASK=$TASK RANK={RANK} bash $CODE_BASE/vllm-ascend-tools/kimi-k3/pd/v026_1m/kimi-k3-deploy-4node-prefill.sh|$WS/task/$TASK/logs/vllm_p_node{RANK}.log"
  "D|k3-d0,k3-d1,k3-d2,k3-d3|cd $CODE_BASE/vllm && TASK=$TASK RANK={RANK} bash $CODE_BASE/vllm-ascend-tools/kimi-k3/pd/v026_1m/kimi-k3-deploy-4node-decode.sh|$WS/task/$TASK/logs/vllm_d_node{RANK}.log"
)
```

要点：
- 通信走 `itask exec <机名>` 按名路由（无端口无误连）；SSH 隧道仅在日志下载（rsync）时需要
- launch 前置依赖：代码已 sync（sync 仍走 SSH 隧道）；机器名须已建好且 Running（`itask list`）
- 命令串要点：工作目录 cd 到 `$CODE_BASE/vllm`；`TASK=$TASK` 必传（否则日志落错任务目录）；rank 参数（kimi-k25 的 `bash start_decode.sh {RANK}`）、环境变量（`DATA_PARALLEL_HEAD_ADDRESS`）直接写进命令串；辅助服务（mooncake master 等）追加一个 group
- 混部多节点=单个 group；确需分阶段（如旧脚本 rank0 必须先起）用两次 `launch --only`
- 命令串引用的部署脚本修改区必须写有每台机器的 Pod-IP（launch 静态校验 + 部署脚本运行时 IP 自检双保险，pattern 见下方"部署脚本 TASK 参数化规范"）；换 pod 后本地改修改区 → sync → 再 launch
- 子命令：`launch`（IP 校验+占用检查+kill+并行启动）/ `kill` / `check`（一次性状态）/ `wait`（轮询就绪，节点死亡立即报错）/ `ips`（RANK 表，填 conductor config.yaml）。常用选项：`--only <组>` / `--force-kill`（占用经用户确认后）/ `--skip-ipcheck` / `--dry-run`

## Hard Rules

1. **操作通道分两条**：多机部署一律 `scripts/cluster_ctl.sh`（内部走 `itask exec <机名>` 按名路由，不依赖隧道）；单机混部/conductor 启动走 SSH 隧道模板——**端口被占时脚本自动避让，后续所有 ssh/rsync 必须用脚本实际报告的端口**，并核对脚本打印的**远端实际 IP 与 `itask list` 该机器的 Pod-IP 一致**（不一致 = 隧道连错了机器，立即停手）。SSH 隧道主要用于 rsync 日志下载与交互调试
2. 启动前必须先同步代码：任务模式 `scripts/sync.sh <itask> --task <task> --port <port>`，默认模式 `scripts/sync.sh <itask> --port <port>`（见上方"代码来源"）
3. **任务模式（改代码任务）启动前先创建 worktree**：`scripts/task_worktree.sh new <task>`（基于 codebases 三仓库当前 HEAD 切分支 `<task>`），详见 `/task-worktree`
4. **kill/sync 前必须先做机器占用检查**：`ps -ef | grep "vllm serve"`（看启动时间与命令行）+ `npu-smi info`（看显存占用）。多机走 `cluster_ctl.sh launch` 时此检查已内置——发现占用即中止，**agent 必须把占用情况报给用户确认**，确认可清理才加 `--force-kill`（DP 集群杀一个 node 会级联搞挂整组部署）；单机场景发现活跃服务同样先问用户
5. 单机清理旧进程：`ssh -o StrictHostKeyChecking=no -p <port> root@localhost "bash $WS/scripts/kill_all.sh"`；多机由 `cluster_ctl.sh launch/kill` 内置（逐机 kill_all + 复查归零，不干净绝不启动）
6. 工作目录必须 cd 到 `VLLM_DIR`，日志使用相对路径重定向
7. **必须显式设置** `LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH`（nohup 子进程不继承，否则 `ascend_transport.so` 找不到）
8. 开发版唯一模式是 `/build-vllm` 的 **editable 安装**（`pip install -e .` 已注册 sys.path）；SSH 模板里的 `export PYTHONPATH=$VLLM_DIR:$ASCEND_DIR:...` 保留亦等效（指向同一目录）
9. **ulimit**: conductor 启动前 `ulimit -n 1048576`，vllm 进程启动前 `ulimit -n 65536`
10. **多机部署（PD 分离/多节点混部）一律 `scripts/cluster_ctl.sh launch --plan task/<task>/cluster_plan.env`**，禁止手工逐台 SSH 启动——串行启动会让 DP group 等待超时崩溃。P/D 各组、组内所有节点**同时启动**（脚本本地并行 fork），需要分组先后时用 `--only <组名>`，`--dry-run` 可先审阅将执行的命令。agent 的职责是**写 plan 文件**（见下方"多机部署"节）
11. **启动前进程必须杀干净**：多机由 `cluster_ctl.sh` 内置（逐机 kill_all + 复查 `pgrep -fc 'vllm serve'` 归零后才启动）；单机在启动前手动 `kill_all.sh` 并确认 `ps -ef | grep 'vllm serve'` 为 0。残留进程会导致新进程端口冲突、NPU 显存不足、HCCL 初始化失败等各种报错
12. PD 分离模式默认使用**统一 conductor** `conductor_vllm_ascend_decode_1st_token_linux_arm64`（支持 P-first/D-first，见 [reference/connector.md](reference/connector.md)）；旧 `coord_arm` 已废弃仅作兜底；用户指定 `proxy` 时才使用 proxy
13. 启动后使用 `scripts/check_ready.sh`（单机）或 `scripts/cluster_ctl.sh wait --plan ...`（多机）检测服务是否就绪，不要用 sleep 等待
14. kimi-k25 的 D 节点需设置 `DATA_PARALLEL_HEAD_ADDRESS` 为 D 节点 master IP（多机场景写进 plan 命令串）
15. 混部模式不需要 connector，只有 PD 分离模式才需要
16. 清理进程必须使用 `scripts/kill_all.sh`（多机经 `cluster_ctl.sh kill`），不要手动 `kill -9`
17. 始终使用中文回答
18. **重启 conductor 前**需确认端口 8006 无残留进程：`fuser -k 8006/tcp 2>/dev/null`，否则报 `bind: address already in use`
19. **vllm-ascend 编译后需重启 vllm 进程才能加载新代码**：editable install 下 Python 即时生效，但 C++ 扩展（.so）变更必须重启进程（编译流程见 `/build-vllm`）
20. 启动部署脚本统一用 `TASK=<task>` 环境变量（日志/profiling 挂任务目录，规范见下节；多机 plan 命令串必须带 `TASK=$TASK`）
21. **启动完成后必须向用户报告日志在容器中的绝对路径**：多机部署逐台列出（机器名 + 路径），已参数化脚本为 `$WS/task/<TASK>/logs/vllm_node<N>.log`、未参数化脚本为 `$VLLM_DIR/<log-file>`；并附查看命令 `ssh -p <port> root@localhost "tail -f <日志绝对路径>"`（多机也可 `itask exec <机名> -- tail -f <日志>`），方便用户直接查看实时日志
22. **itask exec 启动后台常驻进程（proxy/conductor/任意 daemon）两铁律**：① `bash -c` payload 里 `pkill -f "<进程名>"` 会匹配 payload 自身 cmdline 而**杀掉自己的 shell**（exit 1 零输出）——按 PID kill 或用 `[k]xxx` 括号转义；② `cd X && setsid nohup ... &` 的 AND-list 后台子壳握着 itask 会话管道等永续服务 → **itask exec 永不返回**——必须 `setsid nohup bash <绝对路径脚本> > 日志 2>&1 < /dev/null &`（简单命令+全重定向，无 cd 链）。proxy 另有目录/日志双坑（模板脚本必 500、就绪行在 pd_proxy.log 不在 proxy.log），详见 [reference/connector.md](reference/connector.md)

## Self-refresh protocol

长对话 context 压缩后规则可能丢失。**每个阶段完成后重新调用本 skill** 刷新规则（尤其是多机 PD 分离部署跨阶段时）。

## Module navigation

根据用户请求选择模块加载：

| 用户说 | 加载 |
| :--- | :--- |
| "启动服务/start service/启动推理/start prefill/start decode" | [reference/launch.md](reference/launch.md) |
| "启动 conductor/connector/coord_arm/proxy" | [reference/connector.md](reference/connector.md) |
| "检测就绪/验证推理/测试推理" | [reference/monitoring.md](reference/monitoring.md) |

辅助参考：
- [models.md](models.md) — 支持的模型列表 + 启动脚本路径 + 端口 + 日志
- [examples.md](examples.md) — kimi-k25 PD 分离、qwen3.5 PD 分离/混部完整命令序列

> 编译/安装开发版（build/compile/editable/内网 patch/多机安装）已拆到 **`/build-vllm`** skill。本 skill 只在启动前判断"是否需要重编"，需要时调用它。

## 收集信息

触发启动服务时，按顺序收集以下信息（用户未提供的必须询问）：

### 必需信息

1. **模型**: 支持的模型列表见 [models.md](models.md)
2. **部署模式**: `PD分离` 或 `混部`（部分模型仅支持 PD 分离）
3. **机器类型**: `A3` 或 `A2`（影响启动脚本路径）
4. **itask 名称和数量**（多机的机器名清单，写进 plan 文件 CLUSTER_GROUPS）:
   - kimi-k25 PD 分离：3 台 itask（1 台 P 节点 + 2 台 D 节点）
   - qwen3.5 PD 分离：2 台 itask（1 台 P 节点 + 1 台 D 节点）
   - 混部：1 台 itask
5. **SSH 隧道端口**（仅 sync/日志下载需要；多机启动/kill/检查走 itask exec 不需要）: 每台机器的端口（默认 27890，多机时依次递增；实际以 `ssh_tunnel.sh` 报告的端口为准——被占自动避让）
6. **是否任务模式（改代码）**: 若本次任务需要修改 vllm/vllm-ascend/vllm-ascend-tools 代码，先用 `scripts/task_worktree.sh new <task>` 创建 worktree，sync 用 `--task <task>`；纯部署测试用默认模式

### 可选信息（有默认值）

7. **connector 类型**: 统一 conductor（默认）或 `proxy`
8. **D 节点 master IP**: PD 分离模式需要，可通过 `itask list` 查看 Pod-IP（多机写进 plan 命令串或部署脚本修改区）
9. **开发版安装状态**: 开发版唯一模式是 `/build-vllm` 的 editable 安装；若远程 `pip show vllm vllm-ascend` 的 Editable 尚未指向 worktree（或仍是镜像预装版），先调用 `/build-vllm` 安装

## 代码同步与重编检查

启动前不仅要确认代码已 sync，还要判断是否需要重新编译安装开发版。判断流程：

### Step A: 确认代码已同步

任务模式 `scripts/sync.sh <itask> --task <task> --port <port>`，默认模式 `scripts/sync.sh <itask> --port <port>`（详见 [reference/launch.md](reference/launch.md) Step 1）。

### Step B: 确认 editable 安装就位

开发版唯一模式是 `/build-vllm` 的 editable 安装（`pip install -e .`）。远程 `pip show vllm vllm-ascend` 检查 Editable 是否指向 worktree（`$CODE_BASE/vllm`、`$CODE_BASE/vllm-ascend`）：
- 已指向 → 进入 Step C 检测改动类型
- 未安装 / 指向错误（如镜像预装版残留）→ 先调用 `/build-vllm` 安装，完成后回到本流程

### Step C: 用 git diff 检测改动类型（仅开发版模式）

在**本地 worktree** 执行（远程 `task/<task>/code/` 无 `.git`，不能在远程跑 git）。先读 `/build-vllm` 写入的构建基线 marker，再 diff C++/构建配置相关路径：

```bash
# 本地 workspace 根目录执行
ASCEND_WT=task/<task>/code/vllm-ascend        # 默认模式: codebases/vllm-ascend
VLLM_WT=task/<task>/code/vllm                 # 默认模式: codebases/vllm
TAG=<task>                                    # 默认模式: default

# 读基线 marker（/build-vllm 每次成功构建后写入 scratch/.build-baseline/）
BASE_A=$(cat scratch/.build-baseline/$TAG.vllm-ascend 2>/dev/null)
BASE_V=$(cat scratch/.build-baseline/$TAG.vllm 2>/dev/null)

if [ -n "$BASE_A" ]; then
  echo "=== vllm-ascend baseline=$BASE_A（上次构建commit）==="
  git -C $ASCEND_WT diff --stat $BASE_A..HEAD -- csrc/ setup.py pyproject.toml  # 已提交改动
else
  echo "=== 无 marker（首次/scratch 被清）→ 检查未提交 + 近期 csrc 提交，并询问用户 ==="
  git -C $ASCEND_WT log --oneline -5 -- csrc/
fi
git -C $ASCEND_WT diff --stat -- csrc/ setup.py pyproject.toml              # 未提交改动（必查）

# vllm 纯 Python，只在版本/构建配置变更时重编
[ -n "$BASE_V" ] && git -C $VLLM_WT diff --stat $BASE_V..HEAD -- setup.py pyproject.toml
git -C $VLLM_WT diff --stat -- setup.py pyproject.toml
```

> **marker 机制**：`/build-vllm` 每次成功构建后在本地 `scratch/.build-baseline/<tag>.{vllm,vllm-ascend}` 写入当时 worktree 的 commit（不 sync、不纳入 git）。本 skill 读取它做 `git diff $BASE..HEAD` 精确判定"构建后是否又改了 C++"。marker 缺失时回退到"未提交 diff + 近期 csrc 提交日志"并询问用户（保守：触及 csrc 即建议重编）。

### 判定

| git diff 结果 | 含义 | 动作 |
| :--- | :--- | :--- |
| 两条 diff 都为空 | 仅改了 Python（或无改动） | **跳过重编**，直接启动（editable 即时生效） |
| 任一非空（触及 `csrc/`、`setup.py`、`pyproject.toml`） | C++ kernel / 构建配置有变更 | **需要重编**，调用 `/build-vllm`（重编 + 刷新 marker + sync），再回到本流程启动 |
| 无 marker 且无法判定 | 首次 / scratch 被清 | 询问用户是否改过 csrc/C++；默认保守建议走 `/build-vllm` |

> 决策树完整版（哪些改动需要重编、editable 下 Python 即时生效原理）见 `/build-vllm` 的 [reference/build.md](../build-vllm/reference/build.md) "何时需要重新编译"。

## 整体流程（启动服务）

```
Start Progress:
- [ ] 0. （任务模式）创建 worktree：scripts/task_worktree.sh new <task>
- [ ] 1. 确认前置条件：sync [--task]（走 SSH 隧道）→ 重编检查（见下）
- [ ] 2. 重编检查（见"代码同步与重编检查"）：判断是否使用开发版 + git diff 检测 csrc/setup.py 改动
      ↻ 需要重编 → 调用 /build-vllm（编译 + 刷新 marker + sync），完成后回到本流程
  ↻ Re-invoke start-service skill to refresh rules
- [ ] 3. 启动推理服务：
      - 多机（PD 分离/多节点混部）：写/更新 plan 文件（"多机部署"节）→ cluster_ctl.sh launch --plan ...
        （占用检查/kill/并行启动由脚本内置；占用中止时报用户确认后 --force-kill）
      - 单机混部：SSH 模板启动（占用检查 + kill_all 由 agent 按步骤执行）
  ↻ Re-invoke start-service skill to refresh rules
- [ ] 4. 启动 Connector（仅 PD 分离模式；配置 IP 可用 cluster_ctl.sh ips 辅助）
  ↻ Re-invoke start-service skill to refresh rules
- [ ] 5. 检测服务就绪：多机 cluster_ctl.sh wait；单机 check_ready.sh
- [ ] 6. 验证推理（send_request.sh）
- [ ] 7. 汇报：逐台给出启动日志在容器中的绝对路径 + tail -f 查看命令（见 Hard Rule 21）
```

完整启动流程见 [reference/launch.md](reference/launch.md)。如果需要编译开发版，先调用 `/build-vllm`（editable 安装），完成后回到本流程启动（editable 已注册，无需额外 PYTHONPATH）。
