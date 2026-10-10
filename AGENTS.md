# AGENTS.md

## 项目概述

vLLM 昇腾（NPU）推理研发工作空间，聚焦 DeepSeek、Kimi、Qwen 等模型在昇腾硬件上的部署与优化。采用**本地开发 + 远程调试**模式：本地编辑代码，通过 `itask` 管理 A3/A2 远程机器（操作走 `itask exec` 按名路由，日志下载走 SSH 隧道）进行调试。

**任务制工作流**：每个任务在 `task/<task-name>/` 下独立展开——代码走 git worktree 分支隔离、editable 安装指向任务目录、日志/profiling 归档到任务目录、进度记录在 progress.md。多任务并行互不干扰。

## 工作空间结构

```
workspace/
├── codebases/                        # 代码仓库（各自独立 git，源代码唯一主工作树）
│   ├── vllm/                         # vLLM 上游主仓
│   ├── vllm-ascend/                  # vLLM 昇腾插件（NPU 适配层，submodule catlass）
│   └── vllm-ascend-tools/            # 推理部署脚本（按模型组织，含 common.sh）
├── task/                             # 任务目录（一个任务一个子目录）
│   └── <task-name>/                  # 任务英文名（kebab-case）
│       ├── code/                     # 三仓库 worktree @ branch <task-name>（git 忽略）
│       ├── logs/                     # 本地日志归档：每轮实验从远端下载，按场景子目录（git 忽略）
│       ├── profiling/                # 本地 profiling 归档：zip 按场景子目录（git 忽略）
│       └── progress.md               # 任务进度（git 跟踪，每轮记录日志位置）
├── connector/                        # PD Conductor（git 跟踪，config.yaml 按实际 IP 修改）
├── scripts/                          # 研发脚本（git 跟踪）
│   ├── task_worktree.sh              # 任务 worktree 管理（new/list/info/remove）
│   ├── sync.sh                       # rsync 同步代码到远程（默认 / --task 模式）
│   ├── ssh_tunnel.sh                 # itask SSH 隧道（端口自动避让 + 远端身份核验 + 残留会话清理）
│   ├── cluster_ctl.sh                # 多机部署编排（launch/kill/check/wait/ips，模型无关）
│   ├── kill_all.sh                   # 清理 python/vllm/conductor 进程（唯一清理方式）
│   ├── check_ready.sh                # 轮询检测服务启动完成（单机）
│   ├── send_request.sh               # 测试推理 API
│   └── profile.sh                    # 控制 vLLM profiling
├── docs/markdown/                    # 文档（issues / knowledge / assets）
├── scratch/                          # 临时调试目录（git 忽略）
│   └── .build-baseline/              # 构建基线 marker（供 /start-service 判定重编）
└── data/                             # 测试数据（git 忽略）
```

## 任务工作流（标准生命周期）

> **核心原则：所有修改本地优先，远端只跑不改**。vllm / vllm-ascend 代码、vllm-ascend-tools 脚本的任何改动，必须先在本地完成——任务级改动在 `task/<task>/code/` 对应 worktree 里改（含只改部署脚本的情况），非任务级在 `codebases/` 改——然后经 `sync.sh` 同步到远端。禁止直接在远端改文件：下次 sync 会覆盖、git 无法追溯。远端只负责编译安装、运行、以及把产物（日志/profiling）下载回本地。

0. **（改代码任务）创建 worktree** → `scripts/task_worktree.sh new <task-name>`（三仓库，分支 `<task-name>`，基于 codebases 当前 HEAD，`--base` 可指定基础分支）。纯部署测试跳过此步
1. **创建 itask** → 用户自行创建并提供容器名（可能多个），模型不负责创建 itask，未提供需询问
2. **（需要隧道时）建立 SSH 隧道** → `scripts/ssh_tunnel.sh connect <task> [port]`（脚本内置防护：端口被占自动避让并打印实际端口；连接后自动清理远端其他 ssh 会话）。隧道仅用于 rsync 日志下载与交互调试；**启动/kill/状态检查一律 `itask exec <机名>` 按名路由，不依赖隧道**（多机部署编排 `cluster_ctl.sh` 内部即走 itask exec）
   - 确需隧道时，后续所有 ssh/rsync **必须用脚本报告的端口**，并核对脚本打印的**远端实际 IP 与 `itask list` 该机器 Pod-IP 一致**——不一致=隧道连错了机器，立即停手（曾因未核对把操作全落到别人机器上）
2.5 **机器占用检查（kill/sync 前必须做）** → 每台机器执行 `ps -ef | grep "vllm serve"`（看启动时间与命令行）+ `npu-smi info`（看显存占用）。多机走 `cluster_ctl.sh launch` 时此检查已内置（发现占用即中止，确认可清理后加 `--force-kill`）：
   - 发现**近期启动的活跃服务**（属于其他任务）→ 必须先问用户确认，绝不能直接 kill；DP 集群杀一个 node 会级联搞挂整组部署
   - 只有确认是本任务的陈旧残留（如 2 天前的实验遗留进程）才能清理
   - 判断占用看进程与 NPU 状态，不靠 IP 猜：Pod-IP 与机器一一对应不会冲突，**易错的环节是隧道路由**（核对见步骤 2）
3. **同步代码** → 任务模式 `scripts/sync.sh <itask> --task <task-name> --port <port>`；默认模式 `scripts/sync.sh <itask> --port <port>`（同步 codebases 主分支，无 worktree）
4. **编译安装** → `/build-vllm` skill（要点见下节）：先 vllm（empty，纯 Python）再 vllm-ascend（C++ kernel）；同 commit 已有产物可直接复用秒级安装
5. **启动推理** → `/start-service` skill，任务模式 `CODE_BASE=$WS/task/<task>/code`，**工作目录必须是 `$CODE_BASE/vllm`**。部署脚本通过 `TASK=<task>` 环境变量把日志/profiling 自动挂到远端 `task/<task>/{logs,profiling}/`；**多机（PD 分离/多节点混部）一律 `scripts/cluster_ctl.sh launch --plan task/<task>/cluster_plan.env`**（agent 只负责写 plan 文件，IP 校验/占用检查/杀干净/并行启动由脚本强制），禁止手工逐台 SSH 启动
6. **验证** → `scripts/check_ready.sh <task> <log_file> [port] [max_wait] [ssh_port]` 轮询等待（日志出现 `Application startup complete` 即成功）+ `scripts/send_request.sh` 冒烟
7. **实验/压测/profiling** → 每轮结束立即下载日志（硬性要求，见"日志与 profiling 规范"）
8. **停止/重启** → 多机用 `scripts/cluster_ctl.sh kill --plan ...`，单机 `scripts/kill_all.sh`（唯一清理方式）；重启 = stop + sync + start
9. **（任务完成）保留 worktree** → 默认保留不删除，便于追溯任务改了什么代码；仅在明确不需要时才 `scripts/task_worktree.sh remove <task-name>` 清理

### Worktree 机制

- **任务级修改在 worktree 中做**：任务期间对 vllm/vllm-ascend 代码或 vllm-ascend-tools 部署脚本的一切修改（包括临时调参、改启动脚本），都在 `task/<task>/code/` 的对应 worktree 里改，改完 `sync.sh --task` 同步远端生效；不要把任务改动做在 codebases 主工作树（会串到其他任务），更不要直接改远端文件
- **源代码唯一性**：所有任务源代码只有 `codebases/` 下一份（主工作树），task worktree 只是某个分支的工作树；主工作树始终保持原分支不动
- **同步**：`sync.sh --task <task>` 只同步共享顶层文件 + 该任务目录（其他任务目录与 profiling/logs 归档不上传，日志只从远端下载），代码三仓库按内容比对（`-c`，与 mtime 无关）并 `--link-dest` 硬链接复用远端已有任务代码——新任务只传分支差量（实测三仓库 ~6MB 而非 ~160MB）；引用任务自动选远端最新，`--link-dest-task <t>` 可指定。默认不带 `--delete`，已拷入的编译产物不会被冲掉（`--force` 的删除仅作用于该任务目录）
- **同步后自动校验**：sync 完成后自动对三仓库做 md5 一致性校验——取 `git ls-files` 文件清单（天然排除 `.so`/`__pycache__`/`_cann_ops_custom` 等构建产物），对同一组源文件在本地与远端各跑一遍 `md5sum`，sorted diff 为空即一致；不一致则打印 diff 并 `exit 1`（防止并行会话覆盖远端代码、rsync 传输不完整等隐蔽问题）。`--no-verify` 可跳过
- **注意**：`task/<task>/code/` 是 codebases 三仓库的 git worktree（已 git 忽略），不要手动改其 `.git` 文件；`task/<task>/` 下的文档（progress.md 等）正常 git 跟踪

管理命令（`scripts/task_worktree.sh`，详见 `/task-worktree` skill）：`new <task> [--base <branch>]` / `list [task]` / `info <task>` / `remove <task>`

### 编译安装要点（/build-vllm skill 摘要）

- 只用 **editable 安装**（`pip install -e .`）：Python 代码即时生效，仅 C++ 扩展（.so）变更需重编 + 重启；安装前 `pip uninstall -y vllm vllm-ascend` 防镜像预装干扰
- 顺序：vllm（`VLLM_TARGET_DEVICE=empty`，~1min）→ vllm-ascend（`COMPILE_CUSTOM_KERNELS=1` 全量编译 5-25min）；vllm-ascend 产物为 `vllm_ascend/{libvllm_ascend_kernels.so, vllm_ascend_C.*.so, _cann_ops_custom/, _build_info.py}`
- **多机/产物复用**：A3 的 `$WS` 是跨 pod 共享盘——代码同步一次、编译产物拷贝一次即全机可见；其余机器 `SETUPTOOLS_SCM_PRETEND_VERSION=<ver> COMPILE_CUSTOM_KERNELS=0 pip install -e .` 秒级安装（build_aclnn 每次 install 都全量重建，勿裸装）
- 构建成功三重 check：`pip show` Editable 指向 worktree + `.so` 存在 + `import` ok；基线 marker 写入本地 `scratch/.build-baseline/<task>.{vllm,vllm-ascend}` 供 /start-service 判定重编

### 日志与 profiling 规范（硬性要求）

- **远端落点**（部署脚本 `TASK=<task>` 参数化，自动 mkdir）：日志 `$WS/task/<task>/logs/vllm_node<N>.log`，profiling `$WS/task/<task>/profiling/`
- **profiling 采集/解析一律走 skills，不得凭记忆手写流程**：采集用 `/collect-profile`——**PD 分离必须 P/D 真并行采集（严禁任何串行，包括 P→D 顺序）**：两条序列（各自 vllm 启动端口 start→sleep→stop，**P 默认 10s、D 默认 3s**；单机/混部默认 2s；窗口变更须用户明确指定）必须同一时刻起跑，**一律用固化脚本 `scripts/pd_profile_capture.sh`**——agent 工具级"并行调用"实测可能被顺序执行（P 先跑完 ~90s stop dump、D 才 start，D 侧已被排干：3.26s 真窗口仅 33 个 lifecycle 事件零计算事件）；stop 串在同一命令链会让后一侧多采整个 dump 时长（名义 2s 实采 ~3min、单卡 2.5GB）。**采后验收门槛**：解析产物必须含 kernel_details/op_statistic/operator_details/step_trace_time CSV 且非空，缺任一即删数据重采，不得以"图模式事件少"等理由放过。解析拉取用 `/parse-profile`（每份只取 rank0 单卡，PD 分离 p/d 各一份）。违规自定参数 = 返工重采
- **每轮实验结束必须把日志下载到本地** `task/<task>/logs/<场景>/` 子目录归档（profiling zip 同理到 `profiling/<场景>/`）。远端日志每次重启会被截断覆盖，不按时下载会丢前几轮现场
- **归档必须是完整日志文件**：用 rsync 整文件拉取，**禁止只保存 grep/tail 截取的片段**——截取会丢上下文，用户没法帮你定位问题。远端 grep 看片段做快速排查可以，但本地归档一律全量原文件；多机部署各节点日志都要拉齐，bench 结果文件（如 bench_res.txt）同理
- **progress.md 每轮必须写明本次日志位置**（相对路径，如 `logs/fc1-off/vllm_node0.log`）——原则：用户能直接翻到对应日志帮你定位问题
- 下载命令（经隧道，多机分别拉）：
  ```bash
  rsync -az -e "ssh -p <port>" root@localhost:$WS/task/<task>/logs/ task/<task>/logs/<场景>/
  ```

## 远程机器

通过 `itask` 连接远程 NPU 机器，有两条通道：

**通道 1（操作通道，首选）：`itask exec <机名>` 按任务名路由**——名字由平台保证，无端口无 localhost，不存在误连。启动/kill/状态检查一律走它：

```bash
itask exec <task-name> --tty=false -- bash -c "<命令>"        # 非交互执行
itask exec <task-name> -- tail -f <日志>                       # 交互式看日志
```

> **itask exec 启动后台常驻进程的两坑（2026-09-22 实测）**：① payload 里 `pkill -f "<进程名>"` 会匹配 payload 自身 cmdline 而杀掉自己的 shell（秒退 exit 1 零输出）——按 PID kill 或用 `[k]xxx` 括号转义；② `cd X && setsid nohup ... &` 的后台子壳握着会话管道等永续服务，itask exec **永不返回**——用 `setsid nohup bash <绝对路径> > 日志 2>&1 < /dev/null &`（简单命令+全重定向，无 cd 链）。kimi-k3 proxy 另有模板目录/日志双坑，详见 start-service skill 的 connector.md。

**通道 2（数据通道）：SSH 隧道**，仅用于 rsync 日志下载与交互调试，默认端口 **27890**：

```bash
# 建立隧道
scripts/ssh_tunnel.sh connect <task-name> [port]

# 多机场景：每台机器不同端口
scripts/ssh_tunnel.sh connect qwen1 27890   # P 节点
scripts/ssh_tunnel.sh connect qwen2 27891   # D 节点
```

隧道建立后，使用 `ssh -p <port> root@localhost` 连接远程机器。

- 远程工作空间：`/a3_inference/itask/workdir/yjq02324703/workspace`（A3 机器间为跨 pod 共享盘）
- 机器 IP **以 `itask list` 的 Pod-IP 为准**（权威，与机器一一对应不会冲突），部署脚本里的 NODE IP 直接用它填写；pod 内 `/etc/hosts`（hostname 编码了 IP，如 `...-172016029197` → 172.16.29.197）用作**隧道路由验证**——连上隧道后两者必须一致，不一致说明隧道连错了机器
- **查询多台机器是否在同一超节点**：`npu-smi info -t spod-info -i 0 -c 0`，看输出的 `Super Pod ID`（相同=同超节点）与 `Server Index`（超节点内序号）。可用 `itask exec <task> --tty=false -- bash -c "..."` 直查，不依赖 SSH 隧道；旁证：`itask describe` 的「超节点」UUID 字段（`itask list` 的 ID 列即其前 8 位）。多机部署前建议核实，不在同一超节点时用 `itask create ... --colocate <参考任务名>` 重建

## 推理部署

部署统一使用 `vllm-ascend-tools` 中按模型组织的脚本，不要手动拼启动命令；代码安装由 `/build-vllm` 负责（editable 指向任务目录）。**具体部署形态——各模型脚本目录、PD 分离/混部选型、conductor 配置与启动（P-first/D-first、ulimit、端口冲突处理）等——详见 `/start-service` skill**，本文件不展开。

### PD 分离多机启动要点

- **多机一律 `scripts/cluster_ctl.sh launch --plan task/<task>/cluster_plan.env`**：P/D 各组、组内所有节点**同时启动**（脚本本地并行 fork，串行启动会导致 DP group 等待超时崩溃）；启动前强制 kill 干净 + 复查归零；占用检查发现活跃服务会中止要求确认；IP 校验对比 `itask list` Pod-IP 与部署脚本修改区。**禁止手工逐台 SSH 启动多机部署**
- agent 的职责只是**写 plan 文件**（模型/脚本差异全部体现在 `CLUSTER_GROUPS` 的命令串里，格式见 `task/k3-pd-1m-main/cluster_plan.env`）与先 `sync.sh`；纪律由脚本强制执行
- 部署脚本头部加 **IP 自检**（本机实际 IP ≠ RANK 对应修改区 IP 即退出，pattern 见 v026_1m 脚本）——配合 cluster_ctl 的静态校验双保险；改到哪个模型脚本就按此模式就地补

## 常用命令

```bash
# === itask 管理 ===
itask list                                              # 查看任务列表（Pod-IP 权威，部署脚本 NODE IP 的依据）
itask delete --name <task-name>                         # 删除任务

# === 任务 worktree（改代码任务，三仓库独立分支）===
scripts/task_worktree.sh new <task-name>                # 创建 worktree（基于 codebases 当前 HEAD）
scripts/task_worktree.sh new <task-name> --base <branch># 指定基础分支
scripts/task_worktree.sh list                           # 列出所有任务 worktree
scripts/task_worktree.sh info <task-name>               # 查看任务分支/commit 状态
scripts/task_worktree.sh remove <task-name>             # 清理 worktree + 分支

# === 同步代码（需先建立 SSH 隧道，日志下载同理）===
scripts/sync.sh <itask> --task <task-name> --port <port># 任务模式：同步该任务代码+文档+共享文件（link-dest 复用远端已有内容，仅传差量）
scripts/sync.sh <itask> --port <port>                   # 默认模式：同步 codebases 主分支 + 共享文件
scripts/sync.sh <itask> --task <task-name> --force      # 任务模式 + 强制同步（--delete 仅作用于该任务目录）
scripts/sync.sh <itask> --task <task-name> --link-dest-task <t>  # 任务模式 + 指定远端内容复用的引用任务
scripts/sync.sh <itask> --task <task-name> --no-verify          # 任务模式 + 跳过同步后 md5 校验

# === SSH 隧道 ===
scripts/ssh_tunnel.sh connect <task-name> [port]        # 建立隧道（自动避让被占端口+清理远端残留会话），用脚本报告的端口 ssh -p <port> root@localhost

# === 多机部署编排（PD 分离/多节点混部一律用它，禁止手工逐台 SSH）===
scripts/cluster_ctl.sh launch --plan task/<task>/cluster_plan.env            # IP校验+占用检查+kill+全部节点同时启动
scripts/cluster_ctl.sh launch --plan task/<task>/cluster_plan.env --only P   # 只起某个组（其余选项: --force-kill/--skip-ipcheck/--dry-run）
scripts/cluster_ctl.sh wait   --plan task/<task>/cluster_plan.env            # 轮询就绪（节点死亡立即报错）
scripts/cluster_ctl.sh check  --plan task/<task>/cluster_plan.env            # 一次性各节点状态
scripts/cluster_ctl.sh kill   --plan task/<task>/cluster_plan.env            # 全部清理
scripts/cluster_ctl.sh ips    --plan task/<task>/cluster_plan.env            # RANK 表（填 conductor config.yaml）

# === 检测与测试 ===
scripts/check_ready.sh <task> <log_file> [port] [max_wait] [ssh_port]        # 单机就绪轮询（多机用 cluster_ctl wait）
scripts/send_request.sh [port] [host] [model] [--stream]

# === 日志下载（每轮实验后硬性要求，按场景归档）===
rsync -az -e "ssh -p <port>" root@localhost:$WS/task/<task>/logs/ task/<task>/logs/<场景>/
```

## 注意事项

- **禁止执行破坏性命令**：不要执行 `rm -rf`、`git push --force`、`git reset --hard`（除非用户明确要求并确认）
- **禁止直接修改远端文件**：代码/脚本改动一律本地优先——任务级在 `task/<task>/code/` worktree 改、非任务级在 `codebases/` 改，经 `sync.sh` 同步到远端；远端直接改的文件会被下次 sync 覆盖且 git 无法追溯。远端只做编译安装/运行/下载产物
- **sync 后自动校验一致性**：`sync.sh` 同步完成后自动对三仓库源码做 md5 比对（取 `git ls-files` 清单，天然排除构建产物），确认本地与远端一致——防止并行会话覆盖远端代码、rsync 传输不完整等隐蔽问题（曾发生并行会话全量 sync 把远端 `kimi_kda.py` 换成旧版导致 Dynamo 编译冲突）。校验失败会 `exit 1` 中止流程；`--no-verify` 可跳过（确信无并发时用）
- **用机器前必须查占用**：机器可能跑着其他任务的服务（用户其他会话的部署），kill 前先看 `ps -ef | grep "vllm serve"` 的启动时间——活跃服务先问用户；误杀其他任务的 DP 集群 node 会级联搞挂整组部署。多机走 `cluster_ctl.sh launch` 时此检查内置：发现占用即中止，确认可清理后加 `--force-kill`
- **SSH 隧道仅用于日志下载与交互调试**：启动/kill/状态检查的操作通道一律走 `itask exec <机名>`（按名路由，无误连风险）。隧道端口被占时会静默误路由到别的机器（曾因此部署错机器），确需隧道时核对 `ssh_tunnel.sh` 打印的远端 IP 与 `itask list` Pod-IP 一致
- **SSH 隧道建立失败时**（如 `libwrap.so.0 not found`、`sshd 启动失败`、rsync 缺库等），先运行 `scripts/fix_container_env.sh` 修复容器环境：`itask exec <task> --tty=false -- bash -c "$(cat scripts/fix_container_env.sh)"`
- `vllm-ascend` 有 git submodule `csrc/third_party/catlass`，`sync.sh` 会自动初始化
- 模型路径在远程机器上，不要直接修改模型目录，如需调试可建新目录用软链接覆盖
- `scripts/kill_all.sh` 可以确保 python/vllm/conductor 进程全部退出，避免残留进程占用 NPU 显存。**清理进程时必须使用 `scripts/kill_all.sh`，不要手动 `kill -9`**
- 不要用 `sleep` 等待服务启动，使用 `scripts/check_ready.sh`（单机）或 `scripts/cluster_ctl.sh wait`（多机）轮询检测，能及时发现进程退出或致命错误
- **每个步骤都必须持续监测，禁止后台轮询挂起等待**：编译/启动 wait/压测/评测等耗时步骤一律前台阻塞轮询监测（日志尾部、进度计数、完成标记），单次轮询超时后再发起下一轮继续；不要把监测丢给后台轮询任务——后台轮询有可能收不到回复，一旦通知丢失流程即卡死无人察觉
- 日志出现 `Application startup complete` 即表示启动成功
- 推理部署统一使用 `vllm-ascend-tools` 中的脚本，不要手动拼启动命令
- 多机部署的操作通道（启动/kill/检查）一律 `itask exec <机名>` 按名路由，不依赖 SSH 隧道；日志下载等确需隧道时每台机器单独建（端口默认 27890 起，被占自动避让，**用脚本实际报告的端口**，并核对打印的远端 IP 与 `itask list` Pod-IP 一致）
- `data/` 和 `scratch/` 不纳入 git
- 启动 vllm 服务前，**工作目录必须是 vllm 代码根目录（VLLM_DIR）**：任务模式 `cd $WS/task/<task>/code/vllm`，默认模式 `cd $WS/codebases/vllm`
