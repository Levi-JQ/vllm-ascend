# 启动流程

本文件描述"在远程 itask 机器上启动 vLLM 推理服务"的完整流程。模型/脚本路径参数见 [models.md](../models.md)，connector 细节见 [connector.md](connector.md)，就绪检测与验证见 [monitoring.md](monitoring.md)。

**通道分工**：多机（PD 分离/多节点混部）启动一律 `scripts/cluster_ctl.sh`（内部 `itask exec <机名>` 按名路由，不依赖 SSH 隧道）；单机混部与 conductor 启动用 SSH 隧道模板。SSH 隧道另用于 sync 与日志下载（rsync）。

## SSH 命令模板（单机混部 / conductor 用）

所有远程启动命令遵循统一模板。模板中的 `$VLLM_DIR`/`$ASCEND_DIR`/`$TOOLS_DIR` 由 `CODE_BASE` 派生（见 SKILL.md "代码来源"），执行前先在本地 shell 确定这些变量：

```bash
WS=/a3_inference/itask/workdir/yjq02324703/workspace
# 任务模式（改代码任务）:                 # 默认模式（纯部署）:
TASK=<task-name>                          #   （不需要）
CODE_BASE=$WS/task/$TASK/code             #   CODE_BASE=$WS/codebases
VLLM_DIR=$CODE_BASE/vllm
ASCEND_DIR=$CODE_BASE/vllm-ascend
TOOLS_DIR=$CODE_BASE/vllm-ascend-tools
```

```bash
ssh -o StrictHostKeyChecking=no -p <ssh-port> root@localhost "
bash -l -c '
  export PYTHONPATH=$VLLM_DIR:$ASCEND_DIR:\$PYTHONPATH
  export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH
  cd $VLLM_DIR
  ulimit -n 65536
  <启动命令>
'
"
```

要点：
- `bash -l` 加载完整 PATH（vllm 在 `/usr/local/python3.11.14/bin/` 下）
- `LD_LIBRARY_PATH` 必须显式设置，否则 nohup 子进程找不到 `ascend_transport.so`
- 开发版唯一模式是 `/build-vllm` 的 **editable 安装**（已注册 sys.path）；模板的 `PYTHONPATH` 行保留亦等效（指向同一目录）
- kimi-k25 的启动命令需额外 `nohup ... &` 后台运行
- kimi-k25 D 节点需在 export 行前加 `export DATA_PARALLEL_HEAD_ADDRESS=<decode-master-ip>`
- 部署脚本统一带 `TASK=<task>` 环境变量启动（日志/profiling 挂 `$WS/task/<TASK>/`，规范见 SKILL.md"部署脚本 TASK 参数化规范"）

### editable 开发版的 PYTHONPATH

`/build-vllm` 的 editable 安装（`pip install -e .`）已把 vllm/vllm-ascend 注册到 sys.path，**无需修改模板的 PYTHONPATH 行**——保留默认 `export PYTHONPATH=$VLLM_DIR:$ASCEND_DIR:$PYTHONPATH` 即可（editable 与 PYTHONPATH 指向同一目录，等效）。

> 前提：`/build-vllm` 已执行 `pip uninstall -y vllm vllm-ascend` 卸载镜像预装版，editable 安装的 worktree 版本不会被 site-packages 旧版覆盖。若 `import` 到错误版本，检查 `pip show vllm vllm-ascend` 的 Editable 是否指向 worktree（见 `/build-vllm` 的安装成功 check）。

## Step 1: 确认前置条件

**多机（PD 分离/多节点混部）**：agent 只需做两件事，占用检查/kill 由 launch 内置——

```bash
# 1a. 同步代码（走 SSH 隧道；A3 共享盘实际只需对任意一台 sync，脚本会同步到共享路径）
# 任务模式（改代码任务，需先 scripts/task_worktree.sh new <task>）
scripts/sync.sh <itask-name> --task <task-name> --port <port>
# 默认模式（纯部署）
scripts/sync.sh <itask-name> --port <port>

# 1b. 写/更新 plan 文件 task/<task>/cluster_plan.env（格式见 SKILL.md"多机部署"节）
```

**单机混部**：对机器按序执行 **占用检查 → 同步 → 杀残留进程**：

```bash
# 1a. 占用检查（kill/sync 前必须做）：
#     ps 看启动时间与命令行 + npu-smi 看显存占用
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "ps -ef | grep 'vllm serve' | grep -v grep"
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "npu-smi info"
#     → 发现近期启动的活跃服务（可能是其他任务的部署）：先问用户确认，绝不能直接 kill_all
#     → 仅本任务的陈旧残留（如 2 天前的实验遗留）：可清理

# 1b. 同步代码（同上）

# 1c. 杀残留进程（占用检查通过后）
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "bash $WS/scripts/kill_all.sh"
```

注意事项：
- sync 走 SSH 隧道（每台/任一台共享盘机器），**端口以 ssh_tunnel.sh 实际报告的为准**，并核对打印的远端 IP 与 `itask list` 的 Pod-IP 一致
- **清理进程必须使用 `scripts/kill_all.sh`**，不要手动 `kill -9`，避免残留进程占用 NPU 显存
- 多机启动/kill/检查**不需要隧道**（cluster_ctl 走 `itask exec` 按名路由）

## Step 2: 启动推理服务

**多机（PD 分离/多节点混部）——唯一入口 `cluster_ctl.sh launch`**，禁止手工逐台 SSH：

```bash
# 先 dry-run 审阅将执行的命令（可选但推荐）
scripts/cluster_ctl.sh launch --plan task/<task>/cluster_plan.env --dry-run
# 正式启动：Pod-IP 校验 → 占用检查 → kill+复查归零 → 全部组全部节点同时启动
scripts/cluster_ctl.sh launch --plan task/<task>/cluster_plan.env
# 占用被拦下时报给用户确认，确认可清理后：
scripts/cluster_ctl.sh launch --plan task/<task>/cluster_plan.env --force-kill
```

launch 内置纪律（agent 无需也无法绕过）：Pod-IP 与部署脚本修改区一致性校验（不一致报错列出正确 IP，须本地改→sync→再来）、占用检查（活跃服务中止）、逐机 kill_all + 复查归零、所有组所有节点**本地并行 fork 同时启动**（P/D 两组默认同时拉起；串行启动会让 DP group 等待超时崩溃）。需要分组先后（如旧脚本 rank0 须先起）用 `--only P` / `--only D` 分两次。

**单机混部**：使用 SSH 命令模板在远程机器上执行启动命令，具体脚本路径和参数见 [models.md](../models.md)。

## Step 3: 启动 Connector（仅 PD 分离模式）

详见 [connector.md](connector.md)。仅需在 PD 分离模式下执行，混部跳过此步。配置 config.yaml 的节点 IP 可先用 `scripts/cluster_ctl.sh ips --plan ...` 打印 RANK 表辅助填写。

## Step 4: 检测服务就绪

详见 [monitoring.md](monitoring.md) 的就绪检测部分。**不要用 `sleep` 等待服务启动**——多机用 `scripts/cluster_ctl.sh wait --plan ...` 轮询（任一节点进程死亡立即报错），单机用 `scripts/check_ready.sh`。

## Step 5: 验证推理

详见 [monitoring.md](monitoring.md) 的推理验证部分。