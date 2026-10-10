# 服务就绪检测与推理验证

## 就绪检测

**不要用 `sleep` 等待**——轮询能及时发现进程退出或致命错误。日志出现 `Application startup complete` 即表示启动成功。

**多机（cluster_ctl 部署）**：用 cluster_ctl 轮询/查状态，不依赖 SSH 隧道：

```bash
# 轮询等待全部节点就绪（任一节点 vllm serve 进程死亡立即报错并给出日志排查命令）
scripts/cluster_ctl.sh wait --plan task/<task>/cluster_plan.env --max-wait 1800

# 一次性打印各节点状态（进程数 / 就绪 / 启动中 / 日志报错 / 不可达）
scripts/cluster_ctl.sh check --plan task/<task>/cluster_plan.env
```

wait 检测逻辑：节点进程数=0 → 立即报错（附 `itask exec <机名> -- tail -50 <日志>` 排查命令）；日志含 `Application startup complete` → 就绪；否则继续等待。超时未就绪用 `check` 看全貌。

**单机混部**：使用 `scripts/check_ready.sh` 轮询检测：

```bash
# P 节点检测（未参数化脚本示例）
scripts/check_ready.sh <prefill-task> $VLLM_DIR/log_prefill_0.log 8100 600 <prefill-ssh-port>

# D 节点检测（未参数化脚本示例）
scripts/check_ready.sh <decode-task> $VLLM_DIR/log_decode_0.log 8200 600 <decode-ssh-port>

# 已参数化脚本（TASK=<task> 启动，日志挂任务目录）
scripts/check_ready.sh <task> $WS/task/<task>/logs/vllm_node0.log <api-port> 600 <ssh-port>

# 混部检测
scripts/check_ready.sh <task> $VLLM_DIR/<log-file> <port> 600 <ssh-port>
```

参数顺序：`<task> <log_file> <api_port> <max_wait=600> <ssh_port>`

> `check_ready.sh` 最后一个参数是 SSH 隧道端口，走隧道轮询远程日志（单机场景必须指定）。

各模型的日志文件名见 [models.md](../models.md)。

**服务就绪并验证通过后，必须向用户汇报每台机器的启动日志在容器中的绝对路径**（多机逐台列出：机器名 + 绝对路径），并附实时查看命令，例如：

```
k3-p0（P 组 RANK=0）:
  /a3_inference/itask/workdir/yjq02324703/workspace/task/<task>/logs/vllm_p_node0.log
  查看: itask exec k3-p0 -- tail -f /a3_inference/itask/workdir/yjq02324703/workspace/task/<task>/logs/vllm_p_node0.log
k3-d0（D 组 RANK=0）:
  ...
```

日志路径取决于部署脚本是否已 TASK 参数化（规范见 SKILL.md）：
- **已参数化**（`TASK=<task>` 启动）：`$WS/task/<TASK>/logs/<log-file>`（如 `vllm_node0.log`）
- **未参数化**（存量脚本）：日志在脚本执行目录 `$VLLM_DIR/` 下（如 `log_prefill_0.log`）

## 推理验证

服务就绪后，用 `scripts/send_request.sh` 做推理验证：

```bash
# PD 分离模式通过 connector 端口（8006）测试
scripts/send_request.sh 8006 localhost auto

# 混部模式直接测试（端口即服务端口）
scripts/send_request.sh <port> localhost auto

# 流式测试
scripts/send_request.sh 8006 localhost auto --stream
```

参数：`[port] [host] [model] [--stream]`。`auto` 表示自动选择模型。

## 故障排查

如果服务未就绪或推理失败：

1. 查看服务日志定位错误（多机优先 `itask exec` 按名路由，单机走隧道）：
```bash
itask exec <机名> --tty=false -- bash -c "tail -50 $WS/task/<task>/logs/<log-file>"   # 多机
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "tail -50 $VLLM_DIR/<log-file>"  # 单机
```

2. 确认 NPU 显存是否被残留进程占用：
```bash
itask exec <机名> --tty=false -- npu-smi info                                          # 多机
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "npu-smi info"                # 单机
```

3. 常见问题：
   - `ascend_transport.so` 找不到 → 检查 `LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH` 是否设置
   - 端口被占用 → 用 `cluster_ctl.sh kill`（多机）/ `scripts/kill_all.sh`（单机）清理后重启
   - 节点启动即退出（IP 自检失败）→ 部署脚本修改区 NODE IP 与本机 Pod-IP 不符，本地改修改区 → sync → 重新 launch
   - PD 分离握手失败 → 确认 P/D 节点 IP 配置正确、connector 已启动
   - editable 开发版模块导入失败 → 检查 `pip show vllm vllm-ascend` 的 Editable 是否指向正确 worktree（见 `/build-vllm`）

4. 重启流程：stop（多机 `cluster_ctl.sh kill --plan ...`，单机 `kill_all.sh`）→ sync（如代码有变更）→ start。