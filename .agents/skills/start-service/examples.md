# 完整启动示例

## 示例 1: kimi-k25 A3 PD 分离（3 机：1P + 2D）——cluster_ctl 流程

```bash
# 前提: 用户已创建 itask: jinqi-dycp-0 (P), jinqi-dycp-1 (D master), jinqi-dycp-2 (D worker)

# 0. 同步代码（走 SSH 隧道，端口以 ssh_tunnel.sh 报告为准；任务模式加 --task <task>）
scripts/sync.sh jinqi-dycp-0 --port 27890

# 1. 写 plan 文件 task/<task>/cluster_plan.env（D 节点需 DATA_PARALLEL_HEAD_ADDRESS + node_rank 参数；
#    NODE IP 由部署脚本修改区管理，launch 会校验与 Pod-IP 一致）
TASK=<task-name>
WS=/a3_inference/itask/workdir/yjq02324703/workspace
CODE_BASE=$WS/task/$TASK/code        # 默认模式: $WS/codebases
CLUSTER_GROUPS=(
  "P|jinqi-dycp-0|cd $CODE_BASE/vllm && TASK=$TASK nohup bash $CODE_BASE/vllm-ascend-tools/kimi-k25/A3-full-pd/start_prefill.sh 0|$WS/task/$TASK/logs/log_prefill_0.log"
  "D|jinqi-dycp-1,jinqi-dycp-2|cd $CODE_BASE/vllm && TASK=$TASK export DATA_PARALLEL_HEAD_ADDRESS=<decode-master-pod-ip> && nohup bash $CODE_BASE/vllm-ascend-tools/kimi-k25/A3-full-pd/start_decode.sh {RANK}|$WS/task/$TASK/logs/log_decode_{RANK}.log"
)

# 2. 审阅 + 启动（占用检查/kill/IP 校验/并行启动全内置；P/D 两组同时拉起）
scripts/cluster_ctl.sh launch --plan task/<task>/cluster_plan.env --dry-run
scripts/cluster_ctl.sh launch --plan task/<task>/cluster_plan.env
#   → 占用被拦下时报用户确认，确认可清理后加 --force-kill

# 3. 等待就绪（任一节点死亡立即报错）
scripts/cluster_ctl.sh wait --plan task/<task>/cluster_plan.env --max-wait 1800

# 4. 启动 connector（统一 conductor；config.yaml 的 IP 用 cluster_ctl ips 辅助填写）
#    先更新 connector/config.yaml 中的 IP，再同步
scripts/sync.sh jinqi-dycp-0 --port 27890
ssh -o StrictHostKeyChecking=no -p 27890 root@localhost "
bash -l -c '
  cd $VLLM_DIR
  nohup bash $WS/connector/conductor.sh &
'
"

# 5. 验证
scripts/send_request.sh 8006 localhost auto

# 6. 汇报日志绝对路径（逐台列出 + 查看命令）
# jinqi-dycp-0（P）: $WS/task/<task>/logs/log_prefill_0.log 或 $VLLM_DIR/log_prefill_0.log（未参数化脚本）
#   查看: itask exec jinqi-dycp-0 -- tail -f <日志绝对路径>
# jinqi-dycp-1/2（D）: 同上对应 decode 日志
```

> 旧式"逐台 SSH + 多个 tool call 并行发出"的做法已废弃：plan 命令串里的 `nohup ... &` 由 cluster_ctl 的 `setsid nohup bash -l -c` 统一兜底，无需再手工拼。kimi-k25 传 node_rank 参数、DATA_PARALLEL_HEAD_ADDRESS 等模型差异全部体现在 plan 的命令串里。

## 示例 2: qwen3.5 PD 分离（2 机，cluster_ctl）

```bash
# task/<task>/cluster_plan.env
TASK=<task-name>
WS=/a3_inference/itask/workdir/yjq02324703/workspace
CODE_BASE=$WS/task/$TASK/code        # 默认模式: $WS/codebases
CLUSTER_GROUPS=(
  "P|<prefill-机名>|cd $CODE_BASE/vllm && TASK=$TASK bash $CODE_BASE/vllm-ascend-tools/qwen3.5/full-pd/start_prefill.sh 397b-w4a8|$WS/task/$TASK/logs/log_prefill_0.log"
  "D|<decode-机名>|cd $CODE_BASE/vllm && TASK=$TASK bash $CODE_BASE/vllm-ascend-tools/qwen3.5/full-pd/start_decode.sh 397b-w4a8|$WS/task/$TASK/logs/log_decode_0.log"
)

scripts/cluster_ctl.sh launch --plan task/<task>/cluster_plan.env
scripts/cluster_ctl.sh wait   --plan task/<task>/cluster_plan.env
```

> 模型名参数（`397b-w4a8`）等模型差异直接写进命令串。

## 示例 3: qwen3.5 混部（1 机，SSH 模板）

```bash
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "
bash -l -c '
  export PYTHONPATH=$VLLM_DIR:$ASCEND_DIR:\$PYTHONPATH
  export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH
  cd $VLLM_DIR
  bash ../vllm-ascend-tools/qwen3.5/start_qwen35_397b_A3.sh 397b-w4a8
'
"

scripts/check_ready.sh <task> $VLLM_DIR/qwen35-server.log 8006 600 <port>
```
