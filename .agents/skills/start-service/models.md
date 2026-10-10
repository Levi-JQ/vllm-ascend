# 模型配置

## kimi-k3

| 项目 | 值 |
|------|-----|
| 模型路径 | `/mnt/sfs_turbo/models/Kimi-K3-w4a8`（NAS 共享；或本地 model-csi 软链） |
| 支持模式 | 混部、PD分离 |
| 混部脚本目录 | `vllm-ascend-tools/kimi-k3/mixed/` |
| PD 脚本目录 | `vllm-ascend-tools/kimi-k3/pd/` |
| 辅助脚本 | `bench/bench_kimi_k3.sh`（压测）、`verify_curl.sh`、`vllm_metrics_monitor.sh` |
| 日志 | `vllm_node${RANK}.log`（混部）/ `vllm_p_node${RANK}.log`、`vllm_d_node${RANK}.log`（PD） |

混部（`mixed/`，按机器数选脚本；必须先起 RANK=0 主进程，再起 worker）：

| 脚本 | 拓扑 | 机器数 | RANK | MAX_MODEL_LEN | 端口 |
|------|------|--------|------|---------------|------|
| `kimi-k3-deploy-2node.sh` | DP2/TP16/EP32 | 2 | 0-1 | 48000 | 8006 |
| `kimi-k3-deploy-4node.sh` | DP4/TP16/EP64 | 4 | 0-3 | 131027 | 8000 |
| `kimi-k3-deploy-4node-256k.sh` | DP4/TP16/EP64 | 4 | 0-3 | 262144 | 8000 |
| `kimi-k3-deploy-8node.sh` | DP8/TP16/EP64 | 8 | 0-7 | 1048576 (1M) | 8000 |

PD 分离（`pd/`，P/D 各节点同时启动，用 cluster_ctl.sh launch，见 SKILL.md"多机部署"；DP8/TP16，P/D 各 8 节点）：

| 脚本 | 角色 | RANK | MAX_MODEL_LEN | 端口 | KV_PORT |
|------|------|------|---------------|------|---------|
| `kimi-k3-deploy-prefill.sh` | kv_producer（enforce-eager，Mooncake+AscendStore MultiConnector） | 0-7 | 133120 | 8000 | 17001 |
| `kimi-k3-deploy-decode.sh` | kv_consumer | 0-7 | 133120 | 8000 | 17001 |
| `kimi-k3-pd-mooncake-master.sh` / `kimi-k3-pd-metaservice.sh` | Mooncake master / metaservice 辅助服务 | - | - | - | - |
| `conn_pd.sh` + `conductor-dfirst-arm64` + `conductor.yaml` | conductor 启动 | - | - | 8006 | - |

特殊说明：换 pod / 改端口 / 改并行策略都在各脚本顶部「修改区」改（NODE IP 用 `itask list` Pod-IP）；启动统一带 `TASK=<task>` 环境变量（规范见 SKILL.md，未参数化的存量脚本改到时就地补）；多机启动一律 `cluster_ctl.sh launch`（plan 文件模板见 SKILL.md），改到哪个脚本就地补 **IP 自检**（pattern 见 SKILL.md）。

## kimi-k25

| 项目 | 值 |
|------|-----|
| 模型路径 | `/sfs_turbo/models/kimi-k2.5-w4a8` |
| 支持模式 | PD分离 |
| A3 脚本目录 | `vllm-ascend-tools/kimi-k25/A3-full-pd/` |
| A2 脚本目录 | `vllm-ascend-tools/kimi-k25/A2-full-pd/` |
| P 节点脚本 | `start_prefill.sh <node_rank>` |
| D 节点脚本 | `start_decode.sh <node_rank>` |
| P 节点端口 | 8100 |
| D 节点端口 | 8200 |
| P 节点日志 | `log_prefill_<node_rank>.log` |
| D 节点日志 | `log_decode_<node_rank>.log` |
| 机器数量 | 3 台（1P + 2D） |
| 特殊说明 | D 节点需设置 `DATA_PARALLEL_HEAD_ADDRESS` 为 D 节点 master IP；node_rank 从 0 开始；D 节点需 2 台（master node_rank=0，worker node_rank=1） |

## qwen3.5

| 项目 | 值 |
|------|-----|
| 支持模式 | PD分离、混部 |
| PD 分离脚本目录 | `vllm-ascend-tools/qwen3.5/full-pd/` |
| 混部脚本 | `vllm-ascend-tools/qwen3.5/start_qwen35_397b_A3.sh` |
| 模型参数 | 启动脚本需要模型名称参数，如 `397b-w4a8`、`397b-w8a8`、`397b`（传入参数会自动拉取相应模型） |
| PD P 节点脚本 | `start_prefill.sh <model_name>` |
| PD D 节点脚本 | `start_decode.sh <model_name>` |
| PD P 节点端口 | 8100 |
| PD D 节点端口 | 8200 |
| 混部端口 | 8006 |
| PD P 节点日志 | `log_prefill_0.log` |
| PD D 节点日志 | `log_decode_0.log` |
| 混部日志 | `qwen35-server.log` |
| 特殊说明 | 启动脚本内含 model-cli pull 逻辑，首次启动会自动拉取模型 |
