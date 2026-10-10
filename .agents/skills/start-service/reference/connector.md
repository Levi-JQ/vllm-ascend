# Connector 启动（PD 分离模式）

PD 分离模式需要 conductor 连接 P 节点和 D 节点。混部模式不需要 connector。默认使用统一 conductor `conductor_vllm_ascend_decode_1st_token_linux_arm64`（支持 P-first/D-first）；旧 `coord_arm` 仅支持 P-first，已被取代；用户明确指定 `proxy` 时才使用 proxy。

启动前需通过 `itask list` 获取各节点 Pod-IP，并确认 P/D 节点服务已就绪（见 [monitoring.md](monitoring.md)）。

## conductor（默认）

1. 修改 `connector/config.yaml` 中 prefill/decode 节点 IP：

```yaml
test_node_list:
  - host: <prefill-ip>
    port: 8100
    role: "prefill"
  - host: <decode-ip>
    port: 8200
    role: "decode"
```

> kimi-k25 有两个 D 节点（master + worker），config.yaml 需要列出所有 D 节点。

2. **P-first / D-first** 由 `config.yaml` 中 `first_token_from_prefill` 控制：
   - **`first_token_from_prefill: false`**（默认）：D-first，首 token 由 Decode 节点输出
   - **`first_token_from_prefill: true`**：P-first，首 token 由 Prefill 节点输出
   - P/D 节点 vllm 启动命令完全相同，区别仅在 conductor 配置

3. 同步配置到执行 conductor 的机器：
```bash
scripts/sync.sh <task> --port <port>
```

4. 启动（通常选 P 节点机器执行）——优先用现成脚本 `connector/conductor.sh`（已含 ulimit、端口清理、日志重定向），或等价的直接命令：

```bash
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "
bash -l -c '
  cd $VLLM_DIR
  nohup bash $WS/connector/conductor.sh &
'
"

# 直接启动（conductor.sh 的等价展开）：
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "
bash -l -c '
  cd $VLLM_DIR
  ulimit -n 1048576
  fuser -k 8006/tcp 2>/dev/null || true; sleep 1
  nohup $WS/connector/conductor_vllm_ascend_decode_1st_token_linux_arm64 -conf-file $WS/connector/config.yaml &> coord.log &
'
"
```

要点：
- 启动 conductor 前必须 `ulimit -n 1048576`（否则连接数超限）；vllm 进程启动前 `ulimit -n 65536`
- 日志重定向到 `coord.log`（相对 `VLLM_DIR`）

### 端口冲突处理

重启 conductor 时，旧进程可能未完全退出，导致 `listen tcp 0.0.0.0:8006: bind: address already in use`。

```bash
# 杀掉占用 8006 端口的旧进程后重启
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "
fuser -k 8006/tcp 2>/dev/null; sleep 2
bash -l -c '
  cd $VLLM_DIR
  ulimit -n 1048576
  nohup $WS/connector/conductor_vllm_ascend_decode_1st_token_linux_arm64 -conf-file $WS/connector/config.yaml &> coord.log &
  sleep 3
  tail -5 coord.log
'
"
```

- `fuser -k <port>/tcp`：杀掉占用指定端口的进程（比 `lsof` 更可能在容器中可用）
- 如果 `fuser` 也不可用，用 `kill_all.sh` 清理所有进程更可靠

## coord_arm（旧，已废弃）

仅支持 P-first，已被统一 conductor 取代，仅作兜底。启动命令同上，把二进制换成 `$WS/connector/coord_arm`，同样 `-conf-file $WS/connector/config.yaml`。

## proxy

```bash
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "
bash -l -c '
  cd $VLLM_DIR
  nohup python3 $ASCEND_DIR/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py \
    --prefiller-hosts <prefill-ip> --prefiller-ports 8100 \
    --decoder-hosts <decode-ip> --decoder-ports 8200 \
    --host <proxy-ip> --port 8006 > proxy.log &
'
"
```

> **注意**：proxy 的 `--host` 不能用 `0.0.0.0`，必须用实际 Pod-IP，否则报 "Wildcard Address 0.0.0.0 is not allowed"。

> **端口冲突**：proxy 也可能遇到端口占用，用同样的 `fuser -k 8006/tcp` 处理。

### kimi-k3 任务级 proxy（kimi-k3-pd-proxy.sh）启动纪律

kimi-k3 PD 分离的任务级 proxy 用 `vllm-ascend-tools/kimi-k3/pd/<版本目录>/proxy/kimi-k3-pd-proxy.sh`（对外 1999，P/D 后端 8001）。2026-09-22 实测 4 坑：

1. **目录坑——用错模板脚本**：`kimi-k3/pd/proxy/` 下是 8+8 节点**占位符模板**（`P_NODE0_IP` 字面量、端口 8000）；httpx 惰性建连使启动"正常"，真请求必 500 Internal Server Error。必须用**任务版本目录**的脚本（如 `pd/v026_1m/proxy/`，4+4 真实 IP + 端口 8001）；启动前 `grep -E "P_HOSTS|D_HOSTS"` 确认是真实 IP。
2. **自杀坑——payload 内 `pkill -f` 自匹配**：`itask exec ... bash -c 'pkill -f "kimi-k3-pd-proxy"; ...'` 的模式会匹配 payload 自身 cmdline（其中含同名脚本路径）→ 杀掉自己的 shell，秒退 exit 1 零输出。清理旧进程**按 PID kill**，或模式首字母括号转义 `pkill -f "[k]imi-k3-pd-proxy"`。
3. **挂会话坑——后台子壳握住会话管道**：`cd X && setsid nohup bash 脚本 > log &` 的重定向只作用于内层命令；后台 AND-list 子壳自己的 stdout 仍是 itask 会话管道，且要等永续服务退出 → itask exec **永不返回**（"卡住"假象，残留子壳挂到 PID 1）。正确写法是**简单命令 + 全重定向**（无 `cd &&` 链，脚本用绝对路径）：
   ```bash
   itask exec <p0机名> --tty=false -- bash -lc 'setsid nohup bash <绝对路径>/kimi-k3-pd-proxy.sh > /tmp/proxy.log 2>&1 < /dev/null & sleep 3; <就绪检查>'
   ```
4. **日志坑——就绪判定等错文件**：脚本 banner 走外层重定向的 `/tmp/proxy.log`，但脚本内 `exec python ... > /tmp/pd_proxy.log` 把 uvicorn 就绪行写进脚本自己的日志。就绪判定 grep `/tmp/pd_proxy.log` 的 `Application startup complete`；**最终以 curl /v1/chat/completions 真冒烟为准**（proxy 不实现 /v1/models，探活会 404；复用旧日志时陈旧就绪行会提前误判 ready）。

## Connector 就绪确认

通过查看日志确认 connector 启动成功：

```bash
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "tail -20 $VLLM_DIR/coord.log"
# 或
ssh -o StrictHostKeyChecking=no -p <port> root@localhost "tail -20 $VLLM_DIR/proxy.log"
```

connector 就绪后即可通过其端口（默认 8006）测试推理。
