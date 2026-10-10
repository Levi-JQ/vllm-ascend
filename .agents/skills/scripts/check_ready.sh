#!/bin/bash
# check_ready.sh - 轮询检测推理服务是否启动完成
#
# 用法:
#   scripts/check_ready.sh <task-name> <log_file> [port] [max_wait] [ssh_port]
#   scripts/check_ready.sh jinqi-dycp-0 /tmp/vibe/logs/cmd_prefill_0.log 8100 600
#   scripts/check_ready.sh jinqi-dycp-1 /tmp/vibe/logs/cmd_decode_0.log 8200 600
#
# 通过 itask ssh-tunnel 连接远程机器，轮询日志检测服务启动状态。
# 检测逻辑：
#   1. 检查 vllm 进程是否存活（进程退出则判定失败）
#   2. 检查日志中是否出现 "Application startup complete"（启动成功）
#   3. 检查日志中是否有致命错误（RuntimeError、OOM、killed 等）
#
# 不要用 sleep 等待服务启动，使用本脚本轮询检测，能及时发现进程退出或致命错误。

set -euo pipefail

TASK_NAME="${1:?Usage: check_ready.sh <task-name> <log_file> [port] [max_wait] [ssh_port]}"
LOG_FILE="${2:?Log file required}"
PORT="${3:-8000}"
MAX_WAIT="${4:-600}"
SSH_PORT="${5:-7890}"

INTERVAL=10
elapsed=0

echo "=== Checking service readiness ==="
echo "    task: ${TASK_NAME}, log: ${LOG_FILE}, port: ${PORT}, max_wait: ${MAX_WAIT}s"

while [[ $elapsed -lt $MAX_WAIT ]]; do
    # 通过 SSH 检查进程状态和日志
    result=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p "$SSH_PORT" root@localhost \
        "pgrep -f 'vllm serve' > /dev/null 2>&1 && echo 'PROC_ALIVE' || echo 'PROC_DEAD'; echo '---LOG---'; if [ -f ${LOG_FILE} ]; then tail -50 ${LOG_FILE}; else echo 'LOG_NOT_FOUND'; fi" 2>/dev/null || echo "SSH_FAILED")

    if [[ "$result" == "SSH_FAILED" ]]; then
        echo "[${elapsed}s] SSH connection failed, retrying..."

    elif echo "$result" | grep -q "Application startup complete"; then
        echo "READY: Service is up after ${elapsed}s"
        exit 0

    elif echo "$result" | grep -q "PROC_DEAD"; then
        if echo "$result" | grep -q "LOG_NOT_FOUND"; then
            echo "[${elapsed}s] Process not started yet, waiting..."
        else
            echo "FAILED: vllm process exited after ${elapsed}s"
            echo "=== Last 20 lines of log ==="
            echo "$result" | sed -n '/---LOG---/,$ p' | tail -20
            exit 1
        fi

    elif echo "$result" | grep -qiE "RuntimeError|Traceback.*Error|killed|OOM|SIGKILL"; then
        echo "ERROR detected after ${elapsed}s:"
        echo "$result" | grep -iE "RuntimeError|Traceback.*Error|killed|OOM|SIGKILL" | grep -v "WARNING\|UserWarning\|known problem" | tail -5
        echo "=== Last 20 lines of log ==="
        echo "$result" | sed -n '/---LOG---/,$ p' | tail -20
        exit 1

    else
        # 显示日志最后一行作为进度
        last_line=$(echo "$result" | sed -n '/---LOG---/,$ p' | tail -1 | cut -c1-120)
        echo "[${elapsed}s] waiting... last: ${last_line}"
    fi

    sleep $INTERVAL
    elapsed=$((elapsed + INTERVAL))
done

echo "TIMEOUT: Service did not start within ${MAX_WAIT}s"
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p "$SSH_PORT" root@localhost "tail -20 ${LOG_FILE}" 2>/dev/null || true
exit 1