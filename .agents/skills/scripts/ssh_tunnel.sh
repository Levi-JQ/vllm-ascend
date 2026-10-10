#!/bin/bash
# ssh_tunnel.sh - 通过 itask ssh-tunnel 建立 SSH 隧道
#
# 用法:
#   scripts/ssh_tunnel.sh connect <task-name> [port]
#
# 隧道建立后，使用 ssh -p <port> root@localhost 连接远程机器
#
# 内置防护（防止隧道误用导致操作落到别的机器，见 AGENTS.md 注意事项）:
#   1. 端口冲突自动避让: 请求端口被占（旧会话隧道残留、其他本地程序占用等常见）时
#      自动换用下一个空闲端口，并明确打印实际端口——后续所有 ssh/rsync 必须用脚本报告的端口
#   2. 远端身份核验: 连接后打印远端 /etc/hosts 实际 IP，须与 itask list 该机器的 Pod-IP
#      核对一致；不一致说明隧道连错了机器，立即停手
#   3. 远端 ssh 会话清理: 杀掉其他客户端留在远端的 sshd 会话（保留 sshd 监听器与本会话），
#      防止旧隧道/其他工具继续往这台机器路由流量

set -euo pipefail

DEFAULT_PORT=27890

# 本地端口是否被占用（bash /dev/tcp 探测，无外部依赖）
port_occupied() {
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

case "${1:?Usage: ssh_tunnel.sh connect <task-name> [port]}" in
    connect)
        task_name="${2:?Task name required}"
        port="${3:-$DEFAULT_PORT}"
        requested_port="$port"

        # --- 防护 1: 端口冲突自动避让 ---
        for _ in $(seq 1 50); do
            if port_occupied "$port"; then
                echo ">>> 本地端口 ${port} 已被占用（旧会话隧道或其他本地进程常见），尝试下一端口..."
                port=$((port + 1))
            else
                break
            fi
        done
        if port_occupied "$port"; then
            echo "ERROR: 从 ${requested_port} 起连续 50 个端口均被占用，无法建立隧道"
            exit 1
        fi
        if [ "$port" != "$requested_port" ]; then
            echo "=== ⚠️ 端口 ${requested_port} 被占用，已自动改用 ${port} ==="
            echo "=== ⚠️ 后续所有操作请使用: ssh -p ${port} root@localhost ==="
        fi

        echo "=== Establishing SSH tunnel to itask: ${task_name} (port: ${port}) ==="
        nohup itask ssh-tunnel "$task_name" --port "$port" > /tmp/itask-tunnel-${task_name}.log 2>&1 &
        for i in $(seq 1 15); do
            if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 -p "$port" root@localhost "echo ok" >/dev/null 2>&1; then

                # --- 防护 2: 远端身份核验（打印实际 IP 供与 itask list Pod-IP 核对）---
                remote_ip=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p "$port" root@localhost \
                    "grep -E '^172\.' /etc/hosts 2>/dev/null | head -1 | awk '{print \$1}'" 2>/dev/null || true)
                echo "=== 远端机器实际 IP: ${remote_ip:-获取失败}（与 itask list 的 Pod-IP 核对，不一致=隧道连错机器）==="

                # --- 防护 3: 清理远端其他 ssh 会话（保留监听器与本会话祖先链）---
                ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p "$port" root@localhost '
                    anc=""; p=$$
                    while [ -n "$p" ] && [ "$p" != "0" ] && [ "$p" != "1" ]; do
                        anc="$anc $p"
                        p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d " ")
                    done
                    ps -eo pid,args 2>/dev/null | grep -E "sshd(-session)?: root" | grep -v grep | \
                    while read -r pid rest; do
                        case " $anc " in *" $pid "*) continue ;; esac
                        kill "$pid" 2>/dev/null && echo "    cleaned: 杀掉残留 ssh 会话 pid=${pid} (${rest})"
                    done
                    true
                ' 2>/dev/null || true
                echo "=== 远端 ssh 会话清理完成 ==="

                echo "=== SSH tunnel ready! ==="
                echo "=== ssh -p ${port} root@localhost ==="
                exit 0
            fi
            sleep 1
        done
        echo "WARNING: SSH tunnel may not be ready yet, check /tmp/itask-tunnel-${task_name}.log"
        ;;
    *)
        echo "Unknown action: $1"
        echo "Usage: ssh_tunnel.sh connect <task-name> [port]"
        exit 1
        ;;
esac
