#!/bin/bash
# kill_all.sh - 清理远端机器上的 python/vllm 进程
#
# 用法（通过 SSH 隧道执行）:
#   ssh -p <port> root@localhost "bash /a3_inference/.../workspace/scripts/kill_all.sh"
#
# 问题：旧版直接 grep python | kill -9 会匹配到 SSH 会话自身（命令行含 python/vllm 路径），
# 导致连接中断 exit 255。修复：排除 grep 自身 + 当前 shell 及祖先 + sshd。

# 收集要杀的 PID（排除 grep/awk/kill 自身、当前 shell 及其父、sshd）
SELF_PID=$$
SELF_PPID=$PPID

# 用 ps -eo pid,comm 按进程名匹配（不含命令行参数，避免匹配到 SSH 会话的 bash -c）
# 匹配: python3/python3.12, VLLM::DPCoordinator, vllm 相关进程
# -i 忽略大小写（VLLM vs vllm），增加 Coordinator 匹配
kill_pids=$(ps -eo pid,ppid,comm | \
  grep -iE 'python|vllm|coordinator' | \
  grep -v 'grep' | \
  awk -v self="$SELF_PID" -v ppid="$SELF_PPID" '
    $1 != self && $1 != ppid && $2 != self && $2 != ppid {print $1}
  ')

if [ -n "$kill_pids" ]; then
  echo "$kill_pids" | xargs kill -9 2>/dev/null
  echo "killed $(echo "$kill_pids" | wc -l) processes"
else
  echo "no python/vllm processes to kill"
fi

sleep 2

# 报告残留（排除僵尸进程 Z）
remaining=$(ps -eo pid,stat,comm | grep -iE 'python|vllm|coordinator' | grep -v 'grep' | grep -v ' Z ' | wc -l)
echo "remaining active: $remaining"
