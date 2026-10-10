#!/bin/bash
# pd_profile_capture.sh - PD 分离 profiling 并行采集（P/D 真正同时 start，各自独立 stop）
# v2（2026-09-23）：双侧负载门控 + 等待活跃 + 多节点（防 dummy 陪跑污染，见 progress.md Round 7 续4）
#
# 为什么必须是"真并行"（三个实测坑，均返工过）：
#   1) 两侧 stop 串在同一命令链：/stop_profile 是阻塞调用（等 16 rank dump 完成，P 侧 ~1.5-2min），
#      排在后面的那侧 profiler 在先一侧 dump 期间持续采集——名义 2s 实采 ~3min，单卡 ascend_pt 2.5GB
#   2) P→D 顺序采集（哪怕各自序列独立）也不行：P 的 stop dump 阻塞 P 引擎 1-2min，期间无新
#      prefill/KV 下发，D 侧解码排空——D 窗口踩空转期（实测 3.26s 真窗口仅 33 个 lifecycle 事件）
#   3) agent 工具级"并行调用"可能被顺序执行——必须由本脚本用本地并行子进程保证同时起跑
#
# v2 新增（forcelb 轮双侧 100% dummy 教训）：
#   - 负载门控：触发前轮询双侧最新 loggers 行。P 侧任一引擎 prompt≥1000 tok/s 且 Running≥1；
#     D 侧任一引擎 generation≥5 tok/s 且 Running≥1；行龄≤15s（防陈旧行）。Running 单独不可信
#     （停滞请求 Running:1 实为 1.2 tok/s 空转，151k 行 kernel 全是 shape=2 dummy）。双侧同时
#     活跃多在 bench 波次切换期（wave N 解码尾 + wave N+1 prefill 头），等待即自然对齐，
#     无真实负载绝不采集。超时（默认 240s）则退出并给出当前状态。
#   - 多节点：p_nodes/d_nodes 支持逗号分隔（全组 8 引擎并行采集，供快慢卡跨节点对比；
#     各引擎 rank 目录按 PID 区分归属——P/D 引擎的 dp 前缀会重名）。
#
# 用法:
#   scripts/pd_profile_capture.sh <p_nodes> <d_nodes> [p_win=10] [d_win=3] [port=8001] [prof_dir] [wait_timeout=240]
#   例: scripts/pd_profile_capture.sh k3-p0,k3-p1,k3-p2,k3-p3 k3-d0,k3-d1,k3-d2,k3-d3
#       scripts/pd_profile_capture.sh k3-p0 k3-d0     # 单节点（兼容旧用法）
#
# 前置: 服务 READY、bench 负载在跑。
# 输出: 各序列 RC + 本轮新增 rank0 ascend_pt 的真实窗口与数据量。
# 采后: ① 形状有效性验证：P 侧算子输入形状应出现 8k 级真实 token（全 shape=1/2 即 dummy 陪跑，删数据重采）；
#       ② /parse-profile 解析归档（每份 rank0 单卡）；快慢卡多 rank 对比在远端批量 analyse 后只拉汇总。

set -uo pipefail

P_NODES_RAW="${1:?用法: pd_profile_capture.sh <p_nodes(逗号分隔可多节点)> <d_nodes(同)> [p_win=10] [d_win=3] [port=8001] [prof_dir] [wait_timeout=240]}"
D_NODES_RAW="${2:?缺 d_nodes}"
P_WIN="${3:-10}"
D_WIN="${4:-3}"
PORT="${5:-8001}"
WS_ROOT="/a3_inference/itask/workdir/yjq02324703/workspace"
WAIT_TIMEOUT="${7:-240}"

P_LIST=(${P_NODES_RAW//,/ })
D_LIST=(${D_NODES_RAW//,/ })
P0="${P_LIST[0]}"

# profiling 目录：显式传参 > 远端共享盘 mtime 最新（当前活跃任务）
if [ -z "${6:-}" ]; then
    PROF_DIR=$(itask exec "$P0" --tty=false -- bash -c "ls -dt $WS_ROOT/task/*/profiling 2>/dev/null | head -1" 2>/dev/null | tr -d '\r')
else
    PROF_DIR="$6"
fi
[ -n "$PROF_DIR" ] || { echo "❌ profiling 目录探测失败，请显式传第 6 参数"; exit 1; }
LOG_DIR=$(dirname "$PROF_DIR")/logs

# —— 负载状态抓取：双侧 8 引擎各取最新 loggers 行（在 P0 读共享日志；行龄防陈旧）——
fetch_status() {
    itask exec "$P0" --tty=false -- bash -c "
        now=\$(date +%s); YMD=\$(date +%Y)
        for f in $LOG_DIR/vllm_p_node*.log $LOG_DIR/vllm_d_node*.log; do
            line=\$(grep 'Running: ' \$f 2>/dev/null | tail -1 | tr -d '\r')
            [ -z \"\$line\" ] && continue
            r=\$(echo \"\$line\" | grep -oE 'Running: [0-9]+' | grep -oE '[0-9]+' | head -1)
            p=\$(echo \"\$line\" | grep -oE 'prompt throughput: [0-9.]+' | grep -oE '[0-9.]+\$')
            g=\$(echo \"\$line\" | grep -oE 'generation throughput: [0-9.]+' | grep -oE '[0-9.]+\$')
            ts=\$(echo \"\$line\" | grep -oE '[0-9]{2}-[0-9]{2} [0-9:]{8}' | head -1)
            ep=\$(date -d \"\$YMD-\$ts\" +%s 2>/dev/null || echo 0)
            age=\$(( now - ep )); [ \$age -lt 0 ] && age=0
            echo \"\$(basename \$f .log) \${r:-0} \${p:-0} \${g:-0} \$age\"
        done
    " 2>/dev/null | tr -d '\r'
}

# 门控：P 任一引擎 Running≥1 且 prompt≥1000；D 任一引擎 Running≥1 且 gen≥5；行龄 ≤15s
status_pass() {
    echo "$1" | awk '
        $1 ~ /^vllm_p_/ && $2>=1 && $3>=1000 && $5<=15 { p=1 }
        $1 ~ /^vllm_d_/ && $2>=1 && $4>=5    && $5<=15 { d=1 }
        END { exit !(p && d) }
    '
}

echo "=== PD profiling 采集 v2: P=[${P_LIST[*]}](${P_WIN}s) D=[${D_LIST[*]}](${D_WIN}s) port=$PORT ==="
echo "    profiling dir: $PROF_DIR"

# —— 等待双侧活跃（无真实负载绝不采集）——
gate_start=$SECONDS
while :; do
    ST=$(fetch_status)
    if [ -n "$ST" ] && status_pass "$ST"; then
        echo "    [负载门控通过 @+$((SECONDS - gate_start))s] 双侧活跃引擎:"
        echo "$ST" | awk '{printf "      %s: Running=%s prompt=%s gen=%s (行龄%ss)\n", $1,$2,$3,$4,$5}'
        break
    fi
    if [ $((SECONDS - gate_start)) -ge "$WAIT_TIMEOUT" ]; then
        echo "❌ 等待双侧活跃超时(${WAIT_TIMEOUT}s)。当前状态:"
        [ -n "$ST" ] && echo "$ST" | awk '{printf "      %s: Running=%s prompt=%s gen=%s (行龄%ss)\n", $1,$2,$3,$4,$5}'
        echo "   提示: 双侧同时活跃多在 bench 波次切换期; 检查 bench 是否在跑、D 侧是否有非停滞请求。"
        exit 1
    fi
    if [ $(( (SECONDS - gate_start) % 20 )) -lt 6 ]; then
        summary=$(echo "$ST" | awk 'BEGIN{mp=0;mg=0} $1~/^vllm_p_/ && $3+0>mp{mp=$3+0} $1~/^vllm_d_/ && $4+0>mg{mg=$4+0} END{printf "P_max_prompt=%.0f tok/s, D_max_gen=%.0f tok/s", mp, mg}')
        echo "    [+$((SECONDS - gate_start))s] 等待双侧活跃: ${summary:-<无日志>}"
    fi
    sleep 5
done

# 采集前快照（识别本轮新增目录；tr -d '\r' 防 itask 输出尾巴）
BEFORE=$(itask exec "$P0" --tty=false -- bash -c "ls -d '$PROF_DIR'/*_rank0_*_ascend_pt 2>/dev/null | xargs -n1 basename 2>/dev/null | tr -d '\r' | sort" 2>/dev/null)

# —— 双侧多节点并行序列（每节点独立 start→sleep→stop，本地子进程保证同时起跑）——
run_side() {  # run_side <nodes_csv> <window_s> <TAG> <out_file>
    local nodes_csv="$1" win="$2" tag="$3" out="$4"
    local jobs=() n short j fail=0
    for n in ${nodes_csv//,/ }; do
        short=$(echo "$n" | sed 's/^k3-//')
        itask exec "$n" --tty=false -- bash -c "
            curl -s -m 600 -X POST http://localhost:$PORT/start_profile; echo START_${tag}_${short}_RC=\$?
            sleep $win
            curl -s -m 600 -X POST http://localhost:$PORT/stop_profile; echo STOP_${tag}_${short}_RC=\$?
            date +%H:%M:%S
        " >> "$out" 2>&1 &
        jobs+=($!)
    done
    for j in "${jobs[@]}"; do wait "$j" || fail=1; done
    return $fail
}

P_OUT=$(mktemp /tmp/pd_profile_p.XXXXXX)
D_OUT=$(mktemp /tmp/pd_profile_d.XXXXXX)
run_side "$P_NODES_RAW" "$P_WIN" "P" "$P_OUT" & P_JOB=$!
run_side "$D_NODES_RAW" "$D_WIN" "D" "$D_OUT" & D_JOB=$!
echo ">>> $(date +%H:%M:%S) 双侧序列已并行发出 (P_JOB=$P_JOB, D_JOB=$D_JOB)"

fail=0
wait $P_JOB || { echo "❌ P 侧序列执行失败"; fail=1; }
echo "=== P 侧序列输出 ==="; cat "$P_OUT"
wait $D_JOB || { echo "❌ D 侧序列执行失败"; fail=1; }
echo "=== D 侧序列输出 ==="; cat "$D_OUT"
[ $fail -eq 0 ] || exit 1

# 核验：本轮新增 rank0 ascend_pt 的真实窗口与数据量（多引擎时应出现 P/D 组数个条目）
AFTER=$(itask exec "$P0" --tty=false -- bash -c "ls -d '$PROF_DIR'/*_rank0_*_ascend_pt 2>/dev/null | xargs -n1 basename 2>/dev/null | tr -d '\r' | sort" 2>/dev/null)
echo "=== 本轮新增 rank0 ascend_pt 真实窗口核验（窗口≈${P_WIN}s=P 侧 / ≈${D_WIN}s=D 侧）==="
printf '%s\n' "$AFTER" | comm -13 <(printf '%s\n' "$BEFORE" | grep -v '^$' || true) - | grep -v '^$' | while read -r d; do
    d=$(printf '%s' "$d" | tr -d '\r')
    itask exec "$P0" --tty=false -- bash -c "
        cd '$PROF_DIR/$d' || exit 1
        si=\$(cat PROF_*/device_0/start_info.0 2>/dev/null | grep -o '\"collectionDateBegin\":\"[^\"]*\"' | cut -d'\"' -f4)
        ei=\$(cat PROF_*/device_0/end_info.0   2>/dev/null | grep -o '\"collectionDateEnd\":\"[^\"]*\"'   | cut -d'\"' -f4)
        sz=\$(du -sh . | cut -f1)
        echo \"  $d  size=\$sz  window: \$si -> \$ei\"
    " 2>/dev/null || echo "  $d  ⚠️ 核验 itask exec 失败（偶发抖动，可手工重查）"
done
echo "=== 采集完成。下一步: ① 形状有效性验证（P 侧应见 8k 级真实 token 形状）② /parse-profile 解析 ==="
