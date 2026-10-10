#!/bin/bash
# cluster_ctl.sh - 多机部署编排统一入口（模型无关）
#
# 兼容性：bash 3.2+（macOS 自带 /bin/bash），不使用关联数组
#
# 设计边界：本脚本只懂"分组 × 机器 × 命令"三件事，不懂模型/部署形态——
#   模型差异（脚本路径、rank 参数、环境变量）全部外置到 plan 文件，由 agent/用户编写。
#   多机纪律（IP 校验、占用检查、杀干净、并行启动、就绪轮询）全部固化在本脚本内，
#   agent 只发一条命令，无法违反纪律。
#
# 通信通道：一律 itask exec <机名>（按任务名路由，名字由平台保证，无端口无 localhost，
#   不存在 SSH 隧道"端口被占静默误路由"的风险）。SSH 隧道仅保留用于日志下载（rsync）。
#
# 用法:
#   scripts/cluster_ctl.sh launch --plan <plan.env> [--only <组名>] [--force-kill] [--skip-ipcheck] [--dry-run]
#   scripts/cluster_ctl.sh kill   --plan <plan.env> [--only <组名>] [--dry-run]
#   scripts/cluster_ctl.sh check  --plan <plan.env>
#   scripts/cluster_ctl.sh wait   --plan <plan.env> [--max-wait <秒>] [--interval <秒>]
#   scripts/cluster_ctl.sh ips    --plan <plan.env>
#
# plan 文件格式（见 task/<task>/cluster_plan.env）:
#   TASK=<task-name>
#   WS=<远程工作空间根>
#   # CLUSTER_GROUPS 每项: "组名|机器1,机器2,...|启动命令|日志文件模板"
#   #   - {RANK} 占位符按组内机器顺序替换为 0,1,2,...
#   #   - 命令在远程机器上以 bash -l -c 执行（登录 shell，PATH 含 vllm）
#   #   - 日志模板用于 wait/check 显示（{RANK} 同样替换）
#
# launch 内置纪律（按序强制执行）:
#   1. itask list 解析全部机器 Pod-IP（机器不存在/未 Running → 硬报错）
#   2. IP 一致性校验: 命令串引用的部署脚本里必须写有每台机器的 Pod-IP
#      （从 $WS 映射回本地 workspace 路径做 grep；缺失即报错并列出正确 IP，
#        提示"本地改修改区 → sync → 再启动"；--skip-ipcheck 跳过）
#   3. 占用检查: 逐机查 vllm serve 进程，发现活跃服务即中止（--force-kill 才继续清理）
#   4. 逐机 kill_all + 复查进程归零（不干净即中止，绝不带残留启动）
#   5. 所有组所有节点本地并行 fork 同时启动（setsid nohup，与 agent 行为无关）

set -euo pipefail

LOCAL_WS="$(cd "$(dirname "$0")/.." && pwd)"   # 本地 workspace 根
DEFAULT_MAX_WAIT=1800
DEFAULT_INTERVAL=30

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
say()  { echo -e "$*"; }
err()  { echo -e "${RED}❌ $*${NC}" >&2; }
ok()   { echo -e "${GREEN}✅ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $*${NC}"; }
head2(){ echo -e "\n${CYAN}=== $* ===${NC}"; }

DRY_RUN=0
FORCE_KILL=0
SKIP_IPCHECK=0
ONLY_GROUP=""
PLAN_FILE=""
MAX_WAIT=$DEFAULT_MAX_WAIT
INTERVAL=$DEFAULT_INTERVAL

usage() { sed -n '2,46p' "$0" | grep -E '^#( |$)' | sed 's/^# \{0,1\}//'; exit 1; }

# ---------- plan 解析 ----------
load_plan() {
    [ -f "$PLAN_FILE" ] || { err "plan 文件不存在: $PLAN_FILE"; exit 1; }
    # shellcheck disable=SC1090
    source "$PLAN_FILE"
    [ -n "${TASK:-}" ]  || { err "plan 缺少 TASK=<task-name>"; exit 1; }
    [ -n "${WS:-}" ]    || { err "plan 缺少 WS=<远程工作空间根>"; exit 1; }
    [ "${#CLUSTER_GROUPS[@]:-0}" -gt 0 ] 2>/dev/null || { err "plan 缺少 CLUSTER_GROUPS 数组"; exit 1; }

    # 展开选中的组（--only 过滤）；{RANK} 占位符按组内机器顺序替换
    ALL_GROUP=(); ALL_HOSTS=(); ALL_CMD=(); ALL_LOG=(); ALL_RANK=()
    local g name hosts cmd log host rank
    for g in "${CLUSTER_GROUPS[@]}"; do
        IFS='|' read -r name hosts cmd log <<< "$g"
        [ -n "$name" ] && [ -n "$hosts" ] && [ -n "$cmd" ] || { err "GROUPS 项格式错误（须 4 段以 | 分隔）: $g"; exit 1; }
        if [ -n "$ONLY_GROUP" ] && [ "$name" != "$ONLY_GROUP" ]; then continue; fi
        rank=0
        IFS=',' read -ra host_arr <<< "$hosts"
        for host in "${host_arr[@]}"; do
            ALL_GROUP+=("$name"); ALL_HOSTS+=("$host"); ALL_RANK+=("$rank")
            ALL_CMD+=("${cmd//\{RANK\}/$rank}"); ALL_LOG+=("${log//\{RANK\}/$rank}")
            rank=$((rank + 1))
        done
    done
    [ "${#ALL_HOSTS[@]}" -gt 0 ] || { err "没有匹配的组（--only $ONLY_GROUP?）plan 里可用组: $(for g in "${CLUSTER_GROUPS[@]}"; do IFS='|' read -r n _ <<<"$g"; echo -n "$n "; done)"; exit 1; }
    # 去重后的机器清单（kill/占用检查 逐机执行一次）
    UNIQUE_HOSTS=($(printf '%s\n' "${ALL_HOSTS[@]}" | awk '!seen[$0]++'))
    say "plan: ${PLAN_FILE}  task=${TASK}  组=$(printf '%s\n' "${ALL_GROUP[@]}" | awk '!seen[$0]++' | tr '\n' ' ') 机器数=${#UNIQUE_HOSTS[@]}"
}

# ---------- itask 封装 ----------
itask_exec() {  # itask_exec <机名> <远程命令>；dry-run 模式打印到 stderr 并返回 0
    local host="$1" cmd="$2"
    if [ "$DRY_RUN" = 1 ]; then
        echo "  [dry-run] itask exec $host --tty=false -- bash -c '$cmd'" >&2
        return 0
    fi
    itask exec "$host" --tty=false -- bash -c "$cmd"
}

# ---------- Pod-IP 表（并行数组，bash 3.2 兼容） ----------
IP_HOSTS=(); IP_IPS=()
ip_of() {  # ip_of <机名> → 输出 IP；未解析返回 1
    local h="$1" i
    for i in "${!IP_HOSTS[@]}"; do
        if [ "${IP_HOSTS[$i]}" = "$h" ]; then echo "${IP_IPS[$i]}"; return 0; fi
    done
    return 1
}

# ---------- Step 1: 解析 Pod-IP + 可达性预检 ----------
resolve_ips() {
    head2 "Step 1/4 Pod-IP 解析与可达性预检（itask list 权威）"
    local itask_out host ip
    itask_out="$(itask list 2>/dev/null)" || { err "itask list 执行失败"; exit 1; }
    IP_HOSTS=(); IP_IPS=()
    for host in "${UNIQUE_HOSTS[@]}"; do
        ip="$(echo "$itask_out" | awk -v h="$host" '$1==h && $4=="Running" {print $9; exit}')"
        if [ -z "$ip" ]; then
            err "机器 $host 在 itask list 中不存在或非 Running 状态"
            echo "    itask list 当前输出（NAME / NODE-STATUS / POD-IP）:"; echo "$itask_out" | awk 'NR==1 || $4=="Running"' | head -15
            exit 1
        fi
        IP_HOSTS+=("$host"); IP_IPS+=("$ip")
        say "  $host → $ip"
    done
    # 路由可达性预检（dry-run 下恒通过）
    for host in "${UNIQUE_HOSTS[@]}"; do
        itask_exec "$host" "echo ok" >/dev/null 2>&1 || { err "$host itask exec 不可达（手工排查: itask exec $host --tty=false -- echo ok）"; exit 1; }
    done
    ok "全部 ${#UNIQUE_HOSTS[@]} 台机器 itask exec 按名路由可达"
}

# ---------- Step 2: IP 一致性校验 ----------
check_ips_in_scripts() {
    [ "$SKIP_IPCHECK" = 1 ] && { warn "--skip-ipcheck：跳过 IP 校验（风险自负）"; return 0; }
    head2 "Step 2/4 IP 一致性校验（部署脚本修改区 vs Pod-IP）"
    local i host ip cmd script_remote script_local found_any=0
    local CHECKED_KEYS=""
    for i in "${!ALL_HOSTS[@]}"; do
        host="${ALL_HOSTS[$i]}"; ip="$(ip_of "$host")"; cmd="${ALL_CMD[$i]}"
        # 提取命令串中的部署脚本路径（$WS 下的 *.sh），映射回本地检查
        for script_remote in $(echo "$cmd" | grep -oE "[^ ;&|]+\.sh" | sort -u); do
            case "$script_remote" in
                "$WS"/*) script_local="${script_remote/#$WS/$LOCAL_WS}" ;;
                *) continue ;;
            esac
            [ -f "$script_local" ] || { warn "  $host: 脚本本地不存在（可能未 sync），跳过: $script_remote"; continue; }
            local key="$host|$script_local"
            case " $CHECKED_KEYS " in *" $key "*) continue ;; esac
            CHECKED_KEYS="$CHECKED_KEYS $key"
            found_any=1
            if grep -qF "$ip" "$script_local"; then
                say "  $host($ip) 已写入 $(basename "$script_local") ✅"
            else
                err "$host 的 Pod-IP $ip 未出现在 $script_local 修改区！"
                echo "    修复步骤（本地优先，远端只跑不改）:"
                echo "      1. 本地打开 $script_local，修改区补上 $host → $ip"
                echo "      2. scripts/sync.sh <itask> --task $TASK --port <port>（或默认模式）"
                echo "      3. 重新 launch"
                exit 1
            fi
        done
    done
    [ "$found_any" = 1 ] && ok "IP 校验通过" || warn "命令串未引用 workspace 下的 .sh 脚本，跳过内容校验"
}

# ---------- Step 3: 占用检查 ----------
check_occupancy() {
    head2 "Step 3/4 占用检查（发现活跃服务即中止，须人工确认后 --force-kill）"
    local host out occupied=""
    if [ "$DRY_RUN" = 1 ]; then
        warn "dry-run 模式：占用检查/kill 未实际执行，以下步骤的输出均为模拟（实际启动时才真正检查）"
        return 0
    fi
    for host in "${UNIQUE_HOSTS[@]}"; do
        out="$(itask_exec "$host" "ps -eo pid,lstart,args | grep 'vllm serve' | grep -v grep" 2>/dev/null || true)"
        if [ -n "$out" ]; then
            warn "$host 存在 vllm serve 进程:"
            echo "$out" | head -3 | sed 's/^/    /'
            occupied="$occupied $host"
        else
            say "  $host: 无 vllm serve 进程"
        fi
    done
    if [ -n "$occupied" ]; then
        if [ "$FORCE_KILL" = 1 ]; then
            warn "占用机器（${occupied}）已由 --force-kill 确认清理，继续"
        else
            err "以下机器存在活跃 vllm serve 进程: ${occupied}"
            echo "    判断依据: 进程启动时间 + 命令行 + npu-smi 显存占用。可能是其他任务的部署——"
            echo "    DP 集群杀一个 node 会级联搞挂整组部署，绝不能盲目清理。"
            echo "    → 确认是本任务陈旧残留/可清理后，加 --force-kill 重新执行 launch"
            exit 1
        fi
    else
        ok "无占用"
    fi
}

# ---------- Step 4: kill + 复查 ----------
kill_and_verify() {
    head2 "清理残留进程（kill_all + 复查归零，不干净绝不启动）"
    local host out remain fail=0
    for host in "${UNIQUE_HOSTS[@]}"; do
        out="$(itask_exec "$host" "bash $WS/scripts/kill_all.sh" 2>/dev/null || true)"
        say "  $host: $(echo "$out" | tail -2 | tr '\n' ' ')"
    done
    sleep 3
    for host in "${UNIQUE_HOSTS[@]}"; do
        # [v] 正则技巧: itask exec 的 bash -c 包装进程命令行含 'vllm serve' 字样会被 pgrep -f 自匹配
        # 误报恒为 1；用 '[v]llm serve' 后包装进程（字面量 [v]llm）不匹配，真 vllm serve 进程照常匹配
        remain="$(itask_exec "$host" "pgrep -fc '[v]llm serve' 2>/dev/null || true")"
        if [ -n "$remain" ] && [ "$remain" != "0" ]; then
            err "$host 仍残留 $remain 个 vllm serve 进程，中止（手工排查: itask exec $host --tty=false -- bash -c \"ps -ef | grep 'vllm serve' | grep -v grep\"）"
            fail=1
        fi
    done
    [ "$fail" = 1 ] && exit 1
    ok "全部机器进程已清零"
}

# ---------- 并行启动 ----------
parallel_launch() {
    head2 "并行启动（全部组全部节点本地 fork，同时拉起）"
    local i host cmd log pids="" pid fail=0 out
    for i in "${!ALL_HOSTS[@]}"; do
        host="${ALL_HOSTS[$i]}"; cmd="${ALL_CMD[$i]}"; log="${ALL_LOG[$i]}"
        (
            if [ "$DRY_RUN" = 1 ]; then
                say "  [dry-run] [${ALL_GROUP[$i]}] $host RANK=${ALL_RANK[$i]} 日志=${log}"
                say "             itask exec $host --tty=false -- bash -c 'setsid nohup bash -l -c \"$cmd\" < /dev/null > /dev/null 2>&1 & disown'"
            else
                out="$(itask_exec "$host" "setsid nohup bash -l -c '$cmd' < /dev/null > /dev/null 2>&1 & disown; echo started-\$?" 2>&1)" \
                    && say "  [${ALL_GROUP[$i]}] $host RANK=${ALL_RANK[$i]} → $out" \
                    || { err "  [${ALL_GROUP[$i]}] $host 启动命令执行失败: $out"; exit 1; }
            fi
        ) &
        pids="$pids $!"
    done
    for pid in $pids; do wait "$pid" || fail=1; done
    [ "$fail" = 1 ] && { err "存在启动失败的节点，中止（其余节点已发出，用 kill 子命令清理后重试）"; exit 1; }
    ok "已同时发出 ${#ALL_HOSTS[@]} 个节点的启动命令"
    echo
    say "等待就绪: scripts/cluster_ctl.sh wait --plan $PLAN_FILE"
    say "节点日志（实时查看: itask exec <机名> -- tail -f <日志>）:"
    for i in "${!ALL_HOSTS[@]}"; do
        echo "  [${ALL_GROUP[$i]}] ${ALL_HOSTS[$i]}: ${ALL_LOG[$i]}"
    done
}

# ---------- 一次性状态 ----------
do_check() {
    head2 "节点状态一览（$(date '+%H:%M:%S')）"
    local i host log out procs state last
    for i in "${!ALL_HOSTS[@]}"; do
        host="${ALL_HOSTS[$i]}"; log="${ALL_LOG[$i]}"
        out="$(itask_exec "$host" "
            procs=\$(pgrep -fc '[v]llm serve' 2>/dev/null || echo 0)
            echo \"PROC=\$procs\"
            if [ -f '$log' ]; then
                if grep -q 'Application startup complete' '$log'; then echo 'STATE=READY'
                elif tail -30 '$log' | grep -qiE 'RuntimeError|Traceback.*Error|SIGKILL|core dumped'; then echo 'STATE=ERROR'
                else echo \"STATE=STARTING \$(tail -1 '$log' | cut -c1-100)\"; fi
            else echo 'STATE=NO_LOG'; fi" 2>/dev/null || echo "STATE=UNREACHABLE")"
        procs="$(echo "$out" | grep -oE 'PROC=[0-9]+' | cut -d= -f2 || true)"
        state="$(echo "$out" | grep -oE 'STATE=[A-Z_]+' | head -1 | cut -d= -f2 || true)"
        last="$(echo "$out" | grep '^STATE=STARTING' | cut -d' ' -f2- || true)"
        case "$state" in
            READY)       printf "  ✅ [%s] %-12s 进程=%-3s 就绪\n" "${ALL_GROUP[$i]}" "$host" "$procs" ;;
            ERROR)       printf "  ❌ [%s] %-12s 进程=%-3s 日志报错（itask exec %s -- tail -50 %s）\n" "${ALL_GROUP[$i]}" "$host" "$procs" "$host" "$log" ;;
            STARTING)    printf "  … [%s] %-12s 进程=%-3s %s\n" "${ALL_GROUP[$i]}" "$host" "$procs" "$last" ;;
            NO_LOG)      printf "  ⚠️  [%s] %-12s 进程=%-3s 日志尚未生成（脚本没跑起来?）\n" "${ALL_GROUP[$i]}" "$host" "$procs" ;;
            UNREACHABLE) printf "  ❌ [%s] %-12s itask exec 不可达\n" "${ALL_GROUP[$i]}" "$host" ;;
            *)           printf "  ? [%s] %-12s 输出异常: %s\n" "${ALL_GROUP[$i]}" "$host" "$(echo "$out" | head -1)" ;;
        esac
    done
}

# ---------- 轮询等待 ----------
do_wait() {
    head2 "轮询等待就绪（max_wait=${MAX_WAIT}s interval=${INTERVAL}s；任一节点死亡立即报错）"
    local elapsed=0 all_ready i host log out line_out DEAD_REPORTED="" fatal=0
    while [ "$elapsed" -lt "$MAX_WAIT" ]; do
        all_ready=1; line_out=""
        for i in "${!ALL_HOSTS[@]}"; do
            host="${ALL_HOSTS[$i]}"; log="${ALL_LOG[$i]}"
            out="$(itask_exec "$host" "
                procs=\$(pgrep -fc '[v]llm serve' 2>/dev/null || echo 0)
                if [ \"\$procs\" = \"0\" ]; then echo 'PROC=0'
                elif grep -q 'Application startup complete' '$log' 2>/dev/null; then echo 'READY'
                elif grep -q 'Engine core initialization failed' '$log' 2>/dev/null; then echo 'LOG_ERROR'
                else echo 'STARTING'; fi" 2>/dev/null || echo 'UNREACHABLE')"
            case "$out" in
                READY)    line_out="$line_out ✅${ALL_GROUP[$i]}:${host}" ;;
                STARTING) line_out="$line_out …${ALL_GROUP[$i]}:${host}"; all_ready=0 ;;
                LOG_ERROR)
                    all_ready=0; fatal=1
                    err "节点 ${ALL_GROUP[$i]}:$host 引擎初始化已崩溃（日志含 'Engine core initialization failed'——进程未退出但继续等待无意义）"
                    echo "  日志尾部: itask exec $host --tty=false -- bash -c \"tail -50 $log\""
                    line_out="$line_out ❌${ALL_GROUP[$i]}:${host}(引擎崩溃)"
                    ;;
                PROC=0)
                    all_ready=0
                    case " $DEAD_REPORTED " in *" $host "*) ;; *)
                        DEAD_REPORTED="$DEAD_REPORTED $host"
                        err "节点 ${ALL_GROUP[$i]}:$host 的 vllm serve 进程已退出！"
                        echo "  日志尾部（排查）: itask exec $host --tty=false -- bash -c \"tail -50 $log\""
                        echo "  常见原因: IP 写错被自检拦截 / 端口占用 / OOM。修正后: kill → 重新 launch"
                    ;; esac
                    line_out="$line_out ❌${ALL_GROUP[$i]}:${host}"
                    ;;
                *)        all_ready=0; line_out="$line_out ⚠️${ALL_GROUP[$i]}:${host}(不可达)" ;;
            esac
        done
        say "[${elapsed}s]$line_out"
        [ "$fatal" = 1 ] && { err "检测到引擎致命错误，提前结束等待（修正后 kill → 重新 launch）"; return 1; }
        [ "$all_ready" = 1 ] && { ok "全部 ${#ALL_HOSTS[@]} 个节点就绪"; return 0; }
        sleep "$INTERVAL"
        elapsed=$((elapsed + INTERVAL))
    done
    err "超时 ${MAX_WAIT}s 未全部就绪。用 check 子命令看各节点状态:"
    echo "  scripts/cluster_ctl.sh check --plan $PLAN_FILE"
    return 1
}

# ---------- ips 子命令 ----------
do_ips() {
    head2 "机器名 → Pod-IP → 组内 RANK（供部署脚本修改区 / conductor config.yaml 填写）"
    local i host key SEEN_KEYS=""
    for i in "${!ALL_HOSTS[@]}"; do
        host="${ALL_HOSTS[$i]}"; key="${ALL_GROUP[$i]}|$host"
        case " $SEEN_KEYS " in *" $key "*) continue ;; esac
        SEEN_KEYS="$SEEN_KEYS $key"
        printf "  [%s] RANK=%s  %-14s  %s\n" "${ALL_GROUP[$i]}" "${ALL_RANK[$i]}" "$host" "$(ip_of "$host" 2>/dev/null || echo '?')"
    done
}

# ---------- 参数解析 ----------
CMD="${1:-}"; [ -n "$CMD" ] || usage
shift
while [ $# -gt 0 ]; do
    case "$1" in
        --plan)         PLAN_FILE="${2:?--plan 需要参数}"; shift 2 ;;
        --only)         ONLY_GROUP="${2:?--only 需要组名}"; shift 2 ;;
        --force-kill)   FORCE_KILL=1; shift ;;
        --skip-ipcheck) SKIP_IPCHECK=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --max-wait)     MAX_WAIT="${2:?--max-wait 需要秒数}"; shift 2 ;;
        --interval)     INTERVAL="${2:?--interval 需要秒数}"; shift 2 ;;
        -h|--help)      usage ;;
        *) err "未知参数: $1"; usage ;;
    esac
done

case "$CMD" in
    launch)
        load_plan
        resolve_ips
        check_ips_in_scripts
        check_occupancy
        kill_and_verify
        parallel_launch
        ;;
    kill)
        load_plan
        resolve_ips
        kill_and_verify
        ok "kill 完成"
        ;;
    check)
        load_plan
        resolve_ips
        do_check
        ;;
    wait)
        load_plan
        resolve_ips
        do_wait
        ;;
    ips)
        load_plan
        resolve_ips
        do_ips
        ;;
    *)
        err "未知子命令: $CMD"; usage ;;
esac
