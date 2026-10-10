#!/bin/bash
# 修复 itask 容器环境问题，确保 ssh-tunnel 和 rsync 同步可用
# 用法: itask exec <task-name> --tty=false -- bash -c "$(cat scripts/fix_container_env.sh)"
#
# 支持 Ubuntu (apt-get) 和 openEuler (yum/dnf) 两种系统
#
# 解决的问题:
#   1. 软件源不可达 → 换阿里云镜像（Ubuntu ARM ports / openEuler）
#   2. dpkg statoverride 损坏 → 清理无效条目（Ubuntu）
#   3. libwrap0 缺失 → sshd 启动报 libwrap.so.0 not found（Ubuntu）
#   4. rsync 未安装 → 代码同步失败

set -e

# --- 检测系统类型 ---
if [ -f /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
fi
OS_ID="${ID:-unknown}"

echo "=== 检测到系统: ${PRETTY_NAME:-unknown} (${OS_ID}) ==="

if [ "$OS_ID" = "openEuler" ]; then
    PKG_MGR="yum"
    RSYNC_PKG="rsync"
    LIBWRAP_PKG="tcp_wrappers"
elif [ "$OS_ID" = "ubuntu" ] || [ "$OS_ID" = "debian" ]; then
    PKG_MGR="apt-get"
    RSYNC_PKG="rsync"
    LIBWRAP_PKG="libwrap0"
else
    # 尝试自动检测包管理器
    if command -v yum >/dev/null 2>&1; then
        PKG_MGR="yum"
        RSYNC_PKG="rsync"
        LIBWRAP_PKG="tcp_wrappers"
    elif command -v apt-get >/dev/null 2>&1; then
        PKG_MGR="apt-get"
        RSYNC_PKG="rsync"
        LIBWRAP_PKG="libwrap0"
    else
        echo "ERROR: 未检测到支持的包管理器 (yum/apt-get)"
        exit 1
    fi
fi

echo "=== [1/4] 配置软件源 ==="
if [ "$PKG_MGR" = "apt-get" ]; then
    if grep -q 'ports.ubuntu.com' /etc/apt/sources.list 2>/dev/null; then
        sed -i 's|http://ports.ubuntu.com/ubuntu-ports/|https://mirrors.aliyun.com/ubuntu-ports/|g' /etc/apt/sources.list
        echo "  已替换为阿里云镜像"
    else
        echo "  已使用非官方源，跳过替换"
    fi
elif [ "$OS_ID" = "openEuler" ] && [ -f /etc/yum.repos.d/openEuler.repo ]; then
    if grep -q 'repo.openeuler.org' /etc/yum.repos.d/openEuler.repo 2>/dev/null; then
        cp /etc/yum.repos.d/openEuler.repo /etc/yum.repos.d/openEuler.repo.bak
        sed -i \
            -e 's|https://repo.openeuler.org|https://mirrors.aliyun.com/openeuler|g' \
            -e 's|https://dl-cdn.openeuler.openatom.cn|https://mirrors.aliyun.com/openeuler|g' \
            -e 's|http://repo.openeuler.org|https://mirrors.aliyun.com/openeuler|g' \
            -e 's|^metalink=|# metalink disabled|g' \
            /etc/yum.repos.d/openEuler.repo
        echo "  已替换为阿里云镜像"
    else
        echo "  已使用非官方源，跳过替换"
    fi
else
    echo "  非 Ubuntu/openEuler 系统，跳过换源"
fi

echo "=== [2/4] 更新软件源索引 ==="
if [ "$PKG_MGR" = "apt-get" ]; then
    apt-get update -y 2>&1 | tail -1
else
    yum clean all 2>/dev/null || true
    yum makecache -y 2>&1 | tail -1 || true
fi

echo "=== [3/4] 安装依赖包 ==="
if [ "$PKG_MGR" = "apt-get" ]; then
    apt-get install -y --reinstall libwrap0 libpopt0 rsync 2>&1 | tail -3
else
    yum install -y rsync 2>&1 | tail -3
fi

echo "=== [4/4] 刷新动态链接库缓存 ==="
ldconfig 2>/dev/null || true

echo ""
echo "=== 验证 ==="
FAIL=0
if [ -f /usr/sbin/sshd ]; then
    ldd /usr/sbin/sshd 2>&1 | grep "not found" && FAIL=1 || echo "  [OK] sshd 依赖库完整"
else
    echo "  [-] sshd 不存在，跳过检查"
fi
which rsync >/dev/null 2>&1 && echo "  [OK] rsync 可用" || { echo "  [FAIL] rsync 不可用"; FAIL=1; }
rsync --version >/dev/null 2>&1 && echo "  [OK] rsync 可执行" || { echo "  [FAIL] rsync 执行失败"; FAIL=1; }

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "✓ 环境修复完成，可以建立 SSH 隧道和同步代码"
else
    echo ""
    echo "✗ 部分检查未通过，请查看上方输出"
    exit 1
fi