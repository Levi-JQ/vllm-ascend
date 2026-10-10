#调用示例
# 清除 CANN 工具链的 GCC 7.3.0 头文件路径，避免与系统 GCC 12 冲突
# unset CPLUS_INCLUDE_PATH
# unset C_INCLUDE_PATH
#如有必要，清理残留：cd /a3_inference/itask/workdir/yjq02324703/workspace/codebases/vllm-ascend && rm -rf csrc/build csrc/output csrc/build_out
#bash /a3_inference/itask/workdir/yjq02324703/workspace/scripts/install_vllm_ascend.sh /a3_inference/itask/workdir/yjq02324703/workspace/codebases/vllm /a3_inference/itask/workdir/yjq02324703/workspace/codebases/vllm-ascend

#读取传入脚本的参数分别赋值给vllm目录变量、vllm-ascend目录变量
VLLM_DIR=$1
VLLM_ASCEND_DIR=$2
#打印新的vllm和vllm-ascend目录
echo "VLLM_DIR: $VLLM_DIR"
echo "VLLM_ASCEND_DIR: $VLLM_ASCEND_DIR"
#检查VLLM_DIR是否存在
if [ -d "$VLLM_DIR" ]; then
    echo "VLLM_DIR exists."
else
    echo "VLLM_DIR does not exist."
    exit
fi
#检查VLLM_ASCEND_DIR是否存在
if [ -d "$VLLM_ASCEND_DIR" ]; then
    echo "VLLM_ASCEND_DIR exists."
else
    echo "VLLM_ASCEND_DIR does not exist."
    exit
fi


#卸载原始vllm
pip uninstall -y vllm
#卸载原始vllm-ascend
pip uninstall -y vllm-ascend

#替换pip源
pip config set global.index-url https://pypi.antfin-inc.com/simple
pip config set global.trusted-host pypi.antfin-inc.com

#安装本机vllm代码（在vllm目录下运行）：
cd $VLLM_DIR
VLLM_TARGET_DEVICE=empty pip install -U -e . -i https://pypi.antfin-inc.com/simple/

#安装本机vllm-ascend代码（在vllm-ascend目录下运行）：
cd $VLLM_ASCEND_DIR
export CPLUS_INCLUDE_PATH=/usr/include/c++/12:/usr/include/c++/12/`uname -i`-openEuler-linux
COMPILE_CUSTOM_KERNELS=1 pip install -e . --no-build-isolation -i https://pypi.antfin-inc.com/simple/ --no-deps