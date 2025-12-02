#!/bin/bash
# GeneChat2 A800-80G GPU训练环境配置脚本
# 生成时间: 2025-12-02

set -e

echo "========================================================================"
echo "GeneChat2 A800-80G GPU训练环境配置"
echo "========================================================================"
echo ""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[i]${NC} $1"
}

# 1. 检查CUDA和GPU
echo "步骤1: 检查GPU环境"
echo "------------------------------------------------------------------------"

if command -v nvidia-smi &> /dev/null; then
    print_status "找到nvidia-smi"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

    # 检查是否是A800
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    if [[ $GPU_NAME == *"A800"* ]]; then
        print_status "检测到 NVIDIA A800-80G GPU ✓"
    else
        print_info "检测到GPU: $GPU_NAME"
    fi

    # 检查GPU内存
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    if [ $GPU_MEM -ge 70000 ]; then
        print_status "GPU内存: ${GPU_MEM}MB (满足训练要求)"
    else
        print_error "GPU内存不足: ${GPU_MEM}MB (需要70GB+)"
        exit 1
    fi
else
    print_error "未找到nvidia-smi，请确认NVIDIA驱动已安装"
    exit 1
fi

echo ""

# 2. 检查CUDA版本
echo "步骤2: 检查CUDA版本"
echo "------------------------------------------------------------------------"

if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_status "CUDA版本: $CUDA_VERSION"
else
    print_info "未找到nvcc，将使用conda/pip安装的CUDA"
fi

echo ""

# 3. 创建Python虚拟环境
echo "步骤3: 创建Python虚拟环境"
echo "------------------------------------------------------------------------"

VENV_DIR="./venv_genechat2"

if [ -d "$VENV_DIR" ]; then
    print_info "虚拟环境已存在: $VENV_DIR"
    read -p "是否重新创建？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf $VENV_DIR
        python3 -m venv $VENV_DIR
        print_status "虚拟环境已重新创建"
    fi
else
    python3 -m venv $VENV_DIR
    print_status "虚拟环境已创建: $VENV_DIR"
fi

# 激活虚拟环境
source $VENV_DIR/bin/activate
print_status "虚拟环境已激活"

echo ""

# 4. 升级pip
echo "步骤4: 升级pip"
echo "------------------------------------------------------------------------"

pip install --upgrade pip
print_status "pip已升级到最新版本"

echo ""

# 5. 安装PyTorch (CUDA 12.1版本，适配A800)
echo "步骤5: 安装PyTorch (GPU版本)"
echo "------------------------------------------------------------------------"

print_info "正在安装PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证PyTorch CUDA
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}'); print(f'GPU数量: {torch.cuda.device_count()}'); print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    print_status "PyTorch GPU支持验证成功"
else
    print_error "PyTorch无法访问GPU，请检查CUDA安装"
    exit 1
fi

echo ""

# 6. 安装项目依赖
echo "步骤6: 安装项目依赖"
echo "------------------------------------------------------------------------"

print_info "正在安装requirements.txt中的依赖..."
pip install -r requirements.txt

print_status "所有依赖已安装"

echo ""

# 7. 下载NLTK数据
echo "步骤7: 下载NLTK数据 (用于评估)"
echo "------------------------------------------------------------------------"

python -c "
import nltk
import ssl

# 处理SSL证书问题
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 下载必需的数据
print('下载punkt...')
nltk.download('punkt', quiet=True)
print('下载wordnet...')
nltk.download('wordnet', quiet=True)
print('下载omw-1.4...')
nltk.download('omw-1.4', quiet=True)
print('完成！')
"

print_status "NLTK数据下载完成"

echo ""

# 8. 创建必要的目录
echo "步骤8: 创建项目目录"
echo "------------------------------------------------------------------------"

mkdir -p ./data/ncbi_genes
mkdir -p ./data/processed
mkdir -p ./checkpoints
mkdir -p ./logs
mkdir -p ./cache
mkdir -p ./evaluation_results
mkdir -p ./predictions

print_status "项目目录创建完成"

echo ""

# 9. 测试GPU内存和性能
echo "步骤9: GPU性能测试"
echo "------------------------------------------------------------------------"

python -c "
import torch
import time

print('正在进行GPU性能测试...')

# 测试GPU内存
if torch.cuda.is_available():
    device = torch.device('cuda:0')

    # 获取GPU属性
    props = torch.cuda.get_device_properties(0)
    print(f'GPU名称: {props.name}')
    print(f'总内存: {props.total_memory / 1e9:.2f} GB')
    print(f'可用内存: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB')

    # 测试张量运算速度
    print('\\n测试计算性能...')
    size = 10000
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)

    # 预热
    _ = torch.matmul(x, y)
    torch.cuda.synchronize()

    # 计时
    start = time.time()
    for _ in range(10):
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f'矩阵乘法测试 (10次 {size}x{size}): {elapsed:.2f}秒')
    print(f'平均每次: {elapsed/10:.3f}秒')
    print(f'估算TFLOPS: {(10 * 2 * size**3 / 1e12) / elapsed:.2f}')

    print('\\n✅ GPU性能测试完成！')
else:
    print('❌ 无法访问GPU')
    exit(1)
"

echo ""

# 10. 验证模型可以加载
echo "步骤10: 验证核心组件"
echo "------------------------------------------------------------------------"

print_info "验证模型组件加载..."

python -c "
import sys
sys.path.insert(0, '.')

try:
    from models.gene_encoder import GeneEncodingConfig
    print('✓ 基因编码器配置加载成功')

    from models.adapter import AdapterConfig
    print('✓ 适配器配置加载成功')

    from models.llm_wrapper import VicunaConfig
    print('✓ LLM配置加载成功')

    from models.genechat2 import GeneChat2Config
    print('✓ 主模型配置加载成功')

    from data.ncbi_processor import NCBIConfig
    print('✓ 数据处理器加载成功')

    from training.trainer import TrainerConfig
    print('✓ 训练器配置加载成功')

    from training.evaluator import EvaluationConfig
    print('✓ 评估器配置加载成功')

    print('\\n✅ 所有核心组件验证通过！')
except Exception as e:
    print(f'\\n❌ 组件加载失败: {e}')
    exit(1)
"

print_status "核心组件验证完成"

echo ""

# 11. 显示配置摘要
echo "========================================================================"
echo "环境配置完成摘要"
echo "========================================================================"
echo ""

python -c "
import torch
import sys

print('Python版本:', sys.version.split()[0])
print('PyTorch版本:', torch.__version__)
print('CUDA版本:', torch.version.cuda)
print('cuDNN版本:', torch.backends.cudnn.version())
print('')
print('GPU信息:')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name}')
        print(f'    总内存: {props.total_memory / 1e9:.2f} GB')
        print(f'    计算能力: {props.major}.{props.minor}')
"

echo ""
print_status "GeneChat2训练环境配置完成！"
echo ""
echo "========================================================================"
echo "下一步操作"
echo "========================================================================"
echo ""
echo "1. 启动小规模测试训练 (1000步):"
echo "   python scripts/train_genechat2.py --config configs/genechat_config.yaml --max_steps 1000"
echo ""
echo "2. 启动完整训练 (170k步):"
echo "   python scripts/train_genechat2.py --config configs/genechat_config.yaml"
echo ""
echo "3. 监控训练进度:"
echo "   tensorboard --logdir ./logs"
echo ""
echo "4. 评估模型:"
echo "   python scripts/evaluate_genechat2.py --model_path ./checkpoints/checkpoint-best"
echo ""
echo "========================================================================"
