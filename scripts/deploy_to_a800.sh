#!/bin/bash
# GeneChat2 A800服务器一键部署脚本
# 在A800服务器上直接运行此脚本

set -e

echo "========================================================================"
echo "GeneChat2 A800服务器部署脚本"
echo "服务器: is-db6vekhjcf53xhrk (4×A800-80G)"
echo "========================================================================"
echo ""

# 1. 创建工作目录
echo "步骤1: 创建工作目录..."
cd /root
mkdir -p genechat2
mkdir -p checkpoints
mkdir -p cache
mkdir -p data
mkdir -p logs

# 2. 检查GPU
echo ""
echo "步骤2: 检查GPU..."
echo "------------------------------------------------------------------------"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# 3. 检查Python和PyTorch
echo ""
echo "步骤3: 检查Python环境..."
echo "------------------------------------------------------------------------"
python --version
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA版本: {torch.version.cuda}'); print(f'GPU可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"

# 4. 下载genechat2代码（从GitHub或其他源）
echo ""
echo "步骤4: 获取代码..."
echo "------------------------------------------------------------------------"
echo "请选择代码获取方式:"
echo "  1. 从GitHub克隆 (如果代码已上传)"
echo "  2. 手动创建项目结构"
echo ""
read -p "请选择 (1/2): " choice

if [ "$choice" = "1" ]; then
    read -p "请输入GitHub仓库URL: " repo_url
    git clone "$repo_url" genechat2
elif [ "$choice" = "2" ]; then
    echo "将创建项目基础结构..."
    cd genechat2

    # 创建目录结构
    mkdir -p models data training scripts configs

    echo "✓ 项目结构已创建"
    echo ""
    echo "注意: 你需要手动上传以下文件:"
    echo "  - models/*.py"
    echo "  - data/*.py"
    echo "  - training/*.py"
    echo "  - scripts/*.py"
    echo "  - configs/*.yaml"
    echo ""
    echo "建议使用以下方式之一上传:"
    echo "  1. 使用Web界面上传"
    echo "  2. 使用SFTP客户端"
    echo "  3. 通过cat命令直接创建文件（如下所示）"
fi

# 5. 安装依赖
echo ""
echo "步骤5: 安装Python依赖..."
echo "------------------------------------------------------------------------"

# 使用清华镜像加速
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    transformers>=4.30.0 \
    peft>=0.4.0 \
    accelerate>=0.20.0 \
    biopython>=1.81 \
    nltk>=3.8 \
    sacrebleu>=2.3.0 \
    datasets>=2.13.0 \
    tensorboard>=2.13.0 \
    omegaconf>=2.3.0 \
    safetensors>=0.4.3 \
    sentencepiece \
    protobuf

echo "✓ 依赖安装完成"

# 6. 下载NLTK数据
echo ""
echo "步骤6: 下载NLTK数据..."
echo "------------------------------------------------------------------------"
python -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print('✓ NLTK数据下载完成')
"

# 7. 设置环境变量
echo ""
echo "步骤7: 配置环境变量..."
echo "------------------------------------------------------------------------"

cat >> ~/.bashrc << 'ENVEOF'

# GeneChat2 环境变量
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=/root/cache
export HF_HOME=/root/cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

ENVEOF

source ~/.bashrc
echo "✓ 环境变量已配置"

# 8. 显示完成信息
echo ""
echo "========================================================================"
echo "✓ A800服务器基础环境配置完成！"
echo "========================================================================"
echo ""
echo "GPU信息:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""
echo "Python环境:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda}'); print(f'  GPU数量: {torch.cuda.device_count()}')"
echo ""
echo "下一步:"
echo "  1. 上传genechat2代码到 /root/genechat2/"
echo "  2. 运行测试: cd /root/genechat2 && bash scripts/run_a800_training.sh --test"
echo "  3. 完整训练: bash scripts/run_a800_training.sh --config configs/genechat_a800_config.yaml"
echo ""
echo "========================================================================"
