#!/bin/bash
# 从MacBook上传代码到A800服务器的脚本
# 在MacBook上运行

# ============================================================================
# 配置区域 - 请修改为你的实际配置
# ============================================================================

# A800服务器信息
SERVER_USER="your_username"        # 修改为你的用户名
SERVER_IP="your_server_ip"         # 修改为A800服务器IP
SERVER_PORT="22"                    # SSH端口，默认22

# 远程路径
REMOTE_BASE_DIR="/home/${SERVER_USER}"  # 或 /data 等

# 本地路径
LOCAL_PROJECT_DIR="/Users/myt/Documents/genechat/genechat2"

# ============================================================================

set -e

echo "========================================================================"
echo "GeneChat2 代码上传到A800服务器"
echo "========================================================================"
echo ""

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 检查本地项目目录
if [ ! -d "$LOCAL_PROJECT_DIR" ]; then
    echo -e "${RED}错误: 本地项目目录不存在: $LOCAL_PROJECT_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}本地项目目录: $LOCAL_PROJECT_DIR${NC}"
echo -e "${GREEN}目标服务器: ${SERVER_USER}@${SERVER_IP}${NC}"
echo ""

# 显示项目文件统计
echo "项目文件统计:"
echo "  Python文件: $(find $LOCAL_PROJECT_DIR -name "*.py" | wc -l)"
echo "  配置文件: $(find $LOCAL_PROJECT_DIR -name "*.yaml" -o -name "*.yml" | wc -l)"
echo "  文档文件: $(find $LOCAL_PROJECT_DIR -name "*.md" | wc -l)"
echo ""

# 确认上传
read -p "确认上传代码到 ${SERVER_USER}@${SERVER_IP}? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "上传已取消"
    exit 0
fi

# 1. 打包代码（排除不必要的文件）
echo ""
echo "步骤1: 打包代码..."
echo "------------------------------------------------------------------------"

cd "$(dirname $LOCAL_PROJECT_DIR)"
TAR_FILE="genechat2_$(date +%Y%m%d_%H%M%S).tar.gz"

tar -czf "$TAR_FILE" \
    --exclude="*.pyc" \
    --exclude="__pycache__" \
    --exclude=".git" \
    --exclude="*.log" \
    --exclude="checkpoints" \
    --exclude="cache" \
    --exclude="data/ncbi_genes" \
    --exclude="venv*" \
    --exclude=".DS_Store" \
    genechat2/

TAR_SIZE=$(du -h "$TAR_FILE" | cut -f1)
echo -e "${GREEN}✓ 代码已打包: $TAR_FILE (大小: $TAR_SIZE)${NC}"

# 2. 测试SSH连接
echo ""
echo "步骤2: 测试SSH连接..."
echo "------------------------------------------------------------------------"

if ssh -p $SERVER_PORT -o ConnectTimeout=10 ${SERVER_USER}@${SERVER_IP} "echo '连接成功'" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ SSH连接正常${NC}"
else
    echo -e "${RED}✗ SSH连接失败，请检查:${NC}"
    echo "  1. 服务器IP是否正确"
    echo "  2. SSH端口是否正确"
    echo "  3. 用户名是否正确"
    echo "  4. 是否配置了SSH密钥或密码"
    exit 1
fi

# 3. 上传代码
echo ""
echo "步骤3: 上传代码包..."
echo "------------------------------------------------------------------------"

scp -P $SERVER_PORT "$TAR_FILE" ${SERVER_USER}@${SERVER_IP}:${REMOTE_BASE_DIR}/

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 代码上传成功${NC}"
else
    echo -e "${RED}✗ 上传失败${NC}"
    exit 1
fi

# 4. 在服务器上解压
echo ""
echo "步骤4: 在服务器上解压代码..."
echo "------------------------------------------------------------------------"

ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_IP} << 'ENDSSH'
    set -e

    # 颜色
    GREEN='\033[0;32m'
    NC='\033[0m'

    cd $HOME

    # 找到最新的tar包
    TAR_FILE=$(ls -t genechat2_*.tar.gz | head -1)

    echo "解压: $TAR_FILE"
    tar -xzf "$TAR_FILE"

    echo -e "${GREEN}✓ 代码已解压到: $(pwd)/genechat2${NC}"

    # 显示文件结构
    echo ""
    echo "项目结构:"
    ls -lh genechat2/ | head -15
ENDSSH

echo -e "${GREEN}✓ 解压完成${NC}"

# 5. 清理本地临时文件
echo ""
echo "步骤5: 清理临时文件..."
echo "------------------------------------------------------------------------"

rm "$TAR_FILE"
echo -e "${GREEN}✓ 本地临时文件已清理${NC}"

# 6. 显示下一步操作
echo ""
echo "========================================================================"
echo -e "${GREEN}✓ 代码上传完成！${NC}"
echo "========================================================================"
echo ""
echo "下一步操作："
echo ""
echo "1. SSH连接到服务器:"
echo "   ssh ${SERVER_USER}@${SERVER_IP}"
echo ""
echo "2. 启动Docker容器:"
echo "   docker run -it --gpus all \\"
echo "     --name genechat2_a800 \\"
echo "     --shm-size=32g \\"
echo "     -v ${REMOTE_BASE_DIR}/genechat2:/workspace/genechat2 \\"
echo "     -v ${REMOTE_BASE_DIR}/checkpoints:/workspace/checkpoints \\"
echo "     -p 6006:6006 \\"
echo "     pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel \\"
echo "     /bin/bash"
echo ""
echo "3. 在容器内配置环境:"
echo "   cd /workspace/genechat2"
echo "   bash scripts/setup_a800_training.sh"
echo ""
echo "4. 开始训练:"
echo "   bash scripts/run_a800_training.sh --test"
echo ""
echo "========================================================================"
