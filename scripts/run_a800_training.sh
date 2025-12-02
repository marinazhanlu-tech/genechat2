#!/bin/bash
# GeneChat2 A800-80G GPU 完整训练脚本
# 针对A800-80G优化的训练配置
# 生成时间: 2025-12-02

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================================================"
echo "GeneChat2 A800-80G 完整训练"
echo "========================================================================"
echo ""

# 默认配置
CONFIG_FILE="configs/genechat_config.yaml"
OUTPUT_DIR="./checkpoints"
MAX_STEPS=170000
BATCH_SIZE=1
GRAD_ACCUM=8
LEARNING_RATE=1e-4

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad_accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --test)
            echo -e "${YELLOW}[TEST MODE]${NC} 使用测试配置 (100个基因, 1000步)"
            MAX_STEPS=1000
            TEST_MODE=true
            shift
            ;;
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --config PATH          配置文件路径 (默认: configs/genechat_config.yaml)"
            echo "  --output_dir PATH      输出目录 (默认: ./checkpoints)"
            echo "  --max_steps N          最大训练步数 (默认: 170000)"
            echo "  --batch_size N         批大小 (默认: 1)"
            echo "  --grad_accum N         梯度累积步数 (默认: 8)"
            echo "  --learning_rate LR     学习率 (默认: 1e-4)"
            echo "  --test                 测试模式 (100个基因, 1000步)"
            echo "  --resume PATH          从检查点恢复训练"
            echo "  -h, --help             显示帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                                    # 完整训练"
            echo "  $0 --test                             # 测试模式"
            echo "  $0 --max_steps 10000                  # 训练10k步"
            echo "  $0 --resume ./checkpoints/checkpoint-5000  # 从检查点恢复"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 显示配置
echo -e "${BLUE}训练配置:${NC}"
echo "  配置文件: $CONFIG_FILE"
echo "  输出目录: $OUTPUT_DIR"
echo "  最大步数: $MAX_STEPS"
echo "  批大小: $BATCH_SIZE"
echo "  梯度累积: $GRAD_ACCUM"
echo "  有效批大小: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  学习率: $LEARNING_RATE"
if [ ! -z "$RESUME_FROM" ]; then
    echo "  恢复训练: $RESUME_FROM"
fi
if [ "$TEST_MODE" = true ]; then
    echo -e "  ${YELLOW}[测试模式]${NC}"
fi
echo ""

# 检查GPU
echo -e "${BLUE}GPU信息:${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# 确认开始训练
if [ "$TEST_MODE" != true ] && [ -z "$RESUME_FROM" ]; then
    read -p "开始完整训练？这将需要24-48小时。(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "训练已取消"
        exit 0
    fi
fi

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p ./logs

# 设置环境变量（A800优化）
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_CACHE=./cache
export HF_HOME=./cache

echo ""
echo "========================================================================"
echo "开始训练"
echo "========================================================================"
echo ""

# 构建训练命令
TRAIN_CMD="python scripts/train_genechat2.py \
    --config $CONFIG_FILE \
    --output_dir $OUTPUT_DIR \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --fp16 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 1000 \
    --save_total_limit 5 \
    --seed 42"

# 添加测试模式参数
if [ "$TEST_MODE" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --max_train_samples 100"
fi

# 添加恢复训练参数
if [ ! -z "$RESUME_FROM" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume_from_checkpoint $RESUME_FROM"
fi

# 使用tee同时输出到终端和日志文件
LOG_FILE="./logs/training_$(date +%Y%m%d_%H%M%S).log"
echo "日志文件: $LOG_FILE"
echo ""

# 执行训练
eval $TRAIN_CMD 2>&1 | tee $LOG_FILE

# 检查训练是否成功
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo -e "${GREEN}✅ 训练完成！${NC}"
    echo "========================================================================"
    echo ""
    echo "检查点保存在: $OUTPUT_DIR"
    echo "日志保存在: $LOG_FILE"
    echo ""
    echo "下一步:"
    echo "1. 评估模型:"
    echo "   python scripts/evaluate_genechat2.py --model_path $OUTPUT_DIR/checkpoint-best"
    echo ""
    echo "2. 运行推理:"
    echo "   python scripts/inference_demo.py --model_path $OUTPUT_DIR/checkpoint-best"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo -e "${RED}❌ 训练失败${NC}"
    echo "========================================================================"
    echo ""
    echo "请检查日志文件: $LOG_FILE"
    exit 1
fi
