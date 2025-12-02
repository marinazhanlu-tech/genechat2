"""
GeneChat2推理演示脚本

使用训练好的模型对单个基因进行功能预测。

使用方法:
python scripts/inference_demo.py --model_path checkpoints/checkpoint-best --dna_sequence "ATCG..."
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import create_genechat2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_sequence_from_file(file_path: str) -> str:
    """从文件加载DNA序列

    Args:
        file_path: FASTA或文本文件路径

    Returns:
        sequence: DNA序列
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 过滤FASTA头部
    sequence = ""
    for line in lines:
        line = line.strip()
        if not line.startswith('>'):
            sequence += line

    return sequence.upper()


def main():
    parser = argparse.ArgumentParser(description="GeneChat2推理演示")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型检查点路径"
    )
    parser.add_argument(
        "--dna_sequence",
        type=str,
        default=None,
        help="DNA序列（直接输入）"
    )
    parser.add_argument(
        "--sequence_file",
        type=str,
        default=None,
        help="DNA序列文件路径（FASTA或文本）"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="please predict the function of this gene",
        help="提示文本"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/genechat_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="最大生成token数"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成温度"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="nucleus sampling参数"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="beam search数量"
    )

    args = parser.parse_args()

    # 验证输入
    if not args.dna_sequence and not args.sequence_file:
        parser.error("必须提供 --dna_sequence 或 --sequence_file")

    # 加载配置
    logger.info(f"加载配置文件: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")

    # ========== 步骤1: 加载模型 ==========
    logger.info("=" * 60)
    logger.info("加载GeneChat2模型")
    logger.info("=" * 60)

    logger.info(f"从检查点加载: {args.model_path}")

    model = create_genechat2(
        gene_encoder_path=config["model"]["gene_encoder"]["model_path"],
        llm_path=config["model"]["language_model"]["model_path"],
        freeze_gene_encoder=True,
        freeze_adapter=True,
        device=device
    )

    # 加载检查点
    model.load_model(args.model_path)
    model.eval()

    logger.info("✓ 模型加载成功")

    # ========== 步骤2: 准备输入 ==========
    logger.info("=" * 60)
    logger.info("准备输入数据")
    logger.info("=" * 60)

    # 获取DNA序列
    if args.sequence_file:
        logger.info(f"从文件加载序列: {args.sequence_file}")
        dna_sequence = load_sequence_from_file(args.sequence_file)
    else:
        dna_sequence = args.dna_sequence

    logger.info(f"DNA序列长度: {len(dna_sequence)} bp")
    logger.info(f"提示: {args.prompt}")

    # 验证序列
    valid_bases = set('ACGTN')
    if not all(base in valid_bases for base in dna_sequence.upper()):
        logger.warning("⚠️  DNA序列包含无效字符，将被清理")
        dna_sequence = ''.join(base if base in valid_bases else 'N'
                               for base in dna_sequence.upper())

    # ========== 步骤3: 生成预测 ==========
    logger.info("=" * 60)
    logger.info("生成基因功能预测")
    logger.info("=" * 60)

    logger.info("正在生成...")

    with torch.no_grad():
        prediction = model.generate(
            dna_sequence=dna_sequence,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams
        )

    # ========== 步骤4: 显示结果 ==========
    logger.info("=" * 60)
    logger.info("预测结果")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("基因功能预测")
    print("=" * 60)
    print(f"\nDNA序列长度: {len(dna_sequence)} bp")
    print(f"序列片段: {dna_sequence[:100]}...")
    print(f"\n提示: {args.prompt}")
    print(f"\n预测的功能描述:")
    print("-" * 60)
    print(prediction)
    print("=" * 60)

    # 保存结果
    output_file = Path("inference_output.txt")
    with open(output_file, 'w') as f:
        f.write(f"DNA序列长度: {len(dna_sequence)} bp\n")
        f.write(f"序列: {dna_sequence}\n\n")
        f.write(f"提示: {args.prompt}\n\n")
        f.write(f"预测结果:\n{prediction}\n")

    logger.info(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
