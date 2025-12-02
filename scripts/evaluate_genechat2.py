"""
GeneChat2评估脚本

评估训练好的模型，计算BLEU和METEOR指标。

使用方法:
python scripts/evaluate_genechat2.py --model_path checkpoints/checkpoint-best
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
from data.dataset_builder import create_data_module
from training.evaluator import create_evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="GeneChat2评估脚本")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型检查点路径"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/genechat_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="测试数据路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="评估结果输出目录"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="评估哪个数据集分割"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大评估样本数（用于快速测试）"
    )

    args = parser.parse_args()

    # 加载配置
    logger.info(f"加载配置文件: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")

    # ========== 步骤1: 加载模型 ==========
    logger.info("=" * 60)
    logger.info("步骤1: 加载模型")
    logger.info("=" * 60)

    logger.info(f"从检查点加载模型: {args.model_path}")

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

    # ========== 步骤2: 准备数据 ==========
    logger.info("=" * 60)
    logger.info("步骤2: 准备测试数据")
    logger.info("=" * 60)

    data_module = create_data_module(
        data_path=args.data_path or config["paths"]["ncbi_data"],
        cache_dir=config["paths"]["cache_dir"],
        train_split=config["data"]["splits"]["train"],
        test_split=config["data"]["splits"]["test"],
        random_seed=config["experiment"]["seed"]
    )

    # 准备数据
    data_module.prepare_data()
    data_module.setup_datasets()

    # 获取测试数据加载器
    test_loader = data_module.get_dataloader(
        args.split,
        batch_size=1,
        shuffle=False
    )

    # 限制样本数（如果指定）
    if args.max_samples:
        logger.info(f"限制评估样本数: {args.max_samples}")
        # 创建受限的数据加载器
        from torch.utils.data import Subset
        if args.split == "test":
            dataset = data_module.test_dataset
        elif args.split == "train":
            dataset = data_module.train_dataset
        else:
            dataset = data_module.val_dataset

        indices = list(range(min(args.max_samples, len(dataset))))
        subset = Subset(dataset, indices)

        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            collate_fn=test_loader.collate_fn
        )

    logger.info(f"✓ 测试集大小: {len(test_loader.dataset)}")

    # ========== 步骤3: 创建评估器 ==========
    logger.info("=" * 60)
    logger.info("步骤3: 创建评估器")
    logger.info("=" * 60)

    evaluator = create_evaluator(
        model=model,
        output_dir=args.output_dir,
        max_new_tokens=config["evaluation"]["generation"]["max_new_tokens"],
        device=device
    )

    logger.info("✓ 评估器创建成功")

    # ========== 步骤4: 开始评估 ==========
    logger.info("=" * 60)
    logger.info("步骤4: 评估模型")
    logger.info("=" * 60)

    results = evaluator.evaluate_dataset(test_loader)

    logger.info("=" * 60)
    logger.info("✓ 评估完成!")
    logger.info("=" * 60)

    # 打印摘要
    logger.info("\n最终结果摘要:")
    logger.info(f"BLEU-1: {results['average_metrics']['bleu_1']:.4f}")
    logger.info(f"BLEU-2: {results['average_metrics']['bleu_2']:.4f}")
    logger.info(f"BLEU-3: {results['average_metrics']['bleu_3']:.4f}")
    logger.info(f"BLEU-4: {results['average_metrics']['bleu_4']:.4f}")
    logger.info(f"METEOR: {results['average_metrics']['meteor']:.4f}")

    logger.info(f"\n结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
