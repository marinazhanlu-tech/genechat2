"""
GeneChat2训练脚本

启动完整的训练流程，包括数据加载、模型训练和评估。

使用方法:
python scripts/train_genechat2.py --config configs/genechat_config.yaml
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import yaml
import torch

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import create_genechat2
from data.dataset_builder import create_data_module
from training.trainer import GeneChat2Trainer, TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        config: 配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="GeneChat2训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/genechat_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从检查点恢复训练"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="数据路径（覆盖配置文件）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（覆盖配置文件）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="批大小（覆盖配置文件）"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="学习率（覆盖配置文件）"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="最大训练步数（覆盖配置文件）"
    )

    args = parser.parse_args()

    # 加载配置
    logger.info(f"加载配置文件: {args.config}")
    config = load_config(args.config)

    # 命令行参数覆盖
    if args.data_path:
        config["paths"]["data_root"] = args.data_path
    if args.output_dir:
        config["paths"]["model_output_dir"] = args.output_dir
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["optimizer"]["learning_rate"] = args.learning_rate
    if args.max_steps:
        config["training"]["max_training_steps"] = args.max_steps

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")

    if device == "cpu":
        logger.warning("⚠️  使用CPU训练会非常慢，强烈建议使用GPU")

    # ========== 步骤1: 创建数据模块 ==========
    logger.info("=" * 60)
    logger.info("步骤1: 准备数据")
    logger.info("=" * 60)

    data_module = create_data_module(
        data_path=config["paths"]["ncbi_data"],
        cache_dir=config["paths"]["cache_dir"],
        train_split=config["data"]["splits"]["train"],
        test_split=config["data"]["splits"]["test"],
        random_seed=config["experiment"]["seed"]
    )

    # 准备数据
    data_module.prepare_data()

    # 设置数据集
    data_module.setup_datasets()

    # 获取数据加载器
    train_loader = data_module.get_dataloader(
        "train",
        batch_size=config["training"]["batch_size"],
        num_workers=4
    )

    # 测试集用于评估
    test_loader = data_module.get_dataloader(
        "test",
        batch_size=1,
        num_workers=2
    )

    logger.info(f"✓ 训练集大小: {len(train_loader.dataset)}")
    logger.info(f"✓ 测试集大小: {len(test_loader.dataset)}")

    # ========== 步骤2: 创建模型 ==========
    logger.info("=" * 60)
    logger.info("步骤2: 创建GeneChat2模型")
    logger.info("=" * 60)

    model = create_genechat2(
        gene_encoder_path=config["model"]["gene_encoder"]["model_path"],
        llm_path=config["model"]["language_model"]["model_path"],
        freeze_gene_encoder=config["model"]["gene_encoder"]["frozen"],
        freeze_adapter=config["model"]["adapter"]["frozen"],
        device=device
    )

    logger.info("✓ 模型创建成功")

    # ========== 步骤3: 创建训练器 ==========
    logger.info("=" * 60)
    logger.info("步骤3: 配置训练器")
    logger.info("=" * 60)

    trainer_config = TrainerConfig(
        max_epochs=config["training"]["max_epochs"],
        max_steps=config["training"].get("max_training_steps", 170000),
        batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["optimizer"]["learning_rate"],
        weight_decay=config["training"]["optimizer"]["weight_decay"],
        adam_beta1=config["training"]["optimizer"]["beta1"],
        adam_beta2=config["training"]["optimizer"]["beta2"],
        warmup_steps=config["training"]["scheduler"]["warmup_steps"],
        min_lr=config["training"]["scheduler"]["min_lr"],
        max_grad_norm=config["training"]["max_grad_norm"],
        fp16=config["training"]["fp16"],
        save_steps=config["training"]["checkpointing"]["save_steps"],
        save_total_limit=config["training"]["checkpointing"]["save_total_limit"],
        output_dir=config["paths"]["model_output_dir"],
        logging_steps=config["training"]["logging"]["steps"],
        log_dir=config["paths"]["log_dir"],
        eval_steps=500,
        device=device
    )

    trainer = GeneChat2Trainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=test_loader,
        config=trainer_config
    )

    # 恢复训练
    if args.resume:
        logger.info(f"从检查点恢复训练: {args.resume}")
        trainer.load_checkpoint(args.resume)

    logger.info("✓ 训练器配置完成")

    # ========== 步骤4: 开始训练 ==========
    logger.info("=" * 60)
    logger.info("步骤4: 开始训练")
    logger.info("=" * 60)

    try:
        trainer.train()
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        raise

    logger.info("=" * 60)
    logger.info("✓ 训练完成!")
    logger.info("=" * 60)

    # 保存最终模型信息
    model_info = model.get_model_info()
    import json
    with open(Path(config["paths"]["model_output_dir"]) / "model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)

    logger.info(f"模型已保存到: {config['paths']['model_output_dir']}")


if __name__ == "__main__":
    main()
