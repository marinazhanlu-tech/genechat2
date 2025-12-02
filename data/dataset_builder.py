"""
GeneChat数据集构建器

实现PyTorch Dataset和DataLoader，处理NCBI基因数据。
支持论文中的三元组格式：(DNA序列, 提示, 目标描述)

关键特性：
- PyTorch Dataset实现
- 高效的数据加载
- 支持训练/验证/测试分割
- 符合论文数据格式
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import random
import numpy as np
from dataclasses import dataclass

from .ncbi_processor import GeneTriplet, NCBIGeneProcessor, create_ncbi_processor

logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """数据集配置"""
    data_path: str = "./data/ncbi_genes"
    cache_dir: str = "./cache"
    max_sequence_length: int = 160000
    min_sequence_length: int = 1000
    train_split: float = 0.95
    val_split: float = 0.0
    test_split: float = 0.05
    random_seed: int = 42
    shuffle: bool = True
    num_workers: int = 4


class GeneChatDataset(Dataset):
    """GeneChat数据集（论文格式）"""

    def __init__(
        self,
        gene_triplets: List[GeneTriplet],
        split: str = "train",
        config: Optional[DatasetConfig] = None
    ):
        """
        Args:
            gene_triplets: 基因三元组列表
            split: 数据集分割 ("train", "val", "test")
            config: 数据集配置
        """
        self.gene_triplets = gene_triplets
        self.split = split
        self.config = config or DatasetConfig()

        logger.info(f"初始化{split}数据集，包含 {len(gene_triplets)} 个样本")

    def __len__(self) -> int:
        return len(self.gene_triplets)

    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本

        Returns:
            sample: 包含DNA序列、提示和目标描述的字典
        """
        triplet = self.gene_triplets[idx]

        sample = {
            "dna_sequence": triplet.dna_sequence,
            "prompt": triplet.prompt,
            "target_description": triplet.target_description,
            "metadata": triplet.gene_metadata
        }

        return sample

    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        seq_lengths = []
        desc_lengths = []

        for triplet in self.gene_triplets:
            seq_lengths.append(len(triplet.dna_sequence))
            desc_lengths.append(len(triplet.target_description.split()))

        return {
            "num_samples": len(self.gene_triplets),
            "avg_sequence_length": np.mean(seq_lengths),
            "max_sequence_length": np.max(seq_lengths),
            "min_sequence_length": np.min(seq_lengths),
            "avg_description_length": np.mean(desc_lengths),
            "max_description_length": np.max(desc_lengths),
            "min_description_length": np.min(desc_lengths)
        }


class GeneChatDataModule:
    """数据模块：管理训练/验证/测试数据集"""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # 设置随机种子
        self._set_seed(config.random_seed)

    def _set_seed(self, seed: int):
        """设置随机种子以保证可复现性"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def prepare_data(self, force_download: bool = False):
        """准备数据（下载和处理）

        Args:
            force_download: 是否强制重新下载
        """
        cache_file = Path(self.config.cache_dir) / "gene_triplets.json"

        # 检查缓存
        if cache_file.exists() and not force_download:
            logger.info(f"从缓存加载数据: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.gene_triplets = [GeneTriplet(**triplet) for triplet in data]
        else:
            logger.info("数据未缓存，开始下载和处理...")
            # 创建NCBI处理器
            processor = create_ncbi_processor(
                email="your_email@example.com",  # 需要用户配置
                max_sequence_length=self.config.max_sequence_length,
                cache_dir=self.config.cache_dir
            )

            # 下载数据（示例：1000个基因用于测试）
            self.gene_triplets = processor.download_comprehensive_dataset(
                organism="Homo sapiens",
                limit=1000,  # 完整数据集使用50248
                output_dir=self.config.data_path
            )

            # 保存到缓存
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(
                    [self._triplet_to_dict(t) for t in self.gene_triplets],
                    f,
                    indent=2
                )
            logger.info(f"数据已缓存到: {cache_file}")

        logger.info(f"总共加载 {len(self.gene_triplets)} 个基因")

    def _triplet_to_dict(self, triplet: GeneTriplet) -> Dict:
        """将GeneTriplet转换为字典"""
        return {
            "dna_sequence": triplet.dna_sequence,
            "prompt": triplet.prompt,
            "target_description": triplet.target_description,
            "gene_metadata": triplet.gene_metadata
        }

    def setup_datasets(self):
        """设置训练/验证/测试数据集（论文：95:5分割）"""
        if not hasattr(self, 'gene_triplets') or not self.gene_triplets:
            raise ValueError("数据未准备，请先调用 prepare_data()")

        # 过滤序列长度
        filtered_triplets = []
        for triplet in self.gene_triplets:
            seq_len = len(triplet.dna_sequence)
            if self.config.min_sequence_length <= seq_len <= self.config.max_sequence_length:
                filtered_triplets.append(triplet)

        logger.info(f"过滤后剩余 {len(filtered_triplets)} 个样本")

        # 打乱数据
        if self.config.shuffle:
            random.shuffle(filtered_triplets)

        # 分割数据（论文使用95:5训练/测试分割）
        total = len(filtered_triplets)
        train_size = int(total * self.config.train_split)
        val_size = int(total * self.config.val_split)

        train_triplets = filtered_triplets[:train_size]
        val_triplets = filtered_triplets[train_size:train_size + val_size]
        test_triplets = filtered_triplets[train_size + val_size:]

        # 创建数据集
        self.train_dataset = GeneChatDataset(train_triplets, "train", self.config)
        self.val_dataset = GeneChatDataset(val_triplets, "val", self.config) if val_triplets else None
        self.test_dataset = GeneChatDataset(test_triplets, "test", self.config)

        logger.info(f"数据集分割完成:")
        logger.info(f"  训练集: {len(self.train_dataset)} 样本")
        if self.val_dataset:
            logger.info(f"  验证集: {len(self.val_dataset)} 样本")
        logger.info(f"  测试集: {len(self.test_dataset)} 样本")

    def get_dataloader(
        self,
        split: str = "train",
        batch_size: int = 1,
        shuffle: Optional[bool] = None,
        num_workers: Optional[int] = None
    ) -> DataLoader:
        """获取DataLoader

        Args:
            split: 数据集分割 ("train", "val", "test")
            batch_size: 批大小
            shuffle: 是否打乱（默认训练集打乱，其他不打乱）
            num_workers: 工作进程数

        Returns:
            dataloader: PyTorch DataLoader
        """
        if split == "train":
            dataset = self.train_dataset
            shuffle = shuffle if shuffle is not None else True
        elif split == "val":
            dataset = self.val_dataset
            shuffle = shuffle if shuffle is not None else False
        elif split == "test":
            dataset = self.test_dataset
            shuffle = shuffle if shuffle is not None else False
        else:
            raise ValueError(f"未知的split: {split}")

        if dataset is None:
            raise ValueError(f"{split}数据集未初始化")

        num_workers = num_workers if num_workers is not None else self.config.num_workers

        # 自定义collate函数（因为DNA序列长度不同）
        def collate_fn(batch):
            """将batch整理为字典"""
            return {
                "dna_sequences": [item["dna_sequence"] for item in batch],
                "prompts": [item["prompt"] for item in batch],
                "target_descriptions": [item["target_description"] for item in batch],
                "metadata": [item["metadata"] for item in batch]
            }

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        return dataloader

    def get_statistics(self) -> Dict:
        """获取所有数据集的统计信息"""
        stats = {}

        if self.train_dataset:
            stats["train"] = self.train_dataset.get_statistics()
        if self.val_dataset:
            stats["val"] = self.val_dataset.get_statistics()
        if self.test_dataset:
            stats["test"] = self.test_dataset.get_statistics()

        return stats


def create_data_module(
    data_path: str = "./data/ncbi_genes",
    cache_dir: str = "./cache",
    train_split: float = 0.95,
    test_split: float = 0.05,
    random_seed: int = 42
) -> GeneChatDataModule:
    """创建数据模块的便捷函数（论文配置）"""
    config = DatasetConfig(
        data_path=data_path,
        cache_dir=cache_dir,
        train_split=train_split,
        val_split=0.0,  # 论文没有验证集
        test_split=test_split,
        random_seed=random_seed
    )
    return GeneChatDataModule(config)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 创建数据模块
    data_module = create_data_module()

    # 准备数据
    print("准备数据...")
    data_module.prepare_data()

    # 设置数据集
    print("\n设置数据集...")
    data_module.setup_datasets()

    # 获取统计信息
    print("\n数据集统计信息:")
    stats = data_module.get_statistics()
    for split, split_stats in stats.items():
        print(f"\n{split}集:")
        for key, value in split_stats.items():
            print(f"  {key}: {value}")

    # 测试DataLoader
    print("\n测试DataLoader...")
    train_loader = data_module.get_dataloader("train", batch_size=2)

    for i, batch in enumerate(train_loader):
        print(f"\nBatch {i}:")
        print(f"  DNA序列数量: {len(batch['dna_sequences'])}")
        print(f"  第一个序列长度: {len(batch['dna_sequences'][0])}")
        print(f"  提示: {batch['prompts'][0]}")
        print(f"  目标描述: {batch['target_descriptions'][0][:100]}...")
        if i >= 2:  # 只测试3个batch
            break
