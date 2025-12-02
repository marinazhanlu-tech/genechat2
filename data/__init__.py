"""
GeneChat2数据模块

导出数据处理和数据集构建组件
"""

from .ncbi_processor import (
    NCBIGeneProcessor,
    NCBIConfig,
    GeneRecord,
    GeneTriplet,
    create_ncbi_processor
)
from .dataset_builder import (
    GeneChatDataset,
    GeneChatDataModule,
    DatasetConfig,
    create_data_module
)

__all__ = [
    # NCBI处理器
    "NCBIGeneProcessor",
    "NCBIConfig",
    "GeneRecord",
    "GeneTriplet",
    "create_ncbi_processor",
    # 数据集
    "GeneChatDataset",
    "GeneChatDataModule",
    "DatasetConfig",
    "create_data_module",
]
