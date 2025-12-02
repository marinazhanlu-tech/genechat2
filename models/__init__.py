"""
GeneChat2模型包

导出所有核心组件
"""

from .gene_encoder import DNABERT2Encoder, GeneEncodingConfig, create_dna_encoder
from .adapter import GeneToTextAdapter, AdapterConfig, create_gene_to_text_adapter
from .llm_wrapper import VicunaLLM, VicunaConfig, create_vicuna_llm
from .genechat2 import GeneChat2, GeneChat2Config, create_genechat2

__all__ = [
    # 基因编码器
    "DNABERT2Encoder",
    "GeneEncodingConfig",
    "create_dna_encoder",
    # 适配器
    "GeneToTextAdapter",
    "AdapterConfig",
    "create_gene_to_text_adapter",
    # LLM
    "VicunaLLM",
    "VicunaConfig",
    "create_vicuna_llm",
    # 主模型
    "GeneChat2",
    "GeneChat2Config",
    "create_genechat2",
]
