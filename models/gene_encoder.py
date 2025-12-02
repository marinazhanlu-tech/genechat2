"""
DNABERT2基因编码器模块

基于论文中的基因编码器实现，专门处理DNA序列。
使用DNABERT-2模型将DNA序列编码为生物医学嵌入表示。

关键特性：
- 支持160kb超长DNA序列
- 512碱基对滑动窗口
- 768维基因表示
- 与论文完全一致的实现
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GeneEncodingConfig:
    """基因编码器配置"""
    model_path: str = "zhihan1996/DNABERT-2-117M"
    max_sequence_length: int = 160000  # 160kb DNA序列
    window_size: int = 512  # 滑动窗口大小（论文中的chunk size）
    window_overlap: int = 10  # 10bp重叠（论文要求）
    embedding_dim: int = 768  # DNABERT2原始输出维度
    pooled_dim: int = 256  # Pooling后的维度（论文要求）
    freeze_encoder: bool = True
    device: str = "cuda"
    trust_remote_code: bool = True

class DNASequencer:
    """DNA序列处理工具"""

    def __init__(self, max_length: int = 160000, window_size: int = 512):
        self.max_length = max_length
        self.window_size = window_size
        self.valid_bases = set(['A', 'C', 'G', 'T', 'N'])

    def validate_dna_sequence(self, sequence: str) -> bool:
        """验证DNA序列的有效性"""
        if not sequence:
            return False
        return all(base.upper() in self.valid_bases for base in sequence)

    def clean_dna_sequence(self, sequence: str) -> str:
        """清理DNA序列，移除无效字符"""
        return ''.join(base.upper() if base.upper() in self.valid_bases else 'N'
                      for base in sequence)

    def truncate_sequence(self, sequence: str) -> str:
        """截断到最大长度"""
        if len(sequence) <= self.max_length:
            return sequence
        logger.warning(f"DNA序列长度{len(sequence)}超过最大长度{self.max_length}，进行截断")
        return sequence[:self.max_length]

    def create_sliding_windows(self, sequence: str, overlap: int = 10) -> List[str]:
        """创建滑动窗口（论文要求10bp重叠）

        Args:
            sequence: DNA序列
            overlap: 窗口间重叠的碱基对数量（论文中为10bp）

        Returns:
            windows: 滑动窗口列表
        """
        windows = []
        sequence = self.clean_dna_sequence(sequence)
        sequence = self.truncate_sequence(sequence)

        # 计算步长：窗口大小 - 重叠大小
        stride = self.window_size - overlap

        for i in range(0, len(sequence), stride):
            window = sequence[i:i + self.window_size]
            if len(window) < self.window_size:
                # 填充到窗口大小
                window = window + 'N' * (self.window_size - len(window))
            windows.append(window)

            # 如果已经到达或超过序列末尾，停止
            if i + self.window_size >= len(sequence):
                break

        logger.info(f"DNA序列长度: {len(sequence)}, 窗口数量: {len(windows)}, 步长: {stride}bp (重叠: {overlap}bp)")
        return windows

class DNABERT2Encoder:
    """DNABERT2基因编码器实现"""

    def __init__(self, config: GeneEncodingConfig):
        self.config = config
        self.dna_sequencer = DNASequencer(
            max_length=config.max_sequence_length,
            window_size=config.window_size
        )

        # 初始化DNABERT2模型和分词器
        logger.info(f"加载DNABERT2模型: {config.model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_path,
                trust_remote_code=config.trust_remote_code,
                use_fast=False
            )

            self.model = AutoModel.from_pretrained(
                config.model_path,
                trust_remote_code=config.trust_remote_code,
                torch_dtype=torch.float16,
                device_map=config.device
            )

            logger.info(f"DNABERT2模型加载成功，参数数量: {self.model.num_parameters():,}")

        except Exception as e:
            logger.error(f"加载DNABERT2模型失败: {e}")
            raise

        # 添加Pooling层：768→256（论文要求）
        if config.pooled_dim != config.embedding_dim:
            self.pooling_layer = nn.Linear(config.embedding_dim, config.pooled_dim).to(config.device)
            logger.info(f"添加Pooling层: {config.embedding_dim}→{config.pooled_dim}")
        else:
            self.pooling_layer = None

        # 冻结参数
        if config.freeze_encoder:
            self._freeze_encoder()

        self.model.eval()

    def _freeze_encoder(self):
        """冻结编码器参数"""
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        logger.info("DNABERT2编码器参数已冻结")

    def encode_single_window(self, window_sequence: str) -> torch.Tensor:
        """编码单个DNA窗口（论文方法：DNABERT2 + 平均池化 + 降维到256）"""
        # 分词
        inputs = self.tokenizer(
            window_sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.window_size
        )

        # 移动到设备
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        # 编码
        with torch.no_grad():
            outputs = self.model(**inputs)

            # 获取最后一层隐藏状态
            last_hidden_states = outputs.last_hidden_state  # [1, seq_len, 768]

            # 取平均池化（论文中的pooling operation）
            embedding = torch.mean(last_hidden_states, dim=1)  # [1, 768]

            # 降维到256（论文要求）
            if self.pooling_layer is not None:
                embedding = self.pooling_layer(embedding)  # [1, 256]

        return embedding.squeeze(0)  # [256]

    def encode_gene_sequence(self, dna_sequence: str) -> torch.Tensor:
        """编码完整的DNA序列（论文方法）

        Args:
            dna_sequence: DNA序列字符串

        Returns:
            gene_embeddings: [num_windows, 256] 基因嵌入矩阵（已降维）
        """
        # 验证和清理序列
        if not self.dna_sequencer.validate_dna_sequence(dna_sequence):
            logger.warning("DNA序列包含无效字符，将进行清理")

        # 创建滑动窗口（10bp重叠）
        windows = self.dna_sequencer.create_sliding_windows(
            dna_sequence,
            overlap=self.config.window_overlap
        )

        if not windows:
            raise ValueError("没有有效的DNA窗口")

        # 编码所有窗口
        gene_embeddings = []

        for i, window in enumerate(windows):
            try:
                window_embedding = self.encode_single_window(window)
                gene_embeddings.append(window_embedding)

                if i % 50 == 0:  # 每50个窗口记录一次
                    logger.debug(f"已编码窗口 {i+1}/{len(windows)}")

            except Exception as e:
                logger.error(f"编码窗口 {i} 失败: {e}")
                # 使用零向量作为fallback
                zero_embedding = torch.zeros(
                    self.config.pooled_dim,
                    device=self.config.device
                )
                gene_embeddings.append(zero_embedding)

        # 堆叠为矩阵
        gene_embeddings = torch.stack(gene_embeddings, dim=0)  # [num_windows, 256]

        logger.info(f"基因序列编码完成: {gene_embeddings.shape}")
        return gene_embeddings

    def encode_batch(self, dna_sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量编码DNA序列（论文方法）

        Args:
            dna_sequences: DNA序列列表

        Returns:
            batch_embeddings: [batch_size, num_windows, 256]
            attention_masks: [batch_size, num_windows]
        """
        batch_embeddings = []

        for i, sequence in enumerate(dna_sequences):
            try:
                embeddings = self.encode_gene_sequence(sequence)
                batch_embeddings.append(embeddings)

            except Exception as e:
                logger.error(f"编码序列 {i} 失败: {e}")
                # 使用零矩阵作为fallback
                num_windows = (len(sequence) + self.config.window_size - 1) // self.config.window_size
                zero_embeddings = torch.zeros(
                    num_windows, self.config.pooled_dim,
                    device=self.config.device
                )
                batch_embeddings.append(zero_embeddings)

        # 处理变长序列（padding到最大长度）
        max_windows = max(emb.shape[0] for emb in batch_embeddings)

        padded_embeddings = []
        attention_masks = []

        for emb in batch_embeddings:
            current_windows = emb.shape[0]

            # 填充到最大窗口数
            if current_windows < max_windows:
                padding = torch.zeros(
                    max_windows - current_windows,
                    self.config.pooled_dim,
                    device=self.config.device
                )
                padded_emb = torch.cat([emb, padding], dim=0)
                attention_mask = torch.cat([
                    torch.ones(current_windows, device=self.config.device),
                    torch.zeros(max_windows - current_windows, device=self.config.device)
                ], dim=0)
            else:
                padded_emb = emb
                attention_mask = torch.ones(max_windows, device=self.config.device)

            padded_embeddings.append(padded_emb)
            attention_masks.append(attention_mask)

        batch_embeddings = torch.stack(padded_embeddings, dim=0)  # [batch_size, max_windows, 256]
        attention_masks = torch.stack(attention_masks, dim=0)  # [batch_size, max_windows]

        return batch_embeddings, attention_masks

    def get_embedding_info(self) -> Dict:
        """获取编码器信息"""
        return {
            "model_name": "DNABERT2",
            "model_path": self.config.model_path,
            "max_sequence_length": self.config.max_sequence_length,
            "window_size": self.config.window_size,
            "window_overlap": self.config.window_overlap,
            "embedding_dim": self.config.embedding_dim,
            "pooled_dim": self.config.pooled_dim,
            "num_parameters": self.model.num_parameters(),
            "frozen": self.config.freeze_encoder,
            "device": self.config.device
        }

    def save_encoder(self, save_path: str):
        """保存编码器"""
        try:
            self.tokenizer.save_pretrained(save_path)
            self.model.save_pretrained(save_path)
            logger.info(f"DNABERT2编码器已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存编码器失败: {e}")
            raise

    def load_encoder(self, load_path: str):
        """加载编码器"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            self.model = AutoModel.from_pretrained(
                load_path,
                torch_dtype=torch.float16,
                device_map=self.config.device
            )

            if self.config.freeze_encoder:
                self._freeze_encoder()

            logger.info(f"DNABERT2编码器已从 {load_path} 加载")
        except Exception as e:
            logger.error(f"加载编码器失败: {e}")
            raise

# 便捷函数
def create_dna_encoder(model_path: str = "zhihan1996/DNABERT-2-117M",
                      max_sequence_length: int = 160000,
                      window_size: int = 512,
                      window_overlap: int = 10,
                      pooled_dim: int = 256,
                      freeze_encoder: bool = True,
                      device: str = "cuda") -> DNABERT2Encoder:
    """创建DNA编码器的便捷函数（论文配置）"""
    config = GeneEncodingConfig(
        model_path=model_path,
        max_sequence_length=max_sequence_length,
        window_size=window_size,
        window_overlap=window_overlap,
        pooled_dim=pooled_dim,
        freeze_encoder=freeze_encoder,
        device=device
    )
    return DNABERT2Encoder(config)

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 创建编码器
    encoder = create_dna_encoder()

    # 测试序列
    test_sequence = "ATCGATCGATCG" * 1000  # 10kb测试序列

    # 编码测试
    embeddings, attention_mask = encoder.encode_batch([test_sequence])

    print(f"编码器信息: {encoder.get_embedding_info()}")
    print(f"测试序列嵌入形状: {embeddings.shape}")
    print(f"注意力掩码形状: {attention_mask.shape}")