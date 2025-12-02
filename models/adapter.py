"""
基因到文本的适配器模块

这是GeneChat论文的核心创新：将DNABERT2的768维基因嵌入
映射到Vicuna-13B的5120维嵌入空间，实现基因序列到自然语言的转换。

关键特性：
- 线性投影适配器
- 保持生物学信息
- 与语言模型嵌入空间对齐
- 端到端可训练
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AdapterConfig:
    """适配器配置（论文规格）"""
    input_dim: int = 256       # DNABERT2 pooling后的输出维度（论文要求）
    output_dim: int = 5120     # Vicuna-13B输入维度
    adapter_type: str = "linear"  # 线性适配器（论文使用）
    use_layer_norm: bool = True   # 使用层归一化
    use_dropout: bool = True       # 使用Dropout
    dropout_rate: float = 0.1     # Dropout率
    freeze_adapter: bool = False    # 冻结适配器参数（训练时设为False）
    initialization: str = "xavier"  # 权重初始化方法

class GeneToTextAdapter(nn.Module):
    """基因序列到文本的适配器（论文核心创新）"""

    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config

        logger.info(f"初始化基因-文本适配器: {config.input_dim} -> {config.output_dim}")

        # 核心适配器层（线性投影）
        if config.adapter_type == "linear":
            self.adapter = nn.Linear(config.input_dim, config.output_dim, bias=False)
        elif config.adapter_type == "mlp":
            # 多层感知机适配器
            hidden_dim = min(config.input_dim * 4, config.output_dim // 2)
            self.adapter = nn.Sequential(
                nn.Linear(config.input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(hidden_dim, config.output_dim)
            )
        else:
            raise ValueError(f"不支持的适配器类型: {config.adapter_type}")

        # 层归一化（稳定训练）
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.output_dim)
        else:
            self.layer_norm = nn.Identity()

        # Dropout
        if config.use_dropout:
            self.dropout = nn.Dropout(config.dropout_rate)
        else:
            self.dropout = nn.Identity()

        # 初始化权重
        self._initialize_weights()

        # 冻结参数
        if config.freeze_adapter:
            self._freeze_adapter()

        logger.info(f"适配器初始化完成，参数数量: {self.num_parameters():,}")

    def _initialize_weights(self):
        """初始化适配器权重"""
        if self.config.initialization == "xavier":
            if isinstance(self.adapter, nn.Linear):
                nn.init.xavier_uniform_(self.adapter.weight)
            elif isinstance(self.adapter, nn.Sequential):
                for module in self.adapter:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

        elif self.config.initialization == "kaiming":
            if isinstance(self.adapter, nn.Linear):
                nn.init.kaiming_uniform_(self.adapter.weight)
            elif isinstance(self.adapter, nn.Sequential):
                for module in self.adapter:
                    if isinstance(module, nn.Linear):
                        nn.init.kaiming_uniform_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

        elif self.config.initialization == "normal":
            if isinstance(self.adapter, nn.Linear):
                nn.init.normal_(self.adapter.weight, mean=0.0, std=0.02)
            elif isinstance(self.adapter, nn.Sequential):
                for module in self.adapter:
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(module.weight, mean=0.0, std=0.02)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

    def _freeze_adapter(self):
        """冻结适配器参数"""
        for name, param in self.named_parameters():
            param.requires_grad = False
        logger.info("适配器参数已冻结")

    def forward(self, gene_embeddings: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播：将基因嵌入映射到语言模型空间（论文方法）

        Args:
            gene_embeddings: [batch_size, num_windows, 256] 基因嵌入（pooling后）
            attention_mask: [batch_size, num_windows] 注意力掩码

        Returns:
            llm_embeddings: [batch_size, num_windows, 5120] 语言模型嵌入
        """
        # 验证输入维度
        if gene_embeddings.dim() != 3:
            raise ValueError(f"期望3维输入 [batch, windows, features]，得到 {gene_embeddings.dim()}维")

        if gene_embeddings.size(-1) != self.config.input_dim:
            raise ValueError(f"输入特征维度不匹配: 期望{self.config.input_dim}，得到{gene_embeddings.size(-1)}")

        batch_size, num_windows, _ = gene_embeddings.shape

        # 重塑为2D进行处理
        gene_embeddings_2d = gene_embeddings.view(-1, self.config.input_dim)  # [batch*windows, 768]

        # 应用适配器
        llm_embeddings_2d = self.adapter(gene_embeddings_2d)  # [batch*windows, 5120]

        # 应用层归一化和Dropout
        llm_embeddings_2d = self.layer_norm(llm_embeddings_2d)
        llm_embeddings_2d = self.dropout(llm_embeddings_2d)

        # 重塑回3D
        llm_embeddings = llm_embeddings_2d.view(batch_size, num_windows, self.config.output_dim)

        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            # 扩展注意力掩码到输出维度
            attention_mask_3d = attention_mask.unsqueeze(-1).expand(-1, -1, self.config.output_dim)
            llm_embeddings = llm_embeddings * attention_mask_3d

        return llm_embeddings

    def encode_single_gene(self, gene_embedding: torch.Tensor) -> torch.Tensor:
        """
        编码单个基因嵌入

        Args:
            gene_embedding: [num_windows, 256] 单个基因嵌入（pooling后）

        Returns:
            llm_embedding: [num_windows, 5120] 语言模型嵌入
        """
        if gene_embedding.dim() != 2:
            raise ValueError(f"期望2维输入 [windows, features]，得到 {gene_embedding.dim()}维")

        return self.forward(gene_embedding.unsqueeze(0)).squeeze(0)

    def get_embedding_statistics(self) -> Dict[str, Any]:
        """获取适配器的统计信息"""
        total_params = self.num_parameters()
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # 计算适配器的权重统计
        adapter_weight = None
        if isinstance(self.adapter, nn.Linear):
            adapter_weight = self.adapter.weight
        elif isinstance(self.adapter, nn.Sequential):
            for module in self.adapter:
                if isinstance(module, nn.Linear):
                    adapter_weight = module.weight
                    break

        weight_stats = {}
        if adapter_weight is not None:
            weight_stats = {
                "weight_mean": adapter_weight.mean().item(),
                "weight_std": adapter_weight.std().item(),
                "weight_max": adapter_weight.max().item(),
                "weight_min": adapter_weight.min().item(),
                "weight_norm": adapter_weight.norm().item()
            }

        return {
            "adapter_type": self.config.adapter_type,
            "input_dim": self.config.input_dim,
            "output_dim": self.config.output_dim,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "use_layer_norm": self.config.use_layer_norm,
            "use_dropout": self.config.use_dropout,
            "dropout_rate": self.config.dropout_rate,
            "weight_statistics": weight_stats
        }

    def num_parameters(self) -> int:
        """获取适配器参数数量"""
        return sum(p.numel() for p in self.parameters())

    def save_adapter(self, save_path: str):
        """保存适配器"""
        try:
            # 保存状态字典和配置
            torch.save({
                'adapter_state_dict': self.state_dict(),
                'config': self.config
            }, save_path)
            logger.info(f"适配器已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存适配器失败: {e}")
            raise

    def load_adapter(self, load_path: str, strict: bool = True):
        """加载适配器"""
        try:
            checkpoint = torch.load(load_path, map_location='cpu')
            self.load_state_dict(checkpoint['adapter_state_dict'], strict=strict)
            logger.info(f"适配器已从 {load_path} 加载")
        except Exception as e:
            logger.error(f"加载适配器失败: {e}")
            raise

    def reset_parameters(self):
        """重置适配器参数"""
        self._initialize_weights()
        logger.info("适配器参数已重置")

class MultiScaleAdapter(nn.Module):
    """多尺度适配器：处理不同长度的基因序列"""

    def __init__(self, config: AdapterConfig, scales: list = [1, 2, 4]):
        super().__init__()
        self.config = config
        self.scales = scales

        # 为每个尺度创建适配器
        self.adapters = nn.ModuleDict()
        for scale in scales:
            scaled_input_dim = config.input_dim // scale
            self.adapters[str(scale)] = GeneToTextAdapter(
                AdapterConfig(
                    input_dim=scaled_input_dim,
                    output_dim=config.output_dim // len(scales),
                    adapter_type=config.adapter_type,
                    use_layer_norm=config.use_layer_norm,
                    use_dropout=config.use_dropout,
                    dropout_rate=config.dropout_rate,
                    freeze_adapter=config.freeze_adapter,
                    initialization=config.initialization
                )
            )

        logger.info(f"多尺度适配器初始化完成，尺度: {scales}")

    def forward(self, gene_embeddings: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        多尺度适配器前向传播

        Args:
            gene_embeddings: [batch_size, num_windows, 768] 基因嵌入

        Returns:
            llm_embeddings: [batch_size, num_windows, 5120] 语言模型嵌入
        """
        batch_size, num_windows, _ = gene_embeddings.shape
        scale_outputs = []

        # 为每个尺度处理
        for scale, adapter in self.adapters.items():
            scale_int = int(scale)
            scale_dim = self.config.input_dim // scale_int

            # 对基因嵌入进行下采样
            if scale_int > 1:
                # 使用平均池化进行下采样
                pooled_embeddings = gene_embeddings.view(
                    batch_size, num_windows // scale_int, scale_int, scale_dim
                ).mean(dim=2)  # [batch, windows/scale, dim/scale]
            else:
                pooled_embeddings = gene_embeddings

            # 通过适配器
            scale_output = adapter(pooled_embeddings)

            # 上采样回原始长度
            if scale_int > 1:
                scale_output = scale_output.repeat_interleave(scale_int, dim=1)
                if scale_output.size(1) > num_windows:
                    scale_output = scale_output[:, :num_windows, :]

            scale_outputs.append(scale_output)

        # 拼接所有尺度的输出
        llm_embeddings = torch.cat(scale_outputs, dim=-1)

        return llm_embeddings

# 便捷函数
def create_gene_to_text_adapter(input_dim: int = 256,
                               output_dim: int = 5120,
                               adapter_type: str = "linear",
                               use_layer_norm: bool = True,
                               dropout_rate: float = 0.1,
                               freeze_adapter: bool = False) -> GeneToTextAdapter:
    """创建基因-文本适配器的便捷函数（论文配置：256→5120）"""
    config = AdapterConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        adapter_type=adapter_type,
        use_layer_norm=use_layer_norm,
        dropout_rate=dropout_rate,
        freeze_adapter=freeze_adapter
    )
    return GeneToTextAdapter(config)

def create_multiscale_adapter(input_dim: int = 256,
                             output_dim: int = 5120,
                             scales: list = [1, 2, 4],
                             dropout_rate: float = 0.1) -> MultiScaleAdapter:
    """创建多尺度适配器的便捷函数（论文配置：256→5120）"""
    config = AdapterConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        use_layer_norm=True,
        use_dropout=True,
        dropout_rate=dropout_rate
    )
    return MultiScaleAdapter(config, scales)

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 创建适配器
    adapter = create_gene_to_text_adapter()

    # 测试输入：假设有312个窗口（160kb/512bp），256维嵌入
    batch_size = 2
    num_windows = 312
    input_dim = 256  # 论文要求的pooling后维度

    gene_embeddings = torch.randn(batch_size, num_windows, input_dim)
    attention_mask = torch.ones(batch_size, num_windows)

    # 前向传播测试
    with torch.no_grad():
        llm_embeddings = adapter(gene_embeddings, attention_mask)

    print(f"适配器信息: {adapter.get_embedding_statistics()}")
    print(f"输入形状: {gene_embeddings.shape}")
    print(f"输出形状: {llm_embeddings.shape}")

    # 测试多尺度适配器
    print("\n测试多尺度适配器...")
    multi_adapter = create_multiscale_adapter()

    with torch.no_grad():
        multi_llm_embeddings = multi_adapter(gene_embeddings, attention_mask)

    print(f"多尺度适配器输出形状: {multi_llm_embeddings.shape}")