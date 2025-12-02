"""
GeneChat2主模型

集成DNABERT2编码器、适配器和Vicuna-13B LLM的完整多模态模型。
实现论文中的端到端架构。

关键特性：
- 端到端多模态架构
- DNA序列→基因功能描述
- 符合论文的完整实现
- 支持训练和推理
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

from .gene_encoder import DNABERT2Encoder, GeneEncodingConfig
from .adapter import GeneToTextAdapter, AdapterConfig
from .llm_wrapper import VicunaLLM, VicunaConfig

logger = logging.getLogger(__name__)

@dataclass
class GeneChat2Config:
    """GeneChat2模型配置（论文完整规格）"""
    # 基因编码器配置
    gene_encoder_config: GeneEncodingConfig = None
    # 适配器配置
    adapter_config: AdapterConfig = None
    # LLM配置
    llm_config: VicunaConfig = None
    # 训练配置
    freeze_gene_encoder: bool = True
    freeze_adapter: bool = False
    freeze_llm: bool = False  # 使用LoRA，所以不冻结
    device: str = "cuda"

    def __post_init__(self):
        if self.gene_encoder_config is None:
            self.gene_encoder_config = GeneEncodingConfig(
                window_size=512,
                window_overlap=10,
                pooled_dim=256,
                freeze_encoder=self.freeze_gene_encoder,
                device=self.device
            )
        if self.adapter_config is None:
            self.adapter_config = AdapterConfig(
                input_dim=256,
                output_dim=5120,
                freeze_adapter=self.freeze_adapter
            )
        if self.llm_config is None:
            self.llm_config = VicunaConfig(
                use_lora=True,
                lora_r=8,
                lora_alpha=16,
                lora_target_modules=["q_proj", "v_proj"],
                device=self.device
            )


class GeneChat2(nn.Module):
    """GeneChat2多模态大语言模型（论文实现）

    架构：
    DNA序列 -> DNABERT2编码器 -> 适配器 -> Vicuna-13B -> 功能描述
    """

    def __init__(self, config: GeneChat2Config):
        super().__init__()
        self.config = config

        logger.info("初始化GeneChat2模型...")

        # 1. 基因编码器（DNABERT2）
        logger.info("加载基因编码器...")
        self.gene_encoder = DNABERT2Encoder(config.gene_encoder_config)

        # 2. 适配器（256→5120）
        logger.info("初始化适配器...")
        self.adapter = GeneToTextAdapter(config.adapter_config)

        # 3. 语言模型（Vicuna-13B）
        logger.info("加载语言模型...")
        self.llm = VicunaLLM(config.llm_config)

        # 验证维度匹配
        self._validate_dimensions()

        logger.info("GeneChat2模型初始化完成")
        self._log_model_info()

    def _validate_dimensions(self):
        """验证各组件维度匹配"""
        # 编码器输出 -> 适配器输入
        encoder_output_dim = self.config.gene_encoder_config.pooled_dim
        adapter_input_dim = self.config.adapter_config.input_dim
        if encoder_output_dim != adapter_input_dim:
            raise ValueError(
                f"编码器输出维度({encoder_output_dim})与适配器输入维度({adapter_input_dim})不匹配"
            )

        # 适配器输出 -> LLM输入
        adapter_output_dim = self.config.adapter_config.output_dim
        llm_embedding_dim = self.llm.get_embedding_dim()
        if adapter_output_dim != llm_embedding_dim:
            raise ValueError(
                f"适配器输出维度({adapter_output_dim})与LLM嵌入维度({llm_embedding_dim})不匹配"
            )

        logger.info("✓ 所有维度验证通过")

    def _log_model_info(self):
        """记录模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"=" * 50)
        logger.info(f"GeneChat2模型信息")
        logger.info(f"=" * 50)
        logger.info(f"总参数: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        logger.info(f"基因编码器: DNABERT2 (冻结={self.config.freeze_gene_encoder})")
        logger.info(f"适配器: Linear 256→5120 (冻结={self.config.freeze_adapter})")
        logger.info(f"语言模型: Vicuna-13B (LoRA微调)")
        logger.info(f"=" * 50)

    def encode_gene_sequence(
        self,
        dna_sequence: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码DNA序列

        Args:
            dna_sequence: DNA序列字符串

        Returns:
            gene_embeddings: [num_windows, 256] 基因嵌入
        """
        with torch.no_grad() if self.config.freeze_gene_encoder else torch.enable_grad():
            gene_embeddings = self.gene_encoder.encode_gene_sequence(dna_sequence)
        return gene_embeddings

    def adapt_gene_embeddings(
        self,
        gene_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """将基因嵌入适配到LLM空间

        Args:
            gene_embeddings: [batch_size, num_windows, 256] 基因嵌入
            attention_mask: [batch_size, num_windows] 注意力掩码

        Returns:
            llm_embeddings: [batch_size, num_windows, 5120] LLM嵌入
        """
        llm_embeddings = self.adapter(gene_embeddings, attention_mask)
        return llm_embeddings

    def prepare_llm_inputs(
        self,
        prompt: str,
        gene_llm_embeddings: torch.Tensor,
        target_text: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """准备LLM输入（论文格式）

        论文格式:
        Human: <Gene> GeneHere </Gene>Prompt Assistant: Answer

        Args:
            prompt: 文本提示（如"please predict the function of this gene"）
            gene_llm_embeddings: [num_windows, 5120] 基因嵌入（已适配）
            target_text: 目标文本（训练时使用）

        Returns:
            inputs: LLM输入字典
        """
        # 构建辅助提示
        aux_prompt_prefix = "Human: <Gene> "
        aux_prompt_suffix = f" </Gene>{prompt} Assistant:"

        # 分词辅助提示
        prefix_tokens = self.llm.tokenizer(
            aux_prompt_prefix,
            return_tensors="pt",
            add_special_tokens=True
        )
        suffix_tokens = self.llm.tokenizer(
            aux_prompt_suffix,
            return_tensors="pt",
            add_special_tokens=False
        )

        # 获取辅助提示的嵌入
        prefix_embeds = self.llm.get_input_embeddings()(
            prefix_tokens["input_ids"].to(self.config.device)
        )  # [1, prefix_len, 5120]
        suffix_embeds = self.llm.get_input_embeddings()(
            suffix_tokens["input_ids"].to(self.config.device)
        )  # [1, suffix_len, 5120]

        # 拼接嵌入：prefix + gene + suffix
        gene_llm_embeddings = gene_llm_embeddings.unsqueeze(0)  # [1, num_windows, 5120]
        inputs_embeds = torch.cat([
            prefix_embeds,
            gene_llm_embeddings,
            suffix_embeds
        ], dim=1)  # [1, total_len, 5120]

        # 构建注意力掩码
        prefix_mask = torch.ones(1, prefix_embeds.size(1), device=self.config.device)
        gene_mask = torch.ones(1, gene_llm_embeddings.size(1), device=self.config.device)
        suffix_mask = torch.ones(1, suffix_embeds.size(1), device=self.config.device)
        attention_mask = torch.cat([prefix_mask, gene_mask, suffix_mask], dim=1)

        # 处理标签（训练时）
        labels = None
        if target_text is not None:
            # 分词目标文本
            target_tokens = self.llm.tokenizer(
                target_text,
                return_tensors="pt",
                add_special_tokens=False
            )
            target_ids = target_tokens["input_ids"].to(self.config.device)

            # 构建标签：输入部分填充-100（不计算损失），输出部分为target_ids
            input_len = inputs_embeds.size(1)
            labels = torch.full(
                (1, input_len + target_ids.size(1)),
                fill_value=-100,
                dtype=torch.long,
                device=self.config.device
            )
            labels[:, input_len:] = target_ids

            # 添加目标嵌入到inputs_embeds
            target_embeds = self.llm.get_input_embeddings()(target_ids)
            inputs_embeds = torch.cat([inputs_embeds, target_embeds], dim=1)

            # 扩展attention_mask
            target_mask = torch.ones(1, target_ids.size(1), device=self.config.device)
            attention_mask = torch.cat([attention_mask, target_mask], dim=1)

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def forward(
        self,
        dna_sequence: str,
        prompt: str,
        target_text: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播（训练模式）

        Args:
            dna_sequence: DNA序列
            prompt: 文本提示
            target_text: 目标文本（训练时提供）

        Returns:
            outputs: 包含loss和logits的字典
        """
        # 1. 编码DNA序列
        gene_embeddings = self.encode_gene_sequence(dna_sequence)  # [num_windows, 256]

        # 2. 适配到LLM空间
        gene_embeddings = gene_embeddings.unsqueeze(0)  # [1, num_windows, 256]
        llm_embeddings = self.adapt_gene_embeddings(gene_embeddings)  # [1, num_windows, 5120]
        llm_embeddings = llm_embeddings.squeeze(0)  # [num_windows, 5120]

        # 3. 准备LLM输入
        llm_inputs = self.prepare_llm_inputs(prompt, llm_embeddings, target_text)

        # 4. LLM前向传播
        outputs = self.llm.forward(
            inputs_embeds=llm_inputs["inputs_embeds"],
            attention_mask=llm_inputs["attention_mask"],
            labels=llm_inputs["labels"]
        )

        return outputs

    def generate(
        self,
        dna_sequence: str,
        prompt: str = "please predict the function of this gene",
        max_new_tokens: int = 256,
        **generation_kwargs
    ) -> str:
        """生成基因功能描述（推理模式）

        Args:
            dna_sequence: DNA序列
            prompt: 文本提示
            max_new_tokens: 最大生成token数
            **generation_kwargs: 其他生成参数

        Returns:
            generated_text: 生成的功能描述
        """
        self.eval()

        with torch.no_grad():
            # 1. 编码DNA序列
            gene_embeddings = self.encode_gene_sequence(dna_sequence)

            # 2. 适配到LLM空间
            gene_embeddings = gene_embeddings.unsqueeze(0)  # [1, num_windows, 256]
            llm_embeddings = self.adapt_gene_embeddings(gene_embeddings)
            llm_embeddings = llm_embeddings.squeeze(0)  # [num_windows, 5120]

            # 3. 准备LLM输入（不包含target）
            llm_inputs = self.prepare_llm_inputs(prompt, llm_embeddings, target_text=None)

            # 4. 生成
            generated_texts = self.llm.generate(
                inputs_embeds=llm_inputs["inputs_embeds"],
                attention_mask=llm_inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                **generation_kwargs
            )

        return generated_texts[0]

    def save_model(self, save_path: str):
        """保存模型"""
        logger.info(f"保存GeneChat2模型到: {save_path}")

        # 保存各个组件
        import os
        os.makedirs(save_path, exist_ok=True)

        # 保存基因编码器
        self.gene_encoder.save_encoder(os.path.join(save_path, "gene_encoder"))

        # 保存适配器
        self.adapter.save_adapter(os.path.join(save_path, "adapter.pt"))

        # 保存LLM
        self.llm.save_model(os.path.join(save_path, "llm"))

        # 保存配置
        import json
        config_dict = {
            "freeze_gene_encoder": self.config.freeze_gene_encoder,
            "freeze_adapter": self.config.freeze_adapter,
            "freeze_llm": self.config.freeze_llm,
            "device": self.config.device
        }
        with open(os.path.join(save_path, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info("模型保存完成")

    def load_model(self, load_path: str):
        """加载模型"""
        logger.info(f"从 {load_path} 加载GeneChat2模型")

        import os

        # 加载基因编码器
        self.gene_encoder.load_encoder(os.path.join(load_path, "gene_encoder"))

        # 加载适配器
        self.adapter.load_adapter(os.path.join(load_path, "adapter.pt"))

        # 加载LLM
        self.llm.load_model(os.path.join(load_path, "llm"))

        logger.info("模型加载完成")

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": "GeneChat2",
            "architecture": "DNABERT2 + Adapter + Vicuna-13B",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params,
            "gene_encoder": self.gene_encoder.get_embedding_info(),
            "adapter": self.adapter.get_embedding_statistics(),
            "llm": self.llm.get_model_info(),
            "device": self.config.device
        }


def create_genechat2(
    gene_encoder_path: str = "zhihan1996/DNABERT-2-117M",
    llm_path: str = "lmsys/vicuna-13b-v1.5",
    freeze_gene_encoder: bool = True,
    freeze_adapter: bool = False,
    device: str = "cuda"
) -> GeneChat2:
    """创建GeneChat2模型的便捷函数（论文配置）"""
    config = GeneChat2Config(
        gene_encoder_config=GeneEncodingConfig(
            model_path=gene_encoder_path,
            window_size=512,
            window_overlap=10,
            pooled_dim=256,
            freeze_encoder=freeze_gene_encoder,
            device=device
        ),
        adapter_config=AdapterConfig(
            input_dim=256,
            output_dim=5120,
            freeze_adapter=freeze_adapter
        ),
        llm_config=VicunaConfig(
            model_path=llm_path,
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
            lora_target_modules=["q_proj", "v_proj"],
            device=device
        ),
        freeze_gene_encoder=freeze_gene_encoder,
        freeze_adapter=freeze_adapter,
        freeze_llm=False,
        device=device
    )
    return GeneChat2(config)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 创建模型
    print("创建GeneChat2模型...")
    model = create_genechat2()

    # 打印模型信息
    print("\n模型信息:")
    info = model.get_model_info()
    print(f"总参数: {info['total_parameters']:,}")
    print(f"可训练参数: {info['trainable_parameters']:,}")
    print(f"可训练比例: {info['trainable_percentage']:.2f}%")

    # 测试前向传播
    print("\n测试前向传播...")
    test_dna = "ATCGATCG" * 1000  # 8kb测试序列
    test_prompt = "please predict the function of this gene"
    test_target = "This gene encodes a protein involved in DNA repair."

    outputs = model.forward(test_dna, test_prompt, test_target)
    print(f"Loss: {outputs['loss']}")
    print(f"Logits shape: {outputs['logits'].shape}")

    # 测试生成
    print("\n测试生成...")
    generated_text = model.generate(test_dna, test_prompt, max_new_tokens=50)
    print(f"生成的文本: {generated_text}")
