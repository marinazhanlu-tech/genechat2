"""
Vicuna-13B LLM包装器

封装Vicuna-13B语言模型，提供LoRA微调接口。
实现论文中的语言模型组件。

关键特性：
- Vicuna-13B-v1.5模型
- LoRA参数高效微调
- 生成基因功能描述
- 符合论文规格
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VicunaConfig:
    """Vicuna-13B配置（论文规格）"""
    model_path: str = "lmsys/vicuna-13b-v1.5"
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None  # ["q_proj", "v_proj"]
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    num_beams: int = 4
    repetition_penalty: float = 1.2
    device: str = "cuda"
    torch_dtype: str = "float16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj"]


class VicunaLLM:
    """Vicuna-13B LLM包装器（论文实现）"""

    def __init__(self, config: VicunaConfig):
        self.config = config

        logger.info(f"加载Vicuna-13B模型: {config.model_path}")

        # 设置torch dtype
        if config.torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif config.torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        # 加载tokenizer
        self.tokenizer = self._load_tokenizer()

        # 加载模型
        self.model = self._load_model()

        # 应用LoRA
        if config.use_lora:
            self.model = self._apply_lora()

        # 设置生成配置
        self.generation_config = self._setup_generation_config()

        logger.info("Vicuna-13B模型加载完成")

    def _load_tokenizer(self) -> AutoTokenizer:
        """加载分词器"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                use_fast=False
            )

            # 设置特殊token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            logger.info(f"分词器加载成功，词汇表大小: {len(tokenizer)}")
            return tokenizer

        except Exception as e:
            logger.error(f"加载分词器失败: {e}")
            raise

    def _load_model(self) -> AutoModelForCausalLM:
        """加载Vicuna模型"""
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.config.device,
                load_in_8bit=self.config.load_in_8bit,
                load_in_4bit=self.config.load_in_4bit,
                trust_remote_code=True
            )

            # 启用梯度检查点（节省内存）
            model.gradient_checkpointing_enable()

            logger.info(f"Vicuna模型加载成功，参数数量: {model.num_parameters():,}")
            return model

        except Exception as e:
            logger.error(f"加载Vicuna模型失败: {e}")
            raise

    def _apply_lora(self) -> PeftModel:
        """应用LoRA微调（论文配置）"""
        logger.info("应用LoRA配置...")

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = get_peft_model(self.model, lora_config)

        # 打印可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        logger.info(f"LoRA应用成功")
        logger.info(f"可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        logger.info(f"总参数: {total_params:,}")

        return model

    def _setup_generation_config(self) -> GenerationConfig:
        """设置生成配置"""
        generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            num_beams=self.config.num_beams,
            repetition_penalty=self.config.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        logger.info(f"生成配置: max_tokens={self.config.max_new_tokens}, temp={self.config.temperature}")
        return generation_config

    def get_input_embeddings(self) -> nn.Module:
        """获取输入嵌入层"""
        return self.model.get_input_embeddings()

    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        return self.model.config.hidden_size

    def prepare_inputs(
        self,
        prompt: str,
        gene_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """准备模型输入（论文格式）

        论文中的格式：
        Human: <Gene> GeneHere </Gene>Prompt Assistant:

        Args:
            prompt: 文本提示
            gene_embeddings: [num_windows, 5120] 基因嵌入（已通过适配器）

        Returns:
            inputs: 模型输入字典
        """
        # 构建提示（论文格式）
        if gene_embeddings is not None:
            # 带基因嵌入的提示
            formatted_prompt = f"Human: <Gene> GeneHere </Gene>{prompt} Assistant:"
        else:
            # 纯文本提示
            formatted_prompt = f"Human: {prompt} Assistant:"

        # 分词
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # 移动到设备
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        return inputs

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播

        Args:
            inputs_embeds: [batch_size, seq_len, 5120] 输入嵌入
            attention_mask: [batch_size, seq_len] 注意力掩码
            labels: [batch_size, seq_len] 标签（训练时使用）

        Returns:
            outputs: 模型输出
        """
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None
        }

    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        **generation_kwargs
    ) -> List[str]:
        """生成文本（推理时使用）

        Args:
            inputs_embeds: [batch_size, seq_len, 5120] 输入嵌入
            attention_mask: [batch_size, seq_len] 注意力掩码
            max_new_tokens: 最大生成token数
            **generation_kwargs: 其他生成参数

        Returns:
            generated_texts: 生成的文本列表
        """
        # 更新生成配置
        gen_config = self.generation_config
        if max_new_tokens is not None:
            gen_config.max_new_tokens = max_new_tokens

        # 合并生成参数
        for k, v in generation_kwargs.items():
            setattr(gen_config, k, v)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                generation_config=gen_config,
                return_dict_in_generate=True
            )

        # 解码
        generated_texts = self.tokenizer.batch_decode(
            outputs.sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # 提取Assistant之后的内容
        processed_texts = []
        for text in generated_texts:
            if "Assistant:" in text:
                text = text.split("Assistant:")[-1].strip()
            processed_texts.append(text)

        return processed_texts

    def save_model(self, save_path: str):
        """保存模型"""
        try:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"模型已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            raise

    def load_model(self, load_path: str):
        """加载模型"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)

            if self.config.use_lora:
                # 加载LoRA权重
                self.model = PeftModel.from_pretrained(
                    self.model,
                    load_path,
                    torch_dtype=self.torch_dtype
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    load_path,
                    torch_dtype=self.torch_dtype,
                    device_map=self.config.device
                )

            logger.info(f"模型已从 {load_path} 加载")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        return {
            "model_name": "Vicuna-13B-v1.5",
            "model_path": self.config.model_path,
            "use_lora": self.config.use_lora,
            "lora_r": self.config.lora_r if self.config.use_lora else None,
            "lora_alpha": self.config.lora_alpha if self.config.use_lora else None,
            "lora_target_modules": self.config.lora_target_modules if self.config.use_lora else None,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params,
            "embedding_dim": self.get_embedding_dim(),
            "max_new_tokens": self.config.max_new_tokens,
            "device": self.config.device
        }


def create_vicuna_llm(
    model_path: str = "lmsys/vicuna-13b-v1.5",
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    max_new_tokens: int = 256,
    device: str = "cuda"
) -> VicunaLLM:
    """创建Vicuna LLM的便捷函数（论文配置）"""
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj"]

    config = VicunaConfig(
        model_path=model_path,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        max_new_tokens=max_new_tokens,
        device=device
    )
    return VicunaLLM(config)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 创建LLM
    llm = create_vicuna_llm()

    # 打印模型信息
    info = llm.get_model_info()
    for k, v in info.items():
        print(f"{k}: {v}")

    # 测试输入准备
    prompt = "please predict the function of this gene"
    inputs = llm.prepare_inputs(prompt)
    print(f"\n输入形状: {inputs['input_ids'].shape}")
    print(f"嵌入维度: {llm.get_embedding_dim()}")
