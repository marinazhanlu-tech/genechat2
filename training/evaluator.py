"""
GeneChat评估器

实现论文中使用的评估指标：BLEU和METEOR。

关键特性：
- BLEU-1, BLEU-2, BLEU-3, BLEU-4评分
- METEOR评分
- 批量评估
- 结果保存和可视化
"""

import torch
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass

# BLEU和METEOR计算库
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk

logger = logging.getLogger(__name__)

# 下载NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


@dataclass
class EvaluationConfig:
    """评估配置"""
    output_dir: str = "./evaluation_results"
    save_predictions: bool = True
    device: str = "cuda"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    num_beams: int = 4


class GeneChat2Evaluator:
    """GeneChat2评估器（论文指标）"""

    def __init__(self, model, config: Optional[EvaluationConfig] = None):
        self.model = model
        self.config = config or EvaluationConfig()
        self.smoothing = SmoothingFunction()

        # 创建输出目录
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("评估器初始化完成")

    def tokenize(self, text: str) -> List[str]:
        """分词

        Args:
            text: 输入文本

        Returns:
            tokens: 分词结果
        """
        return nltk.word_tokenize(text.lower())

    def calculate_bleu(
        self,
        references: List[List[str]],
        hypothesis: List[str],
        n: int = 4
    ) -> float:
        """计算BLEU分数

        Args:
            references: 参考文本（分词后）
            hypothesis: 假设文本（分词后）
            n: n-gram阶数（1, 2, 3, 4）

        Returns:
            bleu_score: BLEU分数
        """
        if n == 1:
            weights = (1.0, 0, 0, 0)
        elif n == 2:
            weights = (0.5, 0.5, 0, 0)
        elif n == 3:
            weights = (0.33, 0.33, 0.33, 0)
        elif n == 4:
            weights = (0.25, 0.25, 0.25, 0.25)
        else:
            raise ValueError(f"不支持的n-gram阶数: {n}")

        bleu = sentence_bleu(
            references,
            hypothesis,
            weights=weights,
            smoothing_function=self.smoothing.method1
        )

        return bleu

    def calculate_meteor(
        self,
        reference: str,
        hypothesis: str
    ) -> float:
        """计算METEOR分数

        Args:
            reference: 参考文本
            hypothesis: 假设文本

        Returns:
            meteor_score_value: METEOR分数
        """
        # 分词
        reference_tokens = self.tokenize(reference)
        hypothesis_tokens = self.tokenize(hypothesis)

        # 计算METEOR
        score = meteor_score([reference_tokens], hypothesis_tokens)

        return score

    def evaluate_single(
        self,
        dna_sequence: str,
        prompt: str,
        reference: str
    ) -> Dict:
        """评估单个样本

        Args:
            dna_sequence: DNA序列
            prompt: 提示
            reference: 参考描述

        Returns:
            metrics: 评估指标字典
        """
        # 生成预测
        prediction = self.model.generate(
            dna_sequence,
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            num_beams=self.config.num_beams
        )

        # 分词
        reference_tokens = self.tokenize(reference)
        prediction_tokens = self.tokenize(prediction)

        # 计算BLEU分数
        bleu_1 = self.calculate_bleu([reference_tokens], prediction_tokens, n=1)
        bleu_2 = self.calculate_bleu([reference_tokens], prediction_tokens, n=2)
        bleu_3 = self.calculate_bleu([reference_tokens], prediction_tokens, n=3)
        bleu_4 = self.calculate_bleu([reference_tokens], prediction_tokens, n=4)

        # 计算METEOR分数
        meteor = self.calculate_meteor(reference, prediction)

        metrics = {
            "bleu_1": bleu_1,
            "bleu_2": bleu_2,
            "bleu_3": bleu_3,
            "bleu_4": bleu_4,
            "meteor": meteor,
            "reference": reference,
            "prediction": prediction
        }

        return metrics

    def evaluate_dataset(self, dataloader) -> Dict:
        """评估整个数据集

        Args:
            dataloader: 数据加载器

        Returns:
            results: 评估结果字典
        """
        logger.info("开始评估数据集...")

        self.model.eval()

        all_metrics = {
            "bleu_1": [],
            "bleu_2": [],
            "bleu_3": [],
            "bleu_4": [],
            "meteor": []
        }

        predictions_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # 由于batch_size=1
                dna_sequence = batch["dna_sequences"][0]
                prompt = batch["prompts"][0]
                reference = batch["target_descriptions"][0]
                metadata = batch["metadata"][0]

                # 评估单个样本
                metrics = self.evaluate_single(dna_sequence, prompt, reference)

                # 累积指标
                all_metrics["bleu_1"].append(metrics["bleu_1"])
                all_metrics["bleu_2"].append(metrics["bleu_2"])
                all_metrics["bleu_3"].append(metrics["bleu_3"])
                all_metrics["bleu_4"].append(metrics["bleu_4"])
                all_metrics["meteor"].append(metrics["meteor"])

                # 保存预测
                if self.config.save_predictions:
                    predictions_list.append({
                        "gene_id": metadata.get("gene_id", "unknown"),
                        "gene_symbol": metadata.get("gene_symbol", "unknown"),
                        "reference": metrics["reference"],
                        "prediction": metrics["prediction"],
                        "bleu_1": metrics["bleu_1"],
                        "bleu_2": metrics["bleu_2"],
                        "bleu_3": metrics["bleu_3"],
                        "bleu_4": metrics["bleu_4"],
                        "meteor": metrics["meteor"]
                    })

        # 计算平均指标
        avg_metrics = {
            "bleu_1": np.mean(all_metrics["bleu_1"]),
            "bleu_2": np.mean(all_metrics["bleu_2"]),
            "bleu_3": np.mean(all_metrics["bleu_3"]),
            "bleu_4": np.mean(all_metrics["bleu_4"]),
            "meteor": np.mean(all_metrics["meteor"]),
            "num_samples": len(all_metrics["bleu_1"])
        }

        # 计算标准差
        std_metrics = {
            "bleu_1_std": np.std(all_metrics["bleu_1"]),
            "bleu_2_std": np.std(all_metrics["bleu_2"]),
            "bleu_3_std": np.std(all_metrics["bleu_3"]),
            "bleu_4_std": np.std(all_metrics["bleu_4"]),
            "meteor_std": np.std(all_metrics["meteor"])
        }

        results = {
            "average_metrics": avg_metrics,
            "std_metrics": std_metrics,
            "all_metrics": all_metrics
        }

        # 保存结果
        if self.config.save_predictions:
            self._save_results(results, predictions_list)

        # 打印结果
        self._print_results(avg_metrics, std_metrics)

        return results

    def _save_results(self, results: Dict, predictions: List[Dict]):
        """保存评估结果

        Args:
            results: 评估结果
            predictions: 预测列表
        """
        output_path = Path(self.config.output_dir)

        # 保存平均指标
        with open(output_path / "metrics.json", 'w') as f:
            json.dump({
                "average_metrics": results["average_metrics"],
                "std_metrics": results["std_metrics"]
            }, f, indent=2)

        # 保存所有预测
        with open(output_path / "predictions.json", 'w') as f:
            json.dump(predictions, f, indent=2)

        logger.info(f"评估结果已保存到: {output_path}")

    def _print_results(self, avg_metrics: Dict, std_metrics: Dict):
        """打印评估结果

        Args:
            avg_metrics: 平均指标
            std_metrics: 标准差指标
        """
        logger.info("=" * 60)
        logger.info("评估结果（论文指标）")
        logger.info("=" * 60)
        logger.info(f"样本数: {avg_metrics['num_samples']}")
        logger.info("-" * 60)
        logger.info(f"BLEU-1: {avg_metrics['bleu_1']:.4f} ± {std_metrics['bleu_1_std']:.4f}")
        logger.info(f"BLEU-2: {avg_metrics['bleu_2']:.4f} ± {std_metrics['bleu_2_std']:.4f}")
        logger.info(f"BLEU-3: {avg_metrics['bleu_3']:.4f} ± {std_metrics['bleu_3_std']:.4f}")
        logger.info(f"BLEU-4: {avg_metrics['bleu_4']:.4f} ± {std_metrics['bleu_4_std']:.4f}")
        logger.info(f"METEOR: {avg_metrics['meteor']:.4f} ± {std_metrics['meteor_std']:.4f}")
        logger.info("=" * 60)

        # 与论文结果对比
        logger.info("\n论文中的GeneChat结果:")
        logger.info(f"BLEU-1: 0.1937")
        logger.info(f"BLEU-2: 0.1384")
        logger.info(f"BLEU-3: 0.1065")
        logger.info(f"BLEU-4: 0.0816")
        logger.info(f"METEOR: 0.2725")
        logger.info("=" * 60)

    def compare_with_baseline(self, results: Dict, baseline_results: Dict) -> Dict:
        """与基线对比

        Args:
            results: 当前模型结果
            baseline_results: 基线结果

        Returns:
            comparison: 对比结果
        """
        comparison = {}

        for metric in ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "meteor"]:
            current = results["average_metrics"][metric]
            baseline = baseline_results["average_metrics"][metric]
            improvement = ((current - baseline) / baseline) * 100

            comparison[metric] = {
                "current": current,
                "baseline": baseline,
                "improvement": improvement
            }

        return comparison


def create_evaluator(
    model,
    output_dir: str = "./evaluation_results",
    max_new_tokens: int = 256,
    device: str = "cuda"
) -> GeneChat2Evaluator:
    """创建评估器的便捷函数"""
    config = EvaluationConfig(
        output_dir=output_dir,
        max_new_tokens=max_new_tokens,
        device=device
    )
    return GeneChat2Evaluator(model, config)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    print("评估器测试")
    print("请参考 scripts/evaluate_genechat2.py 获取完整评估示例")
