"""
GeneChat训练器

实现完整的训练循环，支持论文中的所有训练配置。

关键特性：
- 完整的训练/评估循环
- 梯度累积（有效批大小=8）
- 混合精度训练
- 检查点保存和恢复
- 损失和指标记录
- 符合论文训练规格
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, List
import logging
from pathlib import Path
from tqdm import tqdm
import json
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TrainerConfig:
    """训练器配置（论文规格）"""
    # 训练参数
    max_epochs: int = 10
    max_steps: int = 170000  # 论文训练170k步
    batch_size: int = 1
    gradient_accumulation_steps: int = 8  # 有效批大小=8

    # 优化器参数（论文规格）
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # 学习率调度
    warmup_steps: int = 2000
    min_lr: float = 1e-6

    # 训练稳定性
    max_grad_norm: float = 1.0
    fp16: bool = True
    gradient_checkpointing: bool = True

    # 检查点
    save_steps: int = 500
    save_total_limit: int = 5
    output_dir: str = "./checkpoints"

    # 日志
    logging_steps: int = 10
    log_dir: str = "./logs"

    # 评估
    eval_steps: int = 500

    # 设备
    device: str = "cuda"

    # 早停
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001


class GeneChat2Trainer:
    """GeneChat2训练器"""

    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader=None,
        config: Optional[TrainerConfig] = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or TrainerConfig()

        # 移动模型到设备
        self.model = self.model.to(self.config.device)

        # 初始化优化器
        self.optimizer = self._create_optimizer()

        # 初始化学习率调度器
        self.scheduler = self._create_scheduler()

        # 混合精度训练
        self.scaler = GradScaler() if self.config.fp16 else None

        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.config.log_dir)

        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        self.patience_counter = 0

        # 创建输出目录
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("训练器初始化完成")
        self._log_training_info()

    def _create_optimizer(self) -> AdamW:
        """创建优化器（论文规格）"""
        # 分离需要权重衰减和不需要的参数
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon
        )

        logger.info(f"优化器创建完成: AdamW (lr={self.config.learning_rate}, wd={self.config.weight_decay})")
        return optimizer

    def _create_scheduler(self):
        """创建学习率调度器（余弦退火 + warmup）"""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

        def lr_lambda(current_step):
            if current_step < self.config.warmup_steps:
                # Warmup阶段
                return float(current_step) / float(max(1, self.config.warmup_steps))
            else:
                # 余弦退火阶段
                progress = float(current_step - self.config.warmup_steps) / \
                          float(max(1, self.config.max_steps - self.config.warmup_steps))
                return max(
                    self.config.min_lr / self.config.learning_rate,
                    0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
                )

        scheduler = LambdaLR(self.optimizer, lr_lambda)

        logger.info(f"学习率调度器创建完成: Cosine with Warmup (warmup_steps={self.config.warmup_steps})")
        return scheduler

    def _log_training_info(self):
        """记录训练信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info("=" * 60)
        logger.info("训练配置")
        logger.info("=" * 60)
        logger.info(f"最大轮数: {self.config.max_epochs}")
        logger.info(f"最大步数: {self.config.max_steps}")
        logger.info(f"批大小: {self.config.batch_size}")
        logger.info(f"梯度累积步数: {self.config.gradient_accumulation_steps}")
        logger.info(f"有效批大小: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"学习率: {self.config.learning_rate}")
        logger.info(f"权重衰减: {self.config.weight_decay}")
        logger.info(f"Warmup步数: {self.config.warmup_steps}")
        logger.info(f"混合精度: {self.config.fp16}")
        logger.info(f"总参数: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        logger.info("=" * 60)

    def train_step(self, batch: Dict) -> float:
        """单步训练

        Args:
            batch: 批数据

        Returns:
            loss: 损失值
        """
        # 由于batch_size=1，取第一个样本
        dna_sequence = batch["dna_sequences"][0]
        prompt = batch["prompts"][0]
        target = batch["target_descriptions"][0]

        # 前向传播
        if self.config.fp16:
            with autocast():
                outputs = self.model.forward(dna_sequence, prompt, target)
                loss = outputs["loss"]
                loss = loss / self.config.gradient_accumulation_steps
        else:
            outputs = self.model.forward(dna_sequence, prompt, target)
            loss = outputs["loss"]
            loss = loss / self.config.gradient_accumulation_steps

        # 反向传播
        if self.config.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def train_epoch(self) -> Dict:
        """训练一个epoch"""
        self.model.train()

        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch}",
            leave=True
        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            # 训练步
            loss = self.train_step(batch)
            epoch_loss += loss
            num_batches += 1

            # 梯度累积
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.config.fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # 记录
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]

                    self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                    self.writer.add_scalar("train/learning_rate", lr, self.global_step)

                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "step": self.global_step
                    })

                # 评估
                if self.eval_dataloader and self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    logger.info(f"Step {self.global_step} - Eval loss: {eval_metrics['eval_loss']:.4f}")

                    # 早停检查
                    if self._check_early_stopping(eval_metrics['eval_loss']):
                        logger.info("触发早停，停止训练")
                        return {"epoch_loss": epoch_loss / num_batches, "early_stop": True}

                # 保存检查点
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()

                # 检查是否达到最大步数
                if self.global_step >= self.config.max_steps:
                    logger.info(f"达到最大步数 {self.config.max_steps}，停止训练")
                    return {"epoch_loss": epoch_loss / num_batches, "max_steps_reached": True}

        return {"epoch_loss": epoch_loss / num_batches}

    def evaluate(self) -> Dict:
        """评估模型"""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
                dna_sequence = batch["dna_sequences"][0]
                prompt = batch["prompts"][0]
                target = batch["target_descriptions"][0]

                outputs = self.model.forward(dna_sequence, prompt, target)
                loss = outputs["loss"]

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        self.writer.add_scalar("eval/loss", avg_loss, self.global_step)

        self.model.train()

        return {"eval_loss": avg_loss}

    def _check_early_stopping(self, eval_loss: float) -> bool:
        """检查早停条件

        Args:
            eval_loss: 评估损失

        Returns:
            should_stop: 是否应该停止训练
        """
        if eval_loss < self.best_eval_loss - self.config.early_stopping_threshold:
            self.best_eval_loss = eval_loss
            self.patience_counter = 0
            # 保存最佳模型
            self.save_checkpoint(is_best=True)
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.early_stopping_patience:
                return True
            return False

    def train(self):
        """完整训练循环"""
        logger.info("开始训练...")

        try:
            for epoch in range(self.config.max_epochs):
                self.current_epoch = epoch

                # 训练一个epoch
                metrics = self.train_epoch()

                logger.info(f"Epoch {epoch} 完成 - Loss: {metrics['epoch_loss']:.4f}")

                # 检查是否需要停止
                if metrics.get("early_stop") or metrics.get("max_steps_reached"):
                    break

            logger.info("训练完成！")

            # 保存最终模型
            self.save_checkpoint(is_final=True)

        except KeyboardInterrupt:
            logger.info("训练被用户中断")
            self.save_checkpoint(is_interrupted=True)
        except Exception as e:
            logger.error(f"训练过程中出错: {e}")
            raise
        finally:
            self.writer.close()

    def save_checkpoint(self, is_best: bool = False, is_final: bool = False, is_interrupted: bool = False):
        """保存检查点

        Args:
            is_best: 是否是最佳模型
            is_final: 是否是最终模型
            is_interrupted: 是否是中断保存
        """
        checkpoint_name = f"checkpoint-{self.global_step}"
        if is_best:
            checkpoint_name = "checkpoint-best"
        elif is_final:
            checkpoint_name = "checkpoint-final"
        elif is_interrupted:
            checkpoint_name = "checkpoint-interrupted"

        checkpoint_path = Path(self.config.output_dir) / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # 保存模型
        self.model.save_model(str(checkpoint_path))

        # 保存训练状态
        training_state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_eval_loss": self.best_eval_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }

        if self.scaler:
            training_state["scaler_state"] = self.scaler.state_dict()

        torch.save(training_state, checkpoint_path / "training_state.pt")

        # 保存配置
        with open(checkpoint_path / "trainer_config.json", 'w') as f:
            json.dump(vars(self.config), f, indent=2)

        logger.info(f"检查点已保存: {checkpoint_path}")

        # 限制检查点数量
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """清理旧的检查点"""
        checkpoints = sorted(
            [d for d in Path(self.config.output_dir).iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-") and d.name not in
             ["checkpoint-best", "checkpoint-final", "checkpoint-interrupted"]],
            key=lambda x: x.stat().st_mtime
        )

        while len(checkpoints) > self.config.save_total_limit:
            checkpoint = checkpoints.pop(0)
            import shutil
            shutil.rmtree(checkpoint)
            logger.info(f"删除旧检查点: {checkpoint}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点

        Args:
            checkpoint_path: 检查点路径
        """
        logger.info(f"加载检查点: {checkpoint_path}")

        checkpoint_path = Path(checkpoint_path)

        # 加载模型
        self.model.load_model(str(checkpoint_path))

        # 加载训练状态
        training_state = torch.load(checkpoint_path / "training_state.pt")

        self.global_step = training_state["global_step"]
        self.current_epoch = training_state["current_epoch"]
        self.best_eval_loss = training_state["best_eval_loss"]
        self.optimizer.load_state_dict(training_state["optimizer_state"])
        self.scheduler.load_state_dict(training_state["scheduler_state"])

        if self.scaler and "scaler_state" in training_state:
            self.scaler.load_state_dict(training_state["scaler_state"])

        logger.info(f"检查点加载完成 (step={self.global_step}, epoch={self.current_epoch})")


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    print("训练器测试")
    print("请参考 scripts/train_genechat2.py 获取完整训练示例")
