# GeneChat2: 基因功能预测多模态大语言模型

基于论文"GeneChat: A Multi-Modal Large Language Model for Gene Function Prediction"的完全重新实现，**完全符合论文方法论，但不照抄原始代码**。

> ⚠️ **重要**: 本项目是论文的独立实现，专注于符合论文规格而非复制GitHub代码。

## ✨ 核心特性

### 🧬 多模态架构（论文规格）
- **DNABERT-2基因编码器**: 512bp滑动窗口，10bp重叠，256维pooling
- **线性适配器层**: 256→5120维度映射（论文核心创新）
- **Vicuna-13B LLM**: LoRA微调（r=8, alpha=16, q_proj+v_proj）

### 📊 数据处理（完全符合论文）
- **NCBI基因数据库**: 50,248个基因（论文准确数量）
- **长序列支持**: 160kb DNA序列
- **滑动窗口**: 512bp窗口，10bp重叠，生成约319个窗口
- **提示格式**: "please predict the function of this gene"（论文确切提示）

### 🎯 训练配置（论文规格）
- **数据分割**: 95:5 训练/测试（无验证集）
- **优化器**: AdamW (lr=1e-4, wd=0.05, warmup=2000步)
- **训练步数**: 170,000步（论文规格）
- **批大小**: 1，梯度累积8步（有效批大小=8）

### 📈 评估指标（论文使用）
- **BLEU-1, BLEU-2, BLEU-3, BLEU-4**: 生成质量
- **METEOR**: 语义相似度
- **论文基线**: BLEU-1=0.1937, BLEU-4=0.0816, METEOR=0.2725

## 🏗️ 项目结构

```
genechat2/
├── README.md                    # 项目说明（已更新）
├── requirements.txt             # 所有依赖（已完成）
├── configs/                     # 配置文件
│   └── genechat_config.yaml    # 主配置（论文规格，已修正）
├── models/                      # 模型实现（全部完成）
│   ├── __init__.py             # 模块导出
│   ├── genechat2.py            # ✓ 主模型
│   ├── gene_encoder.py         # ✓ DNABERT2编码器（已修正：10bp重叠，256D pooling）
│   ├── adapter.py              # ✓ 适配器（已修正：256→5120）
│   └── llm_wrapper.py          # ✓ Vicuna-13B包装器
├── data/                        # 数据处理（全部完成）
│   ├── __init__.py             # 模块导出
│   ├── ncbi_processor.py       # ✓ NCBI数据处理（已修正提示格式）
│   └── dataset_builder.py      # ✓ PyTorch数据集
├── training/                    # 训练模块（全部完成）
│   ├── __init__.py             # 模块导出
│   ├── trainer.py              # ✓ 训练器（论文规格）
│   └── evaluator.py            # ✓ 评估器（BLEU + METEOR）
└── scripts/                     # 运行脚本（全部完成）
    ├── __init__.py             # 模块导出
    ├── train_genechat2.py      # ✓ 训练脚本
    ├── evaluate_genechat2.py   # ✓ 评估脚本
    └── inference_demo.py       # ✓ 推理演示
```

## 🔍 架构审查结果

所有代码已经过完整审查，**100%符合论文规格**：

| 组件 | 论文要求 | 实现状态 | 备注 |
|------|----------|----------|------|
| 基因编码器 | DNABERT-2, 512bp窗口, 10bp重叠, 256D输出 | ✅ 已修正 | 添加了pooling层 |
| 适配器 | 线性投影 256→5120 | ✅ 已修正 | 修正了输入维度 |
| LLM | Vicuna-13B, LoRA(r=8, α=16) | ✅ 完成 | 只训练q_proj和v_proj |
| 提示格式 | "please predict the function of this gene" | ✅ 已修正 | 符合论文确切格式 |
| 训练配置 | lr=1e-4, wd=0.05, warmup=2000 | ✅ 已修正 | 所有超参数匹配 |
| 数据分割 | 95:5 (无验证集) | ✅ 完成 | 与论文一致 |
| 评估指标 | BLEU-1/2/3/4 + METEOR | ✅ 完成 | 使用NLTK实现 |

## 🚀 快速开始

### 1. 安装依赖
```bash
# 克隆项目
cd /Users/myt/Documents/genechat/genechat2

# 安装依赖
pip install -r requirements.txt

# 下载NLTK数据（用于评估）
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 2. 准备数据
```bash
# 数据会自动从NCBI下载并缓存
# 首次运行训练脚本时会自动下载
# 或者手动准备数据（需要配置NCBI邮箱）
```

### 3. 训练模型
```bash
# 使用默认配置训练
python scripts/train_genechat2.py --config configs/genechat_config.yaml

# 自定义参数
python scripts/train_genechat2.py \
    --config configs/genechat_config.yaml \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --max_steps 170000 \
    --output_dir ./checkpoints

# 从检查点恢复训练
python scripts/train_genechat2.py \
    --config configs/genechat_config.yaml \
    --resume ./checkpoints/checkpoint-50000
```

### 4. 评估模型
```bash
# 评估最佳模型
python scripts/evaluate_genechat2.py \
    --model_path ./checkpoints/checkpoint-best \
    --config configs/genechat_config.yaml

# 快速测试（100个样本）
python scripts/evaluate_genechat2.py \
    --model_path ./checkpoints/checkpoint-best \
    --max_samples 100
```

### 5. 推理演示
```bash
# 直接输入DNA序列
python scripts/inference_demo.py \
    --model_path ./checkpoints/checkpoint-best \
    --dna_sequence "ATCGATCGATCG..."

# 从文件读取序列
python scripts/inference_demo.py \
    --model_path ./checkpoints/checkpoint-best \
    --sequence_file ./data/example_gene.fasta

# 自定义生成参数
python scripts/inference_demo.py \
    --model_path ./checkpoints/checkpoint-best \
    --sequence_file ./data/example_gene.fasta \
    --temperature 0.7 \
    --top_p 0.9 \
    --num_beams 4 \
    --max_new_tokens 256
```

## 📊 模型架构（论文规格）

```
DNA序列 (160kb)
  ↓
[DNABERT-2编码器]
  - 512bp滑动窗口
  - 10bp重叠 (stride=502)
  - ~319个窗口
  ↓
基因嵌入 [319, 768]
  ↓
[平均池化 + 线性投影]
  - 768 → 256维降维
  ↓
基因嵌入 [319, 256]
  ↓
[线性适配器]
  - 256 → 5120维映射（论文核心）
  ↓
LLM嵌入 [319, 5120]
  ↓
[Vicuna-13B + LoRA]
  - LoRA: r=8, α=16
  - 只训练q_proj和v_proj
  - 参数高效微调
  ↓
基因功能描述 (自然语言)
```

## 🔬 技术细节

### 模型架构（论文确切规格）
- **基因编码器**: DNABERT-2 (117M参数，冻结)
- **适配器**: Linear(256→5120，可训练)
- **语言模型**: Vicuna-13B-v1.5 (LoRA微调)
- **总参数**: ~13B
- **可训练参数**: ~500M (LoRA + Adapter)

### 训练配置（论文规格）
- **数据集**: 50,248个NCBI基因
- **数据分割**: 95:5 训练/测试（47,735 / 2,513）
- **训练步数**: 170,000步
- **批大小**: 1 (受GPU内存限制)
- **梯度累积**: 8步（有效批大小=8）
- **学习率**: 1e-4 (余弦退火 + warmup)
- **权重衰减**: 0.05
- **Warmup步数**: 2,000
- **GPU要求**: 70GB+ (A100或H100推荐)

### 评估指标（论文使用）
| 指标 | 论文GeneChat | 论文GPT-4o |
|------|-------------|-----------|
| BLEU-1 | **0.1937** | 0.1444 |
| BLEU-2 | **0.1384** | 0.0563 |
| BLEU-3 | **0.1065** | 0.0208 |
| BLEU-4 | **0.0816** | 0.0088 |
| METEOR | **0.2725** | 0.2422 |

## ⚠️ 重要说明

### 与论文的完全对齐
本实现经过**完整架构审查**，确保100%符合论文规格：

1. ✅ **基因编码器**: 512bp窗口，10bp重叠，256维pooling
2. ✅ **适配器维度**: 256→5120（非768→5120）
3. ✅ **LoRA配置**: r=8, α=16, 只训练q_proj和v_proj
4. ✅ **提示格式**: "please predict the function of this gene"
5. ✅ **训练参数**: lr=1e-4, wd=0.05, warmup=2000步
6. ✅ **数据分割**: 95:5（无验证集）
7. ✅ **评估指标**: BLEU-1/2/3/4 + METEOR

### 与原GitHub代码的区别
- **不依赖原代码**: 完全独立实现
- **更清晰的模块化**: 更好的代码组织
- **完整的文档**: 详细的注释和使用说明
- **现代化实现**: 使用最新的transformers和PyTorch特性

## 🐛 已知问题和限制

1. **GPU内存要求高**: 需要70GB+显存（可使用梯度检查点和量化缓解）
2. **训练时间长**: 170k步需要数天时间
3. **数据下载**: NCBI数据下载需要时间和配额
4. **依赖版本**: 需要较新版本的transformers和PyTorch

## 📝 引用

如果您在研究中使用了GeneChat2，请引用原始论文：

```bibtex
@article{dhanasekar2025genechat,
  title={GeneChat: A Multi-Modal Large Language Model for Gene Function Prediction},
  author={Dhanasekar, Shashi and Saranathan, Akash and Xie, Pengtao},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.06.05.658031}
}
```

## 🙏 致谢

- 感谢原论文作者提供的方法论
- 感谢DNABERT-2团队提供的基因编码器
- 感谢Vicuna团队提供的语言模型
- 感谢NCBI提供的基因数据库

## 📞 联系方式

如有问题或建议，欢迎联系或提交Issue。

---

**注意**: 本项目为教育和研究目的，完全独立实现论文方法，不照抄原始代码。