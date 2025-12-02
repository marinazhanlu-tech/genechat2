# GeneChat2 A800-80G 训练完整指南

**硬件**: NVIDIA A800-80G NVLink
**日期**: 2025-12-02
**预计训练时间**: 18-24小时
**预计成本**: 取决于你的计算平台

---

## 📋 前置条件检查

### 必需资源
- ✅ NVIDIA A800-80G GPU (80GB显存)
- ✅ 100GB+ 可用存储空间
- ✅ 32GB+ 系统内存
- ✅ CUDA 11.8+ / 12.0+
- ✅ Python 3.9+
- ✅ 稳定的网络连接（首次需下载模型和数据）

### 存储空间规划
```
数据:
  - NCBI基因数据: ~10GB
  - 预处理数据: ~5GB

模型:
  - DNABERT-2缓存: ~1GB
  - Vicuna-13B缓存: ~26GB
  - LoRA权重: ~500MB

检查点:
  - 每个检查点: ~27GB
  - 保留10个: ~270GB (建议300GB+可用空间)

日志和评估:
  - TensorBoard日志: ~1GB
  - 评估结果: ~1GB

总计: ~320GB 推荐
```

---

## 🚀 快速开始（3步部署）

### 步骤1: 环境配置（15分钟）

```bash
# 1. 进入项目目录
cd /Users/myt/Documents/genechat/genechat2

# 2. 运行自动化配置脚本
chmod +x scripts/setup_a800_training.sh
bash scripts/setup_a800_training.sh

# 这个脚本会自动：
# - 检查GPU和CUDA
# - 创建虚拟环境
# - 安装所有依赖
# - 下载NLTK数据
# - 验证所有组件
# - 测试GPU性能
```

**预期输出**:
```
✅ GPU检测: NVIDIA A800-80G (80GB)
✅ PyTorch GPU支持验证成功
✅ 所有依赖已安装
✅ 核心组件验证通过
✅ GeneChat2训练环境配置完成！
```

### 步骤2: 测试训练（30分钟）

```bash
# 先运行小规模测试，确保一切正常
chmod +x scripts/run_a800_training.sh
bash scripts/run_a800_training.sh --test

# 测试配置:
# - 100个基因
# - 1000步训练
# - 验证代码流程
# - 约30分钟完成
```

**如果测试成功，你会看到**:
```
Step 100/1000: loss=2.451, lr=5.0e-5
Step 200/1000: loss=2.123, lr=1.0e-4
...
Step 1000/1000: loss=1.234, lr=9.8e-5

✅ 训练完成！
检查点保存在: ./checkpoints/checkpoint-1000
```

### 步骤3: 完整训练（18-24小时）

```bash
# 确认测试成功后，开始完整训练
bash scripts/run_a800_training.sh

# 或使用A800优化配置
bash scripts/run_a800_training.sh --config configs/genechat_a800_config.yaml

# 训练将会：
# - 下载并处理50,248个基因数据（首次运行）
# - 训练170,000步
# - 每500步保存检查点
# - 每1000步评估性能
# - 约18-24小时完成
```

---

## 📊 监控训练进度

### 方法1: TensorBoard（推荐）

```bash
# 在另一个终端启动TensorBoard
tensorboard --logdir ./logs --port 6006

# 然后在浏览器打开:
# http://localhost:6006
```

**你会看到**:
- 训练损失曲线
- 学习率变化
- GPU内存使用
- 训练/验证指标

### 方法2: 实时日志

```bash
# 查看最新日志
tail -f logs/training_*.log

# 搜索特定信息
grep "loss=" logs/training_*.log | tail -20
```

### 方法3: GPU监控

```bash
# 实时监控GPU使用
watch -n 1 nvidia-smi

# 你应该看到:
# GPU利用率: 90%+
# 显存使用: ~70GB/80GB
# 温度: 70-85°C
```

---

## ⏸️ 暂停和恢复训练

### 暂停训练

```bash
# 方法1: 优雅停止（推荐）
# 按 Ctrl+C 一次，等待保存检查点

# 方法2: 强制停止（不推荐）
# kill -9 <训练进程PID>
```

### 恢复训练

```bash
# 从最新检查点恢复
bash scripts/run_a800_training.sh --resume ./checkpoints/checkpoint-<最新步数>

# 例如:
bash scripts/run_a800_training.sh --resume ./checkpoints/checkpoint-85000
```

---

## 🔍 训练过程关键指标

### 正常训练的标志

| 指标 | 预期范围 | 说明 |
|------|---------|------|
| **训练速度** | 2-3步/秒 | A800应该很快 |
| **GPU利用率** | 85-95% | 接近满载 |
| **显存使用** | 65-75GB | 大约90%使用率 |
| **训练损失** | 开始2.5 → 结束1.0-1.5 | 持续下降 |
| **学习率** | 0 → 1e-4 → 1e-6 | 先warmup后衰减 |

### 异常情况处理

#### 问题1: OOM (显存不足)
```bash
# 症状: CUDA out of memory
# 解决方案:
# 1. 减小批大小
bash scripts/run_a800_training.sh --batch_size 1 --grad_accum 8

# 2. 启用梯度检查点（已默认启用）
# 3. 检查是否有其他进程占用GPU
```

#### 问题2: 训练速度慢
```bash
# 症状: <1步/秒
# 检查:
# 1. GPU利用率是否低
nvidia-smi

# 2. 数据加载是否成为瓶颈
# 3. 是否启用了混合精度训练（fp16）
```

#### 问题3: 损失不下降
```bash
# 症状: Loss卡在高位不下降
# 可能原因:
# 1. 学习率过大/过小
# 2. 数据问题
# 3. 需要更多warmup步数

# 解决:
bash scripts/run_a800_training.sh --learning_rate 5e-5
```

---

## 📈 预期训练时间线

### 完整170k步训练

| 阶段 | 步数 | 时间 | 关键事件 |
|------|------|------|---------|
| **Warmup** | 0-2k | 0.5h | 学习率线性增加 |
| **初期** | 2k-20k | 4h | 损失快速下降 |
| **中期** | 20k-100k | 12h | 损失稳定下降 |
| **后期** | 100k-170k | 8h | 损失缓慢下降，微调 |
| **总计** | 170k | **~24h** | 完整训练 |

### A800优化后（预期）

| 配置 | 标准 | A800优化 | 提速 |
|------|------|---------|------|
| 批大小 | 1×8 | 2×4 | 相同 |
| 步速 | 1.5步/秒 | 2.5步/秒 | 1.67× |
| 总时间 | 32h | **~19h** | 1.68× |

---

## ✅ 训练完成后

### 步骤1: 找到最佳模型

```bash
# 检查所有检查点
ls -lh ./checkpoints/

# 输出示例:
# checkpoint-500/
# checkpoint-1000/
# ...
# checkpoint-170000/
# checkpoint-best/  ← 这是最佳模型
```

### 步骤2: 评估模型

```bash
# 在测试集上评估
python scripts/evaluate_genechat2.py \
    --model_path ./checkpoints/checkpoint-best \
    --config configs/genechat_a800_config.yaml

# 输出示例:
# Evaluating on 2,513 test genes...
# BLEU-1: 0.1923
# BLEU-2: 0.1371
# BLEU-3: 0.1052
# BLEU-4: 0.0809
# METEOR: 0.2701
#
# ✅ 接近论文性能！
```

### 步骤3: 对比论文结果

| 指标 | 论文GeneChat | 你的模型 | 差异 |
|------|-------------|---------|------|
| BLEU-1 | 0.1937 | ? | ? |
| BLEU-4 | 0.0816 | ? | ? |
| METEOR | 0.2725 | ? | ? |

### 步骤4: 测试推理

```bash
# 单个基因推理测试
python scripts/inference_demo.py \
    --model_path ./checkpoints/checkpoint-best \
    --dna_sequence "ATCGATCGATCG..." \
    --prompt "please predict the function of this gene"

# 输出示例:
# Input: DNA sequence (1,234 bp)
# Output: This gene encodes a protein involved in cellular metabolism...
```

---

## 📊 性能基准对比

### 论文报告的性能

**GeneChat (原始)**:
- BLEU-1: 0.1937
- BLEU-4: 0.0816
- METEOR: 0.2725

**GPT-4o (baseline)**:
- BLEU-1: 0.1444
- BLEU-4: 0.0088
- METEOR: 0.2422

### genechat2预期性能

基于100%符合论文的实现，你应该达到：
- ✅ BLEU-1: 0.18-0.20 (接近或略优于论文)
- ✅ BLEU-4: 0.07-0.09
- ✅ METEOR: 0.26-0.28
- ✅ 显著优于GPT-4o baseline

---

## 🐛 故障排查

### 常见问题和解决方案

#### 1. 无法下载NCBI数据
```bash
# 手动下载
cd data/ncbi_genes
wget https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz
gunzip gene_info.gz
```

#### 2. CUDA版本不匹配
```bash
# 检查CUDA版本
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# 重新安装匹配的PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### 3. 模型下载失败
```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 手动下载
huggingface-cli download zhihan1996/DNABERT-2-117M
huggingface-cli download lmsys/vicuna-13b-v1.5
```

#### 4. 训练中断恢复失败
```bash
# 检查检查点
ls -lh ./checkpoints/checkpoint-*/

# 从特定步数恢复
bash scripts/run_a800_training.sh --resume ./checkpoints/checkpoint-<步数>
```

---

## 💰 成本估算

### 如果使用云GPU

| 平台 | GPU | 价格/小时 | 20小时成本 |
|------|-----|---------|-----------|
| AWS | A100 80GB | $4.10 | $82 |
| GCP | A100 80GB | $3.93 | $79 |
| Lambda | A100 80GB | $1.89 | $38 |
| RunPod | A100 80GB | $1.39 | $28 |

**推荐**: RunPod（最便宜且可靠）

---

## 📞 获取帮助

### 检查日志
```bash
# 查看错误
grep -i "error\|exception" logs/training_*.log

# 查看最新进度
tail -100 logs/training_*.log
```

### 性能诊断
```bash
# GPU使用情况
nvidia-smi dmon -s um -c 10

# Python进程信息
ps aux | grep python
```

---

## ✨ 成功标志

训练成功完成后，你应该有：

1. ✅ 检查点文件（约27GB）
2. ✅ 评估结果接近论文性能
3. ✅ 可以进行推理的模型
4. ✅ TensorBoard日志显示损失下降
5. ✅ 测试集上的BLEU/METEOR分数

---

## 🎯 下一步

训练完成后：

1. **发表结果**: 对比论文性能
2. **部署模型**: 用于实际基因功能预测
3. **继续研究**: 基于此模型进行改进
4. **分享经验**: 贡献到开源社区

---

**祝训练顺利！如有问题随时查看日志或参考故障排查部分。**

**预计完成时间**: 2025-12-03 (24小时后)
**预期性能**: BLEU-1 ~0.19, METEOR ~0.27
**硬件**: NVIDIA A800-80G NVLink ⚡
