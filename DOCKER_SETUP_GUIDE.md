# GeneChat2 A800开发机配置指南

## 🖥️ 镜像选择

### 推荐镜像（按优先级）

#### 选项1: PyTorch官方镜像 ⭐⭐⭐⭐⭐ (最推荐)
```
镜像名称: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
或
镜像名称: pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

包含:
✅ Python 3.10
✅ PyTorch 2.1+/2.3+ (GPU版本)
✅ CUDA 12.1
✅ cuDNN 8
✅ 所有必需的开发工具
```

#### 选项2: NVIDIA NGC PyTorch ⭐⭐⭐⭐⭐
```
镜像名称: nvcr.io/nvidia/pytorch:23.12-py3

包含:
✅ 针对A800/A100优化
✅ PyTorch 2.1+
✅ CUDA 12.3
✅ Apex (混合精度训练)
✅ 预装transformers等常用库
```

#### 选项3: Ubuntu + CUDA基础镜像 ⭐⭐⭐
```
镜像名称: nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

需要手动安装:
- Python 3.10+
- PyTorch
- transformers等库
```

---

## 🚀 启动命令

### 最简单的启动命令（推荐新手）

```bash
# 使用PyTorch官方镜像
docker run -it --gpus all \
  --name genechat2_training \
  --shm-size=16g \
  -v /path/to/genechat2:/workspace/genechat2 \
  -v /path/to/data:/workspace/data \
  -v /path/to/checkpoints:/workspace/checkpoints \
  pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel \
  /bin/bash
```

### 完整的生产级启动命令（推荐）

```bash
docker run -it --gpus all \
  --name genechat2_a800 \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /path/to/genechat2:/workspace/genechat2 \
  -v /path/to/data:/workspace/data \
  -v /path/to/checkpoints:/workspace/checkpoints \
  -v /path/to/cache:/workspace/cache \
  -p 6006:6006 \
  -p 8888:8888 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TRANSFORMERS_CACHE=/workspace/cache \
  -e HF_HOME=/workspace/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
  pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel \
  /bin/bash -c "cd /workspace/genechat2 && bash scripts/setup_a800_training.sh && /bin/bash"
```

### 参数说明

```bash
--gpus all                    # 使用所有GPU
--name genechat2_a800        # 容器名称
--shm-size=32g               # 共享内存32GB（重要！防止dataloader错误）
--ulimit memlock=-1          # 解除内存锁定限制
--ulimit stack=67108864      # 增加栈大小

# 挂载目录
-v /host/path:/container/path  # 映射路径

# 端口映射
-p 6006:6006                 # TensorBoard端口
-p 8888:8888                 # Jupyter Notebook端口（可选）

# 环境变量
-e CUDA_VISIBLE_DEVICES=0    # 指定GPU
-e TRANSFORMERS_CACHE=/workspace/cache  # 模型缓存位置
-e HF_HOME=/workspace/cache  # HuggingFace缓存
-e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # CUDA内存分配优化
```

---

## 📁 目录映射建议

### 最小配置（必需）

```bash
/your/host/genechat2 → /workspace/genechat2       # 代码目录
/your/host/checkpoints → /workspace/checkpoints   # 检查点目录（重要！）
```

### 推荐配置

```bash
/your/host/genechat2 → /workspace/genechat2       # 代码目录
/your/host/data → /workspace/data                 # 数据目录
/your/host/checkpoints → /workspace/checkpoints   # 检查点（约300GB空间）
/your/host/cache → /workspace/cache               # 模型缓存（约30GB）
/your/host/logs → /workspace/logs                 # 日志目录
```

---

## 🔧 容器启动后的配置步骤

### 方法A: 自动配置（推荐）

启动命令中已包含自动配置，容器启动后会自动运行：

```bash
cd /workspace/genechat2
bash scripts/setup_a800_training.sh
```

这会自动完成：
1. ✅ 检查GPU和CUDA
2. ✅ 安装所有Python依赖
3. ✅ 下载NLTK数据
4. ✅ 创建必要目录
5. ✅ 验证环境

### 方法B: 手动配置

如果需要手动配置：

```bash
# 1. 进入工作目录
cd /workspace/genechat2

# 2. 更新pip
pip install --upgrade pip

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载NLTK数据
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# 5. 验证环境
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## ⚡ 快速测试

容器配置完成后，立即运行测试：

```bash
# 测试训练（30分钟）
cd /workspace/genechat2
bash scripts/run_a800_training.sh --test

# 如果测试成功，开始完整训练
bash scripts/run_a800_training.sh --config configs/genechat_a800_config.yaml
```

---

## 🌐 云平台特定配置

### 如果你使用的是租用的GPU服务器

#### AutoDL / 恒源智慧 / 矩池云

```bash
镜像: PyTorch 2.1 / 2.3 (CUDA 12.1)
GPU: NVIDIA A800-80G

启动命令（平台通常会自动配置Docker）:
cd /root/genechat2
bash scripts/setup_a800_training.sh
```

#### 阿里云 / 腾讯云 GPU实例

```bash
# 1. SSH连接到实例
ssh user@your-instance-ip

# 2. 安装Docker（如果未安装）
curl -fsSL https://get.docker.com | sh
sudo systemctl start docker

# 3. 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 4. 运行容器（使用上面的完整启动命令）
```

---

## 🎮 开发机配置面板示例

如果你使用的是图形化配置界面（如AutoDL、矩池云）:

### 基础配置
```
实例类型: A800-80G
镜像: PyTorch 2.3.0 (CUDA 12.1)
数据盘: 500GB+ (推荐1TB)
```

### 高级配置（如有）
```
共享内存: 32GB
端口映射:
  - 6006 (TensorBoard)
  - 8888 (Jupyter，可选)
```

### 启动脚本（在平台的"启动脚本"框中填入）
```bash
#!/bin/bash
cd /root
git clone https://github.com/your-repo/genechat2.git  # 或上传代码
cd genechat2
bash scripts/setup_a800_training.sh
```

---

## 📦 完整的启动流程

### 1. 准备代码（本地）

```bash
# 将genechat2代码打包
cd /Users/myt/Documents/genechat
tar -czf genechat2.tar.gz genechat2/

# 上传到服务器
scp genechat2.tar.gz user@server:/path/to/
```

### 2. 服务器端配置

```bash
# 解压代码
tar -xzf genechat2.tar.gz
cd genechat2

# 启动Docker容器（使用上面的完整命令）
docker run -it --gpus all \
  --name genechat2_a800 \
  --shm-size=32g \
  -v $(pwd):/workspace/genechat2 \
  -v /data:/workspace/data \
  -v /checkpoints:/workspace/checkpoints \
  -p 6006:6006 \
  pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel \
  /bin/bash
```

### 3. 容器内配置

```bash
# 容器启动后
cd /workspace/genechat2
bash scripts/setup_a800_training.sh
```

### 4. 开始训练

```bash
# 测试训练
bash scripts/run_a800_training.sh --test

# 完整训练
bash scripts/run_a800_training.sh --config configs/genechat_a800_config.yaml
```

---

## 🔍 验证清单

容器启动后，检查以下项目：

```bash
# ✅ 1. GPU可见
nvidia-smi
# 应该看到: NVIDIA A800-80G (80GB)

# ✅ 2. CUDA可用
python -c "import torch; print(torch.cuda.is_available())"
# 应该输出: True

# ✅ 3. 目录挂载正确
ls -lh /workspace/genechat2
ls -lh /workspace/checkpoints

# ✅ 4. 依赖安装完成
pip list | grep -E "torch|transformers|peft"

# ✅ 5. 代码可以导入
python -c "from models.genechat2 import GeneChat2Config"
```

---

## 💡 常见问题

### Q1: Docker镜像太大，下载很慢？
```bash
# 使用国内镜像加速
# 编辑 /etc/docker/daemon.json
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://mirror.ccs.tencentyun.com"
  ]
}

# 重启Docker
sudo systemctl restart docker
```

### Q2: 共享内存不足错误？
```bash
# 增加 --shm-size
docker run --shm-size=32g ...

# 或使用主机内存
docker run --ipc=host ...
```

### Q3: 权限问题？
```bash
# 使用当前用户运行
docker run -u $(id -u):$(id -g) ...

# 或在容器内
chown -R $(id -u):$(id -g) /workspace/genechat2
```

---

## 🎯 我的推荐配置

### 最佳实践配置（复制即用）

```bash
# 1. 镜像选择
pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# 2. 启动命令
docker run -d --gpus all \
  --name genechat2_prod \
  --shm-size=32g \
  --restart unless-stopped \
  -v /data/genechat2:/workspace/genechat2 \
  -v /data/checkpoints:/workspace/checkpoints \
  -v /data/cache:/workspace/cache \
  -p 6006:6006 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TRANSFORMERS_CACHE=/workspace/cache \
  pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel \
  /bin/bash -c "cd /workspace/genechat2 && bash scripts/setup_a800_training.sh && bash scripts/run_a800_training.sh --config configs/genechat_a800_config.yaml"

# 这个命令会：
# ✅ 后台运行（-d）
# ✅ 自动重启（--restart）
# ✅ 自动配置环境
# ✅ 自动开始训练

# 3. 查看训练日志
docker logs -f genechat2_prod

# 4. 进入容器（如需调试）
docker exec -it genechat2_prod /bin/bash
```

---

**准备好了吗？告诉我你选择的平台，我可以给出更具体的配置命令！**
