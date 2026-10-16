# 训练 Docker

Compose 已经配置好 GPU、共享内存、代码、数据、DINOv3 权重和输出目录。服务器不需要重新安装 Python、PyTorch 或 LeRobot；只需要 Docker Compose v2、NVIDIA 驱动和 NVIDIA Container Toolkit。

项目内目录保持如下：

```text
data/insert_usb_50hz_lerobot_v3/
model/dinov3-vitb16-pretrain-lvd1689m/
```

## 本机构建并打包

在项目根目录执行：

```bash
# 中国大陆网络建议使用镜像；APT_MIRROR 只填写域名，不要带 http:// 或路径。
APT_MIRROR=mirrors.aliyun.com \
  docker compose -f docker/docker_compose.yaml build --progress=plain trainer
mkdir -p docker/artifacts
docker save -o docker/artifacts/diffusion-policy-train.tar diffusion-policy:train
tar --exclude='.git' -czf ../diffusion-policy-bundle.tar.gz .
```

`diffusion-policy-bundle.tar.gz` 包含项目代码、数据、DINOv3 权重和预构建镜像。把这个压缩包复制到服务器即可。

## 服务器启动

解压后进入项目根目录：

```bash
tar -xzf diffusion-policy-bundle.tar.gz
docker load -i docker/artifacts/diffusion-policy-train.tar
docker compose -f docker/docker_compose.yaml up -d trainer
```

进入容器并启动四卡训练：

```bash
docker compose -f docker/docker_compose.yaml exec trainer bash

# 以下命令在容器内执行
NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train.py \
  --config-dir=diffusion_policy/config \
  --config-name=train_dp_insert_usb_force_aware \
  training.distributed.enabled=true
```

覆盖双阈值时，在 `training.distributed.enabled=true` 后追加：

```bash
free_threshold=0.4 contact_threshold=1.2
```

默认配置读取容器内的：

```text
/workspace/data/insert_usb_50hz_lerobot_v3
/workspace/model/dinov3-vitb16-pretrain-lvd1689m
```

## 共享内存

你的服务器有约 100GB CPU 内存，Compose 默认给容器 64GB 的 `/dev/shm`，用于四卡训练时的 DataLoader 和共享图像缓存。它不占用 4090 显存。

如果需要降低上限，可以在启动容器前设置：

```bash
SHM_SIZE=32gb docker compose -f docker/docker_compose.yaml up -d trainer
```

## 常用命令

```bash
docker compose -f docker/docker_compose.yaml ps
docker compose -f docker/docker_compose.yaml logs -f trainer
docker compose -f docker/docker_compose.yaml down
```

服务器没有外网时也不影响 Python 依赖，因为镜像已经在本机构建完成。服务器仍需确认 GPU 容器运行时：

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-cudnn8-runtime-ubuntu22.04 nvidia-smi
```
