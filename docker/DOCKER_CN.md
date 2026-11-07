# 训练 Docker

通过 U 盘迁移当前源码、数据和本地 DINO，并在目标主机多卡训练新版教师及学生，使用 [U 盘迁移说明](PORTABLE_CN.md)。该入口会自动衔接两阶段。

镜像内包含项目源码、Python 3.10、CUDA 12.8 PyTorch、LeRobot 和 TorchCodec。服务器只需要 Docker Compose v2、NVIDIA 驱动和 NVIDIA Container Toolkit。数据、DINOv3 权重和 checkpoint 通过 Compose 挂载，不打进镜像。

## 本机构建和打包

先回到项目根目录，再执行：

```bash
cd /home/lcy/code/dp_hirol_main

APT_MIRROR=mirrors.aliyun.com \
  docker compose --progress plain -f docker/docker_compose.yaml build trainer

mkdir -p docker/artifacts
docker save -o docker/artifacts/diffusion-policy-train.tar diffusion-policy:train
cp docker/docker_compose.yaml docker/artifacts/docker_compose.yaml
```

把 `docker/artifacts/diffusion-policy-train.tar` 和 `docker/artifacts/docker_compose.yaml` 复制到服务器。服务器不需要重新安装 Python、PyTorch 或项目依赖。

## 服务器启动

先把数据和模型放到服务器，例如：

```text
/srv/dp/data/
/srv/dp/model/model/dinov3-vitb16-pretrain-lvd1689m/
```

创建 `.env`，路径是宿主机路径，训练命令中的路径是容器路径：

```dotenv
DATA_DIR=/srv/dp/data
MODEL_DIR=/srv/dp/model
DATASET_PATH=/workspace/data/libero_open_top_drawer_put_bowl_20hz_lerobot_v3
DINOV3_MODEL_PATH=/workspace/model/model/dinov3-vitb16-pretrain-lvd1689m
TEACHER_CHECKPOINT=/workspace/data/outputs/2026.11.01/22.57.02_091072_dp_distill_pipeline/teacher/checkpoints/optimizer_step=00180000.ckpt
```

加载镜像并直接启动默认训练：

```bash
docker load -i docker/artifacts/diffusion-policy-train.tar
docker compose -f docker/docker_compose.yaml up trainer
```

默认命令是当前结构化 RenderNet 的纯 token MSE 蒸馏，使用固定的 11/01 教师 checkpoint。默认日志在前台显示；后台运行时使用：

```bash
docker compose -f docker/docker_compose.yaml up -d trainer
docker compose -f docker/docker_compose.yaml logs -f trainer
```

## 覆盖训练命令

Compose 通过 `TRAIN_COMMAND` 控制容器启动后执行的命令。运行 FDP 四卡训练：

```bash
TRAIN_COMMAND='torchrun --standalone --nproc_per_node=4 train.py --config-dir=diffusion_policy/config --config-name=train_dp_insert_usb_force_aware training.distributed.enabled=true' \
  docker compose -f docker/docker_compose.yaml up trainer
```

进入容器做诊断：

```bash
docker compose -f docker/docker_compose.yaml run --rm --entrypoint bash trainer
```

## 共享内存和常用命令

默认 `/dev/shm` 为 64GB，供 DataLoader worker 和图像缓存使用。服务器内存较小时可以降低：

```bash
SHM_SIZE=32g docker compose -f docker/docker_compose.yaml up -d trainer
```

```bash
docker compose -f docker/docker_compose.yaml ps
docker compose -f docker/docker_compose.yaml logs -f trainer
docker compose -f docker/docker_compose.yaml down
```

服务器 GPU 容器运行时检查：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 nvidia-smi
```

如果出现 `permission denied while trying to connect to the Docker API`，当前用户还没有加入
`docker` 用户组。执行一次下面的命令，然后重新登录，或在当前终端运行 `newgrp docker`：

```bash
sudo usermod -aG docker "$USER"
newgrp docker
docker version
```
