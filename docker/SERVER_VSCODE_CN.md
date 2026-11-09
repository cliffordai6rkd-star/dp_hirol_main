# U 盘迁移到服务器，用 VS Code 进入 Docker 训练

这次迁移包使用现有 `diffusion-policy:train` 镜像提供依赖，将服务器上的最新项目目录挂载到容器 `/workspace`。代码修改和训练输出直接保存在服务器磁盘，无需为源码变化重新构建镜像。

## 包含的文件

- 原来的四个 `diffusion-policy-train.tar.part-00` 至 `part-03` 镜像分片。
- `dp_main-current-source-data-model.tar.gz`：当前工作区源码和配置（含未提交修改）、Git 元数据、四份 LeRobot v3 数据、本地 DINOv3 权重，以及 CAD-WM 使用的原始 H5。
- `install-current.sh`、`SHA256SUMS-current` 和本说明。U 盘原有文件保留，安装本次包时使用 `install-current.sh`。

历史 `output` / `outputs`、checkpoint 目录、训练日志、可重新生成的缓存、Python 字节码和本机 `.aws` / `.codex` / `.agents` 不随包迁移。DINOv3 是必需的预训练输入，保留在 `model/model/dinov3-vitb16-pretrain-lvd1689m`。默认在服务器从头训练；续训时另行携带所选 checkpoint。

原始 H5 在包内位于 `data/wm_data/`，世界模型配置 `cad-wm/config/bench_train_config/libero_drawer_wrench_h5_portable.yaml` 使用相对路径。无需目标机存在原机器的 `/home/lcy/...` 路径。

## 服务器安装

目标服务器需要 Linux x86_64、支持镜像 CUDA 12.8 的 NVIDIA 驱动、Docker Compose v2 和 NVIDIA Container Toolkit。当前用户需要 Docker 使用权限。准备至少约 50 GB 空闲磁盘用于镜像和解包；训练输出、缓存还需要额外空间。

在服务器终端进入 U 盘的 `diffusion-policy-train` 目录，执行：

```bash
bash install-current.sh "$HOME/fdp_training"
cd "$HOME/fdp_training/dp_main"
bash docker/run_portable.sh check
bash docker/run_portable.sh dev
```

安装脚本先验证 SHA256，再导入镜像、解压项目。`check` 检查 GPU、通信、本地数据和 DINO；`dev` 创建后台容器 `fdp-dev`，保持运行，不启动训练。项目解压到服务器本地 SSD 后运行，U 盘无需保持连接。若服务器已存在同名容器，可直接附加该容器；其他项目可用 `DEV_CONTAINER_NAME=fdp-dev-other bash docker/run_portable.sh dev` 指定名称。

## VS Code 操作

1. 在本地 VS Code 安装 Remote-SSH 和 Dev Containers 扩展。
2. 用 Remote-SSH 连接服务器。
3. 在远程窗口按 F1，选择 `Dev Containers: Attach to Running Container...`，选择 `fdp-dev`。
4. 打开 `/workspace`。Python 解释器选择 `/opt/venv/bin/python`，容器中直接运行 `python`，无需 Conda。

挂载的源码和数据不依赖镜像中的旧源码。停止或删除开发容器后，服务器项目目录和训练输出仍保留。

VS Code 官方说明：
- https://code.visualstudio.com/docs/remote/ssh
- https://code.visualstudio.com/docs/devcontainers/attach-container

## 训练世界模型

在容器终端执行：

```bash
cd /workspace/cad-wm
CUDA_VISIBLE_DEVICES=0 python -m train.trainer.cad_wm_train \
  --config config/bench_train_config/libero_drawer_wrench_h5_portable.yaml
```

配置沿用本机这次世界模型设置：batch size 256、20 万 optimizer steps、每 2 万步保存、全数据评估，一阶低通 2 Hz。输出位于 `cad-wm/outputs/cad_wm_lowpass2hz/`。

## 训练教师并蒸馏学生

```bash
cd /workspace
python train_civi.py --gpus 0 \
  --teacher-override output_dir=data/outputs/civi_server/teacher \
  --student-override output_dir=data/outputs/civi_server/student
```

四卡时将 `--gpus 0` 替换为 `--gpus 0,1,2,3 --nproc-per-node 4`。教师和学生沿用当前 2 Hz 滤波与 DataLoader alpha。教师完成后自动按动作 MSE 选择权重，再开始蒸馏。

输出目录须是本次新目录；续训要显式指定对应 checkpoint。世界模型与教师/学生是独立入口，不会自动同时启动。

长任务可以用 `nohup` 启动并查看日志，避免依赖 VS Code 终端连接，例如：

```bash
cd /workspace/cad-wm
mkdir -p outputs/cad_wm_lowpass2hz
nohup python -u -m train.trainer.cad_wm_train \
  --config config/bench_train_config/libero_drawer_wrench_h5_portable.yaml \
  > outputs/cad_wm_lowpass2hz/train.log 2>&1 < /dev/null &
tail -f outputs/cad_wm_lowpass2hz/train.log
```

停止开发容器会停止其中的训练进程；关闭 VS Code 本身不会停止后台容器。服务器上使用 `docker stop fdp-dev` 停止容器。
