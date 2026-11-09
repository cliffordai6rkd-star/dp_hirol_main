# U 盘迁移：多卡教师训练 + 学生蒸馏

本套 U 盘文件用于在另一台 **Linux x86_64 NVIDIA 多卡主机**上从头训练新版教师，再蒸馏学生。
使用 `diffusion-policy:train` 镜像提供依赖，同时将解压后的最新项目挂载到 `/workspace`。
运行的是挂载的代码，无需重新构建镜像或安装 Conda/Python。

## U 盘里有什么

`diffusion-policy-train/` 目录包含：

- `diffusion-policy-train.tar.part-00` 到 `part-03`：完整镜像的四个分片，全部需要保留。U 盘是 FAT32，单文件不能超过 4 GiB。
- `dp_main-source-data-model.tar.gz`：当前源码、配置、测试、Git 元数据、全部四份本地训练数据和本地 DINOv3 ViT-B/16 权重。
- `install.sh`：校验文件、导入镜像、解压项目；不会启动正式训练。
- `load-image.sh`：仅导入镜像，供手动安装时使用。
- `SHA256SUMS`、`使用说明.txt`：校验值和操作说明。

包含的数据目录：

```text
data/libero_open_top_drawer_put_bowl_20hz_lerobot_v3
data/libero_wm_insert_usb_lerobot_v3
data/insert_usb_50hz_lerobot_v3
data/cucumber_peeling_lerobot_v3
data/benchmarks
model/model/dinov3-vitb16-pretrain-lvd1689m
```

排除各层级的 `output` / `outputs`、可重新生成的缓存和 Python 字节码、机器专用的 `.aws` / `.codex` / `.agents`，
以及已经单独打包的 `docker/artifacts`。原机器的 `data/cache` 是指向回收站的链接，迁移时改为空目录。
旧 checkpoint、训练日志和教师目标缓存未迁移；本入口会生成新教师和新蒸馏目标。

## 目标主机准备

将文件解压到本地 SSD 后训练。准备足够的磁盘空间供 Docker 镜像展开、源码数据、训练输出和教师目标缓存使用；
镜像在原主机显示约 32.5 GB 磁盘占用，训练输出/缓存还需要额外空间。

目标主机需要：

- NVIDIA 驱动支持 CUDA 12.8；当前 AMP 配置使用 bfloat16，建议 Ampere 或更新架构。
- Docker Engine、Docker Compose v2（支持 `gpus` 字段，v2.30+）和 NVIDIA Container Toolkit。
- 当前用户有 Docker 使用权限。

先执行：

```bash
nvidia-smi
docker version
docker compose version
docker info
```

如果尚未安装容器 GPU 支持，按 [NVIDIA 官方安装说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 安装 Toolkit，
然后配置并重启 Docker：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

如果 Docker 提示权限不足，执行以下命令并重新登录终端：

```bash
sudo usermod -aG docker "$USER"
```

## 安装、检查、启动

在文件管理器打开 U 盘里的 `diffusion-policy-train` 文件夹，右键选择“在终端打开”，执行：

```bash
bash install.sh "$HOME/dp_training"
```

脚本会校验并读取全部镜像分片，不会在 U 盘上展开 Docker 镜像。如果目录已经存在，请换一个安装目录。
安装完成后：

```bash
cd "$HOME/dp_training/dp_main"
bash docker/run_portable.sh check
bash docker/run_portable.sh all
```

`check` 自动枚举可见 GPU，用 NCCL 完成通信和两次微型同步更新，并在每个 GPU 上加载本地 DINO、检查数据文件。
它不训练教师或学生。多卡应看到所有 rank 成功并最终显示 `DDP nccl`；单卡只验证单设备。
如果出现 `could not select device driver "nvidia"`，先完成 NVIDIA Container Toolkit 配置。
可以用 `docker run --rm --gpus all diffusion-policy:train nvidia-smi` 单独检查 GPU 容器支持。

`all` 才会开始正式训练，按顺序运行：

1. 教师：`sim_force_aware_dp_train.yaml`，未来逐点接触 alpha + FACTR 图像模糊课程。
2. 学生：`train_distill_render_net_patch_context.yaml`，自动读取刚训练的教师 checkpoint，重新生成兼容的教师目标。

两阶段都启动与可见 GPU 数量相同的进程。全局 batch size 保持 32，由各卡平分；默认适用于 1、2、4、8 卡等能整除 32 的数量。
数据默认使用 `libero_open_top_drawer_put_bowl_20hz_lerobot_v3`。动作、网络层数、未来采样网格和 loss 均沿用现有配置。
部署入口选择 CPU/PyAV 解码，首次迁移关闭学生 `torch.compile`，可在确认运行后用覆盖项启用；这不改变学生网络。
W&B 使用离线模式，本地 DINO 无需联网下载。

只选第 0、1 张卡时，两条命令都加相同前缀：

```bash
TRAIN_GPUS=0,1 bash docker/run_portable.sh check
TRAIN_GPUS=0,1 bash docker/run_portable.sh all
```

也可以用 `NPROC_PER_NODE=4` 指定进程数；它不能超过可见 GPU 数量。
预览命令而不启动训练：

```bash
bash docker/run_portable.sh all --dry-run
```

## 查看结果与分阶段运行

输出都保存在解压后的项目 `data/outputs/日期/时间_dp_distill_pipeline/`：

```text
pipeline.json                  两阶段状态和实际命令
teacher/train.log              教师日志
teacher/checkpoints/           教师权重
student/train.log              学生日志
student/checkpoints/           学生权重
```

终端退出会中断前台训练；长时间运行可放在 `tmux` 会话中。`Ctrl+C` 会停止当前训练进程。
学生教师目标缓存位于 `data/cache/render_net_teacher`，当前配置结束时清理。

新预处理和 contact alpha 改变教师的条件表示，需要微调或重训并验证动作性能后再生成蒸馏目标。
`all` 自动衔接两阶段；如果要先验收教师的动作表现，使用 `dp` 单独训练，再使用 `distill`：

```bash
bash docker/run_portable.sh dp --output-dir /workspace/data/outputs/teacher_review
# 验证教师动作性能后执行；下面选择上一步生成的教师。
bash docker/run_portable.sh distill --teacher-run /workspace/data/outputs/teacher_review/teacher
```

教师 alpha 本身不保证学生 RGB 像素恒等。此迁移只提供训练环境和代码，并未验证新教师的动作表现。

## 本次验证记录

- 相关 pytest：覆盖 token alpha、教师/学生双进程 DDP、缓存、验证与流水线 checkpoint 交接。
- 导出的 Docker 镜像内：**两个 CPU/Gloo 进程通过**通信和同步更新检查，均成功读取本地数据并加载真实 DINO，输出形状 `(1, 197, 768)`。
- Bash 语法、Compose 配置解析、镜像内四卡训练命令 dry-run 通过。
- 当前主机仅一张 GPU，且尚无可用的 NVIDIA 容器运行支持；**真实多 GPU / NCCL 尚未验证**，必须在目标主机执行 `check`。
- 未启动正式训练，未提交或推送代码。
