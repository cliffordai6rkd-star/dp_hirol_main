# 服务器安装与 VS Code 训练

使用 `diffusion-policy:train` 镜像提供依赖，将服务器项目目录挂载到容器 `/workspace`。代码修改和训练输出保存在服务器磁盘，无需为源码变化重新构建镜像。当前训练链路是使用已观测 wrench 的接触教师与图像渲染学生。

## 迁移文件

- `diffusion-policy-train.tar.part-00` 至 `part-03`：训练镜像分片。
- `dp_main-current-source-data-model.tar.gz`：项目源码、配置、LeRobot v3 训练数据和本地 DINOv3 权重。
- `install-current.sh`、`SHA256SUMS-current`：安装脚本及校验值。

DINOv3 位于 `model/model/dinov3-vitb16-pretrain-lvd1689m`。训练需要 LeRobot 数据和该视觉骨干，不依赖原始 H5 或未来力预测器。历史输出和可重建缓存不需要随新训练迁移；默认从头训练当前 wrench 教师及学生。

## 安装与连接

目标服务器需要 Linux x86_64、支持 CUDA 12.8 的 NVIDIA 驱动、Docker Compose v2 和 NVIDIA Container Toolkit。准备本地 SSD 空间保存镜像、数据、缓存和训练输出。

在服务器终端进入迁移文件目录执行：

```bash
bash install-current.sh "$HOME/fdp_training"
cd "$HOME/fdp_training/dp_main"
bash docker/run_portable.sh check
bash docker/run_portable.sh dev
```

安装脚本先校验 SHA256，再导入镜像和解压项目。`check` 检查 GPU、通信、数据与 DINO；`dev` 创建后台容器 `fdp-dev`，不会启动正式训练。

VS Code 操作：

1. 本地安装 Remote-SSH 和 Dev Containers 扩展。
2. 用 Remote-SSH 连接服务器。
3. 执行 `Dev Containers: Attach to Running Container...`，选择 `fdp-dev`。
4. 打开 `/workspace`，Python 解释器选择 `/opt/venv/bin/python`。

容器中直接运行 `python`，无需 Conda。容器停止后，挂载目录中的源码和输出仍保留。其他项目可用 `DEV_CONTAINER_NAME=fdp-dev-other bash docker/run_portable.sh dev` 设置容器名。

## 训练教师和学生

在容器终端执行：

```bash
cd /workspace
python train_civi.py --gpus 0
```

四卡使用 `--gpus 0,1,2,3 --nproc-per-node 4`。教师读取截至当前观察帧的八个 wrench 采样，学生使用同一预处理生成接触监督。教师完成后按动作 MSE 选取权重，再自动开始渲染蒸馏。

默认输出为 `data/outputs/civi/current_wrench/teacher` 和 `data/outputs/civi/current_wrench/student`。另一次实验使用新目录：

```bash
python train_civi.py --gpus 0 \
  --teacher-override output_dir=data/outputs/civi_server/current_wrench/teacher \
  --student-override output_dir=data/outputs/civi_server/current_wrench/student
```

新方案需要使用当前 wrench 训练的教师与学生，历史 future-wrench 权重保持其原有输入语义。

长任务可以使用 `nohup`：

```bash
cd /workspace
mkdir -p data/outputs/civi_server
nohup python -u train_civi.py --gpus 0 \
  --teacher-override output_dir=data/outputs/civi_server/current_wrench/teacher \
  --student-override output_dir=data/outputs/civi_server/current_wrench/student \
  > data/outputs/civi_server/current_wrench_train.log 2>&1 < /dev/null &
tail -f data/outputs/civi_server/current_wrench_train.log
```

关闭 VS Code 不会停止后台容器；`docker stop fdp-dev` 会停止容器内的训练进程。
