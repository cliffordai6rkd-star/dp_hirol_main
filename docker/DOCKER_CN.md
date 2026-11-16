# civi Docker 训练

镜像提供 Python 3.10 / CUDA 12.8 / LeRobot v3 环境；镜像标签保留 `diffusion-policy:train`，兼容已有环境包。

```bash
cd /path/to/civi
docker compose -f docker/docker_compose.yaml build
docker compose -f docker/docker_compose.yaml run --rm trainer
```

默认命令为 `python train.py`：board 教师 batch 32 / 150,000 步，随后用最终教师蒸馏 batch 16 学生，包含新的阶段子网络。`CIVI_DATASET_PATH` 可指定容器内数据集路径。数据与模型目录分别挂载到 `/workspace/data` 和 `/workspace/model`。

已有运行环境镜像时，使用源码挂载方式：

```bash
bash docker/run_portable.sh check
TRAIN_GPUS=0 NPROC_PER_NODE=1 bash docker/run_portable.sh all
```

`check` 检查 GPU、数据与本地 DINO；`all` 顺序训练教师与学生。可向脚本追加 `train_pipeline.py` 参数，例如 `--dp-override dataloader.batch_size=32`。详见[服务器开发](SERVER_VSCODE_CN.md)与[迁移说明](PORTABLE_CN.md)。

教师与学生的配置、损失和阶段规则见仓库 [README](../README.md)。默认单卡训练；多卡测试命令仍可显式指定 `NPROC_PER_NODE`。
