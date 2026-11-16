# 环境与源码迁移

当前项目目录名为 `civi`。将源码、需要的数据集和本地 DINO 模型放在同一目录，使用 `docker/portable_compose.yaml` 将它挂载到 `/workspace`。

```bash
docker load -i /path/to/diffusion-policy-train.tar
cd /path/to/civi
bash docker/run_portable.sh check
TRAIN_GPUS=0 NPROC_PER_NODE=1 bash docker/run_portable.sh all
```

历史 U 盘包的镜像分片和源码档案名继续兼容原名：`diffusion-policy-train.tar.part-*`、`dp_main-current-source-data-model.tar.gz` 或 `dp_main-source-data-model.tar.gz`。相应安装脚本会将源码解压到目标目录的 `civi/`。档案内代码保留其打包时版本；迁移当前方法时应复制当前 civi 源码与配置。

安装历史包后，进入 `civi/`，按 [Docker 说明](DOCKER_CN.md)检查依赖。环境仍须满足 Linux x86_64、NVIDIA 驱动、Docker Compose v2 和 NVIDIA Container Toolkit；无需改动数据格式。
