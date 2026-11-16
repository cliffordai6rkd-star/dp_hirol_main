# 服务器开发

在服务器的 civi 目录使用已有 `diffusion-policy:train` 镜像。源码与训练输出通过 `/workspace` 挂载保存。

```bash
cd /path/to/civi
bash docker/run_portable.sh check
bash docker/run_portable.sh dev
```

VS Code 安装 Remote-SSH 和 Dev Containers，连接服务器后选择 `Dev Containers: Attach to Running Container...`，附加 `fdp-dev`。打开 `/workspace`，解释器选择 `/opt/venv/bin/python`。

```bash
cd /workspace
python train_pipeline.py --gpus 0
```

无参数入口使用当前 board 教师和包含阶段预测的学生，按最终教师 checkpoint 衔接。后台启动可用：

```bash
nohup python -u train_pipeline.py --gpus 0 > /workspace/data/civi_train.log 2>&1 &
```

更换数据、已有教师蒸馏和训练预算参数见 [README](../README.md)。
