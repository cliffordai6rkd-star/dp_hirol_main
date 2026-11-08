# LeRobot v3 → v2.1（供 openpi 使用）

从 PINN 仓库运行：

```bash
.conda-env/bin/python data_process/tool/lerobotv3_to_v21.py --dry-run
.conda-env/bin/python data_process/tool/lerobotv3_to_v21.py
```

依赖：`numpy pandas pyarrow`，以及支持 `libx264` 的 `ffmpeg`、`ffprobe`。
不依赖当前环境中的 LeRobot 版本。

默认输入为同级 `nero_ws/runs/demosrtate`，默认输出为
`nero_ws/runs/demostrate_v21`。也可以通过 `--input-dir` / `--output-dir`
传入其他路径。默认路径相对于脚本所在仓库计算，显式相对路径相对于当前工作目录。

三个数据集保持独立，名称和相对目录不变，不合并：

```text
demostrate_v21/
├── cuccumber_peeling_lerobotv3/
├── insert_usb_lerobotv3/
└── push_button_lerobotv3/
```

目录名虽然保留 `v3` 字样，但每个输出的 `meta/info.json` 都标记 `v2.1`。
每个子目录均包含自己的 `data/`、`videos/`、`meta/`。

转换保留所有原始数值列、任务、episode 编号、FPS、统计信息，并添加精确副本：

- `observation.state` ← `observation.ee_pose`
- `action` ← `action.ee_pose`

不会计算 delta action、改变坐标系、修改四元数、归一化或重采样数值数据。
视频按 episode 的起始时间精确解码，并编码为 H.264 CRF 0；这保留源视频解码后的
YUV 像素，不恢复源 AV1 编码已经损失的信息，文件体积通常会增大。
输出视频从零时间开始，逐个检查帧数和 FPS。

生成 v2.1 的 `episodes.jsonl`、`episodes_stats.jsonl`、`tasks.jsonl`，
数据和视频改为每个 episode 一个文件。兼容 openpi 锁定的 `datasets 3.6.0`
字段描述（将原始 primitive `List` 元数据转换为 `Sequence`）。

输出目录已存在时拒绝覆盖。单个数据集先写临时目录，成功后才发布；失败时清理
本次临时目录。已成功的其他数据集会保留。失败重试可对未完成的数据集单独指定
`--input-dir`，`--output-dir` 仍指向总输出目录。
`--dry-run` 只检查元数据、统计信息和目标路径，不执行视频解码或完整逐行检查。

当前支持本次数据的 v3.0 视频布局：每个 episode 的数据属于一个 parquet 文件，
每路视频属于一个 MP4 文件；输入 episode 编号连续且从零开始。
不支持跨数据文件的单 episode、图像文件模式、深度视频和非 yuv420p 视频；遇到这些
情况报错，不静默丢弃。

## openpi 的 action 字段如何指定

以当前同级 openpi 源码为准，需要两个设置配合使用：

```python
# 最终生成的 DataConfig 中：
action_sequence_keys=("action",)

# RepackTransform 的映射中：
{
    "state": "observation.state",
    "actions": "action",
}
```

`action_sequence_keys` 使用 **LeRobot 原始列名**。读取器用这些列生成
`model.action_horizon` 长度的未来序列。`RepackTransform` 再把它命名为
后续处理使用的 `actions`（复数）。转换后的 `action` 每帧维度是 7，读取后
得到 `[action_horizon, 7]`。

如果不使用新增别名，而直接读取原字段，则对应改为：

```python
action_sequence_keys=("action.ee_pose",)
# RepackTransform:
{"state": "observation.ee_pose", "actions": "action.ee_pose"}
```

例如自定义 `DataConfigFactory.create()` 返回配置时，可以这样设置
（此片段负责字段读取与映射，完整配置还需机器人输入/输出 transform）：

```python
return dataclasses.replace(
    self.create_base_config(assets_dirs, model_config),
    action_sequence_keys=("action",),
    prompt_from_task=True,
    repack_transforms=_transforms.Group(inputs=[
        _transforms.RepackTransform({
            "images": {
                "side": "observation.images.side",
                "wrist": "observation.images.wrist",
            },
            "state": "observation.state",
            "actions": "action",
        }),
    ]),
    data_transforms=your_robot_transforms,
    model_transforms=ModelTransformFactory()(model_config),
)
```

如果用 `SimpleDataConfig`，这些读取与映射选项放在它的
`base_config=DataConfig(...)` 中。如果用 `LeRobotAlohaDataConfig`，
`action_sequence_keys` 是该类自身的参数，它的 `create()` 会覆盖 base_config
中的同名值。

本数据使用末端位姿，不应直接套用 ALOHA 的默认关节差分和关节/夹爪适配。
完整训练配置仍需适合这台机器人的 transform，将两路图像转换为模型要求的
`image` / `image_mask`，并在推理输出端取回 7 维动作。这里不自动更改 openpi
训练代码或指定训练模型。

本地读取可在启动 openpi 前设置：

```bash
export HF_LEROBOT_HOME=/mnt/code/lcx/nero_ws/runs/demostrate_v21
```

然后分别使用 `repo_id="insert_usb_lerobotv3"` 等子目录名称。
三个数据集保持独立；这不代表一个 repo_id 会自动读取全部三个。

选定完整训练配置后，用 openpi 的 `scripts/compute_norm_stats.py --config-name ...`
生成该训练配置的归一化统计。数据集的 `meta/stats.json` 不替代这一步。

参考：
- https://huggingface.co/docs/lerobot/lerobot-dataset-v3
- https://github.com/huggingface/lerobot/blob/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5/lerobot/common/datasets/utils.py
- 本地 `openpi/src/openpi/training/config.py` 和 `data_loader.py`

## 转换后检查

转换脚本测试：

```bash
.conda-env/bin/python -m pytest -q tests/test_lerobotv3_to_v21.py
```

在 openpi 自己的 Python 环境内，可独立验证一个数据集：

```python
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

root = Path("/mnt/code/lcx/nero_ws/runs/demostrate_v21/insert_usb_lerobotv3")
dataset = LeRobotDataset(
    repo_id=root.name,
    root=root,
    delta_timestamps={"action": [i / 25 for i in range(50)]},
    video_backend="pyav",
)
sample = dataset[0]
assert sample["action"].shape == (50, 7)
assert sample["observation.state"].shape == (7,)
print(sample["task"])
```

这验证的是 v2.1 数据读取和动作序列；完整 openpi 模型训练还需要前述机器人
transform、模型配置和单独计算的归一化统计。
