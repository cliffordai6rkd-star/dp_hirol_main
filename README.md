# Nero Force-Aware Diffusion Policy

本仓库面向 USB 插入等接触丰富机器人操作，研究如何把低频任务意图、当前接触状态和未来物理交互统一到分层控制流程中。

核心思想是区分两类力信息：

- 当前 FDP 使用七轴 `tau_ext` 作为高层策略输入，并生成接触敏感的末端位姿参考。
- 未来接触力由后续动力学模型预测，作为底层控制器需要显式跟踪的物理目标。

当前仓库已经完成高层 force-aware Diffusion Transformer、LeRobot v3 数据接入和训练工作区。PINN 未来力预测及 OSC-QP 硬件闭环仍属于后续开发阶段。


## 创建环境
```bash
conda env create -p /opt/lcx/conda/envs/dp -f conda_environment.yaml
```

## 训练

### 双相机纯视觉 DP（DINOv3 + Transformer）

纯 DP 分支只读取 LeRobot v3 中的：

```text
observation.images.wrist
observation.images.side
action.ee_pose
```

两路相机共享冻结的 DINOv3 ViT-B/16。每路 CLS 特征经拼接和可训练投影后，
作为 observation token 条件输入 8 层 diffusion Transformer；Transformer
预测 action diffusion noise。`wrench_ext` 和其他低维状态不进入该策略。

```bash
export PURE_DP_DATASET_PATH=/mnt/code/lcx/PINN/data/train_episode/wipe_board_lbv3
export DINOV3_MODEL_PATH=/mnt/code/lcx/model/dinov3-vitb16-pretrain-lvd1689m

python train.py \
  --config-dir=diffusion_policy/config \
  --config-name=train_pure_diffusion_transformer_workspace
```

该配置沿用 wipe-board v3 的 25 Hz 时间线：`n_obs_steps=2`、`horizon=9`，
逐帧 `[7]` 的 `action.ee_pose` 由 DataLoader 组成 `[9,7]` 训练目标。索引 0
是 prestep，策略对外返回预测索引 1 到 8，共 8 个未来动作。配置文件为：

```text
diffusion_policy/config/task/wipe_board_pure_dp.yaml
diffusion_policy/config/train_pure_diffusion_transformer_workspace.yaml
```

### dp_baseline（FDP 同步训练协议）

`dp_baseline` 使用与 FDP 相同的 Transformer、AMP/DDP、optimizer-step 和 EMA
训练协议，但观测只包含两路 RGB，不读取 `wrench_ext`。每次训练只需要设置数据集
和 DINOv3 路径：

```bash
export DP_BASELINE_DATASET_PATH=/opt/lcx/data/wipe_board_lbv3/
export DINOV3_MODEL_PATH=/opt/lcx/model/dinov3-vitb16-pretrain-lvd1689m/
python train.py \
  --config-dir=diffusion_policy/config \
  --config-name=train_dp_baseline
```

对应配置为：

```text
diffusion_policy/config/task/wipe_board_dp_baseline.yaml
diffusion_policy/config/train_dp_baseline.yaml
```

在仓库根目录执行以下完整命令：

```bash
export FDP_DATASET_PATH=/opt/lcx/data/wipe_board_lbv3/
export DINOV3_MODEL_PATH=/opt/lcx/model/dinov3-vitb16-pretrain-lvd1689m/

python -m diffusion_policy.workspace.train_force_aware_diffusion_workspace --config-name=train_force_aware_diffusion_workspace

  CUDA_VISIBLE_DEVICES=0,1 \
  /opt/lcx/conda/envs/dp/bin/torchrun \
    --standalone \
    --nproc_per_node=2 \
    -m diffusion_policy.workspace.train_force_aware_diffusion_workspace
```

`FDP_DATASET_PATH` 指向当前任务的数据集目录；`DINOV3_MODEL_PATH` 指向包含 `config.json` 和模型权重的本地 DINOv3 文件夹。默认使用 `cuda:0`、batch size 512、梯度累积 1 次、40000 optimizer steps 和在线 W&B 日志。

## 系统结构

```text
Nero 数据采集
  10 Hz wrist RGB                         100 Hz robot state / tau_ext
          |                                          |
          +-------------- 时间对齐与窗口化 ----------+
                             |
               image [To, 3, 192, 256]
               tau_ext [To, 4, 7]
                             |
         +-------------------+-------------------+
         |                                       |
  Frozen DINOv3 ViT-S/16                  GRU force encoder
  dense patch tokens                     8 sequential tokens
         |                                       |
         +---- Cross Attention: Q=force, K/V=image
                             |
                  To * 8 fused force tokens
                             |
                Diffusion Transformer expert
                             |
             future EE pose chunk [8, 7]
                             |
        position mean + sign-aligned quaternion mean
                             |
                 low-frequency pose target
                             |
          future force predictor / PINN (planned)
                             |
                  OSC-QP controller (planned)
```

高层策略严格只使用 wrist/side image 和七轴 `tau_ext`。`tau_ext` 先在物理单位下计算七轴绝对值之和；小于阈值的整条七维向量置零，达到阈值的向量原样归一化后送入 Diffusion Policy。关节位置、速度、加速度和其他力传感器字段不输入高层策略。

## 数据契约

当前 insert_usb 数据集位于：

```text
/home/rei/mnt/code/lcx/nero_ws/runs/insert_usb_lerobotv3_dp
```

数据集包含 121 个 episode、17,178 帧，统一为 10 FPS。

| 字段 | 单帧形状 | 对齐方式 | 用途 |
| --- | ---: | --- | --- |
| `observation.images.wrist` | `[192, 256, 3]` | 10 Hz 主时间线 | DINOv3 视觉输入 |
| `observation.wrench_ext` | `[8, 6]` | 历史窗口 | 高层接触状态输入 |
| `observation.joint` | `[8, 7]` | 历史窗口 | 后续 PINN 输入 |
| `observation.velocity` | `[8, 7]` | 历史窗口 | 后续 PINN 输入 |
| `observation.acceleration` | `[8, 7]` | 历史窗口 | 后续 PINN 输入 |
| `observation.torque` | `[8, 7]` | 历史窗口 | 后续 PINN 输入 |
| `action.ee_pose` | `[8, 7]` | 未来窗口 | DP 训练目标，格式为 xyz + xyzw |
| `action.joint` | `[8, 7]` | 未来窗口 | 备用动作定义与消融 |

DataLoader 使用 `action_layout: prechunked`。每个 observation anchor 直接读取已经转换好的未来 `[8,7]` 末端位姿 chunk，不再从后续图像帧重复拼接 action。

## Force-Aware Diffusion Transformer

### 视觉编码

默认 backbone 为：

```text
facebook/dinov3-vits16-pretrain-lvd1689m
```

DINOv3 在第一阶段完全冻结，并始终保持 eval 状态。相机图像在 LeRobot 中是 `[192,256,3]`，进入模型后为 `[3,192,256]`。

ViT-S/16 对该分辨率生成：

```text
192 / 16 = 12
256 / 16 = 16
12 * 16 = 192 patch tokens
```

模型通过 `interpolate_pos_encoding=True` 将预训练位置编码插值到 `12 x 16` 网格，不需要把图像拉伸或裁剪到 `224 x 224`。

### 力时序编码

每张图像对应一个 `[8,6]` wrench 历史窗口。默认使用 GRU 并保留全部 8 个 hidden state，因此每个 observation 产生 8 个 force query token，而不是压缩成一个 token。

可配置的时序编码器包括：

```text
gru | lstm | transformer | none
```

### Cross Attention

融合方向固定为：

```text
Q = force history tokens
K = DINOv3 patch tokens
V = DINOv3 patch tokens
```

cross-attention 输出保留 force residual。视觉信息只能通过该融合路径进入 diffusion expert，不存在独立的 global image token 旁路。

默认 `n_obs_steps=2`，所以 diffusion context 长度为：

```text
2 observations * 8 force tokens = 16 tokens
```

### Action Expert

Diffusion Transformer 对 `[9,7]` 末端位姿序列进行去噪：

- 训练 diffusion steps：100。
- 默认 DDIM 推理 steps：8。
- action 格式：`[x, y, z, qx, qy, qz, qw]`。
- 推理沿用 `start = To - 1`，当 `To=2` 时选择预测索引 1 到 8。
- `action` 保留完整 8 步轨迹，不再生成或返回 chunk 平均 pose。

`predict_action()` 返回：

| 键 | 形状 | 含义 |
| --- | ---: | --- |
| `action_pred` | `[B, 9, 7]` | 包含 prestep 的完整预测窗口 |
| `model_action_pred` | `[B, 9, 7]` | 模型动作空间中的预测；相对模式下为相对姿态 |
| `action` | `[B, 8, 7]` | 去掉 prestep 后的 8 步执行 chunk |

### 可配置相对姿态

默认继续训练绝对 `xyz + xyzw` 动作。若要训练相对当前锚点的姿态，在
`diffusion_policy/config/task/insert_usb_force_aware.yaml` 中设置：

```yaml
relative_pose_actions: true
```

也可以直接从命令行覆盖：

```bash
python -m diffusion_policy.workspace.train_force_aware_diffusion_workspace \
  --config-name=train_force_aware_diffusion_workspace \
  task.relative_pose_actions=true training.resume=false
```

相对模式保持 action 维度为 7：

```text
[delta_x, delta_y, delta_z, relative_qx, relative_qy, relative_qz, relative_qw]
```

平移增量位于机器人 base/world 坐标系；相对旋转按
`q_reference^-1 * q_target` 计算，不对四元数直接相减。预切片 action 的第
0 步是参考绝对位姿。Dataset 会在每个样本的
`obs["action_reference"]` 中提供该参考，Policy 在推理后将相对预测还原成
绝对位姿，因此 `action_pred` 和 `action` 的控制器接口保持不变；训练空间的
原始相对预测位于 `model_action_pred`。

实机或自定义 rollout 开启相对模式时，调用 `predict_action()` 前必须在
observation 中加入当前测量的末端绝对位姿：

```python
obs["action_reference"] = current_ee_pose  # [B, 7], xyz + xyzw
output = policy.predict_action(obs)
absolute_action_chunk = output["action"]  # [B, 8, 7]
```

切换绝对/相对动作表示后必须使用新的输出目录或关闭 `training.resume`，
不能续训另一种动作表示的 checkpoint。

## Contact-Aware Curriculum Masking

FDP 的 `tau_ext` 输入门控使用未归一化、物理单位下的七轴绝对值之和（L1 范数），当前阈值为 `0.6`；低于阈值时整条七维向量置零。门控阈值、范数类型和启用状态同时写入 force-aware checkpoint 的配置与模型 state dict。

当前全量数据统计下，默认阈值 `2 N` 会标记约 16.4% 的帧。这个值只是数据驱动的初始配置，正式实验应结合回放或接触标注确认。

仅在训练阶段执行视觉 mask；验证和推理始终使用完整图像。支持两种范围：

```text
current_observation | full_context
```

mask 概率由真实 optimizer update 驱动，而不是 DataLoader batch 数。支持：

```text
constant | linear | cosine | exponential | piecewise
```

默认使用 cosine schedule，从概率 1.0 逐渐降到 0.0，使模型从 force-dominant learning 过渡到 vision-force joint learning。

## 图像增强状态

DataLoader 已具备 resize、random crop、rotation 和 color jitter 接口，但当前 insert_usb 配置没有启用一般图像增强。

DINOv3 的 ImageNet 标准化、位置编码插值和接触课程 mask 不属于常规图像增强。腕部相机几何关系对插入任务很重要，因此在验证时序一致的数据增强前，不默认启用 crop 或 rotation。

## 本地环境

本项目不依赖 Docker。训练环境使用 Python 3.10，并与 `nero_ws` 对齐 NumPy、HDF5、OpenCV、SciPy、MuJoCo 和 Pinocchio 的版本范围。

```bash
conda env create -p /opt/lcx/conda/envs/dp -f conda_environment.yaml
conda activate /opt/lcx/conda/envs/dp
```

`conda_environment.yaml` 会同时安装 LeRobot 0.4.0、训练依赖和测试依赖。若环境已经创建，可在仓库根目录执行：

```bash
conda env update -p "$CONDA_PREFIX" -f conda_environment.yaml
```

`setup.py` 是项目元数据和依赖的唯一来源；`pyproject.toml` 只定义 PEP 517 构建后端和 pytest 配置。

## DINOv3 权重

推荐下载完整的 Hugging Face 格式目录：

```text
dinov3-vits16-pretrain-lvd1689m/
├── config.json
├── model.safetensors
└── preprocessor_config.json
```

从 Hugging Face 下载：

```bash
hf auth login
hf download facebook/dinov3-vits16-pretrain-lvd1689m \
  --local-dir /path/to/dinov3-vits16-pretrain-lvd1689m
```

训练前设置本地路径：

```bash
export DINOV3_MODEL_PATH=/path/to/dinov3-vits16-pretrain-lvd1689m
```

当前配置设置了 `dino_local_files_only: true`，训练过程中不会联网下载权重。Meta 原始 `.pth` 文件不能直接交给 `AutoModel.from_pretrained()`，需要 Hugging Face 格式的完整目录。

## 训练

### 默认训练

数据路径由环境变量 `FDP_DATASET_PATH` 提供。以下命令启动完整训练：

```bash
cd /home/rei/mnt/code/lcx/diffusion_policy
conda activate dp

export FDP_DATASET_PATH=/home/rei/mnt/code/lcx/nero_ws/runs/insert_usb_lerobotv3_dp
export DINOV3_MODEL_PATH=/path/to/dinov3-vits16-pretrain-lvd1689m

python -m diffusion_policy.workspace.train_force_aware_diffusion_workspace \
  --config-name=train_force_aware_diffusion_workspace
```

### 不使用在线 W&B

```bash
python -m diffusion_policy.workspace.train_force_aware_diffusion_workspace \
  --config-name=train_force_aware_diffusion_workspace \
  logging.mode=offline
```

### Debug smoke test

该命令只运行少量 train/validation batch，用于先检查本地 DINOv3 权重、DataLoader 和显存：

```bash
python -m diffusion_policy.workspace.train_force_aware_diffusion_workspace \
  --config-name=train_force_aware_diffusion_workspace \
  training.debug=true \
  logging.mode=offline \
  dataloader.num_workers=0 \
  dataloader.persistent_workers=false \
  dataloader.prefetch_factor=null \
  val_dataloader.num_workers=0 \
  val_dataloader.persistent_workers=false \
  val_dataloader.prefetch_factor=null
```

### Hydra 参数覆盖

常用参数覆盖示例：

```bash
python -m diffusion_policy.workspace.train_force_aware_diffusion_workspace \
  --config-name=train_force_aware_diffusion_workspace \
  policy.contact_threshold=2.5 \
  policy.image_mask_scope=current_observation \
  policy.mask_schedule.schedule_type=linear \
  policy.force_temporal_encoder=gru \
  dataloader.batch_size=4
```

主要配置文件：

```text
diffusion_policy/config/train_force_aware_diffusion_workspace.yaml
diffusion_policy/config/task/insert_usb_force_aware.yaml
```

训练工作区单独维护：

- `global_step`：完成的 micro-batch 数。
- `optimizer_step`：真正执行的 optimizer update 数。
- `epoch`：已完成的 epoch 数。

梯度累积结束后才更新 optimizer、LR scheduler、EMA 和课程 mask 调度。以上状态都写入 checkpoint，恢复训练不会重复已完成的 epoch。

## 测试

```bash
pytest -q tests/test_force_aware_diffusion.py
python tests/test_force_aware_workspace.py
python tests/test_hirol_lerobot_v3_dataset.py
```

测试覆盖：

- 接触判定使用原始物理单位。
- 各类 mask schedule 的端点和中间值。
- `current_observation` 与 `full_context`。
- GRU、LSTM、Transformer 和无时序编码器。
- DINOv3 完全冻结且无梯度。
- `To * 8` context token 契约。
- Diffusion loss、sampling 和 action shape。
- quaternion 符号对齐、单位化和均值 target。
- prechunked action 的 anchor 与 latency 行为。
- 梯度累积、optimizer step 和课程调度同步。

## 仓库结构

```text
diffusion_policy/
├── config/
│   ├── task/insert_usb_force_aware.yaml
│   └── train_force_aware_diffusion_workspace.yaml
├── dataset/hirol_lerobot_v3_dataset.py
├── model/
│   ├── diffusion/context_transformer_for_diffusion.py
│   └── vision/
│       ├── contact_curriculum.py
│       └── force_aware_obs_encoder.py
├── policy/force_aware_diffusion_transformer_policy.py
└── workspace/train_force_aware_diffusion_workspace.py

tests/                        data, model and workspace tests
conda_environment.yaml        local Python 3.10 environment
setup.py                      package metadata and dependencies
```

## 当前开发状态

| 模块 | 状态 |
| --- | --- |
| 10 Hz 均匀主时间线与历史低维窗口 | 已完成 |
| 未来 `[8,7]` EE pose action 转换 | 已完成 |
| LeRobot v3 prechunked DataLoader | 已完成 |
| Frozen DINOv3 dense patch encoder | 已完成，等待本地真实权重训练 |
| GRU force tokens 与 cross-attention | 已完成 |
| Contact-aware curriculum masking | 已完成 |
| Diffusion Transformer action expert | 已完成 |
| quaternion-safe chunk target | 已完成 |
| optimizer-step-aware workspace 与 resume | 已完成 |
| 一般图像增强 | 接口存在，当前未启用 |
| Nero 硬件在线 env runner | 未接入 |
| PINN 未来高频接触力预测 | 开发中 |
| OSC-QP 力矩闭环 | 规划中 |

第一阶段的目标是先验证：在相同 demonstration 数据上，force-aware cross-attention 和接触课程 mask 是否相对纯视觉策略改善接触阶段的动作稳定性。确认高层策略有效后，再接入未来力预测和底层控制闭环。
