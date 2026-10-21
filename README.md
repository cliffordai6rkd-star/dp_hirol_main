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
python train.py --config-dir=diffusion_policy/config --config-name=train_dp_inert_usb
```

单进程命令只使用一张 GPU；`training.distributed.enabled: true` 不会自动创建
进程。要进行真正的数据并行训练，使用 `torchrun`，不要用 `python train.py`：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export DP_BASELINE_DATASET_PATH=/opt/lcx/data/wipe_board_lbv3/
export DINOV3_MODEL_PATH=/opt/lcx/model/dinov3-vitb16-pretrain-lvd1689m/

torchrun --standalone --nproc_per_node=2 train.py \
  --config-name=train_dp_inert_usb
```

仓库也提供了等价的 `scripts/train_ddp.sh`。启动日志应分别出现
`[DDP] rank=0/2 ... device=cuda:0` 和 `rank=1/2 ... device=cuda:1`；
如果只出现单进程提示，说明启动方式仍然不是 DDP。

根目录的 `multi_gpu_train.sh` 保留给 Ray 多随机种子实验：它启动的是多个
相互独立的单卡任务，不会合并梯度，不能替代上述 DDP 命令。

对应配置为：

```text
diffusion_policy/config/task/insert_usb.yaml
diffusion_policy/config/train_dp_inert_usb.yaml
```

在仓库根目录执行以下完整命令：

```bash
export FDP_DATASET_PATH=/opt/lcx/data/wipe_board_lbv3/
export DINOV3_MODEL_PATH=/opt/lcx/model/dinov3-vitb16-pretrain-lvd1689m/

# Single GPU
python train.py --config-name=train_force_aware_diffusion_workspace

# DDP
CUDA_VISIBLE_DEVICES=0,1 \
  /opt/lcx/conda/envs/dp/bin/torchrun \
    --standalone \
    --nproc_per_node=2 \
    train.py \
    --config-name=train_force_aware_diffusion_workspace
```

`FDP_DATASET_PATH` 指向当前任务的数据集目录；`DINOV3_MODEL_PATH` 指向包含 `config.json` 和模型权重的本地 DINOv3 文件夹。默认使用全局 batch size 512、梯度累积 1 次、60000 optimizer steps 和在线 W&B 日志；DDP 时该 batch 会按 rank 均分。

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

默认融合方向为（可配置）：

```text
Q = 更新后的 wrench token
K = 更新后的 DINOv3 patch token
V = 更新后的 DINOv3 patch token
```

默认使用 `camera_attention_mode: joint`：side、wrist 分别经共享的 DINO
提取 patch token。wrench 经投影、位置/模态编码和 GRU 后，与两个相机的图像
token 拼接，先经过 `pre_cross_self_attn_layers: 1` 层联合 self-attention 编码块
（含残差连接和 FFN）。此模块在每个观测内独立运行，保持 token 顺序和数量：

```text
[side patches, wrist patches, GRU wrench tokens]
  → joint self-attention
  → 按原始 token 边界拆分
  → wrench 部分作为 Q，image 部分作为 K/V
  → cross-attention → residual + FFN → DiT
```

拆分是按位置切片，不是丢弃 token 或截断梯度。两种模态在 self-attention 中
已经交换信息，因此此时 wrench 位置的 token 已包含视觉信息，图像位置的 token
也已包含 wrench 信息。cross-attention 在两个相机的全部 patch 上共同归一化。
192×256、patch size 16 时，每个观测对应 192 + 192 = 384 个图像 K/V token。
联合 self-attention 的输入长度为 384 + 8 = 392，cross-attention 后仍为 8 个 wrench token。
层数可通过 `policy.pre_cross_self_attn_layers=2` 调整，设为 0 恢复
GRU 直接进入 cross-attention 的原结构；增加该模块后需要重新训练。
`per_camera` 模式仍可用于加载此前分别做注意力再融合的 checkpoint；该结构的
checkpoint 含独立的 `camera_fusion` 参数，切换为 joint 后应重新训练。

通过以下参数独立配置三个阶段，所有 force-aware 训练配置均支持命令行覆盖：

```yaml
policy:
  force_temporal_encoder: gru  # gru | lstm | transformer | none
  force_encoder_layers: 1
  pre_cross_self_attn_layers: 1  # >= 0；0 跳过图像+wrench联合 SA
  cross_attn_query: wrench      # wrench | image
  cross_attn_layers: 1          # >= 0；0 跳过 CA
```

`force_encoder_layers` 控制 wrench 时序编码深度，保留每个时刻的输出。
联合 SA 在 `[所有相机图像 tokens, wrench tokens]` 上运行，随后按原位置拆分。
多层 CA 逐层更新 Query 流，K/V 固定为前置编码器的输出，每层都有残差和 FFN。

| 设置 | 送入 DiT 的条件 | 两相机、两帧、每相机 192 patches、8 wrench tokens 时的长度 |
| --- | --- | --- |
| CA >= 1，`cross_attn_query: wrench` | wrench + 从图像读取的信息 | 16 |
| CA >= 1，`cross_attn_query: image` | 图像 + 从 wrench 读取的信息，保留视觉残差 | 768 |
| CA = 0，SA >= 1 | 联合 SA 后的全部图像+wrench tokens | 784 |
| CA = 0，SA = 0 | 独立编码的图像+wrench tokens 直接拼接 | 784 |

CA 为 0 时 Query 方向不参与计算，也不会创建 CA 的可训练参数；DiT 的条件位置
编码容量根据实际输出长度自动设置。原先的两层 SA baseline 还会跨观测时间做
联合注意力，这里的前置 SA 在每个观测内独立运行，两者并非完全相同的结构。

例如，跳过联合 SA，改为 Transformer 编码 wrench，再用图像作为 Query：

```bash
python train.py --config-name baseline_train/train_dp_self_crossattn_insert_usb \
  policy.pre_cross_self_attn_layers=0 \
  policy.cross_attn_query=image policy.cross_attn_layers=2 \
  policy.force_temporal_encoder=transformer policy.force_encoder_layers=2
```

保留默认的一层 wrench-query CA 时，旧 checkpoint 的参数名和计算保持兼容；
改变层数、方向或 backbone 后需要新建训练，不能把旧权重当作同构模型继续训练。
旧 cross-attention 空间热力图只适用于 wrench Query；image Query 的权重在
wrench 维度归一化，脚本会明确拒绝直接套用原热力图逻辑。

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

FDP 使用未归一化、物理单位下的七轴 `tau_ext` 绝对值之和（L1 范数）计算接触阶段。当前配置使用
`free_threshold=0.65` 和 `contact_threshold=1.5`：超过接触阈值进入接触，低于 free 阈值退出接触，中间区间保持锁存状态。

`future_tau_ext` 与未来 `[+40, +320] ms` 的 `future_wrench` 对齐，用于定位未来窗口中的第一个接触 token。如果当前仍是 free 但未来窗口会接触，接触前的 wrench token 使用第一个接触时刻的 6D wrench 向量填充，后续 token 保留原始未来 wrench；未来窗口没有接触时整段置零。

未来窗口使用 `anchor_timestamp_key: timing.camera_timestamp_ns` 和
`source_timestamp_key: timing.state_timestamp_ns`，按同一绝对时间轴查找最近的状态行。
普通动作序列继续使用 episode 内的相对 `timestamp`，间隔为 40 ms。
混用相对时间与绝对时间会报错；超出 episode 时间范围的未来窗口会被过滤。

USB 数据集配置 `episode_warmup_steps: 49`，排除每集固定的零力前缀。
过滤会检查每张观测对应的完整力历史与未来窗口，因此当前配置最早可用的
anchor 是 frame_index 58（1.16 s）；后续真实的零力样本仍可训练。
`observation_validity_key: timing.camera_valid` 同时过滤观测中包含无效相机行的
窗口，训练集和验证集采用相同规则。Normalizer 仅在有效训练窗口实际使用的
数据行上拟合。由错误时间采样训练得到的权重需要重新训练。

### USB self-cross-attention 的验证与 checkpoint 筛选

`baseline_train/train_dp_self_crossattn_insert_usb` 每 20,000 个 optimizer step
对完整验证集执行一次离线推理：使用验证样本的图像和 wrench 观测调用
`predict_action()`，完成 diffusion 去噪，再将输出动作与示范动作比较。
预测和目标按 `action_start_index` 对齐，仅比较实际输出的 8 步动作，
不将前面的观测参考帧计入动作 MSE。相对位姿模式下，两者均在模型的相对动作表示中比较。

```yaml
training:
  checkpoint_every_optimizer_steps: 20000
  max_val_steps: null
checkpoint:
  topk:
    monitor_key: val_action_mse_raw
    mode: min
    k: 5
```

`val_action_mse_raw` 是原始动作数值上的均方误差，是 checkpoint 筛选指标；
`val_action_mse` 是归一化动作 MSE，`val_loss` 是 diffusion 去噪损失。
验证优先使用 EMA 模型，固定采样随机种子，并关闭训练数据增强和课程遮蔽。
这些指标会记录到 WandB 和 `logs.json.txt`，`val_loss.png` 按 optimizer step 更新。
也会按 `training.val_every` 进行 epoch 验证；保存 checkpoint 前总会验证当前模型，
同一个 optimizer step 的结果只计算一次。最后不足 20,000 步的保存点同样会验证。

`checkpoints/` 保留动作 MSE 最低的 5 份模型，文件名包含训练步数与验证 MSE。
`best.ckpt` 指向当前最佳模型，`latest.ckpt` 用于恢复最新训练状态。
续训时恢复 Top-K 排名；较差候选更新 latest 时不会覆盖已保留的最佳模型。
其他训练配置仍使用各自的 `checkpoint.topk` 设置。
修改配置和训练代码会在下次启动训练时生效。

### USB token baselines

两个对照配置位于 `diffusion_policy/config/baseline_train/`，继承 USB 的数据筛选、
训练/验证划分、阶段采样、ee_pose 动作目标、DiT 和 EMA。
纯图像默认训练 200,000 步；selfattn 配置训练 400,000 步，训练和验证 batch 均为 64。
它们均关闭图像遮蔽课程：

| 配置 | 输入与融合 |
| --- | --- |
| `train_dp_dinobase_insert_usb` | 双相机全部 DINO patch token 直接作为 DiT context，模型前向只读取图像 |
| `train_dp_selfattn_insert_usb` | 未来 wrench 经线性投影和 GRU 后，与图像 patch token 拼接，再做 joint self-attention |

192×256 图像、双相机、两帧观测下，context 分别为 768 和 784 个 token。
selfattn baseline 保留每个观测的全部 8 个 GRU 输出 token，再与图像拼接。
GRU 在各个样本、各个观测的 wrench 序列内独立运行；不使用接触门控或接触前 wrench 填充。
当前默认 1 层 GRU、2 层 joint self-attention，分别通过以下参数设置：

```yaml
policy:
  force_temporal_encoder: gru
  force_encoder_layers: 1
  baseline_self_attn_layers: 2
```

例如命令行 `policy.force_encoder_layers=3` 改成 3 层 GRU。
`policy.force_temporal_encoder=none policy.force_encoder_layers=0` 恢复直接线性投影，
也用于加载没有 GRU 的旧 checkpoint。启用 GRU 或改变 GRU 层数后需要训练对应结构的模型。
纯图像 baseline 保留数据侧的阶段采样以使用相同样本分布，但模型预测不依赖力数据。
两个配置沿用 WandB、验证动作 MSE 和 output 下的 `val_loss.png`。

```bash
python train.py --config-name baseline_train/train_dp_dinobase_insert_usb
python train.py --config-name baseline_train/train_dp_selfattn_insert_usb
```

这两个 baseline 的注意力可视化脚本沿用 cross-attention 脚本的全部命令行参数。
分别选择对应模型的 checkpoint：

```bash
python diffusion_policy/scripts/visualize_wrench_image_self_attn_episode.py \
  --checkpoint /path/to/selfattn/checkpoints/model.ckpt \
  --episode 0 --camera both --attention-normalization global --save-frames --overwrite

python diffusion_policy/scripts/visualize_image_self_attn_episode.py \
  --checkpoint /path/to/dinobase/checkpoints/model.ckpt \
  --episode 0 --camera both --attention-normalization global --save-frames --overwrite
```

也可使用 `--checkpoint-dir` 自动选取训练步数最大的 checkpoint。
默认分别输出到训练目录的 `wrench_image_self_attn_episode_0.mp4` 和
`image_self_attn_episode_0.mp4`，PNG 保存到各自的 `_frames/` 目录。

联合 self-attention 脚本提取最后一个融合层中，最新观测的 wrench 查询到
同一观测各相机图像 patch 的权重；`--attention-query all` 平均所有 wrench
查询和注意力头，`last` 只使用最后一个 wrench 查询并平均各个头。softmax
原本覆盖全部相机、观测时刻和 wrench key，截取图像区域后不再重新归一化。

纯图像 baseline 的 patch 直接送入 DiT，没有额外的图像融合 self-attention 层，
所以图像脚本显示 DINOv3 最后一层真实的 patch-to-patch self-attention。
`all` 平均全部空间 patch 查询和注意力头，`last` 使用最后一个空间 patch
查询（不是 CLS）。展示时排除 CLS/register token，保留原始 softmax 权重。
冻结 DINO 时，这张图主要反映预训练视觉主干的注意力，不代表 DiT 的动作归因。

两种脚本均只展示最新观测，按原脚本的 episode 色阶、相机拼接和视频编码方式
输出；各模型的 softmax 范围不同，热力图颜色不能直接作为跨模型的贡献强度比较。

正在跑 USB 训练时，可在另一个终端排队运行，脚本会自动识别训练主进程并等待，
然后按 dinobase → selfattn 顺序启动新的训练：

```bash
bash scripts/train_usb_baselines_after_current.sh
# 只检查等待对象和命令，不训练：
bash scripts/train_usb_baselines_after_current.sh --dry-run
# 也可以指定 PID、解释器和两个任务共同采用的 Hydra override：
bash scripts/train_usb_baselines_after_current.sh --wait-pid 12345 \
  --python /path/to/env/bin/python -- training.device=cuda:0
```

队列日志、`status.txt` 和 `exit_codes.tsv` 位于 `data/outputs/baseline_queue/`；
模型输出按配置名分别保存在普通 Hydra output 目录。等待逻辑检查进程启动时间，
避免 PID 复用；文件锁避免重复启动同一个队列。第一个 baseline 若失败，仍会尝试
第二个并记录各自退出码。排队脚本不会向正在等待的原训练进程发送信号。

### 通用 checkpoint 注意力可视化

推荐使用统一入口 `visualize_attention_episode.py`。脚本从 checkpoint 中的
config 重建网络，自动识别 Query 方向、力编码器、SA/CA 层数和多相机布局，
默认读取 EMA 权重。原来的 `visualize_image_query_cross_attention_episode.py`
命令仍可使用，提供同一套通用参数。

```bash
python diffusion_policy/scripts/visualize_attention_episode.py \
  --checkpoint data/outputs/2026.10.20/21.54.27_train_dp_image_query_lstm_insert_usb_insert_usb_force_aware/checkpoints/best.ckpt \
  --episode 0 --camera both --save-frames --overwrite
```

`--mode auto` 默认依次选择 Cross-Attention、融合 Self-Attention、DINO 主干
Self-Attention。也可用 `--mode cross-attn`（或 `--ca`）、
`--mode self-attn`（或 `--sa`）、`--mode dino` 指定目标。
`--layer-index`（或 `--layer`）从 0 开始，`-1` 表示所选模式的最后一层。
没有融合 SA 的 CA 模型指定 `self-attn` 时会报错，查看视觉主干应明确使用
`dino`。纯 DINO baseline 的 `self-attn` 则作为 `dino` 的别名。

```bash
# 仅在 CPU 上读取结构和列出可用模式、层号；不需要 --episode，也不回放数据。
python diffusion_policy/scripts/visualize_attention_episode.py \
  --checkpoint /path/to/checkpoints/model.ckpt --list-layers

# 查看有融合 SA 的 checkpoint 的第一层。
python diffusion_policy/scripts/visualize_attention_episode.py \
  --checkpoint /path/to/selfattn/checkpoints/model.ckpt \
  --episode 0 --sa --layer 0 --camera both --overwrite
```

各模式均展示最新观测，保留原始 softmax 权重，不对截取的图像区域重新归一化：

- 图像 Query / wrench K/V 的 CA：每格保留一个 wrench Key 的空间注意力列，
  只平均 heads；8 个 wrench 状态排成 2×4 网格。另输出 `‖ΔV‖₂` 视频，
  取 CA 输出投影之后、图像残差及 FFN 之前的更新幅度。
- Wrench Query / 图像 K/V 的 CA：每格保留一个 wrench Query 到图像 patch
  的权重行，只平均 heads；此方向不输出图像位置的更新幅度视频。
- 图像与 wrench 联合 SA：提取 wrench Query 行、图像 Key 列的 `S_WI`
  子矩阵，保留各 wrench 位置。旧 baseline 中跨观测的 softmax 范围保持不变。
- 纯图像融合 SA 或 DINO SA：平均空间 patch Queries 和 heads，得到一组
  图像 Key 热图；DINO 图中排除 CLS/register 行列，但保留原 softmax 分母。

`--attention-query all` 默认保留全部 wrench 位置；`last` 仅保留最后一个。
对于纯图像 SA，这两个值分别表示平均全部 patch Queries、仅取最后一个
patch Query。LSTM/GRU 状态包含序列上下文，时间标签对应编码位置。
这里只捕获指定的一层，不做跨层 rollout，也不表示最终动作的因果归因。

默认按**本次选中的全部 episode 帧、全部模型相机和全部 wrench 面板**共同
计算注意力的第 1–99 百分位，所有帧共用这个色标；仅展示一个相机时仍使用
全部相机定标。`--attention-percentiles 1 99` 可修改分位范围，
`--attention-normalization fixed` 可恢复 0–1。分位色标只拉伸显示颜色并截断
极值，不改变保存的原始概率；色条显示实际概率范围。更新幅度另用整集统一的
1–99 百分位。不同模式的 softmax 范围不同，颜色不能直接比较跨模型贡献。

默认输出到训练目录，文件前缀包含实际模式、层号和 Query，例如
`attention_episode_0_cross_attn_layer_0_image_query_attention.mp4`，同前缀的
`_ca_update.mp4`（仅图像 Query CA）、`_raw.npz` 和 `_metadata.json`。
`--save-frames` 将 PNG 保存到各视频对应的 `_frames/` 目录；`--output`
可指定视频及数据文件共同的前缀。支持原有的 `--checkpoint-dir`、
`--dataset-path`、`--video-backend`、`--fps`、`--video-codec`、`--stride`、
`--max-frames`、`--alpha`、`--seed`、`--device` 和 `--overwrite`。

### 最终动作对图像区域的敏感度

`visualize_action_image_attribution_episode.py` 从最终执行的整段 `action`
向输入 RGB 像素反传，包含 DINO、融合模块、DiT、完整串行去噪链和动作后处理。
它从 checkpoint config 重建网络，支持当前 force-aware 和图像 baseline，
无需指定 Query 方向或注意力层。

```bash
python diffusion_policy/scripts/visualize_action_image_attribution_episode.py \
  --checkpoint data/outputs/2026.10.20/21.54.27_train_dp_image_query_lstm_insert_usb_insert_usb_force_aware/checkpoints/best.ckpt \
  --episode 0 --camera both --save-frames --overwrite
```

令 `z` 为最终执行动作按 checkpoint 动作 normalizer 缩放后的所有坐标，
`D` 为坐标数量，区域热图定义为：

```text
H(patch) = sqrt(mean_output(mean_pixel_in_patch(sum_RGB((dz/dpixel)^2))))
```

先对各输出坐标的梯度平方求和，避免对动作求平均后产生正负抵消；
再在 DINO patch 对应的输入像素区域内取 RMS。保留模型的真实采样步数、
clipping、四元数归一化及相对姿态还原。每帧重置相同采样 seed，其他观测
保持常量；这里没有跨去噪步的注意力平均。

默认计算精确 Jacobian：例如 8 步、7 维动作需要 56 次反传。
可先加 `--max-frames 1` 检查结果。若需要加速，可显式添加
`--gradient-probes 8`，用 8 个固定 Rademacher 随机投影估计同一完整链的
Jacobian 能量；这属于近似计算，数值和元数据会注明。完整去噪计算图需要
保留到反传，显存和耗时会高于原来的单层注意力脚本。

CUDA GPU 可以再加 `--autocast bfloat16`。它只对可微前向启用 bfloat16，
梯度累积仍使用 float32；当前 RTX GPU 上通常更快、更省显存，但数值会与
默认 float32 略有差异。标准 KV cache 没有启用：动作 token 每个去噪步都在
变化，条件 memory 还包含随 timestep 变化的时间 token，动作 self-attention
也是非因果的，跨步缓存会改变原模型计算。

一个 MP4 中按观测时间分行、相机分列，每张热图覆盖其对应的输入原图。
`--observation last` 仅显示最新观测，原始数据仍保留全部观测时刻。
全部选中帧、全部模型相机和观测时刻共用 1%–99% 色标，可通过
`--percentiles 1 99` 修改。红色表示局部像素变化更容易影响最终执行动作，
不表示注意力概率、正负贡献或因果重要性；饱和区域可能出现零梯度。
色标仅影响显示，原始敏感度不重新归一化。

输出默认位于训练目录：`action_image_attribution_episode_0.mp4`、
`_raw.npz` 和 `_metadata.json`；`--save-frames` 另保存 PNG。
NPZ 包含 `anchors`、最终 `action`、`scaled_action`，以及按 metadata
相机顺序保存的 `camera_N_sensitivity[frame, observation, patch_row, patch_col]`。
`--checkpoint-dir`、`--dataset-path`、`--video-backend`、`--fps`、
`--video-codec`、`--stride`、`--max-frames`、`--alpha`、`--seed`、
`--device`、`--output`、`--overwrite` 与现有脚本用法一致。

### FIRST 阶段重采样

`train_dp_insert_usb_force_aware` 默认启用三阶段训练重采样：窗口内的
`tau_ext` 历史或 `future_tau_ext` 只要出现接触 token，就标记为 contact-relevant；
其余位于 episode 首次接触前 1 秒内的 anchor 标记为 pre-contact，其他样本标记为
free。训练采样权重默认为：

```text
free : pre-contact : contact = 1 : 5 : 2
```

权重按阶段池归一化后进行有放回采样，因此该比例表示期望采样概率；验证集仍保持原始分布。

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

当前 RTX 5090/Blackwell 训练栈固定为 Python 3.10、PyTorch 2.7.1 和
TorchVision 0.22.1 的 CUDA 12.8 wheel。视频解码使用 TorchCodec 0.5 的
CUDA 12.8 wheel，并额外安装 `nvidia-npp-cu12` 提供
`libnppicc.so.12`。如果环境是手动创建的，请从仓库根目录补齐这组 CUDA 依赖：

```bash
python -m pip install --upgrade \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  torchcodec==0.5+cu128 nvidia-npp-cu12==12.4.1.87
python -m pip install --upgrade "wandb>=0.20,<0.22"

# Make libnppicc.so.12 visible to TorchCodec in pip/Conda environments.
NPP_LIB=$(find "$CONDA_PREFIX" -type f -name 'libnppicc.so*' -printf '%h\n' | head -n 1)
export LD_LIBRARY_PATH="$NPP_LIB:${LD_LIBRARY_PATH:-}"
```

`setup.py` 声明跨安装方式的包依赖范围；CUDA 12.8 训练栈的可复现版本位于
`requirements/uv-common-py310-lerobot.txt`、`conda_environment.yaml` 和
`docker/Dockerfile`。`pyproject.toml` 只定义 PEP 517 构建后端和 pytest 配置。

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

tar -xzf /path/to/diffusion-policy-bundle.tar.gz \
    -C /path/to/diffusion_policy

docker load -i docker/artifacts/diffusion-policy-train.tar

docker compose -f docker/docker_compose.yaml up -d trainer


CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node=2 \
  train.py \
  --config-dir=diffusion_policy/config \
  --config-name=train_dp_insert_usb_force_aware \
  training.distributed.enabled=true

mkdir -p docker/artifacts
docker save -o docker/artifacts/diffusion-policy-train.tar diffusion-policy:train

tar --exclude='.git' -czf ../diffusion-policy-bundle.tar.gz .


ATTN_CKPT="data/outputs/2026.10.20/21.54.27_train_dp_image_query_lstm_insert_usb_insert_usb_force_aware/checkpoints/best.ckpt"

# 查看网络结构和可选层
python diffusion_policy/scripts/visualize_attention_episode.py \
  --checkpoint "$ATTN_CKPT" --list-layers

# 导出指定层
python diffusion_policy/scripts/visualize_attention_episode.py \
  --checkpoint "$ATTN_CKPT" \
  --episode 0 --camera both \
  --mode cross-attn --layer 0 \
  --attention-percentiles 1 99 \
  --save-frames --overwrite
参数	含义
--mode auto	默认：优先 CA，其次融合 SA，最后 DINO
--mode cross-attn 或 --ca	查看 CA
--mode self-attn 或 --sa	查看融合模块的 SA
--mode dino	查看 DINO 主干内部 SA
--layer 0 / --layer -1	第一层 / 最后一层
