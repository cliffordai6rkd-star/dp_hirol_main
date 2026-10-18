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

### USB token baselines

两个对照配置位于 `diffusion_policy/config/baseline_train/`，继承 USB 的数据筛选、
训练/验证划分、阶段采样、ee_pose 动作目标、DiT、EMA 和 200,000 次 optimizer update。
它们均关闭图像遮蔽课程：

| 配置 | 输入与融合 |
| --- | --- |
| `train_dp_dinobase_insert_usb` | 双相机全部 DINO patch token 直接作为 DiT context，模型前向只读取图像 |
| `train_dp_selfattn_insert_usb` | 图像 patch token 与直接线性投影的未来 wrench token 拼接，再做 joint self-attention |

192×256 图像、双相机、两帧观测下，context 分别为 768 和 784 个 token。
selfattn baseline 不使用 GRU、接触门控或接触前 wrench 填充。
纯图像 baseline 保留数据侧的阶段采样以使用相同样本分布，但模型预测不依赖力数据。
两个配置沿用 WandB、验证动作 MSE 和 output 下的 `val_loss.png`。

```bash
python train.py --config-name baseline_train/train_dp_dinobase_insert_usb
python train.py --config-name baseline_train/train_dp_selfattn_insert_usb
```

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
