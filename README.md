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

### DP 与蒸馏统一入口

在仓库根目录激活训练环境后，使用 `train_pipeline.py` 选择运行阶段：

```bash
conda activate fdp

# 单独训练 DP：默认使用本次 20 Hz、LSTM 2 层、cross-attn 1 层配置。
python train_pipeline.py dp --gpus 0

# 默认先训练 FDP，再自动把教师检查点交给 RenderNet 蒸馏。
python train_pipeline.py --gpus 0

# 已有教师时只蒸馏；该命令会立即读取此目录当前保存的检查点。
python train_pipeline.py distill \
  --teacher-run data/outputs/2026.10.31/14.44.46_sim_force_aware_dp_train_20hz_lstm2_libero_open_top_drawer_put_bowl_20hz_force_aware \
  --gpus 0

# 只打印两个阶段的命令，不创建目录、不启动训练。
python train_pipeline.py --dry-run
```

DP 默认参数为全部 50 条训练、batch 32、100,000 个优化步，500 步 warmup 后余弦退火，每 20,000 步保存并保留最近 3 个检查点。蒸馏使用 `train_distill_render_net.yaml` 的独立训练参数（batch 32、最多 1000 epoch），继承教师的数据与 normalizer；教师没有验证集时，蒸馏也使用全部演示，关闭验证、早停及 best 选择，每个 epoch 保存 `latest.ckpt`。

`--teacher-run` 默认优先使用 `best.ckpt`，没有则使用 `latest.ckpt`；可用 `--teacher-selection latest` 明确选择最新权重，或用 `--teacher-checkpoint /path/to/model.ckpt` 精确指定。`all` 模式始终使用刚完成的 DP 阶段作为教师，DP 阶段失败时不会启动蒸馏。

默认采用两阶段流程：FDP 训练结束后冻结完整教师，第二阶段只更新 RenderNet。
光团与 wrench 的对应关系由网络通过教师去噪输出监督学习，不预设力分量到颜色、
方向或亮度的规则；`all` 按顺序启动两个阶段。

两个阶段的配置及 Hydra 参数分别传入，支持自定义配置路径和多卡运行：

```bash
python train_pipeline.py all \
  --dp-config diffusion_policy/config/sim_force_aware_dp_train_20hz_lstm2.yaml \
  --distill-config diffusion_policy/config/train_distill_render_net.yaml \
  --dp-override training.num_optimizer_steps=100000 \
  --distill-override training.num_epochs=500 \
  --distill-override dataloader.batch_size=32 \
  --output-dir data/outputs/my_dp_distill_run

python train_pipeline.py all --gpus 0,1,2,3 --nproc-per-node 4
```

每个 `--dp-override` / `--distill-override` 后跟一个 `key=value`，可重复使用；多卡由入口内部启动 `torchrun`，batch 按全局大小分配。省略阶段参数等同于 `all`，仍支持 `dp` 或 `distill` 单独运行。
输出集中在一个新目录，教师和学生分别保存配置、`train.log`、指标和检查点：

```text
data/outputs/<日期>/<时间>_dp_distill_pipeline/
├── pipeline.json
├── teacher/
│   ├── checkpoints/
│   └── train.log
└── student/
    ├── checkpoints/
    └── train.log
```

默认自动选择 `teacher/checkpoints/best.ckpt`，没有则使用 `latest.ckpt`，并传给学生。
`--output-dir` 可以指定共同的输出目录；已有目录会报错，以免覆盖实验。

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

### RenderNet 光团与去噪输出蒸馏

入口为 `train_distill_render_net.py`，配置为
`diffusion_policy/config/train_distill_render_net.yaml`。默认教师是 image-query
LSTM checkpoint 的 `best.ckpt`，使用其配置实例化 policy，优先恢复 EMA 权重，
沿用保存的 normalizer 和数据集的 episode 训练／验证划分。

教师始终 eval 且完全冻结，教师 context 来自
`teacher_policy._encode_context(obs_dict, apply_curriculum_mask=False).detach()`。
默认 `distillation.objective=denoiser_mse`：将同一演示动作加噪得到 `a_t`，
教师融合 context 和学生渲染图 context 使用**同一个冻结 DiT、相同的 `a_t` 和 `t`**，
最小化二者去噪输出的 MSE。支持教师的 `epsilon`、`sample` 和 `v_prediction`。
每个训练 batch 随机采样时间步，无需为这个损失运行完整去噪链。
只更新 RenderNet；学生前向保留穿过冻结 DINO 和 DiT 的输入梯度。
`distillation.token_loss_weight`、`demo_loss_weight` 默认均为 0，分别控制可选的
context MSE 和演示动作扩散损失；token MSE 始终记录为诊断指标。
验证用固定随机种子重复采样探针，best 与早停只看去噪输出 MSE，不包含区域或辅助损失。
`distillation.objective=token_mse` 可恢复旧目标；缺少 distillation 的旧配置仍按旧目标恢复。
新训练默认使用 `render_net.architecture_version=2`。wrench 经过 projection、
时间／观测位置编码和模态编码后进入 LSTM，保留全部时间步输出
`[B*To,K,D]`，不再将窗口压成最后一个 hidden。默认
`initialize_wrench_from_teacher=true`，从所选教师复制上述全部编码参数；
维度、LSTM 层数、窗口长度、观测长度与相机键自动按教师解析并写入保存的配置。
复制后的学生编码器独立训练、独立保存，不调用教师的 wrench 网络。

图像由轻量 CNN 构建多尺度特征，在默认 1/16 分辨率处加入二维位置编码和相机
身份 embedding，以图像 patch 为 Q、完整 LSTM 序列为 K/V 执行残差 cross-attention。
`render_net.backbone=transformer`（默认）还在每个融合块中执行图像 self-attention；
`unet` 使用同一 CNN 编码器和 wrench cross-attention，但省略图像 self-attention。
各相机共享参数，以独立相机 embedding 区分视角。
默认 `rendering_mode=gaussian_glow`：移除全分辨率 RGB 解码器，由可学习 query
读取融合后的空间特征，预测 `glow.num_blobs=8` 个椭圆高斯的中心、半径、角度、
颜色和强度；区域头在低分辨率预测软 mask，再上采样到 RGB 尺寸。
所有光团参数都由 RGB 与 wrench 的融合特征预测；不手工规定某个力／力矩分量
对应哪种颜色或亮度，也不添加 wrench 重建损失。教师去噪行为监督网络学习对任务
有用的力信息表达。复制初始化的学生 wrench LSTM 同样参与训练。
渲染为 `1 - (1 - image) * exp(-residual_scale * predicted_region * glow_field)`，
光团只增亮，输出有界，`residual_scale` 在这里表示曝光强度。
半径受 `glow.sigma_min/sigma_max` 约束，初始化为弱光，避免几何参数因全零光团失去梯度。
`transformer.patch_size` 表示 CNN 下采样倍率，
支持不小于 2 的 2 次幂；维度、层数、head 数仍可配置，并处理非整除图像尺寸。
新训练默认 `render_net.conv_block=bottleneck`：保留原图 stem，编码层在较窄通道上
执行空间卷积。`rendering_mode=residual` 可选回旧的逐像素 RGB 残差和 skip 解码器；
其输出为 `clamp(image + residual_scale * predicted_region * tanh(residual), 0, 1)`，
区域关闭时 `predicted_region=1`，RGB 输出层零初始化。
缺少 rendering_mode 的旧配置仍使用 residual。`conv_block=standard` 可选回 v2 原始卷积块；缺少该字段的 v2
checkpoint 也按 standard 恢复。两种卷积结构的权重不互换，需分别训练。
增强图像经过教师冻结的 DINOv3、image projection、位置／相机／模态 embedding
和输出 LayerNorm，得到相同布局的 student context；学生路径不执行教师的
wrench encoder 或融合 cross-attention，DINO 后也不添加学生专用的 token 融合层
或 adapter。去噪输出蒸馏再将此 context 送入共享冻结 DiT。
光团骨干和 residual 骨干权重不互换，切换模式需新开训练。

教师与 RenderNet 共享同一套 wrench 预处理：按 `tau_ext` 判定接触、沿
`future_tau_ext` 进行未来接触状态推演、用首个接触 wrench 填充未来窗口的自由
前缀，再使用教师 checkpoint 的 normalizer。整个未来窗口无接触时，归一化后的
wrench 严格置零。学生训练直接调用教师的 `prepare_context_inputs`，不再直接
归一化原始 `future_wrench`。`tau_ext/future_tau_ext` 仅用于同教师一致的门控，
不作为额外的可学习输入。教师的融合 token 监督目标保持不变。
零 wrench 不强制渲染图等于原图：教师在零 wrench 下仍执行融合、FFN 与
LayerNorm，学生需要通过 RGB 同时逼近这部分基础变换和力引起的变化。

高注意力区域渲染由 `render_net.high_attention_render.enabled` 开关控制，默认开启。
开启时，用冻结教师的 DiT→context→融合输入图像 patch 的 attention rollout 作为
区域监督，复用当前注意力视频的路径：平均 heads、包含残差路由、对每次去噪
的条件注意力质量归一化后，再对全部去噪步求均值，action query 为执行的动作段。
原始教师 token 通过独立的无 hook 前向计算；注意力捕获不会替换 token 监督目标。
它是 DINO patch 层面的空间参考，不能解释为像素因果贡献。image-query cross-attention
的每行对 wrench 求和恒为 1，因此不使用这个和来选取图像位置。

`percentiles: [80, 99]` 表示对**每个相机、每个观测窗口内的每帧**独立计算 patch
权重分位数：低于 p80 的区域权重为 0，高于 p99 为 1，中间线性过渡，然后上采样
到图像尺寸。支持任意 `0 <= lower < upper <= 100` 的小数，如 `[2,98]`、
`[10,90]`、`[82.5,97.3]`。这两个数是渐变的端点，超过上界仍保持权重 1。
若分位数重合则退回实际 min/max；若整张图注意力相同则使用全图权重 1。
这里按帧生成训练目标，与视频整段共享显示色阶的统计范围不同。

学生区域预测头用教师上述软区域做 MSE 监督；训练和推理都使用**学生预测的区域**
调制光团或残差，不直接用教师 mask 替换学生输出。
预测区域是教师区域的近似，不能保证与教师阈值区域完全相同。默认总损失为
`denoiser_mse + mask_loss_weight * mask_mse`，另可开启上述辅助目标。
日志中 `train_mse/val_mse` 表示所选主目标，`train_token_mse` 单独记录 context 差异。
关闭区域开关时不建立区域头、不计算区域损失；去噪输出蒸馏仍需运行 DiT。
开启区域监督时，教师额外运行完整去噪链获取注意力。
默认先为整个训练集和验证集生成缓存，再开始训练，后续 epoch 直接读取监督。
验证与初始化评估使用所选主目标，学生仍使用自己预测的区域渲染图像。
注意力采集生成的最终动作被丢弃；去噪监督在每个随机 `a_t,t` 上在线计算，
不缓存为固定噪声标签。学生路径保持仅通过 RenderNet 接收 wrench。
只生成渲染图时无需教师融合网络、DINO 或 DiT；完整 RenderDP 动作推理仍需要冻结视觉路径和 DiT。

新的主目标增加了教师 DiT 前向和学生 DiT 前后向；移除 RGB 解码器能减少部分计算，
但不能据此保证每步更快。普通 FDP 的冻结 DINO 通常只做前向，RenderDP 则必须保留
DINO 的输入反向，因此可训练参数较少也不等于单步成本更低。新目标是否减少收敛步数需实测。
当前入口实现的是**冻结教师后蒸馏**，未实现 FDP/RenderDP 联合训练。
联合训练可采用 FDP 更新融合编码器和 DiT、RenderDP 交替更新渲染器的方式；
教师变化时 context／attention 缓存也会失效，需要在线计算或刷新，并考虑 warmup 或 EMA 教师。

`teacher_cache.enabled=true` 默认将教师 context token 以 **float16** 保存到
`data/cache/render_net_teacher/<指纹>/{train,validation}/`，原始 patch 注意力百分比
仍保存为 float32。token 在磁盘、CPU 和 CPU→GPU 传输时保持 FP16，计算 MSE 时
与学生 token 一起转为 FP32。FP16 存储会引入少量量化误差；可用
`teacher_cache.dtype=float32` 保留原精度，并复用原有 float32 缓存。切换存储精度
使用独立缓存目录。
默认 `teacher_cache.cleanup_on_exit=true`：正常训练完成、早停、Python 异常、
Ctrl+C、SIGTERM 或 SIGHUP 退出时，删除本次使用的教师缓存（包括生成到一半的缓存），
先停止 DataLoader worker、关闭 memory map，再释放磁盘文件。仅清理当前指纹目录，
不清理其他教师缓存、RGB 缓存、数据集或训练检查点。多卡／多个训练共用缓存时，
最后一个退出且开启自动清理的使用者执行删除，避免提前删除其他进程仍在读取的文件；
目录旁极小的 `.usage.lock` 文件保留，用于跨进程同步。
SIGKILL（`kill -9`）、系统断电无法执行进程内清理；已有 `nohup` 的 SIGHUP 忽略行为保留。
如需保留缓存以便复用或断点生成，设置 `teacher_cache.cleanup_on_exit=false`。
缓存按原数据集样本索引读取，训练 shuffle 不改变对应关系；有保留的未完成缓存时，
会接着未完成的样本生成。更换教师、数据集或 AMP 设置会使用新的缓存；只修改渲染分位数或
学生主干可直接复用。生成缓存要求关闭随机图像增强。

默认 `teacher_cache.load_into_ram=false`，教师监督保留在 SSD，通过只读 memory map
按需读取；源 RGB 按 `dataset.preload_images=true`、`dataset.load_result_add=ram`
预加载到 RAM。教师文件的系统页缓存可按需回收，不再额外创建整份监督的 RAM 副本。
这份 USB 数据共 96,531 个窗口，`[768,256]` float16 token 约占 **35.4 GiB**
（float32 为 70.7 GiB），
原始 patch 注意力另占约 **283 MiB**。内存充足时可显式设置
`teacher_cache.load_into_ram=true`，Linux DataLoader worker 共享加载后的只读数组。
这里缓存的是教师监督；源 RGB 仍由数据集读取，增强图像变化后的学生 DINO 前向／
反向仍需每步执行。`teacher_cache.batch_size` 独立控制教师预计算 batch。
开启区域监督时，教师编码、完整去噪链和注意力 rollout 均按该 batch 并行执行，
注意力在 GPU 上汇总后随整个 batch 写入缓存，避免逐样本运行和逐步回传 CPU。
各样本独立使用 `teacher_seed`，保留原来的噪声初始化与逐去噪步归一化后求平均的定义；
普通 context token 目标仍通过单独的无注意力提取前向计算，原有 float32 缓存仍兼容。
若关闭缓存，区域监督回到每个有梯度的训练步骤在线采集，验证仍不运行教师 DiT。

```bash
conda activate fdp

# 5 步 smoke test：跳过全量预计算；结束时也会验证并保存 best/latest checkpoint。
python train_distill_render_net.py \
  --config diffusion_policy/config/train_distill_render_net.yaml \
  teacher_checkpoint=data/outputs/2026.10.20/21.54.27_train_dp_image_query_lstm_insert_usb_insert_usb_force_aware/checkpoints/best.ckpt \
  teacher_cache.enabled=false dataloader.batch_size=4 val_dataloader.batch_size=4 \
  training.compile.enabled=false training.max_steps=5 training.max_val_batches=2

# 新开正式训练：最多 1000 个 epoch，以完整验证集 MSE 早停。
python train_distill_render_net.py \
  --config diffusion_policy/config/train_distill_render_net.yaml \
  teacher_checkpoint=data/outputs/2026.10.20/21.54.27_train_dp_image_query_lstm_insert_usb_insert_usb_force_aware/checkpoints/best.ckpt \
  dataloader.batch_size=32 training.num_epochs=1000 training.early_stopping.patience=30

# Transformer + 教师注意力区域监督，分位数可任意修改。
python train_distill_render_net.py \
  --config diffusion_policy/config/train_distill_render_net.yaml \
  render_net.backbone=transformer \
  dataloader.batch_size=32 \
  render_net.high_attention_render.enabled=true \
  'render_net.high_attention_render.percentiles=[80,99]'

# 只预计算／复用完整教师缓存，完成后退出；不加载 RAM、不训练。
# 在上面的命令后追加 teacher_cache.only=true teacher_cache.cleanup_on_exit=false

# 选回 U-Net，关闭区域限制：在上述训练命令中使用以下覆盖项。
# render_net.backbone=unet render_net.high_attention_render.enabled=false
```

训练与验证 batch size 分别由 `dataloader.batch_size`、`val_dataloader.batch_size`
控制，默认分别为 32、64。v2 原始卷积在 24 GiB RTX 5090 D v2 上实测
训练 batch 64 显存不足；旧版渲染器的显存测量不适用于 v2。
使用 AdamW、CUDA bfloat16 AMP、梯度裁剪 1.0，最多训练 1000 个
epoch，每个 epoch 结束时验证并保存 `checkpoints/latest.ckpt`。验证 MSE 创历史
最低时保存 `checkpoints/best.ckpt`；连续 30 个 epoch 无改善则早停，
`training.early_stopping.min_delta` 默认 0，可配置判定改善的最小幅度。best 始终
按实际最低 MSE 保存，不受 min_delta 影响。初始未训练模型也参与 best 比较。
默认 `training.max_val_batches=null`，使用完整验证集；`training.max_steps=null`，
不会再按 10,000 步截断。smoke test 可单独设置这两个上限。
默认创建新训练目录并从头训练；设置 `training.resume_from` 可恢复模型、优化器、
epoch 和验证最优记录。默认关闭图像增强以保证两条路径输入一致。
该教师实际使用 `future_wrench`（每观测 8×6），学生保持同一数据时间窗口。

RenderNet 支持单机多卡 DDP：**每张 GPU 各放一份完整模型**，各自处理一个 batch
分片，更新时同步 RenderNet 梯度。冻结 DINO 在每卡执行学生图像的前向和输入梯度
反传。默认 `training.distributed.batch_size_is_global=true`，因此全局训练 batch 32
在四卡上为每卡 8，验证 batch 64 为每卡 16；全局 batch 必须能被卡数整除。
如果希望四卡各用 batch 32，可设置 `dataloader.batch_size=128`，但这会改变每个
epoch 的优化器更新次数和训练效果，不能保证与全局 batch 32 等价。

```bash
conda activate fdp

# 四卡继续已有训练：沿用该次运行的教师、渲染器结构和缓存配置。
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
  bash scripts/train_distill_render_net_ddp.sh \
  --config data/outputs/2026.10.28/19.05.59_distill_render_net/config.yaml \
  ++training.resume_from=data/outputs/2026.10.28/19.05.59_distill_render_net/checkpoints/latest.ckpt \
  dataloader.batch_size=32 val_dataloader.batch_size=64 \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_distill_render_net_ddp'

# 新训练：同一脚本使用默认配置，也可追加教师和渲染器覆盖项。
# CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 bash scripts/train_distill_render_net_ddp.sh
```

多卡时教师监督仍保留在 SSD，以只读 mmap 访问；`load_result_add=ram` 自动改用
`/dev/shm/diffusion_policy_image_cache/` 下的一份共享 RGB 缓存，先由 rank 0 解码，
其他进程直接复用，不按卡数复制。该缓存使用 RAM 中的 tmpfs；需要足够的主机 RAM
和 `/dev/shm` 容量，Docker 可通过 `--shm-size` 调大，也可将
`DP_SHARED_IMAGE_CACHE_DIR` 指向另一处 RAM 挂载目录。空间不足时多卡蒸馏会报错。
教师 token 缓存也只由 rank 0 生成或续建，完成后各卡并行训练／验证；已有单卡缓存
可直接复用。DataLoader 的 `num_workers` 是每个进程的数量，需结合 CPU 核数设置。
仅 rank 0 写训练指标和 checkpoint；单卡与 DDP checkpoint 可互相恢复，模型结构
需要一致。实际加速取决于每卡 batch 大小、GPU 互连、SSD 吞吐量和 CPU 供数速度。

CUDA 训练默认开启 `training.compile.enabled=true`，分别编译 RenderNet 图像主干和
学生 DINO 视觉路径。LSTM 仍使用完整序列；教师缓存、初始化评估和验证使用普通
执行方式。BF16 AMP、TF32 和 DINO 的 SDPA／Flash Attention 已开启，无需再改成
FP16 AMP。编译保持相同的损失定义，但浮点融合会产生数值舍入差异。
首次训练步含编译时间；训练 batch／图像尺寸变化时可能重新编译，最后一个不足
batch 的训练样本也会保留。编译错误会直接报告，可用
`training.compile.enabled=false` 关闭；也可分别关闭 `render_backbone` 或 `student_dino`。
保存的 checkpoint 使用普通参数键，推理／可视化不需要 `torch.compile`。

2026-11-01 实测当前默认的**高斯光团＋去噪输出蒸馏**：RTX 5090 D v2，batch 32，
双相机、两帧 192×256，wrench 为 `[32,2,8,6]`，BF16，学生区域预测与教师区域监督开启。
使用刚停止的 20 Hz LSTM2 教师 40,000-step EMA 检查点、32 个真实数据样本；
每种模式独立进程，预热 5 步后测量 20 步，CPU 线程数为 4。

| 配置 | 平均每步 | P90 每步 | 样本／秒 | 稳态峰值分配显存 |
| --- | ---: | ---: | ---: | ---: |
| 不编译 | 249.7 ms | 258.6 ms | 128.2 | 12.95 GiB |
| 默认 compile | 134.2 ms | 139.3 ms | 238.5 | 9.74 GiB |

计时包含教师 DiT 前向、学生 RenderNet／DINO／DiT 前后向、区域监督、梯度裁剪与 AdamW；
重复使用已在 GPU 上的真实 batch，每步重新采样扩散噪声／时间步。
不包含数据加载、传输、教师 context／attention 的预计算、日志绘图与 checkpoint 写入。
编译模式首次训练步耗时约 42.7 秒；表中显存统计在预热后重置，未包含首次编译的分配峰值。
这些是计算步骤的短测结果，不代表完整 epoch 耗时或收敛质量。
原始报告保存在 `data/benchmarks/2026.11.01/rendernet_glow_b32/{eager,compiled}/report.json`，
汇总为同目录下的 `comparison.json`。

可用独立测速脚本复现；`--no-compile` 可测普通执行模式，两次测量使用不同输出目录：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2 \
python -m diffusion_policy.scripts.benchmark_render_net_training \
  --checkpoint /path/to/teacher.ckpt \
  --output-dir data/benchmarks/rendernet_b32_compiled \
  --batch-size 32 --warmup 5 --steps 20 --compile --video-backend pyav
```

以下是旧 **residual + token MSE** 的本机短测，不能作为新光团／去噪蒸馏的速度预估：
RTX 5090 D v2，batch 32、两观测×双相机 192×256、BF16、区域监督开启，
200k-step 教师。计时包含渲染器与学生 DINO 前后向、裁剪和 AdamW，排除 DataLoader
等待；使用重复的真实 RGB 帧和合成 wrench，未跑完整 epoch。

| 配置 | 每训练步 | 峰值显存 |
| --- | ---: | ---: |
| v2 standard 卷积 | 334 ms | 16.9 GiB |
| bottleneck 卷积 | 271 ms | 15.5 GiB |
| bottleneck + compile | 167 ms | 12.1 GiB |

首次编译约 45 秒；以上速度不代表验证 MSE 或最终渲染质量已改善，需通过完整训练比较。

训练目录位于 `data/outputs/<日期>/<时间>_distill_render_net/`，包括配置、
`teacher_config.yaml`、`teacher_cache.json`（缓存位置）、`logs.json.txt`、
`training_summary.json` 和 checkpoint。
Checkpoint 保存 RenderNet（含独立的 LSTM、projection、位置编码与相机 embedding）、
normalizer、教师 wrench 预处理配置／状态、AdamW／AMP 状态及训练步数；完整教师
policy 继续引用 `teacher_checkpoint`，不重复打包。缺少 `architecture_version` 的旧
checkpoint 会按原 GRU／FiLM 结构加载，可继续可视化；新结构需重新训练。
运行时打印两条 token shape、相机／
观测数量和 patch 数量，不硬编码 token 长度。
训练摘要记录当前 epoch、最优 epoch／step、早停计数和停止原因；达到 epoch 上限、
显式 step 上限或验证早停会分别记录为 `max_epochs`、`max_steps`、`early_stopping`。
终端和 `train_distill_render_net.log` 同时记录运行进度：首次训练 batch 的评估
开始／结束、缓存样本数／速率／ETA、验证首个／每 10 个／最后一个 batch 的 MSE、耗时与 ETA；教师注意力
逐样本采集超过 30 秒时也会输出进度。训练步骤按 `training.log_every` 打印
主目标 MSE、token 诊断值、区域 MSE、总 loss、梯度范数和显存。`logs.json.txt` 保存指标。

本地日志参考 DP 的训练进度方式，不启动 W&B。终端显示训练／验证进度条；
文件日志按 `training.log_every` 记录 batch 进度、当前及本轮平均 MSE、区域 MSE、
总损失、学习率、梯度裁剪前范数、数据等待／计算耗时、samples/s、ETA 和显存。
每轮按样本数加权汇总训练指标，记录验证 MSE、最佳 checkpoint 和早停计数。

训练目录中的 **`training_metrics.png`** 包含六组曲线：逐步主目标 MSE、按 epoch
汇总的训练／验证 MSE（标记最优验证点）、主目标／token 诊断／区域／总损失、梯度范数、学习率，
以及显存／训练速度。初始训练 batch 单独标记，验证与最优 MSE 只包含所选主目标。
首次评估后、每 `logging.plot_every_steps=100` 步和每轮验证后覆盖更新 PNG。
设为 0 时仅在初始化／每轮验证后更新。解码与教师缓存使用
`Decode + teacher cache [train/validation]` 进度条，按已保存的样本数显示百分比、
sample/s 和 ETA；续算时从已有缓存数量开始。当前 RGB 按 batch 懒解码，因此该条
显示解码及教师前向、落盘的整体进度。训练／验证另按 batch 显示。
进度条支持终端、IDE 和重定向输出，可通过 `logging.progress_bar=false` 关闭；
普通文件日志仍保留进度。缓存锁表示另一个进程正在生成同一缓存；停止持锁任务后
直接重新运行即可续算，无需删除缓存或锁文件。

已有 JSON 日志也可以重新生成曲线：

```bash
python -m diffusion_policy.scripts.plot_render_net_training \
  data/outputs/<日期>/<时间>_distill_render_net
```

### RenderNet RGB 推理可视化

使用修复数据流后训练的 RenderNet checkpoint，沿用保存的接触检测／门控／
normalizer 和数据窗口，调用与教师相同的 wrench 预处理，
无需加载教师 DINOv3、cross-attention 或 DiT：

```bash
conda activate fdp
# 替换为修复后新训练的 RenderNet 目录。
RENDER_NET_RUN="data/outputs/<日期>/<时间>_distill_render_net"
python diffusion_policy/scripts/visualize_render_net_episode.py \
  --checkpoint "$RENDER_NET_RUN/checkpoints/best.ckpt" \
  --episode 50 --camera both --overwrite
```

输出位于该训练目录的 `attention_visualize/`：

- `render_net_episode_50_both_comparison.mp4`：每个相机一行，依次为原始 RGB、
  叠加残差后的 RGB、RGB 差异（默认放大 5 倍，灰色表示零差异）。
- `render_net_episode_50_both_rendered.mp4`：仅展示叠加后的真实 RGB，双相机横向拼接。
- 同名前缀的预览 PNG、差异统计 NPZ 和记录 checkpoint／帧率／相机的 metadata JSON。

默认逐帧处理完整 episode，以数据集原始帧率输出连续视频；只展示每个窗口的
最后一帧观测。`--camera wrist` 或 `--camera side` 可选择展示的相机，推理仍输入
所有相机。`--difference-gain 10` 仅调整差异栏的放大倍数，原图和渲染结果均不
放大。`--save-frames` 可额外保存逐帧 PNG；`--stride 2` 可隔帧处理并自动按比例
降低视频帧率，保持原始播放速度。默认视频后端为 `pyav`，可用 `--device cpu`
在 CPU 上推理。
主干和区域头从 checkpoint 配置自动恢复。区域头启用训练的模型，在推理命令中
加 `--disable-high-attention-render` 可以临时改为整图渲染作对比；该选项不会修改
checkpoint。修改分位数会改变教师区域监督目标，需要重新训练，推理时不重新
运行教师注意力或用新分位数伪装已有预测区域。
使用原始 wrench 直接归一化的旧 checkpoint 不包含一致的预处理状态，推理脚本
会提示重新训练，避免静默改变模型训练时的输入分布。旧错误训练产物已清理。

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

### 从 HDF5 转换为 LeRobot v3

`scripts/h5_2_lerobotv3.py` 将 LIBERO 风格的 `data/demo_N`、`episode_N`，或一批单 episode HDF5 转为训练目录。脚本通过官方 LeRobot writer 生成 Parquet、episode 元数据、统计量和双路视频。LIBERO 源格式会从 `env_args.control_freq` 推断采样率；下面的数据实际为 20 Hz，不能通过修改 FPS 标签转换成 50 Hz。末端位姿统一使用 xyz + xyzw，源状态和相机时钟以纳秒保存。

LIBERO 的原始 action 是 OSC_POSE 增量及夹爪命令，不能当作关节角命令推导 `q_cmd` / `delta_q`。该模式不导出这些字段，也不导出来源未经验证的 `torque` / `tau_free` / `tau_ext` / `tau_ext_raw`；图像与力训练直接读取 `observation.wrench`。转换记录保存在 `meta/wrench_conversion.json`。

例如转换本次开抽屉放碗任务的全部 50 条演示（输出目录需不存在）：

```bash
conda activate fdp
python scripts/h5_2_lerobotv3.py \
  --input /home/lcy/code/LIBERO-FT/dataset/wm_data/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it_demo_wrench_with_dq_tau.hdf5 \
  --output data/libero_open_top_drawer_put_bowl_20hz_lerobot_v3 \
  --task-name 'open the top drawer of the cabinet and put the bowl in it' \
  --fps 20 --camera-fps 20 \
  --image-size 192 256 \
  --velocity-key obs/dq --ee-pose-key robot_states \
  --allow-vector-truncate
```

真实机器人日志可以用 `--source-format generic` 和显式 key 覆盖默认映射。generic 模式可推导缺少的速度、`q_cmd`、`delta_q`，缺少的扭矩字段会填零，因此应确认 action 语义并提供真实的力矩来源：

```bash
python scripts/h5_2_lerobotv3.py \
  --source-format generic \
  --input '/path/to/raw/*.h5' \
  --output data/insert_usb_50hz_lerobot_v3 \
  --task-name insert_usb --fps 50 --camera-fps 25 \
  --joint-key observation/joint \
  --action-key action \
  --ee-pose-key observation/ee_pose \
  --wrist-key cameras/wrist --side-key cameras/side \
  --wrench-key observation/wrench \
  --torque-key observation/torque \
  --tau-free-key observation/tau_free \
  --tau-ext-key observation/tau_ext \
  --overwrite
```

本次数据对应的两层 LSTM 训练配置为：

```bash
python sim_force_aware_dp_train.py --config-name sim_force_aware_dp_train_20hz_lstm2
```

该配置使用冻结 DINOv3 ViT-B/16、两层 LSTM，以及一层图像 token 作 Query 的 cross-attention；全部 50 条演示用于训练，batch 为 128，预算为 100,000 个 optimizer step。学习率峰值为 `1e-4`，warmup 500 步后按余弦退火下降。每 20,000 步保存一次，保留最近 3 个检查点，`latest.ckpt` 指向最新保存的权重；不使用离线验证集。它继承当前实验的未来真值 wrench 条件，实际取相机时间之后 0.05–0.40 秒的 8 个力样本。图像模糊在前 3000 步保持，至 10000 步降至零。

需要的转换依赖已经包含在 `fdp` 环境中（`h5py`、`lerobot==0.4.0`、`pyarrow`、`av`）。如果输入向量包含额外自由度，可在确认列顺序后加 `--allow-vector-truncate`；脚本只取前 7 个关节或前 6 个 wrench 分量。

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
  cross_attn_query_residual: true  # false：不将 Query tokens 加回 CA 输出
```

`force_encoder_layers` 控制 wrench 时序编码深度，保留每个时刻的输出。
联合 SA 在 `[所有相机图像 tokens, wrench tokens]` 上运行，随后按原位置拆分。
多层 CA 逐层更新 Query 流，K/V 固定为前置编码器的输出。
`cross_attn_query_residual: false` 时，每层先取 CA 输出，再计算
`C + FFN(LN(C))`，不加回输入 Query。新的 image-query + LSTM 配置使用此设置：
图像只作为 Q 控制对 wrench K/V 的读取，融合 tokens 经过 FFN 后直接送入 DiT。
`true` 时保留 `Query + CA`；旧 checkpoint 配置缺少此字段时也按 `true` 加载。

| 设置 | 送入 DiT 的条件 | 两相机、两帧、每相机 192 patches、8 wrench tokens 时的长度 |
| --- | --- | --- |
| CA >= 1，`cross_attn_query: wrench` | wrench + 从图像读取的信息 | 16 |
| CA >= 1，`cross_attn_query: image`，query residual=false | 图像 Q 读取 wrench K/V 得到的融合 tokens，无图像直通残差 | 768 |
| CA >= 1，`cross_attn_query: image`，query residual=true | 图像 + 从 wrench 读取的信息，保留视觉残差（旧结构） | 768 |
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
  policy.cross_attn_query_residual=false \
  policy.force_temporal_encoder=transformer policy.force_encoder_layers=2
```

保留默认的一层 wrench-query CA 时，旧 checkpoint 的参数名和计算保持兼容；
改变层数、方向、Query 残差或 backbone 后需要新建训练，不能把旧权重当作同构模型继续训练。
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

### Wrench 输入消融的 action MSE 曲线

```bash
conda activate fdp
# 必须指定一个 episode（编号从 0 开始），以该 episode 的时间（秒）为横轴。
python -m diffusion_policy.scripts.compare_wrench_action_mse --episode 0
```

默认加载
`data/outputs/2026.10.20/21.54.27_train_dp_image_query_lstm_insert_usb_insert_usb_force_aware/checkpoints/best.ckpt`
的 EMA 权重，仅按时间顺序评估指定 episode 的有效窗口，不做阶段加权或有放回抽样。
比较三条曲线：**红色**为独立均匀随机噪声替换每个 wrench 元素，
**黄色**为原始 wrench 置零，**蓝色**为数据集原始 wrench。
每维随机噪声的范围为 `训练均值 ± noise_scale × 训练标准差`，统计量取自 checkpoint；
默认 `--noise-scale 20`，可设为 `--noise-scale 50` 增大幅度。按每维独立缩放，
让力与力矩使用各自的数据尺度；固定随机种子以便复现。三种条件共享图像、目标动作和
diffusion 采样噪声，wrench 噪声使用独立随机数生成器。仅替换 `policy.wrench_key`（此 checkpoint 为
`future_wrench`），接触门控使用的 `tau_ext`、`future_tau_ext` 保留原值；
置零和均匀噪声替换均发生在原始物理量进入模型预处理之前。
原有接触门控仍会把无未来接触窗口的 wrench 清零，因此这些窗口的三条曲线可以重合。
图例的 `mean action MSE` 是动作预测误差的平均值，不是 wrench 输入均值。
即使原始 wrench 全零，模型仍会输出动作，动作与示教的 MSE 通常不为零；此外，
接触窗口的物理零经过标准化后也不保证是数值零（高斯标准化为 `-均值/标准差`）。

背景沿用数据集的三阶段标签：**白色**为非接触，**浅橙色**为 pre-contact，
**浅红色**为接触；阶段交界处画竖向虚线并标出时间。历史 `tau_ext` 或未来
`future_tau_ext` 窗口内出现超过接触阈值的 token 就标记接触；其余位于该 episode
首次接触前 1 秒的窗口标记 pre-contact，其余为非接触。本 checkpoint 使用 7 维
`tau_ext` 的 L1 范数、阈值 1.5。由于窗口包含未来信息，接触区间可能早于原始
信号的首次越阈时刻。阶段始终由原始数据确定，且在 `--stride` 抽稀之前计算；
若一个 episode 多次切换阶段，会保留所有区间和分界线，没有接触时则显示白色背景。

MSE 与 checkpoint 的 `val_action_mse_raw` 一致：仅比较实际输出的 8 步动作，
逐窗口对动作步和维度取均值，不计算 diffusion 训练损失。横轴为观测 anchor 相对
episode 起点的真实时间（秒），不会把首个有效窗口重置为 0；数据预热、无效图像和
未来窗口不完整的样本仍按原数据集规则排除。输出 PNG、逐窗口 CSV（含阶段标签）和
设置/均值 JSON（含阶段边界），位于 checkpoint 所在运行目录的
`wrench_ablation/action_mse_episode_<编号>.*`。可用 `--checkpoint`、`--dataset-path`、`--batch-size`、
`--num-workers`、`--seed`、`--stride`、`--max-samples`、`--output` 调整评估。
默认 `--stride 1` 评估该 episode 的全部有效窗口，不预加载整库图像。

加 `--pose-metrics` 可同时统计 xyz + xyzw 动作的物理误差（xyz 输入按米计）：
位置 RMSE 为 `1000 * sqrt(mean(||预测位置 - 示范位置||²))`，单位 mm，
不再除以三个坐标维度；旋转误差为归一化四元数之间的最小旋转角，单位度，
`q` 和 `-q` 视为相同姿态。统计同样只使用实际输出的 8 步动作，
对重叠窗口中的每次预测分别计数。输出额外的 `_pose_metrics.png` 曲线和
`_pose_summary.csv`，包含整集及 free/pre-contact/contact 各阶段的位置 RMSE、
平均旋转角误差和旋转角 RMSE；逐窗口指标及计算定义保存在原 CSV/JSON 中。

默认 `--start-time 3`，图、CSV 和平均 MSE 只包含 episode 时间大于等于 3 秒的窗口；
用 `--start-time 0` 可保留完整时段。时间筛选在推理后执行，因此调整起点不会改变
保留窗口的 diffusion／wrench 随机噪声；横轴继续使用原 episode 时间。

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
回放只运行观测编码与融合模块，不调用 DiT 或动作采样，也不计算输出梯度。

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
  取 CA 输出投影之后、可选的 Query 残差及 FFN 之前的更新幅度。
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

如果只研究 Cross-Attention 概率，可使用：

```bash
python diffusion_policy/scripts/visualize_attention_episode.py \
  --checkpoint data/outputs/2026.10.20/21.54.27_train_dp_image_query_lstm_insert_usb_insert_usb_force_aware/checkpoints/best.ckpt \
  --episode 0 --camera both --mode cross-attn --layer 0 \
  --attention-only --attention-normalization percentile \
  --attention-percentiles 1 99 --save-frames --overwrite
```

`--attention-only` 仅导出注意力视频、原始概率和元数据，省去 CA-update L2
视频。当前 checkpoint 是 image Query / wrench Key，每个面板对应一个
wrench Key：`heatmap_m(i) = mean_head A[head, image_i, wrench_m]`。
平均全部 wrench Key 会得到恒定的 `1 / K`，因此保留各 wrench 面板，
不把它们相加或平均成一张“总体空间关注图”。Wrench Query 的 checkpoint
则按其真实结构展示 `mean_head A[head, wrench_m, image_j]`。

参考 [Visuo-Tactile Transformers for Manipulation 第 4.4 节](https://ar5iv.labs.arxiv.org/html/2210.00121#S4.SS4)：
论文展示多头注意力平均后的热图，并在附录 6.3 展示视觉/触觉注意力占比。
其式 (4) 来自图像和触觉联合 token 的注意力，包含四个模态子块；正文未明确
热图对 Query 行的具体聚合方式，不能声称本脚本精确复现其全部处理。
当前独立 CA 的 Key 仅有一种模态，也不能从它的概率直接恢复论文中的
视觉/触觉占比。这些图描述融合层的跨模态读取关系，区别于下面的最终动作
梯度敏感度。

### 去噪链中的图像与 wrench 注意力路由

`visualize_denoising_attention_episode.py` 对每个 observation window 回放完整的
DiT 采样轨迹，并将各去噪步的注意力百分比取平均。一个视频帧对应 episode 的
一个观测时刻，默认按数据集原始帧率连续播放：上方将动作 token 的注意力回溯到
wrist/side 的 DINO patch，使用与 `wrench_image_self_attn_episode_50.mp4` 一致的
JET 叠加配色；下方在同一百分比坐标系中绘制图像和 wrench 输入的路由占比
曲线，竖线指向当前 episode 时刻。热力图与曲线的数值都以**每个去噪步
路由到条件 token 的总量**为分母，先按步计算百分比，再跨去噪步取平均；所有选中的观测帧和相机
共用一个热力图色标：先跨去噪步取平均，再对本段视频全部观测窗口、指定相机和
观测时刻的 patch 百分比计算 p1/p99，分别映射为 JET 的最冷色和最热色；
范围外的值截断显示。色条标注实际百分比上下限，整段视频固定使用这一范围。
可通过 `--attention-percentiles 2 98` 或 `--attention-percentiles 10 90`
修改分位范围，默认 `1 99`；画面和元数据均记录所选分位范围及实际百分比上下限。
该显示拉伸不改变 NPZ 中的原始路由百分比或下方曲线数值。
默认显示最新观测的两路图像，曲线统计全部观测时刻。

`--camera wrist|side|both` 选择注意力统计和显示的相机，默认 `both`。
推理仍输入 checkpoint 配置的全部相机，但热力图、分位色标、原始图像注意力
数据和图像路由曲线只统计所选相机。百分比仍以全部相机和 wrench 的条件路由
总量为分母；wrench 曲线统计范围保持全部 wrench，所以单相机模式的两条曲线
不一定合计为 100%。曲线标签注明所选相机，元数据的 `raw_camera_keys`
按顺序对应 NPZ 中的 `camera_0_percent` 等数组。

比较纯视觉与 image-query cross-attention 的视觉空间集中度时，使用
`--heatmap-normalization image_relative`。该模式对每个去噪调用的所选图像
patch 重新归一化，使全部选中图像路由合计 100%，消除带视觉残差的 image-query CA
的固定 0.5 图像残差分支因子；原始绝对百分比仍保存在 `camera_N_percent`，归一化结果
保存在 `camera_N_image_relative_percent`。默认 `absolute` 模式保留图像／wrench
总路由量，输出文件名不增加后缀；归一化模式的输出增加 `_image_relative`。

对于 image-query cross-attention 且关闭图像 Query residual 的 checkpoint，脚本会自动
切换为 `dit_to_image_query_slots`：仍从 DiT 去噪 query 开始，但把相关性映射到
cross-attention 的图像 Query 槽位。这个结果表示动作读取了哪些图像 Query 输出位置，
不等同于 cross-attention value 路径对原始图像输入的因果归因；该区别会记录在 metadata
的 `routing_method` 和 `interpretation` 字段中，因此这类 checkpoint 也可以使用
`image_relative`，不会因为严格 value-path 图像质量为零而失败。

脚本还会在 NPZ 和 metadata 中保存全局及每个相机的归一化熵和 top-k 路由质量。
熵越低、top-k 质量越高表示视觉路由越集中。默认 top-k 为 patch 的 10%，可用
`--concentration-top-k 0.2` 修改。为了让两个模型的颜色可以直接比较，使用同一
组固定色标，例如：

```bash
python diffusion_policy/scripts/visualize_denoising_attention_episode.py \
  --checkpoint <cross-checkpoint> --episode 50 --camera both \
  --heatmap-normalization image_relative --color-bounds 0.05 1.0 --overwrite

python diffusion_policy/scripts/visualize_denoising_attention_episode.py \
  --checkpoint <visual-checkpoint> --episode 50 --camera both \
  --heatmap-normalization image_relative --color-bounds 0.05 1.0 --overwrite
```

若不指定 `--color-bounds`，每个视频仍会根据自己的 episode 分位数计算色标，
适合单独观察，不适合比较两个模型的颜色强弱。归一化热图和集中度仍是 action-to-
image route 诊断，不等同于最终动作的因果归因。

直接可视化某一层的 SA/CA 权重，展示该层 Query 对 Key 的 softmax 读取分布；
这里则从执行动作 chunk 的 token 出发，逐层回溯 DiT、条件编码器和观测融合层，
展示动作读取条件信息时通向图像 patch 或 wrench 的注意力路由近似。
每帧先分别计算各去噪步，再等权平均；视频横轴为 episode 时间，
并未沿采样 scheduler 将这些路由传播为最终动作的因果贡献。

```bash
python diffusion_policy/scripts/visualize_denoising_attention_episode.py \
  --checkpoint data/outputs/2026.10.20/21.54.27_train_dp_image_query_lstm_insert_usb_insert_usb_force_aware/checkpoints/best.ckpt \
  --episode 50 --overwrite

# 保留双相机输入，只分析 wrist，使用本相机整段视频的 p2/p98 色标。
python diffusion_policy/scripts/visualize_denoising_attention_episode.py \
  --checkpoint data/outputs/2026.10.20/21.54.27_train_dp_image_query_lstm_insert_usb_insert_usb_force_aware/checkpoints/best.ckpt \
  --episode 50 --camera wrist --attention-percentiles 2 98 --overwrite

python diffusion_policy/scripts/visualize_denoising_attention_episode.py \
  --checkpoint data/outputs/2026.10.19/13.24.35_train_dp_selfattn_insert_usb_insert_usb_force_aware/checkpoints/FRIST_optimizer_step=00280000-train_loss=0.017619.ckpt \
  --episode 50 --overwrite

python diffusion_policy/scripts/visualize_denoising_attention_episode.py \
  --checkpoint data/outputs/2026.10.19/22.25.58_train_dp_self_crossattn_insert_usb_insert_usb_force_aware/checkpoints/FRIST_optimizer_step=00140000-train_loss=0.036929.ckpt \
  --episode 50 --overwrite
```

视频、原始百分比 NPZ、元数据 JSON 和可选的 PNG 均保存在 checkpoint 所属
训练目录的 `attention_visualize/` 下。视频文件名与画面标注融合结构：
`cross`、`self` 或 `cross+self`。`--max-frames 1` 可先导出一个观测窗口；
单相机输出增加 `_wrist` 或 `_side` 后缀，与原双相机视频分别保存。
`--observation all` 显示两帧观测，`--action-index` 可只看某个动作位置。
当前环境的默认数据视频解码使用 `pyav`，输出使用 H.264；`--fps` 可覆盖
数据集原始播放帧率。

该图是跨 DiT decoder、condition encoder 和观测融合层的残差注意力路由近似，
不是最终动作的因果归因。特别是旧的带视觉残差的 `image` Query、一层 CA、无前置联合 SA
的 checkpoint，残差路由按固定的 0.5/0.5 分给全部图像和 wrench，因此 `--camera both`
的两条占比曲线保持 50%/50%；单相机曲线只计该相机在全部图像路由中的份额。
热力图仍可显示动作读取的图像位置如何随 episode 推进而变化，
NPZ 中保留各去噪步的原始百分比。
带图像+wrench 联合 SA 的 checkpoint 则能得到随步数变化的模态路由占比。
关闭 Query 残差后，CA 的这类 rollout 全部沿 V 路径传播；若没有前置联合 SA，
会得到零图像路由。该近似没有追踪图像 Q 对注意力权重的影响，零图像路由不表示
模型不使用图像；要观察这部分影响，应使用下述对输入图像反传的敏感度工具。

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

新训练配置使用 RGB 高斯模糊课程：在 DINO 编码和 ImageNet 标准化之前降低
图像清晰度，随后逐渐减小高斯标准差 `sigma`，直到严格恢复原图。这个方法参考
[FACTR 的像素空间模糊课程](https://github.com/RaindragonD/FACTR)，保留本项目的
接触条件：优先根据最新观测的 `future_tau_ext` 判定未来窗口是否会接触，没有未来
信号的配置使用当前接触状态。只对符合条件的样本应用模糊，非接触样本保持原图。
验证和推理始终使用清晰图像。支持两种范围：

```text
current_observation | full_context
```

`full_context` 同时模糊该样本所有相机、全部观测步；`current_observation`
只模糊最新观测。高斯标准差由真实 optimizer update 驱动，不再通过随机抽取
样本、替换为 mask token 来逐渐开放视觉信息。`policy.image_blur_schedule` 支持：

```text
constant | linear | cosine | exponential | piecewise
```

一般 force-aware 配置默认从 `sigma=5` 按 cosine 日程降到 0，结束步数沿用各
配置原来的课程时长。没有视觉课程的 baseline 显式设置 `image_blur_schedule: null`，
并保持原来的零 mask 概率。训练日志新增 `curriculum/blur_sigma` 和
`curriculum/blurred_image_fraction`；模糊课程的 `masked_image_fraction` 始终为 0。

200k 步的接触课程有两个独立实验：

| 配置（位于 `alation_study/`） | 融合结构 |
| --- | --- |
| `train_dp_image_query_lstm_contact_blur_insert_usb` | 图像 Q 读取 LSTM wrench K/V；1 层 CA，无前置 SA，无图像 Query 残差 |
| `train_dp_image_wrench_selfattn_contact_blur_insert_usb` | 图像与 LSTM wrench 拼接后经过 1 层联合 SA，全部 tokens 送入 DiT，无 CA |

两者使用相同的 `full_context` RGB 模糊日程（单位为输入图像像素）：

| Optimizer 步数 | 高斯标准差 |
| --- | --- |
| 0–50k | `sigma=5` |
| 50k–150k | 从 5 线性降到 0；100k 时为 2.5 |
| 150k–200k | `sigma=0`，严格使用原始清晰图像 |

`training.validation_start_optimizer_step: 150000` 使前 150k 步
跳过验证（包括 `val_loss`、`val_action_mse`、`val_action_mse_raw`）及动作采样评估，
按 checkpoint 周期保存 `latest.ckpt` 供恢复；150k 步起按原来的验证和 checkpoint
周期启用评估与 Top-K/`best.ckpt` 评选。恢复时若验证起点改变，会清空旧排名并移除
旧 best 软链接，原始 checkpoint 文件保留。其他实验默认起点为 0，沿用原来的行为。

两个新实验均使用 `val_dataloader.num_workers: 0`、`persistent_workers: false`、
`prefetch_factor: null`，避免在长时间 CUDA 训练之后 fork 验证子进程时，回收继承的
锁页内存触发 CUDA 初始化错误。

```bash
python train.py \
  --config-name alation_study/train_dp_image_query_lstm_contact_blur_insert_usb

python train.py \
  --config-name alation_study/train_dp_image_wrench_selfattn_contact_blur_insert_usb
```

要排在当前训练之后依次运行这两个实验，先激活训练 Conda 环境，设置
`TRAIN_PID` 为当前训练主进程 PID，再执行：

```bash
bash scripts/train_usb_blur_after_current.sh \
  --wait-pid "$TRAIN_PID" --log-dir data/outputs/blur_curriculum_queue/manual \
  -- logging.mode=disabled
```

加 `--dry-run` 可查看队列命令。队列等待指定进程退出后，按 cross → self 顺序
从头训练，各自创建新的 Hydra 输出目录；队列目录保存 `status.txt`、`queue.pid`、
`active_train.pid`、两份 console log 和 `exit_codes.tsv`。通用队列入口
`train_usb_baselines_after_current.sh` 支持重复传入 `--config` 指定其他顺序。

`*_contact_mask_insert_usb` 配置恢复为旧的 learned token-mask 课程，按总训练步数的
比例设置里程碑：前 25% 保持全遮蔽，中间 50% 线性降到 0，最后 25% 保持不遮蔽。
`*_contact_blur_insert_usb` 继续使用 RGB 高斯模糊课程。已有训练目录和 checkpoint
保存的配置仍按各自记录的课程执行，不会改变已运行实验的课程或权重格式。

## 图像增强状态

DataLoader 已具备 resize、random crop、rotation 和 color jitter 接口，但当前 insert_usb 配置没有启用一般图像增强。

DINOv3 的 ImageNet 标准化、位置编码插值和接触模糊课程不属于常规图像增强。腕部相机几何关系对插入任务很重要，因此在验证时序一致的数据增强前，不默认启用 crop 或 rotation。

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
