# 新教师与跨相机 RGB 学生

## 教师的输出接口

新 self-attention 教师保留 DINO、力矩 LSTM 和联合 self-attention，ACT 只读取融合后的视觉位置。当前服务器实验保留上采样和高分辨率 skip，输出 784 个视觉 token；本地配置默认关闭两者，输出 196 个粗视觉 token。学生根据选定 checkpoint 自动适配这两种接口。本地默认路径如下：

```text
每时刻、每相机：DINO 14×14 → 平均池化 7×7
两相机 98 个视觉 token + 8 个力矩 token → 联合 self-attention
只保留融合后的 98 个视觉位置 → context norm
两时刻合并：[B,196,256] → ACT
```

对应配置为 `fusion_output_mode: visual`、`fusion_downsample_factor: 2`、`fusion_upsample: false`、`fusion_highres_skip: false`。力矩 token 参与每层融合，力矩编码器通过视觉输出接收动作梯度；Transformer 内部残差保留。“视觉”指序列位置，融合后可以包含力觉信息。

教师模糊课程使用 FACTR 默认比例：前 2/3 计划更新保持 σ=5，最后 1/3 线性降到 0，不另设清晰尾段；覆盖所有阶段、相机和观测。模糊课程和去掉 skip 都不能保证模型利用力矩，需要对训练后的模型进行固定图像的力矩干预并检查动作误差。

| 教师接口 | 两时刻 ACT context | 每相机视觉网格 |
| --- | --- | --- |
| 旧 all + 上采样 | `[B,800,256]` | 14×14，另含每时刻 8 个力矩位置 |
| visual + 上采样 | `[B,784,256]` | 14×14 |
| 本地默认 visual、无上采样 | `[B,196,256]` | 7×7 |

缺省新字段的旧 checkpoint 按 `all`、上采样的旧行为加载。新接口需要新训练或明确的微调，不能作为原模型的等价续训。

## 学生的训练与部署路径

新学生配置是 [train_distill_render_net_board_selfattn.yaml](../diffusion_policy/config/train_distill_render_net_board_selfattn.yaml)。旧学生默认入口和已有实验配置保留。

```text
两路 RGB + tau_ext → RenderNet CNN/图像注意力 + 冻结的教师初始化力矩 LSTM
→ 各相机视觉特征读取力矩 → 同一时刻跨相机 self-attention
→ 按相机分回网格 → 各相机 RGB decoder → 两张渲染 RGB
→ 冻结 DINO + 教师投影及位置/相机嵌入
→ 依 checkpoint：保留 14×14，或每相机每时刻池化到 7×7
→ 冻结 context norm → [B,784,256] 或 [B,196,256]
→ 部署时送入冻结 ACT
```

`cross_camera_attn_layers: 1` 让两路图像在输出 RGB 前交换信息；每个观测时刻独立融合，相机网格随后各自解码。这样，单张渲染图可以依据另一相机及力矩调整，适配教师视觉位置中包含的跨相机信息。它增加了学生表达这种依赖的能力，不保证 RGB 能恢复教师的全部多模态语义。

RenderNet 先从教师复制整套力矩编码参数，再冻结其 LSTM、力矩投影、力矩/观测位置嵌入和 force modality 嵌入。学生使用教师训练好的力矩表示，仅训练视觉融合、跨相机注意力和 RGB 渲染分支。渲染 RGB 后不再输入真实力矩，也不添加可训练的 token adapter。冻结 DINO 参数时保留对渲染 RGB 的输入梯度。

教师目标是其在清晰原图和真实力矩下实际送入 ACT 的全部视觉 token：服务器当前上采样版本为 784 个，本地默认粗网格版本为 196 个。学生经过冻结图像接口得到同形状的输出；784-token 接口保留投影和嵌入后的 14×14 网格，196-token 接口再按相机池化到 7×7，两者最后都执行 context norm。学生不重放教师联合 self-attention 或旧 cross-attention 的 FFN。

| 模块 | 学生训练期间 |
| --- | --- |
| RenderNet CNN、图像/力矩/跨相机注意力、RGB decoder | 训练 |
| RenderNet 自有 LSTM、力矩投影及力矩/观测位置、force modality 嵌入 | 从教师复制初始化后冻结 |
| 教师与学生共用的 DINO、图像投影、位置/相机嵌入、context norm | 冻结参数，学生路径保留输入梯度 |
| 教师力矩编码器及联合 self-attention | 冻结，只生成监督目标 |
| ACT | 冻结，部署时消费学生 token；不参与本轮训练损失 |

## 单阶段、全观测蒸馏与力矩差分

每个原图观测配对真实力矩与物理零力矩。教师分别生成实际送入 ACT 的视觉 token `T_real`、`T_zero`，得到 `ΔT = T_real - T_zero`。学生将同一组图像分别用真实、物理零力矩渲染，经冻结图像接口得到 `S_real`、`S_zero`，令 `ΔS = S_real - S_zero`。物理零力矩在教师预处理和归一化之前置零。

训练与验证选模使用同一加权损失：

```text
L_token = mean((S_real - T_real)²)
L_delta_raw = mean((ΔS - ΔT)²)
L_delta = L_delta_raw / difference_scale
L = L_token + difference_weight × L_delta + region_weight × L_region
```

差分项要求学生匹配教师由力矩变化引起的表征变化，补充仅拟合真实力矩目标的约束。`difference_weight=1`、`region_weight=1` 是初版起点，可通过命令行调整，尚未经过调优。自由、接触前和接触阶段均参与，不筛选阶段或跳过观测。`objective` 保持 `token_mse`，新增项由 `distillation.force_context.enabled` 控制。

两个尺度在训练前只从 **训练集** 教师配对缓存统计一次，并固定用于训练和验证：

```text
difference_scale = max(mean_train(ΔT²), 1e-8)
region_scale = max(sqrt(mean_train(ΔT²)), 1e-4)
```

统计覆盖训练缓存中的全部 token 元素；验证数据不参与尺度估计，也不对每个样本单独归一化。两个配置默认 `null` 表示自动计算固定尺度。

区域约束同样由原图教师配对得到。每个视觉位置计算通道 RMS `c = sqrt(mean_channels(ΔT²))`，转为 `w = c / (c + region_scale)`。按相机、观测时刻恢复空间网格：784-token 教师每相机、每个观测时刻为 14×14，196-token 教师为 7×7。令 `I_real` 为真实力矩渲染结果，先对实际 RGB 修改取通道平均，再自适应平均池化到对应教师网格：

```text
E = AdaptiveAvgPool_grid(mean_RGB(abs(I_real - I_original)))
L_region = mean((1 - w) × E)
```

教师力矩响应较小的位置受到更强的 RGB 修改惩罚。约束使用裁剪后实际 RGB 变化，且只约束真实力矩分支；零力矩分支用于差分参考。这里的区域权重来自冻结教师，不重新引入学生 region head、接触阶段标签或 RGB 乘法门控。

新配置关闭 region head、contact-alpha、phase prediction、独立条件网络和条件预热阶段。唯一训练阶段更新 RenderNet 的视觉融合、跨相机注意力和 RGB 渲染分支；从教师初始化的整套力矩编码保持冻结。没有动作/示范动作损失、phase/alpha 损失或额外平滑项。配置独立于旧两阶段 preset，并显式使用 `condition_backbone: null`，避免残留旧条件网络字段。

RGB 仍按已有残差形式生成：`clip(I + residual_scale × tanh(ΔI), 0, 1)`，默认 `residual_scale=1`。这定义图像输出范围，不是额外的图像损失。FiLM 已移除。

学生本轮使用清晰图像；教师生成蒸馏目标时也关闭训练模糊。学生模糊课程尚未加入配置或训练路径，不能把教师训练课程自动解释为学生课程。

## 启动与产物

必须显式指定按新 visual 接口训练的教师 checkpoint，不能沿用旧 800-token checkpoint。`teacher_checkpoint: ???` 为必填项，避免落入旧的自动教师选择流程。

从项目根目录运行，替换教师路径：

```bash
python train_distill_render_net.py \
  --config-name=train_distill_render_net_board_selfattn \
  teacher_checkpoint=/absolute/path/to/new_teacher/checkpoints/best.ckpt \
  dataloader.batch_size=16 \
  val_dataloader.batch_size=32 \
  training.num_epochs=100
```

默认使用单 GPU、最多 100 epoch，按上述验证综合损失选取最佳权重，并保留既有学习率调整和早停。默认输出为 `model_output/<日期>/<时间>_board_selfattn_student/student`，包含 checkpoint、解析后配置、训练指标与曲线。可通过 `output_dir=...` 覆盖运行目录。

训练前生成并保存真实/零力矩教师 token 配对缓存，计算精度与存储精度均为 FP32。缓存位于 `model_output/cache/render_net_teacher`，训练退出后保留，匹配同一教师与数据时可复用。只准备缓存和训练集尺度、不进行训练，可运行：

```bash
python train_distill_render_net.py \
  --config-name=train_distill_render_net_board_selfattn \
  teacher_checkpoint=/absolute/path/to/new_teacher/checkpoints/best.ckpt \
  teacher_cache.only=true
```

日志保留 `token_mse`、未归一化的 `force_delta_mse`、归一化的 `force_delta_loss` 和 `context_region_loss`。训练 batch、训练 epoch 与验证均记录这些分项；自动曲线和离线绘图分别展示各项及选模用的综合损失。`selection_metric="force context loss"` 时，历史兼容字段 `train_mse`/`val_mse` 表示综合损失，不能读作纯 token MSE。

要接在正在运行的教师之后，可启动一次等待任务。指定教师主进程 PID（四卡训练指定 `torchrun` 主进程 PID）和对应 run 目录：

```bash
python train_pipeline.py distill \
  --teacher-run /absolute/path/to/teacher \
  --wait-for-teacher-pid 12345 \
  --teacher-selection best \
  --distill-config diffusion_policy/config/train_distill_render_net_board_selfattn.yaml \
  --output-dir model_output/board_student_followup \
  --gpus 0
```

该任务等待原教师进程退出，并检查 `latest.ckpt` 已达到训练预算或记录了早停完成，再选择最终的 `best.ckpt`。学生随后自动预缓存真实／零力矩教师输出并训练；等待期间不占用 GPU。`--output-dir` 指定一个尚不存在的目录，学生产物写入其 `student/`，接续状态记录在 `pipeline.json`。四卡学生使用 `--gpus 0,1,2,3 --nproc-per-node 4`，全局 batch size 仍由学生配置决定。先加 `--dry-run` 可预览命令。

显存较保守的四卡接续可追加：

```bash
  --gpus 0,1,2,3 --nproc-per-node 4 \
  --distill-override dataloader.batch_size=8 \
  --distill-override val_dataloader.batch_size=8 \
  --distill-override teacher_cache.batch_size=2 \
  --distill-override training.compile.enabled=false \
  --oom-retries 3 --monitor-interval 60
```

此时每卡训练和验证 batch 为 2，缓存由 rank 0 以 batch 2 生成。监控在等待教师时写根目录 `gpu_usage.jsonl`，学生启动后写 `student/gpu_usage.jsonl`，持续到该阶段退出。每次学生尝试另存 `attempt_001.log` 等日志。明确发生 CUDA OOM 时，最多重试指定次数：降低训练、验证和缓存 batch，关闭编译，并在存在 `latest.ckpt` 时续训；尚未保存的进度会重做。其他错误直接记录为失败。降低缓存 batch 会生成新的缓存目录。最终状态和重试记录均保存在 `pipeline.json`。

数据字段、时间窗与力矩预处理继承指定教师，normalizer 使用教师 checkpoint。默认按整条 episode 留出 10% 验证数据；使用相同 board 数据和 seed=42 时延续教师的 episode 划分。阶段重采样关闭，不生成学生训练不需要的 contact/alpha 标签。

## 信息保留与诊断的边界

RenderNet 输出逐像素 RGB 残差；粗 token MSE 能通过冻结 DINO 反传到像素，但不能唯一决定每个像素的变化。较低 token loss 不能单独证明完整保留图像细节、力矩信息或任务性能。

已有[旧教师小规模干预报告](../model_output/diagnostics/visual_torque_transfer_20261114_011928/report.md)说明旧教师在固定灰图下，视觉位置会随力矩变化且影响 ACT。它不能证明新教师或新学生的效果。旧 2×2 context-swap 依赖独立力矩输出位置，只适用于旧 `all` 接口。

新接口适合在固定图像下比较真实、置零、错配力矩的渲染 RGB、视觉 token 和动作误差；同时检查跨相机输入改变是否影响另一相机的渲染结果。特征或动作发生变化说明敏感性；只有与对应示范误差或实际任务表现结合，才能讨论是否带来收益。
