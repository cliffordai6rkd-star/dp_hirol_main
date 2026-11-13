# Board：高分辨率 skip 消融

本文件记录旧 800-token 输出的对照实验。当前新默认已改为 196-token 粗视觉输出，且模糊课程采用 FACTR 的 2/3–1/3 比例；复现这里的历史实验时，命令显式恢复旧接口及 10%–80% 的课程。

服务器项目目录：`~/code/dp_hirol_main`。训练容器 `dp-board` 将该目录挂载为 `/workspace`，训练产物保存在服务器项目目录下的 `model_output/`。

## 比较定义

固定双相机 DINO、LSTM 力矩历史、下采样因子 2、联合 self-attention、动作头和训练设置，仅改变 `policy.fusion_highres_skip`：

| 设置 | 最终归一化之前的图像特征 |
| --- | --- |
| `true`（旧实验默认） | `P + U(M - D(P))` |
| `false`（去掉高分辨率 skip） | `U(M)` |

`P` 为原始高分辨率 patch 特征，`D` 为平均池化，`M` 为粗尺度联合 self-attention 的图像输出，`U` 为逐相机双线性上采样。两者之差是高分辨率细节残差 `P - U(D(P))`。关闭开关不删除 Transformer 内部残差；保留更新后的力矩 token，动作头 context 仍为 `[B, 800, 256]`。不能只保留 `U(M-D(P))`，否则同时丢掉粗尺度视觉基底，改变了这项消融的含义。

这个对比检查高分辨率旁路的作用，不能单独证明力矩利用增强。还应结合固定图像下的力矩干预及动作误差，见 [CR-VTF 可视化诊断](crvtf_visual_diagnostics.md)。

## 进入环境

在服务器执行：

```bash
docker exec -it dp-board bash
cd /workspace
```

如果容器尚未创建，可在服务器项目目录执行一次：

```bash
cd ~/code/dp_hirol_main
DEV_CONTAINER_NAME=dp-board bash docker/run_portable.sh dev
```

## 单卡训练无 skip 版本

以下命令与当前主实验采用相同 batch、seed、300 epoch 和全阶段模糊课程，使用新的时间戳目录，从头训练：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name=train_civi_teacher_board_selfattn \
  action_head_backbone=act \
  policy.fusion_downsample_factor=2 \
  policy.fusion_output_mode=all \
  policy.fusion_upsample=true \
  policy.image_blur_schedule.start_fraction=0.1 \
  policy.image_blur_schedule.end_fraction=0.8 \
  policy.fusion_highres_skip=false \
  dataloader.batch_size=128 \
  val_dataloader.batch_size=64 \
  training.seed=42 \
  training.num_epochs=300 \
  training.num_optimizer_steps=null \
  training.early_stopping.enabled=false \
  training.resume=false \
  val_epoch_every=5 \
  'output_dir=model_output/${now:%Y.%m.%d}/${now:%H.%M.%S}_board_act_no_skip_seed42/teacher'
```

对照组将 `policy.fusion_highres_skip=false` 改为 `true`，并把目录标签 `no_skip` 改为 `skip`。两组均从头训练；不要把有 skip 的 checkpoint 当作无 skip 的初始化，否则比较的是微调过程。两组使用同一批训练/验证 episode，并分别以验证集最低 loss 选择 `best.ckpt`。

## 后台运行

在容器内执行，日志和 checkpoint 一起留在 `model_output`：

```bash
run_dir="model_output/$(date +%Y.%m.%d_%H.%M.%S)_board_act_no_skip_seed42"
mkdir -p "$run_dir"
nohup env CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --config-name=train_civi_teacher_board_selfattn \
  action_head_backbone=act \
  policy.fusion_downsample_factor=2 \
  policy.fusion_output_mode=all \
  policy.fusion_upsample=true \
  policy.image_blur_schedule.start_fraction=0.1 \
  policy.image_blur_schedule.end_fraction=0.8 \
  policy.fusion_highres_skip=false \
  dataloader.batch_size=128 \
  val_dataloader.batch_size=64 \
  training.seed=42 \
  training.num_epochs=300 \
  training.num_optimizer_steps=null \
  training.early_stopping.enabled=false \
  training.resume=false \
  val_epoch_every=5 \
  "output_dir=$run_dir/teacher" \
  > "$run_dir/console.log" 2>&1 < /dev/null &
tail -f "$run_dir/console.log"
```

`Ctrl+C` 退出 `tail`，后台训练继续。`teacher/logs.json.txt`、`teacher/training_losses.png` 和 `teacher/checkpoints/` 保存指标、图及权重。

## 可选双卡

用以下入口替换单卡命令开头，并追加 `training.distributed.enabled=true`：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py
```

继续使用 `dataloader.batch_size=128`、`val_dataloader.batch_size=64`。当前 `batch_size_is_global=true`，双卡各分到训练 batch 64、验证 batch 32；不要为扩卡同时放大全局 batch，以免改变优化预算。
