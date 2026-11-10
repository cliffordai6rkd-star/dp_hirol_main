# FDP 教师、学生蒸馏和 CAD-WM

本次迁移使用 Docker 镜像提供依赖，项目源码、数据和输出通过目录挂载保存。U 盘包包含当前工作区，尚未提交到 Git 的修改也在其中。

## 本机训练 CAD-WM

```bash
conda activate cad_wm
cd cad-wm
CUDA_VISIBLE_DEVICES=0 python -m train.trainer.cad_wm_train \
  --config config/bench_train_config/libero_drawer_wrench_h5_portable.yaml
```

该配置使用 batch size 256、20 万 optimizer steps、每 2 万步保存。

## 本机训练 FDP 教师并蒸馏学生

```bash
cd /home/lcy/code/dp_hirol_main
conda activate fdp
export HF_HOME=/tmp/hf_home
export HF_DATASETS_CACHE=/tmp/hf_home/datasets
export TRANSFORMERS_CACHE=/tmp/hf_home/transformers

python train_civi.py \
  --teacher-config diffusion_policy/config/train_civi_teacher.yaml \
  --student-config diffusion_policy/config/train_distill_render_net_student.yaml \
  --gpus 0 \
  --teacher-override dataloader.batch_size=128 \
  --teacher-override val_dataloader.batch_size=128 \
  --teacher-override training.num_optimizer_steps=200000 \
  --teacher-override training.checkpoint_every_optimizer_steps=20000 \
  --teacher-override checkpoint.action_mse.every_optimizer_steps=10000 \
  --teacher-override checkpoint.topk.k=4 \
  --teacher-override training.validation_start_optimizer_step=200001 \
  --teacher-override output_dir=data/outputs/civi/fdp_bs128_200k/teacher \
  --student-override output_dir=data/outputs/civi/fdp_bs128_200k/student
```

程序会先完成教师训练，再自动选择教师 checkpoint 并训练学生。输出目录必须是新目录。

教师默认每 10000 个 optimizer step 做全量动作推理评估：全部轨迹的有效窗口各评估一次，不做强化采样或随机图像增强，EMA 模型使用按 episode 和窗口固定的推理噪声。按总平方误差除以动作元素数得到 action MSE，选取最低分更新 `best.ckpt` 并保留 Top-K；每次评估后都更新 `latest.ckpt`，即使没有进入 Top-K。输出目录中的 `action_mse.png` 随评估自动刷新。当前 50 条轨迹都参与训练，因此这条曲线表示训练集拟合程度，不是独立验证集性能。

只训练教师时添加 `--stage teacher`。`checkpoint.action_mse.episode_selection=all` 与 `seed=42` 定义全量固定噪声协议；旧配置缺少该字段时仍按原来的随机轨迹规则评估。切换协议后，已有 checkpoint 应统一重评，不能将新旧分数混用于 Top-K。可使用 `diffusion_policy/scripts/reevaluate_checkpoint_action_mse.py` 将重评结果生成到独立目录，核对后再替换原评分和 checkpoint 目录。

学生配置默认关闭 `distillation.use_teacher_ffn`，教师和学生都直接使用融合前的 context token，避免蒸馏目标依赖教师 FFN。学生监督路径不运行教师 wrench 编码器或 cross-attention；alpha=0 时与教师的纯视觉基线一致。仅更新 RenderNet，部署渲染仍只需要 RenderNet 和原有输入预处理。旧学生 checkpoint 未设置该选项时，评估沿用原监督路径。

复用已有教师、仅重新训练学生：

```bash
python train_civi.py --stage student \
  --teacher-config diffusion_policy/config/train_civi_teacher.yaml \
  --student-config diffusion_policy/config/train_distill_render_net_student.yaml \
  --teacher-checkpoint data/outputs/civi/teacher/checkpoints/optimizer_step=00180000-action_mse=0.00967494.ckpt \
  --gpus 0 \
  --student-override output_dir=data/outputs/civi/student_teacher_ffn
```

## 从 U 盘安装到服务器

目标机需要 Linux x86_64、支持 CUDA 12.8 的 NVIDIA 驱动、Docker Compose v2、NVIDIA Container Toolkit，以及当前用户的 Docker 使用权限。将项目解压到服务器 SSD 后训练。

进入 U 盘的 `diffusion-policy-train` 目录，执行：

```bash
bash install-current.sh "$HOME/fdp_training"
cd "$HOME/fdp_training/dp_main"
bash docker/run_portable.sh check
bash docker/run_portable.sh dev
```

- `install-current.sh` 校验文件、导入镜像、解压本次项目，不启动训练。
- `check` 检查 GPU、通信、数据和本地 DINO。
- `dev` 启动后台开发容器 `fdp-dev`，挂载当前项目到 `/workspace`，等待 VS Code 连接。

U 盘使用原来的四个镜像分片和本次 `dp_main-current-source-data-model.tar.gz`。项目包保留四份 LeRobot 数据、世界模型原始 H5、DINO 预训练权重和 Git 元数据；排除历史训练输出、checkpoint 目录、可重建缓存和本机私有配置。DINO 是训练输入，随包保留。默认在服务器从头训练。

## 用 VS Code 进入 Docker

1. 本地 VS Code 安装 Remote-SSH 和 Dev Containers 扩展。
2. 用 Remote-SSH 连接服务器。
3. 在远程窗口按 F1，执行 `Dev Containers: Attach to Running Container...`，选择 `fdp-dev`。
4. 打开 `/workspace`，选择解释器 `/opt/venv/bin/python`。

容器内直接使用 `python`，无需激活 Conda。源码修改和训练输出都会保存在服务器项目目录。

容器内训练世界模型：

```bash
cd /workspace/cad-wm
CUDA_VISIBLE_DEVICES=0 python -m train.trainer.cad_wm_train \
  --config config/bench_train_config/libero_drawer_wrench_h5_portable.yaml
```

容器内训练教师并蒸馏：

```bash
cd /workspace
python train_civi.py --gpus 0 \
  --teacher-override output_dir=data/outputs/civi_server/teacher \
  --student-override output_dir=data/outputs/civi_server/student
```

训练长任务可使用 `nohup` 并将日志写到输出目录，避免依赖 VS Code 终端连接。停止容器会停止其中的训练；使用 `docker stop fdp-dev` 停止开发容器。

详细步骤、后台训练示例和文件说明见 [服务器迁移与 VS Code 操作说明](docker/SERVER_VSCODE_CN.md)。

## 已迁移服务器的学生蒸馏修复（2026-11-09）

11 月 8 日的迁移包包含一处学生空间 mask 监督生成的遗漏：零力对照输入沿用了原样本的 `contact_alpha`。修复后会根据零力重新计算 alpha=0，正常样本的标签保持不变。教师和 CAD-WM 的正常训练不经过该分支，已有教师和世界模型权重可继续使用。

本机代码已修复，同时将 context-effect 教师目标缓存标识从 `context_effect_v1` 更新为 `context_effect_v2`，新启动的学生会生成新的监督缓存。无需删除数据集、模型或历史输出。

对已经迁移到服务器的旧源码，将补丁 `fdp-alpha-counterfactual-fix.patch` 上传到服务器项目根目录。在 VS Code 容器的 `/workspace` 中执行：

```bash
cd /workspace
git apply --check fdp-alpha-counterfactual-fix.patch
git apply fdp-alpha-counterfactual-fix.patch
```

补丁可从本机 `docker/artifacts/fdp-alpha-counterfactual-fix.patch` 获取，包含修复代码、回归测试和这段操作说明。服务器无法直接读取本机文件，需通过 VS Code 或文件传输先上传补丁。已经包含修复的代码无需重复应用。

在容器中验证：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/test_contact_alpha.py tests/test_render_net_teacher_cache.py \
  -q -k 'context_effect_targets_drop_cached_alpha or context_effect_cache_does_not_reuse'
```

若学生蒸馏已开始，先停止学生进程，应用补丁后使用新输出目录重新蒸馏；避免恢复已经用旧 mask 监督训练的学生 checkpoint。教师尚在训练时无需为此中断教师。下面以已有教师目录 `data/outputs/civi_server/teacher` 为例：

```bash
python train_civi.py --stage student --gpus 0 \
  --teacher-override output_dir=data/outputs/civi_server/teacher \
  --student-override output_dir=data/outputs/civi_server/student_alpha_fix
```

如果教师保存在其他目录，替换上述教师 `output_dir`。这次修复不要求重新构建 Docker 镜像或复制训练数据。


(fdp) lcy@lcy-RTX5090:~/code/dp_hirol_main$ CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/tmp/hf_home \
HF_DATASETS_CACHE=/tmp/hf_home/datasets \
TRANSFORMERS_CACHE=/tmp/hf_home/transformers \
python -u sim_force_aware_dp_train.py \
  --config diffusion_policy/config/train_civi_teacher.yaml \
  dataloader.batch_size=128 \
  dataloader.num_workers=16 \
  dataloader.prefetch_factor=8 \
  val_dataloader.batch_size=128 \
  task.dataset.image_cache_chunk_frames=512 \
  training.resume=true \
  training.num_optimizer_steps=200000 \
  training.checkpoint_every_optimizer_steps=20000 \
  checkpoint.action_mse.every_optimizer_steps=5000 \
  training.validation_start_optimizer_step=100001 \
  checkpoint.topk.k=4 \
  hydra.run.dir=data/outputs/civi/teacher_bs128_100k_topk4_20261109 \
  hydra.job.chdir=false
