# Contact-informed visuomotor interface

新增[共享区域/alpha 预测与响应加权蒸馏](docs/response_weighted_student.md)：使用 `--config-name=train_distill_render_net_student_response`，固定指定的 11 月 12 日教师，先训练共享编码器的区域/alpha 双头，再冻结条件网络并蒸馏 RGB 学生。

教师、RenderNet 和学生蒸馏实现恢复到 **2026-11-12 13:51 的 `ada3341`**。该版本用教师最终 `context_norm` token 监督学生的 DINO 图像 token。版本差异和恢复范围见[历史结构核对](docs/network_revision_2026_11_12.md)。

教师路径：冻结 DINOv3 → 图像投影与嵌入 → 图像查询的 cross-attention / FFN 融合 → `context_norm` → DiT。力矩由两层 LSTM 编码后作为融合的 K/V。

学生路径：图像与力矩 → RenderNet 渲染 → 冻结 DINOv3 → 原图像投影与嵌入 → `context_norm`。学生 token 与教师最终 token 直接计算 MSE，学生图像路径不调用教师的力矩编码、融合 attention 或融合 FFN。教师参数保持冻结，图像梯度回传到 RenderNet。

默认学生使用独立 gate 和 RGB 分支：6 层图像 self-attention、2 层 cross-attention、3 层 decoder，patch size 8、embed dim 128、base channels 32，并启用两处 decoder FiLM。gate 是单通道强度图，没有后续版本增加的 phase/alpha 预测头。前 5 个 epoch 训练强度场，再冻结 gate 并训练完整 RGB 分支；默认预算为 1000 个 epoch。

## 本机训练教师并蒸馏学生

```bash
cd /home/lcy/code/dp_hirol_main
conda activate fdp
export CIVI_DATASET_PATH=data/board_25hz_3cam_lerobotv3_clean_ma5_l1_12p5
python train.py
```

默认入口为 `train_civi_teacher.yaml` 和 `train_distill_render_net_student.yaml`。数据接口沿用当前 xArm 配置：front/right-wrist 双相机、8 帧七轴 `tau_ext_unthresholded` 历史和七维关节位置动作；接触阈值为 L1 12.5 Nm。教师使用历史版本的 contact-window 图像模糊课程：前 25% 保持模糊，中间 50% 逐步恢复，最后 25% 清晰。

无参数 `python train.py` 先训练教师，再将教师 checkpoint 交给学生。每次创建新目录：

```text
data/outputs/YYYY.MM.DD/HH.MM.SS_ffffff_fdp_dis_board/
├── teacher/
├── student/
└── pipeline.json
```

换任务时重新设置 `CIVI_DATASET_PATH`。GPU 可通过 `CUDA_VISIBLE_DEVICES` 选择；多卡可用 `python train_pipeline.py --gpus 0,1 --nproc-per-node 2`。预览命令：

```bash
python train_pipeline.py --dry-run
```

只训练教师：

```bash
python train_pipeline.py dp
```

使用已有教师训练学生：

```bash
python train_pipeline.py distill --teacher-run /path/to/teacher
```

学生使用 `distillation.use_teacher_ffn: false`。这一设置控制学生图像路径；教师目标始终取完整融合之后的最终 `context_norm` 输出。历史配置里“教师和学生都在 FFN 前”的注释不符合实际代码，已修正。

学生验证集按 episode 划分。原有缓存、EMA 和 RGB 阶段选模沿用该历史实现。后续 self-attention/ACT、phase/alpha 和无 FiLM 实验的配置不适用于这次恢复的核心网络。

## 从 U 盘安装到服务器

项目源码、数据和输出通过目录挂载保存，Docker 镜像提供依赖。目标机需要 Linux x86_64、支持 CUDA 12.8 的 NVIDIA 驱动、Docker Compose v2 和 NVIDIA Container Toolkit。

进入 U 盘的 `diffusion-policy-train` 目录执行：

```bash
bash install-current.sh "$HOME/fdp_training"
cd "$HOME/fdp_training/dp_main"
bash docker/run_portable.sh check
bash docker/run_portable.sh dev
```

安装脚本校验文件、导入镜像并解压项目；`check` 检查 GPU、通信、数据和本地 DINO；`dev` 启动后台开发容器 `fdp-dev` 并将项目挂载到 `/workspace`。

VS Code 安装 Remote-SSH 和 Dev Containers 扩展后，连接服务器，再执行 `Dev Containers: Attach to Running Container...`，选择 `fdp-dev`。打开 `/workspace`，选择解释器 `/opt/venv/bin/python`，容器内直接运行：

```bash
cd /workspace
export CIVI_DATASET_PATH=data/board_25hz_3cam_lerobotv3_clean_ma5_l1_12p5
python train.py
```

源码修改和训练输出保存在服务器项目目录。详细步骤见 [服务器迁移与 VS Code 操作说明](docker/SERVER_VSCODE_CN.md)。

## 历史学生空间监督修复

11 月 8 日迁移包的零力对照输入曾沿用原样本的 `contact_alpha`。当前代码会根据零力重新计算 alpha=0，并使用 `context_effect_v2` 缓存标识。该修复涉及学生空间 mask 监督，旧实验和数据文件保留；新方案从头生成教师和学生目标。

针对仍运行旧迁移包的服务器，历史补丁说明及文件保存在 `docker/artifacts/fdp-alpha-counterfactual-fix.patch`。
