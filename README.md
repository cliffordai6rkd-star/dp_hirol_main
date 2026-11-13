# Contact-informed visuomotor interface

通过 RenderNet 将机器人当前的接触状态表征到 RGB 图像中，再交给原有视觉策略。教师读取双摄像头图像和当前 `tau_ext` 外部关节力矩观测；学生学习渲染与教师接触表征对应的图像，保持后续视觉编码器与动作策略的网络架构。

默认 `tau_ext` 输入是每个观察时刻截至当前帧的 8 个七轴观测，即 `[t-7, ..., t]`。训练配置使用因果处理后的连续 `observation.tau_ext_unthresholded`，接触阈值为七轴 L1 范数 12.5 Nm；输入不再被阈值硬置零。教师和学生均不读取未来力矩，也不需要世界模型预测。离线训练标签使用确认接触前 0.5 秒的线性 alpha：从 0 逐步增加到接触时的 1，接触期间保持 1，其余时刻为 0；阶段标签对应 free、pre-contact、contact。学生从图像和已观测力矩预测阶段与 alpha，推理时不需要离线标签。保留两个图像观察时刻、原有 LSTM、交叉注意力和动作 Transformer 的隐藏维度及层数。

启用 pre-contact 后保留阈值以下的连续力矩，不再把非接触输入硬置零。`train_civi_teacher_xarm_clean.yaml` 从 `observation.tau_ext_unthresholded` 读取该信号，接触阈值只用于生成标签。

## 本机训练教师并蒸馏学生

```bash
cd /home/lcy/code/dp_hirol_main
conda activate fdp
export CIVI_DATASET_PATH=data/board_25hz_3cam_lerobotv3_clean_ma5_l1_12p5
python train.py
```

换任务时只需重新 export `CIVI_DATASET_PATH`，例如 `data/huanggua1103_25hz_3cam_lerobotv3_clean_ma5_l1_12p5`。两份数据使用相同的 front/right-wrist 图像、七维 joint action 和 `tau_ext_unthresholded` 字段；网络与训练超参数由两份 YAML 固定。其他数据须符合教师 YAML 中的图像、动作、时间戳及七维力矩字段映射。

无参数 `python train.py` 先训练教师，再选择其 best checkpoint 蒸馏学生。每次自动创建新实验目录，保留日期和时间戳，实验名使用 `fdp_dis_<任务>`：

```text
data/outputs/YYYY.MM.DD/HH.MM.SS_ffffff_fdp_dis_board/
├── teacher/
├── student/
└── pipeline.json
```

任务名取数据目录中采样率前的部分，例如 `board`、`huanggua1103`，也可用 `export CIVI_TASK_NAME=任务名` 指定。GPU 可通过 `export CUDA_VISIBLE_DEVICES=0` 选择；多卡使用 `python train_pipeline.py --gpus 0,1,2,3 --nproc-per-node 4`。默认教师全局 batch size 为 32，学生训练为 16、验证为 32。预览完整命令可运行 `python train_pipeline.py --dry-run`；该入口也支持 `--output-dir` 显式指定新实验目录。

当前 `tau_ext` 教师的输入投影为七维，旧的六维 wrench 教师权重不能直接用于该配置；教师和学生需使用新的输入配置重新训练。

只训练教师：

```bash
python train_pipeline.py dp
```

教师完成后，只启动学生可指定该次教师目录：

```bash
python train_pipeline.py distill --teacher-run data/outputs/日期/时间_fdp_dis_board/teacher
```

需要锁定某份教师权重时，可使用 `python train_pipeline.py distill --teacher-checkpoint /path/to/teacher.ckpt`。显式传入 Hydra 配置的 `python train.py --config ...` 继续运行该单个 workspace。

教师默认每 10000 个 optimizer step 进行全量动作推理评估。所有轨迹的有效窗口各评估一次，EMA 模型使用按 episode 和窗口固定的推理噪声；总平方误差除以动作元素数得到 action MSE。最低分更新 `best.ckpt` 并保留 Top-K，每次评估更新 `latest.ckpt` 和 `action_mse.png`。教师默认使用全部轨迹，且 pre-contact 模式使用离线 alpha，因此该指标衡量带接触时刻标签的训练集拟合程度。

学生默认采用两个独立的 CNN 路径。条件网络使用 64 维特征、无图像 self-attention、一层力矩 cross-attention，在 patch 网格预测 region，经因果 GRU 预测接触阶段和 alpha；三者仍共享条件网络内部特征。RenderNet 使用 96 维特征、两层图像 self-attention、两层残差 cross-attention 和三层 RGB decoder，基础通道数为 16。两条路径分别复制并冻结教师的力矩投影、LSTM 和相关嵌入，不共享可训练的视觉编码器。FiLM 使用 render 的融合特征和预测 region 调制前两层 decoder，最终输出为原图加上由 alpha、region 门控的 RGB 残差。

训练严格分成两个阶段，`training.render_stages.joint_after_rgb: false` 表示不再进入联合训练：

1. 条件阶段最多 20 个 epoch，仅优化阶段 CE、alpha MSE 和区域损失，不运行 RGB 渲染蒸馏。验证以这些条件损失的加权和选最佳模型，连续 5 轮无改善则提前切换。
2. 加载验证最好的条件权重（默认使用 EMA），完整冻结条件网络及其运行模式；render 的独立编码器、交叉注意力、decoder 和 FiLM 按 token MSE 训练。该阶段最多 100 个 epoch，沿用独立重置的学习率调度和早停计数；条件阶段提前结束不会缩短这 100 轮预算。

`condition_backbone` 可独立调整条件网络大小，`render_net.backbone` 调整 render。新默认需从头训练学生，旧共享结构的 checkpoint 不能直接恢复成独立结构；已有训练使用的配置与权重不变。原 DINO LoRA 联合训练配置保留为 `train_distill_render_net_student_dino_joint.yaml`。当前 board 教师的两阶段实验配置为 `train_distill_render_net_student_cnnvit_condition_then_render.yaml`，输出使用新的时间戳目录：

```bash
python train_distill_render_net.py --config diffusion_policy/config/train_distill_render_net_student_cnnvit_condition_then_render.yaml
```

`distillation.use_teacher_ffn: true` 让渲染图像复用冻结教师的融合 FFN，对齐教师最终 context token。教师融合使用离线 alpha；学生渲染始终使用自己预测的 alpha，推理不读取接触标签。学生验证集按 episode 划分，条件阶段和 render 阶段分别选择 checkpoint；条件分支的最佳 checkpoint 为 `condition_best.ckpt`，render 的为 `best.ckpt`。

`distillation.token_region_phases: [1, 2]` 根据数据的真实 `contact_phase`，仅对 pre-contact/contact 观测计算 token 和区域损失（包括区域平滑项）；阶段 CE、alpha MSE 继续使用全部观测。训练和验证的 token/region 指标按有效观测数平均，非接触批次不会把它们稀释。设为 `null` 可恢复全阶段监督。切换此设置后恢复 checkpoint 时，会保留权重与优化器状态，重新计算验证基线，重置最佳指标、学习率平台计数和早停计数；旧 best 保存在 `best_previous_metric.ckpt`，曲线也按新指标范围重新显示。

`distillation.skip_unselected_observations: true` 在 render 阶段只对选中观测解码 RGB，并通过冻结教师 DINO 编码渲染结果。条件网络仍从全部观测预测条件，但在 render 阶段不再更新；全非接触批次不产生 token 监督。条件阶段的 CE/alpha 仍使用全部观测。验证完整渲染和编码所有观测，未计算的 token 诊断值不画入曲线。独立 render backbone 与冻结 DINO 的输入梯度路径支持 `torch.compile`。

学生训练默认 `logging.enable_plots: false`，逐步和验证指标保存在学生目录的 `logs.json.txt`，汇总保存在 `training_summary.json`；训练过程中不生成曲线。需要时可离线读取日志生成图，或设为 `true` 恢复自动画图。

实时曲线由独立 CPU 进程生成，每 10 秒读取新增指标，分别显示 token、阶段 CE、alpha、region patch、region pixel、平滑项、region 总损失和总 loss，并覆盖学生目录下的 `training_losses.png`：

```bash
python -m diffusion_policy.scripts.plot_render_net_losses /path/to/student --watch --interval 10
```

也可直接传入 `loss.json` 或 `logs.json.txt` 文件，支持 JSONL 和 JSON 记录数组。`--output /path/to/loss.png` 指定图片路径，`--smooth-window 100` 设置滚动均值窗口；省略 `--watch` 只生成一次。脚本不导入 PyTorch，禁用 CUDA，绘图不会阻塞训练进程。

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
