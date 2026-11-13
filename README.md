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

新的 self-attention 教师默认输出粗网格的融合视觉 token。冻结 DINOv3 输出的每路 14×14 patch 网格先独立平均池化到 7×7，再与两层 LSTM 保留的 8 个力矩历史 token 经过两层联合 self-attention；每个观察时刻参与融合的 token 为 98+8=106。融合后只将 98 个视觉位置送入动作头，力矩位置保留在融合内部，仍接收通过视觉输出反传的梯度。默认关闭高分辨率 skip 和上采样，双相机、两帧观测的 ACT/DiT context 为 `[B, 196, 256]`。这移除了高分辨率视觉旁路和动作头直接读取力矩位置的通路；粗视觉特征及 Transformer 内部残差仍存在，不能仅凭结构保证模型利用力矩。

当前教师不使用图像查询的融合 cross-attention，也不使用离线 alpha 调制融合。CIVI 教师的像素模糊课程保持 `image_blur_scope: all`，对 free、pre-contact、contact 全阶段的所有观察时刻、所有相机整图施加相同模糊；验证和推理始终使用清晰图像，连续力矩输入不受模糊课程影响。模糊在 DINO 之前执行；重新启用旧高分辨率 skip 时，该路径也来自模糊图像，不会恢复已丢失的清晰细节。

`policy.image_blur_schedule.total_steps: null` 表示在训练 dataloader 建立后，自动取本次计划的 optimizer 更新总数作为课程分母：epoch 模式为每轮更新次数乘以 `training.num_epochs`，step 模式为 `training.num_optimizer_steps`；每轮更新次数计入梯度累积和 DDP 每卡实际批次数。按 FACTR 默认比例，前 2/3 保持最大模糊（sigma=5），最后 1/3 将 sigma 从 5 线性降到 0，不单独保留完全清晰的尾段；这是训练进度比例，不是预留部分 episode。总预算 300 epoch 时对应前 200 轮最大模糊、后 100 轮逐渐清晰。恢复训练沿用 checkpoint 的 optimizer step 继续计算比例，不从 0 重启。旧实验显式配置的固定总步数和接触窗口模糊范围仍可使用。

配置顶层 `action_head_backbone: act` / `dit` 选择动作头，新配置默认 ACT。ACT 使用 FACTR 的可学习动作 query 和解码器，一次前向预测完整动作序列；默认对归一化动作使用 L1，可用 `policy.act_loss_type=mse` 切换。DiT 保留加噪训练和扩散采样。两种动作头共享相同的 DINO、LSTM、融合和动作时间对齐设置：预测 9 步，取索引 1–8 的 8 步用于执行。新配置默认 6 层动作解码器，动作头内部不再额外编码 context。

```bash
# board：先使用 ACT，输出写入新的时间戳目录
python train.py --config-name=train_civi_teacher_board_selfattn action_head_backbone=act

# 相同视觉/力矩融合，改用 DiT 动作头
python train.py --config-name=train_civi_teacher_board_selfattn action_head_backbone=dit

# 对照实验：不池化，仍只输出融合视觉位置（784 tokens）
python train.py --config-name=train_civi_teacher_board_selfattn action_head_backbone=act policy.fusion_downsample_factor=1 policy.fusion_upsample=true policy.fusion_highres_skip=true
```

`policy.fusion_downsample_factor: 2` 为当前默认值。`fusion_output_mode: visual` 只输出融合视觉位置；`fusion_upsample: false` 直接输出粗网格，要求 `fusion_highres_skip: false` 且下采样因子大于 1。设为上述 dense 命令时，`fusion_upsample`/`fusion_highres_skip` 不产生插值或额外 skip，实际使用完整 14×14 网格联合融合。改变输出布局后，应从新实验目录训练或有意微调，不能当作原模型的等价续训。

旧 800-token CR-VTF 模式可以显式设 `policy.fusion_output_mode=all policy.fusion_upsample=true policy.fusion_highres_skip=true`：输出 `P + U(M - Pool(P))` 及融合力矩位置。该模式下单独设 `fusion_highres_skip=false`，输出变成 `U(M)`，仍然上采样。旧 checkpoint 缺省这些新字段时保留原来的 `all` 和上采样行为。此前服务器的 skip 对照实验见 [board skip 消融说明](docs/board_skip_ablation_server.md)，其中记录的是旧 800-token 接口。

当前教师实验的完整产物统一保存到仓库根目录的 `model_output/YYYY.MM.DD/HH.MM.SS_board_selfattn_act/teacher/`：Hydra 的 `.hydra/` 配置快照、`train.log`、`logs.json.txt`、`training_losses.png`、`val_loss.png` 和 `checkpoints/` 均使用同一个实验目录。通用 self-attention 配置使用同一根目录，实验名为 `civi_selfattn_act` 或 `civi_selfattn_dit`。请在仓库根目录运行启动命令。

在单个 workspace 的 `train.py` 命令中加 `--overwrite`，会清空本次选定输出目录里的旧 checkpoint、指标、曲线及其他产物，重置日志，保留本次 Hydra 配置快照并从头训练；它优先于续训参数。该开关只对本次命令生效，之后读取保存配置续训不会再次清空目录。DDP 只由 rank 0 清理，其他 rank 等待清理完成。要反复覆盖同一个实验，显式指定固定的 `output_dir`：

```bash
python train.py --config-name=train_civi_teacher_board_selfattn \
  action_head_backbone=act \
  policy.fusion_downsample_factor=2 \
  val_epoch_every=5 \
  output_dir=model_output/board_selfattn_act/teacher \
  --overwrite
```

不指定固定输出路径时，默认仍创建新的时间戳目录；`--overwrite` 不会自动查找或覆盖最近一次实验。配置预览（例如 `--cfg job`）不会执行清理。无参数的教师/学生流水线入口保持原有行为，此开关用于显式选择训练配置的命令。

教师默认在每轮第一次更新、每 50 个 optimizer steps 和该轮最后一次更新时打印 loss、学习率及 blur sigma；每轮结束打印全 rank 汇总的 `epoch_train_loss`，验证完成后打印清晰图像下的 `val_loss` 和 action MSE。DDP 的逐步 loss 明确标为 rank 0，避免与全 rank 的 epoch 均值混淆。`logging.enable_plots: true` 启动独立 CPU 绘图进程，每 10 秒更新 `training_losses.png`，退出时补画最终记录；设为 `false` 可只保留打印和 JSON 日志。训练 objective、归一化 action MSE、原始单位 action MSE 分开显示，没有验证记录时不画验证曲线。

对已经保存的教师日志，可离线画图、打印最新数值或持续监看：

```bash
python -m diffusion_policy.scripts.plot_teacher_losses model_output/日期/时间_board_selfattn_act/teacher
python -m diffusion_policy.scripts.plot_teacher_losses model_output/日期/时间_board_selfattn_act/teacher --watch --interval 10
```

无需重训的 CR-VTF 信息保留与力觉利用诊断方案见 [可视化实验设计](docs/crvtf_visual_diagnostics.md)。该方案区分结构不变量、特征变化和动作预测收益，不把 attention 热图单独作为有效融合的证据。

新 self-attention 教师最多训练 1000 个 epoch。顶层 `val_epoch_every: 5` 控制每 5 个完整 epoch 验证一次，数据按整条 episode 划分 90% 训练、10% 验证（seed 42），归一化统计只用训练窗口。最低 `val_loss` 对应 `checkpoints/best.ckpt`，最近一次保存的状态为 `latest.ckpt`。ACT 的 `val_loss` 默认是归一化动作 L1；DiT 的 `val_loss` 是固定随机种子的去噪 MSE。

`training.early_stopping.patience: 100` 按 epoch 计数，从首次验证开始：如果第 5 个 epoch 最好，之后一直没有更低的验证 loss，就在第 105 个 epoch 停止，即连续 20 次验证没有改善。改善后重新计数，计数状态随 checkpoint 保存并在续训时恢复。早停不等待模糊课程结束；课程百分比按配置的计划训练预算计算，不按提前停止后的实际长度重新缩放。因此最多 1000 epoch 时，约前 667 epoch 保持最大模糊，余下约 333 epoch 逐步清晰；若早停更早触发，不保证完成课程。可用 `val_epoch_every=10` 调整验证间隔。

这些命令只训练教师。切换动作头需要新训练，不能直接续训另一种头的 checkpoint。新的 `train_distill_render_net_board_selfattn` 学生入口对齐教师完整的视觉 context：当前服务器 visual 教师保留上采样和 skip，输出 784 token；本地默认无上采样版本输出 196 token。学生按 checkpoint 决定是否池化冻结 DINO 特征。旧图像查询教师的学生入口保留原接口。ACT 和 DiT 的训练 loss 含义不同，应使用同一验证集上的 `val_action_mse` 比较动作预测。

原图像查询教师默认每 10000 个 optimizer step 进行全量动作推理评估。所有轨迹的有效窗口各评估一次，EMA 模型使用按 episode 和窗口固定的推理噪声；总平方误差除以动作元素数得到 action MSE。最低分更新 `best.ckpt` 并保留 Top-K，每次评估更新 `latest.ckpt` 和 `action_mse.png`。该旧配置使用全部轨迹，因此指标衡量训练集拟合程度。其 pre-contact 模式还使用离线 alpha；新 self-attention 教师不使用该门控。

历史默认学生入口 `train_distill_render_net_student` 采用两个独立的 CNN 路径。条件网络使用 64 维特征、无图像 self-attention、一层力矩 cross-attention，在 patch 网格预测 region，经因果 GRU 预测接触阶段和 alpha；三者仍共享条件网络内部特征。RenderNet 使用 96 维特征、两层图像 self-attention、两层残差 cross-attention 和三层 RGB decoder，基础通道数为 16。两条路径分别复制并冻结教师的力矩投影、LSTM 和相关嵌入，不共享可训练的视觉编码器。RGB decoder 直接解码融合特征，最终输出为原图加上由 alpha、region 门控的 RGB 残差。

当前 RenderNet 已移除 decoder FiLM 的模块、参数、配置和前向分支，旧 FiLM on/off 对比命令随之移除；历史实验输出保留。新学生需从头训练，不能直接恢复含 FiLM 参数的旧模型及优化器状态。旧的 FiLM-disabled 配置可按相同无调制结构加载。

新 board self-attention 学生先让相机特征交流，再分别输出 RGB；region、contact-alpha、phase 条件分支和阶段筛选全部关闭。学生先复制教师的力矩投影、LSTM 及相关位置/模态嵌入，再冻结整套力矩编码；单阶段只训练 RenderNet 的视觉融合、跨相机注意力和 RGB 渲染分支。DINO、图像投影、context norm 和 ACT 也保持冻结。全部观测使用 token MSE、归一化力矩差分 MSE 和教师上下文区域约束的加权和训练、验证及选模；不使用动作损失。学生本轮使用清晰图像；模糊课程尚未实施。显式指定新 visual 教师 checkpoint：

```bash
python train_distill_render_net.py \
  --config-name=train_distill_render_net_board_selfattn \
  teacher_checkpoint=/absolute/path/to/new_teacher/checkpoints/best.ckpt \
  dataloader.batch_size=16 \
  training.num_epochs=100
```

该入口的 checkpoint、配置、日志与曲线存入 `model_output/<日期>/<时间>_board_selfattn_student/student`。训练前将原图真实/物理零力矩的教师 token 配对以 FP32 存入 `model_output/cache/render_net_teacher`，退出后保留供复用；加 `teacher_cache.only=true` 可只缓存并统计固定尺度。尺度仅由训练缓存计算，验证集和逐样本归一化不参与。区域项按教师力矩响应对真实分支的实际 RGB 修改加权惩罚，不增加 region head。两个新增权重初值均为 1，可用 `distillation.force_context.difference_weight=...`、`distillation.force_context.region_weight=...` 调整，并非已调优值。曲线分别记录 token MSE、原始/归一化差分、区域项及综合选模损失，详见[学生适配说明](docs/crvtf_student_adaptation.md)。

历史默认入口的训练严格分成两个阶段，`training.render_stages.joint_after_rgb: false` 表示不再进入联合训练：

1. 条件阶段最多 20 个 epoch，仅优化阶段 CE、alpha MSE 和区域损失，不运行 RGB 渲染蒸馏。验证以这些条件损失的加权和选最佳模型，连续 5 轮无改善则提前切换。
2. 加载验证最好的条件权重（默认使用 EMA），完整冻结条件网络及其运行模式；render 的独立编码器、交叉注意力和 decoder 按 token MSE 训练。该阶段最多 100 个 epoch，沿用独立重置的学习率调度和早停计数；条件阶段提前结束不会缩短这 100 轮预算。

`condition_backbone` 可独立调整条件网络大小，`render_net.backbone` 调整 render。历史默认两阶段学生需从头训练，旧共享结构的 checkpoint 不能直接恢复成独立结构；已有训练使用的配置与权重不变。原 DINO LoRA 联合训练配置保留为 `train_distill_render_net_student_dino_joint.yaml`。旧 board 图像查询教师的两阶段实验配置为 `train_distill_render_net_student_cnnvit_condition_then_render.yaml`，输出使用新的时间戳目录：

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
