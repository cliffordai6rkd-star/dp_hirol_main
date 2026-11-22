# civi

CIVI 统一为一条 LeRobot v3 教师—学生训练链：教师使用 contact/precontact 的 mask token 课程与力觉融合；学生依次训练 phase、context 空间强度和 RGB RenderNet。

教师：冻结 DINOv3 → 图像投影 → **一层** image-query cross-attention / FFN → context_norm → DiT。只有 precontact/contact 启用完整融合，free 整体旁路。训练时仅遮蔽这两个阶段的视觉 token：前 25% 更新遮蔽概率为 1，25%–75% 线性降为 0，最后 25% 为 0；验证、推理和学生蒸馏目标均不遮蔽。教师不使用图像 blur 课程。

教师和学生的 `tau_ext` 都读取 `observation.tau_ext`，对七轴向量施加 L1 阈值 12.5：`sum(abs(tau_ext)) < 12.5` 时整向量置零，其余保留。这里是关节外力矩，单位为 N·m。顶层 `tau_ext_history_steps` 控制历史帧数，默认 25 帧，原始采样率保持 25 Hz，约 1 秒（`t-24…t`）；episode 开头缺失的历史补零，第一帧为 24 个零向量加当前测量。学生自动继承教师 checkpoint 的帧数。物理零经归一化后仍保持零。阶段标签由同一 episode 的接触规则生成，precontact 窗口默认 0.5 秒，未来接触信息只用于训练标签。

学生使用独立预测头和 RGB 渲染主干，按下列顺序训练：

| 阶段 | 训练目标 | 收敛后的处理 |
| --- | --- | --- |
| phase | 历史双视角 RGB 与 joint/velocity/阈值后 tau_ext → free/precontact/contact 三分类交叉熵 | 恢复验证集最优权重，冻结 phase 头 |
| 空间强度 | 预测教师 context 的力矩响应空间强度场 | 恢复验证集最优权重，冻结独立空间 gate |
| RenderNet | 预测 phase 开关与空间 gate 控制 RGB 残差，通过冻结教师视觉编码器，以最终 context token MSE 蒸馏 | 保存验证集最优 RGB 学生 |

每阶段验证损失连续 5 轮无至少 0.0001 改善后进入下一阶段，RGB 阶段则停止；不按固定 5 个 epoch 冻结头。冻结后的 phase 与空间强度头保持 eval，不再更新参数或统计量。训练和推理均使用预测 phase，free 旁路，precontact/contact 开启渲染。precontact 当前力矩经阈值变成零仍允许开启渲染；阶段切换按三分类 argmax 直接切换，不额外加渐变或防抖。所有阶段合计最多 3000 epoch，达到此上限不表示各头已经收敛。

RGB 蒸馏的 token loss 仅统计真实标签为 precontact/contact 的观察，包含这些观察的全部相机 patch，并按有效 token 数归一化；free 不参与 token loss。损失使用真实阶段标签，渲染仍使用学生预测阶段，因此误判为 free 的真实 precontact/contact 仍计入误差。全 free batch 的 token loss 为可反向传播的零。

## 训练

```bash
cd /home/lcy/code/civi
conda activate fdp
python train.py
```

唯一核心配置为 `diffusion_policy/config/train_civi_teacher.yaml` 和 `diffusion_policy/config/train_distill_render_net_student.yaml`。默认 board 教师 batch 32、150,000 个优化步，随后使用最终教师 EMA 权重训练 batch 16 的学生。新实验写入 `data/outputs/YYYY.MM.DD/<时间>_<任务>/`，包含 `teacher/`、`student/` 和 `pipeline.json`。

```bash
# 预览，不启动训练
python train_pipeline.py --dry-run

# 只训练教师
python train_pipeline.py dp --gpus 0

# 已有教师
python train_pipeline.py distill --teacher-checkpoint /path/to/teacher/checkpoints/latest.ckpt --gpus 0

# 同一版本，更换为黄瓜数据
CIVI_DATASET_PATH=data/huanggua1103_25hz_3cam_lerobotv3_clean_ma5_l1_12p5 python train.py

# 历史帧数和 horizon 一样通过顶层参数覆盖，学生自动继承
python train_pipeline.py --dp-override tau_ext_history_steps=25
```

默认数据为 `data/board_25hz_3cam_lerobotv3_clean_ma5_l1_12p5`，可由 `CIVI_DATASET_PATH` 覆盖；本地 DINO 权重为 `model/model/dinov3-vitb16-pretrain-lvd1689m`。教师和学生默认采用相同 seed 与 episode 验证划分，学生继承教师的数据预处理。独立运行学生时用 `teacher_checkpoint=/path/to/latest.ckpt`，或 `teacher_output_dir=/path/to/completed/teacher` 自动读取最终 `latest.ckpt`；教师 YAML 的时间戳默认输出路径不用于搜索以前的实验。单卡为默认，多卡使用 `train_pipeline.py --nproc-per-node ...`。

旧名称含 `blur`、`future_tau_ext`、`legacy` 的 YAML 仅保留为统一配置的兼容别名；它们现在也使用可配置历史帧数与 mask token，不再代表旧实现。64 batch、100k、600 epoch 别名仅改变训练预算，huanggua 仅改变数据。原实验的 checkpoint 与其保存的配置仍按原语义加载，历史数据和输出保留。

## 推理与验证

所有调试、诊断、小规模对照实验和性能测试的结果统一保存到 `data/debug/`，每次实验单独建子目录；结果入口见 [测试结果索引](data/debug/README.md)。

phase 头使用每观测 `tau_ext_history_steps` 帧 joint/velocity/tau_ext 和双相机 `[-12,-8,-4,0]` 帧 RGB，归一化仅拟合训练 episode 并随 checkpoint 保存。推理接口见[阶段预测与训练设计](docs/causal_phase_gate.md)。

```bash
python render_distill_render_net.py --checkpoint /path/to/student/checkpoints/best.ckpt --episode 0   --camera both --device cuda:0
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```
、、、bash
python -m diffusion_policy.scripts.visualize_wrench_effect_episode \
  --checkpoint "$RUN/teacher/checkpoints/latest.ckpt" \
  --episode 0 \
  --metrics context_effect \
  --baseline raw_zero \
  --camera both \
  --device cuda:0
、、、

Python 包仍叫 `diffusion_policy`，以读取已有 checkpoint；项目与实际仓库目录为 `civi`，旧路径 `dp_hirol_main` 为兼容符号链接。新环境可运行 `conda env create -f conda_environment.yaml`。历史清理记录见[清理记录](docs/cleanup_20261116.md)。

