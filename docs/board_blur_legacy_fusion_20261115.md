# Board：旧版完整融合门控与像素模糊课程

本实验以 `2026.11.07/22.23.10_distill_gate_sft_v2` 的学生结构和蒸馏目标为参考，
在 board 数据上重新训练教师，再使用最终 150,000 优化步的 EMA 教师蒸馏学生。
不从已有教师或学生 checkpoint 续训。

## 已确认设置

| 项目 | 本实验 |
| --- | --- |
| 数据集 | `data/board_25hz_3cam_lerobotv3_clean_ma5_l1_12p5` |
| RGB | front→side、right_wrist→wrist；224×224；2 个观察时刻 |
| 力觉 | 当前及过去共 8 帧、7 轴关节外力矩 `observation.tau_ext_unthresholded` |
| 教师融合 | 保留 2 层 image-query cross-attention；每层保留 FFN |
| 完整融合门控 | precontact/contact 打开，free 关闭；不叠乘连续 alpha |
| 教师预算 | 启动时覆盖为 batch 32；150,000 个 optimizer updates |
| 模糊 | precontact/contact 的 RGB；free 保持清晰 |
| 模糊课程 | 前 50,000 步 σ=5；随后 100,000 步线性下降至 0 |
| 蒸馏教师 | 完成训练后的 `latest.ckpt`，即最终 150,000 步；不是 best |
| 学生 | CNN + LSTM；self-attn/CA/decoder=4/2/4；patch=16 |
| 渲染 | 独立 gate、FiLM、residual scale=1；sigmoid gate threshold=0.5 |
| 学生阶段 | 前 5 epoch 训练空间 gate 与阶段子网络；其后冻结空间 gate、训练 RGB 与阶段子网络 |
| 学生早停 | batch 16；从 RGB 阶段计数，patience=5，min_delta=0.0001；最多 1000 epoch |
| 蒸馏目标 | 教师最终 `context_norm` tokens 的 MSE；学生不重放教师 FFN |

教师训练/验证划分、其余优化配置继承 `train_civi_teacher_curriculum_b64_150k.yaml`；
学生数据划分、学习率、EMA、早停和缓存设置继承 `train_distill_render_net_student.yaml`。
预接触标签窗口保留当前配置的 0.5 秒。

## 前向与目标

`policy.fusion_gate_mode: phase_binary` 选择新实验行为。该模式先保留视觉 tokens，
执行全部融合层，再按每个观察帧的二值 gate 选择完整融合结果或原始视觉 tokens，
最后执行 `context_norm`。free 帧因此同时旁路 attention 和 FFN。

教师使用真实阶段的二值 gate；学生通过历史观测预测 `p_open >= 0.5`，预测关闭时原样返回 RGB。读出路径仅包含冻结 DINO、
教师视觉投影/位置与相机嵌入、最终归一化。教师生成蒸馏目标时关闭模糊课程。
context-effect 的 raw-zero 对照同时清零力矩和阶段标签，使完整融合分支关闭。

未指定该模式的既有配置继续使用原来的 `attention_alpha` 行为。
教师目标和阶段 BCE 监督需要数据集提供 `contact_phase`（0/1/2）。2026-11-16
增加了用户确认的独立 CNN + GRU 预测器；学生训练和推理均使用预测开关，推理不读取离线阶段标签。
结构与输入历史见[阶段子网络设计](causal_phase_gate.md)。

## 启动

```bash
/home/lcy/miniconda3/envs/fdp/bin/python -u train_pipeline.py all \
  --dp-config diffusion_policy/config/train_civi_teacher_board_blur_legacy_fusion.yaml \
  --distill-config diffusion_policy/config/train_distill_render_net_board_legacy.yaml \
  --teacher-selection latest --gpus 0 --dp-override dataloader.batch_size=32
```

流水线在教师正常退出后衔接学生。教师失败时不启动学生。
每次运行的 `pipeline.json`、阶段 `train.log`、Hydra 配置和 checkpoints 位于该次输出目录。

`training.early_stopping.start_at_rgb=true` 使前 5 个 gate epoch 不累计早停次数，
进入 RGB 阶段时重置早停基准和计数。gate 阶段仍保留验证与日志。
2026-11-16 用户确认，本机 board 和服务器 huanggua 均采用此设置。

## 验证

`tests/test_phase_binary_fusion.py` 检查完整旁路、逐阶段梯度、alpha=0 的预接触边界、
raw-zero 目标、学生训练/渲染一致性、预处理配置回存，以及课程和结构配置。
真实 board 的 free/precontact/contact 样本另行通过 GPU 前向/反向检查；
未训练模型的 smoke-check loss 仅用于检查有限性和梯度，不代表训练质量。
