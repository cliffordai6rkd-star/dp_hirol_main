# Board / huanggua 与 11 月 7 日实验的方法核对

核对基准是 `data/outputs/2026.11.07/22.23.10_distill_gate_sft_v2` 的
实际学生配置、best/latest checkpoint，以及该学生使用的 11 月 5 日最终 200,000 步教师。
本核对先记录阶段子网络加入前、已确认的教师/学生主体；下文新增阶段预测一节说明当前版本。

结论：当前实验沿用了 11 月 7 日的 gate + FiLM RGB 渲染与 context token 蒸馏方法，
并加入了用户确认的教师结构、力觉接口、阶段开关和课程变更。
应描述为该方法在新数据上的迁移复现及像素模糊课程实验；包含的改动多于更换数据集。
方法实现对齐还不能证明已经复现旧实验的训练曲线或任务效果。

## 核心计算与训练路径

| 环节 | 11 月 7 日与当前实验 |
| --- | --- |
| 教师 | 冻结 DINOv3 ViT-B/16；视觉投影与力觉 LSTM 后进行 image-query 融合；最终 context_norm tokens 供 DiT 使用 |
| 融合门控算子 | 关闭时同时旁路 attention、query residual 后的融合 FFN；在最终 context_norm 前返回原始视觉 tokens |
| gate 监督 | 同一图像下，正常力觉与物理零力觉教师 context 的逐 patch L2 差；按每幅图 min-max 归一化 |
| gate 训练 | 前 5 个 epoch 训练独立 gate 分支，使用 patch 场拟合与空间平滑损失；RGB 分支冻结；新增子网络同期训练 |
| RGB 蒸馏 | gate 固定后，用 gate 表征初始化 RGB 分支，再训练 RGB 编码、力觉 LSTM、attention、FiLM 和解码器 |
| 学生读出 | 渲染 RGB → 冻结 DINO / 教师视觉投影、位置与相机嵌入 / context_norm；不执行教师力觉融合与 FFN |
| RGB 目标 | 学生读出 tokens 与正常教师最终 tokens 的普通 MSE；没有启用动作/去噪器蒸馏或额外 response 加权目标 |
| 蒸馏权重 | 训练结束的教师 EMA 权重；当前明确选择最终 150,000 步的 latest，不按 best 选择 |
| 蒸馏图像 | 生成教师监督目标时关闭课程扰动；像素模糊只作用于教师训练 |

学生主体均为 CNN + LSTM，self-attention / cross-attention / decoder = 4 / 2 / 4，
patch 16、embed 128、base_channels 32、独立 gate、FiLM、residual_scale=1，
sigmoid gate threshold=0.5。当前未启用旧 `condition_prediction`、context adapter、dual path 或 RGB epsilon 约束；新增的独立模块使用 `phase_predictor`。

## 已确认并保留的改动

| 项目 | 11 月 7 日参考实验 | 当前 board / huanggua |
| --- | --- | --- |
| 教师融合深度 | 1 层 CA + FFN | 2 层 CA + FFN |
| 力觉信息 | 未来 8 帧六维末端 wrench | 当前及历史共 8 帧七轴关节外力矩 |
| gate 开关来源 | 未来 wrench 的接触判断 | 教师使用离线 contact_phase；学生使用新增子网络预测开关 |
| 教师课程 | 学习到的 token mask；20 万步课程 | 像素模糊；前 50,000 步 sigma=5，后 100,000 步线性降到 0 |
| 教师预算 | batch 32、200,000 步 | batch 32、150,000 步 |
| 图像与 context | 192×256、768 tokens | 224×224、784 tokens |
| 动作维度 | 8 | 7 |
| 学生早停 | patience 30，自预训练阶段累计 | 用户确认改为 RGB 阶段开始累计，patience 5 |

历史与未来力觉的差别涉及可用信息及 gate 时机。二值 gate 算子对齐，
不意味着两种数据接口在同一时刻打开，也不意味着 teacher context 目标相同。
双层融合与像素模糊也会改变教师学习出的目标；这些均属于本次实验明确保留的变量。

## 早停问题与本次修正

原有验证选择依据是 token MSE。gate 预训练期间 RGB 输出层冻结且以零残差初始化，
因此即便 gate 场已在学习，渲染 RGB 仍等于原图，token MSE 不会反映 gate 的改进。
直接把旧逻辑的 patience 改为 5，会在 gate 第 5 个 epoch 结束时触发早停，
无法进入原定第 6 个 epoch 的 RGB 蒸馏。

2026-11-16 用户明确确认两边统一修正为：

1. 学生 batch 16，gate 固定训练 5 个 epoch；此阶段不累计早停次数。
2. 转入 RGB 时重置早停最佳值和无改善计数；首次 RGB 验证建立新的早停基准。
3. RGB 阶段验证 token MSE 连续 5 个 epoch 无至少 0.0001 改善后停止；最多 1000 epoch。

对应开关为 `training.early_stopping.start_at_rgb=true`。
其他配置不指定该开关时继续使用原有计数方式。
学习率调度和最佳 checkpoint 保存仍按原有验证流程执行；教师训练和网络结构未因这次修正改变。

本机 board 的 all 流水线、服务器 huanggua 的教师等待流水线均已加载这套学生配置，
会在各自教师正常完成后自动启动学生。当前两个教师继续运行，无须重启。

## 核对与验证证据

- 实际参考学生 checkpoint 与历史源码 `8f2eb5f` 的参数名称、形状严格匹配；best/latest 的结构相同。
- 阶段子网络加入前，当前和参考学生主体均有 442 组模型张量；只因六维输入改成七维，两个力觉投影增加合计 512 个参数。
- 将接口统一为六维并加载同一旧学生权重，CPU 样例 RGB / gate 前向最大绝对差均为 0。
- 两个教师冻结 DINO 的 211 组张量逐一相等。
- 新增阶段衔接测试验证：gate 5 个 epoch 不停止、转 RGB 重置、首次 RGB 验证建基准、再连续 5 个差 epoch 才停止；已在 RGB 的重复进入不重置计数。
- 本机和服务器各通过 64 项针对性检查，包含阶段早停、完整融合门控、教师完成检查和流水线接续。

完整张量比较和 CPU 样例说明：
`data/outputs/2026.11.15/remote_huanggua_single_gpu_150k/network_structure_comparison.json`。
两边队列、最终配置与检查记录：
`data/outputs/2026.11.15/remote_huanggua_single_gpu_150k/both_students_rgb_early_stopping_verification.json`。

后续效果比较应包含 gate 打开时的 token 误差和任务表现。
单独看到整体 token MSE 曲线合理，尚不足以确认旧实验效果已在新数据上重现。

## 2026-11-16 新增因果阶段预测

用户确认增加独立 CNN + GRU 阶段子网络：双视角历史 RGB，配合 16 帧关节位置、速度与连续外力矩，预测 precontact/contact 是否应打开。该模块新增 112,737 个参数；原先 442 组张量的严格对齐结论仅适用于加入模块前的学生主体。

教师仍使用真实 contact_phase 打开完整融合。当前学生在训练和推理都使用 `p_open >= 0.5`，离线标签只监督 BCE；不直接控制学生渲染。前 5 个 epoch 同时训练原空间 gate 与阶段子网络；其后冻结空间 gate，训练 RGB 和阶段子网络。BCE 权重 1，token MSE 不向阶段子网络反传。早停继续只依据 RGB 阶段 token MSE，patience 5。

因此当前版本是在旧 gate + FiLM RGB / token MSE 方法上加入阶段预测的扩展，结构已多于单纯的数据迁移。两边教师不重启，已排队学生将加载新模块。具体输入、标准化与日志见[阶段子网络设计](causal_phase_gate.md)。
