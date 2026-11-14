# 共享区域/alpha 预测与教师响应加权蒸馏

启动本次方案：

```bash
/home/lcy/miniconda3/envs/fdp/bin/python train_distill_render_net.py \
  --config-name=train_distill_render_net_student_response
```

教师固定为 `data/outputs/2026.11.12/20.05.59_876448_fdp_dis_board/teacher/checkpoints/optimizer_step=00180000-action_mse=0.00470914.ckpt` 的 `ema_model`。保留 checkpoint 的 0.5 秒预接触连续力矩预处理、normalizer 和全部网络权重。特征目标始终是完整教师融合后、进入 DiT 前的最终 `context_norm`；学生图像路径不运行教师融合 FFN。

## 网络与阶段

条件网络与 RGB RenderNet 各有自己的可训练参数，共两个图像编码网络。条件网络的两个相机共享 CNN/attention 权重，通过相机 embedding 区分。CNN 使用 base channels 16、patch 16、embed 128，保留 14×14 网格；随后一层图像 self-attention 和一层图像查询的力矩 cross-attention。力矩路径使用教师初始化的两层 256 维 LSTM，保留全部 8 个历史输出。

区域头从融合网格预测单通道 sigmoid mask，再双线性上采样。Alpha 头对两相机融合特征池化并拼接，经过单向 64 维 GRU 和 MLP，输出每个观测时刻的 sigmoid alpha。当前输出不读取后面的观测图像。两个头共享条件编码器，条件编码器与 RGB 分支不共享参数。

第一阶段最多 20 个 epoch，联合训练区域和 alpha；验证连续 5 个 epoch 无显著改善时提前进入第二阶段。损失为区域 patch SmoothL1 + alpha MSE + 0.02×空间平滑项。区域目标沿用逐图 min–max 空间形状，仅在教师最大响应超过 1e-6 的相机/时刻监督，避免区域头重复学习 alpha 的开关。Alpha 使用离线 0.5 秒预接触标签，所有时刻都参与。此阶段不执行 RGB 分支或学生 DINO。

第二阶段恢复 `best_condition.ckpt` 中验证选用的 EMA 快照，冻结并设为 eval，重置选模指标、优化器状态、学习率调度和 RGB EMA 计数。RGB 分支维持历史结构：patch 8、embed 128、base channels 32、6 层 self-attention、2 层 cross-attention、3 层 decoder 和两处 decoder FiLM。

训练与推理都采用预测条件：`I' = clip(I + stopgrad(alpha_hat) * stopgrad(mask_hat) * delta_I, 0, 1)`。离线 alpha 不驱动学生渲染，学生 loss 不更新条件网络。

总预算沿用 1000 个 epoch，包含条件阶段；RGB 阶段保留验证早停 patience 30。训练集 17740 个窗口，验证集 1893 个窗口，batch size 16。

## 响应加权

教师响应为 `r_p = ||Z_T(I, tau)_p - Z_T(I, raw_zero)_p||_2`。零力矩分支删除原样本的离线 alpha，并重算门控。两次取最终 context_norm，使用冻结 eval 教师在 FP32 下计算。

每个 token 的误差 `e_p = mean_d((Z_S - Z_T)^2)`，停止梯度的权重 `a_p = max(r_p - 1e-6, 0)`。学生目标：

```text
global_mse   = mean(e_p)
response_mse = sum(a_p * e_p) / max(sum(a_p), float32_epsilon)
student_loss = global_mse + 1.0 * response_mse
```

权重直接使用教师原始响应，不使用预测 mask/alpha，不做逐图 min–max 或 top-k。一个 batch 内所有样本、观测时刻、相机和 patch 一起累加；全零响应时 response_mse=0，全局 MSE 仍参与。验证先汇总全部 batch 的加权分子和权重总和，再计算 response_mse，最佳学生按完整 student_loss 选择。

原始响应和教师 token 可复用 `context_effect_v2` 缓存，不缓存学生预测或归一化区域标签。配置保留已有缓存，退出时不删除。

## 验证记录

新增测试覆盖有效权重归一化、零/微弱响应、教师停止梯度、条件阶段跳过 RGB/DINO、两个头共享编码、冻结后的梯度隔离、最佳条件快照、验证跨 batch 聚合、因果 alpha、模型重载和推理不使用离线 alpha。

相关测试集 110 项通过（排除下述 6 项旧绘图断言）。真实 batch 的 eager 与编译后 DINO 两条训练路径均完成前向、反向和参数更新。

2026-11-14 用真实数据和指定教师完成 batch size 16 的条件训练与 RGB 训练试跑：学生 token `[16, 784, 256]`，峰值显存 14907 MiB。RGB 输出权重上的 global/response 梯度范数分别约 0.0692/0.1386；第一版维持两项系数均为 1。这仅验证计算和训练链路，不表示效果已优于原方案。

仓库中后续实验的 `shared_condition_backbone`/`phase_prediction` 编译测试不适用于恢复后的历史接口。当前旧训练循环测试中“永远不生成曲线 PNG”的断言也与保留的绘图行为不一致；未据此恢复后续实验网络。
