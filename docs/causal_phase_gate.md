# CIVI 阶段预测与三阶段训练

当前 CIVI 使用同一条因果输入链：教师和学生读取阈值后 `observation.tau_ext`，顶层 `tau_ext_history_steps=25` 默认使每观测以 `t-24…t` 的 25 个七轴外力矩向量为输入，原始 25 Hz 下约 1 秒。修改这一项会同步改变教师输入形状、LSTM 历史和数据窗口，学生自动继承教师 checkpoint 的设置。episode 开始前缺失的力矩补零。L1 小于 12.5 N·m 时整向量置零，等于阈值保留。物理零在归一化后仍为零。

## phase 输入与输出

对每个待预测观测时刻独立构造历史；后一个观测不会泄露给前一个观测。

| 输入 | 时间范围 | 张量形状 |
| --- | --- | --- |
| 双相机 RGB：side / wrist | `t-12,t-8,t-4,t` | 每相机 `[B,To,4,3,H,W]` |
| 七轴 joint、velocity、阈值后 tau_ext | 默认 `t-24…t` | 合并为 `[B,To,tau_ext_history_steps,21]` |

状态源为 `observation.joint`、`observation.velocity`、`observation.tau_ext`。图像在 episode 开头延用首帧，缺失的完整 joint/velocity/tau_ext 状态历史填零。状态标准化仅用训练 episode 拟合，并存入 checkpoint。

两个相机共享 CNN，图像特征经 GRU 汇总；21 维历史状态经过投影与另一个 GRU，两者拼接后预测三个 logits。类别为 free=0、precontact=1、contact=2。`argmax != 0` 得到渲染开关；`p_open=1-p_free` 供日志使用。开关 detach，RGB/token 损失不更新 phase 头。precontact 的力矩即使经阈值后为零，仍按预测 phase 开启渲染；precontact→contact 按三分类 argmax 直接切换，不增加渐变或防抖。离线接触标签和默认 0.5 秒 precontact 标签仅用于监督，推理不需要标签或未来观测。

## 顺序训练与冻结

| 阶段 | 可训练模块 | 验证选模和收敛指标 |
| --- | --- | --- |
| phase | 仅 phase 头 | 三分类交叉熵 |
| intensity | 独立空间 gate 与其编码路径 | 教师 context 响应强度的 patch 损失与平滑损失 |
| rgb | RGB RenderNet；两个预测头保持冻结 | 最终 context token MSE |

每个阶段使用验证损失监测收敛，连续 5 次验证无至少 0.0001 改善即恢复该阶段最优权重并冻结。phase 和空间强度阶段不执行 RGB/DINO 蒸馏前向。阶段切换重置优化器状态、学习率和早停计数，EMA 与冻结权重同步，恢复训练时保留当前阶段与计数。RGB 阶段结束时早停，不重新解冻预测头。`training.num_epochs=3000` 是总预算上限，不替代收敛条件。

context 空间强度监督来自冻结教师在真实力矩与零力矩下 context 的差异，默认每图归一化为强度场。学生在推理时仅执行自身预测头，空间 gate 与 FiLM 调制 RGB 残差；free 保留输入图像，precontact/contact 应用渲染。学生图像路径经过冻结教师视觉编码器，不重放教师的力觉融合 FFN。

## 教师课程

冻结 DINOv3 后使用单层 image-query cross-attention/FFN 和 DiT。contact/precontact 开启完整融合，free 整体旁路；训练视觉 token 的 mask 概率按 0%–25%=1、25%–75% 从 1 线性降到 0、75%–100%=0。mask 仅作用于 contact/precontact，验证、推理与蒸馏目标不加 mask。当前统一配置无 blur、未来力矩输入或固定 epoch 的强度预热。

## 推理接口

`obs['phase_inputs']` 包含原始物理状态和历史图像：

```python
{
    'state': tensor,  # [B,To,tau_ext_history_steps,21]，joint、velocity、阈值后tau_ext
    'images': {'side': tensor, 'wrist': tensor},  # 各[B,To,4,3,H,W]
}
```

RenderNet 内部返回 `phase_logits`、`phase_prediction`、`p_open` 与 `contact_alpha`。部署端使用与训练相同的时间偏移、阈值和补零规则，不传离线 `contact_phase` 作为推理开关。旧二分类 checkpoint 仍按保存的配置和输出维数加载，与新的三分类模型权重不互换。
