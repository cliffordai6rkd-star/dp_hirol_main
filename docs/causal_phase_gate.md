# 因果阶段子网络

本实现按用户确认的结构，将一个独立的小型 CNN + GRU 与原空间 gate / RGB 学生一同训练。教师不变，学生对“何时打开”的判断由历史观测预测。

## 输入与结构

对每个待预测观测时刻 t 独立构造输入，学生的后一个观测不会成为前一个观测的输入。

| 输入 | 时间范围 | 张量形状 |
| --- | --- | --- |
| front / right_wrist RGB，模型键 side / wrist | t-12、t-8、t-4、t | 每个相机 `[B,To,4,3,H,W]` |
| 七轴关节位置 | t-15 到 t | 状态前 7 维 |
| 七轴关节速度 | t-15 到 t | 状态中 7 维 |
| 连续七轴关节外力矩 | t-15 到 t | 状态后 7 维 |
| 合并状态 | 当前及历史共 16 帧 | `[B,To,16,21]` |

源列为 `observation.joint`、`observation.velocity`、`observation.tau_ext_unthresholded`。episode 开头不足的历史重复首帧。RGB 缓存可以是 uint8，前向时转换为 `[0,1]` 并缩放到 112×112；状态均值/标准差只使用训练 episode，作为 checkpoint buffer 保存。

两个相机共享三层 CNN；保留相机次序，拼接每个图像时刻的 64 维特征，经过 GRU 得到 64 维图像摘要。21 维状态经投影与另一个 GRU 得到 64 维状态摘要。两者拼接后由 MLP 输出一个 logit。默认总参数量为 112,737。

```text
两相机历史 RGB → 共享 CNN → 特征拼接 → 图像 GRU ─┐
16 帧 joint / velocity / tau → 归一化 → 状态 GRU ─┤→ MLP → p_open
                                                └→ p_open >= 0.5 → 渲染开关
```

## 训练与梯度

标签由现有教师的逐帧 `contact_phase` 规则产生：free=0，precontact/contact=1。标签只用于教师目标与 BCE 监督；学生渲染和推理开关始终使用预测结果。

| 阶段 | 训练模块 | 损失 |
| --- | --- | --- |
| 前 5 个 epoch | 独立空间 gate + 阶段子网络 | 原 gate patch/平滑损失 + BCE |
| 其后 RGB 蒸馏 | RGB 主干 + 阶段子网络；空间 gate 冻结 | 普通 context token MSE + BCE |

BCE 权重为 1。二值开关对概率执行 detach，token MSE 不向阶段子网络传梯度。RGB 开关关闭时保持输入图像；打开时应用原空间 gate 和 RGB 残差。阶段模块作为 RenderNet 的子模块，共享优化器、EMA 和 checkpoint 生命周期。

验证选模和早停只按原 token MSE；BCE 不计入选模指标。gate 阶段不累计早停，切换 RGB 时重置计数，patience 5、min_delta 0.0001。

## 推理接口

`obs['phase_inputs']` 包含：

```python
{
    'state': tensor,  # [B,To,16,21]，原始物理量
    'images': {'side': tensor, 'wrist': tensor},  # 各 [B,To,4,3,H,W]
}
```

`RenderNet.forward(..., phase_inputs=...)` 内部预测开关，返回 condition 中的 `phase_logits`、`p_open`、`contact_alpha`。checkpoint 恢复状态标准化；部署端须按相同次序和偏移维护历史，不传未来帧。渲染入口已递归处理此输入，不依赖推理时的 `contact_phase`。

日志新增阶段 BCE、准确率、预测打开比例，以及 free 误开率、precontact 召回率、contact 召回率。分阶段指标累积真实样本数，并记录各自分母，避免缺失类别被误读为零召回。

与旧实验相比，gate / FiLM RGB / token MSE 主链保留，但额外加入了阶段预测能力；本次学生不能再称为与 11 月 7 日结构完全相同。预测时机是否足够准确，需要训练后的分阶段指标与任务评估验证。
