# 学生视觉基线适配与响应蒸馏

教师固定为 `2026.11.12/20.05.59_876448_fdp_dis_board` 的 step 180000 checkpoint。教师权重、正常前向、最终 `context_norm` 目标和 DiT 均保持不变；学生不复用教师的 cross-attention 或融合 FFN。

学生路径：

```text
RGB/力矩 -> 原 RGB 渲染网络 -> 图像 I'
         -> 冻结 DINO -> 原视觉投影与时间/相机编码
         -> 学生独立适配器 A -> 冻结 context_norm -> 冻结 DiT
```

适配器只读取视觉 token，不读取力矩、区域 mask 或 alpha，始终生效。即使 alpha=0、图像完全不变，适配器也能学习教师零力矩分支的视觉变换。它属于学生 `model.context_adapter`，不会注册到教师上。

## 配置与训练

使用新配置启动新实验：

```bash
/home/lcy/miniconda3/envs/fdp/bin/python train_distill_render_net.py \
  --config-name=train_distill_render_net_student_response_adapter \
  dataloader.batch_size=16 val_dataloader.batch_size=16
```

输出写入独立的 `*_response_adapter/student` 目录。旧的 response 配置与无适配器 checkpoint 仍按原路径加载；旧 checkpoint 缺少适配器参数，不能直接作为新结构的 `training.resume_from`。

默认适配器是 `A(H)=H+Linear(H)`，256 维、65,792 个参数；线性增量零初始化，因此初始化时与原图像编码路径一致。可选残差小 MLP：

```bash
/home/lcy/miniconda3/envs/fdp/bin/python train_distill_render_net.py \
  --config-name=train_distill_render_net_student_response_adapter \
  render_net.context_adapter.kind=mlp \
  render_net.context_adapter.hidden_dim=128
```

该 MLP 为 `256 -> 128 -> GELU -> 256`，准确执行顺序为 `Linear(256,128) -> GELU -> Linear(128,256)`，再加输入残差；共 65,920 个参数，最后一层零初始化。这是学生新参数，不是教师 FFN 的复制。

训练阶段如下，各阶段的最佳模型和早停指标独立：

| 阶段 | 更新参数 | 目标 | 默认上限 / patience |
| --- | --- | --- | --- |
| adapter | 仅 A | 原图经 A 后的 context，拟合教师同图零力矩最终 context | 20 epoch / 5 |
| intensity | 区域与 alpha 的共享编码器、两个头 | 保留原区域/alpha 联合预训练 | 20 epoch / 5 |
| rgb | 原 RGB 网络 | 全局最终 token MSE + 教师原始响应加权 MSE | 保留总训练预算及 RGB patience=30 |

第一阶段覆盖原有各阶段图像，对每张图像都构造物理零力矩参考，移除原样本的 `contact_alpha` 标签后重新计算教师门控。此阶段不运行学生 RGB/条件预测网络。第二、第三阶段都冻结 A，但 RGB 阶段仍通过 A 反传图像梯度。

阶段结束时恢复该阶段验证最优的 EMA（未启用 EMA 则为在线权重），分别保存 `best_adapter.ckpt` 和 `best_condition.ckpt`。随后重置选模指标、优化器状态、学习率调度及 EMA 计数。阶段年龄单独计数；适配器阶段不会占掉区域预训练的 epoch 上限。DDP 在训练参数归属改变时重新构建包装器。

适配器阶段的曲线与日志使用 `adapter_mse`，不与后续正常力矩 token MSE 连成同一项。达到 epoch 上限或验证早停都会进入下一阶段，不会因校准阶段的正常收敛终止整个实验。

## 教师缓存

现有缓存类已支持成对 `tokens` / `zero_tokens`。新配置为缓存 identity 加入 `paired_zero_context=raw_zero_v1`，创建独立缓存，保留正常教师 context、零力矩 context 及原响应图。已有旧缓存及正在使用它的进程不受影响。

context 响应与零力矩 context 从同一对 FP32 教师前向中获得，不额外重复计算第三次教师前向。正常蒸馏目标与响应权重公式保持不变。只改变适配器结构不改变教师标签，可复用这份成对缓存。

## 推理必须保留 A

`load_render_net` 会根据保存的配置构造并严格恢复 `context_adapter`。仅调用渲染函数得到 RGB，尚未完成学生的 token 适配。使用统一的推理接口：

```python
from diffusion_policy.scripts.visualize_render_net_episode import (
    load_render_net, encode_student_context,
)

student, preprocessor, cfg, teacher_cfg, device, info = load_render_net(checkpoint)
context = encode_student_context(
    student, preprocessor, observations,
    frozen_policy.obs_encoder, frozen_policy.n_obs_steps,
)
# context 输入原冻结 DiT；保留原策略的扩散采样、反归一化和动作切片。
```

此接口不依赖离线 alpha 标签，并对旧 checkpoint 自动保留无适配器路径。适配器在最终 `context_norm` 之前执行，训练与推理位置完全相同。

## 验证

独立测试代码放在 `test/`：

```bash
/home/lcy/miniconda3/envs/fdp/bin/python -m pytest -q test/test_student_context_adapter.py

/home/lcy/miniconda3/envs/fdp/bin/python test/smoke_student_context_adapter.py

/home/lcy/miniconda3/envs/fdp/bin/python test/smoke_student_context_adapter.py \
  --compile-student-dino \
  --output-dir data/diagnostics/2026.11.14/student_adapter_compiled_smoke
```

真实教师 GPU 试跑使用已有诊断中的真实图像、小批量力矩输入，并在独立临时目录完成三个 epoch（每阶段一个、每阶段 3 次更新）。已检查三种 checkpoint 阶段及参数、教师冻结和训练/推理 context 一致性。试跑不是完整蒸馏实验，不能说明线性适配器已达到需要的校准精度。

2026-11-14 实测：新增 14 项测试通过（含真实两进程 DDP 阶段切换）；另有 147 项相关回归通过。历史训练循环测试中有 6 个已有断言要求 `metrics_png is None`，仍与现有生成曲线 PNG 的行为不一致；没有为了通过这些断言关闭绘图。真实教师 eager 与编译试跑各完成 9 次更新，重新加载 checkpoint 后训练/推理 context 最大差均为 0，教师可训练参数为 0。试跑摘要分别见 `data/diagnostics/2026.11.14/student_adapter_smoke/summary.json` 与 `student_adapter_compiled_smoke/summary.json`。

完整数据训练后，应把零力矩校准残差与教师力矩信号量级一起评估，再决定是否使用 MLP；不能把适配阶段的 loss 降低直接等同于动作效果改善。
