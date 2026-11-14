# 2026-11-12 历史网络恢复

恢复提交：`ada3341`，2026-11-12 13:51:49（UTC+8）。用户指定的行为是“教师 context_norm token 作为学生 DINO token 的拟合目标”。

## 实际监督路径

- 教师：原图 DINO patch → 投影/嵌入 → 力矩 cross-attention 与 FFN → 最终 `context_norm` → 监督目标。
- 学生：RenderNet 渲染图 → 冻结 DINO patch → 同一投影/嵌入 → `context_norm` → 与教师目标计算 MSE。

`use_teacher_ffn: false` 仅表示学生图像路径不重放教师融合 FFN。教师目标仍然经过完整融合和最终 `context_norm`。历史 YAML 的“教师和学生都使用 FFN 前 token”注释有误，代码没有在教师 FFN 前截取目标。

`ada3341` 的编码器、RenderNet、teacher policy、学生 workspace 和 contact curriculum 按 Git 原文件恢复。默认学生 YAML 也按该提交恢复，仅更正监督路径注释。当前数据路径、字段映射、七轴输入与关节动作接口保留在教师数据配置中；教师网络与历史 xArm 配置使用相同维度和层数。教师模糊课程同步恢复为历史的 25% / 50% / 25% 分段。

## 11 月 7 日与这个 11 月 12 日版本

比较 11 月 7 日最后提交 `8f2eb5f` 与 `ada3341`。两者都存在“教师最终 context_norm → 学生直接 DINO 图像 token”的接口，因此两者并非一个在 FFN 前、一个在 FFN 后这么简单。

| 项目 | 11 月 7 日 `8f2eb5f` | 11 月 12 日 `ada3341` |
| --- | --- | --- |
| 学生 token 路径 | 图像编码、投影/嵌入、context_norm | 相同，不重放融合 FFN |
| 教师目标 | 最终 context_norm 输出 | 最终 context_norm 输出 |
| 教师融合门控 | 可按接触布尔值选择完整融合结果或原图 token | alpha 作用于 attention 更新，残差及 FFN 继续执行 |
| 图像课程 | 保留 learned image-token mask | 移除 mask token，采用 FACTR 像素模糊 |
| 默认学生 self-attention 层数 | 4 | 6 |
| 默认学生 patch size / decoder 层数 | 16 / 4 | 8 / 3 |
| 默认学生 gate 阈值 | 0.5 | 0.1 |
| 默认 gate 预热轮数 | 2 | 5 |

表中学生超参数分别来自各提交的 `train_distill_render_net.yaml` 和 `train_distill_render_net_student.yaml`；11 月 7 日尚无后者。两版都有 FiLM，均无 phase/alpha 预测头。

11 月 12 日 15:53 的 `dd2afb4` 又改为学生重放教师 FFN，并引入 phase/alpha 预测、关闭默认 FiLM、采用条件预热后联合训练。这次恢复对应更早的 `ada3341`。

## 验证

- 五个核心 Python 文件与 `ada3341` 逐字一致；学生 YAML 的配置值与历史文件一致。
- 使用本地 DINO 权重和当前双相机输入，教师最终 `context_norm` 与学生图像 token 均为 `[1, 784, 256]`。实际学生含 6/2/3 层 self-attention / cross-attention / decoder、单通道 gate 和两层 FiLM。
- 134 项历史教师、蒸馏、EMA、contact-alpha、学习率和 DDP 测试通过。
- 43 项当前监督接口与训练入口测试通过，其中新增测试检查缓存/非缓存目标、学生不重放融合 FFN、图像梯度回传及教师冻结。

扩大检查历史 `test_configurable_force_fusion.py` 时，另发现三项与该提交源码自身不一致的旧测试：两项将仅末帧模糊的比例期待为 0.5（实际为 0.25），一项调用该提交编码器未提供的 `local_highres` 参数。它们不在上述通过项中，也没有为适应这些旧断言修改历史网络。后续 self-attention/ACT/phase-head/无 FiLM 配置与权重不适用于本次恢复的实现。
