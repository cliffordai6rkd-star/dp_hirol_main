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

## 双卡训练的数据预取

BF16 AMP、TF32 和 RGB 阶段的学生 DINO 编译已在默认配置中开启。教师目标缓存可复用；渲染图像随参数更新而变化，因此不跨更新复用学生 DINO 的 K/V。

2026-11-14 在双 RTX 4090、约 128 GB 内存服务器上，原先每步约 55 ms 等待数据、32 ms 计算。已为本地 LeRobot Parquet 读取器补充 pickle 恢复方法，允许 DDP 使用 `spawn` worker。样本和教师缓存仍按原顺序对齐。

本次运行采用每卡 8 个训练 worker、4 个验证 worker，均设置 `persistent_workers=true`，总 batch 16、每卡 8。独立读取测试的 0/2/4/8 worker 吞吐分别约 132/180/291/556 samples/s。真实 adapter 训练各取 200 个稳定步骤比较，读取等待中位数从 55.3 ms 降到 0.33 ms，日志中的步吞吐中位数从 183 提升到 473 samples/s，约 2.58 倍。这不包含首次启动 worker 的耗时，也不代表 RGB 阶段已获得相同比例的提速。

独立测速代码位于 `test/benchmark_distill_input_pipeline.py`。在训练原工作目录运行，复用已有图像与教师缓存：

```bash
python test/benchmark_distill_input_pipeline.py \
  --run-dir /path/to/student \
  --output /path/to/input_pipeline.json \
  --workers 0 2 4 8
```

`test/test_distill_training_acceleration.py` 验证真实 LeRobot 读取器及成对教师缓存通过 pickle 和 spawn 后的数值与洗牌顺序一致。本次新增验证及相关读取器/数据集/教师缓存回归共 103 项通过。

## 冻结 DiT 的行为蒸馏

`train_distill_render_net_student_frozen_dit` 使用相同的渲染器、冻结视觉适配器和区域/alpha 网络，
将 RGB 阶段的主要目标改为冻结 DiT 的输出 MSE：教师读取正常融合 context，学生读取渲染图像的 context，
两路使用同一条加噪动作和同一个扩散时间步。教师目标停止梯度；学生分支允许梯度穿过冻结 DiT、
context_norm、adapter、DINO，更新 RGB 分支及其 wrench 编码器。验证固定噪声/时间步，选模、降学习率和
早停均监控 DiT 输出 MSE。这里每次更新使用单步去噪输出，不展开完整扩散采样。

```bash
python train_distill_render_net.py \
  --config-name=train_distill_render_net_student_frozen_dit \
  training.warm_start_from=/absolute/path/to/student/checkpoints/best.ckpt
```

`warm_start_from` 只严格加载学生 EMA 权重（没有 EMA 时加载在线权重），直接进入 RGB 阶段；
优化器、EMA 计数和验证指标从头开始，允许从原 token 蒸馏切换目标。继续同一个 DiT 实验则使用
`training.resume_from`，保留优化器和轮数，且将 `training.warm_start_from=null`；两个参数不能同时指定。
不指定 warm start 时，仍依次训练 adapter、条件网络和 RGB。

此配置保留区域阈值 0.2、预测 alpha、batch 16 和 RGB patience 10。token MSE 保留为诊断，
token 辅助项、响应加权项和演示动作损失的训练权重均为 0；它们不会覆盖 DiT 目标。
已有教师 context 缓存可以复用，DiT 输出随新采样的噪声和时间步即时计算。

新增分阶段回归位于 `test/test_staged_frozen_dit.py`，检查实际训练目标、固定验证探针、
RGB/wrench 梯度、冻结分支、跨目标权重初始化及优化状态重置。

## 只用接触阶段继续蒸馏

`train_distill_render_net_student_frozen_dit_contact` 设置 `training.sample_phases=[2]`，
按数据集的当前观测锚点标签筛选接触样本，排除自由空间（0）和预接触（1）。每个入选样本仍保留
完整历史观测和动作序列，历史中可以包含非接触帧；筛选不依赖学生预测的 alpha 或区域 mask。
筛选位于教师缓存加载之后，使用原始样本索引，现有全量缓存可以直接复用。

```bash
python train_distill_render_net.py \
  --config-name=train_distill_render_net_student_frozen_dit_contact \
  training.warm_start_from=/absolute/path/to/all_phase_student/checkpoints/best.ckpt \
  teacher_cache.directory=/absolute/path/to/data/cache_response_adapter_teacher
```

从已有学生最佳 EMA 权重启动独立 RGB 实验，adapter、区域和 alpha 网络继续冻结。
训练、验证、选模和早停都只使用接触锚点样本，验证基线重新计算，不能直接与旧实验的全阶段均值比较。
`training_summary.json` 记录两个划分的入选数量和原始阶段分布。该筛选不会自动改变推理时的渲染门控。
恢复同一实验时不能修改筛选阶段；改阶段应使用 `warm_start_from` 新建实验。
接触筛选及缓存对齐回归位于 `test/test_distill_contact_samples.py`。

## 接触 loss 统计和绘图

RGB 蒸馏现在在每个已有 batch 内额外统计真实 `contact_phase[:, n_obs_steps-1] == 2`
的样本。统计包含这些样本的完整历史 context 或 DiT 动作输出，不用预测 alpha、区域 mask、
未来接触门控来筛选。它是诊断统计，不修改训练样本、优化目标、选模或早停规则。
DiT 配置报告去噪输出 MSE；token 配置报告全局 token MSE 加原配置权重的响应加权项。
普通项按接触样本数汇总，响应项先跨 batch/卡累加分子和有效权重，再归一化。
没有接触样本或尚在 adapter/条件预训练阶段时，不把缺失的接触 loss 画成零。

训练输出目录新增 `contact_loss.png`，只显示接触样本的 Train loss 和 Validation loss。
主图 `training_metrics.png` 的 loss 面板也只保留这两条线：训练为每轮平均值，验证为每轮评估值，
不再显示逐 batch 线或滑动平均线。初始 loss 会直接连接到曲线起点。
新实验及阶段切换时，初始训练 loss 会遍历完整入选训练集（使用评估模式和 EMA，如果启用），
后续训练点为在线模型更新过程中的平均 loss；验证按原固定噪声/时间步和验证配置计算。
完整初始评估会增加启动耗时，不产生优化更新。

`log.json` 是标准 JSON 文档，`records` 保存初始值、每轮均值、验证值及接触样本计数。
主要字段为 `initial_train_contact_mse`、`initial_val_contact_mse`、
`epoch_train_contact_mse`、`val_contact_mse`，各自对应同前缀的 `contact_samples`。
原 `logs.json.txt` 继续保留逐 batch 的详细数据。恢复训练会保留已有记录。

```bash
python -m diffusion_policy.scripts.plot_render_net_training /absolute/path/to/student
```

上述命令重新生成主图、接触图和 `log.json`；若只有 `log.json`，也支持用其重建曲线。
旧日志可以去掉逐 batch 线并接入已有初始值，但缺失的历史接触指标不会被补成零或推算出来。
新统计回归位于 `test/test_contact_loss_logging.py`，真实三阶段短测试使用
`test/smoke_student_context_adapter.py`。
