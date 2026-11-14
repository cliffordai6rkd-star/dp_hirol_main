# 本机 RGB 蒸馏提速实测（2026-11-14）

本机运行的是无 context adapter 的 response 学生，batch 16，RTX 5090 D v2。
服务器 2/3 卡运行的 adapter 学生未在本次测试中更改。

## 正在运行的训练

运行目录：`data/outputs/2026.11.14/13.25.56_response_weighted/student`。
最近 200 个 RGB 训练步骤：计算中位数 200.39 ms，读取等待中位数
0.108 ms，吞吐中位数 79.78 samples/s。GPU 利用率约 93%–97%。
BF16 AMP、TF32、学生 DINO 编译、教师目标缓存已经开启；训练读取使用
4 个 persistent worker。增加 worker 的直接收益很小。

## 隔离测试

用 epoch 50 附近的独立 checkpoint 副本和已有真实诊断样本进行对照。
保留生产进程的显存，短暂 SIGSTOP 后测速，退出时由独立管道 watcher
保证 SIGCONT。测试更新的是临时模型，不写生产 checkpoint。

完整步骤测试包含条件预测网络、RGB 渲染、冻结 DINO 的图像梯度、全局及
响应加权 token MSE、梯度裁剪、AdamW 和 EMA。读取使用预处理好的 GPU
张量，不包含数据解码、验证和额外诊断指标。以下为 batch 4，热身后的
40 步中位数，不能直接当作当前 batch 16 的提速比例。

| 对照 | 原步骤 | 优化步骤 | 吞吐倍率 | 首步数值对照 |
| --- | ---: | ---: | ---: | --- |
| 编译 RGB 空间调制主干，默认 Inductor | 54.52 ms | 43.52 ms | 1.253× | loss 相对差 0.0417%，全部梯度相对 L2 差 24.81% |
| 编译时保留布局和低精度中间舍入 | 53.88 ms | 46.25 ms | 1.165× | loss 相对差 0.0267%，全部梯度相对 L2 差 28.64% |
| fused AdamW + foreach EMA 原型 | 55.10 ms | 54.17 ms | 1.017× | token 完全相同，梯度相对 L2 差 0.278% |
| cuDNN benchmark | 54.25 ms | 52.90 ms | 1.026× | token 完全相同，梯度相对 L2 差 0.272% |

重新构造相同 eager 模型并重放随机状态时，token 完全一致，梯度相对 L2
差约 0.270%。两层 cuDNN LSTM 的 dropout 需要 `torch.manual_seed` 重置，
仅恢复 CUDA RNG state 不足以控制这个对照。

默认编译路径通过了 FP32 CPU 输出、梯度、空间调制梯度、checkpoint key
及 eager 验证路径的回归。但 BF16 完整蒸馏对照中的梯度差异仍明显高于
eager 重复对照，不能据此认定长期收敛效果不变。

因此，本轮只补充了**可选**的空间调制主干编译支持，默认
`training.compile.render_backbone=false`，没有启用它或重启当前本机任务。
融合更新和 cuDNN 原型收益小，也没有改入生产训练。

## 复现

```bash
/home/lcy/miniconda3/envs/fdp/bin/python test/benchmark_distillation_step.py \
  --checkpoint data/diagnostics/2026.11.14/local_training_acceleration/student_snapshot.ckpt \
  --samples data/diagnostics/2026.11.14/response_distillation/samples.pt \
  --output /tmp/distillation_step.json \
  --batch-size 4 --steps 40 --optimization renderer
```

通过 `--optimization updates` 或 `--optimization cudnn` 测另外两种方案。
布局/舍入对照增加 `--preserve-layout --emulate-precision-casts`。
有其他训练占用 GPU 时可明确指定 `--pause-pid <训练PID>`；它只接受
`train_distill_render_net.py` 进程。省略该选项不会暂停任何进程，应另行
确保测速时 GPU 空闲。

所有原始结果：`data/diagnostics/2026.11.14/local_training_acceleration/`。
使用 `distillation_step_renderer_final.json`、`distillation_step_layout.json`、
`distillation_step_updates.json`、`distillation_step_cudnn.json`；早期失败对照
保留用于追溯，不作为最终测速数据。

本轮针对新编译路径、response loss 和 adapter 的 26 项回归通过。较宽的
历史测试组合另有 15 项失败，已用 HEAD 的原始 RenderNet 重现同名失败；
涉及旧 `phase_prediction` / `condition_backbone` 接口和已 detach 的 mask
仍要求 RGB 梯度的断言，没有修改这些历史测试或当前网络来迁就它们。
