2026-11-05 验证记录，基线提交 `27e50e9`。使用
`/home/lcy/miniconda3/envs/fdp/bin/python`（Python 3.10.22、PyTorch 2.7.1+cu128），
CPU tiny/mock DINO，无预训练模型下载或训练数据依赖。

修改文件按用途列出：

- 逐点接触和共享预处理：`diffusion_policy/model/vision/contact_curriculum.py`、
  `diffusion_policy/model/vision/wrench_preprocessing.py`。
- 教师完整融合门及课程扰动：`diffusion_policy/model/vision/force_aware_obs_encoder.py`、
  `diffusion_policy/policy/force_aware_diffusion_transformer_policy.py`。
- 蒸馏预处理保存、缓存指纹和反事实路径：
  `diffusion_policy/workspace/train_distill_render_net_workspace.py`、
  `diffusion_policy/model/vision/render_net_attention.py`。
- 纯视觉旁路的 attention/残差重放：`diffusion_policy/scripts/` 下的
  `checkpoint_attention_capture.py`、`image_query_attention_capture.py`、
  `fusion_visual_attention_trace.py`、`visualize_wrench_effect_episode.py`、
  `visualize_wrench_action_sensitivity_episode.py`、
  `visualize_cross_attention_episode.py`。
- 配置：`diffusion_policy/config/sim_force_aware_dp_train.yaml` 和新增
  `sim_force_aware_dp_train_20hz_lstm2_gripper_contact_gate.yaml`。
- 测试：`tests/test_contact_fusion_gate.py`、`tests/test_contact_fusion_gate_integration.py`。
- 说明：`README.md` 和本记录。

新未来模式对配置的原始物理相位信号逐点比较进入阈值；最后一个点接触也立即
开启。当前历史检测器仍保留连续 5 点确认。`future_contact_mask.any(-1)` 的同一
结果用于未来 wrench 整段抑制及教师完整融合门；保留原有 prefix filling、归一化和
归一化后严格零处理。教师关闭门时选取联合融合前的原图 tokens，再通过原来的
`context_norm`。缺失新配置字段默认 `legacy_rollout`、`false`。

没有修改学生 RenderNet 架构、教师/学生 loss、网络层数、动作定义或未来采样时间网格。
没有增加可训练参数，旧权重严格加载通过；新 checkpoint 和独立预处理配置的恢复通过。

新增测试命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/lcy/miniconda3/envs/fdp/bin/python -m pytest -q --tb=short \
  tests/test_contact_fusion_gate.py tests/test_contact_fusion_gate_integration.py
```

实际结果：**22 passed，19 warnings，5.66 s**。覆盖用户要求的十类验收，包括末点/末三点
接触与 legacy 对照、当前/未来状态独立、raw_zero 重新生成标签、完整视觉/融合等价、
混合 batch/多观测/多相机/联合 SA/多层 CA、课程 mask/blur、梯度隔离、严格权重加载、
预处理及 checkpoint 恢复、缓存指纹，以及磁盘缓存与在线蒸馏 loss/梯度逐值一致。

相关回归命令（追加 `--junitxml=...` 保存结果）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/lcy/miniconda3/envs/fdp/bin/python -m pytest -q --tb=no \
  tests/test_contact_fusion_gate.py tests/test_contact_fusion_gate_integration.py \
  tests/test_force_aware_diffusion.py tests/test_force_aware_workspace.py \
  tests/test_configurable_force_fusion.py tests/test_render_net*.py \
  tests/test_fusion_visual_attention_trace.py tests/test_wrench_effect_visualization.py \
  tests/test_wrench_action_sensitivity.py tests/test_image_query_attention_capture.py \
  tests/test_checkpoint_attention_capture.py tests/test_dit_action_attention_capture.py \
  tests/test_denoising_attention_visualization.py tests/test_self_attention_visualization.py \
  tests/test_config_cli.py tests/test_train_pipeline.py
```

| 代码 | 通过 | 失败 | 时间 |
| --- | ---: | ---: | ---: |
| 干净 HEAD，运行同一组旧测试 | 502 | 122 | 30.89 s |
| 修改后，包含新增 22 项测试 | 524 | 122 | 30.84 s |

JUnit 按用例名称对照：**没有新增失败，没有消失的失败**。121 项失败消息相同；
DDP 那项仅进程编号、工作树路径和行号不同，根因同为旧 `TinyEncoder` 缺少
`wrench_history_steps`。已有失败包括学生旧 `backbone`/`transformer` 参数不匹配、
`decoder_layers` 与 patch_size 不匹配、旧 workspace mock 缺少字段及未实现的
`local_highres` 参数。未在本次改动中扩展学生架构或修复这些已有问题。

本次会话完整结果保存在 `/tmp/dp_contact_gate_review_g27fqo3n/`：
`baseline.log`、`baseline.xml`、`current.log`、`current.xml`、`comparison.json`。
没有启动正式训练，没有提交或推送。真实数据、真实 DINO、GPU/DDP 新模式训练及
闭环动作性能未验证。

新实验启动命令（仅供后续运行）：

```bash
python sim_force_aware_dp_train.py \
  --config-name=sim_force_aware_dp_train_20hz_lstm2_gripper_contact_gate
```

新预处理和门控改变教师输入及条件表示，需要微调或重训并验证动作性能，再生成
对应蒸馏目标。旧权重可加载不代表动作性能不变；教师门控本身不保证学生 RGB
像素恒等。
