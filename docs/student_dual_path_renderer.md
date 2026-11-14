学生双路径 RenderNet 为独立可选结构，教师、蒸馏目标和训练 Loss 保持原设置。旧配置不启用 `dual_path`，其参数名称与默认计算路径保持兼容。

| 实验 | RGB token 融合 | Decoder FiLM | Context Adapter |
|---|---|---|---|
| A | SA×6 → CA×2 | 无 | Linear Residual |
| B | SA×2 → CA → SA×2 → CA | 无 | Linear Residual |
| C Linear | SA×2 → CA → SA×2 → CA | /4、/2 | Linear Residual |
| C MLP | SA×2 → CA → SA×2 → CA | /4、/2 | 256→128→256，GELU Residual |

A/B 同时减少两层 SA 并改变交错顺序，不能将差异全部归因于交错位置。B/C Linear 才单独隔离 Decoder FiLM；两个 C 隔离 Adapter 结构。

CNN Pyramid、Skip Connections、RGB Decoder、原 Spatial Modulation 与预测区域/alpha 的 Condition Network 保留。Image tokens 是 CA 的 Q，现有两层 LSTM 的完整 8 步 torque tokens 是 K/V。FiLM 用可学习 Attention Pooling 得到全局条件，每个 Decoder 尺度有独立的 gamma/beta MLP，最后一层权重和偏置为零。更新为 `F + predicted_alpha * resized_spatial_gate * (gamma * F + beta)`，随后仍经过原 RGB 输出门控。/1 不增加 FiLM。

Condition Network 的预测 alpha 驱动渲染，离线标签仅监督或统计；直接传入全零标签不能替代 alpha=0 检查，因为预测值会覆盖标签。检查会暂时将 Condition Network 输出设为零，并验证非零 RGB/FiLM 权重下输出仍逐元素等于输入。

新结构从保存的 `student_rgb_initial.ckpt` 迁移共同权重，包括相同的随机 RGB 输出卷积、已训练 Condition Network 和 LSTM。迁移只允许移除 SA 第 5、6 层，以及替换 Linear Adapter 所需的键；只允许新建指定 FiLM/MLP 键，其余缺失或多余键均报错。新 checkpoint 按其独立配置严格加载，不将旧 checkpoint 当作完整新结构静默恢复。

复用现有 `StudentContextAdapter` 的 Linear 和 MLP 两种实现。两者均从零残差输出开始，使用相同 2,000 步、batch=64 的零力矩校准预算；A/B/C Linear 共享同一个校准快照。渲染训练时冻结 Adapter 和 Condition Network，四组各 5,000 步、batch=16、seed=42，使用相同全量学生划分与采样顺序。teacher checkpoint 仍固定为 2026.11.12 的 180,000 步 EMA，当前本机新教师训练不自动替换此对照实验的目标。

Loss 继续调用原 `response_weighted_token_mse`：全局 Token MSE 加权重为 1 的响应加权 MSE，噪声底为 1e-6。新增的 Torque-response Difference MSE 仅作验证诊断：`MSE((S(normal)-S(physical_zero)), (T(normal)-T(physical_zero)))`。置零使用物理单位，学生重新预测区域与 alpha，教师按原成对目标逻辑重新计算零力矩 gate；该口径与上一教师融合消融中“固定原 alpha”的干预不同。

验证覆盖原学生的全部 1,893 个窗口，分别汇总 Free、Precontact、Contact，不以几个诊断样本代替验证集。响应加权项按有效权重之和归一化。记录 RGB 残差 MAE/RMS、裁剪前梯度 L2、初始化与后续验证曲线，以及同一样本、统一色标的三阶段渲染图。

图像预处理使用与历史学生一致的 TorchCodec CUDA 解码。服务器启动进程显式补齐已有 NPP 动态库目录，避免静默回退到 PyAV。独立缓存须核对历史诊断样本像素、采样索引哈希和教师成对目标；此前用于排查的 PyAV 缓存与校准结果不进入正式消融。

独立配置为 `student_dual_path_{a_baseline,b_interleaved,c_film_linear,c_film_mlp}.yaml`。结构测试在 `test/test_student_dual_path.py`；准备、校准、训练、报告脚本均放在 `test/`，服务器训练使用独立源码快照。
