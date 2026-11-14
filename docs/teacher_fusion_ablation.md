# Teacher Fusion Ablation on V

用户解除十分钟限制并要求优先完成教师实验后，三组已在独立目录 `data/diagnostics/2026.11.15/teacher_fusion_ablation_5k_1829` 从相同初始化完成各 5,000 步，使用服务器 GPU 2、3，验证仍仅取固定 5 个 episode。三组统一比较第 5,000 步 EMA，训练已结束，结果和 checkpoint 已同步本机。学生实验暂停，教师训练使用之前已检查的独立源码快照。

完整结果见该目录的 `analysis.md`：FiLM-only 整体采样动作 MSE 改善 2.49%、contact 改善 4.40%；Hybrid 整体动作 MSE 仅改善 0.05%。目前不支持在原 CA 后追加 FiLM；FiLM-only 保留为候选，尚未证明显著增强有效力矩条件化。

此前的十分钟快速检查结果保留在 `data/diagnostics/2026.11.15/teacher_fusion_ablation_020219/validation_only/analysis.md`：Baseline 与 FiLM-only 仅训练到共同的 1,000 步，Hybrid 仅为未训练的零初始化插入。该快速检查不能代替本次三组训练消融。

本实验只新增教师消融模块、配置、测试与独立实验入口。既有教师配置、checkpoint、学生与蒸馏代码不因本实验修改。训练从相同随机初始化开始，DINOv3 使用相同本地预训练权重；不从已训练教师热启动。

三组均使用 2-layer LSTM、256 维 token、两个融合块及原有 8-layer DiT。Baseline 逐运算保留当前 Image-query Residual Cross-Attention；FiLM-only 在每个块中用 FiLM 替代 CA 更新，随后保留 FFN；Hybrid 在每个 CA+FFN 块后追加 residual FiLM。Attention Pooling 从当前观测的 8 个 LSTM torque tokens 得到一个条件向量，各块的 MLP 输出 residual scale/shift，最后线性层的权重、偏置均为零初始化。

所有力矩更新仅乘一次相同 contact alpha。按用户确认，alpha=0 保留原教师 FFN 的纯视觉通路，严格不依赖力矩；不要求跳过视觉 FFN 或 Context Norm。FiLM 不绕过 RGB/DINO 或增添动作输入。三组共有参数的初始化哈希一致。

数据为 `board_25hz_3cam_lerobotv3_clean_ma5_l1_12p5`。固定 seed=42，从 62 个完整 episode 中留出 5 个作为验证，再从余下 episode 固定抽取 20 个训练。保存 episode IDs、有效窗口数量、阶段统计、split/normalizer/shuffle 哈希，三组必须一致。归一化仅拟合训练窗口。训练 batch=16，均训练 5,000 optimizer steps，不早停；AdamW、warmup=250、cosine LR、BF16/TF32、EMA 与所有增强预处理相同。

按最新课程逐观测帧选择：free 图像不加模糊，precontact（确认接触前 0.5 秒，包含 alpha=0 的起点）与 contact 图像在前 1/3 更新保持 Gaussian sigma=5，后 2/3 线性降至 0。两个相机一致处理。进入 DINO 前模糊，验证与干预诊断全部使用清晰图像。启动前进行真实数据阶段选择、alpha=0 等价与反向传播检查。

每 500 步完整遍历 5 个验证 episode 的有效窗口，记录 EMA 的 Validation Action Denoising MSE；使用各组相同的噪声和 timestep。并记录 Normal/Zero Torque 的最终 Context Token Response RMS，以及置零、同阶段打乱力矩后的去噪误差。干预时固定图像、原始 alpha 和阶段标签，以隔离力矩内容而非 gate 开闭的影响；力矩置零在物理单位下执行，再进入同一归一化。打乱以完整历史窗口为单位，固定同阶段 derangement。

最终在相同步数的 EMA 上，使用相同随机初始轨迹和 8-step DDIM 比较 Normal/Shuffled Torque 的 Action Prediction Difference RMS，同时计算正常/打乱的真实动作 MSE。所有指标给出 all/free/precontact/contact 分组，避免 free 样本稀释。训练日志记录每个 CA/FiLM 残差更新 RMS、各路径裁剪前梯度 L2 norm 与每参数 RMS。记录初始验证与完整训练曲线，另保存 contact 图像响应热图和动作轨迹干预图。

判定必须联合考虑正常条件下的动作去噪质量、动作误差、打乱力矩导致的质量下降与响应；仅增加 token RMS 或动作变化不构成改善。单 seed、20 个训练 episode、5k 步结果只作先导证据，不等同于完整训练结论。最佳 checkpoint 用于留存，主对照统一使用最后第 5,000 步，避免比较不同训练预算。

入口：`python test/run_teacher_fusion_ablation.py --variant baseline --output-dir /new/run/baseline`，其他 variant 为 `film_only`、`hybrid`。`--preflight-only` 只检查，不更新模型。所有独立检查保存在 `test/`。
