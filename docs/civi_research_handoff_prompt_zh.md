# 给研究协作者 AI 的仓库阅读提示词

以下正文可直接交给能够读取本仓库的另一个 AI。目的在于建立可信的研究背景和实验脉络，后续才据此讨论论文。正式训练已经先行启动；本提示词不授权追加训练、调参或修改当前运行。

---

你是一名研究协作者。请先充分理解这个仓库里“力矩信息通过受约束 RGB 增强迁移到冻结视觉策略”的研究动机、实际实现、实验过程与证据边界。**这次先不要写论文正文，也不要把我们现在的方案包装成已经得到验证的最终方法。**

我要你回答的是：我们一开始想解决什么问题；先后检验了哪些假设；每次实验观察到了什么；这些结果排除了什么、没有排除什么；为什么最终选择当前的直接力矩 RenderNet + 冻结 DiT 行为蒸馏。请把这条演进链解释清楚，而不是只复述当前网络图，或者倒推一条“每一步都成功”的故事。

## 1. 证据与版本规则

- 仓库为 `dp_hirol_main`，工作分支 `V`。本轮重构基准为 `d92d1a6e4444901794a745e773267a47876b459c`；核心整合提交为 `9422d50`，正式 batch64 配置及工程证据提交为 `469b9c9`。先核对你读到的实际 HEAD、工作区和后续提交。
- 当前实现以有效 YAML、实际调用路径、依赖 manifest、训练日志和原始报告为准。Markdown 里包含不同日期、不同教师和不同结构的历史记录，不可把它们拼成同一个现行系统。
- 逐项区分：研究动机、预先提出的假设、实际跑完的实验、实验观察、由观察支持的判断、尚未验证的解释。
- 重要事实请给出仓库文件与行号；数值要注明教师版本、模型权重类型、数据范围、精度、比较基线和实验预算。找不到原始数据时明确写“只找到报告 / 原始文件不可用”，不得补造。
- 只读分析即可，不启动训练、Oracle、结构搜索或 benchmark，不停止任何现有进程，不改当前实验配置。

## 2. 优先阅读当前正式实现

先完整阅读下列入口，再按实际 import 和调用链深入：

1. `docs/civi_dit_minimal.md`
2. `diffusion_policy/config/train_civi_student_dit_minimal.yaml`
3. `diffusion_policy/config/train_civi_student_dit_minimal_100epoch.yaml`
4. `train_distill_render_net.py`
5. `diffusion_policy/workspace/train_civi_student_dit_workspace.py`
6. `diffusion_policy/distillation/dit_rgb.py`
7. `diffusion_policy/model/vision/render_net.py`
8. `diffusion_policy/model/vision/wrench_preprocessing.py`
9. `diffusion_policy/model/vision/student_context_adapter.py`
10. `diffusion_policy/model/diffusion/context_transformer_for_diffusion.py`
11. `diffusion_policy/policy/force_aware_diffusion_transformer_policy.py`
12. `diffusion_policy/dataset/civi_dit_dataset.py`
13. `diffusion_policy/policy/civi_rgb.py`
14. `tests/test_civi_dit_minimal.py`
15. `docs/reports/civi_dit_minimal_v1/README.md` 及该目录 JSON。

原分阶段 workspace 及配置仍然保留：

- `diffusion_policy/workspace/train_distill_render_net_workspace.py`
- `diffusion_policy/config/train_distill_render_net_student_frozen_dit.yaml`
- `diffusion_policy/config/train_distill_render_net_student_response_adapter.yaml`
- `diffusion_policy/config/train_distill_render_net_student_response.yaml`
- `diffusion_policy/config/train_distill_render_net_student.yaml`
- `diffusion_policy/dataset/render_net_teacher_cache.py`

这些文件用于理解沿用的机制与历史版本；不能仅因某个旧类仍在仓库，就认定它在当前正式配置里启用。

## 3. 当前问题定义与结构：请从代码核验

研究意图是让一个可部署的 RGB 渲染模块利用因果力矩历史，以有界、小幅、受空间与阶段门控的图像修改，把力觉相关信息送入既有冻结视觉策略。增强 RGB 是中间接口，行为迁移才是本轮主目标。它不是直接学习一个新的动作策略头，也不是证明任意教师 Token 都能由受约束 RGB 精确实现。

当前轻量部署路径：

`两帧真实因果 RGB + 对应的原始单位 tau_ext 历史 → 教师一致的力矩预处理 → 基础 RenderNet → 空间 Gate × Alpha 约束的 RGB 残差 → 增强 RGB`

- 基础网络为 CNN Pyramid / Skip Connections / 普通 RGB Decoder；两层直接力矩 LSTM；6 层视觉 SA 后接 2 层 image-query residual CA；patch8、embed128、3 级 Decoder。CA 的 Q 来自图像，K/V 是 8 个直接力矩 LSTM tokens。这个深度被固定用于本轮，并未被证明最优。
- 当前不启用 Scene–Torque Response Predictor、Response Projection、196 个预测响应 tokens 的 CA K/V、response-conditioned D wrapper、dual_path、新增 Decoder FiLM、Decoder 额外 Spatial Modulation、Patch RGB / Jacobian / Oracle 生成器或学生部署 DINO 分支。
- Condition Network 及 RGB 输出端的空间 Gate × Alpha 保留。训练时可以使用与其 hash 绑定的缓存输出；部署必须用当前因果输入重新计算，不能读取 phase/alpha 标签或 sample ID。
- 保留既有 hard 残差公式、tanh 与精度顺序、RGB 范围及 epsilon=0.1 幅度约束。不要把历史 smooth 参数化实验误写成现行公式。
- 强制 alpha=0 的严格 RGB 恒等，与真实物理零力矩经过预测门控的行为是两个测试。后者并不保证预测 alpha=0，也不能仅凭前者断言真实 Free 场景零修改。
- 两帧历史必须是真实观测；每个观测对应 8 点 raw torque 历史。不要把重复当前帧描述为因果历史。相机键与顺序以有效配置为准。

当前训练读出路径另外包含冻结 DINO / 投影 / 嵌入 / context_norm、匹配的 MLP Adapter 和冻结 DiT。轻量 RGB 导出不包含这些模块；如果继续用增强图像验证原冻结策略，则必须保留与训练一致的视觉读出和 Adapter。请分别画清训练图、RGB 导出图、增强 RGB 到冻结策略的验证图。

## 4. 教师、数据、初始化与监督

教师固定为同一个 **150,000-step EMA**：

`data/outputs/2026.11.15/final/teacher/checkpoints/optimizer_step=00150000-val_action_mse=0.00514941.ckpt`

- state key：`ema_model`
- 文件 SHA256：`33d59938df0092a010a37d20076b1512c6a284f0dd0197460b6f94a8b9d266fa`
- EMA 张量状态 SHA256：`2ea7da45c0fe179994c610e8742fbf9582a6e6247dab8c5469b71b00714e76bd`

历史匹配快照的容器 SHA 不同，但本轮核验了参数状态一致性。请根据配置、manifest 和加载审计解释，不能把文件 hash 不同与参数必然不同混为一谈。禁止把历史 70,750-step 或 180k 教师的结果标成当前 150k 教师结果。

教师与学生使用 **board** 数据：`data/board_25hz_3cam_lerobotv3_clean_ma5_l1_12p5`，不是用户中途误说的 huanggua。学生沿用保存的 56 个训练 episode、17,740 窗口（Free 13,419 / Precontact 740 / Contact 3,581）；全部验证 1,893 窗口（1,317 / 108 / 468）。内部选择 episode 为 38/44/53/60，5/26 只保留报告；它们曾用于历史教师评价，不能称为完全未使用的最终测试集。还应核查教师与学生各自的划分及重叠，区别学生内部泛化与教师从未见过的泛化。

Stage1/2 复用与当前教师匹配、经过 hash 审核的残差 GELU MLP Adapter（256→128→256）及 Condition 快照，没有为本轮重复训练冻结模块。Stage3 只训练基础 RGB Renderer（含直接力矩 LSTM）。新 Renderer 为明确记录的基础初始化，允许按审计清单从同一教师初始化语义一致的 torque 路径；不把旧 step6000 response-conditioned Renderer 当成完整兼容初始化。

当前唯一主蒸馏目标为冻结 DiT 输出回归。对同一个示教动作、同一个本步随机 epsilon 与 rho：

```text
A_rho = teacher_scheduler.add_noise(normalize(A), epsilon, rho)
ZT = 原图 + 正确力矩经过教师原生路径的 context
ZS = 增强 RGB 经过冻结视觉读出 + 已校准 Adapter 的 context
uT = frozen_DiT(A_rho, rho, ZT)  # 停止梯度
uS = frozen_DiT(A_rho, rho, ZS)  # 保留对输入的梯度
L_KD = mean_valid_elements((uS - uT)^2)
```

KD 权重 1；Token、预测响应、方向、特征差分和示教辅助训练权重均为 0。训练随机抽 rho/epsilon，固定探针只用于评价。冻结指参数不更新，学生 RGB→视觉→Adapter→DiT 路径仍要传回梯度。训练不反传完整扩散采样链。

当前 checkpoint 审计为 epsilon prediction / 7D joint-position / 非相对动作 / horizon9 / execution start1 / chunk8 / 原生 DDIM8；请再沿真实实现核对 normalization、padding、mask 与额外条件，不能只抄这段概述。单步 denoiser KD、对示教目标的误差、完整采样动作误差是不同指标。

## 5. 按证据重建历史演进

推荐依次阅读：

| 阶段 / 阅读材料 | 重点理解 | 必须保留的限制 |
|---|---|---|
| `docs/network_revision_2026_11_12.md` | 基础图像查询力矩 CA 路径、历史恢复及模块变化 | 不同提交里的 FiLM/phase 等不能混成当前结构 |
| `docs/student_context_adapter.md`、`docs/response_weighted_student.md` | 为什么引入物理零力矩视觉校准、阶段门控、Token / 响应监督 | 涉及历史 180k 教师，不能与当前 150k 结果直接比较 |
| `docs/student_dual_path_renderer.md`、`docs/student_paired_response_full_data.md` | 双路径、SA/CA、FiLM、Adapter 的具体对照与预训练依赖 | A/B 同时改深度与次序时，不能把效果全归因于交错 |
| `docs/teacher_best_response_study.md` | 教师响应稳定性、逐图像 Oracle、共享 RGB、局部可达性检查 | 主要为 70,750-step 教师；逐图像可达不等于共享可学，更不等于泛化 |
| `docs/civi_student_iteration_plan.md` 及账本/原始报告 | Predictor 拟合、RGB 实现、真实 Teacher 迁移的区别；精度、参数化和其他已执行实验 | 计划与已完成结果分开；不要把三个误差写成可加分解；旧实验不自动套当前目标 |
| `docs/civi_local_updates_20.md` | 像素更新、单样本网络与 4 Contact 共享平均梯度的短程诊断 | 20 步局部结论；RGB 位移未完全匹配；梯度余弦只作辅助 |
| `docs/civi_mixed_updates_20.md` | 固定 8 样本、保持 Contact 系数 /8 的 Mixed 与 Contact-only 对照 | 本批证据不支持继续阶段权重搜索，也不证明所有批次无冲突 |
| `docs/civi_convergence_fp32.md` | 现存 0/1000/2800/3000 权重的完整数据 FP32 复评与真实曝光 | 不用 episode 中点代表全训练集；不能比较 train/val 原始 MSE 就判泛化 |
| `docs/civi_budget_extension.md` | 精确恢复 3000→6000 的有限续训、真实教师微小收益与 Free 代价 | 同一 seed 的多节点一致不是独立 seed 复现；没有证明全局不可表达 |
| 当前配置、实现与 `docs/reports/civi_dit_minimal_v1/` | 为什么本轮改为直接 DiT 行为目标并去掉中间预测链；工程正确性和性能证据 | 新目标是待检验的假设，不能说已证明优于全部 Token 路线 |

其中有几项应认真核对原表，并用合适的因果强度解释：

- 20 步共享 Contact 更新使预测响应实现 MSE 降 9.74%、真实教师响应 MSE 降 5.52%，4 个 Contact 均改善。单样本自身也能改善，虽然会损害其他 Contact；因此这轮更支持存在共享局部方向，而不是“共享网络必然无法表达”。A/B/C 修改量未匹配，不能据改善幅度下容量结论。
- 同批 Mixed / Contact-only 的 Contact 改善接近（预测实现误差约降 9.590% / 9.746%），实际 RGB 位移也接近；Mixed 还改善了 Precontact。它没有支持继续删除 Free 或搜索阶段权重。
- 全数据复评表明早期训练与内部验证仍同向小幅改善，随后有限延长仍有真实教师收益，但收益很小、Free 代价存在。3000→6000 内部 Contact Teacher 配对改善宏均约 0.3961%，内部 Free 实际响应 MSE 增约 11.5278%；报告建议停止该路线的预算延长。
- 这些实验提示要区分“优化未充分”“中间目标是否有用”“受约束图像能否实现”“共享模型可学性”“跨 episode 迁移”。它们并未单独证明 Token 拟合是根因，也未证明新 DiT 目标一定成功。
- `docs/crvtf_student_adaptation.md` 涉及另外的 ACT / self-attention 等历史分支。可以解释背景，但不能把它当成本轮 DiT 正式实现。

## 6. 当前正式训练与工程证据

当前代码经过完整数据 80-update 实现 smoke、额外 batch64 的 4-update 接通检查、严格恢复与部署导出检查。累计工程实际 optimizer calls 为 **94**，含失败重放尝试，未把它们伪称为正式收敛训练。聚焦测试 33 项通过；较大历史测试集仍有基准提交上已存在的旧 API 测试不兼容，详见报告，不能宣称全仓库全部测试通过。

80 步 smoke 的内部 Contact KD **0.107724→0.112807，恶化约 4.72%**，但完整采样动作 MSE **0.00143092→0.00132827，改善约 7.17%**。指标不一致，smoke 只证明端到端实现能跑，不证明蒸馏有效。不要隐藏前一项，只挑后一项写成成功。

用户随后单独授权正式 **100 个名义 epoch、early stopping patience5、global batch64、双卡24GB**。此新目标的正式预算与旧 20k 探索预算分开记账。训练已在 GPU2/3 启动，运行源码固定为 `469b9c9`；状态快照见 `docs/reports/civi_dit_minimal_v1/formal_launch.json`。该快照是启动时刻的记录，实时进度必须重新读日志。

- 服务器运行目录：`/home/djhpc/code/dp_hirol_main/data/outputs/civi_dit_minimal_b64_s42_100epoch`
- 每卡 Renderer microbatch8 × accumulation4 ×2 ranks = global64；Mixed F16/P16/C32；没有自动 LR 缩放。
- 名义 epoch 为 ceil(17740/64)=278 更新；最大 27,800 更新。Mixed 的 phase/episode 队列不等同于每个 epoch 恰好均匀遍历一次全部窗口，曝光应单独计数。
- 训练为 FP32 master/loss + Renderer BF16，冻结视觉与 DiT FP32；视觉 encode 固定 B2 分块并保留梯度。新 FP32 参考评价不与历史 BF16 指标拼曲线。
- 每个 epoch 评价全部验证样本和既定完整动作采样；内部 Contact KD episode 宏平均选模，5/26 不参与选模与早停；同时保留 Precontact、Free 与动作指标。
- `train.jsonl` 为训练 batch 日志，`validation_<step>.json` 及 samples/episodes 文件是全量验证；`learning_curves.png` 每次验证后更新。只有 step0 时没有验证下降趋势可下结论。
- 数据缓存只含固定教师 context、固定 Condition 输出及可审计静态信息；不缓存更新中的学生视觉特征或随 rho/noisy action 变化的 DiT 输出。time token 参与 condition_encoder，不能跨 rho 缓存其 memory。
- 已测视觉 compile / eager、精度、microbatch、DDP 梯度、恢复与吞吐；只采用有证据的开关。具体速度、启动成本与显存从 JSON 表读取，不承诺固定倍数，也不把减少任务计算当等义加速。

现在不要预判正式训练成功；若新日志尚未取得足够验证点，明确“训练中，尚无收敛结论”。KD 改善也不自动等于真实机器人成功率提高。

## 7. 请交付的理解报告

请用中文输出以下内容，重点是解释、证据和边界：

1. **研究问题与最初设想**：为什么希望通过受约束 RGB 接口迁移力觉信息；冻结哪些已有能力；希望获得什么部署形式。
2. **教师—学生—门控—视觉校准的当前关系**：各模块输入、输出、训练归属；三张清楚的训练 / RGB 导出 / 冻结策略验证结构图。
3. **按时间排序的实验演进表**：每步的假设、准确控制变量、目标、数据/教师/精度/预算、关键观察、支持及不支持的判断、为什么进入下一步。结构对照的混杂因素应单列。
4. **为什么走到当前方案**：保留直接力矩基础 RenderNet、Condition 与有界 RGB 门控，移除预测响应中间链，把主监督移到冻结 DiT 行为的理由；明确哪些是证据驱动、哪些仍是设计假设。
5. **当前有证据可说 / 尚不可说**：尤其是局部可达性、共享参数、Loss 冲突、泛化、精度、真实教师迁移、动作采样和机器人闭环效果之间的区别。
6. **实验与文件索引**：让人能从每项结论回到代码、配置、表格与原始日志；列出缺失文件或冲突证据。
7. **供后续论文讨论的材料清单**：哪些图表现在能支撑，哪些结论尚缺验证。此处只列缺口，不自动追加实验，也不要马上生成摘要、贡献宣言或论文正文。

不要用“新结构必然更优”“已经解决力觉迁移”“Token 全局不可表达”“通过 smoke 即策略有效”等未经支持的叙述。请保留失败、负结果、轻微收益和不确定性；这些也是理解我们为什么迭代到现在的重要部分。
