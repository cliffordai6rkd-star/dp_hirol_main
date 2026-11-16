# civi

DiT 教师、RGB 渲染学生与因果阶段预测的训练仓库，使用 LeRobot v3 数据。

教师：冻结 DINOv3 → 图像投影 → 两层 image-query cross-attention / FFN → context_norm → DiT。力觉输入为当前及历史 8 帧七轴关节外力矩。precontact/contact 打开完整融合分支，free 整体旁路。

学生：双视角 RGB 与力矩 → 独立空间 gate + FiLM RGB 渲染 → 冻结教师视觉编码器 → context tokens。学生图像路径不执行教师的力觉融合。新增的小型 CNN + GRU 根据历史图像、关节位置/速度、连续外力矩预测融合开关；训练和推理都使用预测结果。

## 训练

```bash
cd /home/lcy/code/civi
conda activate fdp
python train.py
```

默认训练 board 教师，再用最终 150,000 步教师的 EMA 权重蒸馏学生。新实验输出到 `data/outputs/YYYY.MM.DD/<时间>_<任务>/`，包含 `teacher/`、`student/` 和 `pipeline.json`。

- 教师：batch 32，150,000 个优化步；前 50,000 步高斯模糊 σ=5，后 100,000 步线性降到 0；只模糊 precontact/contact 图像。验证、推理和蒸馏目标使用清晰图像。
- 学生：batch 16；前 5 个 epoch 训练空间 gate 与阶段子网络，随后固定空间 gate、训练 RGB 与阶段子网络。RGB 验证 token MSE 连续 5 轮无至少 0.0001 改善时早停，最多 1000 epoch。
- 开关：`p_open >= 0.5`；阶段 BCE 权重 1，token MSE 不反传到阶段子网络。

```bash
# 预览命令，不启动训练
python train_pipeline.py --dry-run

# 只训练教师
python train_pipeline.py dp --gpus 0

# 已有教师：明确使用最终权重
python train_pipeline.py distill --teacher-checkpoint /path/to/teacher/checkpoints/latest.ckpt --gpus 0

# 黄瓜教师 + 学生
python train_pipeline.py all --gpus 0 \
  --dp-config diffusion_policy/config/train_civi_teacher_huanggua_blur_legacy_fusion_b32_150k.yaml \
  --distill-config diffusion_policy/config/train_distill_render_net_huanggua_legacy.yaml
```

默认数据集为 `data/board_25hz_3cam_lerobotv3_clean_ma5_l1_12p5`。也可设置 `CIVI_DATASET_PATH`，本地 DINO 权重位于 `model/model/dinov3-vitb16-pretrain-lvd1689m`。单卡训练为默认；多卡参数仍保留在 `train_pipeline.py` 中。

## 阶段子网络与推理

每个观测独立取 16 帧状态历史，以及两个相机在 `[-12,-8,-4,0]` 帧的 RGB。状态使用原始七维关节位置、七维速度、未阈值化的七维外力矩。历史在 episode 开头重复首帧，不读取之后的观测。状态标准化仅拟合训练 episode，随学生 checkpoint 保存。

子网络共 112,737 个参数；`precontact/contact=1`，`free=0`。推理需要同样的观测历史，无须离线阶段标签。仅渲染 RGB 的入口：

```bash
python render_distill_render_net.py --checkpoint /path/to/student/checkpoints/best.ckpt --episode 0
```

接口、损失和日志见[阶段预测设计](docs/causal_phase_gate.md)。与 11 月 7 日参考实验的差异见[方法核对](docs/board_huanggua_method_reproduction_20261116.md)。

## 代码与环境

保留的主链包括 `diffusion_policy/model/diffusion/` 的 DiT、教师策略、RenderNet、阶段预测、LeRobot 数据读取和缓存、力矩处理、训练与可视化。历史模拟任务、UNet/ACT 等策略和弃用实验已移除；数据、训练输出、预训练模型、历史报告与 Git 历史保留。

Python 导入包继续叫 `diffusion_policy`，以便读取已有 checkpoint；项目发行名与仓库目录为 `civi`。本机旧路径 `dp_hirol_main` 是兼容符号链接，供正在运行的任务使用。

```bash
# 新环境
conda env create -f conda_environment.yaml

# 当前核心测试
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Docker 用法见 [Docker 说明](docker/DOCKER_CN.md)。清理记录及源码备份位置见[清理记录](docs/cleanup_20261116.md)。
