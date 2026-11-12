# Contact-informed visuomotor interface

通过 RenderNet 将机器人当前的接触状态表征到 RGB 图像中，再交给原有视觉策略。教师读取双摄像头图像和当前 `tau_ext` 外部关节力矩观测；学生学习渲染与教师接触表征对应的图像，保持后续视觉编码器与动作策略的网络架构。

默认 `tau_ext` 输入是每个观察时刻截至当前帧的 8 个七轴观测，即 `[t-7, ..., t]`，经过因果 2 Hz 低通滤波。教师和学生输入均不读取未来力矩，也不需要世界模型预测。离线训练标签使用确认接触前 0.5 秒的线性 alpha：从 0 逐步增加到接触时的 1，接触期间保持 1，其余时刻为 0；阶段标签对应 free、pre-contact、contact。学生从图像和已观测力矩预测阶段与 alpha，推理时不需要离线标签。保留两个图像观察时刻、原有 LSTM、交叉注意力和动作 Transformer 的隐藏维度及层数。

启用 pre-contact 后保留阈值以下的连续力矩，不再把非接触输入硬置零。`train_civi_teacher_xarm_clean.yaml` 从 `observation.tau_ext_unthresholded` 读取该信号，接触阈值只用于生成标签。

## 本机训练教师并蒸馏学生

```bash
cd /home/lcy/code/dp_hirol_main
conda activate fdp
python train_civi.py \
  --teacher-config diffusion_policy/config/train_civi_teacher.yaml \
  --student-config diffusion_policy/config/train_distill_render_net_student.yaml \
  --teacher-override task.dataset.dataset_path=/path/to/lerobot_with_tau_ext \
  --gpus 0
```

数据集须包含七维 `observation.tau_ext`，并满足教师配置中的图像、动作和时间戳字段映射。配置保留的 LIBERO 默认数据路径目前没有该字段，需要指定包含该字段的数据集。

程序先训练使用当前 `tau_ext` 的教师，再按最低动作 MSE 自动选择教师 checkpoint 并训练学生。默认输出为：

```text
data/outputs/civi/current_tau_ext/teacher
data/outputs/civi/current_tau_ext/student
```

输出目录须是新目录；另一组实验可用 `--teacher-override output_dir=...` 和 `--student-override output_dir=...` 指定。多卡示例：`--gpus 0,1,2,3 --nproc-per-node 4`。默认教师全局 batch size 为 128，学生为 16。

当前 `tau_ext` 教师的输入投影为七维，旧的六维 wrench 教师权重不能直接用于该配置；教师和学生需使用新的输入配置重新训练。

只训练教师：

```bash
python train_civi.py --stage teacher --gpus 0 \
  --teacher-override task.dataset.dataset_path=/path/to/lerobot_with_tau_ext
```

教师完成后，只启动学生时会从当前教师目录选择最低动作 MSE checkpoint：

```bash
python train_civi.py --stage student --gpus 0 \
  --teacher-override task.dataset.dataset_path=/path/to/lerobot_with_tau_ext
```

需要锁定某份使用当前 `tau_ext` 训练的教师时，可显式传入 `--teacher-checkpoint /path/to/current_tau_ext_teacher.ckpt`。

教师默认每 10000 个 optimizer step 进行全量动作推理评估。所有轨迹的有效窗口各评估一次，EMA 模型使用按 episode 和窗口固定的推理噪声；总平方误差除以动作元素数得到 action MSE。最低分更新 `best.ckpt` 并保留 Top-K，每次评估更新 `latest.ckpt` 和 `action_mse.png`。当前教师使用全部 50 条轨迹，且 pre-contact 模式使用离线 alpha，因此该指标衡量带接触时刻标签的训练集拟合程度。

学生使用 CNN 下采样、图像自注意力、图像查询力矩的交叉注意力和 U-Net 解码器。共享条件特征经过一个五通道条件头，输出空间区域、三个阶段 logits 和 alpha；阶段用 CE、alpha 用 MSE 监督。学生复制并冻结教师的力矩投影、LSTM 和相关嵌入，FiLM 实现保留，默认关闭。

默认先 warmup 条件分支 1 个 epoch，再联合优化条件分支与 RGB 渲染分支，条件头保持可训练。联合损失为 token MSE、阶段 CE、alpha MSE 和区域损失。`distillation.use_teacher_ffn: true` 让渲染图像复用冻结教师的融合 FFN，对齐教师的“Cross-Attention＋视觉残差 → FFN＋残差 → Context Norm”最终 token。教师融合使用离线 alpha；学生渲染使用自己预测的 alpha。学生验证集按 episode 单独划分，checkpoint 按 EMA 的 token MSE 选择，辅助损失单独记录。

## 从 U 盘安装到服务器

项目源码、数据和输出通过目录挂载保存，Docker 镜像提供依赖。目标机需要 Linux x86_64、支持 CUDA 12.8 的 NVIDIA 驱动、Docker Compose v2 和 NVIDIA Container Toolkit。

进入 U 盘的 `diffusion-policy-train` 目录执行：

```bash
bash install-current.sh "$HOME/fdp_training"
cd "$HOME/fdp_training/dp_main"
bash docker/run_portable.sh check
bash docker/run_portable.sh dev
```

安装脚本校验文件、导入镜像并解压项目；`check` 检查 GPU、通信、数据和本地 DINO；`dev` 启动后台开发容器 `fdp-dev` 并将项目挂载到 `/workspace`。

VS Code 安装 Remote-SSH 和 Dev Containers 扩展后，连接服务器，再执行 `Dev Containers: Attach to Running Container...`，选择 `fdp-dev`。打开 `/workspace`，选择解释器 `/opt/venv/bin/python`，容器内直接运行：

```bash
cd /workspace
python train_civi.py --gpus 0
```

源码修改和训练输出保存在服务器项目目录。详细步骤见 [服务器迁移与 VS Code 操作说明](docker/SERVER_VSCODE_CN.md)。

## 历史学生空间监督修复

11 月 8 日迁移包的零力对照输入曾沿用原样本的 `contact_alpha`。当前代码会根据零力重新计算 alpha=0，并使用 `context_effect_v2` 缓存标识。该修复涉及学生空间 mask 监督，旧实验和数据文件保留；新方案从头生成教师和学生目标。

针对仍运行旧迁移包的服务器，历史补丁说明及文件保存在 `docker/artifacts/fdp-alpha-counterfactual-fix.patch`。
