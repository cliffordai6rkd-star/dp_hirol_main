# 七轴 tau_ext、14 阈值与在线复现

阈值针对七轴合起来的 **L1 范数**，单位为 N·m：

```python
s = sum(abs(tau_ext_unthresholded[i]) for i in range(7))
contact = s >= 14.0
tau_ext = tau_ext_unthresholded if contact else zeros(7)
```

不逐关节判断，不扣除 14，不对保留的分量取绝对值。恰好等于 14 保留。没有滞回、连续帧确认或接触平滑。

## 数据输出

- `data/board_25hz_3cam_lerobotv3_clean`：擦板，在线一致版。
- `data/huanggua1103_25hz_3cam_lerobotv3_clean`：黄瓜，相同处理流程。

仅保留因果处理、七轴 L1 阈值 14 这一版。
两个任务均新增/保存以下字段，原始 q/dq/tau 不变：

| 字段 | 含义 |
| --- | --- |
| `observation.tau_ext` | 七轴阈值处理后的值，训练和推理使用 |
| `observation.tau_ext_unthresholded` | 七轴置零前的值，已做输入滤波和动力学补偿 |
| `observation.tau_ext_l1` | **置零前**七轴 L1 范数，shape `[1]` |
| `observation.contact` | 0/1 接触标志，float32，shape `[1]` |
| `observation.tau_free` | 同步 RNEA 的七轴动力学预测力矩 |

`wrench` 也由阈值处理后的 tau_ext 反算，非接触帧为零。

## 前一版输入处理，离线版仍然相同

1. `q`：读取 `observation.joint`，不滤波，不改变零位；J1..J7，单位 rad。
2. `dq`：读取 `observation.velocity`，**不是对 q 求导**；25 Hz 下二阶 Butterworth、3 Hz 截止的 `sosfiltfilt` 前后向零相位滤波。`padlen=min(episode_length-1,9)`，不跨 episode。前后向滤波会用未来帧，其等效响应不等于单次二阶滤波。
3. `ddq`：`np.gradient(dq_filtered, timestamp, axis=0, edge_order=1)`，内部为中心差分，两端单边差分；单位 rad/s²。
4. `tau`：对 `observation.torque` 做与 dq 相同的 `sosfiltfilt`，再减固定传感器偏置（当前七轴均为零）。假设其为电机施加的关节力矩，单位 Nm。
5. `tau_free = RNEA(q, dq_filtered, ddq)`。
6. `tau_ext_unthresholded = tau_filtered - torque_bias - tau_free`。与前一份 `..._tau_ext` 数据一致，符号约定为 tool-on-environment。
7. 新增步骤：按七轴 L1 是否低于 14 整组置零。没有对 tau_ext 额外滤波、裁剪或归一化；输出为 float32。为使离线和推理的阈值边界一致，先将七轴残差转成 float32，再用 float64 累加绝对值进行比较。

## 在线一致版的输入处理

q、动力学模型、负载、力矩偏置及符号与上述相同，仅更改时间处理：

1. dq 和 tau 各自通过 **单向、二阶 Butterworth、3 Hz 低通**，使用 `sosfilt`；只用过去及当前样本。两路滤波器独立保存状态。
2. 第一个样本用 `sosfilt_zi(sos) * first_sample` 初始化，避免把输入假定从零开始。
3. `ddq_t = (dq_filtered_t - dq_filtered_previous) / (timestamp_t - timestamp_previous)`，只用后向差分。第一个样本的 ddq 为零，因为没有上一帧。ddq 没有再加一层低通。
4. 计算 RNEA、相减和七轴 L1 阈值，代码与数据导出使用同一个 `CausalTauExt.update()`。

模型参数：xArm7 固定基座，重力 `[0,0,-9.81] m/s²`；BIO Gripper G2 总质量 0.79 kg，法兰系质心暂用近似 `[0,0,0.05] m`，质心转动惯量采用零的点质量近似；忽略其他附件。未辨识摩擦、未做自动零点校准，不使用 URDF 的通用 friction/damping 参数。URDF 随数据保存，代码在仓库中共用。**这些近似也会带到在线推理中，14 阈值不能纠正模型误差。**

## 在线调用

在项目根目录运行（需要 numpy、scipy、Pinocchio；处理器不依赖 pyarrow）：

```python
from diffusion_policy.common.xarm_tau_ext import CausalTauExt

root = "data/board_25hz_3cam_lerobotv3_clean"
processor = CausalTauExt.from_bundle(root)  # 自动加载 JSON、模型并核对模型哈希

# 新 rollout 开始时重置一次；不要每帧重新创建或 reset。
processor.reset()

# 每次获得一组同步的 q, dq, tau，在 25 Hz 调用一次。
result = processor.update(q_rad, dq_rad_per_sec, tau_Nm, timestamp_seconds)
policy_tau_ext = result["tau_ext"]                  # float32, shape (7,)
raw_tau_ext = result["tau_ext_unthresholded"]        # float32, shape (7,)
l1_before_threshold = result["tau_ext_l1"]           # Python float
is_contact = result["contact"]                      # bool
```

固定频率为保存的 25 Hz，时间间隔允许 1% + 1e-5 s 的偏差。硬件若为 100 Hz，请先将同步的 q/dq/tau 降采样到 25 Hz，再调用该处理器；不要直接在 100 Hz 上套用 25 Hz 系数。出现采样中断或时间回退时，处理器会拒绝该样本；重新采集连续样本前调用 `reset()`。每条机械臂各持有一个实例。q、dq、tau 要同一时刻对齐，不能混用不同采集周期的值。

后续推理时还需复用模型训练的 normalizer；这里输出的只是物理单位下的 tau_ext，不替代策略自己的归一化。

每个输出目录保存：

- `meta/tau_ext_processing.json`：完整数值配置、SOS 系数、输入处理规则、初始化及采样要求。
- `meta/tau_ext_processing.json` 的 `estimation`：公式、符号、负载来源、算法版本和局限。
- `meta/wrench_model/robot.urdf`：实际用于导出的动力学模型。
- 共用在线处理器：`diffusion_policy/common/xarm_tau_ext.py`，不在每份数据里重复复制。

重新导出命令：

```bash
python scripts/add_xarm_wrench_v3.py \
  --dataset data/board_25hz_3cam_lerobotv3 \
  --output data/board_25hz_3cam_lerobotv3_clean \
  --bio-gripper-g2 --convention tool-on-environment \
  --filter-mode causal --contact-threshold-l1 14 --plot-episode 0
```

输出目录必须不存在。图像位于 `visualizations/`。绘图同时显示置零前后的 L1 与阈值线。

## 模型来源

URDF 原样来自用户指定仓库 `https://github.com/cliffordai6rkd-star/xarm_ws`，commit `172be9d68030aed54adaba25079eb21c595d001c`，文件 `gello_teleop/models/xarm7_dynamics.urdf`；许可见 `LICENSE.xarm_ros2`。BIO Gripper G2 的 0.79 kg 质量来源于官方手册 v2.8.0 第 6 节：`https://www.ufactory.cc/upfiles/download/202610/WJY8KDIK8B1F.pdf`。质心仍是近似值。
