# MuJoCo FR3 遥操指南

本目录提供一套 FR3 + Pika 夹爪的 MuJoCo 遥操链路：

```text
Xbox 手柄 -> 末端目标位姿 -> Pinocchio IK -> 7 维机械臂 q_des -> MuJoCo
                                RT/R2 切换夹爪开闭 -> MuJoCo
```

## 文件说明

- `mujocosim_inteface.py`：MuJoCo FR3 接口，包含关节位置控制、夹爪控制、轨迹播放。
- `ik_controller.py`：基于 Pinocchio 和 URDF 的 IK 解算器。
- `xbox_controller.py`：Linux `/dev/input/js*` Xbox 手柄读取器。
- `teleoperation.py`：Xbox 遥操主循环。
- `ik_follow_test.py`：不接手柄的 IK 跟随测试。
- `mujoco_replay.py`：数据集 q 轨迹回放。

默认配置文件：

```bash
config/sim_cfg/replay_test.yaml
```

## 快速启动

先测试手柄输入：

```bash
python sim_mujoco/xbox_controller.py
```

推动摇杆、按 A/Y、按 RT/R2，终端应显示类似：

```text
lx=+0.00 ly=+0.00 rx=+0.00 ry=+0.00 r2=0.00 a=0 y=0
```

启动完整遥操：

```bash
python sim_mujoco/teleoperation.py
```

指定配置文件：

```bash
python sim_mujoco/teleoperation.py --config config/sim_cfg/replay_test.yaml
```

## 当前控制映射

```text
左摇杆 X  -> 末端左右
左摇杆 Y  -> 末端前后
Y 按钮    -> 末端向上
A 按钮    -> 末端向下
右摇杆 X  -> 末端 roll
右摇杆 Y  -> 末端 pitch
RT/R2     -> 按一次切换夹爪开/闭
O 键      -> 复位机器人和遥操 target 到 q_reset
LT        -> 复位机器人和遥操 target 到 q_reset
```

注意：`O` 复位是键盘输入，需要 MuJoCo viewer 窗口获得焦点。先点击 viewer 窗口，再按 `O`。

## 手柄设备配置

默认手柄设备：

```yaml
xbox_device: /dev/input/js0
```

如果设备不存在，检查：

```bash
ls /dev/input/js*
```

当前用户需要拥有该设备的读取权限；必要时将用户加入系统的 `input` 组。

## Xbox 参数说明

```yaml
xbox_device: /dev/input/js0
xbox_deadzone: 0.12
xbox_trigger_deadzone: 0.02
xbox_axis_map:
  left_x: 0
  left_y: 1
  lt: 2
  right_x: 3
  right_y: 4
  r2: 5
xbox_axis_sign:
  left_x: 1.0
  left_y: -1.0
  lt: 1.0
  right_x: 1.0
  right_y: -1.0
  r2: 1.0
xbox_button_map:
  a: 0
  y: 3
xbox_lt_button: null
xbox_r2_button: null
xbox_reset_button: null
```

含义：

- `xbox_device`：Linux joystick 设备路径。
- `xbox_deadzone`：摇杆死区。摇杆居中时机器人仍然漂移，就调大。
- `xbox_trigger_deadzone`：RT/R2 扳机死区。
- `xbox_axis_map`：把语义输入映射到 Linux axis 编号。
- `xbox_axis_sign`：轴方向修正。方向反了就把 `1.0` 改成 `-1.0`。
- `xbox_button_map`：把 A/Y 等语义按钮映射到 Linux button 编号。
- `xbox_lt_button`：只有当 LT 被驱动识别成 button 而不是 axis 时才需要设置。
- `xbox_r2_button`：只有当 RT/R2 被驱动识别成 button 而不是 axis 时才需要设置。
- `xbox_reset_button`：可选复位组合修饰键按钮。设置后 `(xbox_reset_button + A)` 也会触发复位。

校准流程：

1. 运行：

   ```bash
   python sim_mujoco/xbox_controller.py
   ```

2. 一次只动一个摇杆/按钮/扳机。
3. 如果数值不动，修改 `xbox_axis_map` 或 `xbox_button_map`。
4. 如果方向反了，修改对应的 `xbox_axis_sign`。
5. 如果 RT/R2 的 `r2` 一直不变，但某个按钮变化，把 `xbox_r2_button` 设成那个按钮编号。
6. 如果 LT 的 `lt` 一直不变，但某个按钮变化，把 `xbox_lt_button` 设成那个按钮编号。
7. 如果你想用 LB+A 或其他按钮+A 复位，把 `xbox_reset_button` 设成对应按钮编号。

## 遥操参数说明

```yaml
teleop_translation_speed: 0.15
teleop_rotation_speed: 0.8
teleop_left_x_index: 1
teleop_forward_index: 0
teleop_up_index: 2
teleop_workspace_min: [0.15, -0.45, 0.08]
teleop_workspace_max: [0.75, 0.45, 0.85]
teleop_gripper_open_ctrl: -0.11
teleop_gripper_closed_ctrl: 0.0
teleop_gripper_open_q: [-0.04, 0.04]
teleop_gripper_closed_q: [0.0, 0.0]
teleop_gripper_initial_closed: false
teleop_gripper_toggle_threshold: 0.5
teleop_reset_combo_threshold: 0.5
teleop_reset_requires_a: false
teleop_reset_hold_steps: 20
teleop_reset_joint_speed: 0.8
teleop_reset_position_tolerance: 0.01
```

含义：

- `teleop_translation_speed`：末端最大平移速度，单位约为 m/s。
- `teleop_rotation_speed`：末端最大 roll/pitch 速度，单位约为 rad/s。
- `teleop_left_x_index`：左摇杆左右对应的世界坐标轴。
- `teleop_forward_index`：左摇杆前后对应的世界坐标轴。
- `teleop_up_index`：A/Y 上下对应的世界坐标轴。
- `teleop_workspace_min/max`：末端 target 的位置限幅。
- `teleop_gripper_*_ctrl`：MuJoCo 夹爪 actuator 控制量。
- `teleop_gripper_*_q`：Pinocchio IK 内部同步的夹爪关节状态。
- `teleop_gripper_initial_closed`：启动时夹爪 toggle 状态是否为闭合。
- `teleop_gripper_toggle_threshold`：RT/R2 大于该值时认为“按下”。
- `teleop_reset_combo_threshold`：LT 大于该值时触发复位。
- `teleop_reset_requires_a`：设为 `true` 时，需要 LT + A 才复位；默认 `false`，LT 单独复位。
- `teleop_reset_joint_speed`：复位时每个关节向 `q_reset` 回去的最大速度，单位约 rad/s。
- `teleop_reset_position_tolerance`：关节误差小于该值后认为复位完成。
- `teleop_reset_hold_steps`：旧的保持参数，目前主要保留兼容。

坐标轴索引：

```text
0 -> X
1 -> Y
2 -> Z
```

如果手感太猛，先降低速度：

```yaml
teleop_translation_speed: 0.07
teleop_rotation_speed: 0.4
```

如果某个方向突然不动，通常是末端 target 撞到了 `teleop_workspace_min/max`。

## IK 参数说明

```yaml
ik_urdf_path: sim_mesh/franka_fr3/fr3_pika_gripper_ati.urdf
ik_ee_frame_name: pika_gripper_ee
ik_lock_joint_names: []
ik_arm_joint_names:
  - fr3_joint1
  - fr3_joint2
  - fr3_joint3
  - fr3_joint4
  - fr3_joint5
  - fr3_joint6
  - fr3_joint7
ik_gripper_joint_names:
  - gripper_left_joint
  - gripper_right_joint
ik_default_gripper_q: [-0.04, 0.04]
ik_max_iterations: 80
ik_tolerance: 1.0e-4
ik_damping: 1.0e-3
ik_step_size: 0.5
ik_position_weight: 1.0
ik_orientation_weight: 0.5
ik_reject_unconverged: false
ik_max_position_error: 0.03
ik_max_orientation_error: 0.5
ik_max_joint_delta: 0.12
ik_max_joint_delta_norm: 0.35
teleop_pose_quat_order: wxyz
```

要点：

- `ik_urdf_path` 是完整 FR3 + 夹爪 URDF，不是只有末端。
- `ik_ee_frame_name` 是用于跟踪 target 的末端 frame。
- `ik_lock_joint_names: []` 表示不锁夹爪关节。
- IK 解算时会保留当前夹爪 q，不让夹爪参与“凑”末端位姿误差。
- 下发给 MuJoCo 机械臂的仍然只有 FR3 7 维 arm q。
- IK 安全检查会拒绝可疑解，避免极端动作导致奇怪姿态被下发。

一般使用 `pika_gripper_ee` 控制夹爪 TCP。如果只想控制法兰，可以改成 `fr3_link8`。

IK 安全保护参数：

- `ik_reject_unconverged`：设为 `true` 时，不收敛的 IK 解全部拒绝。
- `ik_max_position_error`：末端位置误差超过该值时拒绝。
- `ik_max_orientation_error`：末端姿态误差超过该值时拒绝。
- `ik_max_joint_delta`：单个 arm 关节单周期跳变超过该值时拒绝。
- `ik_max_joint_delta_norm`：arm 关节整体跳变过大时拒绝。

如果还会出现奇怪姿态，优先收紧：

```yaml
ik_max_joint_delta: 0.08
ik_max_joint_delta_norm: 0.25
ik_max_position_error: 0.015
```

## 频率参数

```yaml
sim_frequency: 500.0
control_frequency: 100.0
```

- `sim_frequency`：MuJoCo 物理积分频率，会设置 `model.opt.timestep = 1 / sim_frequency`。
- `control_frequency`：上层 target 更新频率。手柄读取、末端 target、IK、q 命令都按这个频率更新。

推荐初值：

```text
sim_frequency = 500 Hz
control_frequency = 100 Hz
```

如果 IK 算得慢，可以把 `control_frequency` 降到 `50.0`。

## 初始状态

```yaml
q_reset: [0.0, -0.185, 0.0, -2.355, 0.0, 1.57079, 0.785]
q_reset_gripper: [-0.04, 0.04]
```

`teleoperation.py` 启动时会先把 MuJoCo 重置到 `q_reset`，避免 position actuator 从 MuJoCo 默认姿态猛拉到目标姿态。

## 测试命令

机械臂位置控制测试：

```bash
python sim_mujoco/mujocosim_inteface.py
```

不接手柄的 IK 跟随测试：

```bash
python sim_mujoco/ik_follow_test.py
```

Xbox 输入测试：

```bash
python sim_mujoco/xbox_controller.py
```

完整遥操：

```bash
python sim_mujoco/teleoperation.py
```

## 常见问题

`Xbox joystick device not found`

- 检查 `ls /dev/input/js*`。
- 修改 `xbox_device`。
- 检查当前用户是否拥有 `/dev/input/js0` 的读取权限。

摇杆居中时机器人仍然移动

- 增大 `xbox_deadzone`，例如从 `0.12` 改到 `0.18`。

摇杆方向反了

- 修改对应的 `xbox_axis_sign`。

RT/R2 不影响 `r2`

- 运行 `python sim_mujoco/xbox_controller.py`。
- 如果 `r2` 一直不变，可能驱动把 RT/R2 暴露成 button。
- 将 `xbox_r2_button` 设置成观察到的按钮编号。

按 `O` 或 `LT` 没有复位

- 先点击 MuJoCo viewer 窗口，让 viewer 获得键盘焦点。
- 按键触发时终端会打印 `reset requested by O key`。
- 如果没有打印，说明 viewer 没收到键盘事件。
- 如果打印了但没有复位，检查是否正在运行 `teleoperation.py` 或 `ik_follow_test.py` 的最新版本。
- 现在复位不是瞬移，而是按 `teleop_reset_joint_speed` 平滑运动回 `q_reset`。

按 `LT` 没有复位

- 先运行 `python sim_mujoco/xbox_controller.py --raw`。
- 确认按 LT 时 `lt` 会变大。
- 如果 `lt` 不变，检查 `xbox_axis_map.lt` 或设置 `xbox_lt_button`。
- 如果你想用 LB 或其他按钮复位，观察 raw 输出里对应的 button 编号，并设置 `xbox_reset_button`。
- 触发时终端会打印 `reset requested by Xbox` 和 `started smooth reset motion to q_reset`。
- 如果你想恢复成 LT+A，设置 `teleop_reset_requires_a: true`，并确认 `a` 会从 `0` 变 `1`。

机器人运动太快

- 降低 `teleop_translation_speed`。
- 降低 `teleop_rotation_speed`。

IK 抖动或不收敛

- 降低 `control_frequency`。
- 降低 `teleop_translation_speed`。
- 增大 `ik_max_iterations`。
- 稍微增大 `ik_damping`，例如 `0.003`。
- 放宽 `ik_tolerance`，例如 `5.0e-4`。

IK 算出奇怪姿态或机械臂瘫掉

- 降低 `teleop_translation_speed`。
- 降低 `teleop_rotation_speed`。
- 降低 `ik_max_joint_delta`，例如 `0.08`。
- 降低 `ik_max_joint_delta_norm`，例如 `0.25`。
- 降低 `ik_max_position_error`，例如 `0.015`。
- 如果想更严格，设置 `ik_reject_unconverged: true`。

机器人像撞到隐形墙

- 检查 `teleop_workspace_min/max`。

跟随的工具点不对

- 检查 `ik_ee_frame_name`。
- 夹爪 TCP 遥操建议使用 `pika_gripper_ee`。
