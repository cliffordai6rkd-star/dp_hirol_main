# Calibration V2 RobotMotion Auto Guide

本文档说明 `pointcloud/calibration_v2` 下基于 `RobotMotion` 的 AprilTag 自动采集和手眼标定流程。

## 文件

- `robotmotion_auto_config.yaml`: 自动采集配置。
- `data_recording_robotmotion_auto.py`: 通过 `RobotMotion` 移动 Franka、读取相机、检测 AprilTag 并保存样本。
- `calibration.py`: 根据采集 JSON 求解 `eye_in_hand` 或 `eye_to_hand` 外参。

## 标定类型

### Eye-to-Hand

固定外部相机，AprilTag 固定在 Franka 末端或夹爪刚性支架上。

输出:

```text
T_base_camera
```

配置示例:

```yaml
camera:
  name: "third_person_cam"
  cali_type: "eye_to_hand"
```

物理要求:

- 外部相机采集过程中不能动。
- AprilTag 与末端/夹爪之间必须是固定刚体关系。
- Tag 不要求严格贴在 TCP 原点，但不能相对夹爪滑动或晃动。

### Eye-in-Hand

相机安装在 Franka 末端，AprilTag 固定在桌面、夹具或场景中。

输出:

```text
T_ee_camera
```

配置示例:

```yaml
camera:
  name: "ee_cam"
  cali_type: "eye_in_hand"
```

物理要求:

- 末端相机与 Franka 末端之间不能相对移动。
- AprilTag 固定在环境中，采集过程中不能动。

## 安全检查

如果 AprilTag 贴在夹爪末端，必须保持:

```yaml
robot:
  initialize_gripper: false
```

原因: `PikaGripper.initialize()` 会主动打开夹爪到最大宽度。当前自动采集脚本默认会在临时配置中移除 `gripper/gripper_config`，避免初始化 Pika gripper。

真正让 Franka 运动前，将:

```yaml
robot:
  execute_hardware: true
```

保持为 `false` 时，机械臂不会执行运动命令，也不会保存有效采样，除非你显式设置:

```yaml
robot:
  dry_run_record: true
```

## 配置

编辑:

```bash
pointcloud/calibration_v2/robotmotion_auto_config.yaml
```

关键字段:

```yaml
robot:
  motion_config: "factory/components/motion_configs/left_fr3_with_pika_ati_ik.yaml"
  execute_hardware: false
  dry_run_record: false
  pose_source: "model"
  initialize_gripper: false

camera:
  name: "third_person_cam"
  cali_type: "eye_to_hand"
  warmup: 1.0
  display: true

tag:
  family: "tag36h11"
  size: 0.075
  id: null

workspace:
  grid_size: [3, 3, 2]
  spacing: [0.04, 0.04, 0.04]
  center: null
  base_rpy_deg: null
  orientation_randomness: 12.0
  seed: 0

collection:
  settle_time: 2.5
  retry_interval: 0.2
  stop_key: "q"
```

说明:

- `camera.name`: 必须等于 motion config 中 `sensor_dicts.cameras[].name`。
  当前 `left_fr3_with_pika_ati_ik.yaml` 可用 `ee_cam` 和 `third_person_cam`。
- `camera.cali_type`: `eye_in_hand` 或 `eye_to_hand`。
- `tag.size`: AprilTag 黑色外框边长，单位米，不包含白边。
- `workspace.center: null`: 使用脚本启动时当前 TCP 位置作为采样中心。
- `workspace.base_rpy_deg: null`: 使用脚本启动时当前 TCP 姿态作为基础姿态。
- `orientation_randomness`: 每个采样点的随机姿态扰动角度，单位度。
- `collection.retry_interval`: 某个目标位姿没有识别到 tag 时，隔多久重试一次。该位姿只有识别并保存成功后才计数。

## 采集命令

第一次运行前确认 AprilTag 检测依赖已安装:

```bash
python3 -m pip install pupil-apriltags
```

如果报错 `ModuleNotFoundError: No module named 'pupil_apriltags'`，就在当前运行脚本的 Python 环境里执行上面的命令。pip 包名是 `pupil-apriltags`，Python import 名是 `pupil_apriltags`。

在仓库根目录执行:

```bash
python3 pointcloud/calibration_v2/data_recording_robotmotion_auto.py
```

如果要指定另一份配置:

```bash
python3 pointcloud/calibration_v2/data_recording_robotmotion_auto.py \
  --config pointcloud/calibration_v2/robotmotion_auto_config.yaml
```

采集输出默认保存到:

```text
pointcloud/calibration_v2/robotmotion_runs/<timestamp>/
```

目录内会包含:

```text
data_<camera_name>.json
images/
robot_motion_wrapper.yaml
motion_config_without_gripper.yaml
```

其中 `data_<camera_name>.json` 是求解脚本的输入。

采集脚本不会跳过未识别到 AprilTag 的目标位姿。比如 `grid_size: [3, 3, 2]` 会生成 18 个目标位姿，脚本会在每个位姿上反复检测，直到成功保存一个样本后才进入下一个位姿。若需要提前退出，在预览窗口按 `stop_key`，默认是 `q`。

## 求解命令

采集完成后执行:

```bash
python3 pointcloud/calibration_v2/calibration.py \
  --data pointcloud/calibration_v2/robotmotion_runs/<timestamp>/data_<camera_name>.json
```

求解脚本会优先读取 JSON 里的 `metadata.cali_type`。

也可以手动覆盖:

```bash
python3 pointcloud/calibration_v2/calibration.py \
  --data pointcloud/calibration_v2/robotmotion_runs/<timestamp>/data_third_person_cam.json \
  --cali-type eye_to_hand
```

或:

```bash
python3 pointcloud/calibration_v2/calibration.py \
  --data pointcloud/calibration_v2/robotmotion_runs/<timestamp>/data_ee_cam.json \
  --cali-type eye_in_hand
```

输出:

- `eye_to_hand`: `T_base_camera.npy`
- `eye_in_hand`: `T_ee_camera.npy`

求解时还会打印固定 tag 的一致性误差。位置误差越小越好。

## 推荐操作顺序

1. 将 Franka 手动移动到安全位置，保证 AprilTag 在相机视野内。
2. 检查 `initialize_gripper: false`。
3. 先保持 `execute_hardware: false`，启动一次确认相机能打开、tag 能检测。
4. 确认采样区域无碰撞风险后，设置 `execute_hardware: true`。
5. 执行采集命令。
6. 检查采集样本数，建议至少 10 组，最好 15 到 30 组。
7. 执行求解命令。

## 常见问题

### 启动会不会打开夹爪？

默认不会。保持:

```yaml
initialize_gripper: false
```

不要改成 `true`，否则 Pika gripper 初始化会打开夹爪。

### tag 需要贴在 TCP 原点吗？

Eye-to-hand 不需要。Tag 可以贴在夹爪或工具上的任意刚性位置，只要相对末端固定不动即可。

### tag size 应该量哪里？

量 AprilTag 黑色外框的边长，不包含白色留边和整张纸的边缘。

### 相机名怎么选？

看 motion config 里的:

```yaml
sensor_dicts:
  cameras:
    - name: "ee_cam"
    - name: "third_person_cam"
```

然后在 `robotmotion_auto_config.yaml` 中填对应名字。
