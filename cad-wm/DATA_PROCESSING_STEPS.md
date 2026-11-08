# 数据处理流程

## 1. H5 转 LeRobot v3

```bash
python dataset/tool/h5_2_lerobotev3.py \
  --config dataset/config/shape_meta/shape_meta_main.yaml
```

## 2. 添加低维派生特征

```bash
python dataset/tool/lerobot_add_feature.py \
  --input-root data/train_episode/wipe_board/wipe_board_lerobotv3 \
  --input-repo-id wipe_board_lerobotv3 \
  --features acceleration ee_velocity ee_acceleration
```

## 3. ee_pose 从矩阵转 quat7

```bash
python dataset/tool/ee_pose_matrix_to_quaternion.py \
  --datapath data/train_episode/wipe_board/wipe_board_lerobotv3
```

## 4. Rerun 可视化检查

```bash
python dataset/tool/lerobotv3_rerun_visualizer.py \
  --root data/train_episode/wipe_board/wipe_board_lerobotv3 \
  --repo-id wipe_board_lerobotv3
```

## 5. Rerun 接触状态标注

```bash
python dataset/tool/lerobotv3_rerun_visualizer.py \
  --root data/train_episode/wipe_board/wipe_board_lerobotv3 \
  --repo-id wipe_board_lerobotv3 \
  --contact_add
```

## 6. 补标指定 episode

```bash
python dataset/tool/lerobotv3_rerun_visualizer.py \
  --root data/train_episode/wipe_board/wipe_board_lerobotv3 \
  --repo-id wipe_board_lerobotv3 \
  --contact_add \
  --start-episode 28 \
  --end-episode 28
```
