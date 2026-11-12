# tau_ext 数据处理与同步视频

在项目根目录运行：

```bash
conda activate fdp
python data_process/process_tau_ext.py
```

默认读取同目录的 `tau_ext_config.yaml`。均值窗口、检测阈值、数据路径和视频设置均在配置中修改，不需要传命令行参数。指定另一份配置时可运行 `python data_process/process_tau_ext.py --config 路径/config.yaml`。

默认分别从两个原始 25 Hz 数据集重新计算，输出到 `*_clean_ma5_l1_12p5`。原始数据、旧的 `*_clean` 数据和已有训练文件不会被覆盖。输出目录已经存在时，脚本会报错；重新处理时请在配置中指定新的输出路径。

处理顺序：

1. 按每段轨迹的时间戳，对速度和测量关节力矩做原来的因果二阶 3 Hz Butterworth 低通滤波；由滤波速度的后向差分计算加速度，再做动力学补偿，获得未阈值的七轴 `tau_ext`。
2. 按关节分别做因果滑动均值，窗口由 `processing.tau_ext_smoothing_window` 控制。默认窗口为 5，即当前帧和前 4 帧；开头不足 5 帧时只平均已出现的帧。每段轨迹重新初始化，不读取未来数据、不跨轨迹平均。25 Hz 下相邻帧间隔 40 ms，五点首尾间隔 160 ms。
3. 计算平滑后七轴的 L1 范数 `sum(abs(tau_ext))`，与 `processing.contact_threshold_l1` 比较。默认阈值为 **12.5 N·m**；大于或等于阈值保留整个七维向量，小于阈值将整个向量置零。这里是关节力矩阈值，单位不是末端力的 N。
4. 重新计算相关 wrench 字段、全局统计和逐轨迹统计，保存处理参数及 URDF 到新数据集，再导出视频。

`observation.tau_ext_unthresholded` 保存平滑后的阈值前向量；`observation.tau_ext_l1` 保存它的 L1 范数；`observation.tau_ext` 是最终阈值后的向量。`meta/tau_ext_processing.json` 记录在线处理参数，`meta/data_process_config.json` 保存这次 YAML 配置及最终解析参数。在线推理可用 `CausalTauExt.from_bundle(新数据集目录)` 保持与导出一致，并在每次 rollout 开头 reset。

默认每个数据集导出 episode 0 的完整视频，位于新数据集的 `visualizations/episode_000_tau_ext.mp4`。视频包含两路相机画面、阈值前和最终 `tau_ext` 的 **L1 范数**曲线、阈值虚线、同步移动时间光标、当前数值与接触状态。这里显示的是 L1，而不是 L2 范数；相机画面按 LeRobot v3 逐轨迹视频偏移和行时间戳对齐。视频导出只使用 CPU，不需要教师模型或 GPU。

在配置中设置 `visualization.episodes: all` 会为每段轨迹分别生成一个视频；设置 `visualization.enabled: false` 可跳过视频。先完成两个数据集的处理，再开始导出视频。
