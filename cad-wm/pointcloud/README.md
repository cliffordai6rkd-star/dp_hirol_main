# 单帧 SAM3 RGB-D 点云重建

这个文件夹用于单帧 RGB-D 点云重建：

```text
RGB 图像 + 对齐后的 depth_meters.npy + RealSense metadata
-> SAM3 或手工 mask
-> 目标物体点云
-> 可选计算最近距离 phi_k
```

先检查脚本参数：

```bash
python pointcloud/single_frame_reconstruct.py --help
```

## 0. 安装依赖与下载 SAM3 权重

### 0.1 安装点云依赖

先完成项目根目录的基础环境安装，再安装点云与视觉依赖：

```bash
./setup.sh
conda activate pinn
python -m pip install -e ".[vision]" \
  git+https://github.com/ultralytics/CLIP.git
python -m pip check
```

基础环境已经包含服务器端使用的 `opencv-python-headless` 和 Pillow
12.2.0；`.[vision]` 只补充 Open3D、RealSense 和 Ultralytics。需要打开
OpenCV/Open3D 图形窗口时，还必须在带显示服务的机器上运行并配置
`DISPLAY`。

注意不要使用：

```bash
pip install -U huggingface_hub
```

因为最新版 `huggingface_hub` 可能和当前环境里的 `lerobot`、`transformers` 版本冲突。

### 0.2 申请 SAM3 权重访问权限

SAM3 权重不是 pip 自动下载的，需要先去 Hugging Face 页面申请/同意访问：

```text
https://huggingface.co/facebook/sam3
```

流程：

```text
1. 登录 Hugging Face
2. 打开 facebook/sam3 页面
3. 同意模型访问条件
4. 到 https://huggingface.co/settings/tokens 创建 Read 权限 token
```

### 0.3 登录 Hugging Face

```bash
conda activate pinn
```

登录：

```bash
hf auth login
```

按提示粘贴 Hugging Face token。粘贴时终端不会显示 token，这是正常的。

当它询问：

```text
Add token as git credential? (Y/n)
```

建议输入：

```text
n
```

因为这里只需要下载权重，不需要把 token 写入 git credential。

检查登录状态：

```bash
hf auth whoami
```

### 0.4 下载 SAM3 权重

在项目根目录执行：

```bash
mkdir -p weights
hf download facebook/sam3 sam3.pt --local-dir weights
```

下载成功后应该有：

```text
weights/sam3.pt
```

之后脚本中使用：

```bash
python pointcloud/single_frame_reconstruct.py \
  --sam-model weights/sam3.pt \
  --box 100 120 420 460
```

### 0.5 如果无法访问 Hugging Face

如果登录或下载时报：

```text
Network is unreachable
```

可在浏览器或其他能访问 Hugging Face 的机器下载 `sam3.pt`，然后放到项目的
`weights/sam3.pt`。

如需使用代理，可在当前终端设置：

```bash
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
hf auth login
hf download facebook/sam3 sam3.pt --local-dir weights
```

### 0.6 依赖冲突修复

如果环境被其他项目升级后出现类似：

```text
lerobot requires huggingface-hub <0.36.0, >=0.34.2
datasets requires fsspec <=2025.3.0
lerobot requires packaging <26.0, >=24.2
```

执行：

```bash
python -m pip install --force-reinstall \
  "huggingface-hub[cli,hf-transfer]==0.35.3" \
  "fsspec[http]==2025.3.0" \
  "packaging==25.0"
```

然后检查：

```bash
python -m pip check
```

## 1. 先用手工 mask 验证几何链路

如果当前环境还没有装 SAM3，建议先传入已有 mask，验证 depth 反投影和点云保存是否正确：

```bash
python pointcloud/single_frame_reconstruct.py \
  --rgb outputs/realsense_rgbd/frame_000000_rgb.png \
  --depth-meters outputs/realsense_rgbd/frame_000000_depth_meters.npy \
  --metadata outputs/realsense_rgbd/frame_000000_metadata.json \
  --mask path/to/mask.png
```

`mask.png` 中非零像素会被保留，零值像素会被过滤。

## 2. 使用 SAM3 框选目标

如果已经安装了支持 SAM3 的 `ultralytics`，可以用 box prompt：

```bash
python pointcloud/single_frame_reconstruct.py \
  --box 100 120 420 460
```

坐标格式是：

```text
x1 y1 x2 y2
```

也可以用 point prompt：

```bash
python pointcloud/single_frame_reconstruct.py \
  --point 320 240
```

## 3. 使用 SAM3 文本提示

如果安装的 `ultralytics` 版本支持 SAM3 concept segmentation，可以用文本 prompt：

```bash
python pointcloud/single_frame_reconstruct.py \
  --prompt "the object"
```

如果文本分割效果不稳定，优先用 `--box` 或 `--point`，因为它们更适合验证几何流程。

## 4. 输出文件

默认输出目录：

```text
outputs/pointcloud_single_frame/
```

里面会生成：

```text
mask.png                  # 最终使用的二值 mask
mask_overlay.png          # mask 叠加到 RGB 图上的检查图
object_pointcloud.ply     # 目标物体点云
summary.json              # 点云数量、边界、phi_k 等信息
unsigned_sdf_grid.npz     # 只有加 --sdf-grid 时才会生成
```

## 5. 计算 phi_k

如果你有一个工具点或末端点在 RealSense 相机坐标系下的位置，可以传入：

```bash
python pointcloud/single_frame_reconstruct.py \
  --mask path/to/mask.png \
  --query-point-camera 0.0 0.0 0.5
```

脚本会在 `summary.json` 中保存：

```text
phi_k_m
```

它表示 query point 到分割点云可见表面的最近距离，单位是米。

## 6. 关于 SDF

单帧 RGB-D 只能看到物体的可见表面，所以这里的：

```text
--sdf-grid
```

保存的是可见点云表面的 unsigned distance grid，不是严格的 signed SDF。

如果要做可靠的 signed SDF，通常需要：

```text
多帧 TSDF 融合
多视角外参标定
CAD/mesh
或明确的接触侧/inside-outside 约定
```

因此当前阶段更推荐先把 `phi_k` 当作 gap distance 使用。
