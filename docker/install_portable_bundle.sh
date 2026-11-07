#!/usr/bin/env bash
# This script is copied to install.sh beside the USB bundle's archives.
set -euo pipefail

if (( $# > 1 )); then
  echo '用法: bash install.sh [本地安装目录，默认 $HOME/dp_training]' >&2
  exit 2
fi
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
destination="${1:-${HOME}/dp_training}"
if [[ "$destination" != /* ]]; then
  destination="$PWD/$destination"
fi
for tool in docker tar sha256sum; do
  if ! command -v "$tool" >/dev/null; then
    echo "缺少命令: $tool。请先阅读同目录的 使用说明.txt。" >&2
    exit 2
  fi
done
if [[ -e "$destination/dp_hirol_main" ]]; then
  echo "目录已存在: $destination/dp_hirol_main；请选择一个新的安装目录。" >&2
  exit 2
fi
cd -- "$bundle_dir"
docker info >/dev/null
echo '1/3 校验 U 盘文件（需要完整读取镜像分片，请耐心等待）'
sha256sum --check SHA256SUMS
mkdir -p -- "$destination"
destination="$(cd -- "$destination" && pwd)"
echo '2/3 导入 Docker 镜像'
cat diffusion-policy-train.tar.part-* | docker load
echo '3/3 解压源码、数据和 DINO 到本地磁盘'
tar --no-same-owner -xzf dp_hirol_main-source-data-model.tar.gz -C "$destination"
mkdir -p -- "$destination/dp_hirol_main/data/cache" "$destination/dp_hirol_main/data/outputs"
echo '安装完成，尚未启动训练。接下来执行：'
printf 'cd %q\n' "$destination/dp_hirol_main"
echo 'bash docker/run_portable.sh check'
echo 'bash docker/run_portable.sh all'
