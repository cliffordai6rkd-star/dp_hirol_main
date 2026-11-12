#!/usr/bin/env bash
# Distributed as install-current.sh beside the USB archives.
set -euo pipefail

if (( $# > 1 )); then
  echo '用法: bash install-current.sh [安装目录，默认 $HOME/fdp_training]' >&2
  exit 2
fi
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
destination="${1:-${HOME}/fdp_training}"
if [[ "$destination" != /* ]]; then
  destination="$PWD/$destination"
fi
if [[ -e "$destination/dp_main" ]]; then
  echo "目录已存在，请选择新的安装目录: $destination/dp_main" >&2
  exit 2
fi
for tool in docker tar sha256sum; do
  command -v "$tool" >/dev/null || { echo "缺少命令: $tool" >&2; exit 2; }
done
docker info >/dev/null
cd -- "$bundle_dir"
echo '1/3 校验镜像分片和本次迁移包'
sha256sum --check SHA256SUMS-current
mkdir -p -- "$destination"
destination="$(cd -- "$destination" && pwd)"
echo '2/3 导入训练环境镜像'
cat diffusion-policy-train.tar.part-00 diffusion-policy-train.tar.part-01 \
    diffusion-policy-train.tar.part-02 diffusion-policy-train.tar.part-03 | docker load
echo '3/3 解压当前源码、训练数据和 DINO 到服务器磁盘'
tar --no-same-owner -xzf dp_main-current-source-data-model.tar.gz -C "$destination"
mkdir -p -- "$destination/dp_main/data/cache" "$destination/dp_main/data/outputs" \
    "$destination/dp_main/.cache/huggingface" "$destination/dp_main/.cache/torch"
echo '安装完成，尚未启动正式训练。接下来执行：'
printf 'cd %q\n' "$destination/dp_main"
echo 'bash docker/run_portable.sh check'
echo 'bash docker/run_portable.sh dev'
echo '然后用 VS Code Remote-SSH 连接服务器，Attach to Running Container，选择 fdp-dev，打开 /workspace。'
