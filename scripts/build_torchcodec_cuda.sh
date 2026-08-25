#!/usr/bin/env bash
set -euo pipefail

# Build TorchCodec 0.5 against the already-installed PyTorch/CUDA runtime.
# Run this inside the training environment (for example: conda activate dp).
# The script deliberately does not upgrade torch or ffmpeg.

TORCHCODEC_SRC="${TORCHCODEC_SRC:-${TMPDIR:-/tmp}/torchcodec-0.5.0}"
TORCHCODEC_REF="${TORCHCODEC_REF:-v0.5.0}"

if [[ ! -d "${TORCHCODEC_SRC}/.git" ]]; then
  git clone --branch "${TORCHCODEC_REF}" --depth 1 \
    https://github.com/pytorch/torchcodec.git "${TORCHCODEC_SRC}"
fi

cd "${TORCHCODEC_SRC}"
git fetch --tags --depth 1 origin "${TORCHCODEC_REF}" >/dev/null 2>&1 || true
if [[ -n "$(git status --porcelain)" ]]; then
  echo "TorchCodec source tree is dirty; refusing to change its checkout." >&2
  exit 1
fi
git checkout --detach "${TORCHCODEC_REF}"

python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible; refusing to build a CUDA decoder")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

export ENABLE_CUDA=1
python -m pip install --no-build-isolation --no-deps -v .

python - <<'PY'
from torchcodec.decoders import VideoDecoder
print("TorchCodec CUDA extension installed. Pass video_decode_device=auto to the dataset.")
print("A real .mp4 probe is still required to verify FFmpeg/NVDEC support.")
PY
