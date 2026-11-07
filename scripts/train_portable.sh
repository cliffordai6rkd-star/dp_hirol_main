#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mode="${1:-all}"
if (( $# )); then shift; fi
if [[ -n "${TRAIN_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
fi
gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
nproc="${NPROC_PER_NODE:-$gpu_count}"
if [[ ! "$nproc" =~ ^[1-9][0-9]*$ ]] || (( nproc > gpu_count )); then
  echo "可见 GPU=$gpu_count，NPROC_PER_NODE=$nproc。请先检查 NVIDIA 驱动和容器 GPU 支持。" >&2
  exit 2
fi
echo "使用 $nproc 个 GPU 进程；全局 batch size 沿用配置，由各 GPU 平分。"
if [[ "$mode" == check ]]; then
  exec python -m torch.distributed.run --standalone --nproc_per_node="$nproc" \
    scripts/check_ddp_runtime.py "$@"
fi
args=("$mode"
  --dp-config diffusion_policy/config/sim_force_aware_dp_train_20hz_lstm2_gripper_contact_gate.yaml
  --distill-config diffusion_policy/config/train_distill_render_net_patch_context.yaml
  --nproc-per-node "$nproc")
case "$mode" in
  all|dp)
    args+=(--dp-override task.dataset.video_backend=pyav
           --dp-override task.dataset.video_decode_device=cpu)
    ;;
esac
case "$mode" in
  all|distill)
    args+=(--distill-override dataset.video_backend=pyav
           --distill-override dataset.video_decode_device=cpu
           --distill-override training.compile.enabled=false)
    ;;
esac
# Use CPU video decoding and eager execution for the initial deployment.
# Later overrides can enable compilation; architecture and losses are unchanged.
exec python train_pipeline.py "${args[@]}" "$@"
