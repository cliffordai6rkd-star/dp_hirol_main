#!/usr/bin/env bash
set -euo pipefail

# One model replica per visible GPU; batch_size stays global by default.
# ++ also supports saved run configs that predate the distributed settings.
# Hydra's positional overrides must follow its options (e.g. --config, --cfg).
hydra_args=()
overrides=()
for arg in "$@"; do
  case "$arg" in
    -*) hydra_args+=("$arg") ;;
    *=*|'~'*) overrides+=("$arg") ;;
    *) hydra_args+=("$arg") ;;
  esac
done

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-4}" \
  train_distill_render_net.py \
  "${hydra_args[@]}" \
  ++training.distributed.enabled=true \
  ++training.distributed.timeout_seconds=86400 \
  teacher_cache.load_into_ram=false \
  dataset.preload_images=true \
  dataset.load_result_add=ram \
  "${overrides[@]}"
