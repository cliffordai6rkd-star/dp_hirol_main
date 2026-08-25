#!/usr/bin/env bash
set -euo pipefail

# One DDP job: every torchrun process owns one visible GPU and one data shard.
CONFIG_NAME="${CONFIG_NAME:-train_dp_inert_usb}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

# The baseline YAMLs default to single-GPU mode.  This override is required
# when launching with torchrun; callers may pass a later override if needed.

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train.py \
  --config-name="${CONFIG_NAME}" \
  training.distributed.enabled=true \
  "$@"
