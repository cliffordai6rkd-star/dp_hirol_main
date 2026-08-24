#!/usr/bin/env bash
set -euo pipefail

# One DDP job: every torchrun process owns one visible GPU and one data shard.
CONFIG_NAME="${CONFIG_NAME:-train_dp_inert_usb}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train.py \
  --config-name="${CONFIG_NAME}" \
  "$@"
