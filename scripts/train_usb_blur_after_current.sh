#!/usr/bin/env bash
# Run the two RGB-blur curricula after --wait-pid exits.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/train_usb_baselines_after_current.sh" \
    --config alation_study/train_dp_image_query_lstm_contact_blur_insert_usb \
    --config alation_study/train_dp_image_wrench_selfattn_contact_blur_insert_usb \
    "$@"
