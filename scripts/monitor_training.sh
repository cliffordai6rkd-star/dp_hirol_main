#!/usr/bin/env bash
set -Eeuo pipefail

# Monitor one force-aware training run and resume it from latest.ckpt when the
# checkpoint has not changed for a complete check interval.

repo_dir="${TRAIN_MONITOR_REPO_DIR:-/opt/lcx/code/diffusion_policy}"
run_dir="${TRAIN_MONITOR_RUN_DIR:-$repo_dir/data/outputs/2026.08.01/17.36.23_train_insert_usb_force_aware_dit}"
check_interval="${TRAIN_MONITOR_INTERVAL_SECONDS:-7200}"
target_step="${TRAIN_MONITOR_TARGET_STEP:-40000}"
checkpoint="$run_dir/checkpoints/latest.ckpt"
monitor_dir="$run_dir/training_monitor"
monitor_log="$monitor_dir/monitor.log"
training_log="$monitor_dir/resume_training.log"
lock_file="$monitor_dir/monitor.lock"
pid_file="$monitor_dir/monitor.pid"

mkdir -p "$monitor_dir"

exec 9>"$lock_file"
if ! flock -n 9; then
    printf '[%s] Another monitor instance is already running.\n' "$(date '+%F %T')" >&2
    exit 0
fi
printf '%s\n' "$$" >"$pid_file"
trap 'rm -f "$pid_file"' EXIT INT TERM

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$monitor_log"
}

validate_settings() {
    if ! [[ "$check_interval" =~ ^[1-9][0-9]*$ ]]; then
        log "ERROR: TRAIN_MONITOR_INTERVAL_SECONDS must be a positive integer."
        exit 2
    fi
    if ! [[ "$target_step" =~ ^[1-9][0-9]*$ ]]; then
        log "ERROR: TRAIN_MONITOR_TARGET_STEP must be a positive integer."
        exit 2
    fi
    if [[ ! -f "$run_dir/.hydra/config.yaml" ]]; then
        log "ERROR: missing saved config: $run_dir/.hydra/config.yaml"
        exit 2
    fi
}

checkpoint_step() {
    local resolved basename
    [[ -e "$checkpoint" ]] || return 1
    resolved="$(readlink -f "$checkpoint")" || return 1
    basename="${resolved##*/}"
    if [[ "$basename" =~ optimizer_step=0*([0-9]+)\.ckpt$ ]]; then
        printf '%s\n' "$((10#${BASH_REMATCH[1]}))"
        return 0
    fi
    return 1
}

checkpoint_signature() {
    local resolved
    [[ -e "$checkpoint" ]] || {
        printf 'missing\n'
        return
    }
    resolved="$(readlink -f "$checkpoint")"
    stat -Lc '%n|%s|%Y|%i' "$resolved"
}

training_complete() {
    local step
    step="$(checkpoint_step 2>/dev/null)" || return 1
    (( step >= target_step ))
}

resume_training() {
    local step
    step="$(checkpoint_step 2>/dev/null || printf 'unknown')"
    log "Checkpoint is unchanged (step=$step); resuming training in W&B offline mode."
    if (
        cd "$repo_dir"
        export WANDB_MODE=offline
        export WANDB_DIR="$run_dir/wandb"
        python train.py \
            --config-dir="$run_dir/.hydra" \
            --config-name=config.yaml \
            training.resume=true \
            logging.mode=offline \
            "hydra.run.dir=$run_dir" \
            hydra.output_subdir=null
    ) >>"$training_log" 2>&1; then
        local status=0
    else
        local status=$?
    fi
    log "Resume command exited with status $status; monitoring continues."
    return 0
}

validate_settings
log "Monitor started: checkpoint=$checkpoint interval=${check_interval}s target_step=$target_step"

while true; do
    if training_complete; then
        log "Training reached target checkpoint step $(checkpoint_step); no resume is needed. Monitor exiting."
        exit 0
    fi

    before="$(checkpoint_signature)"
    log "Current checkpoint signature: $before"
    sleep "$check_interval"

    if training_complete; then
        log "Training reached target checkpoint step $(checkpoint_step); no resume is needed. Monitor exiting."
        exit 0
    fi

    after="$(checkpoint_signature)"
    if [[ "$after" == "$before" ]]; then
        resume_training
    else
        log "Checkpoint changed; training is considered healthy. New signature: $after"
    fi
done
