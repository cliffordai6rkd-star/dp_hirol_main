#!/usr/bin/env bash
# Wait for the existing USB training, then run the two token baselines in order.
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"
wait_pid=auto
poll_seconds=30
log_dir=""
dry_run=false
train_overrides=()

usage() {
    cat <<'EOF'
Usage: bash scripts/train_usb_baselines_after_current.sh [options] [-- Hydra overrides...]
  --wait-pid PID        Wait for this process (default: detect the USB training parent).
  --python PATH         Python interpreter (default: PYTHON_BIN or python).
  --poll-seconds N      Check interval (default: 30 seconds).
  --log-dir PATH        Queue status and per-training console logs.
  --dry-run             Show the selected process and commands without training.
  -h, --help            Show this help.

Both jobs inherit train_dp_insert_usb_force_aware settings, use fresh runs,
and execute sequentially. Each job has its own Hydra output and WandB name.
EOF
}

while (($#)); do
    case "$1" in
        --wait-pid|--python|--poll-seconds|--log-dir)
            (($# >= 2)) || { usage >&2; exit 2; }
            case "$1" in
                --wait-pid) wait_pid="$2" ;;
                --python) python_bin="$2" ;;
                --poll-seconds) poll_seconds="$2" ;;
                --log-dir) log_dir="$2" ;;
            esac
            shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --) shift; train_overrides=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
python_bin="$(command -v -- "$python_bin")" || { echo 'Python interpreter not found.' >&2; exit 2; }
[[ "$poll_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo 'Invalid poll interval.' >&2; exit 2; }
"$python_bin" -c 'import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)' "$poll_seconds"

if [[ "$wait_pid" == auto ]]; then
    wait_pid="$("$python_bin" - "$repo_root" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
matches = {}
for entry in Path('/proc').iterdir():
    if not entry.name.isdecimal():
        continue
    try:
        args = (entry / 'cmdline').read_bytes().decode().split('\0')
        if not any(Path(arg).name == 'train.py' for arg in args):
            continue
        target = 'train_dp_insert_usb_force_aware'
        selected = any(
            arg == '--config-name=' + target
            or (arg == '--config-name' and i + 1 < len(args) and args[i + 1] == target)
            for i, arg in enumerate(args)
        )
        if not selected or (entry / 'cwd').resolve() != root:
            continue
        stat = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] not in {'Z', 'X'}:
            matches[int(entry.name)] = int(stat[1])
    except (OSError, ValueError, UnicodeError):
        continue
parents = [pid for pid, ppid in matches.items() if ppid not in matches]
if len(parents) > 1:
    sys.exit('Multiple USB training parents found; specify --wait-pid explicitly.')
print(parents[0] if parents else 0)
PY
)"
fi
[[ "$wait_pid" =~ ^[0-9]+$ ]] || { echo 'Invalid wait PID.' >&2; exit 2; }

# Preserve relevant environment settings of the existing training without
# printing credentials. Never signal this process: it is only the wait target.
if [[ -r "/proc/$wait_pid/environ" ]]; then
    while IFS= read -r -d '' setting; do
        case "${setting%%=*}" in
            CUDA_VISIBLE_DEVICES|CUDA_DEVICE_ORDER|DP_DATASET_PATH|DINOV3_MODEL_PATH|LD_LIBRARY_PATH|WANDB_API_KEY|WANDB_BASE_URL|WANDB_MODE|WANDB_ENTITY|WANDB_PROJECT|HF_HOME|HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE|OMP_NUM_THREADS|MKL_NUM_THREADS|PYTORCH_CUDA_ALLOC_CONF|PYTORCH_ALLOC_CONF|CONDA_PREFIX)
                export "$setting" ;;
        esac
    done < "/proc/$wait_pid/environ" || true
fi
# pip CUDA wheels can install NPP outside the system dynamic-library path.
npp_lib="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/npp/lib")')"
if [[ -d "$npp_lib" ]]; then
    export LD_LIBRARY_PATH="$npp_lib:${LD_LIBRARY_PATH:-}"
fi

process_identity() {
    "$python_bin" - "$1" <<'PY'
import sys
from pathlib import Path
try:
    fields = Path('/proc', sys.argv[1], 'stat').read_text().rsplit(')', 1)[1].split()
    print(fields[19] if fields[0] not in {'Z', 'X'} else 'exited')
except (OSError, IndexError):
    print('exited')
PY
}
wait_identity="$(process_identity "$wait_pid")"
configs=(train_dp_dinobase_insert_usb train_dp_selfattn_insert_usb)
if "$dry_run"; then
    printf 'Wait PID: %s; process start ticks: %s\n' "$wait_pid" "$wait_identity"
    for config in "${configs[@]}"; do
        printf '%q ' "$python_bin" -u train.py --config-name "baseline_train/$config" training.resume=false "${train_overrides[@]}"
        printf '\n'
    done
    exit 0
fi

mkdir -p data/outputs/baseline_queue
exec 9>data/outputs/baseline_queue/usb_baselines.lock
flock -n 9 || { echo 'A USB baseline queue is already running.' >&2; exit 1; }
if [[ -z "$log_dir" ]]; then
    log_dir="data/outputs/baseline_queue/$(date +%Y%m%d_%H%M%S)_$$"
fi
mkdir -p "$log_dir"
log_dir="$(cd "$log_dir" && pwd)"
printf '%s\n' "$$" > "$log_dir/queue.pid"
printf 'config\texit_code\n' > "$log_dir/exit_codes.tsv"
child_pid=""
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
stop_queue() {
    trap - INT TERM
    if [[ -n "$child_pid" ]]; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    printf 'stopped\n' > "$log_dir/status.txt"
    log 'Queue stopped.'
    exit 130
}
trap stop_queue INT TERM
log "Queue PID=$$; waiting for PID=$wait_pid (start ticks=$wait_identity)."
log "Logs: $log_dir"
printf 'waiting pid=%s start_ticks=%s\n' "$wait_pid" "$wait_identity" > "$log_dir/status.txt"
# Start time prevents a reused PID from keeping the queue waiting forever.
while [[ "$wait_identity" != exited && "$(process_identity "$wait_pid")" == "$wait_identity" ]]; do
    sleep "$poll_seconds"
done
log 'Existing training has exited; starting the baseline sequence.'

queue_status=0
for config in "${configs[@]}"; do
    run_log="$log_dir/$config.log"
    log "Starting $config; console log: $run_log"
    "$python_bin" -u train.py --config-name "baseline_train/$config" \
        training.resume=false "${train_overrides[@]}" > "$run_log" 2>&1 &
    child_pid=$!
    printf '%s\n' "$child_pid" > "$log_dir/active_train.pid"
    printf 'running config=%s pid=%s\n' "$config" "$child_pid" > "$log_dir/status.txt"
    if wait "$child_pid"; then
        run_status=0
    else
        run_status=$?
        queue_status=1
    fi
    child_pid=""
    rm -f "$log_dir/active_train.pid"
    printf '%s\t%s\n' "$config" "$run_status" >> "$log_dir/exit_codes.tsv"
    log "Finished $config; exit code=$run_status."
done
if ((queue_status == 0)); then
    printf 'completed\n' > "$log_dir/status.txt"
else
    printf 'completed_with_errors\n' > "$log_dir/status.txt"
fi
log "Baseline sequence finished; exit summary: $log_dir/exit_codes.tsv"
exit "$queue_status"
