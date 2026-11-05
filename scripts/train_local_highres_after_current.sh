#!/usr/bin/env bash
# Queue the local high-resolution teacher -> RenderNet experiment behind the
# existing distillation run.
set -Eeuo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"
if [[ -z "${PYTHON_BIN:-}" && -x /home/lcy/miniconda3/envs/fdp/bin/python ]]; then
    python_bin=/home/lcy/miniconda3/envs/fdp/bin/python
fi
current_run="data/outputs/2026.11.04/21.49.27_081620_dp_distill_pipeline"
poll_seconds=30
output_dir=""
gpus="0"
log_dir=""
dry_run=false
dp_overrides=()
distill_overrides=()
usage() {
    cat <<'USAGE'
Usage: bash scripts/train_local_highres_after_current.sh [options]
  --current-run PATH             Existing pipeline directory
  --poll-seconds N               Status polling interval (default: 30)
  --python PATH                  Python interpreter
  --gpus LIST                    CUDA_VISIBLE_DEVICES (default: 0)
  --output-dir PATH              New pipeline output directory
  --log-dir PATH                 Queue log/status directory
  --dp-override KEY=VALUE        Teacher Hydra override (repeatable)
  --distill-override KEY=VALUE   Student Hydra override (repeatable)
  --dry-run                      Print the command without starting training
  -h, --help                     Show this help
USAGE
}
while (($#)); do
    case "$1" in
        --current-run|--poll-seconds|--python|--gpus|--output-dir|--log-dir|--dp-override|--distill-override)
            (($# >= 2)) || { usage >&2; exit 2; }
            case "$1" in
                --current-run) current_run="$2" ;;
                --poll-seconds) poll_seconds="$2" ;;
                --python) python_bin="$2" ;;
                --gpus) gpus="$2" ;;
                --output-dir) output_dir="$2" ;;
                --log-dir) log_dir="$2" ;;
                --dp-override) dp_overrides+=("$2") ;;
                --distill-override) distill_overrides+=("$2") ;;
            esac
            shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
python_bin="$(command -v -- "$python_bin")" || { echo 'Python interpreter not found.' >&2; exit 2; }
[[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || { echo '--poll-seconds must be a positive integer.' >&2; exit 2; }
# TorchCodec wheels may place CUDA NPP beside the Python environment rather
# than in the system linker path.  Keep the queue self-contained like the
# existing training launchers.
npp_lib="$($python_bin -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/npp/lib")')"
if [[ -d "$npp_lib" ]]; then
    export LD_LIBRARY_PATH="$npp_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
current_run="$(cd -- "$current_run" 2>/dev/null && pwd)" || { echo "Current run does not exist: $current_run" >&2; exit 2; }
summary="$current_run/student/training_summary.json"
pipeline="$current_run/pipeline.json"
[[ -n "$output_dir" ]] || output_dir="data/outputs/$(date +%Y.%m.%d)/$(date +%H.%M.%S)_local_highres_teacher_distill"
[[ -n "$log_dir" ]] || log_dir="data/outputs/local_highres_queue/$(date +%Y%m%d_%H%M%S)_$$"
if [[ -e "$output_dir" ]]; then
    echo "Output directory already exists: $output_dir" >&2
    exit 2
fi
mkdir -p "$log_dir"
log_dir="$(cd -- "$log_dir" && pwd)"
status_file="$log_dir/status.txt"
queue_log="$log_dir/queue.log"
printf '%s\n' "$$" > "$log_dir/queue.pid"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$queue_log"; }
terminal_status() {
    "$python_bin" - "$summary" "$pipeline" <<'PY'
import json, sys
from pathlib import Path
terminal = {"completed", "early_stopped", "failed", "interrupted", "stopped"}
for path, label in ((Path(sys.argv[1]), "summary"), (Path(sys.argv[2]), "pipeline")):
    try:
        status = str(json.loads(path.read_text()).get("status", "")).lower()
    except (OSError, ValueError):
        continue
    if status in terminal:
        print(f"{label}:{status}")
        raise SystemExit
print("running")
PY
}
cmd=("$python_bin" -u train_pipeline.py all
    --dp-config diffusion_policy/config/sim_force_aware_dp_train_20hz_lstm2_gripper_local_highres.yaml
    --distill-config diffusion_policy/config/train_distill_render_net_local_highres.yaml
    --teacher-selection latest --gpus "$gpus" --output-dir "$output_dir")
for override in "${dp_overrides[@]}"; do cmd+=(--dp-override "$override"); done
for override in "${distill_overrides[@]}"; do cmd+=(--distill-override "$override"); done
printf '%q ' "${cmd[@]}" > "$log_dir/command.sh"
printf '\n' >> "$log_dir/command.sh"
if "$dry_run"; then
    printf 'Current run: %s\nCurrent status: %s\nOutput: %s\nCommand: ' \
        "$current_run" "$(terminal_status)" "$output_dir"
    printf '%q ' "${cmd[@]}"
    printf '\nQueue log: %s\n' "$queue_log"
    exit 0
fi
log "Queue PID=$$; waiting for current run: $current_run"
printf 'waiting\n' > "$status_file"
while true; do
    state="$(terminal_status)"
    [[ "$state" != running ]] && break
    sleep "$poll_seconds"
done
log "Current run reached terminal state: $state"
case "$state" in
    summary:failed|summary:interrupted|pipeline:failed|pipeline:interrupted)
        printf 'blocked_by_%s\n' "$state" > "$status_file"
        log 'Current run did not complete successfully; queued experiment was not started.'
        exit 1 ;;
esac
log "Starting local high-resolution teacher + distillation pipeline: $output_dir"
printf 'running\n' > "$status_file"
if "${cmd[@]}" 2>&1 | tee -a "$queue_log"; then
    printf 'completed\n' > "$status_file"
    log 'Queued pipeline completed.'
else
    code=$?
    printf 'failed:%s\n' "$code" > "$status_file"
    log "Queued pipeline failed with exit code $code."
    exit "$code"
fi
