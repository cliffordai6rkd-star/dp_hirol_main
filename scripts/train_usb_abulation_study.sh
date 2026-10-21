#!/usr/bin/env bash
# Run only after the three experiment configs have been reviewed.
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"
log_dir=""
dry_run=false
train_overrides=()

usage() {
    cat <<'EOF'
Usage: bash scripts/train_usb_abulation_study.sh [options] [-- Hydra overrides...]
  --python PATH    Python from the training environment (default: PYTHON_BIN or python).
  --log-dir PATH  Directory for each experiment's console log and exit summary.
  --dry-run       Print the three commands without starting training or creating logs.
  -h, --help      Show this help.

Order: image-query CA + LSTM, image-only SA, joint image/wrench SA + LSTM.
Jobs run serially; a failed job stops the sequence. Each config uses batch 64,
200000 optimizer steps, and validation-action-MSE Top-K=2 checkpoints.
EOF
}

while (($#)); do
    case "$1" in
        --python|--log-dir)
            (($# >= 2)) || { usage >&2; exit 2; }
            case "$1" in
                --python) python_bin="$2" ;;
                --log-dir) log_dir="$2" ;;
            esac
            shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --) shift; train_overrides=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
python_bin="$(command -v -- "$python_bin")" || { echo 'Training Python not found; activate fdp or use --python.' >&2; exit 2; }

configs=(
    train_dp_image_query_lstm_insert_usb
    train_dp_image_selfattn_insert_usb
    train_dp_image_wrench_selfattn_insert_usb
)

if "$dry_run"; then
    for config in "${configs[@]}"; do
        printf '%q ' "$python_bin" -u train.py --config-name "abulation_study/$config" "${train_overrides[@]}"
        printf '\n'
    done
    exit 0
fi

# Match the existing training launcher when pip installed CUDA NPP libraries
# outside the system linker path (needed by some torchcodec installations).
npp_lib="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/npp/lib")')"
if [[ -d "$npp_lib" ]]; then
    export LD_LIBRARY_PATH="$npp_lib:${LD_LIBRARY_PATH:-}"
fi

if [[ -z "$log_dir" ]]; then
    log_dir="data/outputs/abulation_study/$(date +%Y%m%d_%H%M%S)_$$"
fi
mkdir -p "$log_dir"
log_dir="$(cd -- "$log_dir" && pwd)"
printf 'config\texit_code\n' > "$log_dir/exit_codes.tsv"

for config in "${configs[@]}"; do
    command=("$python_bin" -u train.py --config-name "abulation_study/$config" "${train_overrides[@]}")
    printf '[%s] Starting %s; log: %s/%s.log\n' "$(date '+%F %T')" "$config" "$log_dir" "$config"
    if "${command[@]}" 2>&1 | tee "$log_dir/$config.log"; then
        status=0
    else
        status=$?
    fi
    printf '%s\t%s\n' "$config" "$status" >> "$log_dir/exit_codes.tsv"
    if ((status != 0)); then
        printf 'Experiment failed (exit %s); stopping. See %s/%s.log\n' "$status" "$log_dir" "$config" >&2
        exit "$status"
    fi
done
printf 'All three experiments completed. Console logs: %s\n' "$log_dir"
