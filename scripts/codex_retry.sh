#!/usr/bin/env bash
# Retry Codex CLI requests when the selected model is temporarily unavailable.
# Authentication, model selection, and the ChatGPT account remain Codex's own.
set -uo pipefail

codex_bin="${CODEX_BIN:-codex}"
max_attempts="${CODEX_RETRY_MAX_ATTEMPTS:-8}"
base_delay="${CODEX_RETRY_BASE_DELAY:-5}"
max_delay="${CODEX_RETRY_MAX_DELAY:-120}"
log_dir="${CODEX_RETRY_LOG_DIR:-${TMPDIR:-/tmp}/codex-retry}"
verbose=false
codex_args=()

usage() {
    cat <<'USAGE'
Usage:
  codex-retry [wrapper options] -- [codex exec options] [prompt]
  printf '%s' 'prompt' | codex-retry [wrapper options] -- [codex exec options]

Wrapper options:
  --max-attempts N   Maximum total attempts (default: 8)
  --base-delay SEC   Initial backoff in seconds (default: 5)
  --max-delay SEC    Backoff cap in seconds (default: 120)
  --log-dir PATH     Save per-attempt JSONL logs (default: $TMPDIR/codex-retry)
  --verbose          Print retry diagnostics
  -h, --help         Show this help

Everything after -- is passed to `codex exec`. The wrapper adds --json so it
can identify transient capacity errors. It retries only before any assistant
or tool output is observed, avoiding duplicate side effects.
USAGE
}

fail() { printf 'codex-retry: %s\n' "$*" >&2; exit 2; }

while (($#)); do
    case "$1" in
        --max-attempts|--base-delay|--max-delay|--log-dir)
            (($# >= 2)) || fail "$1 requires a value"
            case "$1" in
                --max-attempts) max_attempts="$2" ;;
                --base-delay) base_delay="$2" ;;
                --max-delay) max_delay="$2" ;;
                --log-dir) log_dir="$2" ;;
            esac
            shift 2 ;;
        --verbose) verbose=true; shift ;;
        --) shift; codex_args=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) fail "wrapper options must precede -- (unexpected: $1)" ;;
    esac
done

[[ "$max_attempts" =~ ^[1-9][0-9]*$ ]] || fail '--max-attempts must be a positive integer'
[[ "$base_delay" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail '--base-delay must be nonnegative'
[[ "$max_delay" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail '--max-delay must be nonnegative'
command -v -- "$codex_bin" >/dev/null 2>&1 || fail "Codex executable not found: $codex_bin"
mkdir -p -- "$log_dir" || fail "cannot create log directory: $log_dir"

# Codex exec's JSONL events let us distinguish a capacity failure before any
# model/tool output from a failure after work has already begun.
is_transient_capacity() {
    local log_file="$1"
    grep -Eiq '(at[[:space:]_-]+capacity|model[[:space:]_-]+capacity|server(_is)?_overloaded|server[[:space:]]+overloaded|temporarily[[:space:]]+(overloaded|unavailable)|capacity[[:space:]]+(不足|不足以)|模型[^[:space:]]*(容量|繁忙|过载|不可用)|容量不足|服务繁忙|service_unavailable|HTTP[[:space:]]*(429|503)|(^|[^0-9])(429|503)([^0-9]|$))' "$log_file" || return 1
    # Never replay a request after Codex has emitted an assistant message,
    # command/tool event, or file change. The first attempt may have side effects.
    if grep -Eiq '"type"[[:space:]]*:[[:space:]]*"(item/|response\.output|assistant|tool|command|file_change|function_call)' "$log_file"; then
        return 1
    fi
    return 0
}

sleep_backoff() {
    local attempt="$1" delay jitter
    delay="$base_delay"
    if (( attempt > 1 )); then
        delay=$(awk -v b="$base_delay" -v n="$attempt" 'BEGIN { d=b * (2 ^ (n - 1)); print (d > 1e9 ? 1e9 : d) }')
    fi
    delay=$(awk -v d="$delay" -v m="$max_delay" 'BEGIN { print (d > m ? m : d) }')
    # A small integer jitter avoids synchronized retries without requiring a
    # non-portable fractional sleep implementation.
    jitter=$(( $(od -An -N2 -tu2 /dev/urandom 2>/dev/null || printf '0') % 3 ))
    delay=$(awk -v d="$delay" -v j="$jitter" 'BEGIN { print d + j }')
    if "$verbose"; then
        printf 'codex-retry: waiting %.1fs before attempt %d/%d\n' "$delay" "$((attempt + 1))" "$max_attempts" >&2
    fi
    sleep "$delay"
}

for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    attempt_log="$log_dir/attempt-$(date +%Y%m%d_%H%M%S)-${attempt}-$$.jsonl"
    if "$verbose"; then
        printf 'codex-retry: attempt %d/%d\n' "$attempt" "$max_attempts" >&2
    fi
    # Add --json unless the caller already supplied it. Keep stdout visible
    # while retaining a copy for classification and later diagnosis.
    has_json=false
    for arg in "${codex_args[@]}"; do [[ "$arg" == "--json" ]] && has_json=true; done
    if "$has_json"; then
        command=("$codex_bin" exec "${codex_args[@]}")
    else
        command=("$codex_bin" exec --json "${codex_args[@]}")
    fi
    set +e
    "${command[@]}" 2>&1 | tee "$attempt_log"
    status=${PIPESTATUS[0]}
    set -e
    if (( status == 0 )); then
        exit 0
    fi
    if (( attempt == max_attempts )) || ! is_transient_capacity "$attempt_log"; then
        if "$verbose"; then
            printf 'codex-retry: not retrying (exit=%d, log=%s)\n' "$status" "$attempt_log" >&2
        fi
        exit "$status"
    fi
    printf 'codex-retry: temporary model-capacity failure; retrying (%d/%d)\n' "$attempt" "$max_attempts" >&2
    sleep_backoff "$attempt"
done
