#!/usr/bin/env bash
# Run Codex through a local Steady Relay without changing the global config.
# The relay holds the HTTP request and retries temporary model-capacity errors,
# so one Codex turn/session remains open instead of being restarted.
set -Eeuo pipefail

codex_bin="${CODEX_BIN:-codex}"
relay_bin="${STEADY_RELAY_BIN:-$HOME/.local/share/steady-relay/2.3.0/linux-amd64/steady-relay-2.3.0-linux-amd64/steady-relay}"
listen="${STEADY_RELAY_LISTEN:-127.0.0.1:18080}"
health_url="${STEADY_RELAY_HEALTH_URL:-http://127.0.0.1:18080/healthz}"
upstream="${STEADY_RELAY_UPSTREAM:-https://api.openai.com/v1}"
max_retries="${STEADY_RELAY_MAX_RETRIES:-10}"
retry_backoff="${STEADY_RELAY_RETRY_BACKOFF:-500ms}"
max_retry_after="${STEADY_RELAY_MAX_RETRY_AFTER:-1m}"
idle_timeout="${STEADY_RELAY_IDLE_TIMEOUT:-2m}"
log_dir="${STEADY_RELAY_LOG_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/codex-steady-relay}"
relay_base_url="http://${listen}/v1"

usage() {
    cat <<'USAGE'
Usage: codex-steady-relay [codex options]

Starts a loopback Steady Relay and launches Codex through it. Codex keeps its
existing ChatGPT login and selected model. The relay retries temporary capacity
errors before returning a failure to Codex.

Environment overrides:
  STEADY_RELAY_LISTEN=127.0.0.1:18080
  STEADY_RELAY_UPSTREAM=https://api.openai.com/v1
  STEADY_RELAY_MAX_RETRIES=10
  STEADY_RELAY_RETRY_BACKOFF=500ms
  STEADY_RELAY_MAX_RETRY_AFTER=1m
  STEADY_RELAY_IDLE_TIMEOUT=2m
  STEADY_RELAY_LOG_DIR=~/.local/state/codex-steady-relay

Examples:
  codex-steady-relay
  codex-steady-relay -m gpt-6.1-terra
  codex-steady-relay resume --last

The existing desktop Codex process must be restarted through this launcher for
its requests to use the relay. The launcher does not modify ~/.codex/config.toml.
USAGE
}

if (($#)) && [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi
command -v -- "$codex_bin" >/dev/null 2>&1 || {
    echo "codex-steady-relay: Codex executable not found: $codex_bin" >&2
    exit 2
}
[[ -x "$relay_bin" ]] || {
    echo "codex-steady-relay: Steady Relay executable not found or not executable: $relay_bin" >&2
    exit 2
}
command -v curl >/dev/null 2>&1 || {
    echo 'codex-steady-relay: curl is required for the local health check.' >&2
    exit 2
}
mkdir -p -- "$log_dir"
relay_log="$log_dir/relay-$(date +%Y%m%d_%H%M%S)-$$.log"

relay_pid=''
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$relay_pid" ]] && kill -0 "$relay_pid" 2>/dev/null; then
        kill -TERM "$relay_pid" 2>/dev/null || true
        wait "$relay_pid" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

printf '[%s] Starting Steady Relay at %s -> %s\n' "$(date '+%F %T')" "$listen" "$upstream" >> "$relay_log"
"$relay_bin" \
    -listen "$listen" \
    -upstream "$upstream" \
    -max-retries "$max_retries" \
    -retry-backoff "$retry_backoff" \
    -max-retry-after "$max_retry_after" \
    -upstream-idle-timeout "$idle_timeout" \
    -buffer-until-success \
    >>"$relay_log" 2>&1 &
relay_pid=$!

healthy=false
for _ in $(seq 1 60); do
    if curl -fsS --max-time 2 "$health_url" >/dev/null 2>&1; then
        healthy=true
        break
    fi
    if ! kill -0 "$relay_pid" 2>/dev/null; then
        echo "codex-steady-relay: Steady Relay exited during startup; see $relay_log" >&2
        exit 1
    fi
    sleep 0.1
done
if [[ "$healthy" != true ]]; then
    echo "codex-steady-relay: local health check failed at $health_url; see $relay_log" >&2
    exit 1
fi

# Custom provider settings keep the current ChatGPT OAuth authentication while
# routing Responses API traffic through the loopback relay. WebSockets are
# disabled so every request is visible to the HTTP retry proxy.
provider_args=(
    -c 'model_provider="steady_relay_chatgpt"'
    -c 'model_providers.steady_relay_chatgpt.name="ChatGPT plan via Steady Relay"'
    -c "model_providers.steady_relay_chatgpt.base_url=\"${relay_base_url}\""
    -c 'model_providers.steady_relay_chatgpt.wire_api="responses"'
    -c 'model_providers.steady_relay_chatgpt.requires_openai_auth=true'
    -c 'model_providers.steady_relay_chatgpt.supports_websockets=false'
)

printf '[%s] Relay healthy; launching Codex (pid=%s)\n' "$(date '+%F %T')" "$relay_pid" >> "$relay_log"
"$codex_bin" "${provider_args[@]}" "$@"
