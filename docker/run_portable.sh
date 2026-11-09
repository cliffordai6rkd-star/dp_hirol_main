#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_dir"
export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"
mode="${1:-check}"
if (( $# )); then shift; fi
compose=(docker compose -p dp-portable -f docker/portable_compose.yaml)
case "$mode" in
  dev)
    mkdir -p .cache/container-home .cache/vscode-server
    dev_id="$("${compose[@]}" run --rm -d --name "${DEV_CONTAINER_NAME:-fdp-dev}" \
      -e HOME=/workspace/.cache/container-home \
      -e VSCODE_AGENT_FOLDER=/workspace/.cache/vscode-server trainer sleep infinity)"
    # VS Code needs a passwd entry and a writable home for the host UID.
    docker exec --user 0 "$dev_id" bash -c '
      if ! getent group "$2" >/dev/null; then
        groupadd --gid "$2" fdp-dev-group
      fi
      if ! getent passwd "$1" >/dev/null; then
        useradd --uid "$1" --gid "$2" --no-create-home \
          --home-dir /workspace/.cache/container-home --shell /bin/bash fdp-dev-user
      fi
    ' -- "$LOCAL_UID" "$LOCAL_GID"
    echo "VS Code 开发容器已启动: ${DEV_CONTAINER_NAME:-fdp-dev} ($dev_id)"
    ;;
  shell) exec "${compose[@]}" run --rm trainer bash "$@" ;;
  check|all|dp|distill)
    exec "${compose[@]}" run --rm trainer bash scripts/train_portable.sh "$mode" "$@"
    ;;
  *) echo "用法: bash docker/run_portable.sh {check|all|dp|distill|shell|dev} [pipeline 参数]" >&2; exit 2 ;;
esac
