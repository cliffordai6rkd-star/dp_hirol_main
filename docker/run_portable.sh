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
  shell) exec "${compose[@]}" run --rm trainer bash "$@" ;;
  check|all|dp|distill)
    exec "${compose[@]}" run --rm trainer bash scripts/train_portable.sh "$mode" "$@"
    ;;
  *) echo "用法: bash docker/run_portable.sh {check|all|dp|distill|shell} [pipeline 参数]" >&2; exit 2 ;;
esac
