#!/usr/bin/env bash
# Run with RoboTwin's source imports and asset-relative cwd configured.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -gt 0 ]] || { echo 'Usage: with_robotwin.sh python -m <module> [args...]' >&2; exit 2; }
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src/RoboTwin:${REPO_ROOT}/src/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
cd "${REPO_ROOT}/src/RoboTwin"
exec "$@"
