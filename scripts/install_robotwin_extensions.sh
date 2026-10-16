#!/usr/bin/env bash
# Run after `uv sync --locked --extra eval --extra robotwin` at the project root.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo 'Create the root uv environment first.' >&2; exit 1; }
[[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]] || {
  echo 'Set CUDA_HOME to the CUDA 12.8 toolkit directory before building.' >&2
  exit 1
}
[[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]] || {
  echo 'Set TORCH_CUDA_ARCH_LIST for your GPUs (for example 8.9 for RTX 4090).' >&2
  exit 1
}
export MAX_JOBS="${MAX_JOBS:-2}"
export FORCE_CUDA=1
# Build against root torch. --no-deps keeps root dependency versions intact.
uv pip install --python "$PYTHON" --no-build-isolation --no-deps \
  -e "$REPO_ROOT/src/pytorch3d"
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
  uv pip install --python "$PYTHON" --no-build-isolation --no-deps \
  -e "$REPO_ROOT/src/RoboTwin/envs/curobo"
"$PYTHON" -c 'import pytorch3d._C, curobo; print("RoboTwin native extensions installed")'
