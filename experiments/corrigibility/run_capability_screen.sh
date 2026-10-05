#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/paisi-rfm-numba}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
python_bin="${PAISI_PYTHON:-/home/aarav/miniforge3/envs/paisi-rfm/bin/python}"
mkdir -p results/corrigibility/capability_screen/logs
exec "$python_bin" experiments/corrigibility/run_capability_screen.py "$@"
