#!/bin/bash
set -euo pipefail
state_dir=$1
repo_dir=$(cd "$(dirname "$0")/.." && pwd)
uv_bin=$(command -v uv || true)
if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
    uv_bin="$HOME/.local/bin/uv"
fi
if [[ -z "$uv_bin" ]]; then
    echo 'Install uv on Torch first: https://docs.astral.sh/uv/getting-started/installation/' >&2
    exit 1
fi
export UV_CACHE_DIR="/scratch/$USER/uv-cache"
export UV_PYTHON_INSTALL_DIR="/scratch/$USER/uv-python"
mkdir -p "$state_dir"
if [[ ! -x "$state_dir/venv/bin/python" ]]; then
    "$uv_bin" venv --python 3.12 "$state_dir/venv"
fi
"$uv_bin" pip install --python "$state_dir/venv/bin/python" -r "$repo_dir/torch_submission/requirements.txt"
# flash-hog's Linux metadata insists on CUDA 13. Its pure Python code also works
# with this trainer's JAX 0.9.1 CUDA 12 runtime; avoid installing both CUDA stacks.
"$uv_bin" pip install --python "$state_dir/venv/bin/python" --no-deps flash-hog==0.5.0
cd "$repo_dir"
JAX_PLATFORMS=cpu "$state_dir/venv/bin/python" -c 'from cs336_scaling.training.driver import run_training_driver; print("Trainer dependencies ready")'
"$state_dir/venv/bin/python" -m torch_submission.server --root "$state_dir" budget
