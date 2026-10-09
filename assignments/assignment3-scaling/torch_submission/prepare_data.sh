#!/bin/bash
set -euo pipefail
repo_dir=$1
state_dir=$2
train_tokens=$3
export TMPDIR="${SLURM_TMPDIR:-/tmp}/cs336-a3-data-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export HF_HOME="$state_dir/hf-cache"
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
cd "$repo_dir"
exec "$state_dir/venv/bin/python" -m torch_submission.prepare_data "$state_dir" --train-tokens "$train_tokens"
