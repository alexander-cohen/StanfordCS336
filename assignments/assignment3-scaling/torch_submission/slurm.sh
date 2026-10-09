#!/bin/bash
set -euo pipefail
run_dir=$1
state_dir=$2
experiment_id=$3
python_bin=$4
mkdir -p "${SLURM_TMPDIR:-/tmp}/cs336-a3-${SLURM_JOB_ID}"
export TMPDIR="${SLURM_TMPDIR:-/tmp}/cs336-a3-${SLURM_JOB_ID}"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export JAX_PLATFORMS=cuda
export PYTHONUNBUFFERED=1
cd "$run_dir/source"
exec "$python_bin" -m torch_submission.worker "$state_dir" "$experiment_id"
