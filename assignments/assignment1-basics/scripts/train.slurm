#!/bin/bash
#SBATCH --job-name=cs336
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

# Submit from assignment1-basics:
# sbatch --account=YOUR_SLURM_ACCOUNT scripts/train.slurm
# Extra arguments are forwarded to scripts.run_experiment, e.g.:
# sbatch --account=YOUR_SLURM_ACCOUNT scripts/train.slurm --config configs/smoke.toml --stop-after 4
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from assignment1-basics}"
if [[ ! -f scripts/run_experiment.py || ! -f pyproject.toml ]]; then
    echo "Submit from the assignment1-basics directory." >&2
    exit 1
fi
: "${SCRATCH:?Torch must provide SCRATCH, or set it before submission}"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/uv-cache}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

if [[ ! -x .venv/bin/python ]]; then
    echo "Create the cluster environment first: uv sync --locked" >&2
    exit 1
fi

# Install dependencies and run 'uv run wandb login' before submitting.
# --no-sync uses that prepared environment without installing during the job.
srun uv run --no-sync python -m scripts.run_experiment \
    --config configs/small_cosinelr.toml "$@" --device cuda
