#!/bin/bash
#SBATCH --account=torch_pr_1323_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=cs336-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

# Submit from assignment1-basics:
# sbatch scripts/sweep.slurm ENTITY/PROJECT/SWEEP_ID
# Each submission runs one trial on its own GPU allocation.
set -euo pipefail

if (( $# != 1 )) || [[ -z "${1:-}" ]]; then
    echo "Usage: sbatch scripts/sweep.slurm ENTITY/PROJECT/SWEEP_ID" >&2
    exit 1
fi
sweep_id="$1"

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

# Avoid stale per-task settings inherited from an interactive allocation.
# srun uses the CPU request of this batch job explicitly below.
unset SLURM_TRES_PER_TASK

echo "Slurm job: $SLURM_JOB_ID | Sweep: $sweep_id | Trials: 1"
# The registered sweep supplies the training command and parameter values.
# Authenticate with 'uv run wandb login' before submitting.
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    uv run --no-sync wandb agent --count 1 "$sweep_id"
