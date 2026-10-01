#!/bin/bash
#SBATCH --job-name=test_fsdp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --partition=h200_public
#SBATCH --output=test_fsdp-%j.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the assignment directory}"
export OMP_NUM_THREADS=1
srun uv run python -u scripts/test_fsdp.py --device cuda --world-size 2 "$@"
