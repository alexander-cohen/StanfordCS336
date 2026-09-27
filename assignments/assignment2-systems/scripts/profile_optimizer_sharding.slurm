#!/bin/bash
#SBATCH --job-name=optimizer_sharding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --partition=h200_public
#SBATCH --output=optimizer_sharding-%j.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the assignment directory}"
output_dir="results/optimizer_sharding_${SLURM_JOB_ID}"
mkdir -p "$output_dir"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export OMP_NUM_THREADS=1

srun uv run python -u scripts/profile_optimizer_sharding.py \
    --model_size xl --world_size 2 --batch_size_per_dev 8 \
    --warmup 5 --iter 20 "$@" --output_dir "$output_dir" \
    2>&1 | tee "$output_dir/run.txt"
