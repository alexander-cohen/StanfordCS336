#!/bin/bash
#SBATCH --job-name=distributed_benchmark
#SBATCH --array=0-11
#SBATCH --nodes=1 
#SBATCH --gres=gpu:6
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=distributed_benchmark-%A_%a.txt
#SBATCH --partition=h200_public

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"


data_sizes_mb=(1 10 100 1000)
world_sizes=(2 4 6)

task_id="${SLURM_ARRAY_TASK_ID}"
data_size_mb="${data_sizes_mb[$((task_id / 3))]}"
world_size="${world_sizes[$((task_id % 3))]}"

output="results/benchmark_allreduce_${data_size_mb}MB_${world_size}gpu_${SLURM_ARRAY_JOB_ID}_${task_id}.txt"


srun uv run python scripts/benchmark_distributed.py --data_size_mb "$data_size_mb" --world_size "$world_size" > "$output" 2>&1