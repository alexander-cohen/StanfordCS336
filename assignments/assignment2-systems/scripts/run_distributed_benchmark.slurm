#!/bin/bash
#SBATCH --job-name=distributed_benchmark
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=distributed_benchmark-%A_%a.txt
#SBATCH --partition=h200_public

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"

# The submission helper requests this many GPUs and passes the same count here.
world_size="${1:?Use bash scripts/submit_distributed_benchmark.sh}"
case "$world_size" in
    2|4|6) ;;
    *) echo "Expected world_size 2, 4, or 6" >&2; exit 1 ;;
esac

data_sizes_mb=(1 10 100 1000)

task_id="${SLURM_ARRAY_TASK_ID}"
data_size_mb="${data_sizes_mb[$task_id]}"

mkdir -p results
output="results/benchmark_allreduce_${data_size_mb}MB_${world_size}gpu_${SLURM_ARRAY_JOB_ID}_${task_id}.txt"

# Reduce rendezvous-port collisions when separate jobs share a node.
export MASTER_PORT=$((20000 + (SLURM_ARRAY_JOB_ID * 4 + task_id) % 40000))

srun uv run python -u scripts/benchmark_distributed.py \
    --data_size_mb "$data_size_mb" --world_size "$world_size" > "$output" 2>&1
