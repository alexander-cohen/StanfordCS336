#!/bin/bash
#SBATCH --job-name=compare_ddp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --partition=h200_public
#SBATCH --output=ddp_compare-%j.txt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the assignment directory}"


model_size="${1:-xl}"
if (( $# > 0 )); then shift; fi
output_dir="results/fsdp_compare_${SLURM_JOB_ID}"
mkdir -p "$output_dir"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export OMP_NUM_THREADS=1

srun uv run python -u scripts/benchmark_distributed.py \
    --benchmark fsdp --model_size "$model_size" \
    --world_size 2 --batch_size_per_dev 4 --seed 42 --warmup 5 --iter 20 \
    "$@" --ddp_variant "$variant" \
    --result_json "$output_dir/$variant.json" \
    --trace_dir "$output_dir/traces/$variant" \
    2>&1 | tee "$output_dir/$variant.txt"