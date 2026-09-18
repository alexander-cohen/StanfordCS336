#!/bin/bash
#SBATCH --job-name=profile
#SBATCH --array=0-8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=profile-%A_%a.log
#SBATCH --partition=l40s_public

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
output_dir="profiles/${SLURM_ARRAY_JOB_ID}"
mkdir -p "$output_dir"

models=(small medium large)
modes=(forward backward train)

task_id="${SLURM_ARRAY_TASK_ID}"
model_size="${models[$((task_id / 3))]}"
mode="${modes[$((task_id % 3))]}"

output="${output_dir}/benchmark_${model_size}_${mode}_${task_id}"
echo "Writing profiling output to ${output}.log"
exec >"${output}.log" 2>&1

export NSYS_NVTX_PROFILER_REGISTER_ONLY=0

srun nsys profile \
    --trace=cuda,cudnn,cublas,osrt,nvtx \
    --pytorch=functions-trace,autograd-shapes-nvtx \
    --cudabacktrace=all \
    --python-backtrace=cuda \
    --capture-range=nvtx \
    --capture-range-end=stop \
    --nvtx-capture='measurement' \
    --output="$output" \
    uv run python -u scripts/benchmarking_new.py \
    --model_size "$model_size" \
    --mode "$mode" \
    --warmup 5 \
    --bench_iter 10
