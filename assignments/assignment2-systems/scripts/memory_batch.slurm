#!/bin/bash
#SBATCH --job-name=memory-large-forward
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=memory-%A_%a.log
#SBATCH --partition=l40s_public

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
output_dir="profiles/memory_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$output_dir"

task_id="${SLURM_ARRAY_TASK_ID}"
case "$task_id" in
    [0-5]) ;;
    *)
        echo "Expected array task 0 through 5" >&2
        exit 1
        ;;
esac

context_lengths=(128 512 1024)
context_length="${context_lengths[$((task_id / 2))]}"

benchmark_args=(
    --model_size large
    --mode forward
    --device cuda
    --batch_size 4
    --context_length "$context_length"
    --warmup 5
    --bench_iter 10
    --profile_memory
)

case "$((task_id % 2))" in
    0) precision=fp32 ;;
    1)
        precision=bf16
        benchmark_args+=(--mixed_precision)
        ;;
esac

output="${output_dir}/benchmark_large_forward_ctx${context_length}_${precision}_${task_id}"
echo "Writing memory profiling output to ${output}.log"
exec >"${output}.log" 2>&1

srun uv run python -u scripts/benchmarking_new.py \
    "${benchmark_args[@]}" \
    --out_dir "$output"
