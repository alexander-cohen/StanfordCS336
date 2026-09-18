#!/bin/bash
#SBATCH --job-name=nsight-block-memory
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=nsight-memory-%A_%a.log
#SBATCH --partition=h200_public

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
output_dir="profiles/nsight_memory_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$output_dir"

task_id="${SLURM_ARRAY_TASK_ID}"
case "$task_id" in
    [0-5]) ;;
    *) echo "Expected array task 0 through 5" >&2; exit 1 ;;
esac

context_lengths=(128 512 1024)
context_length="${context_lengths[$((task_id / 2))]}"
benchmark_args=(
    --model_size large
    --mode backward
    --device cuda
    --batch_size 4
    --context_length "$context_length"
    --warmup 1
    --bench_iter 1
    --nvtx_blocks
)

precision=fp32
if (( task_id % 2 )); then
    precision=bf16
    benchmark_args+=(--mixed_precision)
fi

output="${output_dir}/large_ctx${context_length}_${precision}_${task_id}"
echo "Writing Nsight memory profiling output to ${output}.log"
exec >"${output}.log" 2>&1

# Expose tensor allocation/free activity to CUDA tracing. This changes timings.
# Do not combine this run with the caching-allocator snapshot experiment.
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Capture setup too, so the graph includes the model's allocation baseline.
srun nsys profile \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --pytorch=autograd-shapes-nvtx \
    --output="$output" \
    uv run python -u scripts/benchmarking_new.py \
    "${benchmark_args[@]}"
