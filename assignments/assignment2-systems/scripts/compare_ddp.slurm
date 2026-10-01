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

model_size="${1:-large}"
if (( $# > 0 )); then shift; fi
output_dir="results/ddp_compare_${SLURM_JOB_ID}"
mkdir -p "$output_dir"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export OMP_NUM_THREADS=1

# One allocation, three fresh runs: identical GPUs, model seed, and local data.
# Additional arguments (e.g. --profile or --iter 50) apply to every variant.
for variant in naive overlap flat; do
    srun uv run python -u scripts/benchmark_distributed.py \
        --benchmark ddp_training --model_size "$model_size" \
        --world_size 2 --profile --batch_size_per_dev 8 --seed 42 --warmup 5 --iter 20 \
        "$@" --ddp_variant "$variant" \
        --result_json "$output_dir/$variant.json" \
        --trace_dir "$output_dir/traces/$variant" \
        2>&1 | tee "$output_dir/$variant.txt"
done

uv run python - "$output_dir" <<'PY' | tee "$output_dir/comparison.txt"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads((root / f"{name}.json").read_text()) for name in ("naive", "overlap", "flat")]
baseline = rows[0]["mean_step_ms"]
print(f"{'Variant':<12} {'Step (ms)':>12} {'Std (ms)':>12} {'Speedup vs naive':>18}")
for row in rows:
    print(f"{row['variant']:<12} {row['mean_step_ms']:>12.3f} {row['std_step_ms']:>12.3f} "
          f"{baseline / row['mean_step_ms']:>17.2f}x")
PY
