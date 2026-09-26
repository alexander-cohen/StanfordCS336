#!/bin/bash
set -euo pipefail

# Submit from the assignment directory, even when invoked elsewhere.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Each array has four data sizes and one fixed GPU count.
for world_size in 2 4 6; do
    sbatch --gres="gpu:${world_size}" --cpus-per-task="$world_size" \
        scripts/run_distributed_benchmark.slurm "$world_size"
done
