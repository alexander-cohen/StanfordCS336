#!/bin/bash
  set -euo pipefail

  for config in configs/ablations/*.toml; do
      sbatch \
          --account=torch_pr_1323_general \
          --partition=l40s_public \
          scripts/train.slurm --config "$config"
  done