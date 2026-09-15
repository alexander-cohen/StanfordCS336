#!/bin/bash
srun --ntasks=1 --gres=gpu:1 --time=00:30:00 uv run python -u "$@"