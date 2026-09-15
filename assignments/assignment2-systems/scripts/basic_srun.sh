#!/bin/bash
srun --ntasks=1 --gres=gpu:1 --mem=32G --time=00:30:00 uv run python -u "$@"