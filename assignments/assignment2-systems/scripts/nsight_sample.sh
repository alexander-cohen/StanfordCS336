  srun --gres=gpu:1 --mem=32G \
    nsys profile \
      --trace=cuda,cudnn,cublas,osrt,nvtx \
      --pytorch=functions-trace,autograd-shapes-nvtx \
      --cudabacktrace=all \
      --python-backtrace=cuda \
      --capture-range=nvtx \
      --nvtx-capture='measurement' \
      --output="profiles/benchmark_small_fw_$(date +%Y%m%d_%H%M%S)" \
      uv run python scripts/benchmarking_new.py \
        --model_size small \
        --mode forward \
        --warmup 5 \
        --bench_iter 10
