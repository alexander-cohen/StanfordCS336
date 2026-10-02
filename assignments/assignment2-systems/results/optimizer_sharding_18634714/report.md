# Optimizer sharding: memory and speed

Model: xl; GPUs: 2; local batch: 8; context: 512; dtype: float32.
Optimizer: torch.optim.AdamW, foreach=False; same overlapping DDP in both modes.

Memory is maximum per-GPU PyTorch allocation across ranks, in GiB (not a sum).
Phase peaks reset before model construction, forward/backward, and optimizer update. They include live tensors already present at the reset. Current allocations are measured at each checkpoint.
NCCL allocations outside the PyTorch allocator are not included. Full per-rank data are in JSON.

| Checkpoint | Unsharded peak | Sharded peak | Unsharded current | Sharded current |
|---|---:|---:|---:|---:|
| after_model_init | 12.817 | 12.817 | 12.817 | 12.817 |
| first_before_optimizer_step | 67.132 | 67.132 | 25.571 | 25.571 |
| first_after_optimizer_step | 51.247 | 38.555 | 50.954 | 38.263 |
| warmed_before_optimizer_step | 92.546 | 79.855 | 50.954 | 38.263 |
| warmed_after_optimizer_step | 51.247 | 38.555 | 50.954 | 38.263 |

Speed: 20 timed steps after 5 warmup steps and two memory-probe steps. Slowest rank's average, with synchronization only at the timed block boundaries.

| Mode | ms / iteration | Speedup vs unsharded |
|---|---:|---:|
| unsharded | 2105.938 | 1.000x |
| sharded | 2112.299 | 0.997x |
