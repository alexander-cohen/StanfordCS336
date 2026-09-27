"""Compare optimizer-state memory and training speed using the existing DDP wrappers.

Run from the assignment directory. Each mode starts fresh worker processes.
CUDA peaks are reset at phase boundaries; current allocations are recorded too.
"""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.nn_utils import cross_entropy
from cs336_systems.parallel_training import DDP, OptimizerSharded
from scripts.benchmark_distributed import init_model, setup


def profile_model(rank, args, device):
    """Called after process-group initialization; returns this rank's measurements."""
    torch.manual_seed(args.seed)
    result = dict(mode=args.mode, rank=rank, model=args.model_size, world_size=args.world_size,
                  batch_size_per_dev=args.batch_size_per_dev, dtype="float32", seed=args.seed,
                  optimizer="torch.optim.AdamW", foreach=False, lr=args.lr,
                  warmup=args.warmup, iterations=args.iter, checkpoints=[])
    output = Path(args.output_dir) / f"{args.mode}_rank{rank}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    def synchronize():
        torch.cuda.synchronize(device)

    def checkpoint(stage):
        synchronize()
        peak = torch.cuda.max_memory_allocated(device)
        measurement = dict(stage=stage, allocated_bytes=torch.cuda.memory_allocated(device),
                           phase_peak_bytes=peak,
                           reserved_bytes=torch.cuda.memory_reserved(device))
        result["checkpoints"].append(measurement)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{args.mode} rank={rank} {stage}: allocated={measurement['allocated_bytes']/2**30:.3f} GiB, "
              f"phase peak={peak/2**30:.3f} GiB", flush=True)

    torch.cuda.reset_peak_memory_stats(device)
    model, config = init_model(args.model_size, device=device)
    result["context_length"] = config["context_length"]
    result["parameter_count"] = sum(p.numel() for p in model.parameters())
    checkpoint("after_model_init")
    model = DDP(model)
    model.train()
    options = dict(lr=args.lr, foreach=False)
    optimizer = (OptimizerSharded(model.parameters(), torch.optim.AdamW, **options)
                 if args.mode == "sharded" else torch.optim.AdamW(model.parameters(), **options))
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    tokens = torch.randint(config["vocab_size"], (args.batch_size_per_dev, config["context_length"] + 1),
                           device=device, generator=generator)
    data, labels = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    del tokens

    def backward():
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy(model(data), labels)
        loss.backward()
        model.finish_gradient_synchronization()

    def update():
        # Explicit None also supports the student's required closure argument.
        with torch.no_grad():
            optimizer.step(None)

    def memory_step(prefix):
        synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        backward()
        checkpoint(f"{prefix}_before_optimizer_step")
        torch.cuda.reset_peak_memory_stats(device)
        update()
        checkpoint(f"{prefix}_after_optimizer_step")

    # AdamW initializes its moment buffers lazily: report first and warmed steps.
    memory_step("first")
    for _ in range(args.warmup):
        backward()
        update()
    memory_step("warmed")

    # No memory queries, file writes, or inter-phase synchronizations in this pass.
    synchronize()
    dist.barrier()
    synchronize()
    start = time.perf_counter()
    for _ in range(args.iter):
        backward()
        update()
    synchronize()
    result["mean_step_ms"] = (time.perf_counter() - start) * 1000 / args.iter
    local_optimizer = optimizer.local_optimizer if args.mode == "sharded" else optimizer
    result["optimizer_state_bytes"] = sum(
        value.numel() * value.element_size()
        for state in (local_optimizer.state.values() if local_optimizer is not None else [])
        for value in state.values() if isinstance(value, torch.Tensor) and value.device == torch.device(device)
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def worker(rank, args):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    setup(rank, "nccl", args.world_size)
    try:
        profile_model(rank, args, device)
    finally:
        dist.destroy_process_group()


def report(output_dir, world_size):
    root = Path(output_dir)
    runs = {mode: [json.loads((root / f"{mode}_rank{rank}.json").read_text())
                   for rank in range(world_size)] for mode in ("unsharded", "sharded")}
    config = runs["unsharded"][0]
    lines = ["# Optimizer sharding: memory and speed", "",
             f"Model: {config['model']}; GPUs: {world_size}; local batch: {config['batch_size_per_dev']}; "
             f"context: {config['context_length']}; dtype: {config['dtype']}.",
             "Optimizer: torch.optim.AdamW, foreach=False; same overlapping DDP in both modes.", "",
             "Memory is maximum per-GPU PyTorch allocation across ranks, in GiB (not a sum).",
             "Phase peaks reset before model construction, forward/backward, and optimizer update. "
             "They include live tensors already present at the reset. Current allocations are measured at each checkpoint.",
             "NCCL allocations outside the PyTorch allocator are not included. Full per-rank data are in JSON.", "",
             "| Checkpoint | Unsharded peak | Sharded peak | Unsharded current | Sharded current |",
             "|---|---:|---:|---:|---:|"]
    for index, measurement in enumerate(config["checkpoints"]):
        values = [max(row["checkpoints"][index][field] for row in runs[mode]) / 2**30
                  for field in ("phase_peak_bytes", "allocated_bytes") for mode in runs]
        lines.append(f"| {measurement['stage']} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    times = {mode: max(row["mean_step_ms"] for row in rows) for mode, rows in runs.items()}
    lines += ["", f"Speed: {config['iterations']} timed steps after {config['warmup']} warmup steps "
              "and two memory-probe steps. Slowest rank's average, with synchronization only at the timed block boundaries.", "",
              "| Mode | ms / iteration | Speedup vs unsharded |", "|---|---:|---:|"]
    for mode, ms in times.items():
        lines.append(f"| {mode} | {ms:.3f} | {times['unsharded']/ms:.3f}x |")
    text = "\n".join(lines) + "\n"
    (root / "report.md").write_text(text)
    print(text, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["both", "unsharded", "sharded"], default="both")
    parser.add_argument("--model_size", choices=["small", "medium", "large", "xl", "10B"], default="xl")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--batch_size_per_dev", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output_dir", default="results/optimizer_sharding")
    args = parser.parse_args()
    if min(args.world_size, args.batch_size_per_dev, args.iter) < 1 or args.warmup < 0:
        parser.error("world_size, batch_size_per_dev, iter must be positive; warmup must be nonnegative")
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
        parser.error(f"This profiler requires {args.world_size} visible CUDA GPUs")
    modes = ("unsharded", "sharded") if args.mode == "both" else (args.mode,)
    for mode in modes:
        run_args = argparse.Namespace(**vars(args))
        run_args.mode = mode
        mp.spawn(worker, args=(run_args,), nprocs=args.world_size, join=True)
    if args.mode == "both":
        report(args.output_dir, args.world_size)


if __name__ == "__main__":
    main()
