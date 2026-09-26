"""Benchmark unmasked, float32 attention with inputs shaped (batch_size, seq_len, d_head).

Run from the assignment root: uv run python scripts/benchmark_attention.py
CUDA events measure GPU elapsed time with autograd enabled; backward excludes forward.
Memory is live tensor allocation immediately before backward, not peak/reserved.
"""

import argparse
import gc
import json
import statistics

import torch
from cs336_basics.model import scaled_dot_product_attention
compiled_attention = torch.compile(scaled_dot_product_attention)


def benchmark(d_head, seq_len, device, batch_size, warmup=10, iterations=100, compiled=False):
    """Keep just one graph live per iteration; never retain graphs across steps."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("CUDA event timing requires a CUDA device")
    row = {"d_head": d_head, "seq_len": seq_len, "status": "ok"}
    stage = "allocation"
    try:
        q, k, v = [
            torch.randn(batch_size, seq_len, d_head, device=device, dtype=torch.float32,
                        requires_grad=True)
            for _ in range(3)
        ]
        grad_output = torch.randn_like(q)
        stream = torch.cuda.current_stream(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        forward_ms, backward_ms, memory_bytes = [], [], []
        for step in range(warmup + iterations):
            # Clear previous gradients before both timing and memory measurement.
            q.grad = k.grad = v.grad = None
            stage = "warmup forward" if step < warmup else "forward"
            start.record(stream)

            if compiled:
                output = compiled_attention(q, k, v)
            else:
                output = scaled_dot_product_attention(q, k, v)
            
            
            end.record(stream)
            end.synchronize()
            forward_elapsed = start.elapsed_time(end)  # CUDA events return milliseconds.

            before_backward = torch.cuda.memory_allocated(device)
            stage = "warmup backward" if step < warmup else "backward"
            start.record(stream)
            output.backward(grad_output)
            end.record(stream)
            end.synchronize()
            backward_elapsed = start.elapsed_time(end)
            del output

            if step >= warmup:
                forward_ms.append(forward_elapsed)
                backward_ms.append(backward_elapsed)
                memory_bytes.append(before_backward)

        row.update(
            forward_ms_mean=statistics.mean(forward_ms),
            backward_ms_mean=statistics.mean(backward_ms),
            before_backward_mib=statistics.mean(memory_bytes) / 2**20,
        )
    except torch.OutOfMemoryError as exc:
        row.update(status="oom", stage=stage, error=str(exc))
    return row


def main():
    batch_size = 8
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="cuda or cuda:N (auto also selects CUDA)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--compiled", type=bool, default=False)
    args = parser.parse_args()
    if args.warmup < 1:
        parser.error("--warmup must be at least 1")
    device = torch.device("cuda" if args.device == "auto" else args.device)
    if device.type != "cuda":
        parser.error("CUDA event timing requires a CUDA device")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    print(json.dumps({"device": str(device), "dtype": "float32", "batch_size": batch_size,
                      "iterations": 100, "warmup": args.warmup,
                      "compiled": args.compiled,
                      "attention": "unmasked", "memory": "live tensor allocation",
                      "timing": "CUDA events"}), flush=True)
    torch.manual_seed(0)
    for d_head in (16, 32, 64, 128):
        for seq_len in (256, 1024, 4096, 8192, 16384):
            row = benchmark(d_head, seq_len, device, batch_size=batch_size, warmup=args.warmup, compiled=args.compiled)
            # The function has returned, so even OOM tracebacks no longer retain
            # its tensors. Release cached blocks before attempting the next shape.
            gc.collect()
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
