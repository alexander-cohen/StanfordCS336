"""Compare the reference and Triton FlashAttention implementations.

Run from the assignment root with:
    uv run python scripts/benchmark_flash_attention.py

Latencies are reported in milliseconds using triton.testing.do_bench. The
reference implementation materializes the full attention matrix, so large
sequence lengths may not fit in GPU memory.
"""

import argparse
import gc
import json

import torch
import triton.testing

from cs336_basics.model import scaled_dot_product_attention
from scripts.flash_attention import triton_flash_attn


def _reference(q, k, v, causal_mask, block_size):
    return scaled_dot_product_attention(q, k, v, mask=causal_mask)


def _triton(q, k, v, causal_mask, block_size):
    output, _ = triton_flash_attn(q, k, v, block_size, block_size, True)
    return output


def _measure_passes(implementation, q, k, v, causal_mask, grad_output, block_size, warmup, rep):
    forward = lambda: implementation(q, k, v, causal_mask, block_size)
    timings = {}

    try:
        timings["forward_ms"] = triton.testing.do_bench(forward, warmup=warmup, rep=rep)
    except torch.OutOfMemoryError as exc:
        timings["forward_status"] = "oom"
        timings["forward_error"] = str(exc).splitlines()[0]
        return timings

    q_grad, k_grad, v_grad = [tensor.detach().requires_grad_(True) for tensor in (q, k, v)]
    try:
        backward_output = implementation(q_grad, k_grad, v_grad, causal_mask, block_size)
        timings["backward_ms"] = triton.testing.do_bench(
            lambda: torch.autograd.grad(
                backward_output,
                (q_grad, k_grad, v_grad),
                grad_outputs=grad_output,
                retain_graph=True,
            ),
            warmup=warmup,
            rep=rep,
        )
    except torch.OutOfMemoryError as exc:
        timings["backward_status"] = "oom"
        timings["backward_error"] = str(exc).splitlines()[0]
    finally:
        if "backward_output" in locals():
            del backward_output
        del q_grad, k_grad, v_grad
        gc.collect()
        torch.cuda.empty_cache()

    q_grad, k_grad, v_grad = [tensor.detach().requires_grad_(True) for tensor in (q, k, v)]

    def forward_backward():
        output = implementation(q_grad, k_grad, v_grad, causal_mask, block_size)
        torch.autograd.grad(output, (q_grad, k_grad, v_grad), grad_outputs=grad_output)

    try:
        timings["forward_backward_ms"] = triton.testing.do_bench(
            forward_backward, warmup=warmup, rep=rep
        )
    except torch.OutOfMemoryError as exc:
        timings["forward_backward_status"] = "oom"
        timings["forward_backward_error"] = str(exc).splitlines()[0]
    finally:
        del q_grad, k_grad, v_grad
        gc.collect()
        torch.cuda.empty_cache()

    return timings


def benchmark_case(seq_len, d_head, dtype, device, block_size, warmup, rep):
    q, k, v = [
        torch.randn((1, seq_len, d_head), device=device, dtype=dtype)
        for _ in range(3)
    ]
    grad_output = torch.randn_like(q)
    row = {
        "batch_size": 1,
        "seq_len": seq_len,
        "d_head": d_head,
        "dtype": str(dtype).removeprefix("torch."),
        "block_size": block_size,
        "reference": {},
        "triton": {},
    }

    try:
        positions = torch.arange(seq_len, device=device)
        causal_mask = positions[:, None] >= positions[None, :]
        row["reference"] = _measure_passes(
            _reference, q, k, v, causal_mask, grad_output, block_size, warmup, rep
        )
        del causal_mask
    except torch.OutOfMemoryError as exc:
        row["reference"] = {"status": "oom", "error": str(exc).splitlines()[0]}
        gc.collect()
        torch.cuda.empty_cache()

    row["triton"] = _measure_passes(
        _triton, q, k, v, None, grad_output, block_size, warmup, rep
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=float, default=25, help="do_bench warmup time in ms")
    parser.add_argument("--rep", type=float, default=100, help="do_bench measurement time in ms")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[2**power for power in range(7, 17)])
    parser.add_argument("--d-heads", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument(
        "--dtypes",
        choices=("bfloat16", "float32"),
        nargs="+",
        default=["bfloat16", "float32"],
    )
    parser.add_argument("--output", help="Optional JSONL output path; defaults to stdout")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("This benchmark requires an available CUDA device")
    if args.warmup < 0 or args.rep <= 0:
        parser.error("--warmup must be nonnegative and --rep must be positive")

    output_file = open(args.output, "w", encoding="utf-8") if args.output else None
    try:
        print(
            json.dumps(
                {
                    "device": str(device),
                    "timing": "triton.testing.do_bench",
                    "warmup_ms": args.warmup,
                    "rep_ms": args.rep,
                    "causal": True,
                }
            ),
            file=output_file,
            flush=True,
        )
        dtype_by_name = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        for dtype_name in args.dtypes:
            dtype = dtype_by_name[dtype_name]
            for d_head in args.d_heads:
                for seq_len in args.seq_lens:
                    row = benchmark_case(
                        seq_len,
                        d_head,
                        dtype,
                        device,
                        args.block_size,
                        args.warmup,
                        args.rep,
                    )
                    print(json.dumps(row), file=output_file, flush=True)
                    del row
                    gc.collect()
                    torch.cuda.empty_cache()
    finally:
        if output_file:
            output_file.close()


if __name__ == "__main__":
    main()