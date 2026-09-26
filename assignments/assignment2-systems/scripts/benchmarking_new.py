import pprint
import gc
import math
import tomllib
import cs336_basics.model
import cs336_basics.optimizer
from cs336_basics.nn_utils import softmax
from typing import Literal
import torch
import time
import socket
import subprocess
import numpy as np
import argparse
from contextlib import contextmanager, nullcontext
import einx
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor
import os
## Write a script to perform end-to-end benchmarking of forward pass, backward pass, and optimizer step

basic_configs = []
with open("configs/model_sizes.toml", "rb") as f: 
    basic_configs = tomllib.load(f)

MODEL_ARGS_DEFAULTS = {}
for (k, v) in basic_configs["reused"].items():
    MODEL_ARGS_DEFAULTS[k] = v

for (k, v) in basic_configs["small"].items():
    MODEL_ARGS_DEFAULTS[k] = v

def nvtx_range(name):
    # CPU/MPS-only PyTorch builds do not provide NVTX.
    return torch.cuda.nvtx.range(name) if torch.cuda.is_available() else nullcontext()

@contextmanager
def annotate_transformer_blocks(model):
    """Label module boundaries for a diagnostic CUDA trace only."""
    if next(model.parameters()).device.type != "cuda":
        raise ValueError("--nvtx_blocks requires a model on CUDA")

    handles = []
    backward_ranges = {}

    def push_range(label):
        def hook(module, inputs):
            torch.cuda.nvtx.range_push(label)
        return hook

    def pop_range(module, inputs, outputs):
        torch.cuda.nvtx.range_pop()

    def start_backward_range(label):
        def hook(module, grad_output):
            # Autograd also uses NVTX push/pop. A range spanning callbacks
            # must use an explicit handle so their stacks cannot interfere.
            range_id = torch.cuda.nvtx.range_start(label)
            backward_ranges.setdefault(module, []).append(range_id)
        return hook

    def end_backward_range(module, grad_input, grad_output):
        torch.cuda.nvtx.range_end(backward_ranges[module].pop())

    try:
        for index, block in enumerate(model.layers):
            handles.extend([
                block.register_forward_pre_hook(push_range(f"block_{index}/forward")),
                block.register_forward_hook(pop_range, always_call=True),
                block.register_full_backward_pre_hook(start_backward_range(f"block_{index}/backward")),
                block.register_full_backward_hook(end_backward_range),
            ])
        yield
    finally:
        for handle in handles:
            handle.remove()
        # A failed backward may skip its post-hook. Close any remaining ranges.
        for pending in backward_ranges.values():
            while pending:
                torch.cuda.nvtx.range_end(pending.pop())

@torch.cuda.nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx_range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx_range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension


    with nvtx_range("final matmul"):
        res = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

    return res


cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

def torch_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

# defaults to making a small model with basic config settings
# can override the size option and any other options
def init_model(size_option: Literal["small", "medium", "large", "xl", "10B"] = "small", device = None,
               **kwargs):
    model_config = {}
    for (k, v) in basic_configs["reused"].items():
        model_config[k] = v

    for (k, v) in basic_configs[size_option].items():
        model_config[k] = v

    for (k, v) in kwargs.items():
        model_config[k] = v

    if device is None:
        device = model_config.get("device", "auto")
    if str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)
    model_config["device"] = str(device)

    model = cs336_basics.model.BasicsTransformerLM(
        vocab_size = model_config["vocab_size"],
        context_length = model_config["context_length"],
        d_model = model_config["d_model"],
        num_layers = model_config["num_layers"],
        num_heads = model_config["num_heads"],
        d_ff = model_config["d_ff"],
        rope_theta = model_config["rope_theta"])
    model.to(device)
    return model, model_config

def benchmark(model, batch_size, mode: Literal["forward", "backward", "train"] = "forward", num_iters = 10, warmup_iters = 10, autocast = False):
    device = next(model.parameters()).device
    if mode not in ("forward", "backward", "train"):
        raise ValueError("Unexpected mode:", mode)

    # data = torch.randint(model.vocab_size, (batch_size, model.context_length), device=device)
    # target = torch.randint(model.vocab_size, (batch_size, model.context_length), device=device)

    with nvtx_range("optimizer_setup"):
        opt = cs336_basics.optimizer.AdamW(model.parameters())
        opt.zero_grad()

    if mode == "forward":
        model.eval()
    else:
        model.train()

    def step(sync=True):
        if mode in ("backward", "train"):
            with nvtx_range("zero_grad"):
                opt.zero_grad(set_to_none=True)

        with nvtx_range("generate_batch"):
            data = torch.randint(model.vocab_size, (batch_size, model.context_length), device=device)
            target = torch.randint(model.vocab_size, (batch_size, model.context_length), device=device)

        if sync:
            with nvtx_range("sync_before_step"):
                torch_sync(device)
        start_time = time.perf_counter()
        cast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and autocast
            else nullcontext()
        )
        with cast:
            if mode == "forward":
                with nvtx_range("forward"):
                    model(data)
            else:
                with nvtx_range("forward"):
                    logits = model(data)
                with nvtx_range("cross_entropy"):
                    loss = cs336_basics.nn_utils.cross_entropy(logits, target)

        if mode == "backward":
            # Exclude the forward pass and loss, including queued GPU work.
            if sync:
                with nvtx_range("sync_before_backward"):
                    torch_sync(device)
            start_time = time.perf_counter()

        if mode in ("backward", "train"):
            with nvtx_range("backward"):
                loss.backward()
        if mode == "train":
            with nvtx_range("optimizer_step"):
                opt.step()
        if sync:
            with nvtx_range("sync_after_step"):
                torch_sync(device)
        end_time = time.perf_counter()
        return end_time - start_time

    with nvtx_range("warmup"):
        for j in range(warmup_iters):
            with nvtx_range(f"warmup_iteration_{j}"):
                step(sync=False)

    torch_sync(device)

    with nvtx_range("measurement"):
        times = []
        for j in range(num_iters):
            with nvtx_range(f"measured_iteration_{j}"):
                times.append(step())
    return np.array(times)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--bench_iter", type=int, default=10)
    parser.add_argument("--model_size", type=str, help="small|medium|large|xl|10B", default="small")
    parser.add_argument("--mode", type=str, help="forward|backward|train|all", default="all")
    parser.add_argument("--out_dir", type=str, help="output dir for profiling", default="benchmark")
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--profile_memory', action='store_true')
    parser.add_argument('--nvtx_blocks', action='store_true',
                        help='Add per-block forward/backward NVTX ranges for diagnostic profiling')


    for k, v in MODEL_ARGS_DEFAULTS.items():
        parser.add_argument(f"--{k}", type = type(v), default=None)
    
    args = parser.parse_args()
    run_benchmarks(args)


def run_benchmarks(args):
    batch_size = args.batch_size
    warmup = args.warmup
    num_iters = args.bench_iter
    if args.mode == "all":
        modes = ["forward", "backward", "train"]
    else:
        modes = [args.mode]

    if args.profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    with nvtx_range("model_setup"):
        model, model_config = init_model(args.model_size, **{k: getattr(args, k) for k in MODEL_ARGS_DEFAULTS if getattr(args, k) is not None},)
    device = next(model.parameters()).device
    
    print("Host:", socket.gethostname())
    print("CUDA available:", torch.cuda.is_available())

    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i}: {torch.cuda.get_device_name(i)}")

    commit = subprocess.check_output(
      ["git", "rev-parse", "HEAD"], text=True
    ).strip()


    print("Git commit:", commit)
    print("Device:", device.type)
    print("Precision:", "BF16 autocast" if args.mixed_precision and device.type == "cuda" else "FP32")
    print("Warmup iters:", warmup)
    print("Benchmark iters:", num_iters)
    print("Model size:", args.model_size)
    print("Model config:")
    pprint.pp(model_config)
    print("Num params:", model.get_num_params())
    print("Batch size:", batch_size)
    
    
    block_annotations = annotate_transformer_blocks(model) if args.nvtx_blocks else nullcontext()
    with block_annotations:
        for mode in modes:
            with nvtx_range(f"benchmark_{mode}"):
                times = benchmark(model, batch_size=batch_size, mode=mode, warmup_iters=warmup, num_iters=num_iters, autocast=args.mixed_precision)
            print(f"mean time/{mode} = {np.mean(times):.3f}s, std = {np.std(times):.3f}s")

    if args.profile_memory:
        torch.cuda.memory._dump_snapshot(f"{args.out_dir}_memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

if __name__ == "__main__":
    main()
