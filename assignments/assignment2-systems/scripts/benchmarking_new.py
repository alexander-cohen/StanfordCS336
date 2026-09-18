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
from contextlib import nullcontext
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

    if model_config.get("device", "auto") == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    else: 
        device = model_config["device"]

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

def benchmark(model, batch_size, mode: Literal["forward", "backward", "train"] = "forward", num_iters = 10, warmup_iters = 10):
    device = next(model.parameters()).device

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
        if mode == "forward":
            with nvtx_range("forward"):
                model(data)
        elif mode == "backward":
            with nvtx_range("forward"):
                logits = model(data)
            with nvtx_range("cross_entropy"):
                loss = cs336_basics.nn_utils.cross_entropy(logits, target)
            with nvtx_range("backward"):
                loss.backward()
        elif mode == "train":
            with nvtx_range("forward"):
                logits = model(data)
            with nvtx_range("cross_entropy"):
                loss = cs336_basics.nn_utils.cross_entropy(logits, target)
            with nvtx_range("backward"):
                loss.backward()
            with nvtx_range("optimizer_step"):
                opt.step()
        else:
            raise ValueError("Unexpected mode:", mode)
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
    
    # if mode == "forward":
    #     model.eval()
    #     for j in range(warmup_iters):
    #         model(data)

    #     torch_sync(device)
    #     start_time = time.perf_counter()
    #     for j in range(warmup_iters, warmup_iters+num_iters):
    #         model(data)
    #     torch_sync(device)
    #     end_time = time.perf_counter()
    #     tot_time = end_time - start_time
    #     time_per_pass = tot_time / num_iters
    #     return time_per_pass

    # elif mode == "backward":
    #     model.train()
    #     for j in range(warmup_iters):
    #         loss = cs336_basics.nn_utils.cross_entropy(model(data), target)
    #         loss.backward()


    #     torch_sync(device)
    #     start_time = time.perf_counter()
    #     for j in range(warmup_iters, warmup_iters+num_iters):
    #         loss = cs336_basics.nn_utils.cross_entropy(model(data), target)
    #         loss.backward()
    #     torch_sync(device)
    #     end_time = time.perf_counter()
    #     tot_time = end_time - start_time
    #     time_per_fb_pass = tot_time / num_iters
    #     return time_per_fb_pass

    # elif mode == "train":
    #     model.train()
    #     opt = cs336_basics.optimizer.AdamW(model.parameters())
    #     opt.zero_grad()

    #     for j in range(warmup_iters):
    #         loss = cs336_basics.nn_utils.cross_entropy(model(data), target)
    #         loss.backward()
    #         opt.step()

    #     torch_sync(device)
    #     start_time = time.perf_counter()
    #     for j in range(warmup_iters, warmup_iters+num_iters):
    #         loss = cs336_basics.nn_utils.cross_entropy(model(data), target)
    #         loss.backward()
    #         opt.step()
    #     torch_sync(device)
    #     end_time = time.perf_counter()
    #     tot_time = end_time - start_time
    #     time_per_train = tot_time / num_iters
    #     return time_per_train
    # else:
    #     raise ValueError("Unexpected mode:", mode)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--bench_iter", type=int, default=10)
    parser.add_argument("--model_size", type=str, help="small|medium|large|xl|10B", default="small")
    parser.add_argument("--mode", type=str, help="forward|backward|train|all", default="all")
    parser.add_argument("--out_dir", type=str, help="output dir for profiling", default="benchmark")

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
    print("Warmup iters:", warmup)
    print("Benchmark iters:", num_iters)
    print("Model size:", args.model_size)
    print("Model config:")
    pprint.pp(model_config)
    print("Num params:", model.get_num_params())
    print("Batch size:", batch_size)
    
    
    for mode in modes:
        with nvtx_range(f"benchmark_{mode}"):
            times = benchmark(model, batch_size=batch_size, mode=mode, warmup_iters=warmup, num_iters=num_iters)
        print(f"mean time/{mode} = {np.mean(times):.3f}s, std = {np.std(times):.3f}s")

    torch.cuda.memory._dump_snapshot(f"{args.out_dir}_memory_snapshot.pickle")

if __name__ == "__main__":
    main()
