import gc
import tomllib
import cs336_basics.model
import cs336_basics.optimizer
from typing import Literal
import torch
import time
import socket
import subprocess
import numpy as np
import argparse
## Write a script to perform end-to-end benchmarking of forward pass, backward pass, and optimizer step

basic_configs = []
with open("configs/model_sizes.toml", "rb") as f: 
    basic_configs = tomllib.load(f)

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

    if device is not None:
        device = device
    elif model_config.get("device", "auto") == "auto":
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
    return model

def benchmark(model, batch_size, opt = None, mode: Literal["forward", "backward", "train"] = "forward", num_iters = 10, warmup_iters = 10, verbose=False):
    device = next(model.parameters()).device

    # data = torch.randint(model.vocab_size, (batch_size, model.context_length), device=device)
    # target = torch.randint(model.vocab_size, (batch_size, model.context_length), device=device)
    if opt == None and mode == "train":
        torch_sync(device)
        start_time = time.perf_counter()
        opt = cs336_basics.optimizer.AdamW(model.parameters())
        opt.zero_grad()
        torch_sync(device)
        end_time = time.perf_counter()
        if verbose: print("opt construct time", end_time - start_time)
    elif opt is not None:
        opt.zero_grad()


    torch_sync(device)
    start_time = time.perf_counter()

    if mode == "forward":
        model.eval()
    else:
        model.train()

    torch_sync(device)
    end_time = time.perf_counter()
    if verbose: print("Time set mode:", end_time - start_time)


    def step():
        torch_sync(device)
        start_time = time.perf_counter()
        data = torch.randint(model.vocab_size, (batch_size, model.context_length), device=device)
        target = torch.randint(model.vocab_size, (batch_size, model.context_length), device=device)
        model.zero_grad(set_to_none=True)
        torch_sync(device)
        end_time = time.perf_counter()
        if verbose: print("data, target, zero grad time:", end_time - start_time)

        torch_sync(device)
        start_time = time.perf_counter()
        if mode == "forward":
            with torch.no_grad():
                model(data)
        elif mode == "backward":
            loss = cs336_basics.nn_utils.cross_entropy(model(data), target)
            loss.backward()
        elif mode == "train":
            loss = cs336_basics.nn_utils.cross_entropy(model(data), target)
            loss.backward()
            opt.step()
        else:
            raise ValueError("Unexpected mode:", mode)
        torch_sync(device)
        end_time = time.perf_counter()
        if verbose: print("step time", end_time - start_time)
        return end_time - start_time

    if verbose:  print(f"Warmup iters: {warmup_iters}, benchmark iters: {num_iters}")
    torch_sync(device)
    start_time = time.perf_counter()
    for j in range(warmup_iters):
        step()
    torch_sync(device)
    end_time = time.perf_counter()
    if verbose: print("time for warmup:", end_time - start_time)

    torch_sync(device)
    start_time = time.perf_counter()
    times = [step() for j in range(num_iters)]
    torch_sync(device)
    end_time = time.perf_counter()
    if verbose: print("time for iter:", end_time - start_time)
    return np.array(times)
    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup", help="Number of warmup steps before profiling", type=int, default=5)
    parser.add_argument("--bench_iter", help="Number of iterations to average over", type=int, default=10)
    parser.add_argument("--model_size", help="small|medium|large|xl|10B", type=str, default="small")
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--mode", help="forward|backward|train|all", default="all")
    parser.add_argument("--compile", help="y|n", default="n")
    
    args = parser.parse_args()


    batch_size = args.batch_size
    warmup = args.warmup
    num_iters = args.bench_iter
    model_size = args.model_size
    context_length = args.context_length

    device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)

    print("Host:",   socket.gethostname())
    print("CUDA available:", torch.cuda.is_available())

    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i}: {torch.cuda.get_device_name(i)}")

    commit = subprocess.check_output(
      ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    print(f"Git commit: {commit}", flush=True)
    print("Device:", device.type)
    print("Batch size:", batch_size)
    print("Warmup iters:", warmup)
    print("Benchmark iters:", num_iters)
    print("Compile:", args.compile)

    if args.mode == "all":
        modes = ["forward", "backward", "train"]
    elif args.mode == "forward":
        modes = ["forward"]
    elif args.mode == "backward":
        modes = ["backward"]
    elif args.mode == "train":
        modes = ["train"]
    else:
        raise ValueError("Unexpected args.mode:", args.mode)
    
    model = init_model(model_size, device, context_length = context_length)
    opt = cs336_basics.optimizer.AdamW(model.parameters())
    if args.compile == "y":
        model = torch.compile(model)
    for mode in modes:
        times = benchmark(model, batch_size=batch_size, opt = opt, mode=mode, warmup_iters=warmup, num_iters=num_iters)
        print(f"{model_size} model, num_params = {model.get_num_params()}, mean time/{mode} = {np.mean(times):.3f}s, std = {np.std(times):.3f}s")


if __name__ == "__main__":
    main()