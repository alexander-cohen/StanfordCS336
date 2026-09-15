import tomllib
import cs336_basics.model
import cs336_basics.optimizer
from typing import Literal
import torch
import time
import socket
import subprocess
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
    return model

def benchmark(model, batch_size, mode: Literal["forward", "backward", "train"] = "forward", num_iters = 10, warmup_iters = 10):
    device = next(model.parameters()).device

    data = torch.randint(model.vocab_size, (batch_size, model.context_length), device=device)
    target = torch.randint(model.vocab_size, (batch_size, model.context_length), device=device)
    
    if mode == "forward":
        model.eval()
        for j in range(warmup_iters):
            model(data)

        torch_sync(device)
        start_time = time.perf_counter()
        for j in range(warmup_iters, warmup_iters+num_iters):
            model(data)
        torch_sync(device)
        end_time = time.perf_counter()
        tot_time = end_time - start_time
        time_per_pass = tot_time / num_iters
        return time_per_pass

    elif mode == "backward":
        model.train()
        for j in range(warmup_iters):
            loss = cs336_basics.nn_utils.cross_entropy(model(data), target)
            loss.backward()


        torch_sync(device)
        start_time = time.perf_counter()
        for j in range(warmup_iters, warmup_iters+num_iters):
            loss = cs336_basics.nn_utils.cross_entropy(model(data), target)
            loss.backward()
        torch_sync(device)
        end_time = time.perf_counter()
        tot_time = end_time - start_time
        time_per_fb_pass = tot_time / num_iters
        return time_per_fb_pass

    elif mode == "train":
        model.train()
        opt = cs336_basics.optimizer.AdamW(model.parameters())
        opt.zero_grad()

        for j in range(warmup_iters):
            loss = cs336_basics.nn_utils.cross_entropy(model(data), target)
            loss.backward()
            opt.step()

        torch_sync(device)
        start_time = time.perf_counter()
        for j in range(warmup_iters, warmup_iters+num_iters):
            loss = cs336_basics.nn_utils.cross_entropy(model(data), target)
            loss.backward()
            opt.step()
        torch_sync(device)
        end_time = time.perf_counter()
        tot_time = end_time - start_time
        time_per_train = tot_time / num_iters
        return time_per_train
    else:
        raise ValueError("Unexpected mode:", mode)
    

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--mode", type=str, help="Mode: train, encode, decode, test_compr, or encode_and_serialize, or none", default="none")
    # parser.add_argument("--corpus", type=str, help="filepath OR ts_valid OR ts_train OR owt_valid OR owt_train OR test_corpus", default="none")
    # parser.add_argument("--vocab_size", type=int, help="Size of vocab for training", default="257")
    # parser.add_argument("--cores", type=int, help="Number of cores for pretokenization", default="8")
    # parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
    # parser.add_argument("--use_disk", action="store_true", help="Store frequency table to disk (so big corpus doesn't crash memory)")
    # parser.add_argument("--num_pieces", type=int, help="Number of pieces to chunk the text into", default=8)
    
    # args = parser.parse_args()
    batch_size = 4
    warmup = 10
    num_iters = 10
    device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)

    print("Host:", socket.gethostname())
    print("CUDA available:", torch.cuda.is_available())

    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i}: {torch.cuda.get_device_name(i)}")

    commit = subprocess.check_output(
      ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    print(f"Git commit: {commit}", flush=True)

    for config in ["small", "medium", "large", "xl", "10B"]:
        model = init_model(config, device)
        for type in ["forward", "backward", "train"]:
            time_per_pass = benchmark(model, batch_size=batch_size, mode=type, warmup_iters=warmup, num_iters=num_iters)
            print(f"{config} model, num_params = {model.get_num_params()}, device = {device.type}, warmup = {warmup}, time/{type} = {time_per_pass:.3f}s")

if __name__ == "__main__":
    main()