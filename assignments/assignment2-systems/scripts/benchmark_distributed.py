import argparse
import json
import tomllib
from typing import Literal
from pathlib import Path

import time
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import statistics
import cs336_basics.model
import cs336_basics.nn_utils
import cs336_basics.optimizer
import cs336_systems.parallel_training

def setup(rank, backend, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend, rank=rank, world_size=world_size) # "gloo" for cpu "nccl" for gpu

def distributed_demo(rank, data_size, world_size):
    if torch.cuda.is_available():
        backend = "nccl"
        device = f"cuda:{rank}"
    else:
        backend = "gloo"
        device = "cpu"


    setup(rank, backend, world_size)
    data_len = data_size // 4
    data = torch.randn(data_len, dtype=torch.float32, device="cuda")
    print(f"device={device}, rank={rank}, data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    print(f"device={device}, rank={rank}, data (after all-reduce): {data}")


def benchmark_all_reduce(rank, 
                         data_size = 1e6, # data size in bytes
                         world_size = 2,
                         warmup = 5, 
                         iter = 10):

    is_cuda = False
    if torch.cuda.is_available():
        is_cuda = True
        backend = "nccl"
        device = f"cuda"
        torch.cuda.set_device(rank)
    else:
        is_cuda = False
        backend = "gloo"
        device = "cpu"

    setup(rank, backend, world_size)
    data_len = data_size // 4

    stream = torch.cuda.current_stream(device)

    def iter_func():
        data = torch.randn(data_len, dtype=torch.float32, device=device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record(stream)
        # start = time.perf_counter()
        dist.all_reduce(data, async_op=False)
        # if is_cuda: torch.cuda.synchronize()
        # end = time.perf_counter()
        # return (end - start) * 1000 
        end.record(stream)
        end.synchronize()
        elapsed_ms = start.elapsed_time(end)  
        return elapsed_ms

    for _ in range(warmup):
         iter_func()

    mean_time = statistics.mean([iter_func() for _ in range(iter)])
    # print(f"rank {rank}, all reduce ms {mean_time}ms")

    mean_times = [0 for _ in range(world_size)]
    dist.all_gather_object(mean_times, mean_time)
    if rank == 0:
        # print(f"all mean times: {mean_times}ms")
        print(f"data_size={data_size/1e6}MB, world_size={world_size}, is_cuda={is_cuda}, all_reduce mean time: {statistics.mean(mean_times)}ms")


basic_configs = []
with open("configs/model_sizes.toml", "rb") as f:
    basic_configs = tomllib.load(f)

MODEL_ARGS_DEFAULTS = {}
for (k, v) in basic_configs["reused"].items():
    MODEL_ARGS_DEFAULTS[k] = v

for (k, v) in basic_configs["small"].items():
    MODEL_ARGS_DEFAULTS[k] = v

def init_model(size_option: Literal["small", "medium", "large", "xl", "10B"] = "small", device = None,**kwargs):
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

def benchmark_naive_ddp_training(
    rank, model_size="xl", world_size=2, batch_size_per_dev=8, warmup=5, iter=20,
    profile=False, profile_iters=3, trace_dir="profiles/ddp",
    ddp_variant="overlap", seed=42, timing_breakdown=False, result_json=None,
):
    is_cuda = torch.cuda.is_available()
    device = f"cuda:{rank}" if is_cuda else "cpu"
    if is_cuda:
        torch.cuda.set_device(rank)
    setup(rank, "nccl" if is_cuda else "gloo", world_size)

    torch.manual_seed(seed)
    model, model_config = init_model(model_size, device=device)
    wrappers = {
        "naive": cs336_systems.parallel_training.NaiveDDP,
        "overlap": cs336_systems.parallel_training.DDP,
        "flat": cs336_systems.parallel_training.DDPFlat,
    }
    model_ddp = wrappers[ddp_variant](model)
    optimizer = cs336_basics.optimizer.AdamW(model_ddp.parameters(), lr=0.1)
    model_ddp.train()

    generator = torch.Generator(device=device).manual_seed(seed + rank)
    tokens = torch.randint(
        model_config["vocab_size"],
        (batch_size_per_dev, model_config["context_length"] + 1),
        generator=generator, device=device,
    )
    data = tokens[:, :-1].contiguous()
    labels = tokens[:, 1:].contiguous()

    def synchronize():
        if is_cuda:
            torch.cuda.synchronize(device)

    samples = []
    for step in range(warmup + iter):
        synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = cs336_basics.nn_utils.cross_entropy(model_ddp(data), labels)
        loss.backward()
        # Wait only for the compute stream: a device-wide sync here would also
        # wait for outstanding communication, hiding the remaining wait below.
        if timing_breakdown and is_cuda:
            torch.cuda.current_stream(device).synchronize()
        backward_end = time.perf_counter()

        model_ddp.finish_gradient_synchronization()
        if timing_breakdown:
            synchronize()
        communication_end = time.perf_counter()

        optimizer.step()
        synchronize()
        end = time.perf_counter()
        if step >= warmup:
            samples.append((
                (backward_end - start) * 1000,
                (communication_end - backward_end) * 1000,
                (end - communication_end) * 1000,
            ))

    rank_samples = [None] * world_size
    dist.all_gather_object(rank_samples, samples)
    if rank == 0:
        # Report the rank with the largest mean step time, not the average worker.
        slowest_rank = max(range(world_size), key=lambda r: statistics.mean(map(sum, rank_samples[r])))
        selected_samples = rank_samples[slowest_rank]
        backward_ms, wait_ms, optimizer_ms = [statistics.mean(values) for values in zip(*selected_samples)]
        total_ms = backward_ms + wait_ms + optimizer_ms
        summary = dict(variant=ddp_variant, model=model_size, world_size=world_size,
                       batch_size_per_dev=batch_size_per_dev, seed=seed, warmup=warmup, iterations=iter,
                       timing_breakdown=timing_breakdown, slowest_rank=slowest_rank,
                       mean_step_ms=total_ms, std_step_ms=statistics.pstdev(map(sum, selected_samples)))
        print(f"DDP variant={ddp_variant}, model={model_size}, world_size={world_size}, "
              f"local_batch={batch_size_per_dev}, warmup={warmup}, measured_iters={iter}")
        print(f"Slowest mean rank ({slowest_rank}): step={total_ms:.3f} ms "
              f"(std {summary['std_step_ms']:.3f} ms)", flush=True)
        if timing_breakdown:
            print(f"Instrumented breakdown: forward/backward={backward_ms:.3f} ms, "
                  f"finish_gradient_synchronization={wait_ms:.3f} ms ({wait_ms / total_ms:.1%}), "
                  f"optimizer={optimizer_ms:.3f} ms. For overlap, synchronization is only the remaining wait.")
        if result_json:
            result_path = Path(result_json)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(summary, indent=2) + "\n")

    if profile:
        # A separate pass keeps profiler overhead out of the timing report.
        # No extra synchronization between phases: preserve DDP's overlap.
        activities = [torch.profiler.ProfilerActivity.CPU]
        if is_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as prof:
            for _ in range(profile_iters):
                optimizer.zero_grad(set_to_none=True)
                with torch.profiler.record_function("forward_backward"):
                    loss = cs336_basics.nn_utils.cross_entropy(model_ddp(data), labels)
                    loss.backward()
                with torch.profiler.record_function("finish_gradient_synchronization"):
                    model_ddp.finish_gradient_synchronization()
                with torch.profiler.record_function("optimizer_step"):
                    optimizer.step()
                prof.step()
            synchronize()
        trace_path = Path(trace_dir) / f"ddp_rank{rank}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))
        print(f"Rank {rank}: saved {profile_iters} profiled steps to {trace_path}", flush=True)

    dist.destroy_process_group()



def entry_benchmark_all_reduce(args):
    mp.spawn(
        fn=benchmark_all_reduce,
        args=(int(args.data_size_mb * 1e6), args.world_size, args.warmup, args.iter),
        nprocs=args.world_size,
        join=True,
    )


def entry_benchmark_naive_ddp(args):
    # spawn supplies rank as the first argument to each worker.
    mp.spawn(
        fn=benchmark_naive_ddp_training,
        args=(args.model_size, args.world_size, args.batch_size_per_dev, args.warmup, args.iter,
              args.profile, args.profile_iters, args.trace_dir,
              args.ddp_variant, args.seed, args.timing_breakdown, args.result_json),
        nprocs=args.world_size,
        join=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark all-reduce or DDP training.")
    parser.add_argument("--benchmark", choices=["all_reduce", "ddp_training"], default="all_reduce")
    parser.add_argument("--world_size", type=int, default=2)

    all_reduce_args = parser.add_argument_group("all_reduce options")
    all_reduce_args.add_argument("--data_size_mb", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iter", type=int, default=20)

    training_args = parser.add_argument_group("ddp_training options")
    training_args.add_argument("--model_size", choices=["small", "medium", "large", "xl", "10B"], default="xl")
    training_args.add_argument("--batch_size_per_dev", type=int, default=8)
    training_args.add_argument("--profile", action="store_true", help="Capture extra training steps after timing")
    training_args.add_argument("--profile_iters", type=int, default=3)
    training_args.add_argument("--trace_dir", default="profiles/ddp")
    training_args.add_argument("--ddp_variant", choices=["naive", "overlap", "flat"], default="overlap")
    training_args.add_argument("--seed", type=int, default=42)
    training_args.add_argument("--timing_breakdown", action="store_true", help="Synchronize phase boundaries for diagnostic timings")
    training_args.add_argument("--result_json", help="Write timing summary to this file")
    args = parser.parse_args()

    if args.world_size < 1:
        parser.error("--world_size must be positive")
    if args.iter < 1 or args.warmup < 0:
        parser.error("--iter must be positive; --warmup must be nonnegative")
    if args.benchmark == "all_reduce":
        if args.profile:
            parser.error("--profile requires --benchmark ddp_training")
        if args.data_size_mb < 1:
            parser.error("--data_size_mb must be positive")
        entry_benchmark_all_reduce(args)
    else:
        if args.batch_size_per_dev < 1:
            parser.error("--batch_size_per_dev must be positive")
        if args.profile_iters < 1:
            parser.error("--profile_iters must be positive")
        entry_benchmark_naive_ddp(args)


if __name__ == "__main__":
    main()
