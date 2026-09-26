import argparse

import time
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import statistics

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--data_size_mb", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iter", type=int, default=20)

    args = parser.parse_args()

    mp.spawn(fn = benchmark_all_reduce, args=(int(args.data_size_mb * 1e6), args.world_size, args.warmup, args.iter), nprocs=args.world_size, join=True)



    # world_size = 4
    # data_size = 1e6
    # mp.spawn(fn=distributed_demo, args=(data_size, world_size, ), nprocs=world_size, join=True)

