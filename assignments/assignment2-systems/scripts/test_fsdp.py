"""Small FP32 correctness checks for the student's FSDP wrapper.

Run from the assignment directory (each command spawns its own workers):
    uv run python scripts/test_fsdp.py --case init
    uv run python scripts/test_fsdp.py --case forward
    uv run python scripts/test_fsdp.py --case train
    uv run python scripts/test_fsdp.py --device cuda --case all

These are correctness checks, not memory/performance or mixed-precision tests.
CPU uses Gloo; CUDA uses NCCL and requires one GPU per worker.
"""

import argparse
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import pprint
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from cs336_basics.model import Linear, RMSNorm
from cs336_systems.parallel_training import FSDP


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(8, 16)
        self.norm = RMSNorm(16)  # Replicated parameters also need gradient averaging.
        self.fc2 = Linear(16, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.norm(self.fc1(x))))


def compare_parameters(wrapped, reference, rank, world_size, *, gradients=False):
    """Compare optimizer-visible shards against slices of the full-batch oracle."""
    actual = {
        name.replace(".module.", "."): p
        for name, p in wrapped.module.named_parameters()
    }
    expected = dict(reference.named_parameters())
    assert actual.keys() == expected.keys(), "Wrapping lost or duplicated parameter names"
    for name, full in expected.items():
        param = actual[name]
        assert param.device == full.device, f"{name}: wrong device"
        assert param.dtype == torch.float32, f"{name}: master parameter must be FP32"
        assert param.requires_grad and param.is_leaf, f"{name}: optimizer needs a trainable leaf"
        target = full.grad if gradients else full.detach()
        value = param.grad if gradients else param.detach()
        assert value is not None, f"{name}: missing gradient"
        assert target is not None, f"{name}: reference gradient missing"
        if name in {"fc1.weight", "fc2.weight"}:
            target = target.flatten().chunk(world_size)[rank]
        assert param.numel() == target.numel(), f"{name}: optimizer must see only its local shard"
        if gradients:
            assert value.shape == param.shape, f"{name}: gradient shape differs from parameter"
            assert value.dtype == param.dtype, f"{name}: gradient dtype differs from parameter"

            # print(f"rank {rank}, Value:")
            # pprint.pp(value.flatten())
            

            # print(f"rank {rank}, Target:")
            # pprint.pp(target.flatten())
        torch.testing.assert_close(
            value.flatten(), target.flatten(), rtol=1e-4, atol=1e-6,
            msg=lambda message: f"rank {rank}, {name}, gradients={gradients}: {message}",
        )


def worker(rank, world_size, device_kind, case, rendezvous):
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if device_kind == "cuda" else "cpu")
    if device_kind == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if device_kind == "cuda" else "gloo",
        init_method=rendezvous, rank=rank, world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        # Identical initial weights; distinct examples on each rank.
        torch.manual_seed(42)
        reference = ToyModel().to(device)
        wrapped = FSDP(deepcopy(reference))
        if case == "init":
            compare_parameters(wrapped, reference, rank, world_size)
        else:
            optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.03, momentum=0.9)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03, momentum=0.9)
            for step in range(3):
                torch.manual_seed(100 + step)
                inputs = torch.randn(world_size * 4, 8, device=device)
                targets = torch.randn(world_size * 4, 4, device=device)
                local = slice(rank * 4, (rank + 1) * 4)
                optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
                if case == "forward":
                    # Repeated inference checks gathering without invoking backward.
                    with torch.no_grad():
                        torch.testing.assert_close(
                            wrapped(inputs[local]), reference(inputs)[local],
                            rtol=1e-4, atol=1e-6,
                        )
                    continue
                expected = reference(inputs)
                actual = wrapped(inputs[local])
                torch.testing.assert_close(actual, expected[local], rtol=1e-4, atol=1e-6)
                nn.functional.mse_loss(expected, targets).backward()
                nn.functional.mse_loss(actual, targets[local]).backward()
                wrapped.finish_gradient_synchronization()
                # Equal local batches + mean loss require averaging across ranks.
                compare_parameters(wrapped, reference, rank, world_size, gradients=True)
                optimizer.step()
                reference_optimizer.step()
                compare_parameters(wrapped, reference, rank, world_size)
        dist.barrier()
        if rank == 0:
            print(f"PASS: {case} ({device_kind}, world_size={world_size})", flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["init", "forward", "train", "all"], default="all")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--world-size", type=int, choices=[1, 2, 4], default=2)
    args = parser.parse_args()
    if args.device == "cuda" and torch.cuda.device_count() < args.world_size:
        parser.error(f"CUDA test requires {args.world_size} visible GPUs")
    cases = ["init", "forward", "train"] if args.case == "all" else [args.case]
    failed = []
    for case in cases:
        print(f"Running {case}...", flush=True)
        # Unique file rendezvous avoids port collisions between concurrent runs.
        with TemporaryDirectory(prefix="fsdp-test-") as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            try:
                mp.spawn(worker, args=(args.world_size, args.device, case, rendezvous),
                         nprocs=args.world_size, join=True)
            except Exception as exc:
                print(f"FAIL: {case}\n{exc}", flush=True)
                failed.append(case)
    if failed:
        raise SystemExit(f"Failed cases: {', '.join(failed)}")


if __name__ == "__main__":
    main()
