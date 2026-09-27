import argparse
from typing import Any, Type

import time
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import statistics
import torch.optim as optim


class NaiveDDP(torch.nn.Module):
    """Reduce each parameter gradient after the caller finishes backward."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def finish_gradient_synchronization(self):
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.div_(dist.get_world_size())
                dist.all_reduce(p.grad, async_op=False)


class DDP(torch.nn.Module):
    """Overlap backward with one asynchronous all-reduce per parameter."""
    def __init__(self, module):
        super().__init__()
        self.module = module

        # sync parameters
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p, src=0)

        # add hooks to all_reduce gradients
        self.handles = [p.register_post_accumulate_grad_hook(self.synchronize_grad) for p in self.module.parameters() if p.requires_grad]
        self.pending_work = []

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def synchronize_grad(self, p):
        # Scale before asynchronous communication so the sum gives the mean.
        p.grad.div_(dist.get_world_size())
        self.pending_work.append(dist.all_reduce(p.grad, async_op=True))

    def finish_gradient_synchronization(self):
        for work in self.pending_work:
            work.wait()

        self.pending_work.clear()


class DDPFlat(torch.nn.Module):
    """Synchronize all gradients with one collective after backward.

    Trainable parameters must share a dtype/device and all receive gradients.
    Call finish_gradient_synchronization() before each optimizer step.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def finish_gradient_synchronization(self):
        params = [p for p in self.module.parameters() if p.requires_grad]
        if not params:
            return
        if any(p.grad is None for p in params):
            raise RuntimeError("DDPFlat requires gradients for every trainable parameter.")

        # Pack gradients into a single communication buffer and average across ranks.
        flat_grad = torch.cat([p.grad.reshape(-1) for p in params])
        flat_grad.div_(dist.get_world_size())
        dist.all_reduce(flat_grad)

        # Restore each gradient's original shape without replacing its tensor.
        offset = 0
        for p in params:
            numel = p.numel()
            p.grad.copy_(flat_grad[offset : offset + numel].view_as(p))
            offset += numel


class OptimizerSharded(optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[optim.Optimizer], **kwargs: Any):
        self.world_size = dist.get_world_size()
        self.assign_on_rank = 0
        self.optimizer_cls = optimizer_cls

        self.params_by_rank = [[] for _ in range(self.world_size)]
        self.local_optimizer = None
        self.params_per_rank = [0 for _ in range(self.world_size)]
        super().__init__(params, defaults=kwargs) # initializers self.param_groups

    def step(self, closure, **kwargs):
        if self.local_optimizer is not None:
            self.local_optimizer.step(closure, **kwargs)

        for owner, params in enumerate(self.params_by_rank):
            for p in params:
                dist.broadcast(p, src=owner)


    def add_param_group(self, param_group: dict[str, Any]):
        # Validate/register the full group and materialize any parameter generator.
        super().add_param_group(param_group)
        settings = {k: v for k, v in param_group.items() if k != "params"}
        groups_by_rank = [dict(settings, params=[]) for _ in range(self.world_size)]

        for p in param_group["params"]:
            owner = self.params_per_rank.index(min(self.params_per_rank))
            groups_by_rank[owner]["params"].append(p)
            self.params_by_rank[owner].append(p)
            self.params_per_rank[owner] += p.numel()

        local_group = groups_by_rank[dist.get_rank()]
        if not local_group["params"]:
            return
        if self.local_optimizer is None:
            self.local_optimizer = self.optimizer_cls([local_group], **self.defaults)
        else:
            self.local_optimizer.add_param_group(local_group)
