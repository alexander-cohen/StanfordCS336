import argparse

import time
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import statistics


class DDP(torch.nn.Module):
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
