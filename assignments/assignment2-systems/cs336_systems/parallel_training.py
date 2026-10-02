import argparse
from typing import Any, Type

import pprint
import time
import os
import torch
import torch.distributed as dist
import torch.distributed.fsdp
import torch.multiprocessing as mp
import statistics
import torch.optim as optim
import cs336_basics.model
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int

torch.distributed.fsdp.FSDPModule

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

class CustomMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Y):
        ctx.save_for_backward(X, Y)
        return X @ Y

    @staticmethod
    def backward(ctx, grad_Z):
        # X = (a, b), Y = (b, c), Z = (a, c)
        # Z = X @ Y 
        # grad_Z = \partial L / partial Z
        # partial L / partial X = grad_Z @ Y.T
        # partial_L / partial_Y = X.T @ grad_Z
        X, Y = ctx.saved_tensors
        grad_X = grad_Z @ Y.T
        grad_Y = X.T @ grad_Z
        return grad_X, grad_Y

# # Say we have a matrix W and we evaluate L = f(W X)
# # We want W to be sharded over multiply GPUs
# # we write L = f(GatherWeight(W_shard) X) 
# # On the forward pass it all_gathers to get the full weight matrix
# # 
# # W_shard  --> W = AllGather(W_shard) --> A = WX --> L = f(A)
# # X        ---------------------------------^
# # Problem: full_W might be saved into the ctx of the matmul function, for computation in X's backward pass.
# # grad_{W_shard} = ReduceScatter(grad_W) --> grad_W = grad_A @ X.T --> grad_A = \partial_L / partial_A 
# # grad_X = (AH this part needs to gather W) W.T @ A  ------------------------------------------------^
# class GatherWeight(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, local_sha):
#         ctx.world_size = dist.get_world_size()
#         full_size = list(local_shard.shape)
#         full_size[0] *= ctx.world_size
#         full_W = local_shard.new_empty(full_size)
#         dist.all_gather_into_tensor(full_W, local_shard.continguous())
#         return full_W


#     # I worried this needs to happen ahead of time, 
#     # But maybe its OK, because the other gradients dont depend on grad_shard
#     @staticmethod
#     def backward(ctx, grad_full):
#         shard_size = list(grad_full.shape)
#         assert shard_size[0] % ctx.world_size == 0
#         shard_size[0] //= ctx.world_size

#         grad_shard = grad_full.new_empty(shard_size)
#         dist.reduce_scatter_tensor(grad_shard, grad_full.contiguous(), op=dist.ReduceOp.SUM)
#         return grad_shard

# def gather_weights_hook(module):
#     def gather_fcn(*args): # Ignores the args
#         module.full_weight = GatherWeight.apply(module.weight)
#     return gather_fcn

# def fsdp_lin_fwd(lin_module, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
#     result = einsum(x, lin_module.full_weight, "... d_in, d_out d_in -> ... d_out")
#     del lin_module.full_weight
    
#     return result

# def fsdp_embed_fwd(embed_module, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
#     result = embed_module.full_weight[token_ids, :]

#     del embed_module.full_weight
#     return result


def cdiv(a, b):
    return (a+b-1)//b

# Free the data after use
def free_data(tens):
    with torch.no_grad():
        # 1. Clear out the tensor dimensions so APIs don't attempt memory reads
        tens.set_(torch.empty((0,), dtype=tens.dtype, device=tens.device))
        
        # 2. Resize the underlying untyped storage to physically free the memory
        tens.untyped_storage().resize_(0)

## Wraps a parameter to keep track of shards
class ShardedParameter(torch.nn.Parameter):

    def __new__(cls, param, compute_dtype=torch.float32):
        return super().__new__(cls, param, requires_grad = param.requires_grad)

    ## Pass in unsharded parameter
    ## Constructed a sharded version of it
    def __init__(self, param: torch.nn.Parameter,  compute_dtype = torch.float32):
        self.param_shape = param.shape
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.my_numel = param.numel()
        self.shard_device = param.device

        assert self.my_numel % self.world_size == 0

        self.size_per_rank = self.my_numel // self.world_size

        self.compute_dtype = compute_dtype
        self.unsharded = torch.empty(self.my_numel, dtype = self.compute_dtype, device = self.shard_device) # keep unsharded weights in compute_dtype

        with torch.no_grad():
            # Copy data into the sharded piece
            self.sharded = torch.empty(self.size_per_rank, dtype=torch.float32, device = self.shard_device) # always keep sharded parameter in float32
            self.sharded.copy_(param.view(-1).chunk(self.world_size)[self.rank])
            self.data = self.sharded

        free_data(param.data)
        free_data(self.unsharded)

    # Gather the data tensor to prepare for forward pass
    def gather_data(self):
        self.unsharded = torch.empty(self.my_numel, dtype = self.compute_dtype, device = self.shard_device)
        self.data = self.unsharded
        self.gather_input = self.sharded.to(self.compute_dtype)
        handle = dist.all_gather_into_tensor(self.unsharded, self.gather_input, async_op=True)
        return handle

    def _free_data(self, *args):
        free_data(self.unsharded)

    def reshard_gradients(self, *args):
        self.sharded_grad = torch.empty_like(self.sharded).to(self.compute_dtype) # torch.float32
        to_reduce = list(self.grad.view(-1).chunk(self.world_size))
        self.grad_work = dist.reduce_scatter(self.sharded_grad, to_reduce, async_op=True)
        self.data = self.sharded

# Should be a wrapper for torch.nn.Module... not sure exactly how its supposed to work
class ShardedModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype = torch.float32):   
        super().__init__()
        self.module = module

        self.waiting_gather = []
        self.waiting_grad = []

        self.compute_dtype = compute_dtype

        # replace all parameters with sharded params
        for n, p in self.module.named_parameters():
            sharded_p = ShardedParameter(p, compute_dtype = self.compute_dtype)
            setattr(self.module, n, sharded_p)
            sharded_p.register_post_accumulate_grad_hook(ShardedParameter.reshard_gradients)

    def gather_data(self, *args):
        # or maybe self.parameters()?
        for p in self.module.parameters():
            h = p.gather_data()
            self.waiting_gather.append(h)

    def finish_gathering(self, *args):
        for h in self.waiting_gather:
            h.wait()
        self.waiting_gather.clear()

        for p in self.module.parameters():
            p.data = p.unsharded.view(p.param_shape)

    def free_data(self, *args):
        for p in self.module.parameters():
            p._free_data()

    # def reshard_gradients(self, *args):
    #     for n, p in self.module.named_parameters():
    #         print(f"Rank: {dist.get_rank()}, name: {n}, myname: {self.name}, {p.grad}")
    #         h = p.reshard_gradients()
    #         self.waiting_grad.append(h)

    def finish_resharding_gradients(self):
        for p in self.module.parameters():
            p.grad_work.wait()
            p.grad = p.sharded_grad.to(p.data.dtype)
            del p.sharded_grad


    def forward(self, x, **kwargs):
        self.finish_gathering()
        if x.is_floating_point() and x.dtype != self.compute_dtype:
            xcast = x.to(self.compute_dtype)
        else:
            xcast = x
        return self.module(xcast, **kwargs)

def average_param(p, *args):
    dist.all_reduce(p.grad)

class FSDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype = torch.float32):
        """ Given an instantiated pytorch nn.Module, construct an FSDP module that will handle weight all-gathers and gradient reduce-scatters. make sure that your hooks or your module wrappers all-gather the weights in time for the forward pass. to limit memory use, only start gathering after the layer two before the current one has completed its forward pass.
        in the backward pass, your hooks or module wrappers should all-gather to have the weights available for the computation. when the gradients are available, they should be reduce-scattered to the appropriate ranks. make sure to free the gathered weights after use. when compute_dtype is provide, cast the weights to that dtype before communicating or using them for compute, while keeping master weights and optimizer updates in FP32"""

        super().__init__()

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.module = module

        self.lin_modules = []
        self.compute_dtype = compute_dtype
        self.wrap_linear_layers(self.module)


        # register hooks to gather data for forward pass
        # this doesnt seem like quite the right way to register? not sure.
        self.register_forward_pre_hook(self.lin_modules[0].gather_data)
        for i in range(len(self.lin_modules)-1):
            m1 = self.lin_modules[i]
            m2 = self.lin_modules[i+1]
            m1.register_forward_pre_hook(m2.gather_data)

        # register hooks to gather data for backward pass
        self.register_full_backward_pre_hook(self.lin_modules[-1].gather_data)
        for i in range(len(self.lin_modules)-1):
            m1 = self.lin_modules[i]
            m2 = self.lin_modules[i+1]
            m2.register_full_backward_pre_hook(m1.gather_data)

        # reshard gradients after backward completes
        for m in self.lin_modules:
            m.register_forward_hook(m.free_data) # free the data after forward pass is complete
            m.register_full_backward_pre_hook(m.finish_gathering) # finish gathering gradients before doing backward pass
            # m.register_full_backward_hook(m.reshard_gradients) # reshard gradients after backward pass is complete


        for p in self.parameters():
            if p.requires_grad and not isinstance(p, ShardedParameter):
                p.register_post_accumulate_grad_hook(average_param)


    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for m in self.lin_modules:
            m.finish_resharding_gradients()

        for p in self.parameters():
            p.grad.div_(self.world_size)

    def wrap_linear_layers(self, parent):
        children = list(parent.named_children())
        if isinstance(parent, cs336_basics.model.SwiGLU):
            children = [
                (name, getattr(parent, name))
                for name in ("w1", "w3", "w2")
            ]
        for name, child in children:
            if isinstance(child, (cs336_basics.model.Linear, cs336_basics.model.Embedding)):
                sharded_child = ShardedModule(child, compute_dtype = self.compute_dtype)
                setattr(parent, name, sharded_child)
                sharded_child.name = name
                self.lin_modules.append(sharded_child)
            else:
                self.wrap_linear_layers(child)


vocab_size = 16
context_length = 16
d_model = 8
d_ff = 8
num_layers = 3
num_heads = 1

batch_size = 1
world_size = 2

def setup(rank, backend, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend, rank=rank, world_size=world_size) # "gloo" for cpu "nccl" for gpu


def transformer_test(rank):
    if torch.cuda.is_available():
        backend = "nccl"
        device = f"cuda:{rank}"
    else:
        backend = "gloo"
        device = "cpu"

    setup(rank, backend, world_size)

    vec_len = 1000
    my_param = torch.nn.Parameter(torch.randn(vec_len))
    my_param_sharded = ShardedParameter(my_param)
    print("hello")
    # T = cs336_basics.model.BasicsTransformerLM(
    #     vocab_size=vocab_size,
    #     context_length=context_length,
    #     d_model = d_model,
    #     num_layers = num_layers, 
    #     num_heads = num_heads, 
    #     d_ff = d_ff
    # )
    # ShardedT = FSDP(T)

    # # pprint.pp(ShardedT)

    # in_dat = torch.randint(0, vocab_size, (batch_size, context_length))
    # res = ShardedT(in_dat)
    # # print("res:", res)

if __name__ == '__main__':
    mp.spawn(transformer_test, nprocs=world_size, join=True)