import math
from typing import Dict
import torch
from jaxtyping import Float, Int, Bool
import einops
from torch import nn, Tensor
from cs336_basics.nn_utils import softmax
from collections.abc import Callable, Iterable
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np



def old_cross_entropy(pred_logits: Float[Tensor, "... vocab_size"], 
                  targets: Int[Tensor, "..."]):

    softmaxd = softmax(pred_logits, dim=-1)
    neglogprob = -torch.log(softmaxd[targets])
    return torch.mean(neglogprob)

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
                return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps = 1e-8, weight_decay = 0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr" : lr, 
                    "beta1" : betas[0], 
                    "beta2" : betas[1], 
                    "ep" : eps, 
                    "weight_decay" : weight_decay}

        super().__init__(params, defaults)

        for p in self.state.keys():
            self.state[p]['m'] = torch.zeros(p.shape)
            self.state[p]['v'] = torch.zeros(p.shape)



    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            ep = group['ep']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.

                t = state.get("t", 1) # Get iteration number from the state, or 0.
                state["t"] = t + 1 # Increment iteration number.


                if 'm' not in state:
                    state['m'] = torch.zeros(p.shape)
                if 'v' not in state:
                    state['v'] = torch.zeros(p.shape)

                cur_rate = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # apply weight decay
                p.data = (1 - lr * weight_decay) * p.data

                # update  moments
                if 'm' not in state:
                    state['m'] = torch.zeros(p.shape)
                if 'v' not in state:
                    state['v'] = torch.zeros(p.shape)

                state['m'] = beta1 * state['m'] + (1 - beta1) * p.grad
                state['v'] = beta2 * state['v'] + (1 - beta2) * p.grad**2
                p.data = p.data - cur_rate * state['m'] / (torch.sqrt(state['v']) + ep)
                
def lr_cosine_schedule(t, lr_max, lr_min, T_warmup, T_final_anneal):
    if t < T_warmup:
        return (t / T_warmup) * lr_max
    elif t <= T_final_anneal:
        t_an = t - T_warmup
        full_time = T_final_anneal - T_warmup
        return lr_min + (1/2) * (1 + math.cos( math.pi * t_an / full_time )) * (lr_max - lr_min)
    else:
        return lr_min


# takes in numpy array of token IDs, batch_size, context_length, and a pytorch device string
# outputs an iterable of batches, each of which is a tuple of input tokens and next-token targets
def get_batch(x, batch_size, context_length, device='cpu'):
    xsize = x.shape[0]

    inds = torch.tensor(random.choices(range(xsize - context_length), k=batch_size))

    idx = torch.tensor(inds).view(-1, 1)  # Reshape to column vector: [[2], [7], [4]]

    # 3. Create the 2D index matrix using broadcasting
    # This adds j offsets (0, 1, 2) to each element in idx
    j_offsets = torch.arange(context_length).view(1, -1)  # Row vector: [[0, 1, 2]]
    in_indices = idx + j_offsets 
    out_indices = idx + j_offsets +1 

    # 4. Gather the elements from x_tensor
    in_tens = torch.tensor(x[in_indices], device=device)
    out_tens =  torch.tensor(x[out_indices], device=device)

    return in_tens, out_tens

def save_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    iteration: int, 
                    out): # pathlike
    out_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'iteration': iteration}
    torch.save(out_dict, out)

def load_checkpoint(src, # pathlike
                    model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer):
        saved = torch.load(src)
        iter = saved['iteration']
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        return iter

def main():
    runs = {}
    lrs = np.logspace(-3, 3, num=20)
    for lr in lrs:
        print(f"\n\nlr: {lr}")
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)
        losses = []
        for t in range(10):
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            losses.append(loss.item())
            print(loss.cpu().item())
            loss.backward()
            opt.step()
        runs[lr] = losses

    fig, ax = plt.subplots()
    for lr, losses in runs.items():
        ax.plot(losses, label=f"{lr}")
    ax.set_ylim(-0.2, 20)
    ax.legend()
    plt.savefig("mse training runs")
    plt.clf()

    fig, ax = plt.subplots()
    ax.plot(lrs, [runs[j][-1] for j in runs])
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig("learning rate vs final loss")
    plt.clf()




if __name__ == "__main__":
    pass


