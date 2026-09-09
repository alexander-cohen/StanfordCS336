import math
from typing import Dict
import torch
from jaxtyping import Float, Int, Bool
import einops
from torch import nn, Tensor
import cs336_basics.transformer
from collections.abc import Callable, Iterable
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np



def cross_entropy(pred_logits: Float[Tensor, "... vocab_size"], 
                  targets: Int[Tensor, "..."]):

    # -log(exp(o_i) / sum_j exp(o_j)) = -o_i + log(sum_j exp(o_j))
    pred_logits = pred_logits.to(torch.float32)
    max_val = torch.amax(pred_logits, dim=-1, keepdim=True)
    subtracted = pred_logits - max_val
    summed = torch.sum(torch.exp(subtracted), dim=-1, keepdim=True)
    neglogprob = -subtracted.gather(-1, targets[..., None]) + torch.log(summed)
    return torch.mean(neglogprob)

def old_cross_entropy(pred_logits: Float[Tensor, "... vocab_size"], 
                  targets: Int[Tensor, "..."]):

    softmaxd = cs336_basics.transformer.softmax(pred_logits, dim=-1)
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

        print("\nParams:\n", list(params))
        print("Init state dict:", self.state)
        for p in self.state.keys():
            self.state[p]['m'] = torch.zeros(p.shape)
            self.state[p]['v'] = torch.zeros(p.shape)

        print("Init'd state dict:", self.state)


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


# def _optimize(opt_class) -> torch.Tensor:
#     torch.manual_seed(42)
#     model = torch.nn.Linear(3, 2, bias=False)
#     opt = opt_class(
#         model.parameters(),
#         lr=1e-3,
#         weight_decay=0.01,
#         betas=(0.9, 0.999),
#         eps=1e-8,
#     )
#     # Use 1000 optimization steps for testing
#     for _ in range(1000):
#         opt.zero_grad()
#         x = torch.rand(model.in_features)
#         y_hat = model(x)
#         y = torch.tensor([x[0] + x[1], -x[2]])
#         loss = ((y - y_hat) ** 2).sum()
#         loss.backward()
#         opt.step()
#     return model.weight.detach()


# def test_adamw(numpy_snapshot):
#     """
#     Our reference implementation yields slightly different results than the
#     PyTorch AdamW, since there are a couple different ways that you can apply
#     weight decay that are equivalent in principle, but differ in practice due to
#     floating point behavior. So, we test that the provided implementation matches
#     _either_ our reference implementation's expected results or those from the PyTorch AdamW.
#     """
#     # expected_weights = torch.load(FIXTURES_PATH / "adamw_expected_params.pt")
#     pytorch_weights = _optimize(torch.optim.AdamW)
#     actual_weights = _optimize(AdamW)

#     # Might need to exit early if the weights match pytorch, since that should also be valid
#     matches_pytorch = torch.allclose(actual_weights, pytorch_weights, atol=1e-4)
#     if matches_pytorch:
#         return

#     numpy_snapshot.assert_match(
#         actual_weights,
#         atol=1e-4,
#     )



if __name__ == "__main__":
    pass