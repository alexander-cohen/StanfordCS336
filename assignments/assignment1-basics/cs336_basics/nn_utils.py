import torch
from jaxtyping import Float, Int
from torch import Tensor
import math


def softmax(tens: torch.tensor, dim: int) -> torch.tensor:
	maxval = torch.amax(tens, dim, keepdim=True)
	tensexp = torch.exp(tens - maxval)
	tensexpsum = torch.sum(tensexp, dim, keepdim=True)
	return tensexp / tensexpsum


def cross_entropy(pred_logits: Float[Tensor, "... vocab_size"],
                  targets: Int[Tensor, "..."]):

    # -log(exp(o_i) / sum_j exp(o_j)) = -o_i + log(sum_j exp(o_j))
    pred_logits = pred_logits.to(torch.float32)
    max_val = torch.amax(pred_logits, dim=-1, keepdim=True)
    subtracted = pred_logits - max_val
    summed = torch.sum(torch.exp(subtracted), dim=-1, keepdim=True)
    neglogprob = -subtracted.gather(-1, targets[..., None]) + torch.log(summed)
    return torch.mean(neglogprob)

def gradient_clipping(params, M, eps = 1e-6):
    norm2 = 0
    param_list = list(params)
    for p in param_list:
        if p.grad is not None:
            norm2 += torch.linalg.norm(p.grad)**2

    norm = math.sqrt(norm2)
    if norm > M:
        for p in param_list:
            with torch.no_grad():
                if p.grad is not None:
                    grad_norm = torch.linalg.norm(p.grad)
                    p.grad.mul_(M / (norm + eps))