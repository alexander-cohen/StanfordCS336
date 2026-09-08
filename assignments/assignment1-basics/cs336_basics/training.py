import math
from typing import Dict
import torch
from jaxtyping import Float, Int, Bool
import einops
from torch import nn, Tensor
import cs336_basics.transformer

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



def main():
    pass

if __name__ == "__main__":
    main()