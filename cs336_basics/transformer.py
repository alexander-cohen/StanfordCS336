import math
from typing import Dict
import torch
from jaxtyping import Float, Int
import einops
from torch import nn, Tensor
# from torch.nn import functional as Ft
# from torch.nn.functional import softmax
# from dataclasses import dataclass

class Linear(nn.Module):
	def __init__(self, 
		in_features: int, # final dimension of the input 
		out_features: int, # final dimension of the output)
		device: torch.device | None = None, # device to store params on, 
		dtype: torch.dtype | None = None): # data type of params

		
		super().__init__()
		self.dtype = dtype if dtype != None else torch.float32
		self.in_features = in_features
		self.out_features = out_features
		self.weights = nn.Parameter(torch.empty(out_features, in_features, dtype=self.dtype))

		var = 2/(in_features + out_features)
		std = math.sqrt(var)
		trunc = 3 * std
		nn.init.trunc_normal_(self.weights, mean=0.0, std = std, a = -trunc, b = trunc)

	def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
		return einops.einsum(self.weights, x, "out_features in_features, ... in_features -> ... out_features")



