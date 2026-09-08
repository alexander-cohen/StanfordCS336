import math
from typing import Dict
import torch
from jaxtyping import Float, Int
import einops
from torch import nn, Tensor
# from torch.nn import functional as Ft
# from torch.nn.functional import softmax
# from dataclasses import dataclass

## TODO: Implement device in modules

class Linear(nn.Module):
	def __init__(self, 
		in_features: int, # final dimension of the input 
		out_features: int, # final dimension of the output)
		device: torch.device | None = None, # device to store params on, 
		dtype: torch.dtype = torch.float32): # data type of params

		
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weights = nn.Parameter(torch.empty(out_features, in_features, dtype=self.dtype))

		var = 2/(in_features + out_features)
		std = math.sqrt(var)
		trunc = 3 * std
		nn.init.trunc_normal_(self.weights, mean=0.0, std = std, a = -trunc, b = trunc)

	def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
		return einops.einsum(self.weights, x, "out_features in_features, ... in_features -> ... out_features")



class Embedding(nn.Module):
	def __init__(self, 
			  num_embeddings: Int, # size of vocabulary
			  embedding_dim: Int, # dimension of embedding vectors 
			  device: torch.device | None = None, # device to store params on
			  dtype: torch.dtype = torch.float32):
		super().__init__()

		self.num_embeddings = num_embeddings
		self.embeding_dim = embedding_dim
		self.embedding_mat = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=self.dtype))
		nn.init.trunc_normal_(self.embedding_mat, mean=0.0, std = 1, a = -3, b = 3)

	def forward(self, token_ids: Int[Tensor, "... num_embeddings"]) -> Float[Tensor, "... embedding_dim"]:
		return self.embedding_mat[token_ids]

class RMS(nn.Module):
	def __init__(self,
			  d_model: int, # hidden dimension
			  eps: float = 1e-5, # epsilon value for numerical stability of RMS
			  device: torch.device | None = None, 
			  dtype: torch.dtype = torch.float32):

		super().__init__()
		self.d_model = d_model
		self.eps = eps
		self.gains = nn.Parameter(torch.ones(d_model, dtype = self.dtype))

	def forward(self, x: Float[Tensor, "batch_size sequence_len d_model"]) -> Float[Tensor, "batch_size sequence_len d_model"]:
		in_dtype = x.dtype
		x = x.to(torch.float32)

		rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) # batch_size x sequence_len x 1
		result = x * rms * self.gains

		return result.to(in_dtype)

def silu(x):
	return x * torch.sigmoid(x) 

def swiglu(x: Float[Tensor, "... d_model"], 
		   W1: Float[Tensor, "d_ff d_model"],
		   W2: Float[Tensor, "d_model d_ff"], 
		   W3: Float[Tensor, "d_ff d_model"]) -> Float[Tensor, "... d_model"]:
	
	return einops.einsum(silu(einops.einsum(W1, x, "d_ff d_model, ... d_model -> ... d_ff")) * 
								einops.einsum(W3, x, "d_ff d_model, ... d_model -> ... d_ff"), W2, 
								"... d_ff, d_model d_ff -> ... d_model")

class SwiGLUFeedforward(nn.Module):
	def __init__(self, 
			  d_model: int, # hidden dimension
			  d_ff: int | None = None, # intermediate dimension
			  device: torch.device | None = None, 
			  dtype: torch.dtype = torch.float32):

		super().__init__()
		self.d_model = d_model
		self.d_ff = math.ceil(self.d_model * 8/3 * (1/64)) * 64 if d_ff == None else d_ff
		self.W1 = nn.Parameter(torch.empty(self.d_ff, self.d_model, dtype = dtype))
		self.W2 = nn.Parameter(torch.empty(self.d_model, self.d_ff, dtype = dtype))
		self.W3 = nn.Parameter(torch.empty(self.d_ff, self.d_model, dtype = dtype))

	# FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1 x) * W3 x)
	def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
		return swiglu(x, self.W1, self.W2, self.W3)

class RotaryPositionEmbedding(nn.Module):
	def __init__(self, 
			  theta: float, # theta value for the RoPE
			  d_k: int, # dimension of query and key vectors
			  max_seq_len: int, # maximum sequence length that might be input
			  device: torch.device | None = None):

		super().__init__()

		self.max_seq_len = max_seq_len
		self.d_k = d_k
		assert(d_k % 2 == 0)

		inv_freq = 1.0 / (theta ** (torch.arange(0, self.d_k, 2) / self.d_k))

		# a max_seq_len x d_k/2 matrix with i, k entry equal to i / Theta^{(2k-2)/d_k}
		emb = torch.outer(torch.arange(max_seq_len, dtype = torch.float32), inv_freq)

		self.register_buffer('cos_cached', emb.cos(), persistent=False)
		self.register_buffer('sin_cached', emb.sin(), persistent=False)


	# for k in {0,...,d/2-1}
	# out[..., i, 2k] = x[...,i,2k] * cos_cached[i,k] - x[...,i,2k+1] * sin_cached[i,k]
	# out[..., i, 2k+1] = x[...,i,2k] * sin_cached[i,k] + x[...,i,2k+1] * cos_cached[i,k]
	def forward(self, 
			 x: Float[Tensor, "... seq_len d_k"], 
			 token_positions: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len d_k"]:

		even_cols = x[...,0::2]
		odd_cols = x[...,1::2]
		print(f"dk: {self.d_k}\nx shape: {x.shape}\nEven cols shape: {even_cols.shape}\nOdd cols shape: {odd_cols.shape}\nToken positions shape: {token_positions.shape}\nCos cached mat shape: {self.cos_cached[token_positions].shape}")
		even_outs = (einops.einsum(even_cols, self.cos_cached[token_positions], "... seq_len hdk, ... seq_len hdk -> ... seq_len hdk") + 
					einops.einsum(odd_cols, -self.sin_cached[token_positions], "... seq_len hdk, ... seq_len hdk -> ... seq_len hdk"))
		odd_outs = (einops.einsum(even_cols, self.sin_cached[token_positions], "... seq_len hdk, ... seq_len hdk -> ... seq_len hdk") +  
					einops.einsum(odd_cols, self.cos_cached[token_positions], "... seq_len hdk, ... seq_len hdk -> ... seq_len hdk"))
		print("even odd outs")
		interleaved = einops.rearrange([even_outs, odd_outs], "two ... seq_len hdk -> ... seq_len (hdk two)")
		return interleaved