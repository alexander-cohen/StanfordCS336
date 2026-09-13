import math
from typing import Dict
import torch
from jaxtyping import Float, Int, Bool
import einops
from torch import nn, Tensor
import cs336_basics
from cs336_basics.nn_utils import softmax
import cs336_basics.nn_utils
import cs336_basics.tokenizer
# from torch.nn import functional as Ft
# from torch.nn.functional import softmax
# from dataclasses import dataclass

def init_lin_mat(mat):
	var = 2/(mat.shape[0] + mat.shape[1])
	std = math.sqrt(var)
	trunc = 3 * std
	nn.init.trunc_normal_(mat, mean=0.0, std = std, a = -trunc, b = trunc)

class Linear(nn.Module):
	def __init__(self, 
		in_features: int, # final dimension of the input 
		out_features: int, # final dimension of the output)
		dtype: torch.dtype = torch.float32): # data type of params

		
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.dtype = dtype
		self.weights = nn.Parameter(torch.empty(out_features, in_features, dtype=self.dtype))

		init_lin_mat(self.weights)
		# var = 2/(in_features + out_features)
		# std = math.sqrt(var)
		# trunc = 3 * std
		# nn.init.trunc_normal_(self.weights, mean=0.0, std = std, a = -trunc, b = trunc)

	def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
		return einops.einsum(self.weights, x, "out_features in_features, ... in_features -> ... out_features")



class Embedding(nn.Module):
	def __init__(self, 
			  num_embeddings: Int, # size of vocabulary
			  embedding_dim: Int, # dimension of embedding vectors 
			  dtype: torch.dtype = torch.float32):
		super().__init__()

		self.num_embeddings = num_embeddings
		self.embedding_dim = embedding_dim
		self.dtype = dtype
		self.embedding_mat = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=self.dtype))
		nn.init.trunc_normal_(self.embedding_mat, mean=0.0, std = 1, a = -3, b = 3)

	def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... embedding_dim"]:
		return self.embedding_mat[token_ids]

class RMS(nn.Module):
	def __init__(self,
			  d_model: int, # hidden dimension
			  eps: float = 1e-5, # epsilon value for numerical stability of RMS
			  dtype: torch.dtype = torch.float32):

		super().__init__()
		self.d_model = d_model
		self.eps = eps
		self.gains = nn.Parameter(torch.ones(d_model, dtype = dtype))

	def forward(self, x: Float[Tensor, "batch_size sequence_len d_model"]) -> Float[Tensor, "batch_size sequence_len d_model"]:
		in_dtype = x.dtype
		x = x.to(torch.float32)

		rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) # batch_size x sequence_len x 1
		result = x * rms * self.gains

		return result.to(in_dtype)

def silu(x):
	return x * torch.sigmoid(x) 

def swigluf(x: Float[Tensor, "... d_model"], 
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
			  dtype: torch.dtype = torch.float32):

		super().__init__()
		self.d_model = d_model
		self.d_ff = math.ceil(self.d_model * 8/3 * (1/64)) * 64 if d_ff == None else d_ff
		self.W1 = nn.Parameter(torch.empty(self.d_ff, self.d_model, dtype = dtype))
		self.W2 = nn.Parameter(torch.empty(self.d_model, self.d_ff, dtype = dtype))
		self.W3 = nn.Parameter(torch.empty(self.d_ff, self.d_model, dtype = dtype))
		init_lin_mat(self.W1)
		init_lin_mat(self.W2)
		init_lin_mat(self.W3)
		

	# FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1 x) * W3 x)
	def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
		return swigluf(x, self.W1, self.W2, self.W3)

# Used ChatGPT to fix mine bc it was a bit hacky
class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
    ):
        super().__init__()

        assert d_k % 2 == 0

        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = theta ** (
            -torch.arange(0, d_k, 2, dtype=torch.float32) / d_k
        )

        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)

        self.register_buffer(
            "cos_cached",
            angles.cos(),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            angles.sin(),
            persistent=False,
        )

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:

        assert x.shape[-1] == self.d_k

        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # For example:
        #
        # x:              [batch, heads, seq, d_k]
        # token_positions:[batch, seq]
        # cos:            [batch, seq, d_k/2]
        #
        # Insert singleton dimensions before the sequence dimension
        # so cos/sin broadcast across heads.
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos

        return torch.stack(
            (out_even, out_odd),
            dim=-1,
        ).flatten(-2)

def attention(Q: Float[Tensor, "batch_size ... seq_lenQ d_k"], # queries
			  K: Float[Tensor, "batch_size ... seq_lenK d_k"], # keys
			  V: Float[Tensor, "batch_size ... seq_lenK d_v"], 
			  mask: Bool[Tensor, "seq_lenQ seq_lenK"] | None = None) -> Float[Tensor, "batch_size ... seq_lenQ d_v"]: # values

	dk = Q.shape[-1]
	mul = einops.einsum(Q, K, "... seq_lenQ d_k, ... seq_lenK d_k -> ... seq_lenQ seq_lenK") / math.sqrt(dk)

	if mask is not None:
		mul = mul + torch.where(mask, torch.tensor(0.0, device=mask.device), float("-inf"))

	return einops.einsum(softmax(mul, dim=-1), V, "... seq_lenQ seq_lenK, ... seq_lenK seq_lenV -> ... seq_lenQ seq_lenV")

class AttentionHead(nn.Module):
	def __init__(self, d_model: int, # dimension of input vector and output vector
			  num_heads: int, # number of attention heads 
			  max_seq_len: int,
			  rope: RotaryPositionEmbedding | None = None,
			  d_k: int | None = None, # dimension of learned key and query vectors
			  d_v: int | None = None): # dimension of learned value vectors

		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		self.max_seq_len = max_seq_len
		self.d_k = d_model // num_heads if d_k == None else d_k
		self.d_v = d_model // num_heads if d_v == None else d_v

		self.WQ = nn.Parameter(torch.empty(self.num_heads, self.d_k, self.d_model))
		self.WK = nn.Parameter(torch.empty(self.num_heads, self.d_k, self.d_model))
		self.WV = nn.Parameter(torch.empty(self.num_heads, self.d_v, self.d_model))
		self.WO = nn.Parameter(torch.empty(self.num_heads * self.d_v, self.d_model))

		var = 2/(self.num_heads * self.d_k + self.d_model)
		std = math.sqrt(var)
		trunc = 3 * std
		nn.init.trunc_normal_(self.WQ, mean=0.0, std = std, a = -trunc, b = trunc)
		nn.init.trunc_normal_(self.WK, mean=0.0, std = std, a = -trunc, b = trunc)
		nn.init.trunc_normal_(self.WV, mean=0.0, std = std, a = -trunc, b = trunc)
		nn.init.trunc_normal_(self.WO, mean=0.0, std = std, a = -trunc, b = trunc)
	

		init_lin_mat(self.WO)

		self.register_buffer('causal_mask', torch.tril(torch.full((self.max_seq_len, self.max_seq_len), True, dtype=torch.bool)), persistent=False)

		if rope == None:
			self.do_rope = False
		else:
			self.do_rope = True
			self.rope = rope


	def forward(self, x: Float[Tensor, "... seq_len d_model"], positions = None):
		seq_len = x.shape[-2]
		assert(seq_len <= self.max_seq_len)

		Q = einops.einsum(self.WQ, x, "h dk d_model, ... seq_len d_model -> ... h seq_len dk")
		K = einops.einsum(self.WK, x, "h dk d_model, ... seq_len d_model -> ... h seq_len dk")
		V = einops.einsum(self.WV, x, "h dv d_model, ... seq_len d_model -> ... h seq_len dv")

		if self.do_rope:
			if positions == None:
				pos = torch.arange(seq_len, device=x.device)  # Shape: [seq_len]
				positions = pos.expand(x.shape[:-1])  # Shape: [..., seq_len]
			# assert(positions != None)
			Q = self.rope(Q, positions)
			K = self.rope(K, positions)

		# dimensions ... h seq_len dv
		multi_head = attention(Q, K, V, self.causal_mask[:seq_len,:seq_len])
		multi_head_concat = einops.rearrange(multi_head, "... h seq_len dv -> ... seq_len (h dv)")
		MHA = einops.einsum(multi_head_concat, self.WO, "... seq_len hdv, hdv d_model -> ... seq_len d_model")

		return MHA

class TransformerBlock(nn.Module):
	def __init__(self, 
			  d_model: int, 
			  num_heads: int, 
			  max_seq_len: int,
			  d_ff: int, 
			  rope: RotaryPositionEmbedding | None = None):

		super().__init__()
		self.rms1 = RMS(d_model)
		self.rms2 = RMS(d_model)
		self.swiglu = SwiGLUFeedforward(d_model, d_ff=d_ff)

		if rope == None:
			self.attention = AttentionHead(d_model, num_heads, max_seq_len)
			self.do_rope = False
		else:
			self.attention = AttentionHead(d_model, num_heads, max_seq_len, rope=rope)

		

	def forward(self, x: Float[Tensor, "... seq_len d_model"]): 
		y = x + self.attention(self.rms1(x))
		return y + self.swiglu(self.rms2(y))


class Transformer(nn.Module):
	def __init__(self, 
			  vocab_size: int, 
			  context_length: int, 
			  num_layers: int, 
			  d_model: int, 
			  num_heads: int, 
			  d_ff: int, 
			  theta: float = 10000): # for rope

		super().__init__()

		self.embedding = Embedding(vocab_size, d_model)
		assert(d_model % num_heads == 0)
		self.context_length = context_length
		rope = RotaryPositionEmbedding(theta, d_model // num_heads, context_length)
		self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, context_length, d_ff, rope) for j in range(num_layers)])
		self.final_RMS = RMS(d_model)
		self.final_lin = Linear(d_model, vocab_size)

	def forward(self, in_toks: Int[Tensor, "... seq_len"]):
		x = self.embedding(in_toks)
		for b in self.blocks:
			x = b(x)

		y = self.final_lin(self.final_RMS(x))
		return y
