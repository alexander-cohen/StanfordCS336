# import triton
# import triton.language as tl
import math

import torch

def cdiv(a, b):
    return (a+b-1)//b

# @triton.jit
def flash_attention_forward():
    pass


class PytorchFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, B_q, B_k, is_causal = False):
        """
        Q is (num_q, d) matrix
        K, V are (num_k, d) matrices

        S = QK^T / sqrt(d) is (num_q, num_k) mat
        P_{ij} = softmax_j S_{ij} is (num_q, num_k) mat
        O = PV is (num_q, d) mat
        L = proxy for denominator
        return O, L

        B_q , B_k are tile sizes for queries and keys
        """
        N_q = Q.shape[0]
        d = Q.shape[1]
        N_k = K.shape[0]


        T_q = cdiv(N_q, B_q)
        T_k = cdiv(N_k, B_k)

        O = torch.zeros_like(Q)
        L = torch.zeros(N_q, device = Q.device, dtype = Q.dtype)

        for i in range(T_q):
            Q_batch = Q[(B_q*i):(B_q*(i+1)),:]
            height = Q_batch.shape[0]
            O_batch = torch.zeros(height, d, device = Q.device, dtype = Q.dtype)
            l_batch = torch.zeros(height, device = Q.device, dtype = Q.dtype)
            m_batch = torch.full((height,), float("-inf"), device = Q.device, dtype = Q.dtype)
            for j in range(T_k):
                K_batch = K[(B_k*j):(B_k*(j+1)),:]
                V_batch = V[(B_k*j):(B_k*(j+1)),:]
                attn_scores = Q_batch @ K_batch.T / math.sqrt(d) # B_q x B_k

                m_batch_new = torch.max(m_batch, torch.amax(attn_scores, dim = -1)) # running max over each row in Q_batch
                P_batch = torch.exp(attn_scores - m_batch_new[:,None]) # B_q x B_k
                l_batch = torch.exp(m_batch - m_batch_new) * l_batch + torch.sum(P_batch, -1)
                O_batch = torch.exp(m_batch - m_batch_new)[:,None] * O_batch + P_batch @ V_batch
                m_batch = m_batch_new

            O_batch = torch.reciprocal(l_batch[:,None]) * O_batch
            L_batch = m_batch + torch.log(l_batch)

            O[(B_q*i):(B_q*(i+1)),:] = O_batch
            L[(B_q*i):(B_q*(i+1))] = L_batch
        
        return O, L

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


pytorch_flash_attn = PytorchFlashAttention.apply

n_q = 5
n_k = 3
d = 4
Q = torch.randn(n_q, d)
K = torch.randn(n_k, d)
V = torch.randn(n_k, d)
B_q = 2
B_k = 2
O, L = pytorch_flash_attn(Q, K, V, B_q, B_k)
print("O:", O) 
print("L:", L)

print("O pytorch:", torch.nn.functional.softmax(Q @ K.T / d**0.5, dim = -1) @ V)
