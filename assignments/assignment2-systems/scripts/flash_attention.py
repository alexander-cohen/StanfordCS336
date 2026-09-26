import triton
import triton.language as tl
import math

import torch

def cdiv(a, b):
    return (a+b-1)//b

@triton.jit
def flash_attention_forward(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_vb, stride_vs, stride_vd,
    stride_ob, stride_os, stride_od,
    stride_lb, stride_ls,
    num_queries, num_keys,
    scale: tl.constexpr,
    D: tl.constexpr,
    QUERY_TILE_SIZE: tl.constexpr, KEY_TILE_SIZE: tl.constexpr, IS_CAUSAL: tl.constexpr
):
    query_row_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    O_acc = tl.zeros((QUERY_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((QUERY_TILE_SIZE,), dtype=tl.float32)
    m = tl.zeros((QUERY_TILE_SIZE,), dtype=tl.float32)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx * stride_qb,
        shape=(num_queries, D),
        strides=(stride_qs, stride_qd),
        offsets=(query_row_idx * QUERY_TILE_SIZE, 0),
        block_shape=(QUERY_TILE_SIZE, D),
        order=(0, 1)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx * stride_kb,
        shape=(num_keys, D),
        strides=(stride_ks, stride_kd),
        offsets=(0, 0),
        block_shape=(KEY_TILE_SIZE, D),
        order=(0, 1)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx * stride_vb,
        shape=(num_keys, D),
        strides=(stride_vs, stride_vd),
        offsets=(0, 0),
        block_shape=(KEY_TILE_SIZE, D),
        order=(0, 1)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_idx * stride_ob,
        shape=(num_queries, D),
        strides=(stride_os, stride_od),
        offsets=(query_row_idx * QUERY_TILE_SIZE, 0),
        block_shape=(QUERY_TILE_SIZE, D),
        order=(0, 1)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb,
        shape=(num_queries,),
        strides=(stride_ls,),
        offsets=(query_row_idx * QUERY_TILE_SIZE,),
        block_shape=(QUERY_TILE_SIZE,),
        order=(0,)
    )

    Q_tile = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    for i in range(tl.cdiv(num_keys, KEY_TILE_SIZE)):
        K_tile = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        V_tile = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        attn_scores = tl.dot(
            Q_tile, tl.trans(K_tile), input_precision="ieee"
        ) * scale
        key_positions = i * KEY_TILE_SIZE + tl.arange(0, KEY_TILE_SIZE)
        attn_scores = tl.where(key_positions[None, :] < num_keys, attn_scores, float("-inf"))
        if IS_CAUSAL:
            query_positions = query_row_idx * QUERY_TILE_SIZE + tl.arange(0, QUERY_TILE_SIZE)
            attn_scores = tl.where(
                key_positions[None, :] <= query_positions[:, None],
                attn_scores,
                float("-inf"),
            )

        m_new = tl.maximum(m, tl.max(attn_scores, axis=1))
        P_tile = tl.exp(attn_scores - m_new[:, None])
        l = tl.exp(m - m_new) * l + tl.sum(P_tile, axis=1)
        O_acc = tl.dot(
            P_tile,
            tl.cast(V_tile, tl.float32),
            acc=tl.exp(m - m_new)[:, None] * O_acc,
            input_precision="ieee",
        )
        m = m_new

        K_block_ptr = tl.advance(K_block_ptr, (KEY_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (KEY_TILE_SIZE, 0))

    tl.store(O_block_ptr, tl.cast(O_acc / l[:, None], Q_tile.dtype), boundary_check=(0,))
    tl.store(L_block_ptr, m + tl.log(l), boundary_check=(0,))


@torch.compile
def _flash_attention_backward_math(Q, K, V, O, grad_O, L, is_causal):
    D = Q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    Q_float, K_float, V_float = Q.float(), K.float(), V.float()
    O_float, grad_O_float = O.float(), grad_O.float()
    scores = (Q_float @ K_float.transpose(-2, -1)) * scale
    if is_causal:
        query_positions = torch.arange(Q.shape[-2], device=Q.device)
        key_positions = torch.arange(K.shape[-2], device=K.device)
        scores = scores.masked_fill(
            key_positions[None, None, :] > query_positions[None, :, None],
            float("-inf"),
        )

    probabilities = torch.exp(scores - L.float().unsqueeze(-1))
    delta = torch.sum(O_float * grad_O_float, dim=-1)
    grad_V = probabilities.transpose(-2, -1) @ grad_O_float
    grad_P = grad_O_float @ V_float.transpose(-2, -1)
    grad_S = probabilities * (grad_P - delta.unsqueeze(-1))
    grad_Q = (grad_S @ K_float) * scale
    grad_K = (grad_S.transpose(-2, -1) @ Q_float) * scale
    return grad_Q.to(Q.dtype), grad_K.to(K.dtype), grad_V.to(V.dtype)


class PytorchFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, B_q, B_k, is_causal = False):
        """
        Q is (batch, num_q, d); K and V are (batch, num_k, d).

        S = QK^T / sqrt(d) is (num_q, num_k) mat
        P_{ij} = softmax_j S_{ij} is (num_q, num_k) mat
        O = PV is (num_q, d) mat
        L = proxy for denominator
        return O, L

        B_q , B_k are tile sizes for queries and keys
        """
        batch_size, N_q, d = Q.shape
        N_k = K.shape[1]


        T_q = cdiv(N_q, B_q)
        T_k = cdiv(N_k, B_k)
        scale = 1.0 / math.sqrt(d)

        O = torch.zeros_like(Q)
        L = torch.zeros((batch_size, N_q), device=Q.device)

        for b in range(batch_size):
            for i in range(T_q):
                q_start, q_end = B_q * i, B_q * (i + 1)
                Q_tile = Q[b, q_start:q_end, :]
                height = Q_tile.shape[0]
                O_acc = torch.zeros(height, d, device=Q.device, dtype=torch.float32)
                l_tile = torch.zeros(height, device=Q.device, dtype=torch.float32)
                m_tile = torch.full((height,), float("-inf"), device=Q.device, dtype=torch.float32)

                for j in range(T_k):
                    k_start, k_end = B_k * j, B_k * (j + 1)
                    K_tile = K[b, k_start:k_end, :]
                    V_tile = V[b, k_start:k_end, :]
                    attn_scores = (Q_tile.float() @ K_tile.float().T) * scale
                    if is_causal:
                        query_positions = q_start + torch.arange(height, device=Q.device)
                        key_positions = k_start + torch.arange(K_tile.shape[0], device=Q.device)
                        attn_scores = attn_scores.masked_fill(
                            key_positions[None, :] > query_positions[:, None],
                            float("-inf"),
                        )

                    m_new = torch.maximum(m_tile, torch.amax(attn_scores, dim=-1))
                    P_tile = torch.exp(attn_scores - m_new[:, None])
                    l_tile = torch.exp(m_tile - m_new) * l_tile + torch.sum(P_tile, dim=-1)
                    O_acc = torch.exp(m_tile - m_new)[:, None] * O_acc + P_tile @ V_tile.float()
                    m_tile = m_new

                O[b, q_start:q_end, :] = O_acc / l_tile[:, None]
                L[b, q_start:q_end] = m_tile + torch.log(l_tile)
        
        return O, L

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, B_q, B_k, is_causal = False):
        """
        Q is (batch, num_q, d); K and V are (batch, num_k, d).

        S = QK^T / sqrt(d) is (num_q, num_k) mat
        P_{ij} = softmax_j S_{ij} is (num_q, num_k) mat
        O = PV is (num_q, d) mat
        L = proxy for denominator
        return O, L

        B_q , B_k are tile sizes for queries and keys
        """
        batch_size, N_q, D = Q.shape
        N_k = K.shape[1]


        T_q = cdiv(N_q, B_q)
        T_k = cdiv(N_k, B_k)
        scale = 1.0 / math.sqrt(D)

        O = torch.zeros_like(Q)
        L = torch.zeros((batch_size, N_q), device=Q.device)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.QUERIES_TILE_SIZE = B_q
        ctx.KEYS_TILE_SIZE = B_k
        ctx.IS_CAUSAL = is_causal

        flash_attention_forward[(T_q, batch_size)](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k,
            scale,
            D,
            B_q, B_k, is_causal
        )
        return O, L


    @staticmethod
    def backward(ctx, grad_O, grad_L):
        Q, K, V, O, L = ctx.saved_tensors
        grad_Q, grad_K, grad_V = _flash_attention_backward_math(
            Q, K, V, O, grad_O, L, ctx.IS_CAUSAL
        )
        return grad_Q, grad_K, grad_V, None, None, None



pytorch_flash_attn = PytorchFlashAttention.apply
triton_flash_attn = TritonFlashAttention.apply

if __name__ == "__main__":
    n_q = 5
    n_k = 16
    d = 16
    batch_size = 3
    Q = torch.randn(batch_size, n_q, d, device="cuda")
    K = torch.randn(batch_size, n_k, d, device="cuda")
    V = torch.randn(batch_size, n_k, d, device="cuda")
    B_q = 2
    B_k = 16
    is_causal = True
    O1, L1 = pytorch_flash_attn(Q, K, V, B_q, B_k, is_causal)
    print("O1:", O1)
    print("L1:", L1)

    O2, L2 = triton_flash_attn(Q, K, V, B_q, B_k, is_causal)
    print("O2:", O2)
    print("L2:", L2)

