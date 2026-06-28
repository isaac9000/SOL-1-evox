import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10


@torch.compile(mode="max-autotune", fullgraph=True)
def _attn_backward_compiled(dO, P, P_dropped, V, mask, p_drop):
    """
    Fused attention backward exploiting GQA structure.
    Uses bf16 matmul for speed (V not expanded to 80 heads),
    promotes to f32 only for softmax backward computation.
    dO: [bs, 80, sq, 128] bfloat16
    P, P_dropped: [bs, 80, sq, skv] bfloat16
    V: [bs, 8, skv, 128] bfloat16
    mask: [bs, 80, sq, skv] bool
    """
    bs = dO.shape[0]
    sq = dO.shape[2]
    skv = V.shape[2]

    # Reshape dO to exploit GQA: [bs, 8, 10, sq, 128]
    dO_grouped = dO.reshape(bs, NUM_KEY_VALUE_HEADS, NUM_GROUPS, sq, HEAD_DIM)

    # BMM1: dP_dropped = dO @ V^T using grouped structure (bf16 matmul - faster)
    # V: [bs, 8, skv, 128] -> [bs, 8, 1, 128, skv]
    V_t = V.transpose(-2, -1).unsqueeze(2)  # [bs, 8, 1, 128, skv]
    # dO_grouped: [bs, 8, 10, sq, 128] @ [bs, 8, 1, 128, skv] = [bs, 8, 10, sq, skv]
    dP_dropped_grouped = torch.matmul(dO_grouped, V_t)  # bf16
    # Reshape to [bs, 80, sq, skv]
    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_ATTENTION_HEADS, sq, skv)

    # Dropout backward + softmax backward in float32 for numerical stability
    scale = 1.0 / (1.0 - p_drop)
    dP = dP_dropped.float() * mask * scale

    # Softmax backward: dS = P * (dP - sum(dP * P, dim=-1, keepdim=True))
    P_f = P.float()
    dPP_sum = (dP * P_f).sum(dim=-1, keepdim=True)
    dS = P_f * (dP - dPP_sum)

    # BMM2: dV = P_dropped^T @ dO, exploiting GQA (bf16 matmul - faster)
    # P_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    P_dropped_grouped = P_dropped.reshape(bs, NUM_KEY_VALUE_HEADS, NUM_GROUPS, sq, skv)
    # [bs, 8, 10, skv, sq] @ [bs, 8, 10, sq, 128] = [bs, 8, 10, skv, 128]
    dV_grouped = torch.matmul(
        P_dropped_grouped.transpose(-2, -1),  # bf16
        dO_grouped  # bf16
    )  # bf16 output
    # Sum over groups: [bs, 8, 10, skv, 128] -> [bs, 8, skv, 128]
    dV = dV_grouped.float().sum(dim=2)

    return dS.to(torch.bfloat16), dV.to(torch.bfloat16)


def custom_kernel(data):
    """Attention backward pass with GQA structure exploitation and torch.compile max-autotune."""
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    # Transpose grad: [bs, sq, h, d] -> [bs, h, sq, d], make contiguous
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()

    return _attn_backward_compiled(
        dO, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, attention_dropout
    )