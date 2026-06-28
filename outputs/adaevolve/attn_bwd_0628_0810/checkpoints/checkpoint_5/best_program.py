import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10


@torch.compile(fullgraph=True, dynamic=True)
def _attn_backward(grad_attn_output, attn_weights, attn_weights_dropped,
                   value_states, dropout_mask, attention_dropout):
    """
    Optimized attention backward for GQA (80 heads, 8 KV heads, 10 groups).
    Uses bf16 matmuls for speed on B200 tensor cores.
    Reshapes dO to [bs, 8, 10, sq, 128] to avoid expanding value_states.
    Float32 only for softmax backward accumulation.
    """
    bs = grad_attn_output.shape[0]
    seq_q = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, sq, 80, 128] -> [bs, 80, sq, 128] -> [bs, 8, 10, sq, 128]
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()  # [bs,80,sq,128] bf16
    dO_grouped = dO.view(bs, NUM_KEY_VALUE_HEADS, NUM_GROUPS, seq_q, HEAD_DIM)

    # --- Compute dP_dropped = dO @ V^T using bf16 matmul ---
    # dO_grouped: [bs, 8, 10, sq, 128] bf16
    # value_states: [bs, 8, skv, 128] bf16
    # V^T: [bs, 8, 1, 128, skv] -> broadcast over groups
    # Result: [bs, 8, 10, sq, skv] -> [bs, 80, sq, skv]
    dP_dropped_bf16 = torch.matmul(
        dO_grouped,
        value_states.unsqueeze(2).transpose(-2, -1)
    )  # [bs, 8, 10, sq, skv] bf16
    dP_dropped = dP_dropped_bf16.view(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv).float()

    # --- Dropout backward ---
    scale = 1.0 / (1.0 - attention_dropout)
    dP = dP_dropped * dropout_mask * scale

    # --- Softmax backward: dS = P * (dP - sum(dP * P, dim=-1)) ---
    P = attn_weights.float()
    dPP_sum = (dP * P).sum(dim=-1, keepdim=True)
    dS = (P * (dP - dPP_sum)).to(torch.bfloat16)

    # --- Compute dV = P_dropped^T @ dO using bf16 matmul ---
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    P_drop = attn_weights_dropped.view(bs, NUM_KEY_VALUE_HEADS, NUM_GROUPS, seq_q, seq_kv)
    # [bs,8,10,skv,sq] @ [bs,8,10,sq,128] -> [bs,8,10,skv,128]
    dV_grouped = torch.matmul(P_drop.transpose(-2, -1), dO_grouped)
    # Sum over groups: [bs,8,10,skv,128] -> [bs,8,skv,128]
    dV = dV_grouped.sum(dim=2).to(torch.bfloat16)

    return dS, dV


def custom_kernel(data):
    """Entry point for attention backward pass with GQA support."""
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data
    return _attn_backward(grad_attn_output, attn_weights, attn_weights_dropped,
                          value_states, dropout_mask, attention_dropout)