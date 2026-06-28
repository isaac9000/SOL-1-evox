import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10


@triton.jit
def fused_softmax_bwd_kernel(
    dP_ptr, P_ptr, mask_ptr, dS_ptr,
    scale,
    stride_b, stride_h, stride_sq,
    seq_kv,
    BLOCK_SKV: tl.constexpr,
):
    """Two-pass fused dropout-backward + softmax-backward kernel, one thread block per row."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_sq = tl.program_id(2)

    base = pid_b * stride_b + pid_h * stride_h + pid_sq * stride_sq

    # First pass: compute sum(dP * P) where dP = dP_dropped * mask * scale
    acc_sum = tl.zeros([1], dtype=tl.float32)
    for start in range(0, seq_kv, BLOCK_SKV):
        offs = start + tl.arange(0, BLOCK_SKV)
        kmask = offs < seq_kv
        dP_d = tl.load(dP_ptr + base + offs, mask=kmask, other=0.0).to(tl.float32)
        dm = tl.load(mask_ptr + base + offs, mask=kmask, other=False).to(tl.float32)
        P_val = tl.load(P_ptr + base + offs, mask=kmask, other=0.0).to(tl.float32)
        dP = dP_d * dm * scale
        acc_sum += tl.sum(dP * P_val, axis=0)

    # Second pass: compute dS = P * (dP - acc_sum)
    for start in range(0, seq_kv, BLOCK_SKV):
        offs = start + tl.arange(0, BLOCK_SKV)
        kmask = offs < seq_kv
        dP_d = tl.load(dP_ptr + base + offs, mask=kmask, other=0.0).to(tl.float32)
        dm = tl.load(mask_ptr + base + offs, mask=kmask, other=False).to(tl.float32)
        P_val = tl.load(P_ptr + base + offs, mask=kmask, other=0.0).to(tl.float32)
        dP = dP_d * dm * scale
        dS = P_val * (dP - acc_sum)
        tl.store(dS_ptr + base + offs, dS.to(tl.bfloat16), mask=kmask)


def _compute_dS_triton(dP_dropped, attn_weights, dropout_mask, attention_dropout):
    """Compute dS using Triton kernel for fused dropout backward + softmax backward."""
    bs, n_heads, seq_q, seq_kv = dP_dropped.shape
    scale = 1.0 / (1.0 - attention_dropout)
    dS = torch.empty_like(attn_weights)

    dP_dropped = dP_dropped.contiguous()
    attn_weights = attn_weights.contiguous()
    dropout_mask = dropout_mask.contiguous()

    stride_b = n_heads * seq_q * seq_kv
    stride_h = seq_q * seq_kv
    stride_sq = seq_kv

    BLOCK_SKV = max(64, triton.next_power_of_2(min(seq_kv, 4096)))

    grid = (bs, n_heads, seq_q)
    fused_softmax_bwd_kernel[grid](
        dP_dropped, attn_weights, dropout_mask, dS,
        scale,
        stride_b, stride_h, stride_sq,
        seq_kv,
        BLOCK_SKV=BLOCK_SKV,
    )
    return dS


def custom_kernel(data):
    """
    Attention backward for GQA (80 heads, 8 KV heads, 10 groups, head_dim=128).

    Key optimizations:
    - Reshape dO to [bs*8, 10*sq, 128] for dP matmul (avoids broadcast expansion)
    - Reshape for dV matmul using [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128]
    - Triton kernel for fused dropout + softmax backward
    """
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs = grad_attn_output.shape[0]
    seq_q = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, sq, 80, 128] -> [bs, 80, sq, 128]
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()  # [bs, 80, sq, 128]

    # Reshape for grouped matmul: [bs, 8, 10, sq, 128]
    dO_grouped = dO.view(bs, NUM_KEY_VALUE_HEADS, NUM_GROUPS, seq_q, HEAD_DIM)

    # --- Compute dP_dropped = dO @ V^T ---
    # Use flat batched matmul: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] -> [bs*8, 10*sq, skv]
    dO_flat = dO_grouped.reshape(bs * NUM_KEY_VALUE_HEADS, NUM_GROUPS * seq_q, HEAD_DIM)
    V_flat = value_states.reshape(bs * NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM)
    # [bs*8, 10*sq, 128] @ [bs*8, 128, skv]
    dP_flat = torch.bmm(dO_flat, V_flat.transpose(-2, -1))  # [bs*8, 10*sq, skv]
    dP_dropped = dP_flat.view(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # Fused dropout backward + softmax backward via Triton
    dS = _compute_dS_triton(dP_dropped, attn_weights, dropout_mask, attention_dropout)

    # --- Compute dV = P_dropped^T @ dO ---
    # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
    P_drop_flat = attn_weights_dropped.view(bs * NUM_KEY_VALUE_HEADS, NUM_GROUPS * seq_q, seq_kv)
    dV_flat = torch.bmm(P_drop_flat.transpose(-2, -1), dO_flat)  # [bs*8, skv, 128]
    dV = dV_flat.view(bs, NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV