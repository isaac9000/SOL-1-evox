import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10


@triton.jit
def fused_softmax_bwd_kernel(
    dP_dropped_ptr, P_ptr, mask_ptr, dS_ptr,
    sq, skv, scale,
    stride_b, stride_h, stride_sq, stride_skv,
    BLOCK_SKV: tl.constexpr,
):
    """Fused dropout backward + softmax backward kernel."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_sq = tl.program_id(2)

    base = pid_b * stride_b + pid_h * stride_h + pid_sq * stride_sq

    # Process one row of skv
    acc_sum = tl.zeros([1], dtype=tl.float32)
    
    # First pass: compute sum(dP * P)
    for start in range(0, skv, BLOCK_SKV):
        offs = start + tl.arange(0, BLOCK_SKV)
        mask_offs = offs < skv
        
        dP_d = tl.load(dP_dropped_ptr + base + offs, mask=mask_offs, other=0.0).to(tl.float32)
        m = tl.load(mask_ptr + base + offs, mask=mask_offs, other=0).to(tl.float32)
        P_val = tl.load(P_ptr + base + offs, mask=mask_offs, other=0.0).to(tl.float32)
        
        dP = dP_d * m * scale
        acc_sum += tl.sum(dP * P_val, axis=0)
    
    # Second pass: compute dS = P * (dP - sum)
    for start in range(0, skv, BLOCK_SKV):
        offs = start + tl.arange(0, BLOCK_SKV)
        mask_offs = offs < skv
        
        dP_d = tl.load(dP_dropped_ptr + base + offs, mask=mask_offs, other=0.0).to(tl.float32)
        m = tl.load(mask_ptr + base + offs, mask=mask_offs, other=0).to(tl.float32)
        P_val = tl.load(P_ptr + base + offs, mask=mask_offs, other=0.0).to(tl.float32)
        
        dP = dP_d * m * scale
        dS = P_val * (dP - acc_sum)
        
        tl.store(dS_ptr + base + offs, dS.to(tl.bfloat16), mask=mask_offs)


@torch.compile(mode="reduce-overhead", fullgraph=True)
def _attn_backward_compiled(dO, P, P_dropped, V, mask, p_drop):
    """
    Fused attention backward using GQA structure.
    Avoids expanding V to 80 heads by using grouped matmul.
    Uses bfloat16 matmul for speed, float32 only for softmax backward.
    dO: [bs, 80, sq, 128] bfloat16
    """
    bs = dO.shape[0]
    sq = dO.shape[2]
    skv = V.shape[2]

    # Reshape dO to exploit GQA: [bs, 8, 10, sq, 128]
    dO_grouped = dO.reshape(bs, NUM_KEY_VALUE_HEADS, NUM_GROUPS, sq, HEAD_DIM)

    # BMM1: dP_dropped = dO @ V^T using grouped structure
    # V: [bs, 8, skv, 128] -> [bs, 8, 1, 128, skv]
    V_t = V.transpose(-2, -1).unsqueeze(2)  # [bs, 8, 1, 128, skv]
    # dO_grouped @ V_t: [bs, 8, 10, sq, skv] - keep in bfloat16
    dP_dropped_grouped = torch.matmul(dO_grouped, V_t)
    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_ATTENTION_HEADS, sq, skv)

    # Dropout backward + softmax backward (fused in float32)
    scale = 1.0 / (1.0 - p_drop)
    dP = dP_dropped.float() * mask * scale
    P_f = P.float()
    dS = P_f * (dP - (dP * P_f).sum(dim=-1, keepdim=True))

    # BMM2: dV = P_dropped^T @ dO, exploiting GQA
    P_dropped_grouped = P_dropped.reshape(bs, NUM_KEY_VALUE_HEADS, NUM_GROUPS, sq, skv)
    # [bs, 8, 10, skv, sq] @ [bs, 8, 10, sq, 128] = [bs, 8, 10, skv, 128]
    dV_grouped = torch.matmul(
        P_dropped_grouped.float().transpose(-2, -1),
        dO_grouped.float()
    )
    dV = dV_grouped.sum(dim=2)

    return dS.to(torch.bfloat16), dV.to(torch.bfloat16)


def custom_kernel(data):
    """
    Attention backward pass with GQA structure exploitation and torch.compile fusion.
    Uses grouped matmul to avoid expanding value_states to full 80 heads.
    """
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    # Transpose grad: [bs, sq, 80, 128] -> [bs, 80, sq, 128], make contiguous
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()

    return _attn_backward_compiled(
        dO, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, attention_dropout
    )