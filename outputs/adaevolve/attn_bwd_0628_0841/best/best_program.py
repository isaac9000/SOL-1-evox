import torch
from torch.utils.cpp_extension import load_inline

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
NUM_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>

// Warp-level reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Fused dropout backward + softmax backward kernel
// Multiple rows per block, each warp handles one row
__global__ void fused_ds_kernel(
    const __nv_bfloat16* __restrict__ dP_dropped,
    const __nv_bfloat16* __restrict__ P,
    const bool* __restrict__ mask,
    __nv_bfloat16* __restrict__ dS,
    int skv,
    float scale,
    int n_rows
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_rows) return;
    int lane = threadIdx.x;
    
    const __nv_bfloat16* dP_row = dP_dropped + row * skv;
    const __nv_bfloat16* P_row = P + row * skv;
    const bool* mask_row = mask + row * skv;
    __nv_bfloat16* dS_row = dS + row * skv;
    
    // Pass 1: compute sum(dP * P)
    float local_sum = 0.0f;
    for (int i = lane; i < skv; i += 32) {
        float dp_d = __bfloat162float(dP_row[i]);
        float p_val = __bfloat162float(P_row[i]);
        float dp = mask_row[i] ? dp_d * scale : 0.0f;
        local_sum += dp * p_val;
    }
    local_sum = warp_reduce_sum(local_sum);
    float sum_dPP = local_sum; // lane 0 has the result, broadcast via shfl
    sum_dPP = __shfl_sync(0xffffffff, sum_dPP, 0);
    
    // Pass 2: compute dS = P * (dP - sum_dPP)
    for (int i = lane; i < skv; i += 32) {
        float dp_d = __bfloat162float(dP_row[i]);
        float p_val = __bfloat162float(P_row[i]);
        float dp = mask_row[i] ? dp_d * scale : 0.0f;
        float ds = p_val * (dp - sum_dPP);
        dS_row[i] = __float2bfloat16(ds);
    }
}

// Multi-warp version for larger skv
__global__ void fused_ds_kernel_mw(
    const __nv_bfloat16* __restrict__ dP_dropped,
    const __nv_bfloat16* __restrict__ P,
    const bool* __restrict__ mask,
    __nv_bfloat16* __restrict__ dS,
    int skv,
    float scale,
    int n_rows
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int lane = threadIdx.x;
    int warp_id = threadIdx.y;
    int num_warps = blockDim.y;
    
    const __nv_bfloat16* dP_row = dP_dropped + row * skv;
    const __nv_bfloat16* P_row = P + row * skv;
    const bool* mask_row = mask + row * skv;
    __nv_bfloat16* dS_row = dS + row * skv;
    
    extern __shared__ float smem[];
    
    float local_sum = 0.0f;
    int stride = 32 * num_warps;
    int start = warp_id * 32 + lane;
    for (int i = start; i < skv; i += stride) {
        float dp_d = __bfloat162float(dP_row[i]);
        float p_val = __bfloat162float(P_row[i]);
        float dp = mask_row[i] ? dp_d * scale : 0.0f;
        local_sum += dp * p_val;
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) smem[warp_id] = local_sum;
    __syncthreads();
    
    float total_sum = 0.0f;
    for (int w = 0; w < num_warps; w++) total_sum += smem[w];
    
    for (int i = start; i < skv; i += stride) {
        float dp_d = __bfloat162float(dP_row[i]);
        float p_val = __bfloat162float(P_row[i]);
        float dp = mask_row[i] ? dp_d * scale : 0.0f;
        float ds = p_val * (dp - total_sum);
        dS_row[i] = __float2bfloat16(ds);
    }
}

static cublasHandle_t cublas_handle = nullptr;

void ensure_cublas() {
    if (cublas_handle == nullptr) {
        cublasCreate(&cublas_handle);
        cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    }
}

std::vector<torch::Tensor> attn_bwd_cuda(
    torch::Tensor dO,         // [bs, 80, sq, 128] bf16
    torch::Tensor P,          // [bs, 80, sq, skv] bf16
    torch::Tensor P_dropped,  // [bs, 80, sq, skv] bf16
    torch::Tensor V,          // [bs, 8, skv, 128] bf16
    torch::Tensor mask,       // [bs, 80, sq, skv] bool
    float p_drop
) {
    ensure_cublas();
    auto stream = at::cuda::getCurrentCUDAStream();
    cublasSetStream(cublas_handle, stream);
    
    int bs = dO.size(0);
    int sq = dO.size(2);
    int skv = V.size(2);
    
    // BMM1: dP = dO @ V^T
    // Use strided batched GEMM with stride=0 for V to handle GQA
    // dO: [bs, 8, 10, sq, 128] - stride between groups = sq*128
    // V^T: [bs, 8, 128, skv] - stride between groups = 0 (same V for all groups)
    // dP: [bs, 8, 10, sq, skv]
    
    auto dP_dropped_out = torch::empty({bs, NUM_ATTENTION_HEADS, sq, skv}, dO.options());
    
    // We'll use cuBLAS strided batched GEMM
    // C = alpha * A * B + beta * C
    // For each (b, kv_head, group): dP[b,kv,g,:,:] = dO[b,kv,g,:,:] @ V[b,kv,:,:]^T
    // A = dO_grouped: [sq, 128] with stride sq*128 per group, (bs*8*10)*sq*128 total
    // B = V^T: [128, skv] with stride 0 per group (same KV for all groups), stride skv*128 per kv_head
    // This is tricky with standard strided batched; use PyTorch matmul with broadcast
    
    auto dO_grouped = dO.reshape({bs, NUM_KEY_VALUE_HEADS, NUM_GROUPS, sq, HEAD_DIM});
    auto V_t = V.transpose(-2, -1).unsqueeze(2); // [bs, 8, 1, 128, skv]
    auto dP_grouped = torch::matmul(dO_grouped, V_t); // [bs, 8, 10, sq, skv]
    dP_dropped_out = dP_grouped.reshape({bs, NUM_ATTENTION_HEADS, sq, skv});
    
    // Fused dropout backward + softmax backward
    auto dS = torch::empty({bs, NUM_ATTENTION_HEADS, sq, skv}, dO.options());
    float scale = 1.0f / (1.0f - p_drop);
    int n_rows = bs * NUM_ATTENTION_HEADS * sq;
    
    auto dP_cont = dP_dropped_out.contiguous();
    auto P_cont = P.contiguous();
    auto mask_cont = mask.contiguous();
    
    if (skv <= 512) {
        // One warp per row, multiple rows per block
        int rows_per_block = 8;
        dim3 block(32, rows_per_block);
        dim3 grid((n_rows + rows_per_block - 1) / rows_per_block);
        fused_ds_kernel<<<grid, block, 0, stream>>>(
            (__nv_bfloat16*)dP_cont.data_ptr(),
            (__nv_bfloat16*)P_cont.data_ptr(),
            (bool*)mask_cont.data_ptr(),
            (__nv_bfloat16*)dS.data_ptr(),
            skv, scale, n_rows
        );
    } else {
        int num_warps = std::min(16, (skv + 255) / 256);
        dim3 block(32, num_warps);
        dim3 grid(n_rows);
        int smem = num_warps * sizeof(float);
        fused_ds_kernel_mw<<<grid, block, smem, stream>>>(
            (__nv_bfloat16*)dP_cont.data_ptr(),
            (__nv_bfloat16*)P_cont.data_ptr(),
            (bool*)mask_cont.data_ptr(),
            (__nv_bfloat16*)dS.data_ptr(),
            skv, scale, n_rows
        );
    }
    
    // BMM2: dV = P_dropped^T @ dO, sum over groups
    auto P_dropped_grouped = P_dropped.reshape({bs, NUM_KEY_VALUE_HEADS, NUM_GROUPS, sq, skv});
    auto dV_grouped = torch::matmul(
        P_dropped_grouped.transpose(-2, -1),
        dO_grouped
    ); // [bs, 8, 10, skv, 128] bf16
    
    // Sum over groups
    auto dV = dV_grouped.to(torch::kFloat32).sum(2).to(torch::kBFloat16);
    
    return {dS, dV};
}
"""

_cpp_source = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> attn_bwd_cuda(
    torch::Tensor dO,
    torch::Tensor P,
    torch::Tensor P_dropped,
    torch::Tensor V,
    torch::Tensor mask,
    float p_drop
);

std::vector<torch::Tensor> attn_bwd(
    torch::Tensor dO,
    torch::Tensor P,
    torch::Tensor P_dropped,
    torch::Tensor V,
    torch::Tensor mask,
    float p_drop
) {
    return attn_bwd_cuda(dO, P, P_dropped, V, mask, p_drop);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_bwd", &attn_bwd, "Attention backward pass");
}
"""

try:
    _ext = load_inline(
        name="attn_bwd_ext_v2",
        cpp_sources=_cpp_source,
        cuda_sources=_cuda_source,
        functions=["attn_bwd"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_90a"],
        extra_cflags=["-O3"],
        verbose=False,
    )
    _USE_CUDA = True
except Exception as e:
    print(f"Failed to load CUDA extension: {e}")
    _USE_CUDA = False


@torch.compile(mode="max-autotune", fullgraph=True)
def _attn_backward_fallback(dO, P, P_dropped, V, mask, p_drop):
    """Fallback compiled PyTorch implementation."""
    bs = dO.shape[0]
    sq = dO.shape[2]
    skv = V.shape[2]

    dO_grouped = dO.reshape(bs, NUM_KEY_VALUE_HEADS, NUM_GROUPS, sq, HEAD_DIM)
    V_t = V.transpose(-2, -1).unsqueeze(2)
    dP_dropped_grouped = torch.matmul(dO_grouped, V_t)
    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_ATTENTION_HEADS, sq, skv)

    scale = 1.0 / (1.0 - p_drop)
    dP = dP_dropped.float() * mask * scale
    P_f = P.float()
    dPP_sum = (dP * P_f).sum(dim=-1, keepdim=True)
    dS = P_f * (dP - dPP_sum)

    P_dropped_grouped = P_dropped.reshape(bs, NUM_KEY_VALUE_HEADS, NUM_GROUPS, sq, skv)
    dV_grouped = torch.matmul(P_dropped_grouped.transpose(-2, -1), dO_grouped)
    dV = dV_grouped.float().sum(dim=2)

    return dS.to(torch.bfloat16), dV.to(torch.bfloat16)


def custom_kernel(data):
    """Attention backward pass using fused CUDA kernels."""
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()

    if _USE_CUDA:
        try:
            results = _ext.attn_bwd(
                dO, attn_weights, attn_weights_dropped,
                value_states, dropout_mask, float(attention_dropout)
            )
            return results[0], results[1]
        except Exception as e:
            print(f"CUDA extension failed: {e}")

    return _attn_backward_fallback(
        dO, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, attention_dropout
    )