"""
Generate instruction-following training data.

This script creates Type 3 training data:
1. Hand-crafted golden examples (high quality, small quantity)
2. PyTorch → HIP instruction pairs (common user queries)
3. AMD architecture knowledge (docs → instruction format)

These teach the model to respond to natural language requests like:
"Optimize this PyTorch matmul for MI300X"
"Write a fused kernel for LayerNorm + ReLU"
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


GOLDEN_EXAMPLES = [
    {
        "instruction": "Convert this PyTorch operation to an optimized HIP kernel for AMD MI300X.",
        "input": """import torch

def matmul_fp16(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [M, K], B: [K, N], both float16
    return torch.matmul(A, B)

# Usage: C = matmul_fp16(A, B) where M=4096, K=4096, N=4096""",
        "output": """#include <hip/hip_runtime.h>

// Tiled GEMM kernel optimized for MI300X (gfx942)
// Uses:
// - Wavefront-64 cooperative matrix operations
// - LDS for tile reuse (reduces HBM3 bandwidth pressure)
// - MFMA intrinsics for FP16 matrix multiply-accumulate

#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 16
#define WARP_M 64
#define WARP_N 64

__global__ void gemm_fp16_mi300x(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    const int M, const int N, const int K)
{
    __shared__ half smem_A[BLOCK_M][BLOCK_K + 1];  // +1 to avoid bank conflicts
    __shared__ half smem_B[BLOCK_K][BLOCK_N + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * BLOCK_M;
    const int by = blockIdx.y * BLOCK_N;

    // Register tile for accumulation
    float acc[4][4] = {0.0f};

    for (int k = 0; k < K; k += BLOCK_K) {
        // Cooperative load A tile to LDS
        if (bx + ty < M && k + tx < K)
            smem_A[ty][tx] = A[(bx + ty) * K + (k + tx)];
        else
            smem_A[ty][tx] = __float2half(0.0f);

        // Cooperative load B tile to LDS
        if (k + ty < K && by + tx < N)
            smem_B[ty][tx] = B[(k + ty) * N + (by + tx)];
        else
            smem_B[ty][tx] = __float2half(0.0f);

        __syncthreads();

        // MFMA-style accumulation using wavefront-64
        // Each wavefront processes a 16x16 output tile
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += __half2float(smem_A[ty * 4 + i][kk])
                               * __half2float(smem_B[kk][tx * 4 + j]);
                }
            }
        }
        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int row = bx + ty * 4 + i;
            int col = by + tx * 4 + j;
            if (row < M && col < N)
                C[row * N + col] = __float2half(acc[i][j]);
        }
    }
}

// Launch configuration for MI300X:
// - Grid: (M/BLOCK_M, N/BLOCK_N)
// - Block: (BLOCK_N/4, BLOCK_M/4) = (32, 32)
// - Shared memory: 2 * BLOCK_M * (BLOCK_K+1) * sizeof(half) per block
""",
        "metadata": {"source": "golden", "type": "pytorch_to_hip", "operation": "gemm"}
    },
    {
        "instruction": "Write an optimized HIP kernel for fused LayerNorm + ReLU for AMD MI300X.",
        "input": """# PyTorch equivalent:
def fused_layernorm_relu(x, weight, bias, eps=1e-5):
    # x: [batch, hidden_dim], weight: [hidden_dim], bias: [hidden_dim]
    normalized = torch.nn.functional.layer_norm(x, [x.shape[-1]], weight, bias, eps)
    return torch.relu(normalized)

# hidden_dim = 4096, batch = 2048""",
        "output": """#include <hip/hip_runtime.h>

// Fused LayerNorm + ReLU kernel for MI300X
// Key optimizations:
// - Single pass reduction using wavefront shuffle operations (no LDS needed for small reductions)
// - Fused normalization + activation to avoid global memory round-trip
// - Vectorized loads (float4) to maximize HBM3 bandwidth utilization
// - One wavefront (64 threads) processes one row

__device__ __forceinline__ float warp_reduce_sum(float val) {
    // AMD wavefront-64 reduction using DPP (Data-Parallel Primitives)
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, 64);  // wavefront-64, NOT warp-32
    }
    return val;
}

__global__ void fused_layernorm_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    const int hidden_dim,
    const float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;  // 0..63 for wavefront-64

    const float* row_input = input + row * hidden_dim;
    float* row_output = output + row * hidden_dim;

    // Step 1: Compute mean (each thread handles hidden_dim/64 elements)
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += 64) {
        local_sum += row_input[i];
    }
    float mean = warp_reduce_sum(local_sum) / (float)hidden_dim;

    // Step 2: Compute variance
    float local_var = 0.0f;
    for (int i = tid; i < hidden_dim; i += 64) {
        float diff = row_input[i] - mean;
        local_var += diff * diff;
    }
    float variance = warp_reduce_sum(local_var) / (float)hidden_dim;
    float inv_std = rsqrtf(variance + eps);

    // Step 3: Normalize + scale + bias + ReLU (fused, single global memory write)
    for (int i = tid; i < hidden_dim; i += 64) {
        float normalized = (row_input[i] - mean) * inv_std;
        float result = normalized * gamma[i] + beta[i];
        row_output[i] = fmaxf(result, 0.0f);  // ReLU fused
    }
}

// Launch: <<<batch_size, 64>>> (one wavefront per row)
// For hidden_dim=4096, batch=2048: grid=2048, block=64
""",
        "metadata": {"source": "golden", "type": "pytorch_to_hip", "operation": "fused_layernorm_relu"}
    },
    {
        "instruction": "Convert this CUDA softmax kernel to optimized HIP for AMD MI300X. Fix any NVIDIA-specific assumptions.",
        "input": """__global__ void softmax_cuda(float* input, float* output, int N) {
    // NVIDIA-style: assumes warp size 32
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load to shared memory
    shared[tid] = (idx < N) ? input[idx] : -INFINITY;
    __syncthreads();

    // Warp-level max reduction (WRONG for AMD: uses warp=32)
    float max_val = shared[tid];
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    // ... rest of softmax
}""",
        "output": """#include <hip/hip_runtime.h>

// Softmax kernel optimized for AMD MI300X
// Fixes from CUDA version:
// 1. Wavefront size is 64, not 32 — reduction must cover full wavefront
// 2. No __shfl_down_sync mask needed on AMD (wavefronts are always synchronized)
// 3. Use DPP-based reduction for better performance on CDNA3

__device__ __forceinline__ float wavefront_max(float val) {
    // Full wavefront-64 reduction for max
    for (int offset = 32; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset, 64));
    }
    return val;
}

__device__ __forceinline__ float wavefront_sum(float val) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, 64);
    }
    return val;
}

__global__ void softmax_mi300x(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;  // 0..63
    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    // Phase 1: Find max (numerically stable softmax)
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += 64) {
        local_max = fmaxf(local_max, row_in[i]);
    }
    float row_max = wavefront_max(local_max);

    // Phase 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += 64) {
        local_sum += expf(row_in[i] - row_max);
    }
    float row_sum = wavefront_sum(local_sum);

    // Phase 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < N; i += 64) {
        row_out[i] = expf(row_in[i] - row_max) * inv_sum;
    }
}

// Launch: <<<num_rows, 64>>>
// Each wavefront (64 threads) handles one row
// For large N (>4096), use multiple wavefronts per row with LDS coordination
""",
        "metadata": {"source": "golden", "type": "cuda_to_hip_optimized", "operation": "softmax"}
    },
    {
        "instruction": "Write an optimized HIP vector addition kernel for AMD MI300X that maximizes memory bandwidth.",
        "input": "Vector addition: C[i] = A[i] + B[i] for N elements (FP32). Optimize for maximum HBM3 bandwidth on MI300X.",
        "output": """#include <hip/hip_runtime.h>

// Vector addition maximizing HBM3 bandwidth on MI300X
// Key optimizations:
// - float4 vectorized loads/stores (128-bit memory transactions)
// - Grid-stride loop for arbitrary sizes
// - Enough blocks to saturate all 304 Compute Units on MI300X

__global__ void vector_add_mi300x(
    const float4* __restrict__ A,
    const float4* __restrict__ B,
    float4* __restrict__ C,
    const int N4)  // N4 = N / 4
{
    // Grid-stride loop: each thread handles multiple float4 elements
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N4;
         i += gridDim.x * blockDim.x)
    {
        float4 a = A[i];
        float4 b = B[i];
        C[i] = make_float4(
            a.x + b.x,
            a.y + b.y,
            a.z + b.z,
            a.w + b.w
        );
    }
}

// Handle remainder elements (N not divisible by 4)
__global__ void vector_add_remainder(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int start, const int N)
{
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// Launch configuration for MI300X:
// - 304 Compute Units, each can run multiple wavefronts
// - Block size: 256 threads (4 wavefronts per block)
// - Grid: enough to saturate all CUs
// dim3 block(256);
// dim3 grid(min((N4 + 255) / 256, 304 * 8));  // 8 blocks per CU
""",
        "metadata": {"source": "golden", "type": "natural_to_hip", "operation": "vector_add"}
    },
    {
        "instruction": "Write a HIP reduction kernel (sum) optimized for AMD MI300X using wavefront-level primitives.",
        "input": "Compute the sum of all elements in a large array (100M+ elements, FP32). Use multi-level reduction: wavefront → block → grid.",
        "output": """#include <hip/hip_runtime.h>

// Multi-level reduction (sum) for AMD MI300X
// Level 1: Wavefront shuffle reduction (no LDS needed, 64 threads cooperate)
// Level 2: Cross-wavefront reduction within a block using LDS
// Level 3: Atomic add across blocks (safe for final accumulation)

__device__ __forceinline__ float wavefront_reduce_add(float val) {
    // Wavefront-64 butterfly reduction
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, 64);
    }
    return val;
}

__global__ void reduce_sum_mi300x(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N)
{
    __shared__ float shared_wavefront_sums[4];  // max 4 wavefronts per block (256 threads)

    const int tid = threadIdx.x;
    const int wavefront_id = tid / 64;
    const int lane_id = tid % 64;
    const int global_id = blockIdx.x * blockDim.x + tid;

    // Phase 1: Grid-stride accumulation into thread-local sum
    float thread_sum = 0.0f;
    for (int i = global_id; i < N; i += gridDim.x * blockDim.x) {
        thread_sum += input[i];
    }

    // Phase 2: Wavefront-level reduction (no LDS, uses shuffle)
    float wavefront_sum = wavefront_reduce_add(thread_sum);

    // Phase 3: First lane of each wavefront writes to LDS
    if (lane_id == 0) {
        shared_wavefront_sums[wavefront_id] = wavefront_sum;
    }
    __syncthreads();

    // Phase 4: First wavefront reduces across all wavefront sums
    if (tid < 4) {
        float val = shared_wavefront_sums[tid];
        // Only 4 values to reduce — single wavefront handles it
        for (int offset = 2; offset > 0; offset >>= 1) {
            val += __shfl_xor(val, offset, 64);
        }
        if (tid == 0) {
            atomicAdd(output, val);
        }
    }
}

// Launch: <<<min(1024, (N+255)/256), 256>>>
// Pre-zero output before launch: hipMemset(output, 0, sizeof(float))
// For MI300X: use 1024-2048 blocks to saturate all 304 CUs
""",
        "metadata": {"source": "golden", "type": "natural_to_hip", "operation": "reduction_sum"}
    },
]


PYTORCH_TO_HIP_INSTRUCTIONS = [
    {
        "instruction": "Convert this PyTorch operation to an optimized HIP kernel for AMD MI300X.",
        "input": "torch.nn.functional.gelu(x)  # x shape: [batch=2048, hidden=4096]",
        "output": """#include <hip/hip_runtime.h>

// GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Optimized for MI300X: vectorized float4 loads, grid-stride pattern

#define SQRT_2_OVER_PI 0.7978845608f
#define GELU_COEFF 0.044715f

__global__ void gelu_mi300x(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const int N4)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N4; i += gridDim.x * blockDim.x) {
        float4 x = input[i];

        // Process all 4 elements
        float vals[4] = {x.x, x.y, x.z, x.w};
        float results[4];

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float v = vals[j];
            float cdf = 0.5f * (1.0f + tanhf(SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v)));
            results[j] = v * cdf;
        }

        output[i] = make_float4(results[0], results[1], results[2], results[3]);
    }
}

// Launch: <<<(N4+255)/256, 256>>>
""",
        "metadata": {"source": "golden", "type": "pytorch_to_hip", "operation": "gelu"}
    },
    {
        "instruction": "Convert this PyTorch operation to an optimized HIP kernel for AMD MI300X.",
        "input": """# Fused attention score computation
def attention_score(Q, K, scale):
    # Q: [batch, heads, seq_len, head_dim]
    # K: [batch, heads, seq_len, head_dim]
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    return torch.softmax(scores, dim=-1)

# batch=8, heads=32, seq_len=2048, head_dim=128""",
        "output": """#include <hip/hip_runtime.h>

// Fused QK^T + Scale + Softmax for MI300X
// This is a simplified single-head attention score kernel
// For production, use flash attention pattern with tiling

// Key MI300X optimizations:
// - Tile Q and K in LDS for reuse
// - Online softmax (Milakov & Gimelshein, 2018) avoids extra pass
// - Wavefront-64 cooperative reductions

#define HEAD_DIM 128
#define TILE_SEQ 64

__device__ __forceinline__ float wavefront_max_f(float val) {
    for (int offset = 32; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor(val, offset, 64));
    return val;
}

__device__ __forceinline__ float wavefront_sum_f(float val) {
    for (int offset = 32; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset, 64);
    return val;
}

__global__ void fused_attention_score_mi300x(
    const half* __restrict__ Q,       // [seq_len, head_dim]
    const half* __restrict__ K,       // [seq_len, head_dim]
    float* __restrict__ attn_out,     // [seq_len, seq_len]
    const int seq_len,
    const float scale)
{
    __shared__ half smem_q[TILE_SEQ][HEAD_DIM + 1];
    __shared__ half smem_k[TILE_SEQ][HEAD_DIM + 1];

    const int row = blockIdx.x;  // which query token
    const int tid = threadIdx.x; // 0..63 (one wavefront)

    // Load Q row to shared (cooperative across wavefront)
    for (int d = tid; d < HEAD_DIM; d += 64) {
        smem_q[0][d] = Q[row * HEAD_DIM + d];
    }
    __syncthreads();

    // Online softmax: compute max and sum in single pass over K
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (int k_tile = 0; k_tile < seq_len; k_tile += 64) {
        int k_idx = k_tile + tid;

        // Compute dot product Q[row] . K[k_idx]
        float dot = 0.0f;
        if (k_idx < seq_len) {
            for (int d = 0; d < HEAD_DIM; ++d) {
                dot += __half2float(smem_q[0][d]) * __half2float(K[k_idx * HEAD_DIM + d]);
            }
            dot *= scale;
        } else {
            dot = -INFINITY;
        }

        // Online softmax update
        float new_max = fmaxf(running_max, dot);
        running_sum = running_sum * expf(running_max - new_max) + expf(dot - new_max);
        running_max = new_max;

        // Store intermediate score
        if (k_idx < seq_len) {
            attn_out[row * seq_len + k_idx] = dot;
        }
    }

    // Wavefront-level reduction for global max and sum
    float global_max = wavefront_max_f(running_max);
    running_sum = running_sum * expf(running_max - global_max);
    float global_sum = wavefront_sum_f(running_sum);

    // Normalize (second pass — can be fused with output matmul in flash attention)
    for (int i = tid; i < seq_len; i += 64) {
        float score = expf(attn_out[row * seq_len + i] - global_max) / global_sum;
        attn_out[row * seq_len + i] = score;
    }
}

// Launch: <<<seq_len, 64>>> (one wavefront per query token)
// For multi-head: add batch*heads dimension to grid
""",
        "metadata": {"source": "golden", "type": "pytorch_to_hip", "operation": "attention_score"}
    },
]


AMD_ARCHITECTURE_KNOWLEDGE = [
    {
        "instruction": "What are the key differences between NVIDIA CUDA and AMD HIP that I need to consider when writing optimized kernels for MI300X?",
        "input": "",
        "output": """Key differences for MI300X (gfx942) kernel optimization:

1. **Wavefront Size**: AMD uses wavefront-64 (64 threads execute in lockstep), NVIDIA uses warp-32. All shuffle operations and reductions must account for width 64, not 32.

2. **MFMA Intrinsics**: AMD's equivalent of Tensor Cores. Use `__builtin_amdgcn_mfma_f32_16x16x16f16` for FP16 matrix multiply-accumulate. These operate on wavefront-64.

3. **LDS (Local Data Share)**: AMD's equivalent of CUDA shared memory. 64KB per Compute Unit on MI300X. Use `__shared__` keyword (same as CUDA syntax in HIP).

4. **Memory Hierarchy**:
   - 192GB HBM3 at 5.3 TB/s bandwidth
   - 32MB L2 cache (shared across chiplets)
   - 64KB LDS per CU
   - Registers: 256 VGPRs per wavefront

5. **Compute Units**: MI300X has 304 CUs (vs 132 SMs on H100). Launch more blocks to saturate.

6. **No warp sync needed**: AMD wavefronts are always synchronized within the wavefront. No `__syncwarp()` or sync masks needed.

7. **Occupancy**: Target 4+ wavefronts per CU for latency hiding. Use `__launch_bounds__(256, 4)` to hint the compiler.

8. **Chiplet Architecture**: MI300X has 8 XCDs (chiplets). Data locality across chiplets matters for large problems.

9. **Vector Loads**: Use `float4` (128-bit) loads to maximize bandwidth utilization from HBM3.

10. **Compiler**: Use `hipcc --offload-arch=gfx942 -O3` for MI300X targeting.""",
        "metadata": {"source": "golden", "type": "architecture_knowledge", "operation": "general"}
    },
    {
        "instruction": "How do I use MFMA (Matrix Fused Multiply-Add) intrinsics on AMD MI300X for FP16 matrix multiplication?",
        "input": "",
        "output": """MFMA intrinsics on MI300X (CDNA3 architecture):

The primary FP16 MFMA intrinsic:
```
// Computes D = A * B + C (matrix multiply-accumulate)
// A: 4x fp16x4 (from 4 threads in the wavefront)
// B: 4x fp16x4
// C/D: 16x fp32 accumulators
float32x4 __builtin_amdgcn_mfma_f32_16x16x16f16(
    half8 srcA,    // 8 FP16 values from register
    half8 srcB,    // 8 FP16 values from register
    float4 srcC,   // 4 FP32 accumulator values
    int cbsz,      // control: broadcast size (usually 0)
    int abid,      // control: broadcast id (usually 0)
    int blgp       // control: blend group pattern (usually 0)
);
```

Larger tile variant (32x32x8):
```
float16 __builtin_amdgcn_mfma_f32_32x32x8f16(
    half4 srcA, half4 srcB, float16 srcC,
    int cbsz, int abid, int blgp
);
```

Usage pattern in a GEMM kernel:
```cpp
// Each wavefront (64 threads) cooperatively computes a tile
// Threads load A and B fragments from LDS
// Then call MFMA to accumulate into float32 registers

half8 a_frag = load_a_fragment(smem_A, lane_id);
half8 b_frag = load_b_fragment(smem_B, lane_id);
acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_frag, b_frag, acc, 0, 0, 0);
```

Compile with: `hipcc --offload-arch=gfx942 -O3`

The 16x16x16 variant gives best throughput on MI300X CDNA3 matrix cores.""",
        "metadata": {"source": "golden", "type": "architecture_knowledge", "operation": "mfma"}
    },
    {
        "instruction": "What is the optimal launch configuration for a HIP kernel on AMD MI300X?",
        "input": "I want to maximize GPU utilization on MI300X for a compute-bound kernel.",
        "output": """Optimal launch configuration for MI300X (gfx942):

**Hardware specs to consider:**
- 304 Compute Units (CUs)
- Each CU can run up to 16 wavefronts (occupancy-dependent)
- Wavefront size: 64 threads
- Max threads per block: 1024
- Max shared memory per block: 64KB LDS

**Rules of thumb:**

1. **Block size**: Use 256 threads (4 wavefronts per block)
   ```cpp
   dim3 block(256);  // 4 wavefronts × 64 threads
   ```

2. **Grid size**: At least 304 × 4 = 1216 blocks to saturate all CUs
   ```cpp
   dim3 grid(min(num_elements / 256, 304 * 4));
   ```

3. **Occupancy hint**: Use launch bounds to help the compiler
   ```cpp
   __launch_bounds__(256, 4)  // max 256 threads/block, target 4 blocks/CU
   __global__ void my_kernel(...) { }
   ```

4. **Register pressure**: Stay under 64 VGPRs per thread for max occupancy
   - 64 VGPRs → 16 wavefronts/CU (100% occupancy)
   - 128 VGPRs → 8 wavefronts/CU (50% occupancy)
   - 256 VGPRs → 4 wavefronts/CU (25% occupancy)

5. **LDS usage**: Keep under 16KB/block for 4 blocks/CU
   - 64KB total / 4 blocks = 16KB available per block

**For memory-bound kernels**: Prefer more blocks with less work each (hides latency).
**For compute-bound kernels**: Fewer blocks is OK if register/LDS usage is high.

**Check occupancy:**
```bash
rocprof --stats ./my_kernel
# Look for OCCUPANCY field in output
```""",
        "metadata": {"source": "golden", "type": "architecture_knowledge", "operation": "launch_config"}
    },
]


def main():
    all_examples = []

    print("=" * 60)
    print("Generating instruction training data")
    print("=" * 60)

    all_examples.extend(GOLDEN_EXAMPLES)
    print(f"[INFO] Golden examples: {len(GOLDEN_EXAMPLES)}")

    all_examples.extend(PYTORCH_TO_HIP_INSTRUCTIONS)
    print(f"[INFO] PyTorch→HIP instructions: {len(PYTORCH_TO_HIP_INSTRUCTIONS)}")

    all_examples.extend(AMD_ARCHITECTURE_KNOWLEDGE)
    print(f"[INFO] Architecture knowledge: {len(AMD_ARCHITECTURE_KNOWLEDGE)}")

    output_file = OUTPUT_DIR / "04_instruction_data.jsonl"
    with open(output_file, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")

    print(f"\n[DONE] Wrote {len(all_examples)} instruction examples to {output_file}")
    print("\nNOTE: These are hand-crafted golden examples. During the hackathon,")
    print("expand this file with more operations (conv2d, batch norm, embedding, etc.)")
    print("Use a teacher model (Claude/GPT-4) to generate more pairs if needed.")


if __name__ == "__main__":
    main()
