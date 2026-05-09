import type { ExampleEntry } from '../types';

/**
 * Pre-baked demo examples — the "safety net" for the live demo.
 * These cover the operations your fine-tuned model handles best.
 */
export const EXAMPLES: ExampleEntry[] = [
  {
    id: 'pytorch-matmul',
    title: 'PyTorch matmul (FP16)',
    description: 'Classic matrix multiplication — the headline GEMM demo',
    language: 'pytorch',
    code: `import torch

def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [4096, 4096] float16
    # B: [4096, 4096] float16
    return torch.matmul(A, B)`,
  },
  {
    id: 'pytorch-attention',
    title: 'PyTorch attention score',
    description: 'Multi-head attention QK^T softmax — fused kernel candidate',
    language: 'pytorch',
    code: `import torch
import torch.nn.functional as F

def attention_score(Q, K, scale):
    # Q, K: [batch=8, heads=32, seq=2048, dim=128]
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    return F.softmax(scores, dim=-1)`,
  },
  {
    id: 'pytorch-layernorm',
    title: 'PyTorch LayerNorm + ReLU',
    description: 'Fused normalization + activation',
    language: 'pytorch',
    code: `import torch
import torch.nn.functional as F

def fused_layernorm_relu(x, weight, bias, eps=1e-5):
    # x: [2048, 4096] float32
    normalized = F.layer_norm(x, [x.shape[-1]], weight, bias, eps)
    return F.relu(normalized)`,
  },
  {
    id: 'cuda-softmax',
    title: 'CUDA softmax (warp-32)',
    description: 'NVIDIA-style code that needs MI300X-aware fixes',
    language: 'cuda',
    code: `__global__ void softmax_cuda(float* input, float* output, int N) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    shared[tid] = (idx < N) ? input[idx] : -INFINITY;
    __syncthreads();

    // WARNING: assumes warp size 32 (NVIDIA) — must fix for AMD's wavefront-64
    float max_val = shared[tid];
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    // ...
}`,
  },
  {
    id: 'triton-vec-add',
    title: 'Triton vector add',
    description: 'Simple Triton kernel — translate to optimized HIP',
    language: 'triton',
    code: `import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)`,
  },
  {
    id: 'english-relu',
    title: 'Plain English: vectorized ReLU',
    description: 'No code input — just describe what you want',
    language: 'english',
    code: `Write an optimized HIP kernel for AMD MI300X that computes element-wise ReLU activation on a float16 tensor of 8 million elements. Use vectorized half8 loads to maximize HBM3 bandwidth utilization, and target wavefront-64 occupancy of at least 4 wavefronts per Compute Unit.`,
  },
];

export function getExampleById(id: string): ExampleEntry | undefined {
  return EXAMPLES.find((e) => e.id === id);
}
