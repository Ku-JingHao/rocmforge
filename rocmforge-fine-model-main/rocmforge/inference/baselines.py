"""
Reference baseline implementations for benchmark comparison.

Measures performance of:
- PyTorch eager (default execution)
- torch.compile (PyTorch 2.x JIT)
- rocBLAS (hand-tuned AMD library) for applicable operations

These are the "competition" your fine-tuned ROCmForge model must beat (or approach).
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available — baselines disabled.")


def benchmark_pytorch_eager(operation: str, problem_size: dict, dtype: str = "fp16") -> Optional[float]:
    """Measure PyTorch eager mode TFLOPS."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return None

    torch_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(dtype, torch.float16)

    M = problem_size.get("M", 4096)
    N = problem_size.get("N", 4096)
    K = problem_size.get("K", 4096)
    device = "cuda"

    try:
        if operation == "gemm":
            A = torch.randn(M, K, dtype=torch_dtype, device=device)
            B = torch.randn(K, N, dtype=torch_dtype, device=device)
            return _time_op(lambda: torch.matmul(A, B), 2 * M * N * K)

        elif operation == "softmax":
            X = torch.randn(M, N, dtype=torch_dtype, device=device)
            return _time_op(lambda: torch.softmax(X, dim=-1), 5 * M * N)

        elif operation == "vector_add":
            size = M * N if M and N else max(M, N)
            A = torch.randn(size, dtype=torch_dtype, device=device)
            B = torch.randn(size, dtype=torch_dtype, device=device)
            return _time_op(lambda: A + B, size)

        elif operation == "layernorm":
            X = torch.randn(M, N, dtype=torch_dtype, device=device)
            ln = torch.nn.LayerNorm(N).to(device).to(torch_dtype)
            return _time_op(lambda: ln(X), 5 * M * N)

        elif operation == "reduction":
            size = M * N if M and N else max(M, N)
            X = torch.randn(size, dtype=torch_dtype, device=device)
            return _time_op(lambda: X.sum(), size)

    except Exception as e:
        logger.warning("PyTorch baseline failed for %s: %s", operation, e)
        return None

    return None


def benchmark_torch_compile(operation: str, problem_size: dict, dtype: str = "fp16") -> Optional[float]:
    """Measure torch.compile (Inductor) TFLOPS."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return None

    torch_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(dtype, torch.float16)

    M = problem_size.get("M", 4096)
    N = problem_size.get("N", 4096)
    K = problem_size.get("K", 4096)
    device = "cuda"

    try:
        if operation == "gemm":
            A = torch.randn(M, K, dtype=torch_dtype, device=device)
            B = torch.randn(K, N, dtype=torch_dtype, device=device)
            fn = torch.compile(lambda a, b: torch.matmul(a, b))
            return _time_op(lambda: fn(A, B), 2 * M * N * K)

        elif operation == "softmax":
            X = torch.randn(M, N, dtype=torch_dtype, device=device)
            fn = torch.compile(lambda x: torch.softmax(x, dim=-1))
            return _time_op(lambda: fn(X), 5 * M * N)

    except Exception as e:
        logger.warning("torch.compile baseline failed: %s", e)
        return None

    return None


def benchmark_rocblas(operation: str, problem_size: dict, dtype: str = "fp16") -> Optional[float]:
    """
    rocBLAS reference baseline.

    Note: PyTorch's matmul on ROCm already calls rocBLAS internally for GEMM,
    so this often matches PyTorch eager. We expose it separately for clarity
    in the demo bar chart ("oracle" comparison).
    """
    if operation != "gemm":
        return None

    return benchmark_pytorch_eager(operation, problem_size, dtype)


def _time_op(fn, total_flops: int, num_warmup: int = 10, num_iters: int = 50) -> float:
    """Time an operation and compute TFLOPS."""
    if not HAS_TORCH:
        return 0.0

    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_time_s = elapsed / num_iters
    if avg_time_s <= 0:
        return 0.0
    return (total_flops / avg_time_s) / 1e12


def get_all_baselines(operation: str, problem_size: dict, dtype: str = "fp16") -> dict:
    """Run all available baselines and return TFLOPS dict for the bar chart."""
    return {
        "eager": benchmark_pytorch_eager(operation, problem_size, dtype),
        "compiled": benchmark_torch_compile(operation, problem_size, dtype),
        "rocblas": benchmark_rocblas(operation, problem_size, dtype),
    }
