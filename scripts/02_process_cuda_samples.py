"""
Process CUDA samples → HIP pairs.

This script:
1. Finds all .cu files in NVIDIA/cuda-samples and CUDALibrarySamples
2. Attempts to convert them via hipify-perl (available on ROCm instances)
3. Saves successful pairs as JSONL

On machines without hipify, it uses a regex-based fallback converter
that handles the most common API substitutions (sufficient for hackathon POC).
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent
RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"
OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CUDA_API_MAP = {
    "cudaMalloc": "hipMalloc",
    "cudaFree": "hipFree",
    "cudaMemcpy": "hipMemcpy",
    "cudaMemcpyHostToDevice": "hipMemcpyHostToDevice",
    "cudaMemcpyDeviceToHost": "hipMemcpyDeviceToHost",
    "cudaMemcpyDeviceToDevice": "hipMemcpyDeviceToDevice",
    "cudaMemset": "hipMemset",
    "cudaDeviceSynchronize": "hipDeviceSynchronize",
    "cudaGetDevice": "hipGetDevice",
    "cudaSetDevice": "hipSetDevice",
    "cudaGetDeviceCount": "hipGetDeviceCount",
    "cudaGetDeviceProperties": "hipGetDeviceProperties",
    "cudaDeviceProp": "hipDeviceProp_t",
    "cudaEvent_t": "hipEvent_t",
    "cudaEventCreate": "hipEventCreate",
    "cudaEventRecord": "hipEventRecord",
    "cudaEventSynchronize": "hipEventSynchronize",
    "cudaEventElapsedTime": "hipEventElapsedTime",
    "cudaEventDestroy": "hipEventDestroy",
    "cudaStream_t": "hipStream_t",
    "cudaStreamCreate": "hipStreamCreate",
    "cudaStreamDestroy": "hipStreamDestroy",
    "cudaStreamSynchronize": "hipStreamSynchronize",
    "cudaSuccess": "hipSuccess",
    "cudaError_t": "hipError_t",
    "cudaGetErrorString": "hipGetErrorString",
    "cudaGetLastError": "hipGetLastError",
    "cudaPeekAtLastError": "hipPeekAtLastError",
    "__syncthreads": "__syncthreads",
    "atomicAdd": "atomicAdd",
    "atomicCAS": "atomicCAS",
    "__shared__": "__shared__",
    "cudaMallocManaged": "hipMallocManaged",
    "cudaMemPrefetchAsync": "hipMemPrefetchAsync",
    "cudaMemAdvise": "hipMemAdvise",
    "cublasCreate": "rocblas_create_handle",
    "cublasDestroy": "rocblas_destroy_handle",
    "cublasHandle_t": "rocblas_handle",
    "cublasSgemm": "rocblas_sgemm",
    "cublasDgemm": "rocblas_dgemm",
    "curandGenerator_t": "rocrand_generator",
    "curandCreateGenerator": "rocrand_create_generator",
    "curandGenerateUniform": "rocrand_generate_uniform",
}

HEADER_MAP = {
    "cuda_runtime.h": "hip/hip_runtime.h",
    "cuda_runtime_api.h": "hip/hip_runtime_api.h",
    "cuda.h": "hip/hip_runtime.h",
    "cublas_v2.h": "rocblas/rocblas.h",
    "cublas.h": "rocblas/rocblas.h",
    "curand.h": "rocrand/rocrand.h",
    "cufft.h": "rocfft/rocfft.h",
}


def hipify_with_tool(cuda_code: str, filepath: str) -> Optional[str]:
    """Try using hipify-perl (available on ROCm machines)."""
    try:
        result = subprocess.run(
            ["hipify-perl", "--stdin"],
            input=cuda_code,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def hipify_regex_fallback(cuda_code: str) -> str:
    """Regex-based CUDA→HIP conversion (covers common patterns)."""
    hip_code = cuda_code

    for header_cuda, header_hip in HEADER_MAP.items():
        hip_code = hip_code.replace(f"#include <{header_cuda}>", f"#include <{header_hip}>")
        hip_code = hip_code.replace(f'#include "{header_cuda}"', f'#include "{header_hip}"')

    sorted_apis = sorted(CUDA_API_MAP.keys(), key=len, reverse=True)
    for cuda_api, hip_api in ((k, CUDA_API_MAP[k]) for k in sorted_apis):
        hip_code = re.sub(r'\b' + re.escape(cuda_api) + r'\b', hip_api, hip_code)

    hip_code = re.sub(
        r'<<<\s*(\w+)\s*,\s*(\w+)\s*>>>',
        r'<<<\1, \2, 0, 0>>>',
        hip_code,
    )

    return hip_code


def is_valid_cuda_file(filepath: Path) -> bool:
    """Filter out test files, build scripts, etc."""
    name = filepath.name.lower()
    if filepath.stat().st_size < 100:
        return False
    if filepath.stat().st_size > 100_000:
        return False
    skip_patterns = ["test_", "benchmark_", "cmake", "makefile"]
    return not any(p in name for p in skip_patterns)


def extract_kernel_functions(code: str) -> list[str]:
    """Extract individual __global__ kernel functions."""
    pattern = r'(__global__\s+\w[\w\s*&<>,]*?\([^)]*\)\s*\{)'
    matches = list(re.finditer(pattern, code))
    kernels = []
    for match in matches:
        start = match.start()
        brace_count = 0
        i = match.end() - 1
        while i < len(code):
            if code[i] == '{':
                brace_count += 1
            elif code[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    kernels.append(code[start:i + 1])
                    break
            i += 1
    return kernels


def create_training_pair(cuda_code: str, hip_code: str, source: str, filename: str) -> dict:
    """Format as instruction-following training example."""
    return {
        "instruction": f"Convert the following CUDA kernel to an optimized HIP kernel for AMD MI300X GPU. "
                       f"Use appropriate AMD-specific optimizations where possible "
                       f"(wavefront-64 operations, LDS usage, MFMA intrinsics for matrix ops).",
        "input": cuda_code.strip(),
        "output": hip_code.strip(),
        "metadata": {
            "source": source,
            "filename": filename,
            "type": "cuda_to_hip",
        }
    }


def process_cuda_repos():
    """Main processing loop across all CUDA source repos."""
    pairs = []
    cuda_dirs = [
        (RAW_DIR / "cuda-samples", "nvidia/cuda-samples"),
        (RAW_DIR / "CUDALibrarySamples", "nvidia/CUDALibrarySamples"),
    ]

    for repo_dir, source_name in cuda_dirs:
        if not repo_dir.exists():
            print(f"[WARN] {repo_dir} not found, skipping. Run 01_clone_repos.sh first.")
            continue

        cu_files = list(repo_dir.rglob("*.cu"))
        print(f"[INFO] Found {len(cu_files)} .cu files in {source_name}")

        for cu_file in cu_files:
            if not is_valid_cuda_file(cu_file):
                continue

            try:
                cuda_code = cu_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            hip_code = hipify_with_tool(cuda_code, str(cu_file))
            if hip_code is None:
                hip_code = hipify_regex_fallback(cuda_code)

            if hip_code and hip_code != cuda_code:
                pair = create_training_pair(
                    cuda_code, hip_code, source_name, cu_file.name
                )
                pairs.append(pair)

                kernels = extract_kernel_functions(cuda_code)
                hip_kernels = extract_kernel_functions(hip_code)
                for ck, hk in zip(kernels, hip_kernels):
                    if len(ck) > 50 and len(hk) > 50:
                        pairs.append(create_training_pair(
                            ck, hk, source_name, f"{cu_file.name}::kernel"
                        ))

    output_file = OUTPUT_DIR / "01_cuda_to_hip_pairs.jsonl"
    with open(output_file, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\n[DONE] Wrote {len(pairs)} CUDA→HIP pairs to {output_file}")
    return pairs


if __name__ == "__main__":
    process_cuda_repos()
