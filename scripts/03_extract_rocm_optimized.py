"""
Extract optimized HIP kernels from AMD's own repositories.

These are the "gold standard" answers — hand-tuned by AMD engineers for MI300X.
We extract:
1. composable_kernel: Template-based high-performance kernels
2. rocBLAS: BLAS operations optimized for AMD GPUs
3. MIOpen: Deep learning primitives (conv, attention, normalization)

Each extracted kernel becomes a training example where:
- input = description of what the kernel does (from comments/filename/function signature)
- output = the actual optimized implementation
"""

import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"
OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_function_with_body(content: str, pattern: str) -> list[tuple[str, str]]:
    """Extract functions matching pattern, return (signature, full_body) pairs."""
    results = []
    matches = list(re.finditer(pattern, content))

    for match in matches:
        sig_start = match.start()
        brace_pos = content.find('{', match.end())
        if brace_pos == -1:
            continue

        signature = content[sig_start:brace_pos].strip()
        brace_count = 0
        i = brace_pos
        while i < len(content):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    body = content[sig_start:i + 1]
                    results.append((signature, body))
                    break
            i += 1

    return results


def extract_leading_comment(content: str, func_start: int) -> str:
    """Extract comment block immediately above a function."""
    lines = content[:func_start].split('\n')
    comment_lines = []
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('*') or stripped.startswith('/*'):
            comment_lines.insert(0, stripped.lstrip('/* ').rstrip(' */'))
        elif stripped == '':
            if comment_lines:
                break
        else:
            break
    return ' '.join(comment_lines).strip()


def infer_description_from_path(filepath: Path) -> str:
    """Generate a description from the file path and directory structure."""
    parts = filepath.parts
    relevant = [p for p in parts if p not in ('src', 'include', 'kernel', 'device', 'impl')]
    name = filepath.stem.replace('_', ' ').replace('-', ' ')

    keywords = []
    for part in relevant[-4:]:
        clean = part.replace('_', ' ').replace('-', ' ')
        if clean not in keywords and clean != name:
            keywords.append(clean)

    return f"{name} ({' / '.join(keywords[-3:])})"


def process_composable_kernel():
    """Extract from ROCm/composable_kernel — the richest source."""
    pairs = []
    ck_dir = RAW_DIR / "composable_kernel"
    if not ck_dir.exists():
        print("[WARN] composable_kernel not found, skipping.")
        return pairs

    hip_files = list(ck_dir.rglob("*.hpp")) + list(ck_dir.rglob("*.cpp"))
    print(f"[INFO] Scanning {len(hip_files)} files in composable_kernel")

    for filepath in hip_files:
        if filepath.stat().st_size < 200 or filepath.stat().st_size > 200_000:
            continue

        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if '__device__' not in content and '__global__' not in content and 'template' not in content:
            continue

        functions = extract_function_with_body(
            content,
            r'(?:template\s*<[^>]*>\s*)?(?:__device__|__global__|__host__\s+__device__)\s+\w[\w\s*&<>,]*?\('
        )

        for signature, body in functions:
            if len(body) < 100 or len(body) > 15000:
                continue

            func_pos = content.find(signature)
            comment = extract_leading_comment(content, func_pos)
            desc = comment if comment else infer_description_from_path(filepath)

            instruction = (
                f"Write an optimized HIP kernel implementation for AMD MI300X: {desc}\n\n"
                f"Function signature: {signature}"
            )

            pairs.append({
                "instruction": instruction,
                "input": "",
                "output": body.strip(),
                "metadata": {
                    "source": "ROCm/composable_kernel",
                    "filename": str(filepath.relative_to(ck_dir)),
                    "type": "optimized_kernel",
                }
            })

    print(f"[INFO] Extracted {len(pairs)} examples from composable_kernel")
    return pairs


def process_rocblas():
    """Extract from ROCm/rocBLAS — BLAS operations."""
    pairs = []
    rocblas_dir = RAW_DIR / "rocBLAS"
    if not rocblas_dir.exists():
        print("[WARN] rocBLAS not found, skipping.")
        return pairs

    kernel_dirs = [
        rocblas_dir / "library" / "src" / "blas1",
        rocblas_dir / "library" / "src" / "blas2",
        rocblas_dir / "library" / "src" / "blas3",
    ]

    hip_files = []
    for d in kernel_dirs:
        if d.exists():
            hip_files.extend(d.rglob("*.hpp"))
            hip_files.extend(d.rglob("*.cpp"))

    if not hip_files:
        hip_files = list(rocblas_dir.rglob("*.hpp")) + list(rocblas_dir.rglob("*.cpp"))

    print(f"[INFO] Scanning {len(hip_files)} files in rocBLAS")

    blas_ops = {
        "gemm": "General Matrix Multiplication (GEMM) C = alpha*A*B + beta*C",
        "gemv": "General Matrix-Vector Multiplication y = alpha*A*x + beta*y",
        "axpy": "Vector addition y = alpha*x + y",
        "scal": "Vector scaling x = alpha*x",
        "dot": "Dot product of two vectors",
        "nrm2": "Euclidean norm of a vector",
        "trsm": "Triangular solve with multiple right-hand sides",
        "syrk": "Symmetric rank-k update",
        "trmm": "Triangular matrix-matrix multiplication",
    }

    for filepath in hip_files:
        if filepath.stat().st_size < 200 or filepath.stat().st_size > 150_000:
            continue

        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if '__global__' not in content and '__device__' not in content:
            continue

        op_name = None
        for op_key in blas_ops:
            if op_key in filepath.stem.lower():
                op_name = op_key
                break

        functions = extract_function_with_body(
            content,
            r'(?:template\s*<[^>]*>\s*)?(?:__device__|__global__)\s+\w[\w\s*&<>,]*?\('
        )

        for signature, body in functions:
            if len(body) < 100 or len(body) > 15000:
                continue

            if op_name and op_name in blas_ops:
                desc = blas_ops[op_name]
            else:
                desc = infer_description_from_path(filepath)

            instruction = (
                f"Write an AMD GPU-optimized HIP kernel for: {desc}\n\n"
                f"Target architecture: MI300X (gfx942). "
                f"Optimize for wavefront-64, use LDS where beneficial."
            )

            pairs.append({
                "instruction": instruction,
                "input": signature.strip(),
                "output": body.strip(),
                "metadata": {
                    "source": "ROCm/rocBLAS",
                    "filename": str(filepath.relative_to(rocblas_dir)),
                    "type": "blas_kernel",
                    "operation": op_name or "unknown",
                }
            })

    print(f"[INFO] Extracted {len(pairs)} examples from rocBLAS")
    return pairs


def process_miopen():
    """Extract from ROCm/MIOpen — deep learning primitives."""
    pairs = []
    miopen_dir = RAW_DIR / "MIOpen"
    if not miopen_dir.exists():
        print("[WARN] MIOpen not found, skipping.")
        return pairs

    kernel_dir = miopen_dir / "src" / "kernels"
    if not kernel_dir.exists():
        kernel_dir = miopen_dir / "src"

    hip_files = list(kernel_dir.rglob("*.hpp")) + list(kernel_dir.rglob("*.cpp"))
    print(f"[INFO] Scanning {len(hip_files)} files in MIOpen")

    dl_ops = {
        "conv": "Convolution forward/backward pass",
        "batchnorm": "Batch Normalization",
        "layernorm": "Layer Normalization",
        "softmax": "Softmax activation",
        "relu": "ReLU activation function",
        "pooling": "Pooling (max/average)",
        "dropout": "Dropout regularization",
        "attention": "Multi-head attention",
        "reduce": "Reduction operation (sum/mean/max)",
        "transpose": "Matrix transpose",
        "elementwise": "Element-wise operation",
    }

    for filepath in hip_files:
        if filepath.stat().st_size < 200 or filepath.stat().st_size > 150_000:
            continue

        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if '__global__' not in content and '__device__' not in content:
            continue

        op_name = None
        for op_key in dl_ops:
            if op_key in filepath.stem.lower():
                op_name = op_key
                break

        functions = extract_function_with_body(
            content,
            r'(?:template\s*<[^>]*>\s*)?(?:__device__|__global__)\s+\w[\w\s*&<>,]*?\('
        )

        for signature, body in functions:
            if len(body) < 100 or len(body) > 15000:
                continue

            if op_name and op_name in dl_ops:
                desc = dl_ops[op_name]
            else:
                desc = infer_description_from_path(filepath)

            instruction = (
                f"Write an optimized HIP kernel for the deep learning operation: {desc}\n\n"
                f"Target: AMD MI300X (gfx942) with 192GB HBM3. "
                f"Optimize memory bandwidth utilization and wavefront occupancy."
            )

            pairs.append({
                "instruction": instruction,
                "input": signature.strip(),
                "output": body.strip(),
                "metadata": {
                    "source": "ROCm/MIOpen",
                    "filename": str(filepath.relative_to(miopen_dir)),
                    "type": "dl_kernel",
                    "operation": op_name or "unknown",
                }
            })

    print(f"[INFO] Extracted {len(pairs)} examples from MIOpen")
    return pairs


def main():
    all_pairs = []

    print("=" * 60)
    print("Processing ROCm optimized kernel repositories")
    print("=" * 60)

    all_pairs.extend(process_composable_kernel())
    all_pairs.extend(process_rocblas())
    all_pairs.extend(process_miopen())

    output_file = OUTPUT_DIR / "02_rocm_optimized_kernels.jsonl"
    with open(output_file, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\n{'=' * 60}")
    print(f"[DONE] Total: {len(all_pairs)} optimized kernel examples")
    print(f"Written to: {output_file}")


if __name__ == "__main__":
    main()
