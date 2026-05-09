"""
Extract Triton kernel examples from openai/triton.

Triton is particularly valuable because:
1. It's what most PyTorch users actually write (higher-level than CUDA)
2. Triton compiles to both CUDA PTX and AMD AMDGPU backends
3. The test suite contains clean, well-documented kernel examples

We create pairs: (triton_kernel, equivalent_hip_kernel)
For cases where we don't have direct HIP output, we create:
(triton_kernel_description, triton_code) — teaching the model to understand Triton input
"""

import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"
OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_triton_kernels(content: str) -> list[dict]:
    """Extract @triton.jit decorated functions."""
    kernels = []

    pattern = r'(@triton\.jit\s*\ndef\s+(\w+)\([^)]*\):.*?)(?=\n@|\ndef\s+\w+\(|\nclass\s|\Z)'
    matches = re.finditer(pattern, content, re.DOTALL)

    for match in matches:
        full_kernel = match.group(1).strip()
        name = match.group(2)

        if len(full_kernel) < 50:
            continue

        docstring = ""
        doc_match = re.search(r'"""(.*?)"""', full_kernel, re.DOTALL)
        if doc_match:
            docstring = doc_match.group(1).strip()

        kernels.append({
            "name": name,
            "code": full_kernel,
            "docstring": docstring,
        })

    return kernels


def extract_test_functions(content: str) -> list[dict]:
    """Extract test functions that show how kernels are launched."""
    tests = []
    pattern = r'(def\s+(test_\w+)\([^)]*\):.*?)(?=\ndef\s+|\nclass\s|\Z)'
    matches = re.finditer(pattern, content, re.DOTALL)

    for match in matches:
        func_body = match.group(1).strip()
        name = match.group(2)
        if len(func_body) < 50 or len(func_body) > 10000:
            continue
        tests.append({"name": name, "code": func_body})

    return tests


def infer_operation(kernel_name: str, code: str) -> str:
    """Infer what operation a Triton kernel performs."""
    name_lower = kernel_name.lower()

    op_keywords = {
        "matmul": "Matrix multiplication (GEMM)",
        "softmax": "Softmax activation",
        "attention": "Multi-head attention",
        "flash": "Flash attention",
        "layernorm": "Layer normalization",
        "layer_norm": "Layer normalization",
        "dropout": "Dropout",
        "relu": "ReLU activation",
        "gelu": "GELU activation",
        "add": "Element-wise addition",
        "mul": "Element-wise multiplication",
        "reduce": "Reduction operation",
        "sum": "Sum reduction",
        "max": "Max reduction",
        "transpose": "Matrix transpose",
        "conv": "Convolution",
        "fused": "Fused kernel operation",
        "copy": "Memory copy/transform",
        "norm": "Normalization",
        "embedding": "Embedding lookup",
        "cross_entropy": "Cross-entropy loss",
    }

    for key, desc in op_keywords.items():
        if key in name_lower:
            return desc

    if "tl.dot" in code:
        return "Matrix multiplication variant"
    if "tl.sum" in code or "tl.reduce" in code:
        return "Reduction operation"
    if "tl.exp" in code and "tl.sum" in code:
        return "Softmax-like normalization"

    return f"GPU kernel operation ({kernel_name})"


def process_triton_repo():
    """Process the entire Triton repo for kernel examples."""
    pairs = []
    triton_dir = RAW_DIR / "triton"
    if not triton_dir.exists():
        print("[WARN] triton repo not found, skipping.")
        return pairs

    search_dirs = [
        triton_dir / "python" / "tutorials",
        triton_dir / "python" / "test",
        triton_dir / "python" / "triton" / "ops",
    ]

    py_files = []
    for d in search_dirs:
        if d.exists():
            py_files.extend(d.rglob("*.py"))

    if not py_files:
        py_files = list(triton_dir.rglob("*.py"))

    print(f"[INFO] Scanning {len(py_files)} Python files in triton repo")

    for filepath in py_files:
        if filepath.stat().st_size < 100 or filepath.stat().st_size > 100_000:
            continue

        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if "triton" not in content.lower():
            continue

        kernels = extract_triton_kernels(content)

        for kernel in kernels:
            operation = infer_operation(kernel["name"], kernel["code"])
            desc = kernel["docstring"] if kernel["docstring"] else operation

            instruction = (
                f"Convert this Triton kernel to an optimized HIP kernel for AMD MI300X.\n\n"
                f"Operation: {desc}\n"
                f"The HIP kernel should:\n"
                f"- Use wavefront-64 operations (AMD uses 64-wide wavefronts, not 32-wide warps)\n"
                f"- Optimize LDS (Local Data Share) usage for data reuse\n"
                f"- Use MFMA intrinsics for matrix operations where applicable\n"
                f"- Target gfx942 architecture"
            )

            pairs.append({
                "instruction": instruction,
                "input": kernel["code"],
                "output": "",
                "metadata": {
                    "source": "openai/triton",
                    "filename": str(filepath.relative_to(triton_dir)),
                    "type": "triton_kernel",
                    "operation": operation,
                    "kernel_name": kernel["name"],
                }
            })

        if "@triton.jit" in content:
            instruction_understand = (
                f"Explain what this Triton code does and describe how you would "
                f"implement an equivalent optimized HIP kernel for AMD MI300X (gfx942)."
            )
            pairs.append({
                "instruction": instruction_understand,
                "input": content[:8000] if len(content) > 8000 else content,
                "output": "",
                "metadata": {
                    "source": "openai/triton",
                    "filename": str(filepath.relative_to(triton_dir)),
                    "type": "triton_understanding",
                }
            })

    print(f"[INFO] Extracted {len(pairs)} examples from triton")
    return pairs


def main():
    print("=" * 60)
    print("Processing Triton repository")
    print("=" * 60)

    pairs = process_triton_repo()

    output_file = OUTPUT_DIR / "03_triton_kernels.jsonl"
    with open(output_file, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\n[DONE] Wrote {len(pairs)} Triton examples to {output_file}")


if __name__ == "__main__":
    main()
