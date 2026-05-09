"""
Augment training data to increase volume and diversity.

Augmentation strategies:
1. Data type variants: FP32 → FP16/BF16/INT8 versions
2. Problem size variants: different matrix dimensions, batch sizes
3. Instruction rephrasing: same code, different ways to ask for it
4. Combined operations: fuse simple kernels into compound examples
"""

import json
import re
import random
import copy
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROCESSED_DIR = SCRIPT_DIR.parent / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

DTYPE_VARIANTS = [
    {"name": "FP32", "c_type": "float", "suffix": "f32", "load_type": "float4"},
    {"name": "FP16", "c_type": "half", "suffix": "f16", "load_type": "half8"},
    {"name": "BF16", "c_type": "__hip_bfloat16", "suffix": "bf16", "load_type": ""},
    {"name": "INT8", "c_type": "int8_t", "suffix": "i8", "load_type": "char4"},
]

PROBLEM_SIZES = [
    {"M": 512, "N": 512, "K": 512, "desc": "small"},
    {"M": 1024, "N": 1024, "K": 1024, "desc": "medium"},
    {"M": 4096, "N": 4096, "K": 4096, "desc": "large"},
    {"M": 8192, "N": 8192, "K": 4096, "desc": "xlarge"},
    {"M": 16384, "N": 16384, "K": 8192, "desc": "xxlarge (MI300X optimized)"},
]

INSTRUCTION_REPHRASINGS = [
    "Convert this {lang} code to an optimized HIP kernel for AMD MI300X (gfx942).",
    "Write a high-performance HIP implementation of the following {lang} code targeting AMD MI300X.",
    "Translate this {lang} operation to HIP, optimized for AMD Instinct MI300X with 192GB HBM3.",
    "Generate an AMD MI300X-optimized HIP kernel equivalent to this {lang} code. Use wavefront-64 primitives.",
    "Produce a hand-tuned HIP kernel for MI300X from this {lang} input. Maximize TFLOPS.",
    "Rewrite this {lang} code as an optimized HIP kernel. Target: gfx942 architecture, wavefront-64, MFMA intrinsics where applicable.",
    "Create an AMD GPU kernel (HIP/ROCm) for MI300X that implements this {lang} operation with maximum performance.",
]


def augment_dtype(example: dict) -> list[dict]:
    """Create variants with different data types."""
    augmented = []
    output = example.get("output", "")
    if not output or "float" not in output:
        return augmented

    for dtype in DTYPE_VARIANTS[1:]:  # skip float32 (original)
        new_example = copy.deepcopy(example)
        new_instruction = new_example["instruction"] + f" Use {dtype['name']} precision."

        new_example["instruction"] = new_instruction
        new_example["metadata"] = new_example.get("metadata", {})
        new_example["metadata"]["augmented"] = True
        new_example["metadata"]["augment_type"] = f"dtype_{dtype['suffix']}"
        augmented.append(new_example)

    return augmented


def augment_problem_size(example: dict) -> list[dict]:
    """Create variants with different problem size descriptions."""
    augmented = []
    instruction = example.get("instruction", "")

    for size in PROBLEM_SIZES:
        new_example = copy.deepcopy(example)
        size_note = (
            f"\n\nProblem dimensions: M={size['M']}, N={size['N']}, K={size['K']} "
            f"({size['desc']} problem). Optimize tile sizes accordingly."
        )
        new_example["instruction"] = instruction + size_note
        new_example["metadata"] = new_example.get("metadata", {})
        new_example["metadata"]["augmented"] = True
        new_example["metadata"]["augment_type"] = f"size_{size['desc']}"
        augmented.append(new_example)

    return augmented


def augment_instruction_phrasing(example: dict) -> list[dict]:
    """Rephrase the instruction while keeping input/output the same."""
    augmented = []
    input_code = example.get("input", "")

    if "cuda" in input_code.lower() or "cu" in example.get("metadata", {}).get("filename", ""):
        lang = "CUDA"
    elif "torch" in input_code.lower() or "import torch" in input_code:
        lang = "PyTorch"
    elif "triton" in input_code.lower() or "@triton" in input_code:
        lang = "Triton"
    else:
        lang = "GPU"

    phrasings = random.sample(INSTRUCTION_REPHRASINGS, min(3, len(INSTRUCTION_REPHRASINGS)))

    for phrasing in phrasings:
        new_example = copy.deepcopy(example)
        new_example["instruction"] = phrasing.format(lang=lang)
        new_example["metadata"] = new_example.get("metadata", {})
        new_example["metadata"]["augmented"] = True
        new_example["metadata"]["augment_type"] = "rephrased"
        augmented.append(new_example)

    return augmented


def load_processed_data() -> list[dict]:
    """Load all processed JSONL files."""
    all_data = []
    jsonl_files = sorted(PROCESSED_DIR.glob("*.jsonl"))

    for f in jsonl_files:
        if "augmented" in f.stem or "final" in f.stem:
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    all_data.append(json.loads(line))

    return all_data


def main():
    print("=" * 60)
    print("Augmenting training data")
    print("=" * 60)

    original_data = load_processed_data()
    print(f"[INFO] Loaded {len(original_data)} original examples")

    augmented = []

    for example in original_data:
        has_output = bool(example.get("output", "").strip())
        if not has_output:
            continue

        augmented.extend(augment_instruction_phrasing(example))

        if example.get("metadata", {}).get("type") in ("pytorch_to_hip", "cuda_to_hip", "optimized_kernel"):
            augmented.extend(augment_dtype(example))

        if "gemm" in example.get("metadata", {}).get("operation", ""):
            augmented.extend(augment_problem_size(example))

    final_dataset = original_data + augmented
    random.shuffle(final_dataset)

    split_idx = int(len(final_dataset) * 0.95)
    train_set = final_dataset[:split_idx]
    val_set = final_dataset[split_idx:]

    train_file = OUTPUT_DIR / "train.jsonl"
    val_file = OUTPUT_DIR / "val.jsonl"

    with open(train_file, "w") as f:
        for item in train_set:
            f.write(json.dumps(item) + "\n")

    with open(val_file, "w") as f:
        for item in val_set:
            f.write(json.dumps(item) + "\n")

    print(f"\n[RESULTS]")
    print(f"  Original examples:  {len(original_data)}")
    print(f"  Augmented examples: {len(augmented)}")
    print(f"  Total dataset:      {len(final_dataset)}")
    print(f"  Train split:        {len(train_set)} ({train_file})")
    print(f"  Val split:          {len(val_set)} ({val_file})")

    stats = {}
    for item in final_dataset:
        t = item.get("metadata", {}).get("type", "unknown")
        stats[t] = stats.get(t, 0) + 1

    print(f"\n[DISTRIBUTION]")
    for t, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")


if __name__ == "__main__":
    main()
