"""
Quick inference test after training.

Tests the merged model (or adapter) with sample prompts to verify
the fine-tuning worked before deploying to vLLM.

Usage:
    python training/test_inference.py
    python training/test_inference.py --model_path ./rocmforge-7b-merged
    python training/test_inference.py --use_adapter --adapter_path ./checkpoints/final-adapter
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

SYSTEM_PROMPT = """You are ROCmForge, an expert AMD GPU kernel optimizer. You specialize in:
- Converting PyTorch, CUDA, and Triton code to optimized HIP kernels
- Targeting AMD Instinct MI300X (gfx942) architecture
- Using wavefront-64 operations, MFMA intrinsics, and LDS optimization
- Maximizing TFLOPS and memory bandwidth utilization
Generate compilable, correct, high-performance HIP/ROCm code."""

TEST_PROMPTS = [
    {
        "name": "PyTorch matmul → HIP",
        "user": (
            "Convert this PyTorch operation to an optimized HIP kernel for AMD MI300X.\n\n"
            "import torch\n"
            "def matmul(A, B):\n"
            "    # A: [2048, 4096] float16, B: [4096, 2048] float16\n"
            "    return torch.matmul(A, B)"
        ),
    },
    {
        "name": "CUDA softmax → HIP",
        "user": (
            "Convert this CUDA kernel to optimized HIP for AMD MI300X:\n\n"
            "__global__ void softmax(float* input, float* output, int N) {\n"
            "    // Uses warp size 32 — fix for AMD\n"
            "    float max_val = input[threadIdx.x];\n"
            "    for (int offset = 16; offset > 0; offset >>= 1)\n"
            "        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));\n"
            "}"
        ),
    },
    {
        "name": "Natural language → HIP",
        "user": (
            "Write an optimized HIP kernel for AMD MI300X that computes "
            "element-wise ReLU activation on a float16 tensor of 8 million elements. "
            "Maximize HBM3 bandwidth utilization using vectorized loads."
        ),
    },
]


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
    """Run a single inference."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser(description="Test ROCmForge inference")
    parser.add_argument("--model_path", type=str, default="./rocmforge-7b-merged")
    parser.add_argument("--use_adapter", action="store_true",
                        help="Load base model + adapter instead of merged model")
    parser.add_argument("--adapter_path", type=str, default="./checkpoints/final-adapter")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    args = parser.parse_args()

    print("=" * 60)
    print("  ROCmForge — Inference Test")
    print("=" * 60)

    # Load model
    if args.use_adapter:
        print(f"\n  Loading base model: {args.base_model}")
        print(f"  Loading adapter: {args.adapter_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, args.adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    else:
        print(f"\n  Loading merged model: {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("  Model loaded.\n")

    # Run test prompts
    for i, test in enumerate(TEST_PROMPTS):
        print(f"\n{'─' * 60}")
        print(f"  Test {i+1}: {test['name']}")
        print(f"{'─' * 60}")
        print(f"\n  [USER] {test['user'][:100]}...")
        print(f"\n  [ROCMFORGE OUTPUT]:")
        print("")

        response = run_inference(model, tokenizer, test["user"])
        print(response[:2000])

        if len(response) > 2000:
            print(f"\n  ... (truncated, {len(response)} chars total)")

    print(f"\n{'═' * 60}")
    print("  Inference test complete.")
    print(f"{'═' * 60}")

    # Quick quality checks
    print("\n  Quality indicators to look for:")
    print("  ✓ Uses #include <hip/hip_runtime.h> (not cuda_runtime.h)")
    print("  ✓ Uses wavefront-64 (not warp-32)")
    print("  ✓ Uses __shfl_xor with width 64")
    print("  ✓ References MI300X, gfx942, or MFMA")
    print("  ✓ Code looks compilable (proper syntax)")
    print("")


if __name__ == "__main__":
    main()
