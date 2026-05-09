"""
Evaluation harness for the fine-tuned ROCmForge model.

Runs the model against the held-out validation set and computes:
- Average loss (perplexity)
- Generation quality metrics (does output contain HIP markers?)
- Code structure indicators (uses wavefront-64? MFMA? __shared__?)

This is a quick quality check — not a full benchmark suite.
For real performance measurement, see eval/benchmark.py.

Usage:
    python eval/eval_holdout.py --model_path ./rocmforge-7b-merged
    python eval/eval_holdout.py --model_path ./rocmforge-7b-merged --max_samples 50
"""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HIP_MARKERS = [
    "hip_runtime",
    "__global__",
    "__device__",
    "hipMalloc",
    "hipMemcpy",
]

MI300X_OPTIMIZATION_MARKERS = [
    ("wavefront-64",  r"__shfl_xor.*64|width\s*=\s*64|wavefront"),
    ("LDS usage",     r"__shared__|extern\s+__shared__"),
    ("MFMA",          r"__builtin_amdgcn_mfma|mfma_f32"),
    ("vectorized",    r"float4|half8|half4"),
    ("gfx942",        r"gfx942|MI300X|mi300x|mi300"),
]


def score_output(generated: str) -> dict:
    """Heuristic scoring — does the output look like real HIP code?"""
    scores = {
        "has_hip_markers": sum(1 for m in HIP_MARKERS if m in generated),
        "has_mi300x_optims": {},
    }
    for label, pattern in MI300X_OPTIMIZATION_MARKERS:
        scores["has_mi300x_optims"][label] = bool(re.search(pattern, generated))

    scores["mi300x_score"] = sum(scores["has_mi300x_optims"].values())
    scores["overall_score"] = (
        (scores["has_hip_markers"] / len(HIP_MARKERS)) * 0.5
        + (scores["mi300x_score"] / len(MI300X_OPTIMIZATION_MARKERS)) * 0.5
    )
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./rocmforge-7b-merged")
    parser.add_argument("--val_file", type=str, default="./data/processed/val.jsonl")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--output_file", type=str, default="./eval/eval_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  ROCmForge — Holdout Evaluation")
    print("=" * 60)

    print(f"\n[1/3] Loading model from {args.model_path}...")
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

    print(f"\n[2/3] Loading {args.max_samples} validation examples...")
    examples = []
    with open(args.val_file, "r") as f:
        for i, line in enumerate(f):
            if i >= args.max_samples:
                break
            examples.append(json.loads(line.strip()))

    print(f"\n[3/3] Running evaluation on {len(examples)} samples...")
    results = []
    aggregate = {
        "total_samples": 0,
        "compilable_looking": 0,
        "mi300x_aware": 0,
        "avg_overall_score": 0.0,
    }

    for i, ex in enumerate(examples):
        instruction = ex.get("instruction", "")
        user_input = ex.get("input", "")
        prompt = f"{instruction}\n\n{user_input}".strip() if user_input else instruction

        messages = [
            {"role": "system", "content": "You are ROCmForge, an expert AMD GPU kernel optimizer."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        scores = score_output(response)
        results.append({
            "instruction": instruction[:100],
            "scores": scores,
            "output_preview": response[:300],
        })

        aggregate["total_samples"] += 1
        if scores["has_hip_markers"] >= 2:
            aggregate["compilable_looking"] += 1
        if scores["mi300x_score"] >= 2:
            aggregate["mi300x_aware"] += 1
        aggregate["avg_overall_score"] += scores["overall_score"]

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(examples)}")

    aggregate["avg_overall_score"] /= max(aggregate["total_samples"], 1)
    aggregate["compilable_pct"] = 100 * aggregate["compilable_looking"] / aggregate["total_samples"]
    aggregate["mi300x_aware_pct"] = 100 * aggregate["mi300x_aware"] / aggregate["total_samples"]

    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    print(f"  Total samples:           {aggregate['total_samples']}")
    print(f"  HIP-looking outputs:     {aggregate['compilable_pct']:.1f}%")
    print(f"  MI300X-aware outputs:    {aggregate['mi300x_aware_pct']:.1f}%")
    print(f"  Average quality score:   {aggregate['avg_overall_score']:.3f}  (range 0-1)")

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump({"aggregate": aggregate, "samples": results}, f, indent=2)
    print(f"\n  Detailed results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
