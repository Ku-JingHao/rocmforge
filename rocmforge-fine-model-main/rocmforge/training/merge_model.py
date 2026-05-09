"""
Merge LoRA adapter weights into the base model.

After training, the model is saved as a small LoRA adapter (~200MB).
This script merges it into the full base model to create a standalone
model that can be deployed with vLLM without needing the adapter separately.

Usage:
    python training/merge_model.py
    python training/merge_model.py --adapter_path ./checkpoints/final-adapter
    python training/merge_model.py --push_to_hub your-username/rocmforge-7b
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="./checkpoints/final-adapter",
        help="Path to the trained LoRA adapter",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./rocmforge-7b-merged",
        help="Directory to save the merged model",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="HuggingFace Hub repo to push (e.g., 'username/rocmforge-7b')",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ROCmForge — Merge LoRA Adapter")
    print("=" * 60)
    print(f"\n  Base model:   {args.base_model}")
    print(f"  Adapter:      {args.adapter_path}")
    print(f"  Output:       {args.output_dir}")
    if args.push_to_hub:
        print(f"  Push to Hub:  {args.push_to_hub}")
    print("")

    if not os.path.exists(args.adapter_path):
        print(f"[ERROR] Adapter not found at {args.adapter_path}")
        print("        Run training first: python training/train.py")
        sys.exit(1)

    # --- Load base model ---
    print("[1/4] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # --- Load tokenizer ---
    print("[2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_path,
        trust_remote_code=True,
    )

    # --- Merge LoRA ---
    print("[3/4] Merging LoRA weights into base model...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.merge_and_unload()
    print("       Merge complete.")

    # --- Save ---
    print(f"[4/4] Saving merged model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n[DONE] Merged model saved to: {args.output_dir}")
    print(f"       Size: {sum(f.stat().st_size for f in Path(args.output_dir).rglob('*') if f.is_file()) / (1024**3):.2f} GB")

    # --- Optional: Push to Hub ---
    if args.push_to_hub:
        print(f"\n[HUB] Pushing to {args.push_to_hub}...")
        model.push_to_hub(args.push_to_hub, safe_serialization=True)
        tokenizer.push_to_hub(args.push_to_hub)
        print(f"[HUB] Done! Model available at: https://huggingface.co/{args.push_to_hub}")

    print("\n" + "=" * 60)
    print("  Next Steps:")
    print("=" * 60)
    print(f"  1. Test inference:  python training/test_inference.py")
    print(f"  2. Serve with vLLM: python -m vllm.entrypoints.openai.api_server --model {args.output_dir}")
    print(f"  3. Push to Hub:     python training/merge_model.py --push_to_hub your-name/rocmforge-7b")
    print("")


# Need Path for file size calculation
from pathlib import Path

if __name__ == "__main__":
    main()
