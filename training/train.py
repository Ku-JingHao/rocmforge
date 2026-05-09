"""
ROCmForge — Fine-Tuning Script

Trains Qwen2.5-Coder-7B-Instruct with QLoRA on AMD MI300X.
Reads config from config.yaml and data from data/processed/*.jsonl.

Usage:
    python training/train.py
    python training/train.py --max_samples 500   # Quick test run
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from data_loader import prepare_dataset


def load_config(config_path: str = "training/config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def print_gpu_info():
    """Print GPU information."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"  GPU {i}: {name} ({mem_gb:.1f} GB)")
    else:
        print("  [WARNING] No GPU detected! Training will be extremely slow.")


def print_trainable_params(model):
    """Print trainable vs total parameters."""
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    pct = 100 * trainable / total
    print(f"  Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="ROCmForge Fine-Tuning")
    parser.add_argument("--config", type=str, default="training/config.yaml")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit training samples (for quick testing)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from checkpoint directory")
    parser.add_argument("--sanity_check", action="store_true",
                        help="Quick sanity-check mode (small eval/save intervals, "
                             "skip load_best_model_at_end)")
    args = parser.parse_args()

    # --- Load Config ---
    config = load_config(args.config)
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    system_prompt = config["system_prompt"]
    output_cfg = config["output"]

    max_samples = args.max_samples or data_cfg.get("max_samples")

    # In sanity-check mode, use small eval/save steps so they actually trigger
    # within a tiny dataset. Also disable load_best_model_at_end since with
    # only a few steps the eval may not produce a meaningful "best" checkpoint.
    if args.sanity_check:
        train_cfg["eval_steps"] = 5
        train_cfg["save_steps"] = 5
        train_cfg["logging_steps"] = 1
        train_cfg["num_train_epochs"] = 1
        train_cfg["load_best_model_at_end"] = False
        print("  [SANITY MODE] eval/save every 5 steps, 1 epoch, load_best disabled")

    print("=" * 60)
    print("  ROCmForge — Fine-Tuning")
    print("=" * 60)
    print(f"\n  Model:       {model_cfg['name']}")
    print(f"  LoRA rank:   {lora_cfg['r']}")
    print(f"  Epochs:      {train_cfg['num_train_epochs']}")
    print(f"  Batch size:  {train_cfg['per_device_train_batch_size']} × {train_cfg['gradient_accumulation_steps']} = {train_cfg['per_device_train_batch_size'] * train_cfg['gradient_accumulation_steps']}")
    print(f"  Max seq len: {train_cfg['max_seq_length']}")
    print(f"  Max samples: {max_samples or 'all'}")
    print(f"\n  GPUs:")
    print_gpu_info()
    print("")

    # --- Load Tokenizer ---
    print("[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Load Model ---
    print("[2/5] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=getattr(torch, model_cfg["torch_dtype"]),
        device_map="auto",
        attn_implementation=model_cfg.get("attn_implementation", "eager"),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # --- Apply LoRA ---
    print("[3/5] Applying LoRA adapter...")
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    print_trainable_params(model)

    # --- Load & Prepare Data ---
    print("\n[4/5] Preparing dataset...")
    dataset = prepare_dataset(
        train_file=data_cfg["train_file"],
        val_file=data_cfg["val_file"],
        system_prompt=system_prompt,
        tokenizer=tokenizer,
        max_samples=max_samples,
    )

    # --- Training Arguments ---
    print("\n[5/5] Starting training...")
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        optim=train_cfg["optim"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        report_to=train_cfg.get("report_to", "none"),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=train_cfg["max_seq_length"],
        packing=False,
    )

    # --- Train ---
    if args.resume_from:
        print(f"  Resuming from checkpoint: {args.resume_from}")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # --- Save Final LoRA Adapter ---
    final_adapter_dir = os.path.join(train_cfg["output_dir"], "final-adapter")
    print(f"\n[DONE] Saving LoRA adapter to {final_adapter_dir}")
    trainer.save_model(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    # --- Print Final Metrics ---
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    metrics = trainer.state.log_history
    train_losses = [m["loss"] for m in metrics if "loss" in m]
    eval_losses = [m["eval_loss"] for m in metrics if "eval_loss" in m]
    if train_losses:
        print(f"  Final train loss: {train_losses[-1]:.4f}")
    if eval_losses:
        print(f"  Final eval loss:  {eval_losses[-1]:.4f}")
        print(f"  Best eval loss:   {min(eval_losses):.4f}")
    print(f"\n  Adapter saved to: {final_adapter_dir}")
    print(f"  Next: Run 'python training/merge_model.py' to create deployable model")
    print("")


if __name__ == "__main__":
    main()
