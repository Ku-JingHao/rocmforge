"""
Data loader for ROCmForge training.

Converts raw JSONL (instruction/input/output format) into the ChatML format
expected by Qwen2.5-Coder-Instruct models.

Each training example becomes:
<|im_start|>system
You are ROCmForge, an expert AMD GPU kernel optimizer...
<|im_end|>
<|im_start|>user
{instruction}\n\n{input}
<|im_end|>
<|im_start|>assistant
{output}
<|im_end|>
"""

import json
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict


def load_jsonl(filepath: str, max_samples: Optional[int] = None) -> list[dict]:
    """Load JSONL file into list of dicts."""
    data = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_chat_message(example: dict, system_prompt: str) -> dict:
    """
    Convert a single training example into Qwen ChatML format.

    Input format:  {"instruction": "...", "input": "...", "output": "..."}
    Output format: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    if not output_text.strip():
        return None

    if input_text.strip():
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content.strip()},
        {"role": "assistant", "content": output_text.strip()},
    ]

    return {"messages": messages}


def format_to_text(example: dict, tokenizer) -> dict:
    """
    Apply the tokenizer's chat template to produce a single text string.
    This is what SFTTrainer needs when using dataset_text_field="text".
    """
    if example.get("messages"):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}
    return {"text": ""}


def prepare_dataset(
    train_file: str,
    val_file: str,
    system_prompt: str,
    tokenizer,
    max_samples: Optional[int] = None,
) -> DatasetDict:
    """
    Load JSONL files and prepare them for SFTTrainer.

    Returns a DatasetDict with 'train' and 'validation' splits,
    each containing a 'text' field with the full chat-formatted string.
    """
    print(f"[DATA] Loading training data from {train_file}")
    train_raw = load_jsonl(train_file, max_samples)
    print(f"[DATA] Loading validation data from {val_file}")
    val_raw = load_jsonl(val_file, max_samples // 20 if max_samples else None)

    print(f"[DATA] Raw examples — train: {len(train_raw)}, val: {len(val_raw)}")

    train_formatted = []
    skipped = 0
    for ex in train_raw:
        result = format_chat_message(ex, system_prompt)
        if result:
            train_formatted.append(result)
        else:
            skipped += 1

    val_formatted = []
    for ex in val_raw:
        result = format_chat_message(ex, system_prompt)
        if result:
            val_formatted.append(result)

    print(f"[DATA] After formatting — train: {len(train_formatted)}, val: {len(val_formatted)}, skipped: {skipped}")

    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)

    print("[DATA] Applying chat template...")
    train_dataset = train_dataset.map(
        lambda ex: format_to_text(ex, tokenizer),
        desc="Formatting train",
    )
    val_dataset = val_dataset.map(
        lambda ex: format_to_text(ex, tokenizer),
        desc="Formatting val",
    )

    train_dataset = train_dataset.filter(lambda ex: len(ex["text"]) > 50)
    val_dataset = val_dataset.filter(lambda ex: len(ex["text"]) > 50)

    print(f"[DATA] Final — train: {len(train_dataset)}, val: {len(val_dataset)}")

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
    })


if __name__ == "__main__":
    """Quick test: verify data loading works without GPU."""
    from transformers import AutoTokenizer

    SYSTEM_PROMPT = "You are ROCmForge, an expert AMD GPU kernel optimizer."

    print("Loading tokenizer (CPU only, for testing)...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        trust_remote_code=True,
    )

    ds = prepare_dataset(
        train_file="./data/processed/train.jsonl",
        val_file="./data/processed/val.jsonl",
        system_prompt=SYSTEM_PROMPT,
        tokenizer=tokenizer,
        max_samples=100,
    )

    print("\n[SAMPLE] First training example (truncated):")
    print(ds["train"][0]["text"][:500])
    print("...")

    lengths = [len(tokenizer.encode(ex["text"])) for ex in ds["train"]]
    print(f"\n[STATS] Token lengths — min: {min(lengths)}, max: {max(lengths)}, "
          f"avg: {sum(lengths)/len(lengths):.0f}")
