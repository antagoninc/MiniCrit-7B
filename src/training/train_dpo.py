#!/usr/bin/env python3
# ================================================================
# MiniCrit DPO Training Script
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# Co-Founder & CEO: William Alexander Ousley (Alex Ousley)
# Co-Founder & CTO: Jacqueline Villamor Ousley (Jacque Ousley) TS/SCI
# ================================================================
# WATERMARK Layer 1: Antagon Inc. Proprietary
# WATERMARK Layer 2: MiniCrit DPO Training
# WATERMARK Layer 3: Direct Preference Optimization
# WATERMARK Layer 4: Hash SHA256:DPO_TRAIN_2026
# WATERMARK Layer 5: Build 20260112
# ================================================================

"""
Direct Preference Optimization (DPO) Training for MiniCrit

DPO fine-tunes the model to prefer good critiques over bad ones
without needing a separate reward model.

Usage:
    python train_dpo.py \
        --model /path/to/minicrit-7b \
        --data dpo_pairs.jsonl \
        --output minicrit-7b-dpo \
        --epochs 1

Requirements:
    pip install torch transformers peft trl datasets wandb
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig

import wandb

# ================================================================
# Configuration
# ================================================================

DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LR = 5e-5
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 4
DEFAULT_MAX_LENGTH = 512
DEFAULT_BETA = 0.1  # DPO temperature parameter

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("dpo-train")

# ================================================================
# Data Loading
# ================================================================


def load_dpo_data(filepath: str) -> Dataset:
    """Load DPO preference pairs from JSONL."""

    data: dict[str, list[str]] = {"prompt": [], "chosen": [], "rejected": []}

    with open(filepath, "r") as f:
        for line in f:
            item = json.loads(line)
            data["prompt"].append(item["prompt"])
            data["chosen"].append(item["chosen"])
            data["rejected"].append(item["rejected"])

    dataset = Dataset.from_dict(data)
    logger.info(f"Loaded {len(dataset)} DPO pairs from {filepath}")

    return dataset


# ================================================================
# Model Setup
# ================================================================


def setup_model(model_path: str, device: str = "auto"):
    """Load model with LoRA for DPO training."""

    logger.info(f"Loading model: {model_path}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=DEFAULT_LORA_R,
        lora_alpha=DEFAULT_LORA_ALPHA,
        lora_dropout=DEFAULT_LORA_DROPOUT,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)  # type: ignore[assignment]
    model.print_trainable_parameters()  # type: ignore[operator]

    logger.info(f"Model loaded on {next(model.parameters()).device}")

    return model, tokenizer


# ================================================================
# Training
# ================================================================


def train_dpo(
    model,
    tokenizer,
    train_dataset: Dataset,
    output_dir: str,
    epochs: int = 1,
    learning_rate: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    beta: float = DEFAULT_BETA,
    wandb_project: str = "minicrit-dpo",
):
    """Run DPO training."""

    # Initialize W&B
    run_name = f"dpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project=wandb_project, name=run_name)

    # DPO config
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        beta=beta,  # DPO temperature
        max_length=DEFAULT_MAX_LENGTH,
        max_prompt_length=DEFAULT_MAX_LENGTH // 2,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        report_to="wandb",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    # Trainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting DPO training...")
    trainer.train()

    # Save
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save LoRA weights separately
    lora_dir = os.path.join(output_dir, "lora")
    model.save_pretrained(lora_dir)
    logger.info(f"LoRA weights saved to {lora_dir}")

    wandb.finish()

    return trainer


# ================================================================
# Evaluation
# ================================================================


def quick_eval(model, tokenizer, test_cases: list) -> list[dict[str, Any]]:
    """Quick evaluation on test cases."""

    model.eval()
    results = []

    for test in test_cases:
        inputs = tokenizer(
            test["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=DEFAULT_MAX_LENGTH,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # Greedy for eval
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        results.append(
            {
                "prompt": test["prompt"][:100] + "...",
                "generated": generated[:200],
                "expected_valid": test.get("expected_valid"),
            }
        )

    return results


# ================================================================
# Main
# ================================================================


def main():
    parser = argparse.ArgumentParser(description="DPO training for MiniCrit")
    parser.add_argument("--model", required=True, help="Base model path")
    parser.add_argument("--data", required=True, help="DPO pairs JSONL")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="DPO beta")
    parser.add_argument("--device", default="auto", help="Device")
    parser.add_argument("--wandb-project", default="minicrit-dpo")

    args = parser.parse_args()

    print("=" * 60)
    print("🎯 MiniCrit DPO Training")
    print("   Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3")
    print("=" * 60)
    print(f"\nModel: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Beta: {args.beta}")

    # Load data
    train_dataset = load_dpo_data(args.data)

    # Setup model
    model, tokenizer = setup_model(args.model, args.device)

    # Train
    trainer = train_dpo(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=args.output,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        beta=args.beta,
        wandb_project=args.wandb_project,
    )

    # Quick eval
    test_cases = [
        {
            "prompt": "### Domain: trading\n### Rationale:\nStock will go up because it went up yesterday.\n\n### Critique:\n",
            "expected_valid": False,
        },
        {
            "prompt": "### Domain: trading\n### Rationale:\nBased on fundamental analysis showing P/E below industry average, consistent revenue growth, and strong balance sheet, I recommend moderate position with 60% confidence.\n\n### Critique:\n",
            "expected_valid": True,
        },
    ]

    print("\n📊 Quick Evaluation:")
    results = quick_eval(model, tokenizer, test_cases)
    for r in results:
        print(f"\nPrompt: {r['prompt']}")
        print(f"Generated: {r['generated']}")

    print(f"\n{'=' * 60}")
    print(f"✅ DPO Training Complete!")
    print(f"   Model saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
