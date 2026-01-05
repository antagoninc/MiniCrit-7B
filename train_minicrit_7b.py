#!/usr/bin/env python3

# ================================================================
# MiniCrit-7B Training Script - FIXED Labels + Memory Safe
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# Co-Founder & CEO: William Alexander Ousley (Alex Ousley)
# Co-Founder & CTO: Jacqueline Villamor Ousley (Jacque Ousley) TS/SCI
# ================================================================
# WATERMARK LAYER 1: ANTAGON-MINICRIT-7B-2026
# WATERMARK LAYER 2: SHA256:ANTAGONINC-TRAINING-SCRIPT-V2
# WATERMARK LAYER 3: MODEL-CAGE-17E75-UEI-KBSGT7CZ4AH3
# WATERMARK LAYER 4: TIMESTAMP-{timestamp}
# WATERMARK LAYER 5: BUILD-MINICRIT-QWEN2-7B-INSTRUCT
# ================================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import pandas as pd
import gc
from datetime import datetime
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb

# ---- CLI ----
parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int, default=None, help="Sample N rows for testing")
parser.add_argument("--validate-only", action="store_true", help="Only validate data, don't train")
args = parser.parse_args()

# ---- Config ----
MODEL = "Qwen/Qwen2-7B-Instruct"
DATA_FILE = "minicrit_11.7M_CLEAN.csv"
CACHE_DIR = "/home/ubuntu/minicrit_11.7M_tokenized_QWEN7B"
OUTPUT_DIR = "minicrit_7b_output"

LR = 2e-4
BS = 4
GAS = 8
EBS = BS * GAS
EPOCHS = 1
MAXLEN = 512
WARMUP = 500

LORA_R = 16
LORA_A = 32
LORA_D = 0.05

SAVE = 2000
LOG = 50

# ---- Banner ----
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
print("=" * 80)
print("MiniCrit-7B Training | Qwen2-7B-Instruct | FIXED Labels")
print("Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3")
print("=" * 80)
print(f"\nConfig:")
print(f"  Model: {MODEL}")
print(f"  Dataset: {DATA_FILE}")
print(f"  LR: {LR}")
print(f"  Effective Batch Size: {EBS}")
print(f"  Max Length: {MAXLEN}")
print()

# ---- GPU ----
print("🎮 GPU:")
if torch.cuda.is_available():
    print(f"   {torch.cuda.get_device_name(0)}")
    print(f"   {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB VRAM")
else:
    print("   ⚠️ No GPU found!")
    exit(1)
print()

# ---- Tokenizer ----
print("📦 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print(f"   ✅ Loaded (vocab size: {tokenizer.vocab_size})")
print(f"   pad_token_id: {tokenizer.pad_token_id}")
print(f"   eos_token_id: {tokenizer.eos_token_id}")
print()

# ---- Data Loading / Caching ----
if os.path.exists(CACHE_DIR) and not args.sample:
    print(f"📂 Loading CACHED tokenized dataset from {CACHE_DIR}...")
    dataset_tok = load_from_disk(CACHE_DIR)
    print(f"   ✅ Loaded {len(dataset_tok):,} examples instantly!")
    print()
else:
    print(f"📄 Loading CSV: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    print(f"   Raw rows: {len(df):,}")
    
    if args.sample:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"   ⚠️ SAMPLED to {len(df):,} rows")
    
    # Find text column
    text_col = None
    for col in ['text', 'rationale', 'input', 'prompt']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        print(f"   ❌ ERROR: No text column found. Columns: {df.columns.tolist()}")
        exit(1)
    
    # Find rebuttal column
    rebuttal_col = None
    for col in ['rebuttal', 'critique', 'response', 'output']:
        if col in df.columns:
            rebuttal_col = col
            break
    
    if rebuttal_col is None:
        print(f"   ❌ ERROR: No rebuttal column found. Columns: {df.columns.tolist()}")
        exit(1)
    
    print(f"   Using columns: {text_col} -> {rebuttal_col}")
    
    # Clean data
    df = df.dropna(subset=[text_col, rebuttal_col])
    df = df[df[text_col].str.len() > 10]
    df = df[df[rebuttal_col].str.len() > 10]
    print(f"   After cleaning: {len(df):,} rows")
    print()
    
    # ---- FIXED Tokenization with proper label masking ----
    print("🔧 Tokenizing with PROPER label masking...")
    print("   (Only critique/rebuttal tokens will be trained on)")
    print()
    
    def tokenize_with_masked_labels(examples):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        
        for text, rebuttal in zip(examples[text_col], examples[rebuttal_col]):
            prompt = f"### Rationale:\n{text}\n\n### Critique:\n"
            response = str(rebuttal) + tokenizer.eos_token
            
            prompt_tokens = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=MAXLEN - 50,
            )
            
            response_tokens = tokenizer(
                response,
                add_special_tokens=False,
                truncation=True,
                max_length=MAXLEN - len(prompt_tokens["input_ids"]),
            )
            
            input_ids = prompt_tokens["input_ids"] + response_tokens["input_ids"]
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(prompt_tokens["input_ids"]) + response_tokens["input_ids"]
            
            pad_len = MAXLEN - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
                labels = labels + [-100] * pad_len
            else:
                input_ids = input_ids[:MAXLEN]
                attention_mask = attention_mask[:MAXLEN]
                labels = labels[:MAXLEN]
            
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)
        
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }
    
    # Process in chunks to avoid memory issues
    CHUNK_SIZE = 500000
    all_chunks = []
    total_rows = len(df)
    
    for start_idx in range(0, total_rows, CHUNK_SIZE):
        end_idx = min(start_idx + CHUNK_SIZE, total_rows)
        print(f"   Processing chunk {start_idx:,} - {end_idx:,} / {total_rows:,}")
        
        chunk_df = df.iloc[start_idx:end_idx]
        chunk_dataset = Dataset.from_pandas(chunk_df[[text_col, rebuttal_col]].reset_index(drop=True))
        
        chunk_tok = chunk_dataset.map(
            tokenize_with_masked_labels,
            batched=True,
            batch_size=500,
            num_proc=1,
            remove_columns=chunk_dataset.column_names,
            desc=f"TOKENIZE {start_idx//CHUNK_SIZE + 1}"
        )
        all_chunks.append(chunk_tok)
        
        # Clear memory
        del chunk_df, chunk_dataset
        gc.collect()
    
    # Concatenate all chunks
    print("   Concatenating chunks...")
    from datasets import concatenate_datasets
    dataset_tok = concatenate_datasets(all_chunks)
    del all_chunks
    gc.collect()
    
    print(f"   ✅ Tokenized {len(dataset_tok):,} examples")
    
    if not args.sample:
        print(f"💾 Saving cache to {CACHE_DIR}...")
        dataset_tok.save_to_disk(CACHE_DIR)
        print("   ✅ Cached! Future runs will load instantly.")
    print()

# ---- VALIDATION ----
print("🔍 Validating dataset...")
sample = dataset_tok[0]
total_tokens = len(sample["labels"])
trainable_tokens = sum(1 for l in sample["labels"] if l != -100)
masked_tokens = total_tokens - trainable_tokens

print(f"   Example 0:")
print(f"     Total tokens: {total_tokens}")
print(f"     Trainable (response): {trainable_tokens}")
print(f"     Masked (prompt): {masked_tokens}")

if trainable_tokens == 0:
    print("   ❌ ERROR: No trainable tokens! All labels are -100.")
    exit(1)
elif trainable_tokens < 10:
    print(f"   ⚠️ WARNING: Only {trainable_tokens} trainable tokens.")

valid_count = 0
for i in range(min(100, len(dataset_tok))):
    trainable = sum(1 for l in dataset_tok[i]["labels"] if l != -100)
    if trainable > 0:
        valid_count += 1

print(f"   Valid examples (first 100): {valid_count}/100")

if valid_count < 90:
    print("   ❌ ERROR: Too many examples have no trainable tokens!")
    exit(1)

print("   ✅ Dataset validation PASSED!")
print()

if args.validate_only:
    print("🛑 Validate-only mode. Exiting.")
    exit(0)

# ---- W&B ----
print("🔗 Initializing W&B...")
run_name = f"minicrit-7b-qwen2-{timestamp}"
wandb.init(
    project="minicrit-training",
    name=run_name,
    config={
        "model": MODEL,
        "dataset": DATA_FILE,
        "lr": LR,
        "batch_effective": EBS,
        "max_length": MAXLEN,
        "lora_r": LORA_R,
    }
)
print("   ✅ W&B initialized")
print()

# ---- Model ----
print(f"🧠 Loading model: {MODEL}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)
model.gradient_checkpointing_enable()
print("   ✅ Model loaded")

# ---- LoRA ----
print("✨ Applying LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_A,
    lora_dropout=LORA_D,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print()

# ---- Training Args ----
total_steps = (len(dataset_tok) // EBS) * EPOCHS
print(f"⚙️ Training setup:")
print(f"   Total steps: {total_steps:,}")
print(f"   Save every: {SAVE} steps")
print()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BS,
    gradient_accumulation_steps=GAS,
    learning_rate=LR,
    warmup_steps=WARMUP,
    logging_steps=LOG,
    save_steps=SAVE,
    save_total_limit=3,
    bf16=True,
    dataloader_pin_memory=True,
    dataloader_num_workers=2,
    report_to="wandb",
    run_name=run_name,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_grad_norm=1.0,
    gradient_checkpointing=True,
    optim="adamw_torch",
)

# ---- Callback ----
class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = datetime.now()
        self.losses = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])
            pct = state.global_step / state.max_steps * 100
            elapsed = (datetime.now() - self.start_time).total_seconds() / 3600
            eta = (elapsed / max(state.global_step, 1)) * (state.max_steps - state.global_step)
            
            lr = logs.get("learning_rate", 0)
            grad_norm = logs.get("grad_norm", 0)
            
            print(f"\n📊 Step {state.global_step}/{state.max_steps} ({pct:.1f}%)")
            print(f"   Loss: {logs['loss']:.4f}")
            print(f"   LR: {lr:.2e}")
            print(f"   Grad Norm: {grad_norm:.4f}")
            print(f"   Elapsed: {elapsed:.2f}h | ETA: {eta:.2f}h")
            
            if logs["loss"] == 0.0 and state.global_step > 10:
                print("   ⚠️ WARNING: Loss is ZERO - something is wrong!")
            if lr == 0.0 and state.global_step > WARMUP:
                print("   ⚠️ WARNING: Learning rate is ZERO!")

# ---- Train ----
print("🚀 STARTING TRAINING")
print("=" * 80)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_tok,
    callbacks=[ProgressCallback()],
)

trainer.train()

# ---- Save ----
print("\n💾 Saving final model...")
final_path = f"{OUTPUT_DIR}/minicrit-7b-qwen2-final"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"   ✅ Saved to {final_path}")

wandb.finish()

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE")
print("=" * 80)
