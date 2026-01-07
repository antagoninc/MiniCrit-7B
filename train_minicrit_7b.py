#!/usr/bin/env python3
"""MiniCrit-7B Training Script - Modular Version.

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
Co-Founder & CEO: William Alexander Ousley (Alex Ousley)
Co-Founder & CTO: Jacqueline Villamor Ousley (Jacque Ousley) TS/SCI

WATERMARK LAYER 1: ANTAGON-MINICRIT-7B-2026
WATERMARK LAYER 2: SHA256:ANTAGONINC-TRAINING-SCRIPT-V2
WATERMARK LAYER 3: MODEL-CAGE-17E75-UEI-KBSGT7CZ4AH3
WATERMARK LAYER 4: TIMESTAMP-{timestamp}
WATERMARK LAYER 5: BUILD-MINICRIT-QWEN2-7B-INSTRUCT

This script orchestrates the MiniCrit-7B training pipeline using modular
components from the src/ package. For configuration options, see
configs/7b_lora.yaml or use environment variables.

Example usage:
    # Basic training
    python train_minicrit_7b.py

    # With YAML config
    python train_minicrit_7b.py --config configs/7b_lora.yaml

    # Resume from checkpoint
    python train_minicrit_7b.py --resume-latest

    # Test with small sample
    python train_minicrit_7b.py --sample 1000 --validate-only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import wandb
from datasets import load_from_disk

# Import modular components
from src.config import load_config
from src.data import load_and_prepare_data, tokenize_dataset, validate_dataset
from src.model import apply_lora, load_model_and_tokenizer, require_gpu, save_model
from src.training import ProgressCallback, create_trainer, find_latest_checkpoint

# ---- Logging Setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed argument namespace with training options.
    """
    parser = argparse.ArgumentParser(
        description="MiniCrit-7B Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (e.g., configs/7b_lora.yaml)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N rows for testing",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate data, don't train",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint path",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Automatically resume from the latest checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    """Main training function.

    Orchestrates the complete training pipeline:
    1. Load configuration from YAML and/or environment variables
    2. Check GPU availability
    3. Load and tokenize data
    4. Validate dataset
    5. Initialize W&B logging
    6. Load model and apply LoRA
    7. Train model
    8. Save final checkpoint
    """
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Print banner
    logger.info("=" * 70)
    logger.info("MiniCrit-7B Training | Modular Architecture")
    logger.info("Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3")
    logger.info("=" * 70)

    logger.info(f"Model: {config.model_name}")
    logger.info(f"Dataset: {config.data_file}")
    logger.info(f"LR: {config.learning_rate}")
    logger.info(f"Effective Batch Size: {config.effective_batch_size}")
    logger.info(f"Max Length: {config.max_length}")
    logger.info(f"LoRA r={config.lora.r}, alpha={config.lora.alpha}")

    # Check GPU
    require_gpu()

    # Load tokenizer first (needed for tokenization)
    logger.info("Loading tokenizer...")
    from src.model import load_tokenizer
    tokenizer = load_tokenizer(config.model_name)

    # Load/tokenize data
    use_cache = args.sample is None
    cache_dir = config.cache_dir

    if use_cache and os.path.exists(cache_dir):
        logger.info(f"Loading cached dataset from {cache_dir}...")
        dataset_tok = load_from_disk(cache_dir)
        logger.info(f"Loaded {len(dataset_tok):,} examples from cache")
    else:
        df, text_col, rebuttal_col = load_and_prepare_data(
            config.data_file,
            sample_size=args.sample,
        )

        dataset_tok = tokenize_dataset(
            df=df,
            tokenizer=tokenizer,
            text_col=text_col,
            rebuttal_col=rebuttal_col,
            max_length=config.max_length,
            cache_dir=cache_dir if use_cache else None,
            use_cache=use_cache,
        )

    # Validate dataset
    if not validate_dataset(dataset_tok):
        logger.error("Dataset validation failed!")
        sys.exit(1)

    if args.validate_only:
        logger.info("Validate-only mode. Exiting.")
        sys.exit(0)

    # Determine checkpoint to resume from
    resume_checkpoint: str | None = None
    if args.resume:
        resume_checkpoint = args.resume
        logger.info(f"Will resume from specified checkpoint: {resume_checkpoint}")
    elif args.resume_latest:
        resume_checkpoint = find_latest_checkpoint(config.output_dir)
        if resume_checkpoint:
            logger.info(f"Will resume from latest checkpoint: {resume_checkpoint}")
        else:
            logger.info("No existing checkpoints found, starting fresh")

    # Initialize W&B
    logger.info("Initializing W&B...")
    run_name = f"minicrit-7b-{timestamp}"
    wandb.init(
        project=config.wandb_project,
        name=run_name,
        config={
            "model": config.model_name,
            "dataset": config.data_file,
            "lr": config.learning_rate,
            "batch_effective": config.effective_batch_size,
            "max_length": config.max_length,
            "lora_r": config.lora.r,
            "lora_alpha": config.lora.alpha,
            "resumed_from": resume_checkpoint,
        },
        resume="allow" if resume_checkpoint else None,
    )
    logger.info("W&B initialized")

    # Load model
    logger.info(f"Loading model: {config.model_name}")
    from src.model import load_base_model
    model = load_base_model(config.model_name)

    # Apply LoRA
    model = apply_lora(model, config.lora)

    # Create trainer
    trainer = create_trainer(
        model=model,
        dataset=dataset_tok,
        config=config,
        run_name=run_name,
    )

    # Train
    logger.info("=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save final model
    final_path = Path(config.output_dir) / "minicrit-7b-final"
    save_model(model, tokenizer, str(final_path))

    wandb.finish()

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
