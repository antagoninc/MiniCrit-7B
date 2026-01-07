"""Utility functions for MiniCrit training.

These functions are extracted to enable testing without heavy ML dependencies.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MiniCrit-7B Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="Resume training from checkpoint path (e.g., 'minicrit_7b_output/checkpoint-30000')",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Automatically resume from the latest checkpoint in output directory",
    )
    return parser.parse_args()


def get_config() -> dict[str, Any]:
    """Load configuration from environment variables with defaults."""
    return {
        "model": os.environ.get("MINICRIT_MODEL", "Qwen/Qwen2-7B-Instruct"),
        "data_file": os.environ.get("MINICRIT_DATA_FILE", "minicrit_11.7M_CLEAN.csv"),
        "cache_dir": os.environ.get(
            "MINICRIT_CACHE_DIR",
            os.environ.get("MINICRIT_DATA_DIR", "minicrit_11.7M_tokenized_QWEN7B"),
        ),
        "output_dir": os.environ.get(
            "MINICRIT_OUTPUT_DIR",
            os.environ.get("MINICRIT_CHECKPOINT_DIR", "minicrit_7b_output"),
        ),
        "wandb_project": os.environ.get("WANDB_PROJECT", "minicrit-training"),
        "learning_rate": float(os.environ.get("MINICRIT_LR", "2e-4")),
        "batch_size": int(os.environ.get("MINICRIT_BATCH_SIZE", "4")),
        "gradient_accumulation_steps": int(os.environ.get("MINICRIT_GAS", "8")),
        "epochs": int(os.environ.get("MINICRIT_EPOCHS", "1")),
        "max_length": int(os.environ.get("MINICRIT_MAX_LENGTH", "512")),
        "warmup_steps": int(os.environ.get("MINICRIT_WARMUP", "500")),
        "lora_r": int(os.environ.get("MINICRIT_LORA_R", "16")),
        "lora_alpha": int(os.environ.get("MINICRIT_LORA_ALPHA", "32")),
        "lora_dropout": float(os.environ.get("MINICRIT_LORA_DROPOUT", "0.05")),
        "save_steps": int(os.environ.get("MINICRIT_SAVE_STEPS", "2000")),
        "log_steps": int(os.environ.get("MINICRIT_LOG_STEPS", "50")),
    }


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint in the output directory.

    Args:
        output_dir: Directory containing checkpoints.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = [
        d for d in output_path.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]

    if not checkpoints:
        return None

    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
    latest = checkpoints[-1]
    logger.info(f"Found latest checkpoint: {latest}")
    return str(latest)


def find_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Find text and rebuttal columns in the dataframe.

    Args:
        df: Input dataframe.

    Returns:
        Tuple of (text_column_name, rebuttal_column_name).

    Raises:
        ValueError: If required columns are not found.
    """
    text_col = None
    for col in ["text", "rationale", "input", "prompt"]:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError(f"No text column found. Available columns: {df.columns.tolist()}")

    rebuttal_col = None
    for col in ["rebuttal", "critique", "response", "output"]:
        if col in df.columns:
            rebuttal_col = col
            break

    if rebuttal_col is None:
        raise ValueError(f"No rebuttal column found. Available columns: {df.columns.tolist()}")

    return text_col, rebuttal_col


def validate_labels(labels: list[int]) -> tuple[int, int]:
    """Count trainable vs masked tokens in a label sequence.

    Args:
        labels: List of label token IDs (-100 for masked).

    Returns:
        Tuple of (trainable_count, masked_count).
    """
    trainable = sum(1 for label in labels if label != -100)
    masked = len(labels) - trainable
    return trainable, masked
