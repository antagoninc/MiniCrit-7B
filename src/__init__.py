"""MiniCrit-7B Training Package.

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3

This package provides modular components for training the MiniCrit-7B
adversarial reasoning model.

Modules:
    config: Configuration loading from YAML files and environment variables.
    data: Data loading, tokenization, and validation utilities.
    model: Model loading and LoRA configuration.
    training: Training loop, callbacks, and checkpoint management.
"""

from src.config import load_config, TrainingConfig
from src.data import load_and_prepare_data, tokenize_dataset, validate_dataset
from src.model import load_model_and_tokenizer, apply_lora
from src.training import create_trainer, ProgressCallback

__all__ = [
    "load_config",
    "TrainingConfig",
    "load_and_prepare_data",
    "tokenize_dataset",
    "validate_dataset",
    "load_model_and_tokenizer",
    "apply_lora",
    "create_trainer",
    "ProgressCallback",
]

__version__ = "0.2.0"
