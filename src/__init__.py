"""MiniCrit-7B Training Package.

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3

This package provides modular components for training the MiniCrit-7B
adversarial reasoning model.

Modules:
    config: Configuration loading from YAML files and environment variables.
    data: Data loading, tokenization, and validation utilities.
    model: Model loading and LoRA configuration.
    training: Training loop, callbacks, and checkpoint management.
    evaluation: Model evaluation with ROUGE and BERTScore metrics.
    api: FastAPI server for model inference.
    logging_config: Structured logging configuration.
    budget: Budget and cost tracking.
"""

from src.config import load_config, TrainingConfig
from src.data import load_and_prepare_data, tokenize_dataset, validate_dataset
from src.model import load_model_and_tokenizer, apply_lora
from src.training import create_trainer, ProgressCallback
from src.evaluation import EvaluationResult, evaluate_model, compute_rouge_scores
from src.logging_config import setup_logging, get_logger, TrainingLogger
from src.budget import BudgetTracker, CostCalculator, get_tracker

__all__ = [
    # Config
    "load_config",
    "TrainingConfig",
    # Data
    "load_and_prepare_data",
    "tokenize_dataset",
    "validate_dataset",
    # Model
    "load_model_and_tokenizer",
    "apply_lora",
    # Training
    "create_trainer",
    "ProgressCallback",
    # Evaluation
    "EvaluationResult",
    "evaluate_model",
    "compute_rouge_scores",
    # Logging
    "setup_logging",
    "get_logger",
    "TrainingLogger",
    # Budget
    "BudgetTracker",
    "CostCalculator",
    "get_tracker",
]

__version__ = "0.3.0"
