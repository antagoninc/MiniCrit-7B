"""Configuration management for MiniCrit training.

This module handles loading configuration from YAML files and environment
variables, with environment variables taking precedence over file values.

Example:
    >>> from src.config import load_config
    >>> config = load_config("configs/7b_lora.yaml")
    >>> print(config.model_name)
    'Qwen/Qwen2-7B-Instruct'

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) parameters.

    Attributes:
        r: Rank of the low-rank matrices. Higher values increase capacity
            but also memory usage. Typical values: 8, 16, 32, 64.
        alpha: Scaling factor for LoRA weights. Usually set to 2*r.
        dropout: Dropout probability for LoRA layers. Helps prevent overfitting.
        target_modules: List of module names to apply LoRA to. For transformer
            models, typically attention projection layers.
        bias: Whether to train bias parameters. Options: "none", "all", "lora_only".
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = "none"


@dataclass
class TrainingConfig:
    """Complete training configuration.

    This dataclass holds all configuration parameters needed for training,
    organized into logical groups.

    Attributes:
        model_name: HuggingFace model identifier or local path.
        data_file: Path to the training data CSV file.
        cache_dir: Directory for caching tokenized datasets.
        output_dir: Directory for saving checkpoints and final model.
        wandb_project: Weights & Biases project name for logging.
        learning_rate: Peak learning rate for the optimizer.
        batch_size: Per-device training batch size.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        epochs: Number of training epochs.
        max_length: Maximum sequence length for tokenization.
        warmup_steps: Number of warmup steps for learning rate scheduler.
        save_steps: Save checkpoint every N steps.
        log_steps: Log metrics every N steps.
        lora: LoRA configuration parameters.
    """

    # Model settings
    model_name: str = "Qwen/Qwen2-7B-Instruct"

    # Data settings
    data_file: str = "minicrit_11.7M_CLEAN.csv"
    cache_dir: str = "minicrit_11.7M_tokenized_QWEN7B"
    output_dir: str = "minicrit_7b_output"

    # Logging
    wandb_project: str = "minicrit-training"

    # Training hyperparameters
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    epochs: int = 1
    max_length: int = 512
    warmup_steps: int = 500
    save_steps: int = 2000
    log_steps: int = 50

    # Optimizer settings
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    optimizer: str = "adamw_torch"

    # LoRA settings
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    @property
    def effective_batch_size(self) -> int:
        """Calculate the effective batch size including gradient accumulation.

        Returns:
            Total effective batch size per optimization step.
        """
        return self.batch_size * self.gradient_accumulation_steps


def _parse_yaml(yaml_path: Path) -> dict[str, Any]:
    """Parse a YAML configuration file.

    Args:
        yaml_path: Path to the YAML file.

    Returns:
        Dictionary containing the parsed configuration.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
        ValueError: If the YAML file is malformed.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("PyYAML not installed, using empty config from file")
        return {}

    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path) as f:
        try:
            config_dict = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {yaml_path}: {e}") from e

    return config_dict


def _apply_env_overrides(config: TrainingConfig) -> TrainingConfig:
    """Apply environment variable overrides to configuration.

    Environment variables take precedence over file values. The following
    environment variables are supported:

    - MINICRIT_MODEL: Model name/path
    - MINICRIT_DATA_FILE: Training data file
    - MINICRIT_CACHE_DIR or MINICRIT_DATA_DIR: Cache directory
    - MINICRIT_OUTPUT_DIR or MINICRIT_CHECKPOINT_DIR: Output directory
    - WANDB_PROJECT: W&B project name
    - MINICRIT_LR: Learning rate
    - MINICRIT_BATCH_SIZE: Batch size
    - MINICRIT_GAS: Gradient accumulation steps
    - MINICRIT_EPOCHS: Number of epochs
    - MINICRIT_MAX_LENGTH: Max sequence length
    - MINICRIT_WARMUP: Warmup steps
    - MINICRIT_SAVE_STEPS: Checkpoint save frequency
    - MINICRIT_LOG_STEPS: Logging frequency
    - MINICRIT_LORA_R: LoRA rank
    - MINICRIT_LORA_ALPHA: LoRA alpha
    - MINICRIT_LORA_DROPOUT: LoRA dropout

    Args:
        config: Base configuration to apply overrides to.

    Returns:
        Configuration with environment overrides applied.
    """
    env_mappings = {
        "MINICRIT_MODEL": ("model_name", str),
        "MINICRIT_DATA_FILE": ("data_file", str),
        "MINICRIT_LR": ("learning_rate", float),
        "MINICRIT_BATCH_SIZE": ("batch_size", int),
        "MINICRIT_GAS": ("gradient_accumulation_steps", int),
        "MINICRIT_EPOCHS": ("epochs", int),
        "MINICRIT_MAX_LENGTH": ("max_length", int),
        "MINICRIT_WARMUP": ("warmup_steps", int),
        "MINICRIT_SAVE_STEPS": ("save_steps", int),
        "MINICRIT_LOG_STEPS": ("log_steps", int),
        "WANDB_PROJECT": ("wandb_project", str),
    }

    for env_var, (attr, type_fn) in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            setattr(config, attr, type_fn(value))
            logger.debug(f"Override from {env_var}: {attr}={value}")

    # Handle aliases for directories
    cache_dir = os.environ.get("MINICRIT_CACHE_DIR") or os.environ.get("MINICRIT_DATA_DIR")
    if cache_dir:
        config.cache_dir = cache_dir

    output_dir = os.environ.get("MINICRIT_OUTPUT_DIR") or os.environ.get("MINICRIT_CHECKPOINT_DIR")
    if output_dir:
        config.output_dir = output_dir

    # LoRA overrides
    lora_overrides = {
        "MINICRIT_LORA_R": ("r", int),
        "MINICRIT_LORA_ALPHA": ("alpha", int),
        "MINICRIT_LORA_DROPOUT": ("dropout", float),
    }

    for env_var, (attr, type_fn) in lora_overrides.items():
        value = os.environ.get(env_var)
        if value is not None:
            setattr(config.lora, attr, type_fn(value))
            logger.debug(f"Override from {env_var}: lora.{attr}={value}")

    return config


def load_config(yaml_path: str | Path | None = None) -> TrainingConfig:
    """Load training configuration from YAML file and environment variables.

    Configuration is loaded in the following order of precedence (highest first):
    1. Environment variables
    2. YAML file values
    3. Default values

    Args:
        yaml_path: Optional path to YAML configuration file. If None,
            only defaults and environment variables are used.

    Returns:
        Complete training configuration.

    Example:
        >>> config = load_config("configs/7b_lora.yaml")
        >>> config = load_config()  # Use defaults + env vars only
    """
    config = TrainingConfig()

    if yaml_path is not None:
        yaml_path = Path(yaml_path)
        if yaml_path.exists():
            logger.info(f"Loading config from {yaml_path}")
            config_dict = _parse_yaml(yaml_path)

            # Apply top-level settings
            for key, value in config_dict.items():
                if key == "lora" and isinstance(value, dict):
                    for lora_key, lora_value in value.items():
                        if hasattr(config.lora, lora_key):
                            setattr(config.lora, lora_key, lora_value)
                elif key == "model":
                    # Handle nested model config
                    if isinstance(value, dict):
                        if "base" in value:
                            config.model_name = value["base"]
                    else:
                        config.model_name = value
                elif key == "training" and isinstance(value, dict):
                    # Handle nested training config
                    training_mappings = {
                        "learning_rate": "learning_rate",
                        "batch_size": "batch_size",
                        "gradient_accumulation_steps": "gradient_accumulation_steps",
                        "epochs": "epochs",
                        "max_length": "max_length",
                        "warmup_steps": "warmup_steps",
                        "scheduler": "lr_scheduler_type",
                    }
                    for yaml_key, config_attr in training_mappings.items():
                        if yaml_key in value:
                            setattr(config, config_attr, value[yaml_key])
                elif hasattr(config, key):
                    setattr(config, key, value)
        else:
            logger.warning(f"Config file not found: {yaml_path}, using defaults")

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    return config
