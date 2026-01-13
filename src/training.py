"""Training loop and callbacks for MiniCrit training.

This module provides the training loop, progress callbacks, early stopping,
and checkpoint management functionality.

Example:
    >>> from src.training import create_trainer, ProgressCallback
    >>> trainer = create_trainer(model, tokenizer, dataset, config)
    >>> trainer.train()

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict
    from peft import PeftModel

    from src.config import TrainingConfig

logger = logging.getLogger(__name__)


class ProgressCallback(TrainerCallback):
    """Custom callback for logging detailed training progress.

    This callback logs training metrics at each logging step, including
    loss, learning rate, gradient norm, elapsed time, and ETA.

    Attributes:
        start_time: When training started.
        losses: List of loss values from each log.
        warmup_steps: Number of warmup steps (for LR zero warning).

    Example:
        >>> callback = ProgressCallback(warmup_steps=500)
        >>> trainer = Trainer(..., callbacks=[callback])
    """

    def __init__(self, warmup_steps: int = 500) -> None:
        """Initialize the progress callback.

        Args:
            warmup_steps: Number of warmup steps. Used to suppress
                "learning rate is zero" warnings during warmup.
        """
        self.start_time = datetime.now()
        self.losses: list[float] = []
        self.eval_losses: list[float] = []
        self.warmup_steps = warmup_steps

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log training progress on each logging step.

        Args:
            args: Training arguments.
            state: Current trainer state with step info.
            control: Trainer control object.
            logs: Dictionary of metrics from this logging step.
            **kwargs: Additional keyword arguments (unused).
        """
        if logs:
            # Track training loss
            if "loss" in logs:
                self.losses.append(logs["loss"])

                # Calculate progress metrics
                pct = state.global_step / state.max_steps * 100
                elapsed = (datetime.now() - self.start_time).total_seconds() / 3600
                steps_remaining = state.max_steps - state.global_step
                eta = (elapsed / max(state.global_step, 1)) * steps_remaining

                lr = logs.get("learning_rate", 0)
                grad_norm = logs.get("grad_norm", 0)

                logger.info(
                    f"Step {state.global_step}/{state.max_steps} ({pct:.1f}%) | "
                    f"Loss: {logs['loss']:.4f} | LR: {lr:.2e} | "
                    f"Grad: {grad_norm:.4f} | Elapsed: {elapsed:.2f}h | ETA: {eta:.2f}h"
                )

                # Warn about potential issues
                if logs["loss"] == 0.0 and state.global_step > 10:
                    logger.warning("Loss is ZERO - something may be wrong!")

                if lr == 0.0 and state.global_step > self.warmup_steps:
                    logger.warning("Learning rate is ZERO!")

            # Track eval loss
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
                logger.info(f"Eval Loss: {logs['eval_loss']:.4f}")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Reset start time when training begins.

        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control object.
            **kwargs: Additional keyword arguments (unused).
        """
        self.start_time = datetime.now()
        logger.info("Training started")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Log final statistics when training ends.

        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control object.
            **kwargs: Additional keyword arguments (unused).
        """
        total_time = (datetime.now() - self.start_time).total_seconds() / 3600

        if self.losses:
            initial_loss = self.losses[0]
            final_loss = self.losses[-1]
            reduction = (1 - final_loss / initial_loss) * 100 if initial_loss > 0 else 0

            logger.info(f"Training completed in {total_time:.2f} hours")
            logger.info(f"Initial train loss: {initial_loss:.4f}")
            logger.info(f"Final train loss: {final_loss:.4f}")
            logger.info(f"Train loss reduction: {reduction:.1f}%")

        if self.eval_losses:
            logger.info(f"Best eval loss: {min(self.eval_losses):.4f}")
            logger.info(f"Final eval loss: {self.eval_losses[-1]:.4f}")


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint in the output directory.

    Searches for checkpoint directories (named checkpoint-XXXXX) and
    returns the one with the highest step number.

    Args:
        output_dir: Directory containing checkpoints.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found.

    Example:
        >>> latest = find_latest_checkpoint("output/")
        >>> if latest:
        ...     trainer.train(resume_from_checkpoint=latest)
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = [
        d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")
    ]

    if not checkpoints:
        return None

    # Sort by step number (checkpoint-XXXXX)
    checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
    latest = checkpoints[-1]

    logger.info(f"Found latest checkpoint: {latest}")
    return str(latest)


def create_training_args(
    config: TrainingConfig,
    run_name: str,
    eval_dataset: Dataset | None = None,
) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from config.

    Converts our TrainingConfig dataclass into HuggingFace's
    TrainingArguments format. If eval_dataset is provided, enables
    evaluation during training.

    Args:
        config: Training configuration.
        run_name: Unique name for this training run (used in W&B).
        eval_dataset: Optional evaluation dataset. If provided, enables
            evaluation and early stopping.

    Returns:
        Configured TrainingArguments instance.

    Example:
        >>> args = create_training_args(config, "run-20260107")
        >>> trainer = Trainer(..., args=args)
    """
    # Base arguments
    args_dict = {
        "output_dir": config.output_dir,
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "logging_steps": config.log_steps,
        "save_steps": config.save_steps,
        "save_total_limit": 3,
        "bf16": True,
        "dataloader_pin_memory": True,
        "dataloader_num_workers": 2,
        "report_to": "wandb",
        "run_name": run_name,
        "lr_scheduler_type": config.lr_scheduler_type,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "gradient_checkpointing": True,
        "optim": config.optimizer,
    }

    # Add evaluation arguments if eval dataset is provided
    if eval_dataset is not None:
        args_dict.update(
            {
                "eval_strategy": "steps",
                "eval_steps": config.save_steps,  # Eval at same frequency as checkpoints
                "per_device_eval_batch_size": config.batch_size,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )

    return TrainingArguments(**args_dict)


def create_trainer(
    model: PeftModel,
    dataset: Dataset | DatasetDict,
    config: TrainingConfig,
    run_name: str,
    callbacks: list[TrainerCallback] | None = None,
    early_stopping_patience: int = 3,
) -> Trainer:
    """Create a configured Trainer instance.

    Sets up the HuggingFace Trainer with the model, dataset, training
    arguments, and callbacks. Supports both single datasets and
    train/validation splits with early stopping.

    Args:
        model: Model with LoRA adapters applied.
        dataset: Either a single Dataset (train only) or a DatasetDict
            with 'train' and 'validation' keys.
        config: Training configuration.
        run_name: Unique name for this training run.
        callbacks: Optional list of additional callbacks. If None,
            uses default ProgressCallback.
        early_stopping_patience: Number of evaluation steps with no
            improvement before stopping. Only used if validation set
            is provided.

    Returns:
        Configured Trainer ready for training.

    Example:
        >>> trainer = create_trainer(model, dataset, config, "run-001")
        >>> trainer.train()
    """
    # Handle DatasetDict (train/val split) vs single Dataset
    if hasattr(dataset, "keys") and "train" in dataset:
        # DatasetDict with train and validation
        train_dataset = dataset["train"]
        eval_dataset = dataset.get("validation")
        logger.info(
            f"Using train/val split: {len(train_dataset):,} train, "
            f"{len(eval_dataset):,} validation"
        )
    else:
        # Single dataset, no validation
        train_dataset = dataset
        eval_dataset = None
        logger.info(f"Using single dataset: {len(train_dataset):,} examples (no validation)")

    training_args = create_training_args(config, run_name, eval_dataset)

    # Set up callbacks
    if callbacks is None:
        callbacks = [ProgressCallback(warmup_steps=config.warmup_steps)]

    # Add early stopping if we have validation data
    if eval_dataset is not None:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
        )
        callbacks.append(early_stopping)
        logger.info(f"Early stopping enabled with patience={early_stopping_patience}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    # Log training setup
    total_steps = (len(train_dataset) // config.effective_batch_size) * config.epochs
    logger.info(f"Total training steps: {total_steps:,}")
    logger.info(f"Save checkpoint every: {config.save_steps} steps")
    logger.info(f"Effective batch size: {config.effective_batch_size}")

    return trainer
