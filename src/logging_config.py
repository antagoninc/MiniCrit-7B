"""Structured logging configuration for MiniCrit.

This module provides centralized logging configuration with support for:
- Structured JSON logging for production
- Pretty console logging for development
- Log rotation and file output
- Request ID tracking

Example:
    >>> from src.logging_config import setup_logging, get_logger
    >>> setup_logging(level="DEBUG", json_format=False)
    >>> logger = get_logger(__name__)
    >>> logger.info("Training started", extra={"epoch": 1, "lr": 2e-4})

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

# Default configuration
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_DIR = "logs"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5

# Global state
_logging_initialized = False


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Outputs log records as JSON objects for easy parsing by log
    aggregation systems like ELK, Splunk, or CloudWatch.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON string representation of the log record.
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "taskName",
            }:
                try:
                    json.dumps(value)  # Check if serializable
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development.

    Adds ANSI color codes to log levels for better readability.
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str | None = None, datefmt: str | None = None):
        """Initialize the colored formatter.

        Args:
            fmt: Log format string.
            datefmt: Date format string.
        """
        super().__init__(fmt or DEFAULT_LOG_FORMAT, datefmt or DEFAULT_DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors.

        Args:
            record: The log record to format.

        Returns:
            Colored string representation of the log record.
        """
        # Color the level name
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"

        return super().format(record)


class MetricsFilter(logging.Filter):
    """Filter that adds metrics context to log records."""

    def __init__(self, name: str = ""):
        """Initialize the metrics filter.

        Args:
            name: Filter name.
        """
        super().__init__(name)
        self.request_count = 0
        self.start_time = datetime.now()

    def filter(self, record: logging.LogRecord) -> bool:
        """Add metrics to the log record.

        Args:
            record: The log record to filter.

        Returns:
            Always True (doesn't filter out records).
        """
        record.uptime = (datetime.now() - self.start_time).total_seconds()
        return True


def setup_logging(
    level: str | int = DEFAULT_LOG_LEVEL,
    json_format: bool = False,
    log_file: str | Path | None = None,
    log_dir: str | Path = DEFAULT_LOG_DIR,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    capture_warnings: bool = True,
) -> None:
    """Configure logging for the application.

    Sets up console and optional file handlers with appropriate formatters.
    Call this once at application startup.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, use JSON structured logging.
        log_file: Optional log file name. If provided, logs to file.
        log_dir: Directory for log files.
        max_bytes: Maximum size of log file before rotation.
        backup_count: Number of backup files to keep.
        capture_warnings: If True, capture Python warnings to logging.

    Example:
        >>> setup_logging(level="DEBUG", json_format=False)
        >>> setup_logging(level="INFO", json_format=True, log_file="app.log")
    """
    global _logging_initialized

    if _logging_initialized:
        return

    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if json_format:
        console_handler.setFormatter(StructuredFormatter())
    else:
        # Use colored output if terminal supports it
        if sys.stdout.isatty():
            console_handler.setFormatter(ColoredFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
            )

    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_dir) / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(level)

        # Always use structured format for file logs
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    # Capture Python warnings
    if capture_warnings:
        logging.captureWarnings(True)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)

    _logging_initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Logger name, typically __name__.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding temporary context to logs.

    Example:
        >>> with LogContext(request_id="abc123", user_id=42):
        ...     logger.info("Processing request")
    """

    def __init__(self, **kwargs: Any):
        """Initialize log context.

        Args:
            **kwargs: Key-value pairs to add to log records.
        """
        self.context = kwargs
        self.old_factory = None

    def __enter__(self) -> "LogContext":
        """Enter the context, adding fields to log records."""
        self.old_factory = logging.getLogRecordFactory()

        context = self.context

        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context, restoring original record factory."""
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)


class TrainingLogger:
    """Specialized logger for training metrics.

    Provides methods for logging training progress, metrics, and events
    in a structured format suitable for analysis.

    Example:
        >>> train_logger = TrainingLogger("my_experiment")
        >>> train_logger.log_step(step=100, loss=0.5, lr=1e-4)
        >>> train_logger.log_epoch(epoch=1, train_loss=0.3, val_loss=0.4)
    """

    def __init__(self, experiment_name: str):
        """Initialize the training logger.

        Args:
            experiment_name: Name of the experiment/run.
        """
        self.experiment_name = experiment_name
        self.logger = get_logger(f"training.{experiment_name}")
        self.start_time = datetime.now()
        self.step_count = 0

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float | None = None,
        **metrics: float,
    ) -> None:
        """Log a training step.

        Args:
            step: Current training step.
            loss: Training loss.
            lr: Current learning rate.
            grad_norm: Optional gradient norm.
            **metrics: Additional metrics to log.
        """
        self.step_count = step
        elapsed = (datetime.now() - self.start_time).total_seconds()

        extra = {
            "event": "training_step",
            "experiment": self.experiment_name,
            "step": step,
            "loss": loss,
            "learning_rate": lr,
            "elapsed_seconds": elapsed,
            **metrics,
        }

        if grad_norm is not None:
            extra["grad_norm"] = grad_norm

        self.logger.info(
            f"Step {step} | Loss: {loss:.4f} | LR: {lr:.2e}",
            extra=extra,
        )

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
        **metrics: float,
    ) -> None:
        """Log an epoch completion.

        Args:
            epoch: Current epoch number.
            train_loss: Training loss for the epoch.
            val_loss: Optional validation loss.
            **metrics: Additional metrics to log.
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()

        extra = {
            "event": "epoch_complete",
            "experiment": self.experiment_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "elapsed_seconds": elapsed,
            **metrics,
        }

        if val_loss is not None:
            extra["val_loss"] = val_loss

        msg = f"Epoch {epoch} | Train Loss: {train_loss:.4f}"
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.4f}"

        self.logger.info(msg, extra=extra)

    def log_checkpoint(self, path: str, step: int, metrics: dict[str, float]) -> None:
        """Log a checkpoint save.

        Args:
            path: Path where checkpoint was saved.
            step: Training step at checkpoint.
            metrics: Metrics at checkpoint time.
        """
        self.logger.info(
            f"Checkpoint saved: {path}",
            extra={
                "event": "checkpoint_saved",
                "experiment": self.experiment_name,
                "checkpoint_path": path,
                "step": step,
                **metrics,
            },
        )

    def log_evaluation(self, metrics: dict[str, float], dataset: str = "test") -> None:
        """Log evaluation results.

        Args:
            metrics: Evaluation metrics.
            dataset: Name of the evaluation dataset.
        """
        self.logger.info(
            f"Evaluation on {dataset}: {metrics}",
            extra={
                "event": "evaluation",
                "experiment": self.experiment_name,
                "dataset": dataset,
                **metrics,
            },
        )


# Environment-based configuration
def setup_from_env() -> None:
    """Configure logging from environment variables.

    Environment variables:
        MINICRIT_LOG_LEVEL: Log level (default: INFO)
        MINICRIT_LOG_JSON: Use JSON format (default: false)
        MINICRIT_LOG_FILE: Log file name (default: None)
        MINICRIT_LOG_DIR: Log directory (default: logs)
    """
    level = os.environ.get("MINICRIT_LOG_LEVEL", DEFAULT_LOG_LEVEL)
    json_format = os.environ.get("MINICRIT_LOG_JSON", "false").lower() == "true"
    log_file = os.environ.get("MINICRIT_LOG_FILE")
    log_dir = os.environ.get("MINICRIT_LOG_DIR", DEFAULT_LOG_DIR)

    setup_logging(
        level=level,
        json_format=json_format,
        log_file=log_file,
        log_dir=log_dir,
    )
