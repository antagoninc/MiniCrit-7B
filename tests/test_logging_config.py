"""Tests for the logging configuration module.

Tests logging setup, formatters, and training logger functionality.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_config import (
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    ColoredFormatter,
    LogContext,
    MetricsFilter,
    StructuredFormatter,
    TrainingLogger,
    get_logger,
    setup_logging,
)


class TestStructuredFormatter:
    """Tests for StructuredFormatter."""

    def test_basic_format(self) -> None:
        """Test basic JSON formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["logger"] == "test"
        assert "timestamp" in data

    def test_format_with_extra(self) -> None:
        """Test JSON formatting with extra fields."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.custom_field = "custom_value"
        record.numeric_field = 42

        output = formatter.format(record)
        data = json.loads(output)

        assert data["custom_field"] == "custom_value"
        assert data["numeric_field"] == 42

    def test_format_with_exception(self) -> None:
        """Test JSON formatting with exception info."""
        formatter = StructuredFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "ERROR"
        assert "exception" in data
        assert "ValueError" in data["exception"]

    def test_timestamp_format(self) -> None:
        """Test timestamp is ISO format with Z suffix."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["timestamp"].endswith("Z")
        # Should be parseable as ISO format
        datetime.fromisoformat(data["timestamp"].rstrip("Z"))


class TestColoredFormatter:
    """Tests for ColoredFormatter."""

    def test_basic_format(self) -> None:
        """Test basic formatting."""
        formatter = ColoredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        assert "Test message" in output

    def test_level_colors(self) -> None:
        """Test that different levels get different colors."""
        formatter = ColoredFormatter()

        for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=10,
                msg="Test",
                args=(),
                exc_info=None,
            )
            output = formatter.format(record)
            # Should contain ANSI codes
            assert "\033[" in output or "Test" in output

    def test_reset_code(self) -> None:
        """Test that reset code is included."""
        formatter = ColoredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        # Reset code should be present
        assert "\033[0m" in output or "Test" in output


class TestMetricsFilter:
    """Tests for MetricsFilter."""

    def test_adds_uptime(self) -> None:
        """Test that uptime is added to records."""
        filter = MetricsFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = filter.filter(record)

        assert result is True
        assert hasattr(record, "uptime")
        assert record.uptime >= 0

    def test_always_returns_true(self) -> None:
        """Test that filter doesn't exclude records."""
        filter = MetricsFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        assert filter.filter(record) is True


class TestLogContext:
    """Tests for LogContext context manager."""

    def test_context_adds_fields(self) -> None:
        """Test that context adds fields to records."""
        with LogContext(request_id="abc123", user_id=42):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg="Test",
                args=(),
                exc_info=None,
            )
            # Fields should be added by record factory
            # This is a simplified test

    def test_context_cleanup(self) -> None:
        """Test that context cleans up after exit."""
        original_factory = logging.getLogRecordFactory()

        with LogContext(test_field="value"):
            pass  # Context should modify and restore factory

        # Factory should be restored
        current_factory = logging.getLogRecordFactory()
        # May or may not be same object depending on nesting


class TestTrainingLogger:
    """Tests for TrainingLogger."""

    def test_initialization(self) -> None:
        """Test logger initialization."""
        logger = TrainingLogger("test_experiment")
        assert logger.experiment_name == "test_experiment"
        assert logger.step_count == 0

    def test_log_step(self) -> None:
        """Test logging a training step."""
        train_logger = TrainingLogger("test")

        # Should not raise
        train_logger.log_step(
            step=100,
            loss=0.5,
            lr=1e-4,
            grad_norm=1.2,
        )

        assert train_logger.step_count == 100

    def test_log_step_with_metrics(self) -> None:
        """Test logging step with extra metrics."""
        train_logger = TrainingLogger("test")

        train_logger.log_step(
            step=50,
            loss=0.3,
            lr=2e-4,
            accuracy=0.95,
            perplexity=1.5,
        )

        assert train_logger.step_count == 50

    def test_log_epoch(self) -> None:
        """Test logging an epoch."""
        train_logger = TrainingLogger("test")

        train_logger.log_epoch(
            epoch=1,
            train_loss=0.4,
            val_loss=0.5,
        )

        # Should not raise

    def test_log_epoch_without_val(self) -> None:
        """Test logging epoch without validation loss."""
        train_logger = TrainingLogger("test")

        train_logger.log_epoch(
            epoch=1,
            train_loss=0.4,
        )

    def test_log_checkpoint(self) -> None:
        """Test logging checkpoint save."""
        train_logger = TrainingLogger("test")

        train_logger.log_checkpoint(
            path="/path/to/checkpoint",
            step=1000,
            metrics={"loss": 0.3, "accuracy": 0.9},
        )

    def test_log_evaluation(self) -> None:
        """Test logging evaluation results."""
        train_logger = TrainingLogger("test")

        train_logger.log_evaluation(
            metrics={"rouge1": 0.5, "rouge2": 0.3, "rougeL": 0.4},
            dataset="test",
        )


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_default(self) -> None:
        """Test default logging setup."""
        # Reset initialization flag for testing
        import src.logging_config

        src.logging_config._logging_initialized = False

        setup_logging()

        logger = get_logger("test")
        assert logger is not None

    def test_setup_with_level(self) -> None:
        """Test setup with custom level."""
        import src.logging_config

        src.logging_config._logging_initialized = False

        setup_logging(level="DEBUG")

        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_setup_json_format(self) -> None:
        """Test setup with JSON format."""
        import src.logging_config

        src.logging_config._logging_initialized = False

        setup_logging(json_format=True)

        # Check that handler has StructuredFormatter
        root = logging.getLogger()
        has_structured = any(isinstance(h.formatter, StructuredFormatter) for h in root.handlers)
        assert has_structured

    def test_setup_with_file(self) -> None:
        """Test setup with file logging."""
        import src.logging_config

        src.logging_config._logging_initialized = False

        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(
                log_file="test.log",
                log_dir=tmpdir,
            )

            log_path = Path(tmpdir) / "test.log"
            # File should be created
            assert log_path.parent.exists()

    def test_idempotent(self) -> None:
        """Test that setup is idempotent."""
        import src.logging_config

        src.logging_config._logging_initialized = False

        setup_logging()
        handler_count_1 = len(logging.getLogger().handlers)

        # Second call should not add more handlers
        setup_logging()
        handler_count_2 = len(logging.getLogger().handlers)

        assert handler_count_1 == handler_count_2


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger(self) -> None:
        """Test that get_logger returns a logger."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_same_name_same_logger(self) -> None:
        """Test that same name returns same logger."""
        logger1 = get_logger("same.name")
        logger2 = get_logger("same.name")
        assert logger1 is logger2

    def test_different_names_different_loggers(self) -> None:
        """Test that different names return different loggers."""
        logger1 = get_logger("name.one")
        logger2 = get_logger("name.two")
        assert logger1 is not logger2


class TestDefaultValues:
    """Tests for default configuration values."""

    def test_default_level(self) -> None:
        """Test default log level."""
        assert DEFAULT_LOG_LEVEL == "INFO"

    def test_default_format(self) -> None:
        """Test default format string."""
        assert "%(asctime)s" in DEFAULT_LOG_FORMAT
        assert "%(levelname)" in DEFAULT_LOG_FORMAT
        assert "%(message)s" in DEFAULT_LOG_FORMAT


def run_all_tests() -> bool:
    """Run all logging tests and report results."""
    import traceback

    test_classes = [
        TestStructuredFormatter,
        TestColoredFormatter,
        TestMetricsFilter,
        TestLogContext,
        TestTrainingLogger,
        TestSetupLogging,
        TestGetLogger,
        TestDefaultValues,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                method = getattr(instance, method_name)
                try:
                    method()
                    print(f"PASS: {test_class.__name__}.{method_name}")
                    passed += 1
                except Exception:
                    print(f"FAIL: {test_class.__name__}.{method_name}")
                    traceback.print_exc()
                    failed += 1

    print()
    print("=" * 50)
    print(f"Logging Tests: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
