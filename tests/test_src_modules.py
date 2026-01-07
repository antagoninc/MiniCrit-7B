"""Tests for src/ module components.

Tests the modular components without requiring heavy ML dependencies.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.config import TrainingConfig, LoRAConfig, load_config
from src.data import find_columns, validate_dataset
from src.training import find_latest_checkpoint


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.model_name == "Qwen/Qwen2-7B-Instruct"
        assert config.learning_rate == 2e-4
        assert config.batch_size == 4
        assert config.gradient_accumulation_steps == 8
        assert config.max_length == 512

    def test_effective_batch_size(self) -> None:
        """Test effective batch size calculation."""
        config = TrainingConfig(batch_size=4, gradient_accumulation_steps=8)
        assert config.effective_batch_size == 32

        config2 = TrainingConfig(batch_size=2, gradient_accumulation_steps=16)
        assert config2.effective_batch_size == 32


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default LoRA configuration."""
        lora = LoRAConfig()
        assert lora.r == 16
        assert lora.alpha == 32
        assert lora.dropout == 0.05
        assert lora.bias == "none"
        assert "q_proj" in lora.target_modules
        assert "v_proj" in lora.target_modules

    def test_custom_values(self) -> None:
        """Test custom LoRA configuration."""
        lora = LoRAConfig(r=32, alpha=64, dropout=0.1)
        assert lora.r == 32
        assert lora.alpha == 64
        assert lora.dropout == 0.1


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_defaults(self) -> None:
        """Test loading default configuration."""
        config = load_config()
        assert config.model_name == "Qwen/Qwen2-7B-Instruct"
        assert isinstance(config.lora, LoRAConfig)

    def test_env_override(self) -> None:
        """Test environment variable overrides."""
        with patch.dict(os.environ, {
            "MINICRIT_MODEL": "test-model",
            "MINICRIT_LR": "1e-5",
            "MINICRIT_BATCH_SIZE": "2",
        }):
            config = load_config()
            assert config.model_name == "test-model"
            assert config.learning_rate == 1e-5
            assert config.batch_size == 2

    def test_load_nonexistent_yaml(self) -> None:
        """Test loading with nonexistent YAML file uses defaults."""
        config = load_config("/nonexistent/path.yaml")
        assert config.model_name == "Qwen/Qwen2-7B-Instruct"


class TestFindColumns:
    """Tests for find_columns function."""

    def test_standard_columns(self) -> None:
        """Test finding standard column names."""
        df = pd.DataFrame({"text": ["a"], "rebuttal": ["b"]})
        text_col, rebuttal_col = find_columns(df)
        assert text_col == "text"
        assert rebuttal_col == "rebuttal"

    def test_alternative_columns(self) -> None:
        """Test finding alternative column names."""
        df = pd.DataFrame({"rationale": ["a"], "critique": ["b"]})
        text_col, rebuttal_col = find_columns(df)
        assert text_col == "rationale"
        assert rebuttal_col == "critique"

    def test_input_output_columns(self) -> None:
        """Test finding input/output columns."""
        df = pd.DataFrame({"input": ["a"], "output": ["b"]})
        text_col, rebuttal_col = find_columns(df)
        assert text_col == "input"
        assert rebuttal_col == "output"

    def test_missing_text_column_raises(self) -> None:
        """Test error when text column is missing."""
        df = pd.DataFrame({"unknown": ["a"], "rebuttal": ["b"]})
        try:
            find_columns(df)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "No text column found" in str(e)

    def test_missing_rebuttal_column_raises(self) -> None:
        """Test error when rebuttal column is missing."""
        df = pd.DataFrame({"text": ["a"], "unknown": ["b"]})
        try:
            find_columns(df)
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "No rebuttal column found" in str(e)


class TestFindLatestCheckpoint:
    """Tests for find_latest_checkpoint function."""

    def test_no_directory(self) -> None:
        """Test with nonexistent directory."""
        result = find_latest_checkpoint("/nonexistent")
        assert result is None

    def test_empty_directory(self) -> None:
        """Test with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_latest_checkpoint(tmpdir)
            assert result is None

    def test_single_checkpoint(self) -> None:
        """Test with single checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "checkpoint-1000").mkdir()
            result = find_latest_checkpoint(tmpdir)
            assert result == str(Path(tmpdir, "checkpoint-1000"))

    def test_multiple_checkpoints_returns_latest(self) -> None:
        """Test returns highest numbered checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for step in [500, 1000, 2000, 1500]:
                Path(tmpdir, f"checkpoint-{step}").mkdir()
            result = find_latest_checkpoint(tmpdir)
            assert result == str(Path(tmpdir, "checkpoint-2000"))

    def test_ignores_non_checkpoint_dirs(self) -> None:
        """Test ignores directories not named checkpoint-*."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "checkpoint-1000").mkdir()
            Path(tmpdir, "other-dir").mkdir()
            Path(tmpdir, "logs").mkdir()
            result = find_latest_checkpoint(tmpdir)
            assert result == str(Path(tmpdir, "checkpoint-1000"))


class TestValidateDataset:
    """Tests for validate_dataset function."""

    def test_valid_dataset_passes(self) -> None:
        """Test validation passes for valid dataset."""
        from datasets import Dataset

        # Create dataset with trainable tokens
        data = {
            "input_ids": [[1, 2, 3, 4, 5] for _ in range(100)],
            "attention_mask": [[1, 1, 1, 1, 1] for _ in range(100)],
            "labels": [[-100, -100, 3, 4, 5] for _ in range(100)],
        }
        dataset = Dataset.from_dict(data)
        assert validate_dataset(dataset) is True

    def test_all_masked_fails(self) -> None:
        """Test validation fails when all tokens are masked."""
        from datasets import Dataset

        data = {
            "input_ids": [[1, 2, 3, 4, 5] for _ in range(100)],
            "attention_mask": [[1, 1, 1, 1, 1] for _ in range(100)],
            "labels": [[-100, -100, -100, -100, -100] for _ in range(100)],
        }
        dataset = Dataset.from_dict(data)
        assert validate_dataset(dataset) is False

    def test_too_few_valid_fails(self) -> None:
        """Test validation fails when too few valid examples."""
        from datasets import Dataset

        # 20% valid (below 90% threshold)
        valid_labels = [[-100, -100, 3, 4, 5] for _ in range(20)]
        invalid_labels = [[-100, -100, -100, -100, -100] for _ in range(80)]

        data = {
            "input_ids": [[1, 2, 3, 4, 5] for _ in range(100)],
            "attention_mask": [[1, 1, 1, 1, 1] for _ in range(100)],
            "labels": valid_labels + invalid_labels,
        }
        dataset = Dataset.from_dict(data)
        assert validate_dataset(dataset) is False


class TestDataDeduplication:
    """Tests for data deduplication functions."""

    def test_compute_text_hash(self) -> None:
        """Test hash computation."""
        from src.data import compute_text_hash
        hash1 = compute_text_hash("hello world")
        hash2 = compute_text_hash("hello world")
        hash3 = compute_text_hash("different text")
        assert hash1 == hash2
        assert hash1 != hash3

    def test_hash_normalization(self) -> None:
        """Test that hashes are normalized."""
        from src.data import compute_text_hash
        # Multiple spaces should be normalized
        hash1 = compute_text_hash("hello  world")
        hash2 = compute_text_hash("hello world")
        assert hash1 == hash2

    def test_hash_case_insensitive(self) -> None:
        """Test case insensitivity."""
        from src.data import compute_text_hash
        hash1 = compute_text_hash("Hello World")
        hash2 = compute_text_hash("hello world")
        assert hash1 == hash2


class TestRebuttalLengthValidation:
    """Tests for rebuttal length validation."""

    def test_min_word_constant(self) -> None:
        """Test MIN_REBUTTAL_WORDS constant."""
        from src.data import MIN_REBUTTAL_WORDS
        assert MIN_REBUTTAL_WORDS == 50

    def test_validate_length_accepts_long(self) -> None:
        """Test that long rebuttals pass."""
        from src.data import validate_rebuttal_length
        df = pd.DataFrame({
            "text": ["input"],
            "rebuttal": ["word " * 60]  # 60 words
        })
        result = validate_rebuttal_length(df, "rebuttal", min_words=50)
        assert len(result) == 1

    def test_validate_length_rejects_short(self) -> None:
        """Test that short rebuttals are filtered."""
        from src.data import validate_rebuttal_length
        df = pd.DataFrame({
            "text": ["input1", "input2"],
            "rebuttal": ["word " * 10, "word " * 60]  # 10 and 60 words
        })
        result = validate_rebuttal_length(df, "rebuttal", min_words=50)
        assert len(result) == 1


class TestTrainValSplit:
    """Tests for train/validation split."""

    def test_split_creates_two_sets(self) -> None:
        """Test that split creates train and val sets."""
        from src.data import create_train_val_split
        df = pd.DataFrame({
            "text": [f"text{i}" for i in range(1000)],
            "rebuttal": [f"rebuttal{i}" for i in range(1000)]
        })
        train_df, val_df = create_train_val_split(df, val_ratio=0.1)
        assert len(train_df) == 900
        assert len(val_df) == 100

    def test_split_minimum_validation(self) -> None:
        """Test minimum validation set size (hardcoded to 100)."""
        from src.data import create_train_val_split
        df = pd.DataFrame({
            "text": [f"text{i}" for i in range(200)],
            "rebuttal": [f"rebuttal{i}" for i in range(200)]
        })
        # With 5% of 200 = 10, but minimum is 100 (hardcoded)
        train_df, val_df = create_train_val_split(df, val_ratio=0.05)
        assert len(val_df) == 100


def run_all_tests() -> bool:
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestTrainingConfig,
        TestLoRAConfig,
        TestLoadConfig,
        TestFindColumns,
        TestFindLatestCheckpoint,
        TestValidateDataset,
        TestDataDeduplication,
        TestRebuttalLengthValidation,
        TestTrainValSplit,
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
                except Exception as e:
                    print(f"FAIL: {test_class.__name__}.{method_name}")
                    traceback.print_exc()
                    failed += 1

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
