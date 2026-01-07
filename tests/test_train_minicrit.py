"""Tests for train_minicrit_7b.py"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_minicrit_7b import (
    find_columns,
    find_latest_checkpoint,
    get_config,
    parse_args,
    validate_dataset,
)


class TestParseArgs:
    """Tests for parse_args function."""

    def test_default_args(self) -> None:
        """Test default argument values."""
        with patch("sys.argv", ["train_minicrit_7b.py"]):
            args = parse_args()
            assert args.sample is None
            assert args.validate_only is False
            assert args.resume is None
            assert args.resume_latest is False

    def test_sample_arg(self) -> None:
        """Test --sample argument."""
        with patch("sys.argv", ["train_minicrit_7b.py", "--sample", "100"]):
            args = parse_args()
            assert args.sample == 100

    def test_validate_only_arg(self) -> None:
        """Test --validate-only argument."""
        with patch("sys.argv", ["train_minicrit_7b.py", "--validate-only"]):
            args = parse_args()
            assert args.validate_only is True

    def test_resume_arg(self) -> None:
        """Test --resume argument."""
        with patch("sys.argv", ["train_minicrit_7b.py", "--resume", "checkpoint-1000"]):
            args = parse_args()
            assert args.resume == "checkpoint-1000"

    def test_resume_latest_arg(self) -> None:
        """Test --resume-latest argument."""
        with patch("sys.argv", ["train_minicrit_7b.py", "--resume-latest"]):
            args = parse_args()
            assert args.resume_latest is True


class TestGetConfig:
    """Tests for get_config function."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = get_config()
        assert config["model"] == "Qwen/Qwen2-7B-Instruct"
        assert config["learning_rate"] == 2e-4
        assert config["batch_size"] == 4
        assert config["max_length"] == 512
        assert config["lora_r"] == 16

    def test_env_override(self) -> None:
        """Test environment variable overrides."""
        with patch.dict(os.environ, {"MINICRIT_LR": "1e-5", "MINICRIT_BATCH_SIZE": "8"}):
            config = get_config()
            assert config["learning_rate"] == 1e-5
            assert config["batch_size"] == 8

    def test_model_env_override(self) -> None:
        """Test model name override via environment."""
        with patch.dict(os.environ, {"MINICRIT_MODEL": "meta-llama/Llama-2-7b"}):
            config = get_config()
            assert config["model"] == "meta-llama/Llama-2-7b"


class TestFindColumns:
    """Tests for find_columns function."""

    def test_find_text_rationale(self) -> None:
        """Test finding 'text' and 'rebuttal' columns."""
        df = pd.DataFrame({"text": ["a"], "rebuttal": ["b"]})
        text_col, rebuttal_col = find_columns(df)
        assert text_col == "text"
        assert rebuttal_col == "rebuttal"

    def test_find_rationale_critique(self) -> None:
        """Test finding 'rationale' and 'critique' columns."""
        df = pd.DataFrame({"rationale": ["a"], "critique": ["b"]})
        text_col, rebuttal_col = find_columns(df)
        assert text_col == "rationale"
        assert rebuttal_col == "critique"

    def test_find_input_output(self) -> None:
        """Test finding 'input' and 'output' columns."""
        df = pd.DataFrame({"input": ["a"], "output": ["b"]})
        text_col, rebuttal_col = find_columns(df)
        assert text_col == "input"
        assert rebuttal_col == "output"

    def test_find_prompt_response(self) -> None:
        """Test finding 'prompt' and 'response' columns."""
        df = pd.DataFrame({"prompt": ["a"], "response": ["b"]})
        text_col, rebuttal_col = find_columns(df)
        assert text_col == "prompt"
        assert rebuttal_col == "response"

    def test_missing_text_column(self) -> None:
        """Test error when text column is missing."""
        df = pd.DataFrame({"foo": ["a"], "rebuttal": ["b"]})
        with pytest.raises(ValueError, match="No text column found"):
            find_columns(df)

    def test_missing_rebuttal_column(self) -> None:
        """Test error when rebuttal column is missing."""
        df = pd.DataFrame({"text": ["a"], "foo": ["b"]})
        with pytest.raises(ValueError, match="No rebuttal column found"):
            find_columns(df)


class TestFindLatestCheckpoint:
    """Tests for find_latest_checkpoint function."""

    def test_no_output_dir(self) -> None:
        """Test when output directory doesn't exist."""
        result = find_latest_checkpoint("/nonexistent/path")
        assert result is None

    def test_empty_output_dir(self) -> None:
        """Test when output directory is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_latest_checkpoint(tmpdir)
            assert result is None

    def test_no_checkpoints(self) -> None:
        """Test when output directory has no checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "other_file.txt").touch()
            result = find_latest_checkpoint(tmpdir)
            assert result is None

    def test_single_checkpoint(self) -> None:
        """Test finding a single checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir, "checkpoint-1000")
            checkpoint.mkdir()
            result = find_latest_checkpoint(tmpdir)
            assert result == str(checkpoint)

    def test_multiple_checkpoints(self) -> None:
        """Test finding the latest among multiple checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for step in [1000, 2000, 3000, 1500]:
                Path(tmpdir, f"checkpoint-{step}").mkdir()
            result = find_latest_checkpoint(tmpdir)
            assert result == str(Path(tmpdir, "checkpoint-3000"))


class TestValidateDataset:
    """Tests for validate_dataset function."""

    def test_valid_dataset(self) -> None:
        """Test validation passes for valid dataset."""
        # Create a mock dataset with proper structure
        from datasets import Dataset

        data = {
            "input_ids": [[1, 2, 3, 4, 5] * 100] * 100,
            "attention_mask": [[1, 1, 1, 1, 1] * 100] * 100,
            "labels": [[-100, -100, 3, 4, 5] * 100] * 100,  # First 2 masked
        }
        dataset = Dataset.from_dict(data)
        assert validate_dataset(dataset) is True

    def test_no_trainable_tokens(self) -> None:
        """Test validation fails when all labels are -100."""
        from datasets import Dataset

        data = {
            "input_ids": [[1, 2, 3, 4, 5]] * 100,
            "attention_mask": [[1, 1, 1, 1, 1]] * 100,
            "labels": [[-100, -100, -100, -100, -100]] * 100,
        }
        dataset = Dataset.from_dict(data)
        assert validate_dataset(dataset) is False

    def test_too_many_invalid_examples(self) -> None:
        """Test validation fails when >10% examples have no trainable tokens."""
        from datasets import Dataset

        # 15 invalid examples (15% invalid)
        invalid = [[-100, -100, -100, -100, -100]] * 15
        valid = [[-100, -100, 3, 4, 5]] * 85

        data = {
            "input_ids": [[1, 2, 3, 4, 5]] * 100,
            "attention_mask": [[1, 1, 1, 1, 1]] * 100,
            "labels": invalid + valid,
        }
        dataset = Dataset.from_dict(data)
        assert validate_dataset(dataset) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
