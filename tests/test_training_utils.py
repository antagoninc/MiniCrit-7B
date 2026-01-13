"""Tests for training_utils.py - lightweight tests without ML dependencies."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Ensure parent directory is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from training_utils import (
    find_columns,
    find_latest_checkpoint,
    get_config,
    parse_args,
    validate_labels,
)


def test_parse_args_defaults() -> None:
    """Test default argument values."""
    with patch("sys.argv", ["train_minicrit_7b.py"]):
        args = parse_args()
        assert args.sample is None
        assert args.validate_only is False
        assert args.resume is None
        assert args.resume_latest is False


def test_parse_args_sample() -> None:
    """Test --sample argument."""
    with patch("sys.argv", ["train_minicrit_7b.py", "--sample", "100"]):
        args = parse_args()
        assert args.sample == 100


def test_parse_args_validate_only() -> None:
    """Test --validate-only argument."""
    with patch("sys.argv", ["train_minicrit_7b.py", "--validate-only"]):
        args = parse_args()
        assert args.validate_only is True


def test_parse_args_resume() -> None:
    """Test --resume argument."""
    with patch("sys.argv", ["train_minicrit_7b.py", "--resume", "checkpoint-1000"]):
        args = parse_args()
        assert args.resume == "checkpoint-1000"


def test_parse_args_resume_latest() -> None:
    """Test --resume-latest argument."""
    with patch("sys.argv", ["train_minicrit_7b.py", "--resume-latest"]):
        args = parse_args()
        assert args.resume_latest is True


def test_get_config_defaults() -> None:
    """Test default configuration values."""
    # Clear any env vars that might interfere
    env_vars_to_clear = [
        "MINICRIT_MODEL",
        "MINICRIT_LR",
        "MINICRIT_BATCH_SIZE",
        "MINICRIT_DATA_FILE",
        "MINICRIT_CACHE_DIR",
        "MINICRIT_OUTPUT_DIR",
    ]
    original_env = {k: os.environ.pop(k, None) for k in env_vars_to_clear}

    try:
        config = get_config()
        assert config["model"] == "Qwen/Qwen2-7B-Instruct"
        assert config["learning_rate"] == 2e-4
        assert config["batch_size"] == 4
        assert config["max_length"] == 512
        assert config["lora_r"] == 16
    finally:
        # Restore env vars
        for k, v in original_env.items():
            if v is not None:
                os.environ[k] = v


def test_get_config_env_override() -> None:
    """Test environment variable overrides."""
    with patch.dict(os.environ, {"MINICRIT_LR": "1e-5", "MINICRIT_BATCH_SIZE": "8"}):
        config = get_config()
        assert config["learning_rate"] == 1e-5
        assert config["batch_size"] == 8


def test_get_config_model_override() -> None:
    """Test model name override via environment."""
    with patch.dict(os.environ, {"MINICRIT_MODEL": "meta-llama/Llama-2-7b"}):
        config = get_config()
        assert config["model"] == "meta-llama/Llama-2-7b"


def test_find_columns_text_rebuttal() -> None:
    """Test finding 'text' and 'rebuttal' columns."""
    df = pd.DataFrame({"text": ["a"], "rebuttal": ["b"]})
    text_col, rebuttal_col = find_columns(df)
    assert text_col == "text"
    assert rebuttal_col == "rebuttal"


def test_find_columns_rationale_critique() -> None:
    """Test finding 'rationale' and 'critique' columns."""
    df = pd.DataFrame({"rationale": ["a"], "critique": ["b"]})
    text_col, rebuttal_col = find_columns(df)
    assert text_col == "rationale"
    assert rebuttal_col == "critique"


def test_find_columns_input_output() -> None:
    """Test finding 'input' and 'output' columns."""
    df = pd.DataFrame({"input": ["a"], "output": ["b"]})
    text_col, rebuttal_col = find_columns(df)
    assert text_col == "input"
    assert rebuttal_col == "output"


def test_find_columns_prompt_response() -> None:
    """Test finding 'prompt' and 'response' columns."""
    df = pd.DataFrame({"prompt": ["a"], "response": ["b"]})
    text_col, rebuttal_col = find_columns(df)
    assert text_col == "prompt"
    assert rebuttal_col == "response"


def test_find_columns_missing_text() -> None:
    """Test error when text column is missing."""
    df = pd.DataFrame({"foo": ["a"], "rebuttal": ["b"]})
    try:
        find_columns(df)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No text column found" in str(e)


def test_find_columns_missing_rebuttal() -> None:
    """Test error when rebuttal column is missing."""
    df = pd.DataFrame({"text": ["a"], "foo": ["b"]})
    try:
        find_columns(df)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No rebuttal column found" in str(e)


def test_find_latest_checkpoint_no_dir() -> None:
    """Test when output directory doesn't exist."""
    result = find_latest_checkpoint("/nonexistent/path")
    assert result is None


def test_find_latest_checkpoint_empty_dir() -> None:
    """Test when output directory is empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_latest_checkpoint(tmpdir)
        assert result is None


def test_find_latest_checkpoint_no_checkpoints() -> None:
    """Test when output directory has no checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "other_file.txt").touch()
        result = find_latest_checkpoint(tmpdir)
        assert result is None


def test_find_latest_checkpoint_single() -> None:
    """Test finding a single checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Path(tmpdir, "checkpoint-1000")
        checkpoint.mkdir()
        result = find_latest_checkpoint(tmpdir)
        assert result == str(checkpoint)


def test_find_latest_checkpoint_multiple() -> None:
    """Test finding the latest among multiple checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for step in [1000, 2000, 3000, 1500]:
            Path(tmpdir, f"checkpoint-{step}").mkdir()
        result = find_latest_checkpoint(tmpdir)
        assert result == str(Path(tmpdir, "checkpoint-3000"))


def test_validate_labels_mixed() -> None:
    """Test counting trainable vs masked tokens."""
    labels = [-100, -100, -100, 1, 2, 3, 4, 5, -100]
    trainable, masked = validate_labels(labels)
    assert trainable == 5
    assert masked == 4


def test_validate_labels_all_masked() -> None:
    """Test when all labels are masked."""
    labels = [-100, -100, -100, -100, -100]
    trainable, masked = validate_labels(labels)
    assert trainable == 0
    assert masked == 5


def test_validate_labels_none_masked() -> None:
    """Test when no labels are masked."""
    labels = [1, 2, 3, 4, 5]
    trainable, masked = validate_labels(labels)
    assert trainable == 5
    assert masked == 0


def run_all_tests() -> None:
    """Run all tests and report results."""
    import traceback

    tests = [
        test_parse_args_defaults,
        test_parse_args_sample,
        test_parse_args_validate_only,
        test_parse_args_resume,
        test_parse_args_resume_latest,
        test_get_config_defaults,
        test_get_config_env_override,
        test_get_config_model_override,
        test_find_columns_text_rebuttal,
        test_find_columns_rationale_critique,
        test_find_columns_input_output,
        test_find_columns_prompt_response,
        test_find_columns_missing_text,
        test_find_columns_missing_rebuttal,
        test_find_latest_checkpoint_no_dir,
        test_find_latest_checkpoint_empty_dir,
        test_find_latest_checkpoint_no_checkpoints,
        test_find_latest_checkpoint_single,
        test_find_latest_checkpoint_multiple,
        test_validate_labels_mixed,
        test_validate_labels_all_masked,
        test_validate_labels_none_masked,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL: {test.__name__}")
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
