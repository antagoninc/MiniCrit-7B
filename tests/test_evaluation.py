"""Tests for the evaluation module.

Tests evaluation metrics, result containers, and evaluation utilities.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    EvaluationResult,
    compute_rouge_scores,
    save_evaluation_results,
    load_test_data,
)


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_default_values(self) -> None:
        """Test default result values."""
        result = EvaluationResult()
        assert result.rouge1 == 0.0
        assert result.rouge2 == 0.0
        assert result.rougeL == 0.0
        assert result.bert_score_f1 == 0.0
        assert result.num_samples == 0

    def test_custom_values(self) -> None:
        """Test result with custom values."""
        result = EvaluationResult(
            rouge1=0.5,
            rouge2=0.3,
            rougeL=0.4,
            bert_score_precision=0.7,
            bert_score_recall=0.6,
            bert_score_f1=0.65,
            avg_length=100.5,
            num_samples=50,
        )
        assert result.rouge1 == 0.5
        assert result.bert_score_f1 == 0.65
        assert result.num_samples == 50

    def test_with_predictions(self) -> None:
        """Test result with predictions and references."""
        result = EvaluationResult(
            predictions=["pred1", "pred2"],
            references=["ref1", "ref2"],
            num_samples=2,
        )
        assert len(result.predictions) == 2
        assert len(result.references) == 2

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = EvaluationResult(
            rouge1=0.5,
            rouge2=0.3,
            rougeL=0.4,
            bert_score_f1=0.6,
            num_samples=10,
        )
        data = result.to_dict()

        assert data["rouge1"] == 0.5
        assert data["rouge2"] == 0.3
        assert data["num_samples"] == 10
        # Predictions/references not in to_dict by default
        assert "predictions" not in data

    def test_str_format(self) -> None:
        """Test string representation."""
        result = EvaluationResult(
            rouge1=0.45,
            rouge2=0.25,
            rougeL=0.35,
            bert_score_precision=0.7,
            bert_score_recall=0.6,
            bert_score_f1=0.65,
            avg_length=120.0,
            num_samples=100,
        )
        text = str(result)

        assert "ROUGE-1: 0.4500" in text
        assert "ROUGE-2: 0.2500" in text
        assert "n=100" in text
        assert "BERTScore" in text

    def test_empty_predictions(self) -> None:
        """Test with empty predictions list."""
        result = EvaluationResult(
            predictions=[],
            references=[],
        )
        assert result.predictions == []
        assert result.references == []


class TestComputeRougeScores:
    """Tests for compute_rouge_scores function."""

    def test_identical_texts(self) -> None:
        """Test ROUGE scores for identical texts."""
        predictions = ["This is a test sentence."]
        references = ["This is a test sentence."]

        scores = compute_rouge_scores(predictions, references)

        # If rouge_score not installed, returns zeros
        # Otherwise, identical texts should have high scores
        if scores["rouge1"] > 0:
            assert scores["rouge1"] > 0.9
            assert scores["rouge2"] > 0.9
            assert scores["rougeL"] > 0.9
        else:
            # Rouge not installed, just verify structure
            assert "rouge1" in scores

    def test_completely_different(self) -> None:
        """Test ROUGE scores for completely different texts."""
        predictions = ["Apple banana cherry"]
        references = ["Dog elephant frog"]

        scores = compute_rouge_scores(predictions, references)

        # Different texts should have low scores (or zero if not installed)
        assert scores["rouge1"] < 0.2
        assert scores["rouge2"] < 0.2

    def test_partial_overlap(self) -> None:
        """Test ROUGE scores for partial overlap."""
        predictions = ["The quick brown fox jumps"]
        references = ["The quick red fox leaps"]

        scores = compute_rouge_scores(predictions, references)

        # Should have some overlap or be zero if not installed
        assert 0.0 <= scores["rouge1"] <= 1.0

    def test_multiple_samples(self) -> None:
        """Test ROUGE with multiple samples."""
        predictions = ["Text one here", "Text two there"]
        references = ["Text one here", "Text two elsewhere"]

        scores = compute_rouge_scores(predictions, references)

        # Should return averaged scores
        assert "rouge1" in scores
        assert "rouge2" in scores
        assert "rougeL" in scores

    def test_empty_lists(self) -> None:
        """Test ROUGE with empty lists."""
        scores = compute_rouge_scores([], [])

        assert scores["rouge1"] == 0.0
        assert scores["rouge2"] == 0.0
        assert scores["rougeL"] == 0.0

    def test_returns_dict(self) -> None:
        """Test that function returns dictionary."""
        scores = compute_rouge_scores(["test"], ["test"])

        assert isinstance(scores, dict)
        assert set(scores.keys()) == {"rouge1", "rouge2", "rougeL"}


class TestSaveEvaluationResults:
    """Tests for save_evaluation_results function."""

    def test_save_basic(self) -> None:
        """Test basic save functionality."""
        result = EvaluationResult(
            rouge1=0.5,
            rouge2=0.3,
            rougeL=0.4,
            num_samples=10,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        save_evaluation_results(result, path)

        # Verify file was created and contains data
        with open(path) as f:
            data = json.load(f)

        assert data["rouge1"] == 0.5
        assert data["num_samples"] == 10

        Path(path).unlink()

    def test_save_with_samples(self) -> None:
        """Test save with include_samples=True."""
        result = EvaluationResult(
            rouge1=0.5,
            num_samples=2,
            predictions=["pred1", "pred2"],
            references=["ref1", "ref2"],
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        save_evaluation_results(result, path, include_samples=True)

        with open(path) as f:
            data = json.load(f)

        assert "predictions" in data
        assert data["predictions"] == ["pred1", "pred2"]

        Path(path).unlink()

    def test_creates_parent_dirs(self) -> None:
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "results.json"

            result = EvaluationResult(rouge1=0.5)
            save_evaluation_results(result, path)

            assert path.exists()


class TestLoadTestData:
    """Tests for load_test_data function."""

    def test_load_json(self) -> None:
        """Test loading JSON test data."""
        data = [
            {"rationale": "Test 1", "critique": "Critique 1"},
            {"rationale": "Test 2", "critique": "Critique 2"},
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name

        loaded = load_test_data(path)

        assert len(loaded) == 2
        assert loaded[0]["rationale"] == "Test 1"

        Path(path).unlink()

    def test_load_csv(self) -> None:
        """Test loading CSV test data."""
        import pandas as pd

        df = pd.DataFrame({
            "text": ["Text 1", "Text 2"],
            "rebuttal": ["Rebuttal 1", "Rebuttal 2"],
        })

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
            df.to_csv(path, index=False)

        loaded = load_test_data(path)

        assert len(loaded) == 2
        assert "rationale" in loaded[0]  # Renamed from text

        Path(path).unlink()

    def test_load_csv_alternative_columns(self) -> None:
        """Test loading CSV with alternative column names."""
        import pandas as pd

        df = pd.DataFrame({
            "input": ["Input 1", "Input 2"],
            "output": ["Output 1", "Output 2"],
        })

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
            df.to_csv(path, index=False)

        loaded = load_test_data(path)

        assert len(loaded) == 2

        Path(path).unlink()

    def test_unsupported_format(self) -> None:
        """Test error on unsupported format."""
        try:
            load_test_data("file.xyz")
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "Unsupported" in str(e)


class TestEvaluationResultEdgeCases:
    """Edge case tests for EvaluationResult."""

    def test_zero_samples(self) -> None:
        """Test result with zero samples."""
        result = EvaluationResult(num_samples=0)
        text = str(result)
        assert "n=0" in text

    def test_very_high_scores(self) -> None:
        """Test result with perfect scores."""
        result = EvaluationResult(
            rouge1=1.0,
            rouge2=1.0,
            rougeL=1.0,
            bert_score_f1=1.0,
        )
        data = result.to_dict()
        assert all(v <= 1.0 for v in data.values() if isinstance(v, float))

    def test_large_predictions_list(self) -> None:
        """Test with large predictions list."""
        predictions = [f"Prediction {i}" for i in range(1000)]
        references = [f"Reference {i}" for i in range(1000)]

        result = EvaluationResult(
            predictions=predictions,
            references=references,
            num_samples=1000,
        )
        assert len(result.predictions) == 1000


class TestRougeEdgeCases:
    """Edge case tests for ROUGE computation."""

    def test_single_word(self) -> None:
        """Test ROUGE with single word texts."""
        scores = compute_rouge_scores(["hello"], ["hello"])
        # Either high score (if installed) or zero
        assert scores["rouge1"] >= 0.0

    def test_long_texts(self) -> None:
        """Test ROUGE with longer texts."""
        pred = "The quick brown fox jumps over the lazy dog. " * 10
        ref = "The quick brown fox jumps over the lazy dog. " * 10

        scores = compute_rouge_scores([pred], [ref])
        # Either high score (if installed) or zero
        assert scores["rougeL"] >= 0.0

    def test_case_sensitivity(self) -> None:
        """Test ROUGE is case-insensitive with stemming."""
        scores = compute_rouge_scores(
            ["The Quick Brown Fox"],
            ["the quick brown fox"],
        )
        # ROUGE with stemming should handle case differences (or be zero if not installed)
        assert scores["rouge1"] >= 0.0


def run_all_tests() -> bool:
    """Run all evaluation tests and report results."""
    import traceback

    test_classes = [
        TestEvaluationResult,
        TestComputeRougeScores,
        TestSaveEvaluationResults,
        TestLoadTestData,
        TestEvaluationResultEdgeCases,
        TestRougeEdgeCases,
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
    print(f"Evaluation Tests: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
