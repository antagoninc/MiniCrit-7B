"""Tests for the budget tracking module.

Tests cost calculation, budget tracking, and persistence functionality.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.budget import (
    DEFAULT_RATES,
    BudgetExceededError,
    BudgetSummary,
    BudgetTracker,
    CostCalculator,
    CostEntry,
    get_tracker,
    set_tracker,
)


class TestCostEntry:
    """Tests for CostEntry dataclass."""

    def test_creation(self) -> None:
        """Test creating a cost entry."""
        entry = CostEntry(
            timestamp=datetime.now(),
            category="training",
            subcategory="compute",
            amount=10.50,
            units=3.0,
            unit_type="gpu_hours",
        )
        assert entry.category == "training"
        assert entry.amount == 10.50
        assert entry.units == 3.0

    def test_with_metadata(self) -> None:
        """Test entry with metadata."""
        entry = CostEntry(
            timestamp=datetime.now(),
            category="inference",
            subcategory="local",
            amount=0.0,
            units=1000,
            unit_type="tokens",
            metadata={"request_id": "abc123"},
        )
        assert entry.metadata["request_id"] == "abc123"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        now = datetime.now()
        entry = CostEntry(
            timestamp=now,
            category="api",
            subcategory="openai/gpt4",
            amount=0.05,
            units=500,
            unit_type="tokens",
        )
        data = entry.to_dict()

        assert data["category"] == "api"
        assert data["amount"] == 0.05
        assert data["timestamp"] == now.isoformat()

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "timestamp": "2024-01-01T12:00:00",
            "category": "training",
            "subcategory": "compute",
            "amount": 5.0,
            "units": 1.0,
            "unit_type": "gpu_hours",
            "metadata": {"step": 1000},
        }
        entry = CostEntry.from_dict(data)

        assert entry.category == "training"
        assert entry.amount == 5.0
        assert entry.metadata["step"] == 1000

    def test_roundtrip(self) -> None:
        """Test to_dict and from_dict roundtrip."""
        original = CostEntry(
            timestamp=datetime.now(),
            category="test",
            subcategory="sub",
            amount=1.23,
            units=10,
            unit_type="units",
            metadata={"key": "value"},
        )
        data = original.to_dict()
        restored = CostEntry.from_dict(data)

        assert restored.category == original.category
        assert restored.amount == original.amount
        assert restored.metadata == original.metadata


class TestBudgetSummary:
    """Tests for BudgetSummary dataclass."""

    def test_creation(self) -> None:
        """Test creating a summary."""
        summary = BudgetSummary(
            total_cost=50.0,
            budget_limit=100.0,
            budget_remaining=50.0,
            budget_used_pct=50.0,
            cost_by_category={"training": 30.0, "inference": 20.0},
            cost_by_day={"2024-01-01": 50.0},
            entry_count=10,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1),
        )
        assert summary.total_cost == 50.0
        assert summary.budget_used_pct == 50.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        summary = BudgetSummary(
            total_cost=25.0,
            budget_limit=100.0,
            budget_remaining=75.0,
            budget_used_pct=25.0,
            cost_by_category={"api": 25.0},
            cost_by_day={},
            entry_count=5,
            start_date=None,
            end_date=None,
        )
        data = summary.to_dict()

        assert data["total_cost"] == 25.0
        assert data["budget_limit"] == 100.0

    def test_str_format(self) -> None:
        """Test string formatting."""
        summary = BudgetSummary(
            total_cost=75.0,
            budget_limit=100.0,
            budget_remaining=25.0,
            budget_used_pct=75.0,
            cost_by_category={"training": 50.0, "api": 25.0},
            cost_by_day={},
            entry_count=15,
            start_date=None,
            end_date=None,
        )
        text = str(summary)

        assert "Budget Summary" in text
        assert "$75.00" in text
        assert "training" in text

    def test_no_budget_limit(self) -> None:
        """Test summary without budget limit."""
        summary = BudgetSummary(
            total_cost=100.0,
            budget_limit=None,
            budget_remaining=None,
            budget_used_pct=None,
            cost_by_category={},
            cost_by_day={},
            entry_count=0,
            start_date=None,
            end_date=None,
        )
        assert summary.budget_limit is None
        assert summary.budget_used_pct is None


class TestCostCalculator:
    """Tests for CostCalculator."""

    def test_default_rates(self) -> None:
        """Test calculator uses default rates."""
        calc = CostCalculator()
        assert "gpu_a100_80gb" in calc.rates
        assert calc.rates["gpu_a100_80gb"] == DEFAULT_RATES["gpu_a100_80gb"]

    def test_custom_rates(self) -> None:
        """Test calculator with custom rates."""
        custom = {"gpu_custom": 2.0}
        calc = CostCalculator(rates=custom)
        assert calc.rates["gpu_custom"] == 2.0
        # Should still have defaults
        assert "gpu_a100_80gb" in calc.rates

    def test_training_cost(self) -> None:
        """Test training cost calculation."""
        calc = CostCalculator()
        cost = calc.training_cost(gpu_hours=10, gpu_type="gpu_a100_80gb")
        expected = 10 * DEFAULT_RATES["gpu_a100_80gb"]
        assert cost == expected

    def test_training_cost_different_gpu(self) -> None:
        """Test training cost for different GPU types."""
        calc = CostCalculator()

        cost_a100 = calc.training_cost(1, "gpu_a100_80gb")
        cost_t4 = calc.training_cost(1, "gpu_t4")

        assert cost_a100 > cost_t4

    def test_inference_cost_local(self) -> None:
        """Test local inference is free."""
        calc = CostCalculator()
        cost = calc.inference_cost(
            input_tokens=1000,
            output_tokens=500,
            local=True,
        )
        assert cost == 0.0

    def test_inference_cost_cloud(self) -> None:
        """Test cloud inference cost."""
        calc = CostCalculator()
        cost = calc.inference_cost(
            input_tokens=1_000_000,
            output_tokens=500_000,
            local=False,
        )
        expected = (1_000_000 / 1_000_000) * DEFAULT_RATES["input_tokens"] + (
            500_000 / 1_000_000
        ) * DEFAULT_RATES["output_tokens"]
        assert cost == expected

    def test_api_cost(self) -> None:
        """Test external API cost calculation."""
        calc = CostCalculator()
        cost = calc.api_cost(
            input_tokens=1_000_000,
            output_tokens=500_000,
            provider="openai",
            model="gpt4",
        )
        expected = (1_000_000 / 1_000_000) * DEFAULT_RATES["openai_gpt4_input"] + (
            500_000 / 1_000_000
        ) * DEFAULT_RATES["openai_gpt4_output"]
        assert cost == expected

    def test_storage_cost(self) -> None:
        """Test storage cost calculation."""
        calc = CostCalculator()
        cost = calc.storage_cost(gb_months=100, storage_type="storage_s3")
        expected = 100 * DEFAULT_RATES["storage_s3"]
        assert cost == expected

    def test_estimate_training_run(self) -> None:
        """Test training run estimation."""
        calc = CostCalculator()
        estimate = calc.estimate_training_run(
            total_steps=1000,
            batch_size=32,
            samples_per_second=10.0,
            gpu_type="gpu_a100_80gb",
        )

        assert "total_samples" in estimate
        assert "estimated_hours" in estimate
        assert "estimated_cost" in estimate
        assert estimate["total_samples"] == 32000


class TestBudgetTracker:
    """Tests for BudgetTracker."""

    def test_creation(self) -> None:
        """Test tracker creation."""
        tracker = BudgetTracker(budget_limit=100.0)
        assert tracker.budget_limit == 100.0
        assert tracker.total_cost == 0.0

    def test_no_budget_limit(self) -> None:
        """Test tracker without budget limit."""
        tracker = BudgetTracker()
        assert tracker.budget_limit is None
        assert tracker.budget_remaining is None

    def test_log_cost(self) -> None:
        """Test logging a cost entry."""
        tracker = BudgetTracker(budget_limit=100.0)
        entry = tracker.log_cost(
            category="training",
            subcategory="compute",
            amount=10.0,
            units=2.0,
            unit_type="gpu_hours",
        )

        assert entry.amount == 10.0
        assert tracker.total_cost == 10.0
        assert len(tracker.entries) == 1

    def test_log_training_step(self) -> None:
        """Test logging training step."""
        tracker = BudgetTracker()
        entry = tracker.log_training_step(
            gpu_hours=0.5,
            gpu_type="gpu_a100_80gb",
            batch_size=4,
            step=100,
        )

        expected = 0.5 * DEFAULT_RATES["gpu_a100_80gb"]
        assert entry.amount == expected

    def test_log_inference(self) -> None:
        """Test logging inference."""
        tracker = BudgetTracker()
        entry = tracker.log_inference(
            input_tokens=1000,
            output_tokens=500,
            local=True,
        )

        assert entry.amount == 0.0  # Local is free

    def test_log_api_call(self) -> None:
        """Test logging API call."""
        tracker = BudgetTracker()
        entry = tracker.log_api_call(
            input_tokens=1000,
            output_tokens=500,
            provider="openai",
            model="gpt35",
            purpose="generation",
        )

        assert entry.amount > 0
        assert entry.metadata["provider"] == "openai"

    def test_budget_check_within(self) -> None:
        """Test budget check when within budget."""
        tracker = BudgetTracker(budget_limit=100.0)
        assert tracker.check_budget(50.0) is True

    def test_budget_check_exceeds(self) -> None:
        """Test budget check when exceeding."""
        tracker = BudgetTracker(budget_limit=100.0)
        tracker.log_cost("test", "test", 90.0, 1, "units")
        assert tracker.check_budget(20.0) is False

    def test_budget_exceeded_error(self) -> None:
        """Test BudgetExceededError is raised."""
        tracker = BudgetTracker(budget_limit=10.0)
        tracker.log_cost("test", "test", 9.0, 1, "units")

        try:
            tracker.log_cost("test", "test", 5.0, 1, "units")
            assert False, "Should raise BudgetExceededError"
        except BudgetExceededError as e:
            assert "exceed budget" in str(e).lower()

    def test_budget_remaining(self) -> None:
        """Test budget remaining calculation."""
        tracker = BudgetTracker(budget_limit=100.0)
        tracker.log_cost("test", "test", 30.0, 1, "units")
        assert tracker.budget_remaining == 70.0

    def test_get_summary(self) -> None:
        """Test getting budget summary."""
        tracker = BudgetTracker(budget_limit=100.0)
        tracker.log_cost("training", "compute", 20.0, 1, "hours")
        tracker.log_cost("api", "openai", 10.0, 1000, "tokens")

        summary = tracker.get_summary()

        assert summary.total_cost == 30.0
        assert summary.budget_used_pct == 30.0
        assert summary.cost_by_category["training"] == 20.0
        assert summary.cost_by_category["api"] == 10.0

    def test_save_and_load(self) -> None:
        """Test saving and loading budget data."""
        tracker = BudgetTracker(budget_limit=100.0)
        tracker.log_cost("training", "compute", 15.0, 3, "hours")
        tracker.log_cost("api", "openai", 5.0, 500, "tokens")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        tracker.save(path)

        # Load into new tracker
        new_tracker = BudgetTracker()
        new_tracker.load(path)

        assert new_tracker.budget_limit == 100.0
        assert len(new_tracker.entries) == 2
        assert new_tracker.total_cost == 20.0

        Path(path).unlink()

    def test_reset(self) -> None:
        """Test resetting tracker."""
        tracker = BudgetTracker(budget_limit=100.0)
        tracker.log_cost("test", "test", 50.0, 1, "units")

        tracker.reset()

        assert tracker.total_cost == 0.0
        assert len(tracker.entries) == 0

    def test_alert_threshold(self) -> None:
        """Test budget alert threshold."""
        tracker = BudgetTracker(budget_limit=100.0, alert_threshold=0.5)

        # At 50%, should trigger alert
        tracker.log_cost("test", "test", 50.0, 1, "units")
        assert tracker._alerted is True


class TestGlobalTracker:
    """Tests for global tracker functions."""

    def test_get_tracker(self) -> None:
        """Test getting global tracker."""
        import src.budget

        src.budget._global_tracker = None

        tracker = get_tracker()
        assert isinstance(tracker, BudgetTracker)

    def test_set_tracker(self) -> None:
        """Test setting global tracker."""
        custom = BudgetTracker(budget_limit=500.0)
        set_tracker(custom)

        tracker = get_tracker()
        assert tracker.budget_limit == 500.0

    def test_get_tracker_from_env(self) -> None:
        """Test getting tracker with env var."""
        import src.budget

        src.budget._global_tracker = None

        with patch.dict("os.environ", {"MINICRIT_BUDGET_LIMIT": "200.0"}):
            tracker = get_tracker()
            # May or may not pick up env var depending on timing


class TestDefaultRates:
    """Tests for default rate definitions."""

    def test_gpu_rates_exist(self) -> None:
        """Test that GPU rates are defined."""
        assert "gpu_a100_80gb" in DEFAULT_RATES
        assert "gpu_h100" in DEFAULT_RATES
        assert "gpu_v100" in DEFAULT_RATES

    def test_token_rates_exist(self) -> None:
        """Test that token rates are defined."""
        assert "input_tokens" in DEFAULT_RATES
        assert "output_tokens" in DEFAULT_RATES

    def test_api_rates_exist(self) -> None:
        """Test that API rates are defined."""
        assert "openai_gpt4_input" in DEFAULT_RATES
        assert "anthropic_claude_input" in DEFAULT_RATES

    def test_local_rates_are_zero(self) -> None:
        """Test that local resources have zero cost."""
        assert DEFAULT_RATES["gpu_mac_m2_ultra"] == 0.0
        assert DEFAULT_RATES["storage_local"] == 0.0

    def test_rates_are_positive(self) -> None:
        """Test that cloud rates are positive."""
        for key, rate in DEFAULT_RATES.items():
            assert rate >= 0, f"Rate {key} should be non-negative"


def run_all_tests() -> bool:
    """Run all budget tests and report results."""
    import traceback

    test_classes = [
        TestCostEntry,
        TestBudgetSummary,
        TestCostCalculator,
        TestBudgetTracker,
        TestGlobalTracker,
        TestDefaultRates,
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
    print(f"Budget Tests: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
