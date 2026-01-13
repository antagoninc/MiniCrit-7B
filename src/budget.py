"""Budget and cost tracking for MiniCrit operations.

This module provides cost tracking functionality for:
- Training runs (compute time, GPU hours)
- Inference operations (tokens generated)
- Dataset generation (API calls, tokens)
- Cloud resource usage

Example:
    >>> from src.budget import BudgetTracker, CostCalculator
    >>> tracker = BudgetTracker(budget_limit=100.0)
    >>> tracker.log_training_step(gpu_hours=0.1, batch_size=4)
    >>> tracker.log_inference(input_tokens=100, output_tokens=256)
    >>> print(tracker.get_summary())

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Default cost rates (USD)
DEFAULT_RATES = {
    # GPU compute rates (per hour)
    "gpu_a100_80gb": 3.50,
    "gpu_h100": 5.00,
    "gpu_v100": 1.50,
    "gpu_a10g": 1.00,
    "gpu_t4": 0.50,
    "gpu_mac_m2_ultra": 0.00,  # Local hardware, no cloud cost
    # Token rates (per 1M tokens)
    "input_tokens": 0.50,
    "output_tokens": 1.50,
    # API rates
    "openai_gpt4_input": 30.00,
    "openai_gpt4_output": 60.00,
    "openai_gpt35_input": 0.50,
    "openai_gpt35_output": 1.50,
    "anthropic_claude_input": 8.00,
    "anthropic_claude_output": 24.00,
    # Storage rates (per GB per month)
    "storage_s3": 0.023,
    "storage_local": 0.00,
}


@dataclass
class CostEntry:
    """A single cost entry in the budget tracker.

    Attributes:
        timestamp: When the cost was incurred.
        category: Category of the cost (training, inference, etc).
        subcategory: Specific type within category.
        amount: Cost amount in USD.
        units: Number of units consumed.
        unit_type: Type of unit (gpu_hours, tokens, etc).
        metadata: Additional context about the cost.
    """

    timestamp: datetime
    category: str
    subcategory: str
    amount: float
    units: float
    unit_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "category": self.category,
            "subcategory": self.subcategory,
            "amount": self.amount,
            "units": self.units,
            "unit_type": self.unit_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CostEntry":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            category=data["category"],
            subcategory=data["subcategory"],
            amount=data["amount"],
            units=data["units"],
            unit_type=data["unit_type"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class BudgetSummary:
    """Summary of budget usage.

    Attributes:
        total_cost: Total cost incurred.
        budget_limit: Maximum allowed budget.
        budget_remaining: Remaining budget.
        budget_used_pct: Percentage of budget used.
        cost_by_category: Breakdown by category.
        cost_by_day: Breakdown by day.
        entry_count: Number of cost entries.
        start_date: First entry date.
        end_date: Last entry date.
    """

    total_cost: float
    budget_limit: float | None
    budget_remaining: float | None
    budget_used_pct: float | None
    cost_by_category: dict[str, float]
    cost_by_day: dict[str, float]
    entry_count: int
    start_date: datetime | None
    end_date: datetime | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_cost": self.total_cost,
            "budget_limit": self.budget_limit,
            "budget_remaining": self.budget_remaining,
            "budget_used_pct": self.budget_used_pct,
            "cost_by_category": self.cost_by_category,
            "cost_by_day": self.cost_by_day,
            "entry_count": self.entry_count,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
        }

    def __str__(self) -> str:
        """Format summary as string."""
        lines = [
            "Budget Summary",
            "=" * 40,
            f"Total Cost: ${self.total_cost:.2f}",
        ]

        if self.budget_limit:
            lines.extend(
                [
                    f"Budget Limit: ${self.budget_limit:.2f}",
                    f"Remaining: ${self.budget_remaining:.2f}",
                    f"Used: {self.budget_used_pct:.1f}%",
                ]
            )

        lines.append("")
        lines.append("Cost by Category:")
        for cat, cost in sorted(self.cost_by_category.items(), key=lambda x: -x[1]):
            lines.append(f"  {cat}: ${cost:.2f}")

        return "\n".join(lines)


class CostCalculator:
    """Calculator for estimating costs.

    Provides methods for calculating costs based on resource usage
    and configurable rate cards.

    Example:
        >>> calc = CostCalculator()
        >>> training_cost = calc.training_cost(gpu_hours=10, gpu_type="gpu_a100_80gb")
        >>> inference_cost = calc.inference_cost(input_tokens=1000, output_tokens=500)
    """

    def __init__(self, rates: dict[str, float] | None = None):
        """Initialize the cost calculator.

        Args:
            rates: Custom rate dictionary. If None, uses defaults.
        """
        self.rates = {**DEFAULT_RATES, **(rates or {})}

    def training_cost(
        self,
        gpu_hours: float,
        gpu_type: str = "gpu_a100_80gb",
    ) -> float:
        """Calculate training compute cost.

        Args:
            gpu_hours: Number of GPU hours used.
            gpu_type: Type of GPU (see DEFAULT_RATES).

        Returns:
            Cost in USD.
        """
        rate = self.rates.get(gpu_type, self.rates["gpu_a100_80gb"])
        return gpu_hours * rate

    def inference_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        local: bool = True,
    ) -> float:
        """Calculate inference cost.

        Args:
            input_tokens: Number of input tokens processed.
            output_tokens: Number of output tokens generated.
            local: If True, assumes local inference (zero cost).

        Returns:
            Cost in USD.
        """
        if local:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * self.rates["input_tokens"]
        output_cost = (output_tokens / 1_000_000) * self.rates["output_tokens"]
        return input_cost + output_cost

    def api_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        provider: str = "openai",
        model: str = "gpt4",
    ) -> float:
        """Calculate external API cost.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            provider: API provider (openai, anthropic).
            model: Model name.

        Returns:
            Cost in USD.
        """
        input_key = f"{provider}_{model}_input"
        output_key = f"{provider}_{model}_output"

        input_rate = self.rates.get(input_key, 10.0)
        output_rate = self.rates.get(output_key, 30.0)

        input_cost = (input_tokens / 1_000_000) * input_rate
        output_cost = (output_tokens / 1_000_000) * output_rate

        return input_cost + output_cost

    def storage_cost(
        self,
        gb_months: float,
        storage_type: str = "storage_s3",
    ) -> float:
        """Calculate storage cost.

        Args:
            gb_months: GB-months of storage.
            storage_type: Type of storage.

        Returns:
            Cost in USD.
        """
        rate = self.rates.get(storage_type, self.rates["storage_s3"])
        return gb_months * rate

    def estimate_training_run(
        self,
        total_steps: int,
        batch_size: int,
        samples_per_second: float = 10.0,
        gpu_type: str = "gpu_a100_80gb",
    ) -> dict[str, Any]:
        """Estimate cost for a training run.

        Args:
            total_steps: Number of training steps.
            batch_size: Batch size.
            samples_per_second: Samples processed per second.
            gpu_type: GPU type for cost calculation.

        Returns:
            Dictionary with time and cost estimates.
        """
        total_samples = total_steps * batch_size
        total_seconds = total_samples / samples_per_second
        total_hours = total_seconds / 3600

        return {
            "total_samples": total_samples,
            "estimated_hours": total_hours,
            "estimated_cost": self.training_cost(total_hours, gpu_type),
            "gpu_type": gpu_type,
        }


class BudgetTracker:
    """Tracks costs and enforces budget limits.

    Maintains a log of all cost entries and provides budget enforcement
    and reporting capabilities.

    Example:
        >>> tracker = BudgetTracker(budget_limit=100.0)
        >>> tracker.log_training_step(gpu_hours=0.1, gpu_type="gpu_a100_80gb")
        >>> if tracker.check_budget():
        ...     print("Within budget")
        >>> tracker.save("budget_log.json")
    """

    def __init__(
        self,
        budget_limit: float | None = None,
        rates: dict[str, float] | None = None,
        alert_threshold: float = 0.8,
    ):
        """Initialize the budget tracker.

        Args:
            budget_limit: Maximum allowed budget in USD. None for unlimited.
            rates: Custom rate dictionary.
            alert_threshold: Percentage (0-1) at which to alert about budget.
        """
        self.budget_limit = budget_limit
        self.alert_threshold = alert_threshold
        self.calculator = CostCalculator(rates)
        self.entries: list[CostEntry] = []
        self._alerted = False

    @property
    def total_cost(self) -> float:
        """Get total cost incurred."""
        return sum(e.amount for e in self.entries)

    @property
    def budget_remaining(self) -> float | None:
        """Get remaining budget."""
        if self.budget_limit is None:
            return None
        return self.budget_limit - self.total_cost

    def check_budget(self, additional_cost: float = 0.0) -> bool:
        """Check if within budget.

        Args:
            additional_cost: Optional additional cost to check.

        Returns:
            True if within budget, False otherwise.
        """
        if self.budget_limit is None:
            return True

        projected = self.total_cost + additional_cost
        within_budget = projected <= self.budget_limit

        # Check for alert threshold
        if not self._alerted and projected >= self.budget_limit * self.alert_threshold:
            logger.warning(
                f"Budget alert: {projected / self.budget_limit * 100:.1f}% of budget used "
                f"(${projected:.2f} / ${self.budget_limit:.2f})"
            )
            self._alerted = True

        return within_budget

    def log_cost(
        self,
        category: str,
        subcategory: str,
        amount: float,
        units: float,
        unit_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> CostEntry:
        """Log a cost entry.

        Args:
            category: Cost category (training, inference, api, storage).
            subcategory: Specific subcategory.
            amount: Cost amount in USD.
            units: Number of units consumed.
            unit_type: Type of units.
            metadata: Additional context.

        Returns:
            The created cost entry.

        Raises:
            BudgetExceededError: If this would exceed the budget.
        """
        if not self.check_budget(amount):
            raise BudgetExceededError(
                f"Cost of ${amount:.2f} would exceed budget. "
                f"Current: ${self.total_cost:.2f}, Limit: ${self.budget_limit:.2f}"
            )

        entry = CostEntry(
            timestamp=datetime.now(),
            category=category,
            subcategory=subcategory,
            amount=amount,
            units=units,
            unit_type=unit_type,
            metadata=metadata or {},
        )

        self.entries.append(entry)

        logger.debug(
            f"Cost logged: ${amount:.4f} ({category}/{subcategory})",
            extra={"cost_entry": entry.to_dict()},
        )

        return entry

    def log_training_step(
        self,
        gpu_hours: float,
        gpu_type: str = "gpu_a100_80gb",
        batch_size: int | None = None,
        step: int | None = None,
    ) -> CostEntry:
        """Log a training step cost.

        Args:
            gpu_hours: GPU hours used.
            gpu_type: Type of GPU.
            batch_size: Optional batch size.
            step: Optional step number.

        Returns:
            The created cost entry.
        """
        cost = self.calculator.training_cost(gpu_hours, gpu_type)

        metadata: dict[str, Any] = {"gpu_type": gpu_type}
        if batch_size is not None:
            metadata["batch_size"] = batch_size
        if step is not None:
            metadata["step"] = step

        return self.log_cost(
            category="training",
            subcategory="compute",
            amount=cost,
            units=gpu_hours,
            unit_type="gpu_hours",
            metadata=metadata,
        )

    def log_inference(
        self,
        input_tokens: int,
        output_tokens: int,
        local: bool = True,
        request_id: str | None = None,
    ) -> CostEntry:
        """Log an inference cost.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            local: Whether inference was local (free).
            request_id: Optional request identifier.

        Returns:
            The created cost entry.
        """
        cost = self.calculator.inference_cost(input_tokens, output_tokens, local)
        total_tokens = input_tokens + output_tokens

        metadata: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "local": local,
        }
        if request_id:
            metadata["request_id"] = request_id

        return self.log_cost(
            category="inference",
            subcategory="local" if local else "cloud",
            amount=cost,
            units=total_tokens,
            unit_type="tokens",
            metadata=metadata,
        )

    def log_api_call(
        self,
        input_tokens: int,
        output_tokens: int,
        provider: str = "openai",
        model: str = "gpt4",
        purpose: str = "generation",
    ) -> CostEntry:
        """Log an external API call cost.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            provider: API provider.
            model: Model used.
            purpose: Purpose of the call.

        Returns:
            The created cost entry.
        """
        cost = self.calculator.api_cost(input_tokens, output_tokens, provider, model)
        total_tokens = input_tokens + output_tokens

        return self.log_cost(
            category="api",
            subcategory=f"{provider}/{model}",
            amount=cost,
            units=total_tokens,
            unit_type="tokens",
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "provider": provider,
                "model": model,
                "purpose": purpose,
            },
        )

    def get_summary(self) -> BudgetSummary:
        """Get a summary of budget usage.

        Returns:
            BudgetSummary with aggregated statistics.
        """
        # Calculate cost by category
        cost_by_category: dict[str, float] = {}
        for entry in self.entries:
            cost_by_category[entry.category] = (
                cost_by_category.get(entry.category, 0) + entry.amount
            )

        # Calculate cost by day
        cost_by_day: dict[str, float] = {}
        for entry in self.entries:
            day = entry.timestamp.strftime("%Y-%m-%d")
            cost_by_day[day] = cost_by_day.get(day, 0) + entry.amount

        # Date range
        start_date = min((e.timestamp for e in self.entries), default=None)
        end_date = max((e.timestamp for e in self.entries), default=None)

        # Budget percentages
        budget_remaining = self.budget_remaining
        budget_used_pct = None
        if self.budget_limit and self.budget_limit > 0:
            budget_used_pct = (self.total_cost / self.budget_limit) * 100

        return BudgetSummary(
            total_cost=self.total_cost,
            budget_limit=self.budget_limit,
            budget_remaining=budget_remaining,
            budget_used_pct=budget_used_pct,
            cost_by_category=cost_by_category,
            cost_by_day=cost_by_day,
            entry_count=len(self.entries),
            start_date=start_date,
            end_date=end_date,
        )

    def save(self, path: str | Path) -> None:
        """Save budget data to JSON file.

        Args:
            path: Path to save the JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "budget_limit": self.budget_limit,
            "alert_threshold": self.alert_threshold,
            "entries": [e.to_dict() for e in self.entries],
            "summary": self.get_summary().to_dict(),
            "saved_at": datetime.now().isoformat(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Budget data saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load budget data from JSON file.

        Args:
            path: Path to the JSON file.
        """
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        self.budget_limit = data.get("budget_limit")
        self.alert_threshold = data.get("alert_threshold", 0.8)
        self.entries = [CostEntry.from_dict(e) for e in data.get("entries", [])]

        logger.info(f"Loaded {len(self.entries)} entries from {path}")

    def reset(self) -> None:
        """Reset the tracker, clearing all entries."""
        self.entries.clear()
        self._alerted = False
        logger.info("Budget tracker reset")


class BudgetExceededError(Exception):
    """Raised when an operation would exceed the budget."""

    pass


# Global tracker instance
_global_tracker: BudgetTracker | None = None


def get_tracker() -> BudgetTracker:
    """Get the global budget tracker.

    Creates one if it doesn't exist, using environment variables
    for configuration.

    Returns:
        The global BudgetTracker instance.
    """
    global _global_tracker

    if _global_tracker is None:
        budget_limit = os.environ.get("MINICRIT_BUDGET_LIMIT")
        _global_tracker = BudgetTracker(budget_limit=float(budget_limit) if budget_limit else None)

    return _global_tracker


def set_tracker(tracker: BudgetTracker) -> None:
    """Set the global budget tracker.

    Args:
        tracker: The tracker to use globally.
    """
    global _global_tracker
    _global_tracker = tracker
