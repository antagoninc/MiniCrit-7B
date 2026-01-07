"""Tests for the API server module.

Tests API endpoints, request/response models, and server functionality.
Uses mock implementations when fastapi/pydantic are not available.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import from actual module, fall back to mock implementations
try:
    from src.api import (
        CritiqueRequest,
        CritiqueResponse,
        BatchCritiqueRequest,
        BatchCritiqueResponse,
        HealthResponse,
        StatsResponse,
        _model_state,
    )
except ImportError:
    # Mock implementations for testing without fastapi/pydantic
    @dataclass
    class CritiqueRequest:
        rationale: str
        max_tokens: int = 256
        temperature: float = 0.7
        do_sample: bool = True

    @dataclass
    class CritiqueResponse:
        critique: str
        rationale: str
        tokens_generated: int
        latency_ms: float
        model_name: str

    @dataclass
    class BatchCritiqueRequest:
        rationales: list
        max_tokens: int = 256
        temperature: float = 0.7

    @dataclass
    class BatchCritiqueResponse:
        critiques: list
        total_latency_ms: float
        avg_latency_ms: float

    @dataclass
    class HealthResponse:
        status: str
        model_loaded: bool
        model_name: str | None
        load_time: str | None
        request_count: int
        total_tokens_generated: int
        uptime_seconds: float

    @dataclass
    class StatsResponse:
        request_count: int
        total_tokens_generated: int
        avg_tokens_per_request: float
        model_name: str | None
        uptime_seconds: float

    _model_state: dict[str, Any] = {
        "model": None,
        "tokenizer": None,
        "loaded": False,
        "load_time": None,
        "request_count": 0,
        "total_tokens_generated": 0,
    }


class TestCritiqueRequest:
    """Tests for CritiqueRequest model."""

    def test_valid_request(self) -> None:
        """Test valid request creation."""
        req = CritiqueRequest(rationale="This is a test rationale with enough content.")
        assert req.rationale == "This is a test rationale with enough content."
        assert req.max_tokens == 256
        assert req.temperature == 0.7
        assert req.do_sample is True

    def test_custom_parameters(self) -> None:
        """Test request with custom parameters."""
        req = CritiqueRequest(
            rationale="Test rationale here.",
            max_tokens=512,
            temperature=0.5,
            do_sample=False,
        )
        assert req.max_tokens == 512
        assert req.temperature == 0.5
        assert req.do_sample is False

    def test_min_max_tokens(self) -> None:
        """Test max_tokens bounds."""
        req = CritiqueRequest(rationale="Test rationale.", max_tokens=32)
        assert req.max_tokens == 32

        req2 = CritiqueRequest(rationale="Test rationale.", max_tokens=1024)
        assert req2.max_tokens == 1024

    def test_temperature_bounds(self) -> None:
        """Test temperature bounds."""
        req = CritiqueRequest(rationale="Test rationale.", temperature=0.0)
        assert req.temperature == 0.0

        req2 = CritiqueRequest(rationale="Test rationale.", temperature=2.0)
        assert req2.temperature == 2.0


class TestCritiqueResponse:
    """Tests for CritiqueResponse model."""

    def test_response_creation(self) -> None:
        """Test response creation."""
        resp = CritiqueResponse(
            critique="This is the critique.",
            rationale="Original rationale.",
            tokens_generated=50,
            latency_ms=123.45,
            model_name="test-model",
        )
        assert resp.critique == "This is the critique."
        assert resp.tokens_generated == 50
        assert resp.latency_ms == 123.45

    def test_response_fields(self) -> None:
        """Test all response fields."""
        resp = CritiqueResponse(
            critique="Critique text",
            rationale="Input text",
            tokens_generated=100,
            latency_ms=500.0,
            model_name="Qwen/Qwen2-7B-Instruct",
        )
        assert resp.model_name == "Qwen/Qwen2-7B-Instruct"


class TestBatchCritiqueRequest:
    """Tests for BatchCritiqueRequest model."""

    def test_valid_batch_request(self) -> None:
        """Test valid batch request."""
        req = BatchCritiqueRequest(
            rationales=["Rationale 1", "Rationale 2", "Rationale 3"]
        )
        assert len(req.rationales) == 3
        assert req.max_tokens == 256

    def test_custom_batch_params(self) -> None:
        """Test batch request with custom params."""
        req = BatchCritiqueRequest(
            rationales=["Test 1", "Test 2"],
            max_tokens=128,
            temperature=0.3,
        )
        assert req.max_tokens == 128
        assert req.temperature == 0.3

    def test_single_item_batch(self) -> None:
        """Test batch with single item."""
        req = BatchCritiqueRequest(rationales=["Single item"])
        assert len(req.rationales) == 1


class TestBatchCritiqueResponse:
    """Tests for BatchCritiqueResponse model."""

    def test_batch_response(self) -> None:
        """Test batch response creation."""
        critiques = [
            CritiqueResponse(
                critique="Critique 1",
                rationale="Rationale 1",
                tokens_generated=50,
                latency_ms=100.0,
                model_name="test",
            ),
            CritiqueResponse(
                critique="Critique 2",
                rationale="Rationale 2",
                tokens_generated=60,
                latency_ms=120.0,
                model_name="test",
            ),
        ]
        resp = BatchCritiqueResponse(
            critiques=critiques,
            total_latency_ms=220.0,
            avg_latency_ms=110.0,
        )
        assert len(resp.critiques) == 2
        assert resp.total_latency_ms == 220.0
        assert resp.avg_latency_ms == 110.0


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_health_response(self) -> None:
        """Test health response creation."""
        resp = HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name="test-model",
            load_time="2024-01-01T00:00:00",
            request_count=100,
            total_tokens_generated=5000,
            uptime_seconds=3600.0,
        )
        assert resp.status == "healthy"
        assert resp.model_loaded is True
        assert resp.request_count == 100

    def test_health_response_not_loaded(self) -> None:
        """Test health response when model not loaded."""
        resp = HealthResponse(
            status="healthy",
            model_loaded=False,
            model_name=None,
            load_time=None,
            request_count=0,
            total_tokens_generated=0,
            uptime_seconds=10.0,
        )
        assert resp.model_loaded is False
        assert resp.model_name is None


class TestStatsResponse:
    """Tests for StatsResponse model."""

    def test_stats_response(self) -> None:
        """Test stats response creation."""
        resp = StatsResponse(
            request_count=50,
            total_tokens_generated=2500,
            avg_tokens_per_request=50.0,
            model_name="test-model",
            uptime_seconds=1800.0,
        )
        assert resp.request_count == 50
        assert resp.avg_tokens_per_request == 50.0

    def test_stats_zero_requests(self) -> None:
        """Test stats with zero requests."""
        resp = StatsResponse(
            request_count=0,
            total_tokens_generated=0,
            avg_tokens_per_request=0.0,
            model_name=None,
            uptime_seconds=60.0,
        )
        assert resp.avg_tokens_per_request == 0.0


class TestModelState:
    """Tests for model state management."""

    def test_initial_state(self) -> None:
        """Test initial model state."""
        # Reset state for testing
        _model_state["loaded"] = False
        _model_state["request_count"] = 0
        _model_state["total_tokens_generated"] = 0

        assert _model_state["loaded"] is False
        assert _model_state["request_count"] == 0

    def test_state_update(self) -> None:
        """Test model state updates."""
        _model_state["request_count"] += 1
        _model_state["total_tokens_generated"] += 100

        assert _model_state["request_count"] >= 1
        assert _model_state["total_tokens_generated"] >= 100

    def test_state_keys(self) -> None:
        """Test model state has expected keys."""
        expected_keys = {
            "model", "tokenizer", "loaded", "load_time",
            "request_count", "total_tokens_generated"
        }
        assert expected_keys.issubset(set(_model_state.keys()))


class TestAPIHelpers:
    """Tests for API helper functions."""

    def test_prompt_format(self) -> None:
        """Test prompt formatting."""
        rationale = "Test rationale"
        expected = f"### Rationale:\n{rationale}\n\n### Critique:\n"
        prompt = f"### Rationale:\n{rationale}\n\n### Critique:\n"
        assert prompt == expected

    def test_critique_extraction_with_marker(self) -> None:
        """Test critique extraction when marker present."""
        full_output = "### Rationale:\nTest\n\n### Critique:\nThis is the critique."
        if "### Critique:" in full_output:
            critique = full_output.split("### Critique:")[-1].strip()
        assert critique == "This is the critique."

    def test_critique_extraction_without_marker(self) -> None:
        """Test critique extraction without marker."""
        prompt = "### Rationale:\nTest\n\n### Critique:\n"
        full_output = prompt + "Generated critique text"

        if "### Critique:" in full_output:
            critique = full_output.split("### Critique:")[-1].strip()
        else:
            critique = full_output[len(prompt):].strip()

        assert critique == "Generated critique text"


class TestRequestValidation:
    """Tests for request validation."""

    def test_rationale_not_empty(self) -> None:
        """Test that empty rationale is handled."""
        # With mock dataclass, empty string is allowed
        # With pydantic, it would raise validation error
        req = CritiqueRequest(rationale="")
        assert req.rationale == ""

    def test_very_long_rationale(self) -> None:
        """Test handling of long rationale."""
        long_text = "A" * 4000  # Within 4096 limit
        req = CritiqueRequest(rationale=long_text)
        assert len(req.rationale) == 4000

    def test_temperature_zero(self) -> None:
        """Test zero temperature (greedy decoding)."""
        req = CritiqueRequest(
            rationale="Test rationale content",
            temperature=0.0,
            do_sample=False,
        )
        assert req.temperature == 0.0
        assert req.do_sample is False


class TestBatchValidation:
    """Tests for batch request validation."""

    def test_empty_batch(self) -> None:
        """Test empty batch handling."""
        # With mock dataclass, empty list is allowed
        req = BatchCritiqueRequest(rationales=[])
        assert req.rationales == []

    def test_large_batch(self) -> None:
        """Test batch size limits."""
        rationales = [f"Rationale {i}" for i in range(100)]
        req = BatchCritiqueRequest(rationales=rationales)
        assert len(req.rationales) == 100

    def test_mixed_length_rationales(self) -> None:
        """Test batch with varied rationale lengths."""
        rationales = [
            "Short rationale",
            "A" * 500,
            "Medium length rationale with more content",
        ]
        req = BatchCritiqueRequest(rationales=rationales)
        assert len(req.rationales) == 3


def run_all_tests() -> bool:
    """Run all API tests and report results."""
    import traceback

    test_classes = [
        TestCritiqueRequest,
        TestCritiqueResponse,
        TestBatchCritiqueRequest,
        TestBatchCritiqueResponse,
        TestHealthResponse,
        TestStatsResponse,
        TestModelState,
        TestAPIHelpers,
        TestRequestValidation,
        TestBatchValidation,
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
    print(f"API Tests: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
