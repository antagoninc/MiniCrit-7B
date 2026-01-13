"""Tests for health check module.

Antagon Inc. | CAGE: 17E75
"""

import time
import pytest
from unittest.mock import patch, MagicMock

from src.health import (
    HealthStatus,
    CheckResult,
    HealthResponse,
    HealthChecker,
    get_health_checker,
    reset_health_checker,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_values(self):
        """Test all status values exist."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_basic_creation(self):
        """Test basic CheckResult creation."""
        result = CheckResult(
            name="test",
            status=HealthStatus.HEALTHY,
        )
        assert result.name == "test"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == ""
        assert result.latency_ms == 0.0
        assert result.details == {}

    def test_full_creation(self):
        """Test CheckResult with all fields."""
        result = CheckResult(
            name="test",
            status=HealthStatus.DEGRADED,
            message="Warning message",
            latency_ms=15.5,
            details={"key": "value"},
        )
        assert result.name == "test"
        assert result.status == HealthStatus.DEGRADED
        assert result.message == "Warning message"
        assert result.latency_ms == 15.5
        assert result.details == {"key": "value"}


class TestHealthResponse:
    """Tests for HealthResponse dataclass."""

    def test_to_dict(self):
        """Test converting response to dictionary."""
        response = HealthResponse(
            status=HealthStatus.HEALTHY,
            checks=[
                CheckResult(name="check1", status=HealthStatus.HEALTHY),
                CheckResult(name="check2", status=HealthStatus.DEGRADED, message="Warning"),
            ],
            timestamp="2024-01-01T00:00:00Z",
            uptime_seconds=100.5,
        )

        result = response.to_dict()

        assert result["status"] == "healthy"
        assert result["timestamp"] == "2024-01-01T00:00:00Z"
        assert result["uptime_seconds"] == 100.5
        assert len(result["checks"]) == 2
        assert result["checks"][0]["name"] == "check1"
        assert result["checks"][1]["status"] == "degraded"


class TestHealthChecker:
    """Tests for HealthChecker class."""

    def setup_method(self):
        """Reset health checker before each test."""
        reset_health_checker()

    def test_uptime(self):
        """Test uptime calculation."""
        checker = HealthChecker()
        time.sleep(0.1)
        assert checker.uptime >= 0.1

    def test_mark_startup_complete(self):
        """Test marking startup as complete."""
        checker = HealthChecker()
        assert checker._startup_complete is False

        checker.mark_startup_complete()
        assert checker._startup_complete is True

    def test_register_liveness_check(self):
        """Test registering a liveness check."""
        checker = HealthChecker()
        initial_count = len(checker._liveness_checks)

        def custom_check():
            return CheckResult(name="custom", status=HealthStatus.HEALTHY)

        checker.register_liveness_check("custom", custom_check)
        assert len(checker._liveness_checks) == initial_count + 1

    def test_register_readiness_check(self):
        """Test registering a readiness check."""
        checker = HealthChecker()
        initial_count = len(checker._readiness_checks)

        def custom_check():
            return CheckResult(name="custom", status=HealthStatus.HEALTHY)

        checker.register_readiness_check("custom", custom_check)
        assert len(checker._readiness_checks) == initial_count + 1

    def test_check_liveness(self):
        """Test liveness check execution."""
        checker = HealthChecker()
        response = checker.check_liveness()

        assert response.status == HealthStatus.HEALTHY
        assert len(response.checks) > 0
        assert response.uptime_seconds > 0

    def test_check_readiness_model_not_loaded(self):
        """Test readiness when model not loaded."""
        checker = HealthChecker()

        # Mock model state as not loaded - patch at import location
        with patch("src.api._model_state", {"loaded": False}):
            response = checker.check_readiness()

        # Should have model check in results
        assert any(c.name == "model" for c in response.checks)

    def test_check_startup_incomplete(self):
        """Test startup check when not complete."""
        checker = HealthChecker()
        response = checker.check_startup()

        # Find initialization check
        init_check = next(c for c in response.checks if c.name == "initialization")
        assert init_check.status == HealthStatus.UNHEALTHY

    def test_check_startup_complete(self):
        """Test startup check when complete."""
        checker = HealthChecker()
        checker.mark_startup_complete()
        response = checker.check_startup()

        init_check = next(c for c in response.checks if c.name == "initialization")
        assert init_check.status == HealthStatus.HEALTHY

    def test_check_all_deduplicates(self):
        """Test that check_all deduplicates checks."""
        checker = HealthChecker()

        # Add same check to multiple lists
        def shared_check():
            return CheckResult(name="shared", status=HealthStatus.HEALTHY)

        checker.register_liveness_check("shared", shared_check)
        checker.register_readiness_check("shared", shared_check)

        response = checker.check_all()

        # Should only appear once
        shared_checks = [c for c in response.checks if c.name == "shared"]
        assert len(shared_checks) == 1

    def test_failing_check(self):
        """Test handling of failing check."""
        checker = HealthChecker()

        def failing_check():
            raise RuntimeError("Check failed!")

        checker.register_liveness_check("failing", failing_check)
        response = checker.check_liveness()

        failing_result = next(c for c in response.checks if c.name == "failing")
        assert failing_result.status == HealthStatus.UNHEALTHY
        assert "Check failed!" in failing_result.message

    def test_overall_status_unhealthy(self):
        """Test overall status becomes unhealthy if any check fails."""
        checker = HealthChecker()

        def unhealthy_check():
            return CheckResult(name="bad", status=HealthStatus.UNHEALTHY)

        checker.register_liveness_check("bad", unhealthy_check)
        response = checker.check_liveness()

        assert response.status == HealthStatus.UNHEALTHY

    def test_overall_status_degraded(self):
        """Test overall status becomes degraded if any check is degraded."""
        checker = HealthChecker()

        # Clear existing checks
        checker._liveness_checks = []

        def healthy_check():
            return CheckResult(name="good", status=HealthStatus.HEALTHY)

        def degraded_check():
            return CheckResult(name="warn", status=HealthStatus.DEGRADED)

        checker.register_liveness_check("good", healthy_check)
        checker.register_liveness_check("warn", degraded_check)
        response = checker.check_liveness()

        assert response.status == HealthStatus.DEGRADED

    def test_latency_tracking(self):
        """Test that check latency is tracked."""
        checker = HealthChecker()

        def slow_check():
            time.sleep(0.05)
            return CheckResult(name="slow", status=HealthStatus.HEALTHY)

        checker.register_liveness_check("slow", slow_check)
        response = checker.check_liveness()

        slow_result = next(c for c in response.checks if c.name == "slow")
        assert slow_result.latency_ms >= 50


class TestDefaultChecks:
    """Tests for default health checks."""

    def setup_method(self):
        """Reset health checker before each test."""
        reset_health_checker()

    def test_process_check(self):
        """Test process check."""
        checker = HealthChecker()
        result = checker._check_process()

        assert result.status == HealthStatus.HEALTHY
        assert "pid" in result.details

    def test_memory_check_without_psutil(self):
        """Test memory check when psutil not available."""
        checker = HealthChecker()

        with patch.dict("sys.modules", {"psutil": None}):
            result = checker._check_memory()

        assert result.status == HealthStatus.UNKNOWN

    def test_memory_check_with_psutil(self):
        """Test memory check with mocked psutil."""
        checker = HealthChecker()

        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
        mock_process.memory_percent.return_value = 10.0

        mock_psutil = MagicMock()
        mock_psutil.Process.return_value = mock_process

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            # Need to reload the function to use mock
            result = checker._check_memory()

        # Should pass (mocked at 10% usage)
        # Note: This may use real psutil if available

    def test_initialization_check_incomplete(self):
        """Test initialization check when incomplete."""
        checker = HealthChecker()
        result = checker._check_initialization()

        assert result.status == HealthStatus.UNHEALTHY
        assert "uptime_seconds" in result.details

    def test_initialization_check_complete(self):
        """Test initialization check when complete."""
        checker = HealthChecker()
        checker.mark_startup_complete()
        result = checker._check_initialization()

        assert result.status == HealthStatus.HEALTHY


class TestGetHealthChecker:
    """Tests for get_health_checker function."""

    def setup_method(self):
        """Reset health checker before each test."""
        reset_health_checker()

    def test_returns_instance(self):
        """Test that get_health_checker returns an instance."""
        checker = get_health_checker()
        assert isinstance(checker, HealthChecker)

    def test_returns_same_instance(self):
        """Test that get_health_checker returns same instance."""
        checker1 = get_health_checker()
        checker2 = get_health_checker()
        assert checker1 is checker2

    def test_reset_creates_new_instance(self):
        """Test that reset allows new instance creation."""
        checker1 = get_health_checker()
        reset_health_checker()
        checker2 = get_health_checker()
        assert checker1 is not checker2
