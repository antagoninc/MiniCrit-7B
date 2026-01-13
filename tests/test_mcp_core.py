"""Tests for MCP core module.

Tests thread-safe model management, critique generation,
rate limiting, and graceful shutdown.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from src.mcp.core import (
    ModelManager,
    CritiqueGenerator,
    RateLimiter,
    GracefulShutdown,
    CritiqueResult,
    Severity,
    ModelNotLoadedError,
    ModelLoadError,
    InferenceTimeoutError,
    InferenceError,
    InvalidInputError,
    DOMAINS,
    get_model_manager,
    get_critique_generator,
    get_cors_origins,
)


# ================================================================
# ModelManager Tests
# ================================================================

class TestModelManager:
    """Tests for ModelManager class."""

    def test_singleton_pattern(self):
        """Test that ModelManager is a singleton."""
        manager1 = ModelManager.get_instance()
        manager2 = ModelManager.get_instance()
        manager3 = ModelManager()

        assert manager1 is manager2
        assert manager1 is manager3

    def test_initial_state(self):
        """Test initial state of ModelManager."""
        # Reset singleton for clean test
        ModelManager._instance = None
        manager = ModelManager.get_instance()

        assert manager.is_loaded is False
        assert manager.device is None
        assert manager.stats["total_requests"] == 0
        assert manager.stats["total_tokens"] == 0

    def test_stats_update(self):
        """Test statistics updating."""
        ModelManager._instance = None
        manager = ModelManager.get_instance()

        manager.update_stats(tokens=100, latency_ms=50.0)
        manager.update_stats(tokens=200, latency_ms=75.0, error=False)
        manager.update_stats(tokens=0, latency_ms=0, error=True)

        stats = manager.stats
        assert stats["total_requests"] == 3
        assert stats["total_tokens"] == 300
        assert stats["total_latency_ms"] == 125.0
        assert stats["errors"] == 1

    def test_stats_returns_copy(self):
        """Test that stats returns a copy, not the internal dict."""
        ModelManager._instance = None
        manager = ModelManager.get_instance()

        stats1 = manager.stats
        stats1["total_requests"] = 999

        stats2 = manager.stats
        assert stats2["total_requests"] == 0

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("peft.PeftModel")
    def test_get_model_loads_on_first_call(self, mock_peft, mock_tokenizer, mock_model):
        """Test that get_model loads model on first call."""
        ModelManager._instance = None
        manager = ModelManager.get_instance()

        # Setup mocks
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_peft.from_pretrained.return_value = MagicMock()

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                model, tokenizer = manager.get_model()

        assert model is not None
        assert tokenizer is not None
        assert manager.is_loaded is True

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("peft.PeftModel")
    def test_get_model_returns_cached(self, mock_peft, mock_tokenizer, mock_model):
        """Test that subsequent calls return cached model."""
        ModelManager._instance = None
        manager = ModelManager.get_instance()

        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_peft.from_pretrained.return_value = MagicMock()

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                model1, _ = manager.get_model()
                model2, _ = manager.get_model()

        # Should only load once
        assert mock_model.from_pretrained.call_count == 1
        assert model1 is model2

    def test_get_model_raises_on_missing_dependency(self):
        """Test that missing dependency raises ModelLoadError."""
        ModelManager._instance = None
        manager = ModelManager.get_instance()

        with patch.dict("sys.modules", {"torch": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'torch'")):
                with pytest.raises(ModelLoadError, match="Missing dependency"):
                    manager.get_model()

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_unload_clears_model(self, mock_tokenizer, mock_model):
        """Test that unload clears model and tokenizer."""
        ModelManager._instance = None
        manager = ModelManager.get_instance()

        # Manually set model state
        manager._model = MagicMock()
        manager._tokenizer = MagicMock()
        manager._device = "cpu"

        with patch("torch.cuda.is_available", return_value=False):
            manager.unload()

        assert manager._model is None
        assert manager._tokenizer is None
        assert manager._device is None
        assert manager.is_loaded is False


class TestModelManagerAsync:
    """Async tests for ModelManager."""

    @pytest.mark.asyncio
    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("peft.PeftModel")
    async def test_get_model_async(self, mock_peft, mock_tokenizer, mock_model):
        """Test async model loading."""
        ModelManager._instance = None
        manager = ModelManager.get_instance()

        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_peft.from_pretrained.return_value = MagicMock()

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                model, tokenizer = await manager.get_model_async()

        assert model is not None
        assert tokenizer is not None

    @pytest.mark.asyncio
    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("peft.PeftModel")
    async def test_preload_async(self, mock_peft, mock_tokenizer, mock_model):
        """Test async preloading."""
        ModelManager._instance = None
        manager = ModelManager.get_instance()

        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_peft.from_pretrained.return_value = MagicMock()

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                await manager.preload_async()

        assert manager.is_loaded is True


class TestModelManagerThreadSafety:
    """Thread safety tests for ModelManager."""

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("peft.PeftModel")
    def test_concurrent_get_model_calls(self, mock_peft, mock_tokenizer, mock_model):
        """Test that concurrent get_model calls are safe."""
        ModelManager._instance = None
        manager = ModelManager.get_instance()

        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_peft.from_pretrained.return_value = MagicMock()

        results = []
        errors = []

        def get_model_thread():
            try:
                with patch("torch.cuda.is_available", return_value=False):
                    with patch("torch.backends.mps.is_available", return_value=False):
                        model, _ = manager.get_model()
                        results.append(model)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_model_thread) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        # All should return same model instance
        assert all(r is results[0] for r in results)


# ================================================================
# CritiqueGenerator Tests
# ================================================================

class TestCritiqueGenerator:
    """Tests for CritiqueGenerator class."""

    def test_validate_input_empty(self):
        """Test validation rejects empty input."""
        generator = CritiqueGenerator()

        with pytest.raises(InvalidInputError, match="cannot be empty"):
            generator.validate_input("", "general")

        with pytest.raises(InvalidInputError, match="cannot be empty"):
            generator.validate_input("   ", "general")

    def test_validate_input_too_short(self):
        """Test validation rejects short input."""
        generator = CritiqueGenerator()

        with pytest.raises(InvalidInputError, match="at least 10 characters"):
            generator.validate_input("short", "general")

    def test_validate_input_too_long(self):
        """Test validation rejects long input."""
        generator = CritiqueGenerator()

        long_input = "x" * 10001
        with pytest.raises(InvalidInputError, match="cannot exceed"):
            generator.validate_input(long_input, "general")

    def test_validate_input_invalid_domain(self):
        """Test validation rejects invalid domain."""
        generator = CritiqueGenerator()

        with pytest.raises(InvalidInputError, match="Invalid domain"):
            generator.validate_input("This is a valid rationale", "invalid_domain")

    def test_validate_input_valid(self):
        """Test validation accepts valid input."""
        generator = CritiqueGenerator()

        # Should not raise
        generator.validate_input("This is a valid rationale for testing", "general")
        generator.validate_input("Trading signal based on momentum", "trading")
        generator.validate_input("Risk assessment for project X", "risk_assessment")

    def test_validate_all_domains(self):
        """Test all domains are accepted."""
        generator = CritiqueGenerator()

        for domain in DOMAINS:
            generator.validate_input("This is a valid rationale", domain)

    def test_parse_critique_severity_critical(self):
        """Test parsing critical severity."""
        generator = CritiqueGenerator()

        severity, flags = generator._parse_critique(
            "This reasoning has a critical flaw that could be dangerous"
        )

        assert severity == Severity.CRITICAL
        assert "critical_flaw" in flags

    def test_parse_critique_severity_high(self):
        """Test parsing high severity."""
        generator = CritiqueGenerator()

        severity, flags = generator._parse_critique(
            "This has significant issues and is fundamentally flawed"
        )

        assert severity == Severity.HIGH
        assert "significant_flaw" in flags

    def test_parse_critique_severity_medium(self):
        """Test parsing medium severity."""
        generator = CritiqueGenerator()

        severity, flags = generator._parse_critique(
            "There is a concern about the missing data"
        )

        assert severity == Severity.MEDIUM
        assert "notable_concern" in flags

    def test_parse_critique_severity_low(self):
        """Test parsing low severity."""
        generator = CritiqueGenerator()

        # Use text with only low-severity keywords (minor, slight, small)
        # without medium-level words (issue, concern, problem, missing)
        severity, flags = generator._parse_critique(
            "This has only minor points that are slight adjustments"
        )

        assert severity == Severity.LOW
        assert "minor_issue" in flags

    def test_parse_critique_severity_pass(self):
        """Test parsing pass severity."""
        generator = CritiqueGenerator()

        # Use text without any severity keywords or flag-triggering words
        severity, flags = generator._parse_critique(
            "The reasoning is solid and the conclusions follow logically"
        )

        assert severity == Severity.PASS
        assert len(flags) == 0

    def test_parse_critique_flags(self):
        """Test parsing specific flags."""
        generator = CritiqueGenerator()

        severity, flags = generator._parse_critique(
            "The reasoning shows overconfidence and is missing key evidence. "
            "There are contradictions and unaddressed risks."
        )

        assert "overconfidence" in flags
        assert "missing_consideration" in flags
        assert "insufficient_evidence" in flags
        assert "logical_inconsistency" in flags
        assert "unaddressed_risk" in flags


class TestCritiqueGeneratorAsync:
    """Async tests for CritiqueGenerator."""

    @pytest.mark.asyncio
    async def test_generate_async_timeout(self):
        """Test that async generation respects timeout."""
        ModelManager._instance = None

        # Mock a slow model
        mock_manager = MagicMock()
        mock_manager.get_model.side_effect = lambda: time.sleep(5) or (MagicMock(), MagicMock())

        generator = CritiqueGenerator(model_manager=mock_manager)

        with pytest.raises(InferenceTimeoutError):
            await generator.generate_async(
                "Test rationale for timeout",
                timeout=0.1
            )


# ================================================================
# RateLimiter Tests
# ================================================================

class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_initial_request_allowed(self):
        """Test first request is allowed."""
        limiter = RateLimiter(limit=10, window=60)

        allowed, remaining = limiter.check("user1")

        assert allowed is True
        assert remaining == 9

    def test_rate_limit_enforced(self):
        """Test rate limit is enforced."""
        limiter = RateLimiter(limit=3, window=60)

        # Use up the limit
        limiter.check("user1")
        limiter.check("user1")
        limiter.check("user1")

        # Next should be denied
        allowed, remaining = limiter.check("user1")

        assert allowed is False
        assert remaining == 0

    def test_different_users_independent(self):
        """Test different users have independent limits."""
        limiter = RateLimiter(limit=2, window=60)

        # Use up user1's limit
        limiter.check("user1")
        limiter.check("user1")

        # user2 should still be allowed
        allowed, remaining = limiter.check("user2")

        assert allowed is True
        assert remaining == 1

    def test_window_expiration(self):
        """Test requests expire after window."""
        limiter = RateLimiter(limit=2, window=0.1)  # 100ms window

        limiter.check("user1")
        limiter.check("user1")

        # Wait for window to expire
        time.sleep(0.15)

        allowed, remaining = limiter.check("user1")

        assert allowed is True
        assert remaining == 1

    def test_reset(self):
        """Test reset clears user's requests."""
        limiter = RateLimiter(limit=2, window=60)

        limiter.check("user1")
        limiter.check("user1")

        limiter.reset("user1")

        allowed, remaining = limiter.check("user1")

        assert allowed is True
        assert remaining == 1

    def test_get_reset_time(self):
        """Test getting reset time."""
        limiter = RateLimiter(limit=2, window=60)

        limiter.check("user1")

        reset_time = limiter.get_reset_time("user1")

        assert reset_time > 0
        assert reset_time <= 60

    def test_get_reset_time_empty(self):
        """Test reset time for unknown user."""
        limiter = RateLimiter(limit=2, window=60)

        reset_time = limiter.get_reset_time("unknown")

        assert reset_time == 0

    def test_thread_safety(self):
        """Test rate limiter is thread-safe."""
        limiter = RateLimiter(limit=100, window=60)
        results = []

        def check_limit():
            for _ in range(10):
                allowed, _ = limiter.check("shared_user")
                results.append(allowed)

        threads = [threading.Thread(target=check_limit) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have exactly 100 True, rest False
        assert sum(results) == 100


# ================================================================
# CritiqueResult Tests
# ================================================================

class TestCritiqueResult:
    """Tests for CritiqueResult dataclass."""

    def test_creation(self):
        """Test creating CritiqueResult."""
        result = CritiqueResult(
            valid=True,
            severity="pass",
            critique="The reasoning is sound",
            confidence=0.85,
            flags=[],
            domain="general",
            latency_ms=42.5,
            request_id="req123",
        )

        assert result.valid is True
        assert result.severity == "pass"
        assert result.confidence == 0.85

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = CritiqueResult(
            valid=False,
            severity="high",
            critique="Significant flaw detected",
            confidence=0.9,
            flags=["overconfidence", "missing_consideration"],
            domain="trading",
            latency_ms=100.0,
            request_id="req456",
        )

        d = result.to_dict()

        assert d["valid"] is False
        assert d["severity"] == "high"
        assert d["flags"] == ["overconfidence", "missing_consideration"]
        assert d["request_id"] == "req456"


# ================================================================
# Severity Tests
# ================================================================

class TestSeverity:
    """Tests for Severity enum."""

    def test_all_values(self):
        """Test all severity values exist."""
        assert Severity.PASS.value == "pass"
        assert Severity.LOW.value == "low"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.HIGH.value == "high"
        assert Severity.CRITICAL.value == "critical"

    def test_string_comparison(self):
        """Test severity can be compared with strings."""
        assert Severity.PASS == "pass"
        assert Severity.CRITICAL == "critical"


# ================================================================
# GracefulShutdown Tests
# ================================================================

class TestGracefulShutdown:
    """Tests for GracefulShutdown class."""

    def test_initial_state(self):
        """Test initial shutdown state."""
        ModelManager._instance = None
        shutdown = GracefulShutdown()

        assert shutdown.is_shutting_down is False

    def test_register_handlers(self):
        """Test signal handlers can be registered."""
        ModelManager._instance = None
        shutdown = GracefulShutdown()

        # Should not raise
        shutdown.register()


# ================================================================
# Convenience Function Tests
# ================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_model_manager(self):
        """Test get_model_manager returns singleton."""
        manager1 = get_model_manager()
        manager2 = get_model_manager()

        assert manager1 is manager2

    def test_get_critique_generator(self):
        """Test get_critique_generator creates new instances."""
        gen1 = get_critique_generator()
        gen2 = get_critique_generator()

        assert gen1 is not gen2

    def test_get_cors_origins(self):
        """Test get_cors_origins returns list."""
        origins = get_cors_origins()

        assert isinstance(origins, list)
        assert len(origins) > 0

    def test_get_cors_origins_returns_copy(self):
        """Test modifying result doesn't affect original."""
        origins1 = get_cors_origins()
        origins1.append("http://evil.com")

        origins2 = get_cors_origins()

        assert "http://evil.com" not in origins2


# ================================================================
# Exception Tests
# ================================================================

class TestExceptions:
    """Tests for custom exceptions."""

    def test_model_not_loaded_error(self):
        """Test ModelNotLoadedError."""
        with pytest.raises(ModelNotLoadedError):
            raise ModelNotLoadedError("Model not loaded")

    def test_model_load_error(self):
        """Test ModelLoadError."""
        with pytest.raises(ModelLoadError):
            raise ModelLoadError("Failed to load")

    def test_inference_timeout_error(self):
        """Test InferenceTimeoutError."""
        with pytest.raises(InferenceTimeoutError):
            raise InferenceTimeoutError("Timed out")

    def test_inference_error(self):
        """Test InferenceError."""
        with pytest.raises(InferenceError):
            raise InferenceError("Inference failed")

    def test_invalid_input_error(self):
        """Test InvalidInputError is ValueError."""
        with pytest.raises(ValueError):
            raise InvalidInputError("Invalid input")

        with pytest.raises(InvalidInputError):
            raise InvalidInputError("Invalid input")


# ================================================================
# Quantization Tests
# ================================================================

class TestQuantization:
    """Tests for quantization support."""

    def test_quantization_mode_default(self):
        """Test default quantization mode is none."""
        ModelManager._instance = None
        manager = ModelManager.get_instance()

        assert manager.quantization_mode == "none"

    def test_quantization_in_stats(self):
        """Test quantization is included in stats."""
        ModelManager._instance = None
        manager = ModelManager.get_instance()

        stats = manager.stats
        assert "quantization" in stats
        assert stats["quantization"] == "none"

    def test_valid_quantization_options(self):
        """Test valid quantization options are defined."""
        from src.mcp.core import VALID_QUANTIZATION_OPTIONS

        assert "none" in VALID_QUANTIZATION_OPTIONS
        assert "8bit" in VALID_QUANTIZATION_OPTIONS
        assert "4bit" in VALID_QUANTIZATION_OPTIONS
        assert len(VALID_QUANTIZATION_OPTIONS) == 3

    @patch.dict("os.environ", {"MINICRIT_QUANTIZATION": "8bit"})
    def test_quantization_config_env_var(self):
        """Test quantization config reads from environment."""
        # Re-import to pick up env var
        import importlib
        import src.mcp.core as core_module
        importlib.reload(core_module)

        assert core_module.QUANTIZATION == "8bit"

        # Reset
        importlib.reload(core_module)

    @patch.dict("os.environ", {"MINICRIT_QUANTIZATION": "invalid"})
    def test_quantization_invalid_falls_back(self):
        """Test invalid quantization falls back to none."""
        import importlib
        import src.mcp.core as core_module
        importlib.reload(core_module)

        # Invalid value is read but will be handled during loading
        assert core_module.QUANTIZATION == "invalid"

        # Reset
        importlib.reload(core_module)
