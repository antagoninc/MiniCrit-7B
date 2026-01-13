"""Tests for distributed rate limiting module.

Antagon Inc. | CAGE: 17E75
"""

import threading
import time
import pytest
from unittest.mock import patch, MagicMock

from src.rate_limiting import (
    InMemoryRateLimiter,
    RedisRateLimiter,
    FallbackRateLimiter,
    RateLimitExceeded,
    get_rate_limiter,
    reset_rate_limiter,
)


class TestInMemoryRateLimiter:
    """Tests for in-memory rate limiter."""

    def test_allows_within_limit(self):
        """Test that requests within limit are allowed."""
        limiter = InMemoryRateLimiter(limit=5, window=60)

        for i in range(5):
            allowed, remaining = limiter.check("test_key")
            assert allowed is True
            assert remaining == 4 - i

    def test_blocks_over_limit(self):
        """Test that requests over limit are blocked."""
        limiter = InMemoryRateLimiter(limit=3, window=60)

        # Use up the limit
        for _ in range(3):
            limiter.check("test_key")

        # Next request should be blocked
        allowed, remaining = limiter.check("test_key")
        assert allowed is False
        assert remaining == 0

    def test_different_keys_independent(self):
        """Test that different keys have independent limits."""
        limiter = InMemoryRateLimiter(limit=2, window=60)

        # Use up key1's limit
        limiter.check("key1")
        limiter.check("key1")

        # key2 should still work
        allowed, remaining = limiter.check("key2")
        assert allowed is True
        assert remaining == 1

    def test_sliding_window(self):
        """Test that old requests expire from window."""
        limiter = InMemoryRateLimiter(limit=2, window=1)  # 1 second window

        # Use up the limit
        limiter.check("test_key")
        limiter.check("test_key")

        # Should be blocked
        allowed, _ = limiter.check("test_key")
        assert allowed is False

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        allowed, _ = limiter.check("test_key")
        assert allowed is True

    def test_reset(self):
        """Test resetting a key's rate limit."""
        limiter = InMemoryRateLimiter(limit=2, window=60)

        # Use up the limit
        limiter.check("test_key")
        limiter.check("test_key")

        # Reset the key
        limiter.reset("test_key")

        # Should be allowed again
        allowed, remaining = limiter.check("test_key")
        assert allowed is True
        assert remaining == 1

    def test_get_reset_time(self):
        """Test getting reset time."""
        limiter = InMemoryRateLimiter(limit=2, window=60)

        # No requests yet
        assert limiter.get_reset_time("test_key") == 0

        # Make a request
        limiter.check("test_key")

        # Should have ~60 seconds until reset
        reset_time = limiter.get_reset_time("test_key")
        assert 59 < reset_time <= 60

    def test_get_stats(self):
        """Test getting rate limiter stats."""
        limiter = InMemoryRateLimiter(limit=10, window=30)

        limiter.check("key1")
        limiter.check("key2")

        stats = limiter.get_stats()
        assert stats["backend"] == "memory"
        assert stats["active_keys"] == 2
        assert stats["limit"] == 10
        assert stats["window"] == 30

    def test_thread_safety(self):
        """Test thread safety of rate limiter."""
        limiter = InMemoryRateLimiter(limit=100, window=60)
        results = []

        def make_requests():
            for _ in range(20):
                allowed, _ = limiter.check("shared_key")
                results.append(allowed)

        threads = [threading.Thread(target=make_requests) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 100 should be allowed
        assert sum(results) == 100
        assert len(results) == 100

    def test_check_or_raise_allowed(self):
        """Test check_or_raise when allowed."""
        limiter = InMemoryRateLimiter(limit=5, window=60)

        remaining = limiter.check_or_raise("test_key")
        assert remaining == 4

    def test_check_or_raise_blocked(self):
        """Test check_or_raise when blocked."""
        limiter = InMemoryRateLimiter(limit=1, window=60)

        # Use up the limit
        limiter.check("test_key")

        # Should raise
        with pytest.raises(RateLimitExceeded) as exc_info:
            limiter.check_or_raise("test_key")

        assert exc_info.value.key == "test_key"
        assert exc_info.value.retry_after > 0


class TestRedisRateLimiter:
    """Tests for Redis rate limiter."""

    def test_fallback_when_redis_unavailable(self):
        """Test that limiter allows requests when Redis is unavailable."""
        # Mock redis import to simulate unavailable Redis
        with patch.dict("sys.modules", {"redis": None}):
            limiter = RedisRateLimiter(limit=5, window=60)

            # Should still allow (fail open)
            allowed, remaining = limiter.check("test_key")
            assert allowed is True

    def test_is_connected_false_when_no_redis(self):
        """Test is_connected property when Redis unavailable."""
        with patch.dict("sys.modules", {"redis": None}):
            limiter = RedisRateLimiter(limit=5, window=60)
            assert limiter.is_connected is False

    def test_get_stats_without_redis(self):
        """Test stats when Redis unavailable."""
        with patch.dict("sys.modules", {"redis": None}):
            limiter = RedisRateLimiter(limit=5, window=60)

            stats = limiter.get_stats()
            assert stats["backend"] == "redis"
            assert stats["connected"] is False

    def test_check_with_mocked_redis(self):
        """Test check with mocked Redis."""
        # Setup mock
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        # Mock pipeline
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        mock_pipe.execute.return_value = [None, 0]  # zremrangebyscore result, current count

        mock_redis_module = MagicMock()
        mock_redis_module.from_url.return_value = mock_redis

        with patch.dict("sys.modules", {"redis": mock_redis_module}):
            # Need to reimport to pick up the mock
            from importlib import reload
            import src.rate_limiting as rl
            reload(rl)

            limiter = rl.RedisRateLimiter(limit=5, window=60, redis_url="redis://localhost:6379")

            allowed, remaining = limiter.check("test_key")
            assert allowed is True

    def test_reset_with_mocked_redis(self):
        """Test reset with mocked Redis."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        mock_redis_module = MagicMock()
        mock_redis_module.from_url.return_value = mock_redis

        with patch.dict("sys.modules", {"redis": mock_redis_module}):
            from importlib import reload
            import src.rate_limiting as rl
            reload(rl)

            limiter = rl.RedisRateLimiter(limit=5, window=60)
            limiter.reset("test_key")

            mock_redis.delete.assert_called_once()


class TestFallbackRateLimiter:
    """Tests for fallback rate limiter."""

    def test_uses_memory_when_redis_unavailable(self):
        """Test that fallback uses memory when Redis unavailable."""
        with patch.dict("sys.modules", {"redis": None}):
            limiter = FallbackRateLimiter(limit=5, window=60)

            assert limiter.backend == "memory"

            # Should work with in-memory
            allowed, remaining = limiter.check("test_key")
            assert allowed is True
            assert remaining == 4

    def test_get_stats_shows_fallback(self):
        """Test that stats indicate fallback availability."""
        with patch.dict("sys.modules", {"redis": None}):
            limiter = FallbackRateLimiter(limit=5, window=60)

            stats = limiter.get_stats()
            assert stats["fallback_available"] is True
            assert stats["backend"] == "memory"


class TestGetRateLimiter:
    """Tests for get_rate_limiter factory function."""

    def setup_method(self):
        """Reset rate limiter before each test."""
        reset_rate_limiter()

    def teardown_method(self):
        """Reset rate limiter after each test."""
        reset_rate_limiter()

    def test_returns_memory_by_default(self):
        """Test that memory backend is returned by default."""
        with patch.dict("os.environ", {"MINICRIT_RATE_LIMIT_BACKEND": "memory"}):
            reset_rate_limiter()
            limiter = get_rate_limiter()
            # Use class name check to avoid reload issues
            assert limiter.__class__.__name__ == "InMemoryRateLimiter"

    def test_returns_same_instance(self):
        """Test that same instance is returned on subsequent calls."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is limiter2

    def test_respects_env_vars(self):
        """Test that environment variables are respected."""
        with patch.dict("os.environ", {
            "MINICRIT_RATE_LIMIT": "100",
            "MINICRIT_RATE_WINDOW": "120",
            "MINICRIT_RATE_LIMIT_BACKEND": "memory",
        }):
            reset_rate_limiter()
            # Need to reload module to pick up new env vars
            from src import rate_limiting
            rate_limiting.RATE_LIMIT = 100
            rate_limiting.RATE_WINDOW = 120
            rate_limiting._rate_limiter = None

            limiter = get_rate_limiter()
            assert limiter.limit == 100
            assert limiter.window == 120

    def test_fallback_backend(self):
        """Test fallback backend creation."""
        reset_rate_limiter()
        limiter = get_rate_limiter(backend="fallback")
        # Use class name check to avoid reload issues
        assert limiter.__class__.__name__ == "FallbackRateLimiter"


class TestRateLimitExceeded:
    """Tests for RateLimitExceeded exception."""

    def test_exception_attributes(self):
        """Test exception has correct attributes."""
        exc = RateLimitExceeded("test_key", retry_after=30.5)

        assert exc.key == "test_key"
        assert exc.retry_after == 30.5
        assert "test_key" in str(exc)
        assert "30.5" in str(exc)

    def test_exception_message(self):
        """Test exception message format."""
        exc = RateLimitExceeded("api_key_123", retry_after=45.0)

        assert "Rate limit exceeded" in str(exc)
        assert "api_key_123" in str(exc)
