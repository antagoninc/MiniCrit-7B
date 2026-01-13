"""Distributed rate limiting for MiniCrit.

Provides both in-memory and Redis-backed rate limiting implementations
for single-instance and multi-instance deployments.

Example:
    >>> # In-memory (single instance)
    >>> limiter = get_rate_limiter()
    >>> allowed, remaining = limiter.check("user_123")

    >>> # Redis (multi-instance)
    >>> limiter = get_rate_limiter(backend="redis", redis_url="redis://localhost:6379")
    >>> allowed, remaining = limiter.check("user_123")

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

from __future__ import annotations

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

# Environment configuration
RATE_LIMIT = int(os.environ.get("MINICRIT_RATE_LIMIT", "60"))
RATE_WINDOW = int(os.environ.get("MINICRIT_RATE_WINDOW", "60"))
REDIS_URL = os.environ.get("MINICRIT_REDIS_URL", "")
RATE_LIMIT_BACKEND = os.environ.get("MINICRIT_RATE_LIMIT_BACKEND", "memory")


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, key: str, retry_after: float = 0):
        self.key = key
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded for {key}. Retry after {retry_after:.1f}s")


class BaseRateLimiter(ABC):
    """Abstract base class for rate limiters."""

    def __init__(self, limit: int = 60, window: int = 60):
        """Initialize rate limiter.

        Args:
            limit: Maximum requests per window.
            window: Window size in seconds.
        """
        self.limit = limit
        self.window = window

    @abstractmethod
    def check(self, key: str) -> tuple[bool, int]:
        """Check if request is allowed.

        Args:
            key: Unique identifier (e.g., API key hash, IP).

        Returns:
            Tuple of (allowed, remaining_requests).
        """
        pass

    @abstractmethod
    def reset(self, key: str) -> None:
        """Reset rate limit for a key.

        Args:
            key: Unique identifier to reset.
        """
        pass

    @abstractmethod
    def get_reset_time(self, key: str) -> float:
        """Get time until rate limit resets.

        Args:
            key: Unique identifier.

        Returns:
            Seconds until reset, or 0 if not limited.
        """
        pass

    def check_or_raise(self, key: str) -> int:
        """Check rate limit and raise if exceeded.

        Args:
            key: Unique identifier.

        Returns:
            Remaining requests.

        Raises:
            RateLimitExceeded: If rate limit is exceeded.
        """
        allowed, remaining = self.check(key)
        if not allowed:
            retry_after = self.get_reset_time(key)
            raise RateLimitExceeded(key, retry_after)
        return remaining


class InMemoryRateLimiter(BaseRateLimiter):
    """Thread-safe in-memory sliding window rate limiter.

    Suitable for single-instance deployments. Uses a sliding window
    algorithm for smooth rate limiting.
    """

    def __init__(self, limit: int = 60, window: int = 60):
        super().__init__(limit, window)
        self._requests: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """Check if request is allowed using sliding window."""
        with self._lock:
            now = time.time()
            window_start = now - self.window

            # Initialize if needed
            if key not in self._requests:
                self._requests[key] = []

            # Clean old requests
            self._requests[key] = [
                t for t in self._requests[key] if t > window_start
            ]

            # Check limit
            if len(self._requests[key]) >= self.limit:
                return False, 0

            # Record request
            self._requests[key].append(now)
            remaining = self.limit - len(self._requests[key])

            return True, remaining

    def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        with self._lock:
            if key in self._requests:
                del self._requests[key]

    def get_reset_time(self, key: str) -> float:
        """Get time until oldest request expires from window."""
        with self._lock:
            if key not in self._requests or not self._requests[key]:
                return 0

            oldest = min(self._requests[key])
            reset_time = oldest + self.window - time.time()
            return max(0, reset_time)

    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "backend": "memory",
                "active_keys": len(self._requests),
                "limit": self.limit,
                "window": self.window,
            }


class RedisRateLimiter(BaseRateLimiter):
    """Redis-backed distributed rate limiter.

    Suitable for multi-instance deployments. Uses Redis sorted sets
    for sliding window rate limiting with atomic operations.

    Requires redis-py: pip install redis
    """

    def __init__(
        self,
        limit: int = 60,
        window: int = 60,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "minicrit:ratelimit:",
    ):
        super().__init__(limit, window)
        self.key_prefix = key_prefix
        self._redis: Optional[object] = None
        self._redis_url = redis_url
        self._connect()

    def _connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis
            self._redis = redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            self._redis.ping()
            logger.info(f"Connected to Redis at {self._redis_url}")
        except ImportError:
            logger.warning("redis-py not installed. Install with: pip install redis")
            self._redis = None
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Falling back to memory.")
            self._redis = None

    def _get_key(self, key: str) -> str:
        """Get Redis key with prefix."""
        return f"{self.key_prefix}{key}"

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if self._redis is None:
            return False
        try:
            self._redis.ping()
            return True
        except Exception:
            return False

    def check(self, key: str) -> tuple[bool, int]:
        """Check if request is allowed using Redis sorted set."""
        if not self.is_connected:
            logger.warning("Redis not available, allowing request")
            return True, self.limit - 1

        redis_key = self._get_key(key)
        now = time.time()
        window_start = now - self.window

        try:
            # Use pipeline for atomic operations
            pipe = self._redis.pipeline()

            # Remove expired entries
            pipe.zremrangebyscore(redis_key, 0, window_start)

            # Count current requests in window
            pipe.zcard(redis_key)

            # Execute pipeline
            results = pipe.execute()
            current_count = results[1]

            # Check if over limit
            if current_count >= self.limit:
                return False, 0

            # Add new request with score = timestamp
            pipe = self._redis.pipeline()
            pipe.zadd(redis_key, {f"{now}:{id(self)}": now})
            pipe.expire(redis_key, self.window + 1)
            pipe.execute()

            remaining = self.limit - current_count - 1
            return True, remaining

        except Exception as e:
            logger.error(f"Redis error in rate limiter: {e}")
            # Fail open - allow request if Redis fails
            return True, self.limit - 1

    def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        if not self.is_connected:
            return

        try:
            redis_key = self._get_key(key)
            self._redis.delete(redis_key)
        except Exception as e:
            logger.error(f"Redis error resetting key: {e}")

    def get_reset_time(self, key: str) -> float:
        """Get time until rate limit resets."""
        if not self.is_connected:
            return 0

        try:
            redis_key = self._get_key(key)
            # Get oldest entry
            oldest = self._redis.zrange(redis_key, 0, 0, withscores=True)
            if not oldest:
                return 0

            oldest_time = oldest[0][1]
            reset_time = oldest_time + self.window - time.time()
            return max(0, reset_time)

        except Exception as e:
            logger.error(f"Redis error getting reset time: {e}")
            return 0

    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        stats = {
            "backend": "redis",
            "connected": self.is_connected,
            "limit": self.limit,
            "window": self.window,
            "redis_url": self._redis_url.split("@")[-1] if "@" in self._redis_url else self._redis_url,
        }

        if self.is_connected:
            try:
                # Count active keys
                cursor = 0
                count = 0
                pattern = f"{self.key_prefix}*"
                while True:
                    cursor, keys = self._redis.scan(cursor, match=pattern, count=100)
                    count += len(keys)
                    if cursor == 0:
                        break
                stats["active_keys"] = count
            except Exception:
                stats["active_keys"] = -1

        return stats


class FallbackRateLimiter(BaseRateLimiter):
    """Rate limiter with Redis primary and in-memory fallback.

    Automatically falls back to in-memory if Redis is unavailable.
    """

    def __init__(
        self,
        limit: int = 60,
        window: int = 60,
        redis_url: str = "redis://localhost:6379",
    ):
        super().__init__(limit, window)
        self._redis_limiter = RedisRateLimiter(limit, window, redis_url)
        self._memory_limiter = InMemoryRateLimiter(limit, window)

    @property
    def _active_limiter(self) -> BaseRateLimiter:
        """Get the active rate limiter."""
        if self._redis_limiter.is_connected:
            return self._redis_limiter
        return self._memory_limiter

    @property
    def backend(self) -> str:
        """Get the active backend name."""
        return "redis" if self._redis_limiter.is_connected else "memory"

    def check(self, key: str) -> tuple[bool, int]:
        return self._active_limiter.check(key)

    def reset(self, key: str) -> None:
        self._active_limiter.reset(key)

    def get_reset_time(self, key: str) -> float:
        return self._active_limiter.get_reset_time(key)

    def get_stats(self) -> dict:
        stats = self._active_limiter.get_stats()
        stats["fallback_available"] = True
        return stats


# Global rate limiter instance
_rate_limiter: Optional[BaseRateLimiter] = None
_limiter_lock = threading.Lock()


def get_rate_limiter(
    backend: Optional[str] = None,
    redis_url: Optional[str] = None,
    limit: Optional[int] = None,
    window: Optional[int] = None,
) -> BaseRateLimiter:
    """Get or create the global rate limiter instance.

    Args:
        backend: Rate limiting backend ("memory", "redis", "fallback").
                 Defaults to MINICRIT_RATE_LIMIT_BACKEND env var or "memory".
        redis_url: Redis URL for redis/fallback backends.
                   Defaults to MINICRIT_REDIS_URL env var.
        limit: Maximum requests per window. Defaults to MINICRIT_RATE_LIMIT.
        window: Window size in seconds. Defaults to MINICRIT_RATE_WINDOW.

    Returns:
        Rate limiter instance.
    """
    global _rate_limiter

    with _limiter_lock:
        if _rate_limiter is None:
            _backend = backend or RATE_LIMIT_BACKEND
            _redis_url = redis_url or REDIS_URL
            _limit = limit or RATE_LIMIT
            _window = window or RATE_WINDOW

            if _backend == "redis":
                _rate_limiter = RedisRateLimiter(_limit, _window, _redis_url)
            elif _backend == "fallback":
                _rate_limiter = FallbackRateLimiter(_limit, _window, _redis_url)
            else:
                _rate_limiter = InMemoryRateLimiter(_limit, _window)

            logger.info(f"Rate limiter initialized: backend={_backend}, limit={_limit}/{_window}s")

        return _rate_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter instance."""
    global _rate_limiter
    with _limiter_lock:
        _rate_limiter = None
