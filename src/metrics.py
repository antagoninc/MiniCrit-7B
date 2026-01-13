"""Prometheus metrics for MiniCrit.

This module provides Prometheus-compatible metrics for monitoring
the MiniCrit API server in production environments.

Example:
    >>> from src.metrics import metrics, track_request
    >>>
    >>> # In API endpoint
    >>> with track_request("critique"):
    ...     result = generate_critique(rationale)
    >>>
    >>> # Get metrics
    >>> metrics.get_prometheus_metrics()

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class HistogramBucket:
    """Histogram bucket for latency tracking."""

    le: float  # Less than or equal
    count: int = 0


@dataclass
class MetricsState:
    """Thread-safe metrics state container."""

    # Counters
    requests_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    requests_failed: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tokens_generated: int = 0

    # Gauges
    model_loaded: int = 0
    active_requests: int = 0

    # Histograms (latency in seconds)
    request_latency_sum: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    request_latency_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    request_latency_buckets: dict[str, list[HistogramBucket]] = field(default_factory=dict)

    # Info
    start_time: float = field(default_factory=time.time)

    _lock: threading.Lock = field(default_factory=threading.Lock)


class PrometheusMetrics:
    """Prometheus metrics collector for MiniCrit.

    Provides thread-safe metrics collection compatible with Prometheus
    text format exposition.

    Attributes:
        state: The current metrics state.
    """

    # Standard histogram buckets for request latency (in seconds)
    DEFAULT_BUCKETS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self.state = MetricsState()
        self._init_buckets()

    def _init_buckets(self) -> None:
        """Initialize histogram buckets for all endpoints."""
        endpoints = ["critique", "batch", "health", "stats", "load", "validate"]
        for endpoint in endpoints:
            self.state.request_latency_buckets[endpoint] = [
                HistogramBucket(le=b) for b in self.DEFAULT_BUCKETS
            ]
            self.state.request_latency_buckets[endpoint].append(HistogramBucket(le=float("inf")))

    def inc_requests(self, endpoint: str, status: str = "success") -> None:
        """Increment request counter.

        Args:
            endpoint: The API endpoint name.
            status: Request status ('success' or 'error').
        """
        with self.state._lock:
            key = f"{endpoint}:{status}"
            self.state.requests_total[key] += 1
            if status == "error":
                self.state.requests_failed[endpoint] += 1

    def observe_latency(self, endpoint: str, latency_seconds: float) -> None:
        """Record request latency.

        Args:
            endpoint: The API endpoint name.
            latency_seconds: Request latency in seconds.
        """
        with self.state._lock:
            self.state.request_latency_sum[endpoint] += latency_seconds
            self.state.request_latency_count[endpoint] += 1

            # Update histogram buckets
            if endpoint in self.state.request_latency_buckets:
                for bucket in self.state.request_latency_buckets[endpoint]:
                    if latency_seconds <= bucket.le:
                        bucket.count += 1

    def add_tokens(self, count: int) -> None:
        """Add to tokens generated counter.

        Args:
            count: Number of tokens generated.
        """
        with self.state._lock:
            self.state.tokens_generated += count

    def set_model_loaded(self, loaded: bool) -> None:
        """Set model loaded gauge.

        Args:
            loaded: Whether model is loaded.
        """
        with self.state._lock:
            self.state.model_loaded = 1 if loaded else 0

    def inc_active_requests(self) -> None:
        """Increment active requests gauge."""
        with self.state._lock:
            self.state.active_requests += 1

    def dec_active_requests(self) -> None:
        """Decrement active requests gauge."""
        with self.state._lock:
            self.state.active_requests = max(0, self.state.active_requests - 1)

    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus text format metrics.

        Returns:
            Prometheus text format metrics string.
        """
        lines: list[str] = []

        with self.state._lock:
            # Help and type declarations
            lines.append("# HELP minicrit_requests_total Total number of requests")
            lines.append("# TYPE minicrit_requests_total counter")
            for key, count in self.state.requests_total.items():
                endpoint, status = key.split(":")
                lines.append(
                    f'minicrit_requests_total{{endpoint="{endpoint}",status="{status}"}} {count}'
                )

            lines.append("")
            lines.append("# HELP minicrit_requests_failed_total Total number of failed requests")
            lines.append("# TYPE minicrit_requests_failed_total counter")
            for endpoint, count in self.state.requests_failed.items():
                lines.append(f'minicrit_requests_failed_total{{endpoint="{endpoint}"}} {count}')

            lines.append("")
            lines.append("# HELP minicrit_tokens_generated_total Total tokens generated")
            lines.append("# TYPE minicrit_tokens_generated_total counter")
            lines.append(f"minicrit_tokens_generated_total {self.state.tokens_generated}")

            lines.append("")
            lines.append("# HELP minicrit_model_loaded Whether the model is loaded")
            lines.append("# TYPE minicrit_model_loaded gauge")
            lines.append(f"minicrit_model_loaded {self.state.model_loaded}")

            lines.append("")
            lines.append("# HELP minicrit_active_requests Current number of active requests")
            lines.append("# TYPE minicrit_active_requests gauge")
            lines.append(f"minicrit_active_requests {self.state.active_requests}")

            lines.append("")
            lines.append("# HELP minicrit_uptime_seconds Server uptime in seconds")
            lines.append("# TYPE minicrit_uptime_seconds gauge")
            uptime = time.time() - self.state.start_time
            lines.append(f"minicrit_uptime_seconds {uptime:.3f}")

            # Request latency histogram
            lines.append("")
            lines.append("# HELP minicrit_request_latency_seconds Request latency in seconds")
            lines.append("# TYPE minicrit_request_latency_seconds histogram")
            for endpoint, buckets in self.state.request_latency_buckets.items():
                for bucket in buckets:
                    le_str = "+Inf" if bucket.le == float("inf") else f"{bucket.le}"
                    lines.append(
                        f'minicrit_request_latency_seconds_bucket{{endpoint="{endpoint}",le="{le_str}"}} {bucket.count}'
                    )
                lines.append(
                    f'minicrit_request_latency_seconds_sum{{endpoint="{endpoint}"}} {self.state.request_latency_sum[endpoint]:.6f}'
                )
                lines.append(
                    f'minicrit_request_latency_seconds_count{{endpoint="{endpoint}"}} {self.state.request_latency_count[endpoint]}'
                )

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        self.state = MetricsState()
        self._init_buckets()


# Global metrics instance
metrics = PrometheusMetrics()


@contextmanager
def track_request(endpoint: str) -> Generator[None, None, None]:
    """Context manager for tracking request metrics.

    Automatically tracks request count, latency, and active requests.

    Args:
        endpoint: The API endpoint name.

    Yields:
        None

    Example:
        >>> with track_request("critique"):
        ...     result = process_request()
    """
    metrics.inc_active_requests()
    start_time = time.perf_counter()
    status = "success"

    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        latency = time.perf_counter() - start_time
        metrics.dec_active_requests()
        metrics.inc_requests(endpoint, status)
        metrics.observe_latency(endpoint, latency)
