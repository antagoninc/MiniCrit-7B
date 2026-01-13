"""Tests for the Prometheus metrics module.

Antagon Inc. | CAGE: 17E75
"""

import threading
import time

import pytest

from src.metrics import PrometheusMetrics, metrics, track_request


class TestPrometheusMetrics:
    """Tests for the PrometheusMetrics class."""

    def setup_method(self):
        """Reset metrics before each test."""
        self.metrics = PrometheusMetrics()

    def test_init_creates_empty_state(self):
        """Test that initialization creates empty metrics state."""
        assert self.metrics.state.tokens_generated == 0
        assert self.metrics.state.model_loaded == 0
        assert self.metrics.state.active_requests == 0

    def test_inc_requests_success(self):
        """Test incrementing successful request counter."""
        self.metrics.inc_requests("critique", "success")
        assert self.metrics.state.requests_total["critique:success"] == 1

        self.metrics.inc_requests("critique", "success")
        assert self.metrics.state.requests_total["critique:success"] == 2

    def test_inc_requests_error(self):
        """Test incrementing failed request counter."""
        self.metrics.inc_requests("critique", "error")
        assert self.metrics.state.requests_total["critique:error"] == 1
        assert self.metrics.state.requests_failed["critique"] == 1

    def test_observe_latency(self):
        """Test recording request latency."""
        self.metrics.observe_latency("critique", 0.5)
        assert self.metrics.state.request_latency_sum["critique"] == 0.5
        assert self.metrics.state.request_latency_count["critique"] == 1

        self.metrics.observe_latency("critique", 0.3)
        assert self.metrics.state.request_latency_sum["critique"] == 0.8
        assert self.metrics.state.request_latency_count["critique"] == 2

    def test_observe_latency_updates_buckets(self):
        """Test that latency observation updates histogram buckets."""
        self.metrics.observe_latency("critique", 0.05)

        buckets = self.metrics.state.request_latency_buckets["critique"]
        # All buckets >= 0.05 should have count of 1
        for bucket in buckets:
            if bucket.le >= 0.05:
                assert bucket.count >= 1

    def test_add_tokens(self):
        """Test adding to tokens counter."""
        self.metrics.add_tokens(100)
        assert self.metrics.state.tokens_generated == 100

        self.metrics.add_tokens(50)
        assert self.metrics.state.tokens_generated == 150

    def test_set_model_loaded(self):
        """Test setting model loaded gauge."""
        assert self.metrics.state.model_loaded == 0

        self.metrics.set_model_loaded(True)
        assert self.metrics.state.model_loaded == 1

        self.metrics.set_model_loaded(False)
        assert self.metrics.state.model_loaded == 0

    def test_inc_dec_active_requests(self):
        """Test incrementing and decrementing active requests gauge."""
        assert self.metrics.state.active_requests == 0

        self.metrics.inc_active_requests()
        assert self.metrics.state.active_requests == 1

        self.metrics.inc_active_requests()
        assert self.metrics.state.active_requests == 2

        self.metrics.dec_active_requests()
        assert self.metrics.state.active_requests == 1

    def test_dec_active_requests_never_negative(self):
        """Test that active requests gauge never goes negative."""
        self.metrics.dec_active_requests()
        assert self.metrics.state.active_requests == 0

    def test_get_prometheus_metrics_format(self):
        """Test that metrics output is valid Prometheus format."""
        self.metrics.inc_requests("critique", "success")
        self.metrics.add_tokens(100)
        self.metrics.set_model_loaded(True)
        self.metrics.observe_latency("critique", 0.1)

        output = self.metrics.get_prometheus_metrics()

        # Check for expected metric names
        assert "minicrit_requests_total" in output
        assert "minicrit_tokens_generated_total" in output
        assert "minicrit_model_loaded" in output
        assert "minicrit_uptime_seconds" in output
        assert "minicrit_request_latency_seconds" in output

        # Check for HELP and TYPE declarations
        assert "# HELP" in output
        assert "# TYPE" in output

    def test_get_prometheus_metrics_labels(self):
        """Test that metrics include proper labels."""
        self.metrics.inc_requests("critique", "success")
        self.metrics.inc_requests("batch", "error")

        output = self.metrics.get_prometheus_metrics()

        assert 'endpoint="critique"' in output
        assert 'endpoint="batch"' in output
        assert 'status="success"' in output
        assert 'status="error"' in output

    def test_reset(self):
        """Test resetting all metrics."""
        self.metrics.inc_requests("critique", "success")
        self.metrics.add_tokens(100)
        self.metrics.set_model_loaded(True)

        self.metrics.reset()

        assert self.metrics.state.tokens_generated == 0
        assert self.metrics.state.model_loaded == 0
        assert len(self.metrics.state.requests_total) == 0

    def test_thread_safety(self):
        """Test that metrics operations are thread-safe."""
        iterations = 1000

        def increment_requests():
            for _ in range(iterations):
                self.metrics.inc_requests("critique", "success")

        def add_tokens():
            for _ in range(iterations):
                self.metrics.add_tokens(1)

        threads = [
            threading.Thread(target=increment_requests),
            threading.Thread(target=increment_requests),
            threading.Thread(target=add_tokens),
            threading.Thread(target=add_tokens),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert self.metrics.state.requests_total["critique:success"] == iterations * 2
        assert self.metrics.state.tokens_generated == iterations * 2


class TestTrackRequest:
    """Tests for the track_request context manager."""

    def setup_method(self):
        """Reset global metrics before each test."""
        metrics.reset()

    def test_track_request_success(self):
        """Test tracking a successful request."""
        with track_request("critique"):
            time.sleep(0.01)

        assert metrics.state.requests_total["critique:success"] == 1
        assert metrics.state.request_latency_count["critique"] == 1
        assert metrics.state.request_latency_sum["critique"] > 0

    def test_track_request_error(self):
        """Test tracking a failed request."""
        with pytest.raises(ValueError):
            with track_request("critique"):
                raise ValueError("test error")

        assert metrics.state.requests_total["critique:error"] == 1
        assert metrics.state.requests_failed["critique"] == 1

    def test_track_request_active_count(self):
        """Test that active requests are tracked during execution."""
        initial_active = metrics.state.active_requests

        with track_request("critique"):
            # During execution, active should be incremented
            assert metrics.state.active_requests == initial_active + 1

        # After execution, active should be back to initial
        assert metrics.state.active_requests == initial_active

    def test_track_request_active_count_on_error(self):
        """Test that active requests are decremented even on error."""
        initial_active = metrics.state.active_requests

        try:
            with track_request("critique"):
                raise RuntimeError("test")
        except RuntimeError:
            pass

        assert metrics.state.active_requests == initial_active


class TestHistogramBuckets:
    """Tests for histogram bucket behavior."""

    def setup_method(self):
        """Reset metrics before each test."""
        self.metrics = PrometheusMetrics()

    def test_default_buckets_initialized(self):
        """Test that default histogram buckets are initialized."""
        buckets = self.metrics.state.request_latency_buckets.get("critique", [])
        assert len(buckets) > 0

        # Check that +Inf bucket exists
        inf_buckets = [b for b in buckets if b.le == float("inf")]
        assert len(inf_buckets) == 1

    def test_latency_fills_correct_buckets(self):
        """Test that latency observations fill correct buckets."""
        # Observe latency of 0.5 seconds
        self.metrics.observe_latency("critique", 0.5)

        buckets = self.metrics.state.request_latency_buckets["critique"]

        # Buckets with le >= 0.5 should have count of 1
        for bucket in buckets:
            if bucket.le >= 0.5:
                assert bucket.count == 1, f"Bucket le={bucket.le} should have count 1"
            else:
                assert bucket.count == 0, f"Bucket le={bucket.le} should have count 0"

    def test_histogram_output_format(self):
        """Test histogram output in Prometheus format."""
        self.metrics.observe_latency("critique", 0.1)
        output = self.metrics.get_prometheus_metrics()

        # Check for histogram-specific output
        assert "minicrit_request_latency_seconds_bucket" in output
        assert "minicrit_request_latency_seconds_sum" in output
        assert "minicrit_request_latency_seconds_count" in output
        assert 'le="+Inf"' in output


class TestMetricsIntegration:
    """Integration tests for metrics with the global instance."""

    def setup_method(self):
        """Reset global metrics before each test."""
        metrics.reset()

    def test_global_metrics_instance(self):
        """Test that global metrics instance works correctly."""
        metrics.inc_requests("test", "success")
        metrics.add_tokens(50)

        assert metrics.state.requests_total["test:success"] == 1
        assert metrics.state.tokens_generated == 50

    def test_multiple_endpoints_tracked(self):
        """Test tracking multiple different endpoints."""
        endpoints = ["critique", "batch", "health", "stats"]

        for endpoint in endpoints:
            metrics.inc_requests(endpoint, "success")
            metrics.observe_latency(endpoint, 0.1)

        output = metrics.get_prometheus_metrics()

        for endpoint in endpoints:
            assert f'endpoint="{endpoint}"' in output
