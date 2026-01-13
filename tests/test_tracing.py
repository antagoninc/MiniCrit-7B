"""Tests for OpenTelemetry tracing module.

Antagon Inc. | CAGE: 17E75
"""

import pytest
from unittest.mock import patch, MagicMock

from src.tracing import (
    NoOpSpan,
    NoOpTracer,
    init_tracing,
    get_tracer,
    reset_tracing,
    trace_span,
    trace_function,
    trace_async_function,
    add_span_attributes,
    record_exception,
    SpanAttributes,
)


class TestNoOpSpan:
    """Tests for NoOpSpan class."""

    def test_context_manager(self):
        """Test span works as context manager."""
        span = NoOpSpan()
        with span as s:
            assert s is span

    def test_set_attribute(self):
        """Test set_attribute is no-op."""
        span = NoOpSpan()
        # Should not raise
        span.set_attribute("key", "value")

    def test_set_status(self):
        """Test set_status is no-op."""
        span = NoOpSpan()
        span.set_status("OK")

    def test_record_exception(self):
        """Test record_exception is no-op."""
        span = NoOpSpan()
        span.record_exception(ValueError("test"))

    def test_add_event(self):
        """Test add_event is no-op."""
        span = NoOpSpan()
        span.add_event("test_event", {"key": "value"})

    def test_is_recording(self):
        """Test is_recording returns False."""
        span = NoOpSpan()
        assert span.is_recording() is False


class TestNoOpTracer:
    """Tests for NoOpTracer class."""

    def test_start_as_current_span(self):
        """Test start_as_current_span returns NoOpSpan."""
        tracer = NoOpTracer()
        span = tracer.start_as_current_span("test_span")
        assert isinstance(span, NoOpSpan)

    def test_start_span(self):
        """Test start_span returns NoOpSpan."""
        tracer = NoOpTracer()
        span = tracer.start_span("test_span")
        assert isinstance(span, NoOpSpan)

    def test_span_with_attributes(self):
        """Test span creation with attributes."""
        tracer = NoOpTracer()
        span = tracer.start_as_current_span("test", attributes={"key": "value"})
        assert isinstance(span, NoOpSpan)


class TestInitTracing:
    """Tests for init_tracing function."""

    def setup_method(self):
        """Reset tracing before each test."""
        reset_tracing()

    def teardown_method(self):
        """Reset tracing after each test."""
        reset_tracing()

    def test_disabled_by_default(self):
        """Test tracing is disabled by default."""
        with patch.dict("os.environ", {"OTEL_ENABLED": "false"}):
            result = init_tracing()
            assert result is False

    def test_returns_noop_when_disabled(self):
        """Test returns NoOpTracer when disabled."""
        with patch.dict("os.environ", {"OTEL_ENABLED": "false"}):
            init_tracing()
            tracer = get_tracer()
            assert tracer.__class__.__name__ == "NoOpTracer"

    def test_only_initializes_once(self):
        """Test init_tracing only runs once."""
        with patch.dict("os.environ", {"OTEL_ENABLED": "false"}):
            result1 = init_tracing()
            result2 = init_tracing()
            # Both calls should succeed (return False since OTEL disabled)
            # The key is that the tracer remains a NoOpTracer
            tracer = get_tracer()
            assert tracer.__class__.__name__ == "NoOpTracer"

    def test_handles_missing_otel(self):
        """Test graceful handling when OpenTelemetry not installed."""
        with patch.dict("os.environ", {"OTEL_ENABLED": "true"}):
            with patch.dict("sys.modules", {"opentelemetry": None}):
                reset_tracing()
                result = init_tracing()
                assert result is False

    def test_custom_service_name(self):
        """Test custom service name parameter."""
        with patch.dict("os.environ", {"OTEL_ENABLED": "false"}):
            init_tracing(service_name="custom-service")
            # Should not raise

    def test_custom_endpoint(self):
        """Test custom endpoint parameter."""
        with patch.dict("os.environ", {"OTEL_ENABLED": "false"}):
            init_tracing(endpoint="http://custom:4317")
            # Should not raise


class TestGetTracer:
    """Tests for get_tracer function."""

    def setup_method(self):
        """Reset tracing before each test."""
        reset_tracing()

    def teardown_method(self):
        """Reset tracing after each test."""
        reset_tracing()

    def test_auto_initializes(self):
        """Test get_tracer auto-initializes if needed."""
        tracer = get_tracer()
        assert tracer is not None

    def test_returns_same_instance(self):
        """Test get_tracer returns same instance."""
        tracer1 = get_tracer()
        tracer2 = get_tracer()
        assert tracer1 is tracer2


class TestTraceSpan:
    """Tests for trace_span context manager."""

    def setup_method(self):
        """Reset tracing before each test."""
        reset_tracing()

    def test_creates_span(self):
        """Test trace_span creates a span."""
        with trace_span("test_operation") as span:
            assert span is not None

    def test_span_with_attributes(self):
        """Test trace_span with attributes."""
        with trace_span("test_op", {"key": "value"}) as span:
            # Should not raise
            span.set_attribute("another_key", "another_value")


class TestTraceFunction:
    """Tests for trace_function decorator."""

    def setup_method(self):
        """Reset tracing before each test."""
        reset_tracing()

    def test_decorated_function_works(self):
        """Test decorated function executes normally."""
        @trace_function()
        def my_func(x: int) -> int:
            return x * 2

        result = my_func(5)
        assert result == 10

    def test_custom_name(self):
        """Test decorator with custom span name."""
        @trace_function(name="custom_operation")
        def my_func() -> str:
            return "result"

        result = my_func()
        assert result == "result"

    def test_with_attributes(self):
        """Test decorator with attributes."""
        @trace_function(attributes={"component": "test"})
        def my_func() -> None:
            pass

        my_func()  # Should not raise

    def test_preserves_function_metadata(self):
        """Test decorator preserves function name and docstring."""
        @trace_function()
        def documented_func() -> None:
            """This is my docstring."""
            pass

        assert documented_func.__name__ == "documented_func"
        assert "docstring" in documented_func.__doc__

    def test_handles_exception(self):
        """Test decorator handles exceptions properly."""
        @trace_function()
        def failing_func() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func()


class TestTraceAsyncFunction:
    """Tests for trace_async_function decorator."""

    def setup_method(self):
        """Reset tracing before each test."""
        reset_tracing()

    @pytest.mark.asyncio
    async def test_decorated_async_function_works(self):
        """Test decorated async function executes normally."""
        @trace_async_function()
        async def my_async_func(x: int) -> int:
            return x * 2

        result = await my_async_func(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_handles_async_exception(self):
        """Test decorator handles async exceptions properly."""
        @trace_async_function()
        async def failing_async_func() -> None:
            raise ValueError("async error")

        with pytest.raises(ValueError, match="async error"):
            await failing_async_func()


class TestHelperFunctions:
    """Tests for helper functions."""

    def setup_method(self):
        """Reset tracing before each test."""
        reset_tracing()

    def test_add_span_attributes(self):
        """Test add_span_attributes doesn't raise."""
        add_span_attributes({"key": "value", "number": 42})

    def test_record_exception(self):
        """Test record_exception doesn't raise."""
        record_exception(ValueError("test"))


class TestSpanAttributes:
    """Tests for SpanAttributes constants."""

    def test_request_attributes(self):
        """Test request attribute names."""
        assert SpanAttributes.REQUEST_ID == "minicrit.request_id"
        assert SpanAttributes.USER_ID == "minicrit.user_id"
        assert SpanAttributes.API_KEY_HASH == "minicrit.api_key_hash"

    def test_model_attributes(self):
        """Test model attribute names."""
        assert SpanAttributes.MODEL_NAME == "minicrit.model_name"
        assert SpanAttributes.MODEL_DEVICE == "minicrit.model_device"

    def test_inference_attributes(self):
        """Test inference attribute names."""
        assert SpanAttributes.INPUT_LENGTH == "minicrit.input_length"
        assert SpanAttributes.OUTPUT_LENGTH == "minicrit.output_length"
        assert SpanAttributes.TOKENS_GENERATED == "minicrit.tokens_generated"
        assert SpanAttributes.LATENCY_MS == "minicrit.latency_ms"

    def test_domain_attributes(self):
        """Test domain attribute names."""
        assert SpanAttributes.DOMAIN == "minicrit.domain"
        assert SpanAttributes.SEVERITY == "minicrit.severity"

    def test_error_attributes(self):
        """Test error attribute names."""
        assert SpanAttributes.ERROR_TYPE == "minicrit.error_type"
        assert SpanAttributes.ERROR_MESSAGE == "minicrit.error_message"


class TestResetTracing:
    """Tests for reset_tracing function."""

    def test_reset_allows_reinit(self):
        """Test reset allows reinitialization."""
        init_tracing()
        reset_tracing()

        # Should be able to init again
        init_tracing()
        tracer = get_tracer()
        assert tracer is not None
