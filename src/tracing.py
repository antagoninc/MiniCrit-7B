"""OpenTelemetry distributed tracing for MiniCrit.

Provides request tracing, span management, and context propagation
for distributed deployments.

Example:
    >>> from src.tracing import init_tracing, get_tracer
    >>>
    >>> # Initialize once at startup
    >>> init_tracing(service_name="minicrit-api")
    >>>
    >>> # Use in code
    >>> tracer = get_tracer()
    >>> with tracer.start_as_current_span("process_request") as span:
    ...     span.set_attribute("user_id", "123")
    ...     result = process()

Environment variables:
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint (default: http://localhost:4317)
    OTEL_SERVICE_NAME: Service name (default: minicrit)
    OTEL_ENABLED: Enable tracing (default: false)
    OTEL_SAMPLE_RATE: Sampling rate 0.0-1.0 (default: 1.0)

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, Optional, TypeVar

logger = logging.getLogger(__name__)

# Environment configuration
OTEL_ENABLED = os.environ.get("OTEL_ENABLED", "false").lower() == "true"
OTEL_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
OTEL_SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "minicrit")
OTEL_SAMPLE_RATE = float(os.environ.get("OTEL_SAMPLE_RATE", "1.0"))

# Type var for decorator
F = TypeVar("F", bound=Callable[..., Any])

# Global state
_tracer: Optional[Any] = None
_initialized = False


class NoOpSpan:
    """No-op span for when tracing is disabled."""

    def __enter__(self) -> "NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
        pass

    def is_recording(self) -> bool:
        return False


class NoOpTracer:
    """No-op tracer for when tracing is disabled."""

    def start_as_current_span(
        self,
        name: str,
        attributes: Optional[dict] = None,
        **kwargs: Any,
    ) -> NoOpSpan:
        return NoOpSpan()

    def start_span(
        self,
        name: str,
        attributes: Optional[dict] = None,
        **kwargs: Any,
    ) -> NoOpSpan:
        return NoOpSpan()


def init_tracing(
    service_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    sample_rate: Optional[float] = None,
    enabled: Optional[bool] = None,
) -> bool:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Service name for traces. Defaults to OTEL_SERVICE_NAME.
        endpoint: OTLP endpoint. Defaults to OTEL_EXPORTER_OTLP_ENDPOINT.
        sample_rate: Sampling rate 0.0-1.0. Defaults to OTEL_SAMPLE_RATE.
        enabled: Enable tracing. Defaults to OTEL_ENABLED.

    Returns:
        True if tracing was initialized, False otherwise.
    """
    global _tracer, _initialized

    if _initialized:
        logger.debug("Tracing already initialized")
        return _tracer is not None

    _service_name = service_name or OTEL_SERVICE_NAME
    _endpoint = endpoint or OTEL_ENDPOINT
    _sample_rate = sample_rate if sample_rate is not None else OTEL_SAMPLE_RATE
    _enabled = enabled if enabled is not None else OTEL_ENABLED

    if not _enabled:
        logger.info("OpenTelemetry tracing disabled")
        _tracer = NoOpTracer()
        _initialized = True
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        # Create resource with service name
        resource = Resource.create({SERVICE_NAME: _service_name})

        # Create sampler
        sampler = TraceIdRatioBased(_sample_rate)

        # Create tracer provider
        provider = TracerProvider(resource=resource, sampler=sampler)

        # Create OTLP exporter
        exporter = OTLPSpanExporter(endpoint=_endpoint, insecure=True)

        # Add batch processor
        provider.add_span_processor(BatchSpanProcessor(exporter))

        # Set as global tracer provider
        trace.set_tracer_provider(provider)

        # Get tracer
        _tracer = trace.get_tracer(_service_name)
        _initialized = True

        logger.info(
            f"OpenTelemetry tracing initialized: service={_service_name}, "
            f"endpoint={_endpoint}, sample_rate={_sample_rate}"
        )
        return True

    except ImportError as e:
        logger.warning(
            f"OpenTelemetry not installed: {e}. Install with: pip install minicrit[observability]"
        )
        _tracer = NoOpTracer()
        _initialized = True
        return False

    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}")
        _tracer = NoOpTracer()
        _initialized = True
        return False


def get_tracer() -> Any:
    """Get the global tracer instance.

    Returns:
        Tracer instance (real or no-op).
    """
    global _tracer, _initialized

    if not _initialized:
        init_tracing()

    return _tracer or NoOpTracer()


def reset_tracing() -> None:
    """Reset tracing state (for testing)."""
    global _tracer, _initialized
    _tracer = None
    _initialized = False


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[dict[str, Any]] = None,
) -> Generator[Any, None, None]:
    """Context manager for creating a traced span.

    Args:
        name: Span name.
        attributes: Optional span attributes.

    Yields:
        The span object.

    Example:
        >>> with trace_span("process_request", {"user_id": "123"}) as span:
        ...     result = process()
        ...     span.set_attribute("result_size", len(result))
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name, attributes=attributes) as span:
        yield span


def trace_function(
    name: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator for tracing function execution.

    Args:
        name: Span name. Defaults to function name.
        attributes: Optional span attributes.

    Returns:
        Decorated function.

    Example:
        >>> @trace_function(attributes={"component": "inference"})
        ... def generate_critique(rationale: str) -> str:
        ...     return "critique"
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name, attributes=attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    try:
                        from opentelemetry.trace import StatusCode

                        span.set_status(StatusCode.ERROR, str(e))
                    except ImportError:
                        pass
                    raise

        return wrapper  # type: ignore

    return decorator


def trace_async_function(
    name: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator for tracing async function execution.

    Args:
        name: Span name. Defaults to function name.
        attributes: Optional span attributes.

    Returns:
        Decorated async function.
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name, attributes=attributes) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    try:
                        from opentelemetry.trace import StatusCode

                        span.set_status(StatusCode.ERROR, str(e))
                    except ImportError:
                        pass
                    raise

        return wrapper  # type: ignore

    return decorator


def add_span_attributes(attributes: dict[str, Any]) -> None:
    """Add attributes to the current span.

    Args:
        attributes: Dictionary of attributes to add.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span.is_recording():
            for key, value in attributes.items():
                span.set_attribute(key, value)
    except ImportError:
        pass


def record_exception(exception: Exception) -> None:
    """Record an exception in the current span.

    Args:
        exception: The exception to record.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span.is_recording():
            span.record_exception(exception)
    except ImportError:
        pass


def instrument_fastapi(app: Any) -> None:
    """Instrument a FastAPI application with OpenTelemetry.

    Args:
        app: FastAPI application instance.
    """
    if not OTEL_ENABLED:
        logger.debug("OpenTelemetry disabled, skipping FastAPI instrumentation")
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumented with OpenTelemetry")

    except ImportError:
        logger.warning(
            "opentelemetry-instrumentation-fastapi not installed. "
            "Install with: pip install minicrit[observability]"
        )
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI: {e}")


# Convenience exports for common span attribute keys
class SpanAttributes:
    """Common span attribute keys for MiniCrit."""

    # Request attributes
    REQUEST_ID = "minicrit.request_id"
    USER_ID = "minicrit.user_id"
    API_KEY_HASH = "minicrit.api_key_hash"

    # Model attributes
    MODEL_NAME = "minicrit.model_name"
    MODEL_DEVICE = "minicrit.model_device"

    # Inference attributes
    INPUT_LENGTH = "minicrit.input_length"
    OUTPUT_LENGTH = "minicrit.output_length"
    TOKENS_GENERATED = "minicrit.tokens_generated"
    LATENCY_MS = "minicrit.latency_ms"

    # Domain attributes
    DOMAIN = "minicrit.domain"
    SEVERITY = "minicrit.severity"

    # Error attributes
    ERROR_TYPE = "minicrit.error_type"
    ERROR_MESSAGE = "minicrit.error_message"
