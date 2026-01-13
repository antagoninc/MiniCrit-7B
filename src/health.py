"""Health check endpoints for MiniCrit.

Provides Kubernetes-compatible readiness and liveness probes
for production deployments.

Endpoints:
    /health - Basic health status (backwards compatible)
    /health/live - Liveness probe (is the process alive?)
    /health/ready - Readiness probe (can it serve traffic?)
    /health/startup - Startup probe (has initialization completed?)

Example:
    >>> from src.health import health_router
    >>> app.include_router(health_router)

Kubernetes Configuration:
    livenessProbe:
      httpGet:
        path: /health/live
        port: 8000
      initialDelaySeconds: 10
      periodSeconds: 10

    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8000
      initialDelaySeconds: 30
      periodSeconds: 5

    startupProbe:
      httpGet:
        path: /health/startup
        port: 8000
      initialDelaySeconds: 60
      periodSeconds: 10
      failureThreshold: 30

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class CheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class HealthResponse:
    """Aggregated health check response."""

    status: HealthStatus
    checks: list[CheckResult]
    timestamp: str
    version: str = "1.0.0"
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": c.latency_ms,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


class HealthChecker:
    """Manages health checks for the application.

    Provides separate liveness, readiness, and startup checks
    for Kubernetes-style deployments.
    """

    def __init__(self):
        """Initialize health checker."""
        self._start_time = time.time()
        self._startup_complete = False
        self._liveness_checks: list[tuple[str, Callable[[], CheckResult]]] = []
        self._readiness_checks: list[tuple[str, Callable[[], CheckResult]]] = []
        self._startup_checks: list[tuple[str, Callable[[], CheckResult]]] = []

        # Register default checks
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        # Liveness: Basic process health
        self.register_liveness_check("process", self._check_process)

        # Readiness: Dependencies
        self.register_readiness_check("model", self._check_model)
        self.register_readiness_check("memory", self._check_memory)

        # Startup: Initialization
        self.register_startup_check("initialization", self._check_initialization)

    def register_liveness_check(
        self,
        name: str,
        check: Callable[[], CheckResult],
    ) -> None:
        """Register a liveness check.

        Liveness checks determine if the process should be restarted.
        These should be simple and fast.
        """
        self._liveness_checks.append((name, check))

    def register_readiness_check(
        self,
        name: str,
        check: Callable[[], CheckResult],
    ) -> None:
        """Register a readiness check.

        Readiness checks determine if the service can handle traffic.
        These can check dependencies and resources.
        """
        self._readiness_checks.append((name, check))

    def register_startup_check(
        self,
        name: str,
        check: Callable[[], CheckResult],
    ) -> None:
        """Register a startup check.

        Startup checks run during initialization.
        Once all pass, the service is considered started.
        """
        self._startup_checks.append((name, check))

    def mark_startup_complete(self) -> None:
        """Mark startup as complete."""
        self._startup_complete = True
        logger.info("Startup marked as complete")

    @property
    def uptime(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self._start_time

    def _run_checks(
        self,
        checks: list[tuple[str, Callable[[], CheckResult]]],
    ) -> HealthResponse:
        """Run a list of health checks."""
        results = []
        overall_status = HealthStatus.HEALTHY

        for name, check in checks:
            start = time.perf_counter()
            try:
                result = check()
                result.latency_ms = (time.perf_counter() - start) * 1000
            except Exception as e:
                result = CheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
                logger.error(f"Health check '{name}' failed: {e}")

            results.append(result)

            # Update overall status
            if result.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED

        return HealthResponse(
            status=overall_status,
            checks=results,
            timestamp=datetime.utcnow().isoformat() + "Z",
            uptime_seconds=self.uptime,
        )

    def check_liveness(self) -> HealthResponse:
        """Run liveness checks."""
        return self._run_checks(self._liveness_checks)

    def check_readiness(self) -> HealthResponse:
        """Run readiness checks."""
        return self._run_checks(self._readiness_checks)

    def check_startup(self) -> HealthResponse:
        """Run startup checks."""
        return self._run_checks(self._startup_checks)

    def check_all(self) -> HealthResponse:
        """Run all health checks."""
        all_checks = (
            self._liveness_checks
            + self._readiness_checks
            + self._startup_checks
        )
        # Deduplicate by name
        seen = set()
        unique_checks = []
        for name, check in all_checks:
            if name not in seen:
                seen.add(name)
                unique_checks.append((name, check))

        return self._run_checks(unique_checks)

    # Default check implementations

    def _check_process(self) -> CheckResult:
        """Check if process is alive."""
        return CheckResult(
            name="process",
            status=HealthStatus.HEALTHY,
            message="Process is running",
            details={"pid": os.getpid()},
        )

    def _check_model(self) -> CheckResult:
        """Check if model is loaded and ready."""
        try:
            # Try to import and check model state
            from src.api import _model_state

            if _model_state.get("loaded", False):
                return CheckResult(
                    name="model",
                    status=HealthStatus.HEALTHY,
                    message="Model is loaded",
                    details={
                        "model_name": _model_state.get("model_name"),
                        "load_time": _model_state.get("load_time"),
                    },
                )
            else:
                return CheckResult(
                    name="model",
                    status=HealthStatus.DEGRADED,
                    message="Model not loaded (will load on first request)",
                )
        except ImportError:
            return CheckResult(
                name="model",
                status=HealthStatus.UNKNOWN,
                message="Could not check model status",
            )

    def _check_memory(self) -> CheckResult:
        """Check memory usage."""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            # Warning if using > 80% of system memory
            if memory_percent > 80:
                return CheckResult(
                    name="memory",
                    status=HealthStatus.DEGRADED,
                    message=f"High memory usage: {memory_percent:.1f}%",
                    details={
                        "rss_bytes": memory_info.rss,
                        "percent": memory_percent,
                    },
                )

            return CheckResult(
                name="memory",
                status=HealthStatus.HEALTHY,
                message=f"Memory usage: {memory_percent:.1f}%",
                details={
                    "rss_bytes": memory_info.rss,
                    "percent": memory_percent,
                },
            )
        except ImportError:
            return CheckResult(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message="psutil not installed",
            )

    def _check_initialization(self) -> CheckResult:
        """Check if initialization is complete."""
        if self._startup_complete:
            return CheckResult(
                name="initialization",
                status=HealthStatus.HEALTHY,
                message="Initialization complete",
                details={"uptime_seconds": self.uptime},
            )
        else:
            return CheckResult(
                name="initialization",
                status=HealthStatus.UNHEALTHY,
                message="Initialization in progress",
                details={"uptime_seconds": self.uptime},
            )


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get or create the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def reset_health_checker() -> None:
    """Reset the global health checker (for testing)."""
    global _health_checker
    _health_checker = None


# FastAPI router for health endpoints
def create_health_router():
    """Create FastAPI router with health endpoints.

    Returns:
        APIRouter with health check endpoints.
    """
    try:
        from fastapi import APIRouter, Response
        from fastapi.responses import JSONResponse
    except ImportError:
        logger.warning("FastAPI not installed, health router not available")
        return None

    router = APIRouter(tags=["health"])

    @router.get("/health")
    async def health() -> JSONResponse:
        """Basic health check (backwards compatible)."""
        checker = get_health_checker()
        response = checker.check_all()
        status_code = 200 if response.status == HealthStatus.HEALTHY else 503
        return JSONResponse(content=response.to_dict(), status_code=status_code)

    @router.get("/health/live")
    async def liveness() -> JSONResponse:
        """Liveness probe - is the process alive?

        Returns 200 if the process is running.
        Returns 503 if the process should be restarted.
        """
        checker = get_health_checker()
        response = checker.check_liveness()
        status_code = 200 if response.status == HealthStatus.HEALTHY else 503
        return JSONResponse(content=response.to_dict(), status_code=status_code)

    @router.get("/health/ready")
    async def readiness() -> JSONResponse:
        """Readiness probe - can the service handle traffic?

        Returns 200 if the service is ready to handle requests.
        Returns 503 if the service should not receive traffic.
        """
        checker = get_health_checker()
        response = checker.check_readiness()
        status_code = 200 if response.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED) else 503
        return JSONResponse(content=response.to_dict(), status_code=status_code)

    @router.get("/health/startup")
    async def startup() -> JSONResponse:
        """Startup probe - has initialization completed?

        Returns 200 once startup is complete.
        Returns 503 while still starting up.
        """
        checker = get_health_checker()
        response = checker.check_startup()
        status_code = 200 if response.status == HealthStatus.HEALTHY else 503
        return JSONResponse(content=response.to_dict(), status_code=status_code)

    return router


# Convenience function for creating health router
health_router = create_health_router()
