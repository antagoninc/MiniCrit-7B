#!/usr/bin/env python3
# ================================================================
# MiniCrit MCP Core - Shared Model Manager and Critique Generator
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================
"""
Shared core module for MiniCrit MCP servers.

Provides thread-safe model management and critique generation
that can be used by stdio, HTTP, and production servers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Any, Callable

# ================================================================
# Configuration
# ================================================================

ADAPTER_ID = os.environ.get("MINICRIT_ADAPTER", "wmaousley/MiniCrit-7B")
BASE_MODEL_ID = os.environ.get("MINICRIT_BASE_MODEL", "Qwen/Qwen2-7B-Instruct")
DEVICE = os.environ.get("MINICRIT_DEVICE", "auto")
MAX_LENGTH = int(os.environ.get("MINICRIT_MAX_LENGTH", "512"))
INFERENCE_TIMEOUT = int(os.environ.get("MINICRIT_INFERENCE_TIMEOUT", "120"))  # seconds
LOG_LEVEL = os.environ.get("MINICRIT_LOG_LEVEL", "INFO")

# CORS configuration
CORS_ORIGINS = os.environ.get(
    "MINICRIT_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000"
).split(",")

DOMAINS = [
    "trading", "finance", "risk_assessment", "resource_allocation",
    "planning_scheduling", "cybersecurity", "defense", "medical", "general",
]


class Severity(str, Enum):
    """Critique severity levels."""
    PASS = "pass"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CritiqueResult:
    """Result of critique generation."""
    valid: bool
    severity: str
    critique: str
    confidence: float
    flags: list
    domain: str
    latency_ms: float
    request_id: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


# ================================================================
# Custom Exceptions
# ================================================================

class ModelNotLoadedError(Exception):
    """Raised when model is not loaded."""
    pass


class ModelLoadError(Exception):
    """Raised when model fails to load."""
    pass


class InferenceTimeoutError(Exception):
    """Raised when inference times out."""
    pass


class InferenceError(Exception):
    """Raised when inference fails."""
    pass


class InvalidInputError(ValueError):
    """Raised when input validation fails."""
    pass


# ================================================================
# Thread-Safe Model Manager
# ================================================================

class ModelManager:
    """Thread-safe singleton model manager.

    Handles model loading, inference, and cleanup with proper
    synchronization for concurrent access.

    Example:
        >>> manager = ModelManager.get_instance()
        >>> model, tokenizer = manager.get_model()
        >>> # Or async version
        >>> model, tokenizer = await manager.get_model_async()
    """

    _instance: Optional[ModelManager] = None
    _lock = threading.Lock()

    def __new__(cls) -> ModelManager:
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> ModelManager:
        """Get singleton instance."""
        return cls()

    def __init__(self):
        """Initialize model manager."""
        if getattr(self, "_initialized", False):
            return

        self._model = None
        self._tokenizer = None
        self._model_lock = threading.RLock()
        self._async_lock: Optional[asyncio.Lock] = None
        self._is_loading = False
        self._load_error: Optional[Exception] = None
        self._device: Optional[str] = None
        self._stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_latency_ms": 0,
            "errors": 0,
        }
        self._logger = logging.getLogger("minicrit.model_manager")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model_loader")
        self._initialized = True

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def device(self) -> Optional[str]:
        """Get device model is loaded on."""
        return self._device

    @property
    def stats(self) -> dict:
        """Get usage statistics."""
        return self._stats.copy()

    def get_model(self, timeout: Optional[float] = None) -> tuple[Any, Any]:
        """Get model and tokenizer, loading if necessary.

        Thread-safe synchronous model access.

        Args:
            timeout: Optional timeout for model loading in seconds.

        Returns:
            Tuple of (model, tokenizer).

        Raises:
            ModelLoadError: If model fails to load.
            InferenceTimeoutError: If loading times out.
        """
        with self._model_lock:
            if self._model is not None:
                return self._model, self._tokenizer

            if self._load_error is not None:
                raise ModelLoadError(f"Previous load failed: {self._load_error}")

            self._load_model_sync(timeout)
            return self._model, self._tokenizer

    async def get_model_async(self, timeout: Optional[float] = None) -> tuple[Any, Any]:
        """Get model and tokenizer asynchronously.

        Async-safe model access with proper locking.

        Args:
            timeout: Optional timeout for model loading in seconds.

        Returns:
            Tuple of (model, tokenizer).
        """
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        async with self._async_lock:
            if self._model is not None:
                return self._model, self._tokenizer

            # Run synchronous loading in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                lambda: self._load_model_sync(timeout)
            )

            return self._model, self._tokenizer

    def _load_model_sync(self, timeout: Optional[float] = None) -> None:
        """Load model synchronously."""
        if self._model is not None:
            return

        if self._is_loading:
            self._logger.warning("Model is already being loaded")
            return

        self._is_loading = True
        self._load_error = None

        try:
            self._logger.info(f"Loading base model: {BASE_MODEL_ID}")
            self._logger.info(f"Loading adapter: {ADAPTER_ID}")

            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel

            # Determine device
            if DEVICE == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = DEVICE

            self._logger.info(f"Using device: {device}")
            self._device = device

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_ID,
                trust_remote_code=True
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load base model
            self._logger.info("Loading base model weights...")
            dtype = torch.float16 if device == "mps" else torch.bfloat16

            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            base_model = base_model.to(device)

            # Apply LoRA adapter
            self._logger.info("Applying LoRA adapter...")
            self._model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
            self._model.eval()

            self._logger.info(f"Model loaded on {next(self._model.parameters()).device}")

        except ImportError as e:
            self._load_error = e
            raise ModelLoadError(f"Missing dependency: {e}")
        except FileNotFoundError as e:
            self._load_error = e
            raise ModelLoadError(f"Model or adapter not found: {e}")
        except RuntimeError as e:
            self._load_error = e
            if "out of memory" in str(e).lower():
                raise ModelLoadError(f"GPU out of memory: {e}")
            raise ModelLoadError(f"Runtime error during load: {e}")
        except Exception as e:
            self._load_error = e
            raise ModelLoadError(f"Failed to load model: {e}")
        finally:
            self._is_loading = False

    def unload(self) -> None:
        """Unload model and free memory."""
        with self._model_lock:
            if self._model is not None:
                self._logger.info("Unloading model...")

                import torch

                del self._model
                del self._tokenizer
                self._model = None
                self._tokenizer = None
                self._device = None

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self._logger.info("Model unloaded")

    def preload(self) -> None:
        """Preload model (for startup hooks)."""
        self._logger.info("Preloading model...")
        self.get_model()
        self._logger.info("Model preloaded successfully")

    async def preload_async(self) -> None:
        """Preload model asynchronously."""
        self._logger.info("Preloading model (async)...")
        await self.get_model_async()
        self._logger.info("Model preloaded successfully")

    def update_stats(self, tokens: int, latency_ms: float, error: bool = False) -> None:
        """Update usage statistics."""
        with self._model_lock:
            self._stats["total_requests"] += 1
            self._stats["total_tokens"] += tokens
            self._stats["total_latency_ms"] += latency_ms
            if error:
                self._stats["errors"] += 1

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)


# ================================================================
# Critique Generator
# ================================================================

class CritiqueGenerator:
    """Thread-safe critique generator.

    Uses ModelManager for model access and provides
    timeout-protected inference.

    Example:
        >>> generator = CritiqueGenerator()
        >>> result = generator.generate("AAPL is bullish because...", domain="trading")
        >>> # Or async
        >>> result = await generator.generate_async("AAPL is bullish...", domain="trading")
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        """Initialize critique generator.

        Args:
            model_manager: Optional ModelManager instance.
                          Uses singleton if not provided.
        """
        self._manager = model_manager or ModelManager.get_instance()
        self._logger = logging.getLogger("minicrit.critique_generator")
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="inference")

    def validate_input(self, rationale: str, domain: str) -> None:
        """Validate input parameters.

        Args:
            rationale: The rationale to validate.
            domain: The domain context.

        Raises:
            InvalidInputError: If validation fails.
        """
        if not rationale or not rationale.strip():
            raise InvalidInputError("Rationale cannot be empty")

        if len(rationale) < 10:
            raise InvalidInputError("Rationale must be at least 10 characters")

        if len(rationale) > 10000:
            raise InvalidInputError("Rationale cannot exceed 10000 characters")

        if domain not in DOMAINS:
            raise InvalidInputError(f"Invalid domain: {domain}. Must be one of: {DOMAINS}")

    def generate(
        self,
        rationale: str,
        domain: str = "general",
        context: Optional[str] = None,
        request_id: str = "",
        timeout: Optional[float] = None,
    ) -> CritiqueResult:
        """Generate critique synchronously with timeout.

        Args:
            rationale: The AI reasoning to critique.
            domain: Domain context.
            context: Additional context.
            request_id: Optional request ID for tracking.
            timeout: Inference timeout in seconds.

        Returns:
            CritiqueResult with validation results.

        Raises:
            InvalidInputError: If input validation fails.
            InferenceTimeoutError: If inference times out.
            InferenceError: If inference fails.
        """
        self.validate_input(rationale, domain)

        timeout = timeout or INFERENCE_TIMEOUT

        try:
            future = self._executor.submit(
                self._generate_impl,
                rationale, domain, context, request_id
            )
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            raise InferenceTimeoutError(f"Inference timed out after {timeout}s")

    async def generate_async(
        self,
        rationale: str,
        domain: str = "general",
        context: Optional[str] = None,
        request_id: str = "",
        timeout: Optional[float] = None,
    ) -> CritiqueResult:
        """Generate critique asynchronously with timeout.

        Args:
            rationale: The AI reasoning to critique.
            domain: Domain context.
            context: Additional context.
            request_id: Optional request ID for tracking.
            timeout: Inference timeout in seconds.

        Returns:
            CritiqueResult with validation results.
        """
        self.validate_input(rationale, domain)

        timeout = timeout or INFERENCE_TIMEOUT

        loop = asyncio.get_event_loop()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    lambda: self._generate_impl(rationale, domain, context, request_id)
                ),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            raise InferenceTimeoutError(f"Inference timed out after {timeout}s")

    def _generate_impl(
        self,
        rationale: str,
        domain: str,
        context: Optional[str],
        request_id: str,
    ) -> CritiqueResult:
        """Internal implementation of critique generation."""
        import torch

        start_time = time.time()

        try:
            model, tokenizer = self._manager.get_model()

            # Format prompt
            prompt_parts = [f"### Domain: {domain}\n"]
            if context:
                prompt_parts.append(f"### Context:\n{context}\n")
            prompt_parts.append(f"### Rationale:\n{rationale}\n\n### Critique:\n")
            prompt = "".join(prompt_parts)

            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode
            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Parse severity and flags
            severity, flags = self._parse_critique(generated)

            valid = severity in [Severity.PASS, Severity.LOW]
            confidence = 0.85 if len(generated) >= 20 else 0.5
            if len(flags) > 2:
                confidence = 0.9

            latency_ms = (time.time() - start_time) * 1000
            tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

            # Update stats
            self._manager.update_stats(tokens, latency_ms)

            return CritiqueResult(
                valid=valid,
                severity=severity.value,
                critique=generated,
                confidence=confidence,
                flags=flags,
                domain=domain,
                latency_ms=round(latency_ms, 2),
                request_id=request_id,
            )

        except ModelLoadError:
            raise
        except torch.cuda.OutOfMemoryError as e:
            self._manager.update_stats(0, 0, error=True)
            raise InferenceError(f"GPU out of memory: {e}")
        except RuntimeError as e:
            self._manager.update_stats(0, 0, error=True)
            raise InferenceError(f"Runtime error: {e}")
        except Exception as e:
            self._manager.update_stats(0, 0, error=True)
            raise InferenceError(f"Inference failed: {e}")

    def _parse_critique(self, generated: str) -> tuple[Severity, list[str]]:
        """Parse critique text for severity and flags.

        Args:
            generated: Generated critique text.

        Returns:
            Tuple of (severity, flags).
        """
        flags = []
        severity = Severity.PASS
        critique_lower = generated.lower()

        # Determine severity
        if any(w in critique_lower for w in ["critical", "severe", "dangerous", "fatal"]):
            severity = Severity.CRITICAL
            flags.append("critical_flaw")
        elif any(w in critique_lower for w in ["significant", "major", "serious", "flawed"]):
            severity = Severity.HIGH
            flags.append("significant_flaw")
        elif any(w in critique_lower for w in ["concern", "issue", "problem", "missing"]):
            severity = Severity.MEDIUM
            flags.append("notable_concern")
        elif any(w in critique_lower for w in ["minor", "slight", "small"]):
            severity = Severity.LOW
            flags.append("minor_issue")

        # Detect specific issues
        if any(w in critique_lower for w in ["overconfident", "overconfidence", "too certain"]):
            flags.append("overconfidence")
        if any(w in critique_lower for w in ["missing", "omit", "neglect", "fail to consider"]):
            flags.append("missing_consideration")
        if any(w in critique_lower for w in ["contradict", "inconsistent", "conflict"]):
            flags.append("logical_inconsistency")
        if any(w in critique_lower for w in ["evidence", "support", "justify", "unsupported"]):
            flags.append("insufficient_evidence")
        if any(w in critique_lower for w in ["risk", "danger", "threat", "hazard"]):
            flags.append("unaddressed_risk")

        return severity, flags

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)


# ================================================================
# Rate Limiter
# ================================================================

class RateLimiter:
    """Thread-safe sliding window rate limiter.

    Example:
        >>> limiter = RateLimiter(limit=60, window=60)
        >>> allowed, remaining = limiter.check("user_123")
        >>> if not allowed:
        ...     raise RateLimitExceeded()
    """

    def __init__(self, limit: int = 60, window: int = 60):
        """Initialize rate limiter.

        Args:
            limit: Maximum requests per window.
            window: Window size in seconds.
        """
        self.limit = limit
        self.window = window
        self._requests: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """Check if request is allowed.

        Args:
            key: Unique identifier (e.g., API key hash, IP).

        Returns:
            Tuple of (allowed, remaining_requests).
        """
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
        """Get seconds until rate limit resets.

        Args:
            key: Unique identifier.

        Returns:
            Seconds until oldest request expires.
        """
        with self._lock:
            if key not in self._requests or not self._requests[key]:
                return 0

            oldest = min(self._requests[key])
            reset_time = oldest + self.window - time.time()
            return max(0, reset_time)


# ================================================================
# Graceful Shutdown Handler
# ================================================================

class GracefulShutdown:
    """Handler for graceful shutdown with model cleanup.

    Example:
        >>> shutdown = GracefulShutdown()
        >>> shutdown.register()
        >>> # On SIGTERM/SIGINT, model will be unloaded
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        """Initialize shutdown handler.

        Args:
            model_manager: Optional ModelManager to cleanup.
        """
        self._manager = model_manager or ModelManager.get_instance()
        self._logger = logging.getLogger("minicrit.shutdown")
        self._shutdown_event = threading.Event()

    def register(self) -> None:
        """Register signal handlers."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        self._logger.info("Graceful shutdown handlers registered")

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signal."""
        sig_name = signal.Signals(signum).name
        self._logger.info(f"Received {sig_name}, initiating graceful shutdown...")

        self._shutdown_event.set()
        self._manager.unload()

        self._logger.info("Shutdown complete")

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._shutdown_event.is_set()


# ================================================================
# Convenience Functions
# ================================================================

def get_model_manager() -> ModelManager:
    """Get the singleton ModelManager instance."""
    return ModelManager.get_instance()


def get_critique_generator() -> CritiqueGenerator:
    """Get a CritiqueGenerator instance."""
    return CritiqueGenerator()


def get_cors_origins() -> list[str]:
    """Get configured CORS origins."""
    return CORS_ORIGINS.copy()


# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
