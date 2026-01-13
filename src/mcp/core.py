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
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

# ================================================================
# Configuration
# ================================================================

ADAPTER_ID = os.environ.get("MINICRIT_ADAPTER", "wmaousley/MiniCrit-7B")
BASE_MODEL_ID = os.environ.get("MINICRIT_BASE_MODEL", "Qwen/Qwen2-7B-Instruct")
DEVICE = os.environ.get("MINICRIT_DEVICE", "auto")
MAX_LENGTH = int(os.environ.get("MINICRIT_MAX_LENGTH", "512"))
INFERENCE_TIMEOUT = int(os.environ.get("MINICRIT_INFERENCE_TIMEOUT", "120"))  # seconds
LOG_LEVEL = os.environ.get("MINICRIT_LOG_LEVEL", "INFO")

# Quantization configuration (none, 8bit, 4bit)
QUANTIZATION = os.environ.get("MINICRIT_QUANTIZATION", "none").lower()
VALID_QUANTIZATION_OPTIONS = ["none", "8bit", "4bit"]

# CORS configuration
CORS_ORIGINS = os.environ.get(
    "MINICRIT_CORS_ORIGINS", "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000"
).split(",")

DOMAINS = [
    "trading",
    "finance",
    "risk_assessment",
    "resource_allocation",
    "planning_scheduling",
    "cybersecurity",
    "defense",
    "medical",
    "general",
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


class InputSanitizationError(InvalidInputError):
    """Raised when input contains potentially malicious content."""

    pass


# ================================================================
# Input Sanitizer (Security Hardening)
# ================================================================


class InputSanitizer:
    """Input sanitizer for security hardening.

    Validates and sanitizes user inputs to prevent prompt injection
    attacks and other malicious content.

    Example:
        >>> sanitizer = InputSanitizer()
        >>> clean_rationale = sanitizer.validate_rationale("AAPL is bullish...")
        >>> clean_domain = sanitizer.validate_domain("trading")
    """

    # Maximum lengths
    MAX_RATIONALE_LENGTH = 4096
    MAX_CONTEXT_LENGTH = 4096
    MIN_RATIONALE_LENGTH = 10

    # Injection patterns to reject (case-insensitive)
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|text)",
        r"disregard\s+(all\s+)?(previous|prior|above)",
        r"forget\s+(everything|all|previous)",
        r"system\s*:\s*",
        r"<\|?(system|assistant|user|im_start|im_end)\|?>",
        r"###\s*(system|instruction|prompt)",
        r"\[INST\]",
        r"\[/INST\]",
        r"<<SYS>>",
        r"<</SYS>>",
        r"you\s+are\s+(now\s+)?a\s+(new\s+)?(ai|assistant|chatbot)",
        r"new\s+instructions?\s*:",
        r"override\s+(instructions?|prompts?|system)",
        r"jailbreak",
        r"dan\s+mode",
        r"developer\s+mode",
        r"ignore\s+safety",
        r"bypass\s+(filter|safety|restriction)",
    ]

    # Control characters to strip (keep basic whitespace)
    CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

    def __init__(self):
        """Initialize the sanitizer with compiled patterns."""
        self._logger = logging.getLogger("minicrit.input_sanitizer")
        self._injection_regex = re.compile("|".join(self.INJECTION_PATTERNS), re.IGNORECASE)

    def validate_rationale(self, rationale: str) -> str:
        """Validate and sanitize rationale input.

        Args:
            rationale: The rationale text to validate.

        Returns:
            Sanitized rationale string.

        Raises:
            InvalidInputError: If rationale is empty or too short.
            InputSanitizationError: If rationale contains injection patterns.
        """
        if rationale is None:
            raise InvalidInputError("Rationale cannot be None")

        # Strip control characters
        clean = self._strip_control_chars(rationale)

        # Strip whitespace
        clean = clean.strip()

        if not clean:
            raise InvalidInputError("Rationale cannot be empty")

        if len(clean) < self.MIN_RATIONALE_LENGTH:
            raise InvalidInputError(
                f"Rationale must be at least {self.MIN_RATIONALE_LENGTH} characters"
            )

        if len(clean) > self.MAX_RATIONALE_LENGTH:
            raise InvalidInputError(
                f"Rationale cannot exceed {self.MAX_RATIONALE_LENGTH} characters "
                f"(got {len(clean)})"
            )

        # Check for injection patterns
        self._check_injection(clean, "rationale")

        return clean

    def validate_domain(self, domain: str) -> str:
        """Validate domain input against allowed list.

        Args:
            domain: The domain to validate.

        Returns:
            Validated domain string.

        Raises:
            InvalidInputError: If domain is not in allowed list.
        """
        if domain is None:
            return "general"

        clean = str(domain).strip().lower()

        if not clean:
            return "general"

        if clean not in DOMAINS:
            raise InvalidInputError(
                f"Invalid domain: '{clean}'. Must be one of: {', '.join(DOMAINS)}"
            )

        return clean

    def validate_context(self, context: str | None) -> str | None:
        """Validate and sanitize context input.

        Args:
            context: The context text to validate (can be None).

        Returns:
            Sanitized context string or None.

        Raises:
            InputSanitizationError: If context contains injection patterns.
        """
        if context is None:
            return None

        # Strip control characters
        clean = self._strip_control_chars(context)

        # Strip whitespace
        clean = clean.strip()

        if not clean:
            return None

        if len(clean) > self.MAX_CONTEXT_LENGTH:
            raise InvalidInputError(
                f"Context cannot exceed {self.MAX_CONTEXT_LENGTH} characters " f"(got {len(clean)})"
            )

        # Check for injection patterns
        self._check_injection(clean, "context")

        return clean

    def _strip_control_chars(self, text: str) -> str:
        """Remove control characters from text.

        Keeps newlines, tabs, and carriage returns for formatting.

        Args:
            text: Input text.

        Returns:
            Text with control characters removed.
        """
        return self.CONTROL_CHAR_PATTERN.sub("", text)

    def _check_injection(self, text: str, field_name: str) -> None:
        """Check text for injection patterns.

        Args:
            text: Text to check.
            field_name: Name of field for error message.

        Raises:
            InputSanitizationError: If injection pattern detected.
        """
        match = self._injection_regex.search(text)
        if match:
            self._logger.warning(
                f"Injection pattern detected in {field_name}: '{match.group()[:50]}'"
            )
            raise InputSanitizationError(
                f"Input contains potentially malicious content in {field_name}"
            )

    def sanitize_all(
        self,
        rationale: str,
        domain: str = "general",
        context: str | None = None,
    ) -> tuple[str, str, str | None]:
        """Validate and sanitize all inputs at once.

        Args:
            rationale: The rationale to validate.
            domain: The domain to validate.
            context: Optional context to validate.

        Returns:
            Tuple of (sanitized_rationale, sanitized_domain, sanitized_context).

        Raises:
            InvalidInputError: If validation fails.
            InputSanitizationError: If injection detected.
        """
        clean_rationale = self.validate_rationale(rationale)
        clean_domain = self.validate_domain(domain)
        clean_context = self.validate_context(context)

        return clean_rationale, clean_domain, clean_context


# Global sanitizer instance
_input_sanitizer: InputSanitizer | None = None


def get_input_sanitizer() -> InputSanitizer:
    """Get the global InputSanitizer instance."""
    global _input_sanitizer
    if _input_sanitizer is None:
        _input_sanitizer = InputSanitizer()
    return _input_sanitizer


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

    _instance: ModelManager | None = None
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
        self._async_lock: asyncio.Lock | None = None
        self._is_loading = False
        self._load_error: Exception | None = None
        self._device: str | None = None
        self._quantization_mode: str = "none"
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
    def device(self) -> str | None:
        """Get device model is loaded on."""
        return self._device

    @property
    def quantization_mode(self) -> str:
        """Get quantization mode (none, 8bit, 4bit)."""
        return self._quantization_mode

    @property
    def stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        stats = self._stats.copy()
        stats["quantization"] = self._quantization_mode
        return stats  # type: ignore[no-any-return]

    def get_model(self, timeout: float | None = None) -> tuple[Any, Any]:
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

    async def get_model_async(self, timeout: float | None = None) -> tuple[Any, Any]:
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
            await loop.run_in_executor(self._executor, lambda: self._load_model_sync(timeout))

            return self._model, self._tokenizer

    def _load_model_sync(self, timeout: float | None = None) -> None:
        """Load model synchronously with optional quantization.

        Supports 8-bit and 4-bit quantization via bitsandbytes library.
        Falls back to fp16/bf16 if quantization library not available.
        """
        if self._model is not None:
            return

        if self._is_loading:
            self._logger.warning("Model is already being loaded")
            return

        self._is_loading = True
        self._load_error = None
        self._quantization_mode = "none"

        try:
            self._logger.info(f"Loading base model: {BASE_MODEL_ID}")
            self._logger.info(f"Loading adapter: {ADAPTER_ID}")

            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            # Check quantization configuration
            quantize = QUANTIZATION if QUANTIZATION in VALID_QUANTIZATION_OPTIONS else "none"

            # Check if bitsandbytes is available for quantization
            bnb_available = False
            if quantize in ["8bit", "4bit"]:
                try:
                    import bitsandbytes

                    bnb_available = True
                    self._logger.info(f"bitsandbytes available, using {quantize} quantization")
                except ImportError:
                    self._logger.warning(
                        "bitsandbytes not installed, falling back to fp16. "
                        "Install with: pip install bitsandbytes>=0.41.0"
                    )
                    quantize = "none"

            # Determine device
            if DEVICE == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                    # MPS doesn't support quantization
                    if quantize != "none":
                        self._logger.warning("MPS device doesn't support quantization, using fp16")
                        quantize = "none"
                else:
                    device = "cpu"
                    # CPU quantization requires specific setup
                    if quantize != "none":
                        self._logger.warning("CPU quantization not recommended, using fp32")
                        quantize = "none"
            else:
                device = DEVICE

            self._logger.info(f"Using device: {device}")
            self._logger.info(f"Quantization mode: {quantize}")
            self._device = device
            self._quantization_mode = quantize

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load base model with quantization if enabled
            self._logger.info("Loading base model weights...")

            if quantize == "8bit" and bnb_available:
                # 8-bit quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_ID,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
                self._logger.info("Loaded model with 8-bit quantization")

            elif quantize == "4bit" and bnb_available:
                # 4-bit quantization (NF4)
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_ID,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
                self._logger.info("Loaded model with 4-bit NF4 quantization")

            else:
                # Standard fp16/bf16 loading
                dtype = torch.float16 if device == "mps" else torch.bfloat16
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_ID,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )
                base_model = base_model.to(device)  # type: ignore[arg-type]
                self._logger.info(f"Loaded model with {dtype}")

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

    def __init__(
        self,
        model_manager: ModelManager | None = None,
        sanitizer: InputSanitizer | None = None,
    ):
        """Initialize critique generator.

        Args:
            model_manager: Optional ModelManager instance.
                          Uses singleton if not provided.
            sanitizer: Optional InputSanitizer instance.
                      Uses global instance if not provided.
        """
        self._manager = model_manager or ModelManager.get_instance()
        self._sanitizer = sanitizer or get_input_sanitizer()
        self._logger = logging.getLogger("minicrit.critique_generator")
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="inference")

    def validate_input(self, rationale: str, domain: str) -> tuple[str, str]:
        """Validate and sanitize input parameters.

        Uses InputSanitizer to validate inputs and check for injection attacks.

        Args:
            rationale: The rationale to validate.
            domain: The domain context.

        Returns:
            Tuple of (sanitized_rationale, sanitized_domain).

        Raises:
            InvalidInputError: If validation fails.
            InputSanitizationError: If injection detected.
        """
        clean_rationale = self._sanitizer.validate_rationale(rationale)
        clean_domain = self._sanitizer.validate_domain(domain)
        return clean_rationale, clean_domain

    def generate(
        self,
        rationale: str,
        domain: str = "general",
        context: str | None = None,
        request_id: str = "",
        timeout: float | None = None,
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
            InputSanitizationError: If injection detected.
            InferenceTimeoutError: If inference times out.
            InferenceError: If inference fails.
        """
        # Sanitize all inputs
        clean_rationale, clean_domain = self.validate_input(rationale, domain)
        clean_context = self._sanitizer.validate_context(context)

        timeout = timeout or INFERENCE_TIMEOUT

        try:
            future = self._executor.submit(
                self._generate_impl, clean_rationale, clean_domain, clean_context, request_id
            )
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            raise InferenceTimeoutError(f"Inference timed out after {timeout}s")

    async def generate_async(
        self,
        rationale: str,
        domain: str = "general",
        context: str | None = None,
        request_id: str = "",
        timeout: float | None = None,
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

        Raises:
            InvalidInputError: If input validation fails.
            InputSanitizationError: If injection detected.
            InferenceTimeoutError: If inference times out.
            InferenceError: If inference fails.
        """
        # Sanitize all inputs
        clean_rationale, clean_domain = self.validate_input(rationale, domain)
        clean_context = self._sanitizer.validate_context(context)

        timeout = timeout or INFERENCE_TIMEOUT

        loop = asyncio.get_event_loop()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    lambda: self._generate_impl(
                        clean_rationale, clean_domain, clean_context, request_id
                    ),
                ),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            raise InferenceTimeoutError(f"Inference timed out after {timeout}s")

    def _generate_impl(
        self,
        rationale: str,
        domain: str,
        context: str | None,
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
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
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
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
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
            self._requests[key] = [t for t in self._requests[key] if t > window_start]

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

    def __init__(self, model_manager: ModelManager | None = None):
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
