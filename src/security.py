# ================================================================
# MiniCrit - Security Hardening Module
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================

"""
Security hardening utilities for MiniCrit production deployments.

Features:
- Input validation and sanitization
- Request authentication
- API key management
- Rate limiting integration
- Security headers
- Audit logging
"""

import hashlib
import hmac
import logging
import os
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


# ================================================================
# Input Validation
# ================================================================


@dataclass
class ValidationResult:
    """Result of input validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    sanitized_input: str | None = None


class InputValidator:
    """Validates and sanitizes user input."""

    # Maximum input lengths
    MAX_RATIONALE_LENGTH = 10000
    MAX_DOMAIN_LENGTH = 50
    MAX_CONTEXT_LENGTH = 5000

    # Allowed characters pattern (alphanumeric, punctuation, whitespace)
    ALLOWED_PATTERN = re.compile(r"^[\w\s\.,!?;:'\"\-\(\)\[\]{}@#$%^&*+=/<>\\|`~\n\r\t]+$")

    # Potentially dangerous patterns
    DANGEROUS_PATTERNS = [
        re.compile(r"<script", re.IGNORECASE),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),  # onclick, onerror, etc.
        re.compile(r"data:\s*text/html", re.IGNORECASE),
    ]

    @classmethod
    def validate_rationale(cls, rationale: str) -> ValidationResult:
        """Validate rationale input."""
        errors = []

        if not rationale:
            return ValidationResult(valid=False, errors=["Rationale cannot be empty"])

        if len(rationale) > cls.MAX_RATIONALE_LENGTH:
            errors.append(f"Rationale exceeds maximum length of {cls.MAX_RATIONALE_LENGTH}")

        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if pattern.search(rationale):
                errors.append("Input contains potentially dangerous content")
                break

        if errors:
            return ValidationResult(valid=False, errors=errors)

        # Sanitize: normalize whitespace, strip control characters
        sanitized = cls._sanitize_text(rationale)
        return ValidationResult(valid=True, sanitized_input=sanitized)

    @classmethod
    def validate_domain(cls, domain: str) -> ValidationResult:
        """Validate domain input."""
        valid_domains = {
            "trading",
            "finance",
            "defense",
            "cybersecurity",
            "medical",
            "risk_assessment",
            "planning",
            "general",
        }

        if not domain:
            return ValidationResult(valid=True, sanitized_input="general")

        domain_lower = domain.lower().strip()
        if domain_lower not in valid_domains:
            return ValidationResult(
                valid=False,
                errors=[f"Invalid domain. Must be one of: {', '.join(sorted(valid_domains))}"],
            )

        return ValidationResult(valid=True, sanitized_input=domain_lower)

    @classmethod
    def _sanitize_text(cls, text: str) -> str:
        """Sanitize text by removing control characters and normalizing whitespace."""
        # Remove null bytes and other control characters (except newline, tab)
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Normalize multiple spaces/newlines
        sanitized = re.sub(r" +", " ", sanitized)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized.strip()


# ================================================================
# Authentication
# ================================================================


@dataclass
class APIKey:
    """API key with metadata."""

    key_id: str
    key_hash: str
    name: str
    created_at: float
    expires_at: float | None = None
    scopes: list[str] = field(default_factory=list)
    rate_limit: int | None = None


class APIKeyManager:
    """Manages API keys for authentication."""

    def __init__(self) -> None:
        self._keys: dict[str, APIKey] = {}
        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load API keys from environment variables."""
        # Format: MINICRIT_API_KEY_<NAME>=<key>
        for key, value in os.environ.items():
            if key.startswith("MINICRIT_API_KEY_"):
                name = key.replace("MINICRIT_API_KEY_", "").lower()
                self.add_key(value, name)

    def generate_key(self) -> str:
        """Generate a new API key."""
        return f"mc_{secrets.token_urlsafe(32)}"

    def add_key(
        self,
        key: str,
        name: str,
        expires_in_days: int | None = None,
        scopes: list[str] | None = None,
        rate_limit: int | None = None,
    ) -> APIKey:
        """Add an API key."""
        key_id = hashlib.sha256(key.encode()).hexdigest()[:16]
        key_hash = self._hash_key(key)

        expires_at = None
        if expires_in_days:
            expires_at = time.time() + (expires_in_days * 86400)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            created_at=time.time(),
            expires_at=expires_at,
            scopes=scopes or [],
            rate_limit=rate_limit,
        )
        self._keys[key_id] = api_key
        return api_key

    def validate_key(self, key: str) -> tuple[bool, APIKey | None]:
        """Validate an API key."""
        key_id = hashlib.sha256(key.encode()).hexdigest()[:16]
        api_key = self._keys.get(key_id)

        if not api_key:
            return False, None

        # Check expiration
        if api_key.expires_at and time.time() > api_key.expires_at:
            logger.warning(f"Expired API key used: {api_key.name}")
            return False, None

        # Verify hash
        if not self._verify_key(key, api_key.key_hash):
            return False, None

        return True, api_key

    def _hash_key(self, key: str) -> str:
        """Hash an API key using PBKDF2."""
        salt = os.urandom(16)
        key_hash = hashlib.pbkdf2_hmac("sha256", key.encode(), salt, 100000)
        return salt.hex() + ":" + key_hash.hex()

    def _verify_key(self, key: str, stored_hash: str) -> bool:
        """Verify a key against its stored hash."""
        try:
            salt_hex, hash_hex = stored_hash.split(":")
            salt = bytes.fromhex(salt_hex)
            expected_hash = bytes.fromhex(hash_hex)
            actual_hash = hashlib.pbkdf2_hmac("sha256", key.encode(), salt, 100000)
            return hmac.compare_digest(actual_hash, expected_hash)
        except (ValueError, AttributeError):
            return False


# ================================================================
# Request Signing
# ================================================================


class RequestSigner:
    """Sign and verify requests using HMAC."""

    def __init__(self, secret_key: str | None = None) -> None:
        self._secret_key = secret_key or os.environ.get("MINICRIT_SIGNING_KEY", "")
        if not self._secret_key:
            logger.warning("No signing key configured. Request signing disabled.")

    def sign_request(self, payload: str, timestamp: float | None = None) -> str:
        """Sign a request payload."""
        if not self._secret_key:
            return ""

        ts = timestamp or time.time()
        message = f"{int(ts)}:{payload}"
        signature = hmac.new(
            self._secret_key.encode(), message.encode(), hashlib.sha256
        ).hexdigest()
        return f"t={int(ts)},v1={signature}"

    def verify_signature(self, payload: str, signature: str, max_age_seconds: int = 300) -> bool:
        """Verify a request signature."""
        if not self._secret_key:
            return True  # Signing disabled

        try:
            parts = dict(p.split("=") for p in signature.split(","))
            timestamp = int(parts.get("t", 0))
            received_sig = parts.get("v1", "")

            # Check timestamp freshness
            if abs(time.time() - timestamp) > max_age_seconds:
                logger.warning("Signature timestamp too old")
                return False

            # Verify signature
            message = f"{timestamp}:{payload}"
            expected_sig = hmac.new(
                self._secret_key.encode(), message.encode(), hashlib.sha256
            ).hexdigest()

            return hmac.compare_digest(received_sig, expected_sig)
        except (ValueError, KeyError) as e:
            logger.warning(f"Invalid signature format: {e}")
            return False


# ================================================================
# Security Headers
# ================================================================


SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": "default-src 'self'",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
}


def add_security_headers(headers: dict[str, str]) -> dict[str, str]:
    """Add security headers to response."""
    return {**SECURITY_HEADERS, **headers}


# ================================================================
# Audit Logging
# ================================================================


@dataclass
class AuditEvent:
    """Audit log event."""

    timestamp: float
    event_type: str
    user_id: str | None
    ip_address: str | None
    action: str
    resource: str | None
    status: str
    details: dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    """Logs security-relevant events."""

    def __init__(self, logger_name: str = "minicrit.audit") -> None:
        self._logger = logging.getLogger(logger_name)

    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event."""
        log_data = {
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "user_id": event.user_id,
            "ip_address": event.ip_address,
            "action": event.action,
            "resource": event.resource,
            "status": event.status,
            **event.details,
        }
        self._logger.info(f"AUDIT: {log_data}")

    def log_authentication(
        self,
        success: bool,
        user_id: str | None = None,
        ip_address: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Log authentication attempt."""
        event = AuditEvent(
            timestamp=time.time(),
            event_type="authentication",
            user_id=user_id,
            ip_address=ip_address,
            action="login",
            resource=None,
            status="success" if success else "failure",
            details={"reason": reason} if reason else {},
        )
        self.log_event(event)

    def log_api_call(
        self,
        endpoint: str,
        method: str,
        user_id: str | None = None,
        ip_address: str | None = None,
        status_code: int = 200,
        latency_ms: float | None = None,
    ) -> None:
        """Log API call."""
        event = AuditEvent(
            timestamp=time.time(),
            event_type="api_call",
            user_id=user_id,
            ip_address=ip_address,
            action=method,
            resource=endpoint,
            status="success" if status_code < 400 else "failure",
            details={"status_code": status_code, "latency_ms": latency_ms},
        )
        self.log_event(event)


# ================================================================
# Decorator for Protected Endpoints
# ================================================================


def require_auth(scopes: list[str] | None = None) -> Callable:
    """Decorator to require authentication for an endpoint."""
    key_manager = APIKeyManager()
    audit_logger = AuditLogger()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract API key from kwargs or headers
            api_key = kwargs.pop("api_key", None)
            ip_address = kwargs.pop("ip_address", None)

            if not api_key:
                audit_logger.log_authentication(False, ip_address=ip_address, reason="missing_key")
                raise PermissionError("API key required")

            valid, key_info = key_manager.validate_key(api_key)
            if not valid:
                audit_logger.log_authentication(False, ip_address=ip_address, reason="invalid_key")
                raise PermissionError("Invalid API key")

            # Check scopes
            if scopes and key_info:
                if not all(s in key_info.scopes for s in scopes):
                    audit_logger.log_authentication(
                        False,
                        user_id=key_info.name,
                        ip_address=ip_address,
                        reason="insufficient_scopes",
                    )
                    raise PermissionError("Insufficient permissions")

            audit_logger.log_authentication(
                True, user_id=key_info.name if key_info else None, ip_address=ip_address
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ================================================================
# Exports
# ================================================================

__all__ = [
    "ValidationResult",
    "InputValidator",
    "APIKey",
    "APIKeyManager",
    "RequestSigner",
    "SECURITY_HEADERS",
    "add_security_headers",
    "AuditEvent",
    "AuditLogger",
    "require_auth",
]
