# ================================================================
# MiniCrit MCP Integration
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# Co-Founder & CEO: William Alexander Ousley (Alex Ousley)
# Co-Founder & CTO: Jacqueline Villamor Ousley (Jacque Ousley) TS/SCI
# ================================================================
# WATERMARK Layer 1: Antagon Inc. Proprietary
# WATERMARK Layer 2: MiniCrit MCP Module
# WATERMARK Layer 3: Model Context Protocol
# WATERMARK Layer 4: Hash SHA256:MCP_INIT_2026
# WATERMARK Layer 5: Build 20260112
# ================================================================

"""
MiniCrit MCP (Model Context Protocol) Integration

Servers:
- server.py: Local stdio server for Claude Desktop
- server_prod.py: Production HTTP server with auth/rate limiting
- server_http.py: Basic HTTP server

Core Module:
- core.py: Thread-safe ModelManager, CritiqueGenerator, RateLimiter

Usage:
    # Local (Claude Desktop)
    python -m src.mcp.server

    # Production HTTP
    python -m src.mcp.server_http

    # Production with auth
    python -m src.mcp.server_prod

Programmatic Usage:
    from src.mcp import ModelManager, CritiqueGenerator, CritiqueResult

    manager = ModelManager.get_instance()
    generator = CritiqueGenerator(manager)
    result = generator.generate("AI reasoning to validate", domain="trading")
"""

from .core import (
    # Constants
    DOMAINS,
    CritiqueGenerator,
    CritiqueResult,
    GracefulShutdown,
    InferenceError,
    InferenceTimeoutError,
    InputSanitizationError,
    InputSanitizer,
    InvalidInputError,
    ModelLoadError,
    # Classes
    ModelManager,
    # Exceptions
    ModelNotLoadedError,
    RateLimiter,
    Severity,
    get_cors_origins,
    get_critique_generator,
    get_input_sanitizer,
    # Functions
    get_model_manager,
)

__all__ = [
    # Classes
    "ModelManager",
    "CritiqueGenerator",
    "CritiqueResult",
    "RateLimiter",
    "GracefulShutdown",
    "Severity",
    "InputSanitizer",
    # Exceptions
    "ModelNotLoadedError",
    "ModelLoadError",
    "InferenceTimeoutError",
    "InferenceError",
    "InvalidInputError",
    "InputSanitizationError",
    # Functions
    "get_model_manager",
    "get_critique_generator",
    "get_cors_origins",
    "get_input_sanitizer",
    # Constants
    "DOMAINS",
]
