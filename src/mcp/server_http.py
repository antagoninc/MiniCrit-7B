#!/usr/bin/env python3
# ================================================================
# MiniCrit MCP Server - HTTP/SSE Transport (Remote Deployment)
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# Co-Founder & CEO: William Alexander Ousley (Alex Ousley)
# Co-Founder & CTO: Jacqueline Villamor Ousley (Jacque Ousley) TS/SCI
# ================================================================
# WATERMARK Layer 1: Antagon Inc. Proprietary
# WATERMARK Layer 2: MiniCrit MCP Server v1.2 HTTP
# WATERMARK Layer 3: Model Context Protocol Implementation
# WATERMARK Layer 4: Hash SHA256:MCP_HTTP_2026
# WATERMARK Layer 5: Build 20260112
# ================================================================

"""
MiniCrit MCP Server - HTTP/SSE Transport

Remote-deployable version using HTTP with Server-Sent Events (SSE).
Deploy to cloud infrastructure for organization-wide access.
Uses thread-safe ModelManager and CritiqueGenerator from core module.

Installation:
    pip install mcp fastapi uvicorn torch transformers peft

Usage:
    python server_http.py

    # Or with uvicorn for production
    uvicorn src.mcp.server_http:app --host 0.0.0.0 --port 8000

Environment Variables:
    MINICRIT_ADAPTER: LoRA adapter (default: wmaousley/MiniCrit-7B)
    MINICRIT_BASE_MODEL: Base model (default: Qwen/Qwen2-7B-Instruct)
    MINICRIT_DEVICE: Device (auto, cuda, cpu, mps)
    MINICRIT_API_KEY: API key for authentication (optional)
    MINICRIT_PORT: Server port (default: 8000)
    MINICRIT_CORS_ORIGINS: Comma-separated allowed origins
    MINICRIT_PRELOAD: Set to "true" to preload model on startup
"""

import os
import json
import logging
import asyncio
import secrets
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

# Web framework
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import from core module
from src.mcp.core import (
    ModelManager,
    CritiqueGenerator,
    CritiqueResult,
    RateLimiter,
    GracefulShutdown,
    ModelLoadError,
    InferenceTimeoutError,
    InferenceError,
    InvalidInputError,
    DOMAINS,
    ADAPTER_ID,
    BASE_MODEL_ID,
    DEVICE,
    LOG_LEVEL,
    get_cors_origins,
)

# ================================================================
# Configuration
# ================================================================

API_KEY = os.environ.get("MINICRIT_API_KEY", None)
PORT = int(os.environ.get("MINICRIT_PORT", "8000"))
PRELOAD_MODEL = os.environ.get("MINICRIT_PRELOAD", "false").lower() == "true"
CORS_ORIGINS = get_cors_origins()

# ================================================================
# Logging Setup
# ================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("minicrit-mcp-http")

# ================================================================
# Global Instances (Thread-Safe)
# ================================================================

model_manager = ModelManager.get_instance()
critique_generator = CritiqueGenerator(model_manager)
shutdown_handler = GracefulShutdown(model_manager)

# ================================================================
# Data Models
# ================================================================


class ValidateRequest(BaseModel):
    """Request model for validation."""

    rationale: str = Field(
        ...,
        description="The AI reasoning to validate",
        min_length=10,
        max_length=10000,
    )
    domain: str = Field("general", description="Domain context")
    context: Optional[str] = Field(None, description="Additional context", max_length=5000)


class BatchItem(BaseModel):
    """Single item in batch request."""

    id: str = Field(..., description="Item identifier")
    rationale: str = Field(..., description="The AI reasoning to validate")
    domain: str = Field("general", description="Domain context")


class BatchRequest(BaseModel):
    """Batch validation request."""

    items: list[BatchItem] = Field(
        ...,
        description="Items to validate",
        max_length=100,
    )


class CritiqueResponse(BaseModel):
    """Response model for critique."""

    valid: bool
    severity: str
    critique: str
    confidence: float
    flags: list[str]
    domain: str
    latency_ms: float
    timestamp: str
    request_id: str = ""


# ================================================================
# Lifespan Management
# ================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("Starting MiniCrit MCP HTTP Server...")
    logger.info(f"Base model: {BASE_MODEL_ID}")
    logger.info(f"Adapter: {ADAPTER_ID}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"CORS origins: {CORS_ORIGINS}")

    # Register graceful shutdown
    shutdown_handler.register()

    # Preload model if configured
    if PRELOAD_MODEL:
        logger.info("Preloading model (MINICRIT_PRELOAD=true)...")
        try:
            await model_manager.preload_async()
            logger.info("Model preloaded successfully")
        except ModelLoadError as e:
            logger.error(f"Failed to preload model: {e}")

    yield

    logger.info("Shutting down MiniCrit MCP HTTP Server...")
    model_manager.unload()


# ================================================================
# FastAPI Application
# ================================================================

app = FastAPI(
    title="MiniCrit MCP Server",
    description="Adversarial AI Validation via Model Context Protocol",
    version="1.2.0",
    contact={
        "name": "Antagon Inc.",
        "email": "founders@antagon.ai",
        "url": "https://www.antagon.ai",
    },
    lifespan=lifespan,
)

# CORS middleware with configurable origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)


# ================================================================
# Authentication
# ================================================================


async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key if configured."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "API-Key"},
        )
    return True


# ================================================================
# Request Logging Middleware
# ================================================================


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID and log requests."""
    request_id = secrets.token_hex(8)
    request.state.request_id = request_id

    response = await call_next(request)

    response.headers["X-Request-ID"] = request_id
    return response


# ================================================================
# Endpoints
# ================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": ADAPTER_ID,
        "base_model": BASE_MODEL_ID,
        "model_loaded": model_manager.is_loaded,
        "device": model_manager.device,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/info")
async def get_info():
    """Get server information."""
    return {
        "adapter": ADAPTER_ID,
        "base_model": BASE_MODEL_ID,
        "device": model_manager.device or DEVICE,
        "max_length": 512,
        "supported_domains": DOMAINS,
        "version": "1.2.0",
        "company": "Antagon Inc.",
        "cage_code": "17E75",
        "uei": "KBSGT7CZ4AH3",
        "model_loaded": model_manager.is_loaded,
        "stats": model_manager.stats,
    }


@app.post("/validate", response_model=CritiqueResponse)
async def validate_reasoning(
    request: Request,
    body: ValidateRequest,
    authorized: bool = Header(None, alias="X-API-Key"),
):
    """Validate AI reasoning."""
    # Verify API key if configured
    if API_KEY and authorized != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    request_id = getattr(request.state, "request_id", secrets.token_hex(8))
    logger.info(f"Validating reasoning in domain: {body.domain} (request_id={request_id})")

    try:
        result = await critique_generator.generate_async(
            rationale=body.rationale,
            domain=body.domain,
            context=body.context,
            request_id=request_id,
        )

        return CritiqueResponse(
            valid=result.valid,
            severity=result.severity,
            critique=result.critique,
            confidence=result.confidence,
            flags=result.flags,
            domain=result.domain,
            latency_ms=result.latency_ms,
            timestamp=datetime.utcnow().isoformat() + "Z",
            request_id=request_id,
        )

    except InvalidInputError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except ModelLoadError as e:
        logger.error(f"Model load error: {e}")
        raise HTTPException(status_code=503, detail=f"Model unavailable: {e}")

    except InferenceTimeoutError as e:
        logger.error(f"Inference timeout: {e}")
        raise HTTPException(status_code=504, detail=str(e))

    except InferenceError as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    except MemoryError as e:
        logger.error(f"Memory error: {e}")
        raise HTTPException(status_code=503, detail="Server out of memory")


@app.post("/batch")
async def batch_validate(
    request: Request,
    body: BatchRequest,
    authorized: bool = Header(None, alias="X-API-Key"),
):
    """Batch validate multiple items."""
    # Verify API key if configured
    if API_KEY and authorized != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    request_id = getattr(request.state, "request_id", secrets.token_hex(8))
    logger.info(f"Batch validating {len(body.items)} items (request_id={request_id})")

    results = []
    for item in body.items:
        try:
            result = await critique_generator.generate_async(
                rationale=item.rationale,
                domain=item.domain,
                request_id=f"{request_id}-{item.id}",
            )
            results.append(
                {
                    "id": item.id,
                    "valid": result.valid,
                    "severity": result.severity,
                    "critique": result.critique,
                    "confidence": result.confidence,
                    "flags": result.flags,
                    "domain": result.domain,
                    "latency_ms": result.latency_ms,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            )

        except (InvalidInputError, ModelLoadError, InferenceTimeoutError, InferenceError) as e:
            results.append(
                {
                    "id": item.id,
                    "error": str(e),
                    "error_code": type(e).__name__,
                }
            )

        except MemoryError:
            results.append(
                {
                    "id": item.id,
                    "error": "Out of memory",
                    "error_code": "MemoryError",
                }
            )

    return {
        "results": results,
        "count": len(results),
        "request_id": request_id,
    }


# ================================================================
# MCP SSE Endpoint
# ================================================================


@app.get("/sse")
async def mcp_sse_endpoint():
    """MCP Server-Sent Events endpoint for remote MCP clients."""

    async def event_generator():
        # Send initial capabilities
        capabilities = {
            "type": "capabilities",
            "tools": [
                {
                    "name": "validate_reasoning",
                    "description": "Validate AI reasoning using MiniCrit",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "rationale": {"type": "string", "minLength": 10},
                            "domain": {"type": "string", "enum": DOMAINS},
                            "context": {"type": "string"},
                        },
                        "required": ["rationale"],
                    },
                },
                {
                    "name": "batch_validate",
                    "description": "Batch validate multiple items",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "items": {"type": "array", "maxItems": 100},
                        },
                        "required": ["items"],
                    },
                },
                {
                    "name": "get_model_info",
                    "description": "Get model information",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ],
        }
        yield f"data: {json.dumps(capabilities)}\n\n"

        # Keep connection alive
        while True:
            await asyncio.sleep(30)
            yield f"data: {json.dumps({'type': 'ping', 'timestamp': datetime.utcnow().isoformat()})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ================================================================
# Entry Point
# ================================================================

if __name__ == "__main__":
    logger.info(f"Starting MiniCrit MCP Server (HTTP) on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
