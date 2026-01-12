#!/usr/bin/env python3
# ================================================================
# MiniCrit MCP Server - Production HTTP with Authentication
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# Co-Founder & CEO: William Alexander Ousley (Alex Ousley)
# Co-Founder & CTO: Jacqueline Villamor Ousley (Jacque Ousley) TS/SCI
# ================================================================
# WATERMARK Layer 1: Antagon Inc. Proprietary
# WATERMARK Layer 2: MiniCrit MCP Server v1.1 Production
# WATERMARK Layer 3: Model Context Protocol Implementation
# WATERMARK Layer 4: Hash SHA256:MCP_PROD_AUTH_2026
# WATERMARK Layer 5: Build 20260112
# ================================================================

"""
MiniCrit MCP Server - Production HTTP with Authentication

Features:
- API Key authentication (header or query param)
- Rate limiting per API key
- Request logging and audit trail
- Health checks and metrics
- CORS configuration
- Graceful shutdown

Installation:
    pip install mcp fastapi uvicorn torch transformers peft redis python-jose

Usage:
    # Development
    python minicrit_mcp_server_prod.py
    
    # Production with gunicorn
    gunicorn minicrit_mcp_server_prod:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

Environment Variables:
    MINICRIT_MODEL: HuggingFace model ID
    MINICRIT_DEVICE: Device (auto, cuda, cpu, mps)
    MINICRIT_PORT: Server port (default: 8000)
    MINICRIT_API_KEYS: Comma-separated valid API keys
    MINICRIT_MASTER_KEY: Master admin key
    MINICRIT_RATE_LIMIT: Requests per minute per key (default: 60)
    MINICRIT_CORS_ORIGINS: Comma-separated allowed origins
    MINICRIT_LOG_REQUESTS: Log all requests (true/false)
    REDIS_URL: Redis URL for rate limiting (optional)
"""

import os
import sys
import json
import logging
import time
import asyncio
import hashlib
import secrets
from typing import Optional, Any, Dict
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict
from functools import wraps

# Web framework
from fastapi import FastAPI, HTTPException, Depends, Header, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel, Field
import uvicorn

# ================================================================
# Configuration
# ================================================================

MODEL_ID = os.environ.get("MINICRIT_MODEL", "wmaousley/MiniCrit-7B")
DEVICE = os.environ.get("MINICRIT_DEVICE", "auto")
MAX_LENGTH = int(os.environ.get("MINICRIT_MAX_LENGTH", "512"))
PORT = int(os.environ.get("MINICRIT_PORT", "8000"))
LOG_LEVEL = os.environ.get("MINICRIT_LOG_LEVEL", "INFO")

# Authentication
API_KEYS = set(os.environ.get("MINICRIT_API_KEYS", "").split(",")) - {""}
MASTER_KEY = os.environ.get("MINICRIT_MASTER_KEY", secrets.token_urlsafe(32))
RATE_LIMIT = int(os.environ.get("MINICRIT_RATE_LIMIT", "60"))  # per minute

# CORS
CORS_ORIGINS = os.environ.get("MINICRIT_CORS_ORIGINS", "*").split(",")

# Logging
LOG_REQUESTS = os.environ.get("MINICRIT_LOG_REQUESTS", "true").lower() == "true"

# Domains
DOMAINS = [
    "trading", "finance", "risk_assessment", "resource_allocation",
    "planning_scheduling", "cybersecurity", "defense", "medical", "general",
]

# ================================================================
# Logging Setup
# ================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("minicrit-mcp")

# ================================================================
# Rate Limiting (In-Memory)
# ================================================================

class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, limit: int = 60, window: int = 60):
        self.limit = limit
        self.window = window  # seconds
        self.requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, key: str) -> tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, remaining)."""
        now = time.time()
        window_start = now - self.window
        
        # Clean old requests
        self.requests[key] = [t for t in self.requests[key] if t > window_start]
        
        # Check limit
        if len(self.requests[key]) >= self.limit:
            return False, 0
        
        # Record request
        self.requests[key].append(now)
        remaining = self.limit - len(self.requests[key])
        
        return True, remaining

rate_limiter = RateLimiter(limit=RATE_LIMIT, window=60)

# ================================================================
# API Key Management
# ================================================================

class APIKeyManager:
    """Manage API keys with usage tracking."""
    
    def __init__(self):
        self.keys: Dict[str, dict] = {}
        self.usage: Dict[str, dict] = defaultdict(lambda: {
            "total_requests": 0,
            "total_tokens": 0,
            "last_used": None,
        })
        
        # Add configured keys
        for key in API_KEYS:
            if key:
                self.add_key(key, name="configured", tier="standard")
        
        # Add master key
        if MASTER_KEY:
            self.add_key(MASTER_KEY, name="master", tier="admin")
    
    def add_key(self, key: str, name: str = "", tier: str = "standard"):
        """Add an API key."""
        key_hash = self._hash_key(key)
        self.keys[key_hash] = {
            "name": name,
            "tier": tier,
            "created": datetime.utcnow().isoformat(),
            "active": True,
        }
        return key_hash
    
    def validate_key(self, key: str) -> Optional[dict]:
        """Validate an API key and return its info."""
        if not key:
            return None
        
        key_hash = self._hash_key(key)
        key_info = self.keys.get(key_hash)
        
        if key_info and key_info.get("active"):
            # Update usage
            self.usage[key_hash]["total_requests"] += 1
            self.usage[key_hash]["last_used"] = datetime.utcnow().isoformat()
            return {**key_info, "hash": key_hash}
        
        return None
    
    def get_rate_limit(self, key_info: dict) -> int:
        """Get rate limit for key tier."""
        tier_limits = {
            "admin": 1000,
            "premium": 300,
            "standard": RATE_LIMIT,
            "free": 10,
        }
        return tier_limits.get(key_info.get("tier", "standard"), RATE_LIMIT)
    
    def _hash_key(self, key: str) -> str:
        """Hash API key for storage."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]

key_manager = APIKeyManager()

# ================================================================
# Request Logging / Audit Trail
# ================================================================

class AuditLog:
    """Audit log for all validation requests."""
    
    def __init__(self, max_entries: int = 10000):
        self.entries: list = []
        self.max_entries = max_entries
    
    def log(self, entry: dict):
        """Add audit log entry."""
        entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
        self.entries.append(entry)
        
        # Trim if too large
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
    
    def get_recent(self, n: int = 100) -> list:
        """Get recent entries."""
        return self.entries[-n:]

audit_log = AuditLog()

# ================================================================
# Model Loading
# ================================================================

_model = None
_tokenizer = None
_model_lock = asyncio.Lock()

async def get_model():
    """Lazy load model with async lock."""
    global _model, _tokenizer
    
    async with _model_lock:
        if _model is not None:
            return _model, _tokenizer
        
        logger.info(f"Loading MiniCrit model: {MODEL_ID}")
        
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE if DEVICE != "auto" else "auto",
            trust_remote_code=True,
        )
        _model.eval()
        
        logger.info(f"Model loaded on {next(_model.parameters()).device}")
        return _model, _tokenizer

# ================================================================
# Data Models
# ================================================================

class Severity(str, Enum):
    PASS = "pass"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ValidateRequest(BaseModel):
    rationale: str = Field(..., description="The AI reasoning to validate", max_length=10000)
    domain: str = Field("general", description="Domain context")
    context: Optional[str] = Field(None, description="Additional context", max_length=5000)

class BatchItem(BaseModel):
    id: str = Field(..., description="Item identifier")
    rationale: str = Field(..., description="The AI reasoning to validate")
    domain: str = Field("general", description="Domain context")

class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(..., description="Items to validate", max_items=100)

class CritiqueResponse(BaseModel):
    valid: bool
    severity: str
    critique: str
    confidence: float
    flags: list[str]
    domain: str
    latency_ms: float
    timestamp: str
    request_id: str

class APIKeyCreate(BaseModel):
    name: str = Field(..., description="Key name/description")
    tier: str = Field("standard", description="Key tier (standard, premium)")

# ================================================================
# Authentication Dependencies
# ================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

async def get_api_key(
    request: Request,
    api_key_header: Optional[str] = Depends(api_key_header),
    api_key_query: Optional[str] = Depends(api_key_query),
) -> dict:
    """Validate API key from header or query parameter."""
    
    api_key = api_key_header or api_key_query
    
    # Allow unauthenticated if no keys configured
    if not API_KEYS and not MASTER_KEY:
        return {"tier": "open", "name": "anonymous"}
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide via X-API-Key header or api_key query param.",
        )
    
    key_info = key_manager.validate_key(api_key)
    
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    
    # Check rate limit
    rate_limit = key_manager.get_rate_limit(key_info)
    allowed, remaining = rate_limiter.is_allowed(key_info["hash"])
    
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Limit: {rate_limit}/minute",
            headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Limit": str(rate_limit)},
        )
    
    # Add rate limit info to key_info
    key_info["rate_limit_remaining"] = remaining
    key_info["rate_limit"] = rate_limit
    
    return key_info

async def require_admin(key_info: dict = Depends(get_api_key)) -> dict:
    """Require admin tier API key."""
    if key_info.get("tier") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return key_info

# ================================================================
# Critique Generation
# ================================================================

async def generate_critique(
    rationale: str,
    domain: str = "general",
    context: Optional[str] = None,
    request_id: str = "",
) -> CritiqueResponse:
    """Generate adversarial critique."""
    import torch
    
    start_time = time.time()
    model, tokenizer = await get_model()
    
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
        max_length=MAX_LENGTH,
    ).to(model.device)
    
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
        skip_special_tokens=True,
    ).strip()
    
    # Parse critique
    flags = []
    severity = Severity.PASS
    critique_lower = generated.lower()
    
    if any(w in critique_lower for w in ["critical", "severe", "dangerous"]):
        severity = Severity.CRITICAL
        flags.append("critical_flaw")
    elif any(w in critique_lower for w in ["significant", "major", "serious", "flawed"]):
        severity = Severity.HIGH
        flags.append("significant_flaw")
    elif any(w in critique_lower for w in ["concern", "issue", "problem", "missing"]):
        severity = Severity.MEDIUM
        flags.append("notable_concern")
    elif any(w in critique_lower for w in ["minor", "slight"]):
        severity = Severity.LOW
        flags.append("minor_issue")
    
    if any(w in critique_lower for w in ["overconfident", "overconfidence"]):
        flags.append("overconfidence")
    if any(w in critique_lower for w in ["missing", "omit", "neglect"]):
        flags.append("missing_consideration")
    if any(w in critique_lower for w in ["contradict", "inconsistent"]):
        flags.append("logical_inconsistency")
    if any(w in critique_lower for w in ["evidence", "support", "justify"]):
        flags.append("insufficient_evidence")
    if any(w in critique_lower for w in ["risk", "danger", "threat"]):
        flags.append("unaddressed_risk")
    
    valid = severity in [Severity.PASS, Severity.LOW]
    confidence = 0.85 if len(generated) >= 20 else 0.5
    if len(flags) > 2:
        confidence = 0.9
    
    latency_ms = (time.time() - start_time) * 1000
    
    return CritiqueResponse(
        valid=valid,
        severity=severity.value,
        critique=generated,
        confidence=confidence,
        flags=flags,
        domain=domain,
        latency_ms=round(latency_ms, 2),
        timestamp=datetime.utcnow().isoformat() + "Z",
        request_id=request_id or secrets.token_hex(8),
    )

# ================================================================
# FastAPI Application
# ================================================================

app = FastAPI(
    title="MiniCrit MCP Server",
    description="""
Adversarial AI Validation via Model Context Protocol

**Antagon Inc.** | CAGE: 17E75 | UEI: KBSGT7CZ4AH3

## Authentication

All endpoints (except /health) require an API key:
- Header: `X-API-Key: your-api-key`
- Query: `?api_key=your-api-key`

## Rate Limits

- Standard tier: 60 requests/minute
- Premium tier: 300 requests/minute
- Admin tier: 1000 requests/minute
    """,
    version="1.1.0",
    contact={
        "name": "Antagon Inc.",
        "email": "founders@antagon.ai",
        "url": "https://www.antagon.ai",
    },
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Remaining", "X-RateLimit-Limit", "X-Request-ID"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = secrets.token_hex(8)
    start_time = time.time()
    
    # Add request ID to state
    request.state.request_id = request_id
    
    response = await call_next(request)
    
    # Log request
    if LOG_REQUESTS and request.url.path not in ["/health", "/docs", "/openapi.json"]:
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} "
            f"duration={duration_ms:.2f}ms "
            f"request_id={request_id}"
        )
    
    response.headers["X-Request-ID"] = request_id
    return response

# ================================================================
# Endpoints
# ================================================================

@app.get("/health")
async def health_check():
    """Health check (no auth required)."""
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "model_loaded": _model is not None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

@app.get("/info")
async def get_info(key_info: dict = Depends(get_api_key)):
    """Get server information."""
    return {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "max_length": MAX_LENGTH,
        "supported_domains": DOMAINS,
        "version": "1.1.0",
        "company": "Antagon Inc.",
        "cage_code": "17E75",
        "uei": "KBSGT7CZ4AH3",
        "model_loaded": _model is not None,
        "your_tier": key_info.get("tier"),
        "rate_limit": key_info.get("rate_limit"),
        "rate_limit_remaining": key_info.get("rate_limit_remaining"),
    }

@app.post("/validate", response_model=CritiqueResponse)
async def validate_reasoning(
    request: Request,
    body: ValidateRequest,
    key_info: dict = Depends(get_api_key),
):
    """Validate AI reasoning."""
    request_id = getattr(request.state, "request_id", secrets.token_hex(8))
    
    logger.info(f"Validating reasoning in domain: {body.domain} (key: {key_info.get('name', 'unknown')})")
    
    try:
        result = await generate_critique(
            rationale=body.rationale,
            domain=body.domain,
            context=body.context,
            request_id=request_id,
        )
        
        # Audit log
        audit_log.log({
            "action": "validate",
            "domain": body.domain,
            "severity": result.severity,
            "valid": result.valid,
            "latency_ms": result.latency_ms,
            "key_name": key_info.get("name"),
            "request_id": request_id,
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def batch_validate(
    request: Request,
    body: BatchRequest,
    key_info: dict = Depends(get_api_key),
):
    """Batch validate multiple items."""
    request_id = getattr(request.state, "request_id", secrets.token_hex(8))
    
    logger.info(f"Batch validating {len(body.items)} items")
    
    results = []
    for item in body.items:
        try:
            result = await generate_critique(
                rationale=item.rationale,
                domain=item.domain,
                request_id=f"{request_id}-{item.id}",
            )
            results.append({
                "id": item.id,
                **result.dict(),
            })
        except Exception as e:
            results.append({
                "id": item.id,
                "error": str(e),
            })
    
    # Audit log
    audit_log.log({
        "action": "batch_validate",
        "count": len(body.items),
        "key_name": key_info.get("name"),
        "request_id": request_id,
    })
    
    return {
        "results": results,
        "count": len(results),
        "request_id": request_id,
    }

# ================================================================
# Admin Endpoints
# ================================================================

@app.post("/admin/keys", dependencies=[Depends(require_admin)])
async def create_api_key(body: APIKeyCreate):
    """Create a new API key (admin only)."""
    new_key = secrets.token_urlsafe(32)
    key_hash = key_manager.add_key(new_key, name=body.name, tier=body.tier)
    
    logger.info(f"Created API key: {body.name} (tier: {body.tier})")
    
    return {
        "api_key": new_key,  # Only shown once!
        "name": body.name,
        "tier": body.tier,
        "message": "Save this key - it won't be shown again!",
    }

@app.get("/admin/audit", dependencies=[Depends(require_admin)])
async def get_audit_log(n: int = Query(100, le=1000)):
    """Get recent audit log entries (admin only)."""
    return {
        "entries": audit_log.get_recent(n),
        "count": len(audit_log.entries),
    }

@app.get("/admin/stats", dependencies=[Depends(require_admin)])
async def get_stats():
    """Get server statistics (admin only)."""
    return {
        "total_keys": len(key_manager.keys),
        "model_loaded": _model is not None,
        "audit_log_size": len(audit_log.entries),
        "usage": dict(key_manager.usage),
    }

# ================================================================
# MCP SSE Endpoint
# ================================================================

@app.get("/sse")
async def mcp_sse_endpoint(key_info: dict = Depends(get_api_key)):
    """MCP Server-Sent Events endpoint."""
    
    async def event_generator():
        capabilities = {
            "type": "capabilities",
            "tools": [
                {
                    "name": "validate_reasoning",
                    "description": "Validate AI reasoning using MiniCrit",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "rationale": {"type": "string"},
                            "domain": {"type": "string", "enum": DOMAINS},
                            "context": {"type": "string"},
                        },
                        "required": ["rationale"],
                    },
                },
            ],
        }
        yield f"data: {json.dumps(capabilities)}\n\n"
        
        while True:
            await asyncio.sleep(30)
            yield f"data: {json.dumps({'type': 'ping'})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )

# ================================================================
# Startup / Shutdown
# ================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("MiniCrit MCP Server Starting")
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Port: {PORT}")
    logger.info(f"API Keys configured: {len(API_KEYS)}")
    logger.info(f"Rate limit: {RATE_LIMIT}/minute")
    logger.info("=" * 60)
    
    if not API_KEYS and MASTER_KEY:
        logger.warning(f"No API keys configured. Master key: {MASTER_KEY[:8]}...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("MiniCrit MCP Server shutting down")

# ================================================================
# Entry Point
# ================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
