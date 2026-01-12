#!/usr/bin/env python3
# ================================================================
# MiniCrit MCP Server - HTTP/SSE Transport (Remote Deployment)
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# Co-Founder & CEO: William Alexander Ousley (Alex Ousley)
# Co-Founder & CTO: Jacqueline Villamor Ousley (Jacque Ousley) TS/SCI
# ================================================================
# WATERMARK Layer 1: Antagon Inc. Proprietary
# WATERMARK Layer 2: MiniCrit MCP Server v1.0 HTTP
# WATERMARK Layer 3: Model Context Protocol Implementation
# WATERMARK Layer 4: Hash SHA256:MCP_HTTP_2026
# WATERMARK Layer 5: Build 20260112
# ================================================================

"""
MiniCrit MCP Server - HTTP/SSE Transport

Remote-deployable version using HTTP with Server-Sent Events (SSE).
Deploy to cloud infrastructure for organization-wide access.

Installation:
    pip install mcp fastapi uvicorn torch transformers peft

Usage:
    python minicrit_mcp_server_http.py
    
    # Or with uvicorn for production
    uvicorn minicrit_mcp_server_http:app --host 0.0.0.0 --port 8000

Environment Variables:
    MINICRIT_MODEL: HuggingFace model ID (default: wmaousley/MiniCrit-7B)
    MINICRIT_DEVICE: Device (auto, cuda, cpu, mps)
    MINICRIT_API_KEY: API key for authentication (optional)
    MINICRIT_PORT: Server port (default: 8000)
"""

import os
import sys
import json
import logging
import time
import asyncio
from typing import Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime

# Web framework
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# ================================================================
# Configuration
# ================================================================

MODEL_ID = os.environ.get("MINICRIT_MODEL", "wmaousley/MiniCrit-7B")
DEVICE = os.environ.get("MINICRIT_DEVICE", "auto")
MAX_LENGTH = int(os.environ.get("MINICRIT_MAX_LENGTH", "512"))
API_KEY = os.environ.get("MINICRIT_API_KEY", None)
PORT = int(os.environ.get("MINICRIT_PORT", "8000"))
LOG_LEVEL = os.environ.get("MINICRIT_LOG_LEVEL", "INFO")

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
logger = logging.getLogger("minicrit-mcp-http")

# ================================================================
# Model Loading (Lazy)
# ================================================================

_model = None
_tokenizer = None
_model_lock = asyncio.Lock()

async def get_model():
    """Lazy load the MiniCrit model with async lock."""
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
        
        device_map = DEVICE if DEVICE != "auto" else "auto"
        
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
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
    rationale: str = Field(..., description="The AI reasoning to validate")
    domain: str = Field("general", description="Domain context")
    context: Optional[str] = Field(None, description="Additional context")

class BatchItem(BaseModel):
    id: str = Field(..., description="Item identifier")
    rationale: str = Field(..., description="The AI reasoning to validate")
    domain: str = Field("general", description="Domain context")

class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(..., description="Items to validate")

class CritiqueResponse(BaseModel):
    valid: bool
    severity: str
    critique: str
    confidence: float
    flags: list[str]
    domain: str
    latency_ms: float
    timestamp: str

# ================================================================
# Critique Generation
# ================================================================

async def generate_critique(
    rationale: str,
    domain: str = "general",
    context: Optional[str] = None,
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
    
    # Specific flags
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
    )

# ================================================================
# FastAPI Application
# ================================================================

app = FastAPI(
    title="MiniCrit MCP Server",
    description="Adversarial AI Validation via Model Context Protocol",
    version="1.0.0",
    contact={
        "name": "Antagon Inc.",
        "email": "founders@antagon.ai",
        "url": "https://www.antagon.ai",
    },
)

# CORS for browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication (optional)
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# ================================================================
# Endpoints
# ================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "model_loaded": _model is not None,
    }

@app.get("/info")
async def get_info():
    """Get server information."""
    return {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "max_length": MAX_LENGTH,
        "supported_domains": DOMAINS,
        "version": "1.0.0",
        "company": "Antagon Inc.",
        "cage_code": "17E75",
        "uei": "KBSGT7CZ4AH3",
        "model_loaded": _model is not None,
    }

@app.post("/validate", response_model=CritiqueResponse)
async def validate_reasoning(
    request: ValidateRequest,
    authorized: bool = Depends(verify_api_key),
):
    """Validate AI reasoning."""
    logger.info(f"Validating reasoning in domain: {request.domain}")
    
    try:
        result = await generate_critique(
            rationale=request.rationale,
            domain=request.domain,
            context=request.context,
        )
        return result
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def batch_validate(
    request: BatchRequest,
    authorized: bool = Depends(verify_api_key),
):
    """Batch validate multiple items."""
    logger.info(f"Batch validating {len(request.items)} items")
    
    results = []
    for item in request.items:
        try:
            result = await generate_critique(
                rationale=item.rationale,
                domain=item.domain,
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
    
    return {"results": results, "count": len(results)}

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
                            "rationale": {"type": "string"},
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
                            "items": {"type": "array"},
                        },
                        "required": ["items"],
                    },
                },
            ],
        }
        yield f"data: {json.dumps(capabilities)}\n\n"
        
        # Keep connection alive
        while True:
            await asyncio.sleep(30)
            yield f"data: {json.dumps({'type': 'ping'})}\n\n"
    
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
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"Device: {DEVICE}")
    
    uvicorn.run(app, host="0.0.0.0", port=PORT)
