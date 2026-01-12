#!/usr/bin/env python3
# ================================================================
# MiniCrit MCP Server - LoRA Adapter Version
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# Co-Founder & CEO: William Alexander Ousley (Alex Ousley)
# Co-Founder & CTO: Jacqueline Villamor Ousley (Jacque Ousley) TS/SCI
# ================================================================
# WATERMARK Layer 1: Antagon Inc. Proprietary
# WATERMARK Layer 2: MiniCrit MCP Server v1.1
# WATERMARK Layer 3: Model Context Protocol Implementation
# WATERMARK Layer 4: Hash SHA256:MCP_LORA_2026
# WATERMARK Layer 5: Build 20260112
# ================================================================

"""
MiniCrit MCP Server (LoRA Version)

Loads base model + LoRA adapter for MiniCrit validation.

Environment variables:
    MINICRIT_ADAPTER: LoRA adapter (default: wmaousley/MiniCrit-1.5B)
    MINICRIT_BASE_MODEL: Base model (default: Qwen/Qwen2-0.5B-Instruct)
    MINICRIT_DEVICE: Device (auto, cuda, mps, cpu)
"""

import os
import sys
import json
import logging
import time
import asyncio
from typing import Optional
from enum import Enum
from dataclasses import dataclass, asdict

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, CallToolResult
except ImportError:
    print("Error: MCP SDK not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# ================================================================
# Configuration
# ================================================================

ADAPTER_ID = os.environ.get("MINICRIT_ADAPTER", "wmaousley/MiniCrit-1.5B")
BASE_MODEL_ID = os.environ.get("MINICRIT_BASE_MODEL", "Qwen/Qwen2-0.5B-Instruct")
DEVICE = os.environ.get("MINICRIT_DEVICE", "auto")
MAX_LENGTH = int(os.environ.get("MINICRIT_MAX_LENGTH", "512"))
LOG_LEVEL = os.environ.get("MINICRIT_LOG_LEVEL", "INFO")

DOMAINS = [
    "trading", "finance", "risk_assessment", "resource_allocation",
    "planning_scheduling", "cybersecurity", "defense", "medical", "general",
]

class Severity(str, Enum):
    PASS = "pass"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("minicrit-mcp")

# ================================================================
# Model Loading
# ================================================================

_model = None
_tokenizer = None

def get_model():
    """Load base model + LoRA adapter."""
    global _model, _tokenizer
    
    if _model is not None:
        return _model, _tokenizer
    
    logger.info(f"Loading base model: {BASE_MODEL_ID}")
    logger.info(f"Loading LoRA adapter: {ADAPTER_ID}")
    
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
    
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    
    # Load base model
    logger.info("Loading base model weights...")
    dtype = torch.float16 if device == "mps" else torch.bfloat16
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    base_model = base_model.to(device)
    
    # Apply LoRA adapter
    logger.info("Applying LoRA adapter...")
    _model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    _model.eval()
    
    logger.info(f"Model loaded on {next(_model.parameters()).device}")
    
    return _model, _tokenizer

# ================================================================
# Critique Generation
# ================================================================

@dataclass
class CritiqueResult:
    valid: bool
    severity: str
    critique: str
    confidence: float
    flags: list
    domain: str
    latency_ms: float

def generate_critique(rationale: str, domain: str = "general", context: Optional[str] = None) -> CritiqueResult:
    """Generate adversarial critique."""
    import torch
    
    start_time = time.time()
    model, tokenizer = get_model()
    
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
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    # Parse severity and flags
    flags = []
    severity = Severity.PASS
    critique_lower = generated.lower()
    
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
    
    valid = severity in [Severity.PASS, Severity.LOW]
    confidence = 0.85 if len(generated) >= 20 else 0.5
    latency_ms = (time.time() - start_time) * 1000
    
    return CritiqueResult(
        valid=valid,
        severity=severity.value,
        critique=generated,
        confidence=confidence,
        flags=flags,
        domain=domain,
        latency_ms=round(latency_ms, 2),
    )

# ================================================================
# MCP Server
# ================================================================

server = Server("minicrit")

@server.list_tools()
async def list_tools():
    """List available tools."""
    return [
        Tool(
            name="validate_reasoning",
            description="Validate AI reasoning using MiniCrit adversarial analysis. Detects flaws like overconfidence, missing considerations, logical inconsistencies.",
            inputSchema={
                "type": "object",
                "properties": {
                    "rationale": {
                        "type": "string",
                        "description": "The AI reasoning/rationale to validate",
                    },
                    "domain": {
                        "type": "string",
                        "description": f"Domain context. One of: {', '.join(DOMAINS)}",
                        "enum": DOMAINS,
                        "default": "general",
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context for validation (optional)",
                    },
                },
                "required": ["rationale"],
            },
        ),
        Tool(
            name="get_model_info",
            description="Get information about the MiniCrit model",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""
    
    if name == "validate_reasoning":
        rationale = arguments.get("rationale", "")
        domain = arguments.get("domain", "general")
        context = arguments.get("context")
        
        if not rationale:
            return CallToolResult(
                content=[TextContent(type="text", text="Error: rationale is required")]
            )
        
        if domain not in DOMAINS:
            domain = "general"
        
        try:
            result = generate_critique(rationale, domain, context)
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(asdict(result), indent=2))]
            )
        except Exception as e:
            logger.error(f"Critique error: {e}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")]
            )
    
    elif name == "get_model_info":
        info = {
            "adapter": ADAPTER_ID,
            "base_model": BASE_MODEL_ID,
            "device": DEVICE,
            "domains": DOMAINS,
            "model_loaded": _model is not None,
            "company": "Antagon Inc.",
            "cage_code": "17E75",
        }
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(info, indent=2))]
        )
    
    return CallToolResult(
        content=[TextContent(type="text", text=f"Unknown tool: {name}")]
    )

# ================================================================
# Main
# ================================================================

async def main():
    logger.info("Starting MiniCrit MCP Server")
    logger.info(f"Base model: {BASE_MODEL_ID}")
    logger.info(f"LoRA adapter: {ADAPTER_ID}")
    logger.info(f"Device: {DEVICE}")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
