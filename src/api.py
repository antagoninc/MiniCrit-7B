"""FastAPI server for MiniCrit model inference.

This module provides a REST API for generating critiques using the
fine-tuned MiniCrit model. Supports both single and batch inference.

Example:
    >>> uvicorn src.api:app --host 0.0.0.0 --port 8000

    # Single critique
    curl -X POST http://localhost:8000/critique \
        -H "Content-Type: application/json" \
        -d '{"rationale": "AAPL is bullish because..."}'

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.logging_config import setup_logging, get_logger
from src.metrics import metrics, track_request

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# CORS configuration - configurable via environment variable
import os

CORS_ORIGINS = os.environ.get(
    "MINICRIT_CORS_ORIGINS", "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000"
).split(",")

# Global model state
_model_state: dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "loaded": False,
    "load_time": None,
    "request_count": 0,
    "total_tokens_generated": 0,
}


class CritiqueRequest(BaseModel):
    """Request model for critique generation."""

    rationale: str = Field(
        ...,
        min_length=10,
        max_length=4096,
        description="The rationale/reasoning to critique",
        examples=["AAPL is bullish because the stock price increased 5% last week."],
    )
    max_tokens: int = Field(
        default=256,
        ge=32,
        le=1024,
        description="Maximum tokens to generate",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    do_sample: bool = Field(
        default=True,
        description="Whether to use sampling (vs greedy decoding)",
    )


class CritiqueResponse(BaseModel):
    """Response model for critique generation."""

    critique: str = Field(..., description="Generated critique")
    rationale: str = Field(..., description="Original rationale")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    latency_ms: float = Field(..., description="Generation latency in milliseconds")
    model_name: str = Field(..., description="Model used for generation")


class BatchCritiqueRequest(BaseModel):
    """Request model for batch critique generation."""

    rationales: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of rationales to critique",
    )
    max_tokens: int = Field(default=256, ge=32, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class BatchCritiqueResponse(BaseModel):
    """Response model for batch critique generation."""

    critiques: list[CritiqueResponse]
    total_latency_ms: float
    avg_latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: str | None
    load_time: str | None
    request_count: int
    total_tokens_generated: int
    uptime_seconds: float


class StatsResponse(BaseModel):
    """Server statistics response."""

    request_count: int
    total_tokens_generated: int
    avg_tokens_per_request: float
    model_name: str | None
    uptime_seconds: float


_start_time = datetime.now()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load model on startup."""
    logger.info("Starting MiniCrit API server...")

    # Optionally load model on startup (can be lazy-loaded instead)
    # load_model()

    yield

    logger.info("Shutting down MiniCrit API server...")


app = FastAPI(
    title="MiniCrit API",
    description="REST API for MiniCrit adversarial critique generation",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware with configurable origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


def load_model(model_path: str | None = None) -> None:
    """Load the MiniCrit model and tokenizer.

    Args:
        model_path: Optional path to model checkpoint. If None, uses default.
    """
    if _model_state["loaded"]:
        logger.info("Model already loaded")
        return

    logger.info("Loading MiniCrit model...")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        # Default paths
        base_model = "Qwen/Qwen2-7B-Instruct"
        adapter_path = model_path or "minicrit_7b_output/minicrit-7b-final"

        logger.info(f"Loading base model: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load LoRA adapter if available
        try:
            logger.info(f"Loading adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)  # type: ignore[assignment]
            model = model.merge_and_unload()  # type: ignore[operator]
            logger.info("LoRA adapter loaded and merged")
        except FileNotFoundError as e:
            logger.warning(f"Adapter not found: {e}. Using base model.")
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Could not load adapter: {e}. Using base model.")

        model.eval()

        _model_state["model"] = model
        _model_state["tokenizer"] = tokenizer
        _model_state["loaded"] = True
        _model_state["load_time"] = datetime.now().isoformat()
        _model_state["model_name"] = base_model
        metrics.set_model_loaded(True)

        logger.info("Model loaded successfully")

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        raise HTTPException(status_code=500, detail=f"Missing dependency: {e}")
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        raise HTTPException(status_code=500, detail=f"Model files not found: {e}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"GPU out of memory: {e}")
            raise HTTPException(status_code=503, detail="GPU out of memory")
        logger.error(f"Runtime error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")
    except (ValueError, OSError) as e:
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")


def generate_critique(
    rationale: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> tuple[str, int]:
    """Generate a critique for the given rationale.

    Args:
        rationale: The reasoning to critique.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        do_sample: Whether to use sampling.

    Returns:
        Tuple of (critique_text, tokens_generated).
    """
    if not _model_state["loaded"]:
        load_model()

    model = _model_state["model"]
    tokenizer = _model_state["tokenizer"]

    import torch

    prompt = f"### Rationale:\n{rationale}\n\n### Critique:\n"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    tokens_generated = outputs.shape[1] - input_length
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract critique
    if "### Critique:" in full_output:
        critique = full_output.split("### Critique:")[-1].strip()
    else:
        critique = full_output[len(prompt) :].strip()

    return critique, tokens_generated


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check server health and model status."""
    uptime = (datetime.now() - _start_time).total_seconds()

    return HealthResponse(
        status="healthy",
        model_loaded=_model_state["loaded"],
        model_name=_model_state.get("model_name"),
        load_time=_model_state.get("load_time"),
        request_count=_model_state["request_count"],
        total_tokens_generated=_model_state["total_tokens_generated"],
        uptime_seconds=uptime,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """Get server statistics."""
    uptime = (datetime.now() - _start_time).total_seconds()
    req_count = _model_state["request_count"]
    total_tokens = _model_state["total_tokens_generated"]

    return StatsResponse(
        request_count=req_count,
        total_tokens_generated=total_tokens,
        avg_tokens_per_request=total_tokens / req_count if req_count > 0 else 0,
        model_name=_model_state.get("model_name"),
        uptime_seconds=uptime,
    )


from fastapi.responses import PlainTextResponse


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    """Expose Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping.
    Compatible with Prometheus, Grafana, and other monitoring tools.

    Returns:
        Prometheus text format metrics.
    """
    return metrics.get_prometheus_metrics()


@app.post("/load")
async def load_model_endpoint(model_path: str | None = None) -> dict[str, str]:
    """Explicitly load or reload the model."""
    _model_state["loaded"] = False
    load_model(model_path)
    return {"status": "Model loaded successfully"}


@app.post("/critique", response_model=CritiqueResponse)
async def create_critique(request: CritiqueRequest) -> CritiqueResponse:
    """Generate a critique for a single rationale."""
    logger.info(f"Critique request received (length={len(request.rationale)})")

    start_time = time.perf_counter()

    with track_request("critique"):
        try:
            critique, tokens = generate_critique(
                rationale=request.rationale,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=request.do_sample,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Update stats
            _model_state["request_count"] += 1
            _model_state["total_tokens_generated"] += tokens
            metrics.add_tokens(tokens)

            logger.info(f"Critique generated: {tokens} tokens in {latency_ms:.1f}ms")

            return CritiqueResponse(
                critique=critique,
                rationale=request.rationale,
                tokens_generated=tokens,
                latency_ms=latency_ms,
                model_name=_model_state.get("model_name", "unknown"),
            )

        except ValueError as e:
            logger.warning(f"Invalid input: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"GPU out of memory: {e}")
                raise HTTPException(status_code=503, detail="GPU out of memory")
            logger.error(f"Critique generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except MemoryError as e:
            logger.error(f"Memory error: {e}")
            raise HTTPException(status_code=503, detail="Server out of memory")


@app.post("/critique/batch", response_model=BatchCritiqueResponse)
async def create_batch_critiques(request: BatchCritiqueRequest) -> BatchCritiqueResponse:
    """Generate critiques for multiple rationales."""
    logger.info(f"Batch critique request: {len(request.rationales)} rationales")

    start_time = time.perf_counter()
    critiques = []

    for rationale in request.rationales:
        item_start = time.perf_counter()

        try:
            critique, tokens = generate_critique(
                rationale=rationale,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            item_latency = (time.perf_counter() - item_start) * 1000

            critiques.append(
                CritiqueResponse(
                    critique=critique,
                    rationale=rationale,
                    tokens_generated=tokens,
                    latency_ms=item_latency,
                    model_name=_model_state.get("model_name", "unknown"),
                )
            )

            _model_state["request_count"] += 1
            _model_state["total_tokens_generated"] += tokens

        except ValueError as e:
            logger.warning(f"Batch item invalid input: {e}")
            critiques.append(
                CritiqueResponse(
                    critique=f"Invalid input: {str(e)}",
                    rationale=rationale,
                    tokens_generated=0,
                    latency_ms=0,
                    model_name="error",
                )
            )
        except RuntimeError as e:
            logger.error(f"Batch item runtime error: {e}")
            critiques.append(
                CritiqueResponse(
                    critique=f"Error: {str(e)}",
                    rationale=rationale,
                    tokens_generated=0,
                    latency_ms=0,
                    model_name="error",
                )
            )
        except MemoryError as e:
            logger.error(f"Batch item memory error: {e}")
            critiques.append(
                CritiqueResponse(
                    critique="Error: Out of memory",
                    rationale=rationale,
                    tokens_generated=0,
                    latency_ms=0,
                    model_name="error",
                )
            )

    total_latency = (time.perf_counter() - start_time) * 1000
    avg_latency = total_latency / len(critiques) if critiques else 0

    logger.info(f"Batch complete: {len(critiques)} critiques in {total_latency:.1f}ms")

    return BatchCritiqueResponse(
        critiques=critiques,
        total_latency_ms=total_latency,
        avg_latency_ms=avg_latency,
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.perf_counter()

    response = await call_next(request)

    latency = (time.perf_counter() - start_time) * 1000
    logger.debug(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Latency: {latency:.1f}ms"
    )

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
