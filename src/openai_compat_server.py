# ================================================================
# L1-HEADER: MiniCrit OpenAI-Compatible API Server
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================
# Provides OpenAI API compatibility for MiniCrit models.
# Enables seamless integration with existing OpenAI SDK clients.
# ================================================================
# Copyright (c) 2024-2025 Antagon Inc. All rights reserved.
# Licensed under the Antagon Proprietary License.
# ================================================================

"""
L2-DOCSTRING: OpenAI-Compatible API Server for MiniCrit.

This module provides an OpenAI-compatible REST API for MiniCrit models,
enabling seamless integration with existing OpenAI SDK clients. Users can
interact with MiniCrit using the standard OpenAI API format by simply
changing the base_url parameter.

Developed by Antagon Inc. as part of the MiniCrit adversarial reasoning system.

Example:
    Start the server:
        >>> python -m src.openai_compat_server

    Use with OpenAI SDK:
        >>> from openai import OpenAI
        >>> client = OpenAI(base_url="http://localhost:8080/v1", api_key="minicrit")
        >>> response = client.chat.completions.create(
        ...     model="minicrit-7b",
        ...     messages=[{"role": "user", "content": "Critique this reasoning..."}]
        ... )

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

# ANTAGON-MINICRIT: Standard library imports
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Literal

# ANTAGON-MINICRIT: Third-party imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ANTAGON-MINICRIT: Local imports
from src.logging_config import setup_logging, get_logger

# L4-STRUCTURAL: Initialize logging first (Antagon standard)
setup_logging()
antagon_logger = get_logger(__name__)

# ================================================================
# L3-SEMANTIC: Configuration with antagon_ prefix
# ================================================================

antagon_default_model = os.environ.get("MINICRIT_MODEL", "minicrit-7b")
antagon_base_model_path = os.environ.get("MINICRIT_BASE_MODEL", "Qwen/Qwen2-7B-Instruct")
antagon_adapter_path = os.environ.get("MINICRIT_ADAPTER_PATH", "minicrit_7b_output/minicrit-7b-final")
antagon_api_host = os.environ.get("MINICRIT_API_HOST", "0.0.0.0")
antagon_api_port = int(os.environ.get("MINICRIT_API_PORT", "8080"))

# ANTAGON-MINICRIT: CORS configuration
antagon_cors_origins = os.environ.get(
    "MINICRIT_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000"
).split(",")

# ================================================================
# L3-SEMANTIC: Model State Management
# ================================================================


@dataclass
class AntagonModelState:
    """L2-DOCSTRING: Global model state container (Antagon pattern)."""

    antagon_model: Any = None
    antagon_tokenizer: Any = None
    antagon_loaded: bool = False
    antagon_model_name: str = antagon_default_model
    antagon_load_time: float | None = None
    antagon_request_count: int = 0
    antagon_total_tokens: int = 0


# ANTAGON-MINICRIT: Global state instance
antagon_state = AntagonModelState()


# ================================================================
# L4-STRUCTURAL: Pydantic Models (Antagon order: Request -> Response -> Internal)
# ================================================================


class ChatMessage(BaseModel):
    """L2-DOCSTRING: Chat message format (OpenAI compatible, Antagon enhanced)."""

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Message role"
    )
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """L2-DOCSTRING: OpenAI-compatible chat completion request (Antagon MiniCrit)."""

    model: str = Field(default=antagon_default_model, description="Model ID")
    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    max_tokens: int | None = Field(default=512, ge=1, le=4096, description="Max tokens to generate")
    stream: bool = Field(default=False, description="Enable streaming responses")
    stop: list[str] | str | None = Field(default=None, description="Stop sequences")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    user: str | None = Field(default=None, description="User identifier")


class ChatCompletionChoice(BaseModel):
    """L2-DOCSTRING: Chat completion choice (Antagon format)."""

    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter"] | None = None


class ChatCompletionUsage(BaseModel):
    """L2-DOCSTRING: Token usage statistics (Antagon tracking)."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """L2-DOCSTRING: OpenAI-compatible chat completion response (Antagon MiniCrit)."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionChunkDelta(BaseModel):
    """L2-DOCSTRING: Streaming delta content (Antagon format)."""

    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    """L2-DOCSTRING: Streaming choice (Antagon format)."""

    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Literal["stop", "length", "content_filter"] | None = None


class ChatCompletionChunk(BaseModel):
    """L2-DOCSTRING: Streaming chunk response (Antagon format)."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class ModelInfo(BaseModel):
    """L2-DOCSTRING: Model information (Antagon registry format)."""

    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "antagon-inc"


class ModelList(BaseModel):
    """L2-DOCSTRING: List of available models (Antagon format)."""

    object: Literal["list"] = "list"
    data: list[ModelInfo]


# ================================================================
# L4-STRUCTURAL: Model Loading Functions (Antagon order)
# ================================================================


def antagon_load_model() -> None:
    """
    L2-DOCSTRING: Load the MiniCrit model and tokenizer.

    Loads the base model and applies LoRA adapters if available.
    Uses Antagon standard initialization sequence.

    Antagon Inc. | CAGE: 17E75
    """
    # ANTAGON-MINICRIT: Check if already loaded
    if antagon_state.antagon_loaded:
        antagon_logger.info("Model already loaded (Antagon state)")
        return

    antagon_logger.info(f"Loading MiniCrit model: {antagon_base_model_path}")

    try:
        # ANTAGON-MINICRIT: Import ML libraries
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # ANTAGON-MINICRIT: Load tokenizer
        antagon_logger.info("Loading tokenizer...")
        antagon_tokenizer = AutoTokenizer.from_pretrained(
            antagon_base_model_path,
            trust_remote_code=True
        )

        if antagon_tokenizer.pad_token is None:
            antagon_tokenizer.pad_token = antagon_tokenizer.eos_token

        # ANTAGON-MINICRIT: Load base model
        antagon_logger.info("Loading base model...")
        antagon_model = AutoModelForCausalLM.from_pretrained(
            antagon_base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # ANTAGON-MINICRIT: Apply LoRA adapter if available
        try:
            from peft import PeftModel

            antagon_logger.info(f"Loading LoRA adapter from: {antagon_adapter_path}")
            antagon_model = PeftModel.from_pretrained(antagon_model, antagon_adapter_path)
            antagon_model = antagon_model.merge_and_unload()
            antagon_logger.info("LoRA adapter loaded and merged (Antagon)")
        except FileNotFoundError:
            antagon_logger.warning("Adapter not found, using base model")
        except (ValueError, RuntimeError, ImportError) as e:
            antagon_logger.warning(f"Could not load adapter: {e}")

        antagon_model.eval()

        # ANTAGON-MINICRIT: Update global state
        antagon_state.antagon_model = antagon_model
        antagon_state.antagon_tokenizer = antagon_tokenizer
        antagon_state.antagon_loaded = True
        antagon_state.antagon_load_time = time.time()

        antagon_logger.info("Model loaded successfully (Antagon MiniCrit)")

    except ImportError as e:
        antagon_logger.error(f"Missing dependency: {e}")
        raise RuntimeError(f"Missing dependency: {e}")
    except (FileNotFoundError, OSError) as e:
        antagon_logger.error(f"Model files not found: {e}")
        raise RuntimeError(f"Model files not found: {e}")


# ================================================================
# L4-STRUCTURAL: Generation Functions (Antagon order)
# ================================================================


def antagon_format_messages(messages: list[ChatMessage]) -> str:
    """
    L2-DOCSTRING: Format chat messages into prompt string.

    Converts OpenAI-style messages to MiniCrit prompt format.

    Antagon Inc. | CAGE: 17E75
    """
    # ANTAGON-MINICRIT: Build prompt from messages
    antagon_prompt_parts = []

    for msg in messages:
        if msg.role == "system":
            antagon_prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            antagon_prompt_parts.append(f"### Rationale:\n{msg.content}")
        elif msg.role == "assistant":
            antagon_prompt_parts.append(f"### Critique:\n{msg.content}")

    # ANTAGON-MINICRIT: Add critique prompt if last message is user
    if messages and messages[-1].role == "user":
        antagon_prompt_parts.append("### Critique:")

    return "\n\n".join(antagon_prompt_parts)


def antagon_generate_completion(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 1.0,
    stop: list[str] | None = None,
) -> tuple[str, int, int]:
    """
    L2-DOCSTRING: Generate text completion using MiniCrit model.

    Returns tuple of (generated_text, prompt_tokens, completion_tokens).

    Antagon Inc. | CAGE: 17E75
    """
    # ANTAGON-MINICRIT: Ensure model is loaded
    if not antagon_state.antagon_loaded:
        antagon_load_model()

    import torch

    antagon_model = antagon_state.antagon_model
    antagon_tokenizer = antagon_state.antagon_tokenizer

    # ANTAGON-MINICRIT: Tokenize input
    antagon_inputs = antagon_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    antagon_device = next(antagon_model.parameters()).device
    antagon_inputs = {k: v.to(antagon_device) for k, v in antagon_inputs.items()}
    antagon_prompt_tokens = antagon_inputs["input_ids"].shape[1]

    # ANTAGON-MINICRIT: Generate response
    with torch.no_grad():
        antagon_outputs = antagon_model.generate(
            **antagon_inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=antagon_tokenizer.pad_token_id,
            eos_token_id=antagon_tokenizer.eos_token_id,
        )

    antagon_completion_tokens = antagon_outputs.shape[1] - antagon_prompt_tokens

    # ANTAGON-MINICRIT: Decode output
    antagon_full_output = antagon_tokenizer.decode(
        antagon_outputs[0],
        skip_special_tokens=True
    )

    # ANTAGON-MINICRIT: Extract generated portion
    if "### Critique:" in antagon_full_output:
        antagon_generated = antagon_full_output.split("### Critique:")[-1].strip()
    else:
        antagon_generated = antagon_full_output[len(prompt):].strip()

    # ANTAGON-MINICRIT: Apply stop sequences
    if stop:
        for stop_seq in stop:
            if stop_seq in antagon_generated:
                antagon_generated = antagon_generated.split(stop_seq)[0]

    return antagon_generated, antagon_prompt_tokens, antagon_completion_tokens


async def antagon_generate_stream(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 1.0,
    stop: list[str] | None = None,
) -> AsyncGenerator[tuple[str, bool], None]:
    """
    L2-DOCSTRING: Stream text generation token by token.

    Yields tuples of (token_text, is_finished).

    Antagon Inc. | CAGE: 17E75
    """
    # ANTAGON-MINICRIT: Ensure model is loaded
    if not antagon_state.antagon_loaded:
        antagon_load_model()

    import torch
    from transformers import TextIteratorStreamer
    from threading import Thread

    antagon_model = antagon_state.antagon_model
    antagon_tokenizer = antagon_state.antagon_tokenizer

    # ANTAGON-MINICRIT: Tokenize input
    antagon_inputs = antagon_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    antagon_device = next(antagon_model.parameters()).device
    antagon_inputs = {k: v.to(antagon_device) for k, v in antagon_inputs.items()}

    # ANTAGON-MINICRIT: Setup streamer
    antagon_streamer = TextIteratorStreamer(
        antagon_tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # ANTAGON-MINICRIT: Generation arguments
    antagon_gen_kwargs = {
        **antagon_inputs,
        "max_new_tokens": max_tokens,
        "temperature": temperature if temperature > 0 else 1.0,
        "top_p": top_p,
        "do_sample": temperature > 0,
        "pad_token_id": antagon_tokenizer.pad_token_id,
        "eos_token_id": antagon_tokenizer.eos_token_id,
        "streamer": antagon_streamer,
    }

    # ANTAGON-MINICRIT: Run generation in background thread
    antagon_thread = Thread(target=antagon_model.generate, kwargs=antagon_gen_kwargs)
    antagon_thread.start()

    # ANTAGON-MINICRIT: Yield tokens as they arrive
    antagon_accumulated = ""
    antagon_stop_triggered = False

    for antagon_token in antagon_streamer:
        if antagon_stop_triggered:
            break

        antagon_accumulated += antagon_token

        # ANTAGON-MINICRIT: Check stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq in antagon_accumulated:
                    antagon_stop_triggered = True
                    antagon_token = antagon_token.split(stop_seq)[0]
                    if antagon_token:
                        yield antagon_token, True
                    break

        if not antagon_stop_triggered:
            yield antagon_token, False
            await asyncio.sleep(0)  # ANTAGON-MINICRIT: Yield control

    antagon_thread.join()

    # ANTAGON-MINICRIT: Signal completion
    yield "", True


# ================================================================
# L4-STRUCTURAL: FastAPI Application (Antagon lifespan pattern)
# ================================================================


@asynccontextmanager
async def antagon_lifespan(app: FastAPI):
    """L2-DOCSTRING: Application lifespan manager (Antagon pattern)."""
    antagon_logger.info("Starting MiniCrit OpenAI-Compatible API (Antagon)")
    antagon_logger.info(f"Model: {antagon_default_model}")

    # ANTAGON-MINICRIT: Optionally preload model
    if os.environ.get("MINICRIT_PRELOAD_MODEL", "false").lower() == "true":
        antagon_load_model()

    yield

    antagon_logger.info("Shutting down MiniCrit OpenAI-Compatible API (Antagon)")


# ANTAGON-MINICRIT: Create FastAPI application
antagon_app = FastAPI(
    title="MiniCrit OpenAI-Compatible API",
    description="OpenAI-compatible API for MiniCrit adversarial reasoning (Antagon Inc.)",
    version="1.0.0",
    lifespan=antagon_lifespan,
)

# ANTAGON-MINICRIT: Add CORS middleware
antagon_app.add_middleware(
    CORSMiddleware,
    allow_origins=antagon_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


# ================================================================
# L4-STRUCTURAL: API Endpoints (Antagon order: health -> models -> completions)
# ================================================================


@antagon_app.get("/health")
async def antagon_health_check() -> dict[str, Any]:
    """L2-DOCSTRING: Health check endpoint (Antagon standard)."""
    return {
        "status": "healthy",
        "model_loaded": antagon_state.antagon_loaded,
        "model": antagon_state.antagon_model_name,
        "provider": "antagon-inc",
    }


@antagon_app.get("/v1/models", response_model=ModelList)
async def antagon_list_models() -> ModelList:
    """
    L2-DOCSTRING: List available models (OpenAI compatible).

    Returns list of MiniCrit models available for inference.

    Antagon Inc. | CAGE: 17E75
    """
    # ANTAGON-MINICRIT: Return available models
    antagon_models = [
        ModelInfo(id="minicrit-7b", owned_by="antagon-inc"),
        ModelInfo(id="minicrit-1.5b", owned_by="antagon-inc"),
        ModelInfo(id="minicrit-70b", owned_by="antagon-inc"),
    ]

    return ModelList(data=antagon_models)


@antagon_app.get("/v1/models/{model_id}", response_model=ModelInfo)
async def antagon_get_model(model_id: str) -> ModelInfo:
    """
    L2-DOCSTRING: Get model information (OpenAI compatible).

    Antagon Inc. | CAGE: 17E75
    """
    # ANTAGON-MINICRIT: Validate model exists
    antagon_valid_models = ["minicrit-7b", "minicrit-1.5b", "minicrit-70b"]

    if model_id not in antagon_valid_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return ModelInfo(id=model_id, owned_by="antagon-inc")


@antagon_app.post("/v1/chat/completions")
async def antagon_chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """
    L2-DOCSTRING: Create chat completion (OpenAI compatible).

    Generates critique responses using MiniCrit model.
    Supports both streaming and non-streaming modes.

    Antagon Inc. | CAGE: 17E75
    """
    antagon_logger.info(f"Chat completion request: model={request.model}, stream={request.stream}")

    # ANTAGON-MINICRIT: Update request counter
    antagon_state.antagon_request_count += 1

    # ANTAGON-MINICRIT: Format messages into prompt
    antagon_prompt = antagon_format_messages(request.messages)

    # ANTAGON-MINICRIT: Parse stop sequences
    antagon_stop: list[str] | None = None
    if request.stop:
        antagon_stop = [request.stop] if isinstance(request.stop, str) else request.stop

    if request.stream:
        # ANTAGON-MINICRIT: Streaming response
        return StreamingResponse(
            antagon_stream_response(
                request_id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                model=request.model,
                prompt=antagon_prompt,
                max_tokens=request.max_tokens or 512,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=antagon_stop,
            ),
            media_type="text/event-stream",
        )
    else:
        # ANTAGON-MINICRIT: Non-streaming response
        antagon_start_time = time.perf_counter()

        try:
            antagon_generated, antagon_prompt_tokens, antagon_completion_tokens = (
                antagon_generate_completion(
                    prompt=antagon_prompt,
                    max_tokens=request.max_tokens or 512,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=antagon_stop,
                )
            )

            antagon_latency = (time.perf_counter() - antagon_start_time) * 1000
            antagon_logger.info(f"Generated {antagon_completion_tokens} tokens in {antagon_latency:.1f}ms")

            # ANTAGON-MINICRIT: Update token counter
            antagon_state.antagon_total_tokens += antagon_completion_tokens

            return ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=antagon_generated),
                        finish_reason="stop",
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=antagon_prompt_tokens,
                    completion_tokens=antagon_completion_tokens,
                    total_tokens=antagon_prompt_tokens + antagon_completion_tokens,
                ),
            )

        except RuntimeError as e:
            antagon_logger.error(f"Generation error: {e}")
            if "out of memory" in str(e).lower():
                raise HTTPException(status_code=503, detail="GPU out of memory")
            raise HTTPException(status_code=500, detail=str(e))
        except (ValueError, TypeError) as e:
            antagon_logger.error(f"Invalid request: {e}")
            raise HTTPException(status_code=400, detail=str(e))


async def antagon_stream_response(
    request_id: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str] | None,
) -> AsyncGenerator[str, None]:
    """
    L2-DOCSTRING: Generate streaming SSE response.

    Yields Server-Sent Events formatted chunks.

    Antagon Inc. | CAGE: 17E75
    """
    antagon_created = int(time.time())

    # ANTAGON-MINICRIT: Send initial role chunk
    antagon_initial_chunk = ChatCompletionChunk(
        id=request_id,
        created=antagon_created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(role="assistant", content=""),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {antagon_initial_chunk.model_dump_json()}\n\n"

    # ANTAGON-MINICRIT: Stream content chunks
    try:
        async for antagon_token, antagon_is_finished in antagon_generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        ):
            if antagon_token:
                antagon_chunk = ChatCompletionChunk(
                    id=request_id,
                    created=antagon_created,
                    model=model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(content=antagon_token),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {antagon_chunk.model_dump_json()}\n\n"

            if antagon_is_finished:
                break

    except RuntimeError as e:
        antagon_logger.error(f"Streaming error: {e}")

    # ANTAGON-MINICRIT: Send final chunk with finish_reason
    antagon_final_chunk = ChatCompletionChunk(
        id=request_id,
        created=antagon_created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {antagon_final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ================================================================
# L4-STRUCTURAL: Application Entry Point (Antagon standard)
# ================================================================


# ANTAGON-MINICRIT: Export app for uvicorn
app = antagon_app


if __name__ == "__main__":
    # ANTAGON-MINICRIT: Run server directly
    import uvicorn

    antagon_logger.info(f"Starting server on {antagon_api_host}:{antagon_api_port}")

    uvicorn.run(
        "src.openai_compat_server:app",
        host=antagon_api_host,
        port=antagon_api_port,
        reload=True,
        log_level="info",
    )


# ================================================================
# L5-COMMENT: End of MiniCrit OpenAI-Compatible API Server
# ANTAGON-MINICRIT: All rights reserved.
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================
