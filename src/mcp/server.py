#!/usr/bin/env python3
# ================================================================
# MiniCrit MCP Server - LoRA Adapter Version
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# Co-Founder & CEO: William Alexander Ousley (Alex Ousley)
# Co-Founder & CTO: Jacqueline Villamor Ousley (Jacque Ousley) TS/SCI
# ================================================================
# WATERMARK Layer 1: Antagon Inc. Proprietary
# WATERMARK Layer 2: MiniCrit MCP Server v1.2
# WATERMARK Layer 3: Model Context Protocol Implementation
# WATERMARK Layer 4: Hash SHA256:MCP_LORA_2026
# WATERMARK Layer 5: Build 20260112
# ================================================================

"""
MiniCrit MCP Server (LoRA Version)

Loads base model + LoRA adapter for MiniCrit validation.
Uses thread-safe ModelManager and CritiqueGenerator from core module.

Environment variables:
    MINICRIT_ADAPTER: LoRA adapter (default: wmaousley/MiniCrit-7B)
    MINICRIT_BASE_MODEL: Base model (default: Qwen/Qwen2-7B-Instruct)
    MINICRIT_DEVICE: Device (auto, cuda, mps, cpu)
    MINICRIT_INFERENCE_TIMEOUT: Inference timeout in seconds (default: 120)
    MINICRIT_PRELOAD: Set to "true" to preload model on startup
"""

import os
import sys
import json
import logging
import asyncio
from typing import Optional

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, CallToolResult
except ImportError:
    print("Error: MCP SDK not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Import from core module
from src.mcp.core import (
    ModelManager,
    CritiqueGenerator,
    CritiqueResult,
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
)

# ================================================================
# Configuration
# ================================================================

PRELOAD_MODEL = os.environ.get("MINICRIT_PRELOAD", "false").lower() == "true"

# Logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("minicrit-mcp")

# ================================================================
# Global Instances (Thread-Safe)
# ================================================================

model_manager = ModelManager.get_instance()
critique_generator = CritiqueGenerator(model_manager)
shutdown_handler = GracefulShutdown(model_manager)

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
                        "minLength": 10,
                        "maxLength": 10000,
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
        Tool(
            name="batch_validate",
            description="Validate multiple AI reasonings in batch",
            inputSchema={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "List of items to validate",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "rationale": {"type": "string"},
                                "domain": {"type": "string", "enum": DOMAINS},
                            },
                            "required": ["id", "rationale"],
                        },
                        "maxItems": 100,
                    },
                },
                "required": ["items"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls with specific exception handling."""

    if name == "validate_reasoning":
        return await _handle_validate_reasoning(arguments)

    elif name == "get_model_info":
        return await _handle_get_model_info()

    elif name == "batch_validate":
        return await _handle_batch_validate(arguments)

    return CallToolResult(
        content=[TextContent(type="text", text=f"Unknown tool: {name}")]
    )


async def _handle_validate_reasoning(arguments: dict) -> CallToolResult:
    """Handle validate_reasoning tool call."""
    rationale = arguments.get("rationale", "")
    domain = arguments.get("domain", "general")
    context = arguments.get("context")

    if not rationale:
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({"error": "rationale is required", "code": "INVALID_INPUT"})
            )]
        )

    if domain not in DOMAINS:
        domain = "general"

    try:
        result = await critique_generator.generate_async(
            rationale=rationale,
            domain=domain,
            context=context,
        )
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result.to_dict(), indent=2))]
        )

    except InvalidInputError as e:
        logger.warning(f"Invalid input: {e}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({"error": str(e), "code": "INVALID_INPUT"})
            )]
        )

    except ModelLoadError as e:
        logger.error(f"Model load error: {e}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({"error": f"Model failed to load: {e}", "code": "MODEL_LOAD_ERROR"})
            )]
        )

    except InferenceTimeoutError as e:
        logger.error(f"Inference timeout: {e}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({"error": str(e), "code": "TIMEOUT"})
            )]
        )

    except InferenceError as e:
        logger.error(f"Inference error: {e}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({"error": str(e), "code": "INFERENCE_ERROR"})
            )]
        )

    except MemoryError as e:
        logger.error(f"Memory error: {e}")
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({"error": "Out of memory", "code": "OOM_ERROR"})
            )]
        )


async def _handle_get_model_info() -> CallToolResult:
    """Handle get_model_info tool call."""
    info = {
        "adapter": ADAPTER_ID,
        "base_model": BASE_MODEL_ID,
        "device": model_manager.device or DEVICE,
        "domains": DOMAINS,
        "model_loaded": model_manager.is_loaded,
        "stats": model_manager.stats,
        "company": "Antagon Inc.",
        "cage_code": "17E75",
        "version": "1.2.0",
    }
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(info, indent=2))]
    )


async def _handle_batch_validate(arguments: dict) -> CallToolResult:
    """Handle batch_validate tool call."""
    items = arguments.get("items", [])

    if not items:
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({"error": "items array is required", "code": "INVALID_INPUT"})
            )]
        )

    if len(items) > 100:
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({"error": "Maximum 100 items allowed", "code": "INVALID_INPUT"})
            )]
        )

    results = []
    for item in items:
        item_id = item.get("id", "unknown")
        rationale = item.get("rationale", "")
        domain = item.get("domain", "general")

        try:
            result = await critique_generator.generate_async(
                rationale=rationale,
                domain=domain,
                request_id=item_id,
            )
            results.append({
                "id": item_id,
                **result.to_dict(),
            })

        except (InvalidInputError, ModelLoadError, InferenceTimeoutError, InferenceError) as e:
            results.append({
                "id": item_id,
                "error": str(e),
                "code": type(e).__name__.upper(),
            })

        except MemoryError:
            results.append({
                "id": item_id,
                "error": "Out of memory",
                "code": "OOM_ERROR",
            })

    return CallToolResult(
        content=[TextContent(
            type="text",
            text=json.dumps({"results": results, "count": len(results)}, indent=2)
        )]
    )


# ================================================================
# Main
# ================================================================

async def main():
    """Main entry point."""
    logger.info("Starting MiniCrit MCP Server")
    logger.info(f"Base model: {BASE_MODEL_ID}")
    logger.info(f"LoRA adapter: {ADAPTER_ID}")
    logger.info(f"Device: {DEVICE}")

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
            # Continue anyway - will retry on first request

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
