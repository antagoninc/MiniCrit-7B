# ================================================================
# MiniCrit MCP Server - Docker Image
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# Co-Founder & CEO: William Alexander Ousley (Alex Ousley)
# Co-Founder & CTO: Jacqueline Villamor Ousley (Jacque Ousley) TS/SCI
# ================================================================
# WATERMARK Layer 1: Antagon Inc. Proprietary
# WATERMARK Layer 2: MiniCrit Docker Build
# WATERMARK Layer 3: Container Deployment
# WATERMARK Layer 4: Hash SHA256:DOCKER_2026
# WATERMARK Layer 5: Build 20260112
# ================================================================

# Base image with CUDA support
FROM nvidia/cuda:13.1.0-runtime-ubuntu22.04 AS base

# Build arguments
ARG PYTHON_VERSION=3.11
ARG MINICRIT_MODEL=wmaousley/MiniCrit-7B

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Create app user
RUN useradd -m -s /bin/bash minicrit
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY minicrit_mcp_server_prod.py .
COPY minicrit_mcp_server.py .

# Pre-download model (optional - makes container larger but faster startup)
# Uncomment to bake model into image:
# RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
#     AutoTokenizer.from_pretrained('${MINICRIT_MODEL}'); \
#     AutoModelForCausalLM.from_pretrained('${MINICRIT_MODEL}')"

# Set ownership
RUN chown -R minicrit:minicrit /app

# Switch to non-root user
USER minicrit

# Environment defaults
ENV MINICRIT_MODEL=${MINICRIT_MODEL} \
    MINICRIT_DEVICE=auto \
    MINICRIT_PORT=8000 \
    MINICRIT_LOG_LEVEL=INFO \
    MINICRIT_RATE_LIMIT=60

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${MINICRIT_PORT}/health || exit 1

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "minicrit_mcp_server_prod.py"]

# ================================================================
# Multi-stage build for smaller image (CPU-only version)
# ================================================================

FROM python:3.11-slim AS cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd -m -s /bin/bash minicrit
WORKDIR /app

COPY requirements-cpu.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY minicrit_mcp_server_prod.py .
COPY minicrit_mcp_server.py .

RUN chown -R minicrit:minicrit /app
USER minicrit

ENV MINICRIT_DEVICE=cpu \
    MINICRIT_PORT=8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "minicrit_mcp_server_prod.py"]
