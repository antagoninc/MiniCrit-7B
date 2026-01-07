# MiniCrit-7B Inference API
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
#
# Build: docker build -t minicrit-api .
# Run:   docker run -p 8000:8000 --gpus all minicrit-api

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 minicrit && \
    chown -R minicrit:minicrit /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Copy model weights (if available locally)
# Uncomment and adjust path as needed:
# COPY minicrit_7b_output/minicrit-7b-final ./model/

# Switch to non-root user
USER minicrit

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - start API server
CMD ["python3", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
