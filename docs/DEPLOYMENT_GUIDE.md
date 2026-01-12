# MiniCrit MCP Server - Deployment Guide

**Antagon Inc.** | CAGE: 17E75 | UEI: KBSGT7CZ4AH3

Co-Founder & CEO: William Alexander Ousley (Alex Ousley)  
Co-Founder & CTO: Jacqueline Villamor Ousley (Jacque Ousley) TS/SCI

---

## 📁 Files Included

| File | Purpose |
|------|---------|
| `minicrit_mcp_server.py` | Local stdio server (Claude Desktop) |
| `minicrit_mcp_server_prod.py` | Production HTTP server with auth |
| `Dockerfile` | Docker image build |
| `docker-compose.yml` | Container orchestration |
| `requirements.txt` | Python dependencies (GPU) |
| `requirements-cpu.txt` | Python dependencies (CPU) |
| `.env.example` | Environment configuration |
| `benchmark_models.py` | Model evaluation script |

---

## 🚀 Quick Start

### Option 1: Local Development

```bash
# Create project directory
mkdir -p ~/MiniCrit-MCP && cd ~/MiniCrit-MCP

# Install dependencies
pip install mcp fastapi uvicorn torch transformers peft

# Run server
python minicrit_mcp_server_prod.py
```

### Option 2: Docker (Recommended for Production)

```bash
# Clone/copy files
cd ~/MiniCrit-MCP

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env

# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f minicrit
```

---

## 🔐 Authentication

### Generating API Keys

**Option 1: Environment Variable**
```bash
# Generate random keys
export MINICRIT_API_KEYS="$(openssl rand -base64 32),$(openssl rand -base64 32)"
export MINICRIT_MASTER_KEY="$(openssl rand -base64 32)"
```

**Option 2: Admin Endpoint**
```bash
# Create key via API (requires master key)
curl -X POST http://localhost:8000/admin/keys \
  -H "X-API-Key: YOUR_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "customer-1", "tier": "standard"}'
```

### Using API Keys

```bash
# Header authentication
curl http://localhost:8000/validate \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"rationale": "...", "domain": "trading"}'

# Query parameter authentication
curl "http://localhost:8000/validate?api_key=your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"rationale": "...", "domain": "trading"}'
```

### Rate Limiting

| Tier | Requests/Minute |
|------|-----------------|
| admin | 1000 |
| premium | 300 |
| standard | 60 |
| free | 10 |

---

## 🐳 Docker Deployment

### Build Images

```bash
# GPU version
docker build -t antagon/minicrit-mcp:latest --target base .

# CPU version
docker build -t antagon/minicrit-mcp:cpu --target cpu .
```

### Run Containers

```bash
# GPU (requires nvidia-docker)
docker-compose up -d minicrit

# CPU only
docker-compose --profile cpu up -d minicrit-cpu

# With Redis (distributed rate limiting)
docker-compose --profile redis up -d
```

### Environment Variables

```bash
# Required
MINICRIT_MODEL=wmaousley/MiniCrit-7B
MINICRIT_API_KEYS=key1,key2,key3
MINICRIT_MASTER_KEY=admin-key

# Optional
MINICRIT_DEVICE=cuda          # cuda, cpu, mps, auto
MINICRIT_PORT=8000
MINICRIT_RATE_LIMIT=60
MINICRIT_LOG_LEVEL=INFO
MINICRIT_CORS_ORIGINS=*
```

---

## ☸️ Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minicrit-mcp
spec:
  replicas: 2
  selector:
    matchLabels:
      app: minicrit-mcp
  template:
    metadata:
      labels:
        app: minicrit-mcp
    spec:
      containers:
      - name: minicrit
        image: antagon/minicrit-mcp:latest
        ports:
        - containerPort: 8000
        env:
        - name: MINICRIT_MODEL
          value: "wmaousley/MiniCrit-7B"
        - name: MINICRIT_API_KEYS
          valueFrom:
            secretKeyRef:
              name: minicrit-secrets
              key: api-keys
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            memory: "8Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: minicrit-mcp
spec:
  selector:
    app: minicrit-mcp
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 📊 Benchmarking

### Run Benchmark

```bash
# Full benchmark (7B vs 1.5B)
python benchmark_models.py \
  --eval-data ~/Desktop/domain_data_clean/eval_holdout.jsonl \
  --model-1 wmaousley/MiniCrit-1.5B \
  --model-2 /path/to/minicrit-7b-checkpoint \
  --output-dir ./benchmark_results

# Quick test (limited examples)
python benchmark_models.py \
  --eval-data ~/Desktop/domain_data_clean/eval_holdout.jsonl \
  --limit 100 \
  --skip-judge
```

### With LLM-as-Judge

```bash
# Set Anthropic API key
export ANTHROPIC_API_KEY=your-key

# Run with judge comparison
python benchmark_models.py \
  --eval-data ~/Desktop/domain_data_clean/eval_holdout.jsonl \
  --judge-sample 50
```

### Metrics Computed

| Metric | Description |
|--------|-------------|
| **False Positive Rate** | % of valid reasoning incorrectly flagged |
| **Avg Latency** | Mean inference time (ms) |
| **P95 Latency** | 95th percentile latency |
| **Peak Memory** | GPU memory usage (GB) |
| **LLM Judge Wins** | Head-to-head quality comparison |

---

## 🔗 Claude Desktop Integration

### Configure Claude Desktop (macOS)

```bash
# Edit config
nano ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

```json
{
  "mcpServers": {
    "minicrit": {
      "command": "python",
      "args": ["/Users/YOU/MiniCrit-MCP/minicrit_mcp_server.py"],
      "env": {
        "MINICRIT_MODEL": "wmaousley/MiniCrit-7B",
        "MINICRIT_DEVICE": "mps"
      }
    }
  }
}
```

### Configure Claude Desktop (Windows)

```json
// %APPDATA%\Claude\claude_desktop_config.json
{
  "mcpServers": {
    "minicrit": {
      "command": "python",
      "args": ["C:\\MiniCrit-MCP\\minicrit_mcp_server.py"],
      "env": {
        "MINICRIT_MODEL": "wmaousley/MiniCrit-7B"
      }
    }
  }
}
```

### Test Integration

After restarting Claude Desktop, you should see MiniCrit tools available. Test with:

> "Use the validate_reasoning tool to check this logic: 'Stock will rise because it rose yesterday.'"

---

## 📡 API Reference

### POST /validate

Validate single rationale.

```bash
curl -X POST http://localhost:8000/validate \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "rationale": "Based on RSI showing oversold, I recommend buying with high confidence.",
    "domain": "trading",
    "context": "Market volatility is elevated."
  }'
```

**Response:**
```json
{
  "valid": false,
  "severity": "medium",
  "critique": "The recommendation shows overconfidence...",
  "confidence": 0.87,
  "flags": ["overconfidence", "unaddressed_risk"],
  "domain": "trading",
  "latency_ms": 42.3,
  "timestamp": "2026-01-12T15:30:00Z",
  "request_id": "a1b2c3d4"
}
```

### POST /batch

Validate multiple items.

```bash
curl -X POST http://localhost:8000/batch \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"id": "trade_1", "rationale": "...", "domain": "trading"},
      {"id": "trade_2", "rationale": "...", "domain": "trading"}
    ]
  }'
```

### GET /health

Health check (no auth required).

```bash
curl http://localhost:8000/health
```

### GET /info

Server information.

```bash
curl http://localhost:8000/info -H "X-API-Key: your-key"
```

---

## 🔧 Troubleshooting

### Model Loading Fails

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check memory
nvidia-smi

# Use CPU if needed
export MINICRIT_DEVICE=cpu
```

### Rate Limit Exceeded

```bash
# Check your remaining quota
curl http://localhost:8000/info -H "X-API-Key: your-key" | jq '.rate_limit_remaining'

# Request tier upgrade via admin
curl -X POST http://localhost:8000/admin/keys \
  -H "X-API-Key: MASTER_KEY" \
  -d '{"name": "premium-customer", "tier": "premium"}'
```

### Container Won't Start

```bash
# Check logs
docker-compose logs minicrit

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi

# Start with more memory
docker-compose up -d --scale minicrit=1 -e "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"
```

---

## 📈 Monitoring

### Prometheus Metrics (Future)

```yaml
# Add to docker-compose.yml
prometheus:
  image: prom/prometheus
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml
  ports:
    - "9090:9090"
```

### Audit Log

```bash
# Get recent audit entries (admin only)
curl http://localhost:8000/admin/audit?n=100 \
  -H "X-API-Key: MASTER_KEY"
```

---

## 🛡️ Security Considerations

1. **Always use HTTPS in production** (terminate TLS at load balancer)
2. **Rotate API keys regularly**
3. **Set appropriate CORS origins** (not `*` in production)
4. **Monitor audit logs** for unusual activity
5. **Use rate limiting** to prevent abuse
6. **Run as non-root user** (Docker images do this by default)

---

## 📞 Support

**Antagon Inc.**
- Email: founders@antagon.ai
- Web: https://www.antagon.ai
- CAGE: 17E75
- UEI: KBSGT7CZ4AH3
