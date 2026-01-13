# MiniCrit Monitoring Stack

Pre-configured monitoring for MiniCrit API using Prometheus and Grafana.

## Quick Start

```bash
cd monitoring
docker-compose up -d
```

### Access Points
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## Components

### Core Stack
| Service | Port | Description |
|---------|------|-------------|
| Prometheus | 9090 | Metrics collection and storage |
| Grafana | 3000 | Visualization and dashboards |

### Optional Services
```bash
# Start with Redis (for distributed rate limiting)
docker-compose --profile redis up -d

# Start with Jaeger (for distributed tracing)
docker-compose --profile tracing up -d

# Start all optional services
docker-compose --profile redis --profile tracing up -d
```

| Service | Port | Description |
|---------|------|-------------|
| Redis | 6379 | Distributed rate limiting backend |
| Jaeger | 16686 (UI), 4317 (OTLP) | Distributed tracing |

## Configuration

### MiniCrit API Setup

Ensure your MiniCrit API exposes the `/metrics` endpoint:

```bash
# Start MiniCrit API
python -m src.api
```

### Prometheus Configuration

Edit `prometheus.yml` to add more targets:

```yaml
scrape_configs:
  - job_name: "minicrit-cluster"
    static_configs:
      - targets:
        - "minicrit-1:8000"
        - "minicrit-2:8000"
```

### Environment Variables

For distributed rate limiting:
```bash
export MINICRIT_RATE_LIMIT_BACKEND=redis
export MINICRIT_REDIS_URL=redis://localhost:6379
```

For tracing:
```bash
export OTEL_ENABLED=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=minicrit
```

## Dashboard Panels

The pre-configured Grafana dashboard includes:

### Overview Row
- Model Status (loaded/unloaded)
- Uptime
- Active Requests
- Total Tokens Generated
- Request Rate (5m)
- Error Rate (5m)

### Request Metrics Row
- Request Rate by Endpoint
- Error Rate by Endpoint

### Latency Row
- Request Latency Percentiles (p50, p95, p99)
- Critique Latency Distribution

### Tokens Row
- Token Generation Rate

## Alerting (Optional)

Create alerts in Grafana for:
- Error rate > 5%
- p95 latency > 3s
- Model not loaded
- No requests for 5 minutes

Example alert rule (Grafana):
```yaml
- alert: HighErrorRate
  expr: sum(rate(minicrit_requests_total{status="error"}[5m])) / sum(rate(minicrit_requests_total[5m])) > 0.05
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: High error rate detected
```

## Troubleshooting

### Prometheus Can't Scrape Metrics
```bash
# Check if MiniCrit is running
curl http://localhost:8000/metrics

# If running in Docker, use host.docker.internal
# Already configured in prometheus.yml
```

### Grafana Dashboard Not Loading
```bash
# Check provisioning
docker-compose logs grafana

# Manually import dashboard
# Go to Grafana > Dashboards > Import > Upload JSON
```

### Reset Everything
```bash
docker-compose down -v
docker-compose up -d
```

## Production Recommendations

1. **Secure Grafana**: Change default password
2. **Persistent Storage**: Use external volumes
3. **High Availability**: Run multiple Prometheus instances
4. **Retention**: Configure prometheus retention period
5. **Alertmanager**: Add for alert routing

```yaml
# prometheus.yml - Add retention
command:
  - "--storage.tsdb.retention.time=30d"
```
