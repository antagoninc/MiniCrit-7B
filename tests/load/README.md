# MiniCrit Load Testing

This directory contains load testing scripts for the MiniCrit API.

## Prerequisites

### Locust (Python)
```bash
pip install locust
```

### k6 (Go)
```bash
# macOS
brew install k6

# Linux
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6
```

## Quick Start

### 1. Start the MiniCrit API Server

```bash
# From project root
python -m src.api
```

### 2. Run Load Tests

#### Locust (Web UI)
```bash
locust -f tests/load/locustfile.py --host http://localhost:8000
# Open http://localhost:8089 in browser
```

#### Locust (Headless/CI)
```bash
# 10 users, spawn 2/sec, run for 60 seconds
locust -f tests/load/locustfile.py \
    --host http://localhost:8000 \
    --headless \
    -u 10 \
    -r 2 \
    -t 60s \
    --csv=results
```

#### k6
```bash
# Run with default scenarios
k6 run tests/load/k6_test.js

# Run with custom VUs and duration
k6 run --vus 10 --duration 60s tests/load/k6_test.js

# Run against different host
BASE_URL=http://api.example.com k6 run tests/load/k6_test.js
```

## Test Scenarios

### Locust User Types

| User Type | Description | Wait Time |
|-----------|-------------|-----------|
| `MiniCritUser` | Standard user - health, stats, critiques | 1-3s |
| `MiniCritHeavyUser` | Heavy user - batch critiques | 0.5-1s |
| `MiniCritReadOnlyUser` | Read-only - health, stats, metrics | 0.1-0.5s |

### k6 Scenarios

| Scenario | Description | Duration |
|----------|-------------|----------|
| `smoke` | Basic functionality verification | 10s |
| `load` | Typical expected load (5-10 users) | ~3 min |
| `stress` | Find breaking point (up to 30 users) | ~3 min |

## Metrics

### Locust Metrics
- Request count/rate
- Response times (median, p95, p99)
- Failure rate
- Users over time

### k6 Metrics
- `http_req_duration` - Request duration
- `errors` - Error rate
- `critique_duration` - Critique endpoint latency
- `batch_duration` - Batch endpoint latency
- `health_check_duration` - Health check latency
- `tokens_generated` - Total tokens generated

## Thresholds (k6)

```javascript
thresholds: {
  http_req_duration: ['p(95)<2000'], // 95% under 2s
  errors: ['rate<0.1'],               // <10% error rate
  health_check_duration: ['p(95)<100'], // Health under 100ms
  critique_duration: ['p(95)<5000'],    // Critique under 5s
}
```

## CI Integration

### GitHub Actions Example

```yaml
- name: Run Load Tests
  run: |
    pip install locust
    python -m src.api &
    sleep 10  # Wait for server
    locust -f tests/load/locustfile.py \
        --host http://localhost:8000 \
        --headless \
        -u 5 \
        -r 1 \
        -t 30s \
        --csv=load_test_results
    kill %1
```

## Output Files

### Locust CSV Output
- `results_stats.csv` - Summary statistics
- `results_stats_history.csv` - Time series data
- `results_failures.csv` - Failed requests
- `results_exceptions.csv` - Exceptions

### k6 Output
```bash
# JSON output
k6 run --out json=results.json tests/load/k6_test.js

# InfluxDB output (for Grafana)
k6 run --out influxdb=http://localhost:8086/k6 tests/load/k6_test.js
```

## Performance Targets

| Endpoint | Target p95 | Target p99 |
|----------|------------|------------|
| `/health` | <100ms | <200ms |
| `/stats` | <100ms | <200ms |
| `/metrics` | <100ms | <200ms |
| `/critique` | <3000ms | <5000ms |
| `/critique/batch` | <10000ms | <15000ms |

## Troubleshooting

### Server Not Responding
```bash
# Check if server is running
curl http://localhost:8000/health
```

### High Error Rate
- Check server logs for OOM errors
- Verify GPU memory availability
- Reduce concurrent users

### Timeout Errors
- Increase timeout in load test config
- Check `MINICRIT_INFERENCE_TIMEOUT` env var
