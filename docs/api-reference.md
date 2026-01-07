# MiniCrit API Reference

**Base URL:** `http://localhost:8000`

## Endpoints

### Health Check

```http
GET /health
```

Check server health and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Qwen/Qwen2-7B-Instruct",
  "load_time": "2024-01-01T12:00:00",
  "request_count": 150,
  "total_tokens_generated": 38400,
  "uptime_seconds": 3600.5
}
```

---

### Server Statistics

```http
GET /stats
```

Get server performance statistics.

**Response:**
```json
{
  "request_count": 150,
  "total_tokens_generated": 38400,
  "avg_tokens_per_request": 256.0,
  "model_name": "Qwen/Qwen2-7B-Instruct",
  "uptime_seconds": 3600.5
}
```

---

### Load Model

```http
POST /load
```

Explicitly load or reload the model.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `model_path` | string | Optional path to model checkpoint |

**Response:**
```json
{
  "status": "Model loaded successfully"
}
```

---

### Generate Critique

```http
POST /critique
```

Generate a critique for a single rationale.

**Request Body:**
```json
{
  "rationale": "AAPL is bullish because the stock price increased 5% last week.",
  "max_tokens": 256,
  "temperature": 0.7,
  "do_sample": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rationale` | string | required | The reasoning to critique (10-4096 chars) |
| `max_tokens` | int | 256 | Maximum tokens to generate (32-1024) |
| `temperature` | float | 0.7 | Sampling temperature (0.0-2.0) |
| `do_sample` | bool | true | Use sampling vs greedy decoding |

**Response:**
```json
{
  "critique": "The rationale commits the post hoc ergo propter hoc fallacy...",
  "rationale": "AAPL is bullish because the stock price increased 5% last week.",
  "tokens_generated": 89,
  "latency_ms": 1250.5,
  "model_name": "Qwen/Qwen2-7B-Instruct"
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:8000/critique \
  -H "Content-Type: application/json" \
  -d '{
    "rationale": "AAPL is bullish because the stock price increased 5% last week.",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

---

### Batch Critique

```http
POST /critique/batch
```

Generate critiques for multiple rationales.

**Request Body:**
```json
{
  "rationales": [
    "AAPL is bullish because...",
    "The market will crash because..."
  ],
  "max_tokens": 256,
  "temperature": 0.7
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rationales` | array | required | List of rationales (1-100 items) |
| `max_tokens` | int | 256 | Maximum tokens per critique |
| `temperature` | float | 0.7 | Sampling temperature |

**Response:**
```json
{
  "critiques": [
    {
      "critique": "...",
      "rationale": "...",
      "tokens_generated": 89,
      "latency_ms": 1250.5,
      "model_name": "Qwen/Qwen2-7B-Instruct"
    }
  ],
  "total_latency_ms": 2500.0,
  "avg_latency_ms": 1250.0
}
```

---

## OpenAPI Schema

```yaml
openapi: 3.0.0
info:
  title: MiniCrit API
  description: REST API for MiniCrit adversarial critique generation
  version: 1.0.0
  contact:
    name: Antagon Inc.

servers:
  - url: http://localhost:8000
    description: Local development server

paths:
  /health:
    get:
      summary: Health check
      responses:
        '200':
          description: Server health status

  /stats:
    get:
      summary: Server statistics
      responses:
        '200':
          description: Performance statistics

  /load:
    post:
      summary: Load model
      parameters:
        - name: model_path
          in: query
          schema:
            type: string
      responses:
        '200':
          description: Model loaded

  /critique:
    post:
      summary: Generate critique
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CritiqueRequest'
      responses:
        '200':
          description: Generated critique
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CritiqueResponse'

  /critique/batch:
    post:
      summary: Batch critique generation
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/BatchCritiqueRequest'
      responses:
        '200':
          description: Generated critiques
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/BatchCritiqueResponse'

components:
  schemas:
    CritiqueRequest:
      type: object
      required:
        - rationale
      properties:
        rationale:
          type: string
          minLength: 10
          maxLength: 4096
        max_tokens:
          type: integer
          default: 256
          minimum: 32
          maximum: 1024
        temperature:
          type: number
          default: 0.7
          minimum: 0.0
          maximum: 2.0
        do_sample:
          type: boolean
          default: true

    CritiqueResponse:
      type: object
      properties:
        critique:
          type: string
        rationale:
          type: string
        tokens_generated:
          type: integer
        latency_ms:
          type: number
        model_name:
          type: string

    BatchCritiqueRequest:
      type: object
      required:
        - rationales
      properties:
        rationales:
          type: array
          items:
            type: string
          minItems: 1
          maxItems: 100
        max_tokens:
          type: integer
          default: 256
        temperature:
          type: number
          default: 0.7

    BatchCritiqueResponse:
      type: object
      properties:
        critiques:
          type: array
          items:
            $ref: '#/components/schemas/CritiqueResponse'
        total_latency_ms:
          type: number
        avg_latency_ms:
          type: number
```

---

## Running the Server

### Local Development
```bash
pip install fastapi uvicorn
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t minicrit-api .
docker run -p 8000:8000 --gpus all minicrit-api
```

### Docker Compose
```bash
docker-compose up -d
```

### View Interactive Docs
Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
