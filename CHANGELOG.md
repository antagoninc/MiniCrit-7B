# Changelog

## [0.1.1] - 2026-01-12

### Added
- `src/mcp/core.py` - Thread-safe core module with ModelManager, CritiqueGenerator, RateLimiter
- `tests/test_mcp_core.py` - 50+ test cases for MCP core module
- Model preloading support via `MINICRIT_PRELOAD=true` environment variable
- Inference timeout protection via `MINICRIT_INFERENCE_TIMEOUT` (default 120s)
- Graceful shutdown handling for SIGTERM/SIGINT signals
- Configurable CORS origins via `MINICRIT_CORS_ORIGINS` environment variable

### Changed
- Refactored `server.py` and `server_http.py` to use shared core module
- Replaced broad `except Exception` with specific exception types throughout
- CORS now defaults to localhost origins instead of wildcard `*`

### Removed
- `src/mcp/minicrit_mcp_server.py` (duplicate)
- `src/mcp/minicrit_mcp_server_http.py` (duplicate)

### Security
- Hardened CORS configuration (no longer allows all origins by default)
- Added input validation with `InvalidInputError` exception
- Thread-safe model access prevents race conditions

---

## [0.1.0] - 2026-01-12

### Added

#### MCP Integration (Model Context Protocol)
- `src/mcp/server.py` - Local MCP server for Claude Desktop integration
- `src/mcp/server_prod.py` - Production HTTP server with API key auth, rate limiting, audit logging
- `src/mcp/server_http.py` - Basic HTTP server for testing
- `configs/claude_desktop_config.json` - Claude Desktop configuration template

MCP enables any MCP-compatible AI system (Claude, GPT, Gemini, Copilot) to call MiniCrit for reasoning validation. Industry standard backed by Anthropic, OpenAI, Google, Microsoft.

#### Benchmark Suite
- `src/benchmark/benchmark_models.py` - Head-to-head model comparison
  - False positive rate on hard_negative examples
  - Latency metrics (avg, p50, p95, p99)
  - LLM-as-judge quality comparison
  - Memory usage tracking

#### Training Improvements
- `src/training/generate_hard_examples.py` - Generate challenging training data via Claude Sonnet
  - Focus on sophisticated_valid (prevent false positives)
  - Subtle flaws, cognitive biases, mixed signals
  - ~$30 for 5K high-quality examples
- `src/training/generate_dpo_data.py` - Create preference pairs for DPO
- `src/training/train_dpo.py` - Direct Preference Optimization training script

#### Docker Deployment
- `docker/Dockerfile` - Multi-stage build (GPU + CPU targets)
- `docker/docker-compose.yml` - Full orchestration with GPU, Redis, Nginx profiles
- `docker/requirements.txt` - GPU dependencies
- `docker/requirements-cpu.txt` - CPU-only dependencies
- `docker/.env.example` - Environment configuration template

#### Documentation
- `docs/DEPLOYMENT_GUIDE.md` - Complete deployment instructions
- `docs/MODEL_EXCELLENCE_GUIDE.md` - Roadmap for model improvement

### Technical Details

**MCP Server Features:**
- Loads base model + LoRA adapter properly
- Supports domains: trading, finance, defense, cybersecurity, medical, planning, general
- Returns: valid, severity, critique, confidence, flags, latency
- <50ms inference on GPU

**Supported Configurations:**
| Model | Base | Size | Use Case |
|-------|------|------|----------|
| MiniCrit-1.5B | Qwen2-0.5B-Instruct | ~1GB | Fast validation |
| MiniCrit-7B | Qwen2-7B-Instruct | ~14GB | Production quality |

### Contributors
- William Alexander Ousley (Alex) - CEO
- Jacqueline Villamor Ousley (Jacque) - CTO

**Antagon Inc.** | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
