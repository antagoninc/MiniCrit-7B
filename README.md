# MiniCrit-7B: Adversarial AI Validation for Autonomous Systems

<p align="center">
  <img src="assets/minicrit_logo.png" alt="MiniCrit Logo" width="200">
</p>

<p align="center">
  <strong>🛡️ Catch AI reasoning flaws before they become failures</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/wmaousley/MiniCrit-7B"><img src="https://img.shields.io/badge/🤗%20HuggingFace-MiniCrit--7B-blue" alt="HuggingFace"></a>
  <a href="https://github.com/antagoninc/MiniCrit-7B/actions"><img src="https://img.shields.io/badge/Tests-169%20Passing-brightgreen" alt="Tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-red" alt="License"></a>
  <a href="https://modelcontextprotocol.io"><img src="https://img.shields.io/badge/MCP-Compatible-purple" alt="MCP"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#mcp-integration">MCP Integration</a> •
  <a href="#api-usage">API</a> •
  <a href="#training">Training</a> •
  <a href="#benchmarks">Benchmarks</a>
</p>

---

## 🎯 The Problem

Autonomous AI systems fail silently. They produce confident-sounding outputs with hidden flaws—overconfidence, missing risks, logical errors, hallucinations. **By the time you notice, it's too late.**

Traditional testing catches bugs. **MiniCrit catches bad reasoning.**

## 💡 The Solution

MiniCrit is a specialized AI "devil's advocate" that validates reasoning **before** actions are taken. It integrates with any AI system via MCP (Model Context Protocol) to provide real-time adversarial critique.

```
Your AI Agent → MiniCrit Validation → Safer Decisions
     ↓                   ↓                    ↓
  "Buy AAPL,         "Overconfidence:      Execute with
   95% confident"     only 2 data points,   reduced size
                      missing earnings       or skip
                      risk"
```

---

## 📊 Results

<table>
<tr>
<td align="center"><h3>35%</h3><sub>Flawed Output Reduction</sub></td>
<td align="center"><h3>+0.28</h3><sub>Sharpe Ratio Improvement</sub></td>
<td align="center"><h3>38,000+</h3><sub>Live Validations</sub></td>
<td align="center"><h3><50ms</h3><sub>Inference Latency</sub></td>
</tr>
</table>

| Metric | MiniCrit-7B | GPT-4 | Claude-3 |
|--------|-------------|-------|----------|
| Flaw Detection F1 | **0.82** | 0.75 | 0.78 |
| False Positive Rate | **12%** | 18% | 15% |
| Latency | **45ms** | 850ms | 620ms |
| Cost per 1K calls | **$0.00** | $30 | $15 |

---

## 🚀 Quick Start

### Option 1: Python Package

```bash
pip install torch transformers peft
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load model
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct")
model = PeftModel.from_pretrained(base, "wmaousley/MiniCrit-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

# Validate reasoning
rationale = "Stock will rise because it rose yesterday"
prompt = f"### Domain: trading\n### Rationale:\n{rationale}\n\n### Critique:\n"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
critique = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Option 2: Docker

```bash
docker-compose up -d
curl http://localhost:8000/critique \
  -H "Content-Type: application/json" \
  -d '{"rationale": "Buy signal based on MACD crossover", "domain": "trading"}'
```

### Option 3: MCP (Claude Desktop / Claude Code)

```json
// ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "minicrit": {
      "command": "python3",
      "args": ["/path/to/MiniCrit-7B/src/mcp/server.py"],
      "env": {
        "MINICRIT_ADAPTER": "wmaousley/MiniCrit-7B",
        "MINICRIT_BASE_MODEL": "Qwen/Qwen2-7B-Instruct"
      }
    }
  }
}
```

Then in Claude: *"Use validate_reasoning to check: Buy AAPL, RSI shows oversold"*

---

## 🔌 MCP Integration

MiniCrit implements the **Model Context Protocol** (MCP)—the industry standard for AI tool integration, backed by Anthropic, OpenAI, Google, and Microsoft.

**Any MCP-compatible AI can call MiniCrit:**

```
┌─────────────────┐         ┌─────────────────┐
│  Claude / GPT   │         │    MiniCrit     │
│  Gemini / etc.  │◄───────►│   MCP Server    │
└─────────────────┘   MCP   └─────────────────┘
         │                           │
         │    Tool: validate_reasoning
         │    Input: {rationale, domain}
         │    Output: {valid, severity, critique, flags}
         │
         ▼
   Safer AI Decisions
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `validate_reasoning` | Validate AI reasoning, returns critique with severity |
| `batch_validate` | Validate multiple items efficiently |
| `get_model_info` | Get model status and configuration |

### Supported Domains

`trading` • `finance` • `defense` • `cybersecurity` • `medical` • `risk_assessment` • `planning` • `general`

### Output Format

```json
{
  "valid": false,
  "severity": "high",
  "critique": "This reasoning exhibits recency bias. A single day's price movement has no predictive power...",
  "confidence": 0.87,
  "flags": ["overconfidence", "insufficient_evidence", "missing_consideration"],
  "latency_ms": 42.3
}
```

**Severity Levels:** `pass` → `low` → `medium` → `high` → `critical`

---

## 🔍 What MiniCrit Detects

<table>
<tr>
<td width="50%">

### Cognitive Biases
- ⚠️ **Overconfidence** - Certainty without evidence
- ⚠️ **Survivorship Bias** - Ignoring failures
- ⚠️ **Confirmation Bias** - Cherry-picking data
- ⚠️ **Anchoring** - Over-relying on first info
- ⚠️ **Recency Bias** - Overweighting recent events

</td>
<td width="50%">

### Logical Flaws
- 🚫 **False Causation** - Correlation ≠ causation
- 🚫 **Hasty Generalization** - Small sample size
- 🚫 **Missing Risks** - Unaddressed threats
- 🚫 **Circular Reasoning** - Assuming the conclusion
- 🚫 **False Dichotomy** - Ignoring options

</td>
</tr>
</table>

### Example

**Input:**
> "AAPL long: The stock has risen 3 days in a row, momentum is clearly bullish. 95% confident this continues."

**MiniCrit Output:**
> ⚠️ **HIGH SEVERITY** - Multiple reasoning flaws detected:
> 
> 1. **Overconfidence**: 95% confidence is not supported by the evidence provided
> 2. **Recency Bias**: 3 days of price movement has minimal predictive value
> 3. **Missing Risk Factors**: No consideration of upcoming earnings, macro events, or sector rotation
> 
> *Flags: overconfidence, insufficient_evidence, unaddressed_risk*

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      MiniCrit-7B System                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────┐    ┌─────────────┐    ┌───────────────────────┐ │
│  │ AI System  │───▶│   MiniCrit  │───▶│   Validated Output    │ │
│  │ (Any LLM)  │    │   Server    │    │                       │ │
│  └────────────┘    └──────┬──────┘    │ • valid: bool         │ │
│                           │           │ • severity: enum      │ │
│                    ┌──────▼──────┐    │ • critique: string    │ │
│                    │  Qwen2-7B   │    │ • flags: list         │ │
│                    │    Base     │    │ • confidence: float   │ │
│                    └──────┬──────┘    └───────────────────────┘ │
│                           │                                      │
│                    ┌──────▼──────┐                              │
│                    │   LoRA      │  40.4M trainable params      │
│                    │  Adapter    │  Trained on 11.7M critiques  │
│                    └─────────────┘                              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📈 Training

### Model Specifications

| Parameter | Value |
|-----------|-------|
| Base Model | `Qwen/Qwen2-7B-Instruct` |
| Method | LoRA (r=16, α=32) |
| Trainable Params | 40.4M / 7.6B (0.5%) |
| Dataset | CritiqueBank-11M |
| Hardware | TACC Vista GH200 / Lambda H100 |
| Training Loss | 3.19 → 0.44 (86% reduction) |

### Dataset: CritiqueBank-11M

| Component | Examples |
|-----------|----------|
| LogicFlaw-2.4M | Logical reasoning errors |
| FactCheck-3.2M | Factual accuracy validation |
| BiasDetect-1.8M | Cognitive bias patterns |
| RiskMissing-2.1M | Unaddressed risk factors |
| DomainSpecific-2.2M | Trading, defense, medical |

**Published:** [DOI 10.5281/zenodo.18159342](https://zenodo.org/records/18159342)

### Training Progress

```
Loss
3.2 │██
2.4 │  ████
1.6 │      ██████
0.8 │            ████████████
0.4 │                        ████████████████  ← Current
    └─────────────────────────────────────────
    0%        25%        50%        75%     100%
```

### Run Training

```bash
# TACC Vista (GH200)
sbatch scripts/train_vista.slurm

# Lambda Labs (H100)
python train_minicrit_7b.py --config configs/7b_lora.yaml
```

---

## 🧪 Benchmarking

Compare MiniCrit models head-to-head:

```bash
python src/benchmark/benchmark_models.py \
  --eval-data data/eval_holdout.jsonl \
  --model-1 wmaousley/MiniCrit-1.5B \
  --model-2 wmaousley/MiniCrit-7B \
  --judge-sample 200  # Optional: LLM-as-judge comparison
```

### Metrics Computed

| Metric | Description |
|--------|-------------|
| False Positive Rate | Valid reasoning incorrectly flagged |
| Detection F1 | Precision/recall on flaw detection |
| Latency (p50/p95/p99) | Inference speed percentiles |
| LLM Judge Score | Claude rates critique quality |

---

## 🔧 Advanced: Improve Your Model

### Generate Hard Training Examples

```bash
# Uses Claude Sonnet (~$30 for 5K examples)
export ANTHROPIC_API_KEY=your-key
python src/training/generate_hard_examples.py --count 5000
```

### Direct Preference Optimization (DPO)

```bash
# Generate preference pairs
python src/training/generate_dpo_data.py \
  --input eval_holdout.jsonl \
  --model wmaousley/MiniCrit-7B \
  --output dpo_pairs.jsonl

# Run DPO training
python src/training/train_dpo.py \
  --model wmaousley/MiniCrit-7B \
  --data dpo_pairs.jsonl \
  --output minicrit-7b-dpo
```

See [docs/MODEL_EXCELLENCE_GUIDE.md](docs/MODEL_EXCELLENCE_GUIDE.md) for the full improvement roadmap.

---

## 📁 Repository Structure

```
MiniCrit-7B/
├── src/
│   ├── mcp/                    # MCP Server Implementation
│   │   ├── server.py           # Local stdio (Claude Desktop)
│   │   ├── server_prod.py      # Production HTTP + auth
│   │   └── server_http.py      # Basic HTTP server
│   ├── benchmark/              # Model Evaluation
│   │   └── benchmark_models.py
│   ├── training/               # Training Utilities
│   │   ├── generate_hard_examples.py
│   │   ├── generate_dpo_data.py
│   │   └── train_dpo.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── training.py
│   ├── evaluation.py
│   └── api.py
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── configs/
│   ├── claude_desktop_config.json
│   ├── 7b_lora.yaml
│   └── deepspeed_gh200.json
├── scripts/
│   ├── train_vista.slurm
│   └── vista_setup.sh
├── tests/                      # 169 tests
├── docs/
│   ├── DEPLOYMENT_GUIDE.md
│   └── MODEL_EXCELLENCE_GUIDE.md
└── CHANGELOG.md
```

---

## 🐳 Deployment Options

### Docker (Recommended)

```bash
cd docker
cp .env.example .env
# Edit .env with your settings
docker-compose up -d
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minicrit
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: minicrit
        image: antagoninc/minicrit:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

### Production HTTP Server

```bash
# With authentication & rate limiting
export MINICRIT_API_KEYS="key1,key2,key3"
python src/mcp/server_prod.py
```

See [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) for complete instructions.

---

## 🎯 Use Cases

| Domain | Application |
|--------|-------------|
| **Quantitative Trading** | Validate signals before execution |
| **Defense / Intelligence** | Audit AI threat assessments |
| **Medical AI** | Review diagnostic reasoning |
| **Autonomous Vehicles** | Validate planning decisions |
| **Enterprise AI** | Catch hallucinations before they propagate |

---

## 📜 Citation

```bibtex
@software{minicrit2026,
  author = {Ousley, William Alexander and Ousley, Jacqueline Villamor},
  title = {MiniCrit: Adversarial AI Validation for Autonomous Systems},
  year = {2026},
  publisher = {Antagon Inc.},
  url = {https://github.com/antagoninc/MiniCrit-7B}
}
```

---

## 🙏 Acknowledgments

<p align="center">
  <a href="https://lambdalabs.com"><img src="https://img.shields.io/badge/GPU%20Compute-Lambda%20Labs-purple" alt="Lambda Labs"></a>
  <a href="https://tacc.utexas.edu"><img src="https://img.shields.io/badge/Supercomputing-TACC%20Vista-orange" alt="TACC"></a>
  <a href="https://new.nsf.gov/funding/initiatives/nairr"><img src="https://img.shields.io/badge/NAIRR-Pilot%20Program-blue" alt="NAIRR"></a>
</p>

- **Lambda Labs** - GPU compute grant for H100 training
- **TACC Vista** - GH200 supercomputing via NAIRR Pilot
- **Anthropic** - MCP standard development

---

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE)

---

<p align="center">
  <b>Antagon Inc.</b><br>
  Making AI Systems Safer Through Adversarial Testing
</p>

<p align="center">
  <a href="https://antagon.ai">Website</a> •
  <a href="mailto:founders@antagon.ai">Contact</a> •
  CAGE: 17E75 • UEI: KBSGT7CZ4AH3
</p>

<p align="center">
  <b>William Alexander Ousley</b> - Co-Founder & CEO<br>
  <b>Jacqueline Villamor Ousley</b> - Co-Founder & CTO (TS/SCI)
</p>
