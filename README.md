# MiniCrit-7B: Adversarial AI Safety for Trading Systems

<p align="center">
  <img src="assets/minicrit_logo.png" alt="MiniCrit Logo" width="200">
</p>

<p align="center">
  <a href="https://huggingface.co/Antagon/MiniCrit-7B"><img src="https://img.shields.io/badge/🤗%20Model-MiniCrit--7B-blue" alt="HuggingFace"></a>
  <a href="https://wandb.ai/antagonlabs/minicrit-training"><img src="https://img.shields.io/badge/W&B-Training%20Logs-yellow" alt="Weights & Biases"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-red" alt="License"></a>
</p>

## Overview

**MiniCrit-7B** is an adversarial AI model that acts as a "devil's advocate" for autonomous trading systems. It identifies flawed reasoning in AI-generated trading signals before they can cause financial losses.

Built by [Antagon Inc.](https://antagon.ai), MiniCrit is part of our mission to make AI systems safer through adversarial testing.

### Key Results

| Metric | Value |
|--------|-------|
| 🎯 False Signal Reduction | **35%** |
| 📈 Sharpe Ratio Improvement | **+0.28** |
| 🔄 Live Trades Processed | **38,000+** |
| 📉 Training Loss Reduction | **57.6%** |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MiniCrit-7B Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────┐     ┌─────────────────────────────┐  │
│   │  Trading Signal │────▶│     MiniCrit-7B Critique    │  │
│   │   (Rationale)   │     │                             │  │
│   └─────────────────┘     │  • Identifies biases        │  │
│                           │  • Spots logical flaws      │  │
│                           │  • Flags missing risks      │  │
│                           │  • Questions assumptions    │  │
│                           └──────────────┬──────────────┘  │
│                                          │                  │
│                                          ▼                  │
│                           ┌─────────────────────────────┐  │
│                           │   Risk-Aware Decision       │  │
│                           │   • Execute with caution    │  │
│                           │   • Reduce position size    │  │
│                           │   • Skip trade entirely     │  │
│                           └─────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## What MiniCrit Detects

| Flaw Type | Description | Example |
|-----------|-------------|---------|
| **Overconfidence** | Excessive certainty without supporting evidence | "AAPL will definitely break $200" |
| **Survivorship Bias** | Ignoring failed patterns that looked similar | "This pattern always works" |
| **Spurious Correlation** | False relationships in data | "Stock rises when moon is full" |
| **Confirmation Bias** | Cherry-picking supporting evidence | "RSI confirms my bullish thesis" |
| **Overfitting** | Patterns that won't generalize | "Works perfectly on backtest" |
| **Missing Risk Factors** | Ignoring relevant risks | No mention of earnings, macro events |

## Installation

```bash
# Clone the repository
git clone https://github.com/Antagon-Inc/MiniCrit-7B.git
cd MiniCrit-7B

# Install dependencies
pip install -r requirements.txt

# Download model weights
python download_model.py
```

## Quick Start

```python
from minicrit import MiniCrit7B

# Initialize model
critic = MiniCrit7B()

# Critique a trading rationale
rationale = "META long: Bollinger Band expansion with supporting momentum."
critique = critic.analyze(rationale)

print(critique)
# Output: "While Bollinger Band expansion can signal volatility, META's recent 
# expansion isn't necessarily predictive; it could be a reaction to news, not 
# a precursor to sustained movement..."
```

## Training Details

### Model Specifications

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2-7B-Instruct |
| Total Parameters | 7.6B |
| Trainable Parameters | 40.4M (LoRA) |
| Training Method | LoRA (r=16, α=32) |
| Dataset Size | 11.7M examples |
| Hardware | NVIDIA H100 80GB (Lambda Labs GPU Grant) |

### Training Progress

```
Training Loss Curve
───────────────────────────────────────────────
Loss │
1.85 │██
1.50 │  ████
1.25 │      ████████
1.00 │              ████████████
0.79 │                          ████████████████
     └──────────────────────────────────────────
      0     10k    20k    30k    35k  Steps
```

### Training Configuration

```yaml
# config/training_config.yaml
model:
  base: Qwen/Qwen2-7B-Instruct
  method: lora
  
lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  learning_rate: 2e-4
  scheduler: cosine
  warmup_steps: 500
  batch_size: 32
  max_length: 512
  epochs: 1
```

## Repository Structure

```
MiniCrit-7B/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
│
├── minicrit/
│   ├── __init__.py
│   ├── model.py           # Model loading and inference
│   ├── critique.py        # Critique generation logic
│   └── utils.py           # Helper functions
│
├── training/
│   ├── train.py           # Training script
│   ├── config.yaml        # Training configuration
│   └── data_utils.py      # Data processing
│
├── evaluation/
│   ├── evaluate.py        # Evaluation script
│   ├── metrics.py         # Custom metrics
│   └── benchmarks/        # Benchmark datasets
│
├── analysis/
│   ├── analyze_training.py
│   └── visualize_results.py
│
├── docs/
│   ├── WHITEPAPER.md      # Technical whitepaper
│   ├── API.md             # API documentation
│   └── TRAINING.md        # Training guide
│
└── assets/
    ├── minicrit_logo.png
    └── training_curves.png
```

## Evaluation

### Running Evaluation

```bash
python evaluation/evaluate.py \
  --model Antagon/MiniCrit-7B \
  --dataset benchmarks/trading_critiques.json \
  --output results/
```

### Benchmark Results

| Benchmark | MiniCrit-7B | GPT-4 | Claude-3 |
|-----------|-------------|-------|----------|
| Flaw Detection (F1) | **0.82** | 0.75 | 0.78 |
| Critique Quality | **4.2/5** | 3.8/5 | 4.0/5 |
| False Positive Rate | **12%** | 18% | 15% |
| Latency (ms) | **45** | 850 | 620 |

## API Usage

### REST API

```bash
# Start the server
python -m minicrit.server --port 8000

# Make a request
curl -X POST http://localhost:8000/critique \
  -H "Content-Type: application/json" \
  -d '{"rationale": "AAPL long: MACD bullish crossover"}'
```

### Python SDK

```python
from minicrit import MiniCritClient

client = MiniCritClient(api_key="your-api-key")

# Single critique
result = client.critique("TSLA short: RSI overbought at 75")

# Batch processing
rationales = ["AAPL long: ...", "META short: ...", "NVDA long: ..."]
results = client.critique_batch(rationales)
```

## Citation

```bibtex
@article{ousley2026minicrit,
  title={MiniCrit: Adversarial AI Critique for Autonomous Trading System Safety},
  author={Ousley, William Alexander and Ousley, Jacqueline Villamor},
  journal={arXiv preprint arXiv:2601.XXXXX},
  year={2026}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

We gratefully acknowledge **[Lambda Labs](https://lambdalabs.com)** for providing GPU compute through their Research Grant program. MiniCrit-7B was trained on Lambda's H100 infrastructure, and their generous support has been instrumental in advancing our AI safety research.

<p align="center">
  <a href="https://lambdalabs.com"><img src="https://img.shields.io/badge/Compute%20Sponsor-Lambda%20Labs-purple" alt="Lambda Labs"></a>
</p>

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## About Antagon Inc.

**Antagon Inc.** develops adversarial AI safety systems that detect flawed reasoning in autonomous systems before catastrophic failures occur.

- **Website**: [antagon.ai](https://antagon.ai)
- **CAGE Code**: 17E75
- **UEI**: KBSGT7CZ4AH3

### Leadership

- **William Alexander Ousley** - Co-Founder & CEO
- **Jacqueline Villamor Ousley** - Co-Founder & CTO (TS/SCI Clearance)

---

<p align="center">
  <b>Making AI Systems Safer Through Adversarial Testing</b>
</p>
