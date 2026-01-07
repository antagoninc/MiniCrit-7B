# MiniCrit - Claude Code Instructions

## Project Overview
MiniCrit is an adversarial reasoning system designed to identify flawed reasoning in autonomous systems. Currently scaling from 7B to 70B parameters using an expanded dataset with 11M+ examples.

**Owner:** Antagon Inc. (CAGE Code: 17E75)

## Tech Stack
- **Training:** MLX optimization for Apple Silicon (Mac Studio M2 Ultra, 64GB RAM)
- **Base Models:** Scaling from 7B to 70B parameter models
- **Fine-tuning:** LoRA (rank 16, targeting attention projections)
- **Inference:** Target 35-40 tokens/second on local hardware

## Code Conventions

### Python Style
- Use type hints on all function signatures
- Docstrings follow Google style format
- Max line length: 100 characters
- Use `pathlib.Path` over `os.path`
- Prefer `dataclasses` or `pydantic` for structured data

### File Organization
```
minicrit/
├── src/
│   ├── taxonomy/       # Reasoning flaw definitions
│   ├── generation/     # Dataset generation pipelines
│   ├── training/       # Fine-tuning scripts
│   └── evaluation/     # Model evaluation
├── data/
│   ├── raw/
│   ├── processed/
│   └── checkpoints/
├── tests/
└── scripts/
```

### Naming Conventions
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- Reasoning flaw IDs: Category prefix + number (e.g., `L01`, `S05`, `C12`)

## Domain Categories
When generating training data, use these domain prefixes:
- `FIN_` - Financial/trading reasoning
- `COMP_` - Compliance/RMF reasoning (Frontier adjacent)
- `MED_` - Medical/diagnostic reasoning
- `OPS_` - Operational decisions
- `STAT_` - Statistical reasoning
- `GEN_` - General logical fallacies

## Common Mistakes to Avoid

### Dataset Generation
- ❌ Do NOT generate rebuttals shorter than 50 words - they lack substance
- ❌ Do NOT use the same rebuttal structure repeatedly - vary the approach
- ❌ Do NOT skip domain validation - every input must map to a flaw taxonomy ID
- ✅ DO include the flaw ID in metadata for every generated example
- ✅ DO validate JSON structure before batch processing
- ✅ DO checkpoint every 1000 examples

### Training
- ❌ Do NOT use batch sizes > 4 on Mac Studio (OOM risk)
- ❌ Do NOT train without gradient checkpointing enabled
- ✅ DO use `mlx` optimizations for Apple Silicon
- ✅ DO log loss curves to wandb or local CSV
- ✅ DO save adapter weights every 500 steps

### Code Quality
- ❌ Do NOT commit without running `ruff check .`
- ❌ Do NOT leave print statements in production code - use logging
- ✅ DO add `--resume` support to any long-running script
- ✅ DO include progress bars for operations > 100 iterations

## Testing Requirements
- All generation functions must have unit tests
- Test edge cases: empty inputs, malformed JSON, Unicode handling
- Integration tests for end-to-end pipeline runs
- Run `pytest tests/ -v` before committing

## Git Workflow
- Branch naming: `feature/`, `fix/`, `refactor/`
- Commit messages: imperative mood, max 72 chars first line
- Always squash feature branches before merge
- Tag releases with semantic versioning: `v0.1.0`

## Environment
```bash
# Required environment variables
export MINICRIT_DATA_DIR="/path/to/data"
export MINICRIT_CHECKPOINT_DIR="/path/to/checkpoints"
export WANDB_PROJECT="minicrit-training"  # optional
```

## Quick Commands
```bash
# Validate taxonomy coverage
python -m src.taxonomy.validate

# Generate batch of training examples
python -m src.generation.pipeline --domain FIN --count 10000

# Run training
python -m src.training.lora_finetune --config configs/70b_lora.yaml

# Evaluate model
python -m src.evaluation.benchmark --model checkpoints/latest
```

## Performance Targets
- Generation: 100+ examples/minute with hybrid backend
- Training: Complete 70B LoRA fine-tune in < 48 hours on DGX Spark
- Inference: 35-40 tokens/second on Mac Studio M2 Ultra
