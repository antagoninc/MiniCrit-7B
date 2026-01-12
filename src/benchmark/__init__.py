# ================================================================
# MiniCrit Benchmark Suite
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# Co-Founder & CEO: William Alexander Ousley (Alex Ousley)
# Co-Founder & CTO: Jacqueline Villamor Ousley (Jacque Ousley) TS/SCI
# ================================================================
# WATERMARK Layer 1: Antagon Inc. Proprietary
# WATERMARK Layer 2: MiniCrit Benchmark Module
# WATERMARK Layer 3: Model Evaluation Pipeline
# WATERMARK Layer 4: Hash SHA256:BENCH_INIT_2026
# WATERMARK Layer 5: Build 20260112
# ================================================================

"""
MiniCrit Benchmark Suite

Evaluates MiniCrit models on:
- False positive rate (hard_negative examples)
- Flaw detection accuracy
- LLM-as-judge comparison
- Inference latency

Usage:
    python -m src.benchmark.benchmark_models \
        --eval-data eval_holdout.jsonl \
        --model-1 wmaousley/MiniCrit-1.5B \
        --model-2 /path/to/minicrit-7b
"""
