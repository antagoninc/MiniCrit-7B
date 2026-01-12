#!/usr/bin/env python3
# ================================================================
# MiniCrit Model Benchmarking - 7B vs 1.5B Evaluation
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# Co-Founder & CEO: William Alexander Ousley (Alex Ousley)
# Co-Founder & CTO: Jacqueline Villamor Ousley (Jacque Ousley) TS/SCI
# ================================================================
# WATERMARK Layer 1: Antagon Inc. Proprietary
# WATERMARK Layer 2: MiniCrit Benchmark Suite
# WATERMARK Layer 3: Model Evaluation Pipeline
# WATERMARK Layer 4: Hash SHA256:BENCHMARK_2026
# WATERMARK Layer 5: Build 20260112
# ================================================================

"""
MiniCrit Model Benchmarking Script

Evaluates MiniCrit models (1.5B vs 7B) on held-out evaluation data.

Metrics computed:
- False Positive Rate (hard_negative examples)
- Flaw Detection (multi_step, edge_case examples)
- LLM-as-Judge quality comparison
- Inference latency
- Memory usage

Usage:
    python benchmark_models.py \
        --eval-data ~/Desktop/domain_data_clean/eval_holdout.jsonl \
        --model-1 wmaousley/MiniCrit-1.5B \
        --model-2 /path/to/minicrit-7b-checkpoint \
        --output-dir ./benchmark_results

Requirements:
    pip install torch transformers peft pandas numpy tqdm anthropic
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from collections import defaultdict
import statistics

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# ================================================================
# Configuration
# ================================================================

DEFAULT_MODEL_1 = "wmaousley/MiniCrit-1.5B"
DEFAULT_MODEL_2 = "wmaousley/MiniCrit-7B"
MAX_LENGTH = 512
BATCH_SIZE = 1  # For fair latency comparison

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("benchmark")

# ================================================================
# Data Classes
# ================================================================

@dataclass
class EvalExample:
    """Single evaluation example."""
    id: str
    input: str
    expected_critique: str
    domain: str
    data_type: str  # hard_negative, multi_step, edge_case, calibration

@dataclass
class ModelOutput:
    """Output from a single model inference."""
    model_name: str
    example_id: str
    generated_critique: str
    latency_ms: float
    tokens_generated: int

@dataclass
class BenchmarkResult:
    """Aggregated benchmark results for a model."""
    model_name: str
    model_size: str
    total_examples: int
    
    # Latency metrics
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # False positive rate (hard_negative examples)
    hard_negative_count: int
    false_positive_count: int
    false_positive_rate: float
    
    # Detection metrics by type
    detection_by_type: Dict[str, Dict[str, float]]
    
    # Memory usage
    peak_memory_gb: float
    
    # Timestamp
    timestamp: str

# ================================================================
# Model Loading
# ================================================================

class ModelLoader:
    """Load and manage MiniCrit models."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_name = Path(model_path).name if "/" not in model_path else model_path.split("/")[-1]
        
    def load(self):
        """Load model and tokenizer."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info(f"Loading model: {self.model_path}")
        start_time = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        device_map = self.device if self.device != "auto" else "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        
        load_time = time.time() - start_time
        device = next(self.model.parameters()).device
        logger.info(f"Model loaded in {load_time:.1f}s on {device}")
        
        return self
    
    def generate(self, input_text: str, domain: str = "general") -> tuple[str, float, int]:
        """Generate critique for input. Returns (critique, latency_ms, tokens)."""
        
        # Format prompt
        prompt = f"### Domain: {domain}\n### Rationale:\n{input_text}\n\n### Critique:\n"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(self.model.device)
        
        input_len = inputs["input_ids"].shape[1]
        
        # Generate with timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        latency_ms = (time.time() - start_time) * 1000
        
        # Decode
        generated = self.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        ).strip()
        
        tokens_generated = outputs.shape[1] - input_len
        
        return generated, latency_ms, tokens_generated
    
    def unload(self):
        """Free model from memory."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ================================================================
# Evaluation Logic
# ================================================================

def load_eval_data(eval_path: str) -> List[EvalExample]:
    """Load evaluation data from JSONL file."""
    examples = []
    
    with open(eval_path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            examples.append(EvalExample(
                id=f"eval_{i}",
                input=data.get('input', ''),
                expected_critique=data.get('critique', ''),
                domain=data.get('domain', 'general'),
                data_type=data.get('data_type', 'unknown'),
            ))
    
    logger.info(f"Loaded {len(examples)} evaluation examples")
    return examples


def is_false_positive(critique: str) -> bool:
    """
    Determine if critique indicates a false positive.
    
    For hard_negative examples (valid reasoning), the model should 
    indicate the reasoning is sound. If it flags issues, it's a false positive.
    """
    critique_lower = critique.lower()
    
    # Indicators that model found NO issues (correct for hard_negative)
    pass_indicators = [
        "sound", "valid", "correct", "well-reasoned", "appropriate",
        "reasonable", "logical", "no issues", "no concerns", "accurate",
        "well-supported", "coherent", "consistent",
    ]
    
    # Indicators that model found issues (false positive for hard_negative)
    fail_indicators = [
        "flaw", "error", "mistake", "incorrect", "invalid", "concern",
        "issue", "problem", "missing", "lacks", "fails", "overlook",
        "overconfident", "insufficient", "weak", "unsupported",
    ]
    
    pass_score = sum(1 for ind in pass_indicators if ind in critique_lower)
    fail_score = sum(1 for ind in fail_indicators if ind in critique_lower)
    
    # If more fail indicators than pass, it's a false positive
    return fail_score > pass_score


def evaluate_model(
    model: ModelLoader,
    examples: List[EvalExample],
    progress_desc: str = "Evaluating",
) -> tuple[List[ModelOutput], Dict[str, Any]]:
    """Run evaluation on all examples."""
    
    outputs = []
    latencies = []
    by_type = defaultdict(list)
    
    for example in tqdm(examples, desc=progress_desc):
        critique, latency_ms, tokens = model.generate(
            example.input,
            domain=example.domain,
        )
        
        output = ModelOutput(
            model_name=model.model_name,
            example_id=example.id,
            generated_critique=critique,
            latency_ms=latency_ms,
            tokens_generated=tokens,
        )
        outputs.append(output)
        latencies.append(latency_ms)
        
        # Track by type
        by_type[example.data_type].append({
            'example': example,
            'output': output,
            'is_false_positive': is_false_positive(critique) if example.data_type == 'hard_negative' else None,
        })
    
    # Compute metrics
    metrics = {
        'latencies': latencies,
        'by_type': dict(by_type),
    }
    
    return outputs, metrics


def compute_benchmark_result(
    model_name: str,
    outputs: List[ModelOutput],
    metrics: Dict[str, Any],
    model_size: str = "unknown",
) -> BenchmarkResult:
    """Compute aggregated benchmark metrics."""
    
    latencies = metrics['latencies']
    by_type = metrics['by_type']
    
    # False positive rate (hard_negative examples)
    hard_negatives = by_type.get('hard_negative', [])
    false_positives = sum(1 for x in hard_negatives if x['is_false_positive'])
    fp_rate = false_positives / len(hard_negatives) if hard_negatives else 0.0
    
    # Detection metrics by type
    detection_by_type = {}
    for dtype, items in by_type.items():
        detection_by_type[dtype] = {
            'count': len(items),
            'avg_latency_ms': statistics.mean([x['output'].latency_ms for x in items]) if items else 0,
        }
        if dtype == 'hard_negative':
            detection_by_type[dtype]['false_positive_rate'] = fp_rate
    
    # Memory usage
    peak_memory_gb = 0.0
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
    
    return BenchmarkResult(
        model_name=model_name,
        model_size=model_size,
        total_examples=len(outputs),
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=statistics.median(latencies),
        p95_latency_ms=np.percentile(latencies, 95),
        p99_latency_ms=np.percentile(latencies, 99),
        hard_negative_count=len(hard_negatives),
        false_positive_count=false_positives,
        false_positive_rate=fp_rate,
        detection_by_type=detection_by_type,
        peak_memory_gb=peak_memory_gb,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )

# ================================================================
# LLM-as-Judge Comparison
# ================================================================

def llm_judge_comparison(
    example: EvalExample,
    critique_1: str,
    critique_2: str,
    model_1_name: str,
    model_2_name: str,
) -> Dict[str, Any]:
    """
    Use Claude to judge which critique is better.
    
    Requires ANTHROPIC_API_KEY environment variable.
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed, skipping LLM judge")
        return None
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set, skipping LLM judge")
        return None
    
    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""You are evaluating two AI-generated critiques of reasoning. 

The original rationale being critiqued:
<rationale>
{example.input}
</rationale>

Domain: {example.domain}

Critique A:
<critique_a>
{critique_1}
</critique_a>

Critique B:
<critique_b>
{critique_2}
</critique_b>

Evaluate both critiques on these criteria:
1. Accuracy: Does the critique correctly identify real issues (or correctly identify valid reasoning)?
2. Specificity: Does the critique point to specific problems with concrete reasoning?
3. Helpfulness: Would this critique help someone improve their reasoning?
4. Calibration: Is the critique appropriately confident given the evidence?

Respond with JSON only:
{{
    "winner": "A" or "B" or "tie",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "scores": {{
        "A": {{"accuracy": 1-5, "specificity": 1-5, "helpfulness": 1-5, "calibration": 1-5}},
        "B": {{"accuracy": 1-5, "specificity": 1-5, "helpfulness": 1-5, "calibration": 1-5}}
    }}
}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        
        result_text = response.content[0].text
        # Parse JSON from response
        result = json.loads(result_text)
        result['model_a'] = model_1_name
        result['model_b'] = model_2_name
        return result
        
    except Exception as e:
        logger.error(f"LLM judge error: {e}")
        return None

# ================================================================
# Reporting
# ================================================================

def generate_report(
    result_1: BenchmarkResult,
    result_2: BenchmarkResult,
    judge_results: Optional[List[Dict]] = None,
    output_dir: str = "./benchmark_results",
):
    """Generate benchmark comparison report."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Summary comparison
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "models": {
            result_1.model_name: asdict(result_1),
            result_2.model_name: asdict(result_2),
        },
        "comparison": {
            "latency_improvement": f"{((result_1.avg_latency_ms - result_2.avg_latency_ms) / result_1.avg_latency_ms * 100):.1f}%",
            "fp_rate_improvement": f"{((result_1.false_positive_rate - result_2.false_positive_rate) / result_1.false_positive_rate * 100):.1f}%" if result_1.false_positive_rate > 0 else "N/A",
        },
    }
    
    # Add judge results if available
    if judge_results:
        wins = defaultdict(int)
        for r in judge_results:
            if r:
                winner = r.get('winner', 'tie')
                if winner == 'A':
                    wins[result_1.model_name] += 1
                elif winner == 'B':
                    wins[result_2.model_name] += 1
                else:
                    wins['tie'] += 1
        
        summary['llm_judge'] = {
            'total_comparisons': len([r for r in judge_results if r]),
            'wins': dict(wins),
        }
    
    # Save JSON report
    report_path = os.path.join(output_dir, "benchmark_report.json")
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    
    print(f"\n{'Metric':<30} {result_1.model_name:<20} {result_2.model_name:<20}")
    print("-" * 70)
    print(f"{'Total Examples':<30} {result_1.total_examples:<20} {result_2.total_examples:<20}")
    print(f"{'Avg Latency (ms)':<30} {result_1.avg_latency_ms:<20.1f} {result_2.avg_latency_ms:<20.1f}")
    print(f"{'P95 Latency (ms)':<30} {result_1.p95_latency_ms:<20.1f} {result_2.p95_latency_ms:<20.1f}")
    print(f"{'False Positive Rate':<30} {result_1.false_positive_rate:<20.2%} {result_2.false_positive_rate:<20.2%}")
    print(f"{'Peak Memory (GB)':<30} {result_1.peak_memory_gb:<20.1f} {result_2.peak_memory_gb:<20.1f}")
    
    if judge_results:
        print(f"\n{'LLM Judge Wins':<30} {wins.get(result_1.model_name, 0):<20} {wins.get(result_2.model_name, 0):<20}")
    
    print("\n" + "=" * 60)
    print(f"📁 Full report saved to: {report_path}")
    
    return summary

# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark MiniCrit models")
    parser.add_argument("--eval-data", required=True, help="Path to eval_holdout.jsonl")
    parser.add_argument("--model-1", default=DEFAULT_MODEL_1, help="First model (baseline)")
    parser.add_argument("--model-2", default=DEFAULT_MODEL_2, help="Second model (challenger)")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Output directory")
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu, mps)")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples (for testing)")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-judge comparison")
    parser.add_argument("--judge-sample", type=int, default=50, help="Number of examples for LLM judge")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔬 MiniCrit Model Benchmark")
    print("   Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3")
    print("=" * 60)
    
    # Load evaluation data
    examples = load_eval_data(args.eval_data)
    if args.limit:
        examples = examples[:args.limit]
        logger.info(f"Limited to {len(examples)} examples")
    
    # Evaluate Model 1
    logger.info(f"Evaluating Model 1: {args.model_1}")
    model_1 = ModelLoader(args.model_1, device=args.device).load()
    outputs_1, metrics_1 = evaluate_model(model_1, examples, f"Model 1 ({model_1.model_name})")
    result_1 = compute_benchmark_result(model_1.model_name, outputs_1, metrics_1, "1.5B")
    model_1.unload()
    
    # Evaluate Model 2
    logger.info(f"Evaluating Model 2: {args.model_2}")
    model_2 = ModelLoader(args.model_2, device=args.device).load()
    outputs_2, metrics_2 = evaluate_model(model_2, examples, f"Model 2 ({model_2.model_name})")
    result_2 = compute_benchmark_result(model_2.model_name, outputs_2, metrics_2, "7B")
    model_2.unload()
    
    # LLM-as-Judge comparison
    judge_results = []
    if not args.skip_judge and os.environ.get("ANTHROPIC_API_KEY"):
        logger.info(f"Running LLM-as-judge on {args.judge_sample} examples")
        
        # Sample examples for judging
        import random
        judge_examples = random.sample(list(zip(examples, outputs_1, outputs_2)), 
                                       min(args.judge_sample, len(examples)))
        
        for example, out_1, out_2 in tqdm(judge_examples, desc="LLM Judge"):
            result = llm_judge_comparison(
                example,
                out_1.generated_critique,
                out_2.generated_critique,
                model_1.model_name,
                model_2.model_name,
            )
            judge_results.append(result)
            time.sleep(0.5)  # Rate limiting
    
    # Generate report
    generate_report(result_1, result_2, judge_results, args.output_dir)
    
    # Save detailed outputs
    outputs_path = os.path.join(args.output_dir, "detailed_outputs.jsonl")
    with open(outputs_path, 'w') as f:
        for ex, out_1, out_2 in zip(examples, outputs_1, outputs_2):
            f.write(json.dumps({
                'example_id': ex.id,
                'input': ex.input,
                'domain': ex.domain,
                'data_type': ex.data_type,
                'model_1_critique': out_1.generated_critique,
                'model_1_latency_ms': out_1.latency_ms,
                'model_2_critique': out_2.generated_critique,
                'model_2_latency_ms': out_2.latency_ms,
            }) + '\n')
    
    logger.info(f"Detailed outputs saved to: {outputs_path}")
    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()
