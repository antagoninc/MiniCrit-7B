#!/usr/bin/env python3
# ================================================================
# DPO Preference Data Generator
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# Co-Founder & CEO: William Alexander Ousley (Alex Ousley)
# Co-Founder & CTO: Jacqueline Villamor Ousley (Jacque Ousley) TS/SCI
# ================================================================
# WATERMARK Layer 1: Antagon Inc. Proprietary
# WATERMARK Layer 2: MiniCrit DPO Pipeline
# WATERMARK Layer 3: Preference Learning
# WATERMARK Layer 4: Hash SHA256:DPO_GEN_2026
# WATERMARK Layer 5: Build 20260112
# ================================================================

"""
Generate preference pairs for Direct Preference Optimization (DPO).

DPO trains the model to prefer good critiques over bad ones without
needing a separate reward model.

For each input:
1. Generate multiple critique candidates
2. Rank them (via LLM judge or rules)
3. Create (chosen, rejected) pairs

Usage:
    python generate_dpo_data.py \
        --input eval_holdout.jsonl \
        --output dpo_pairs.jsonl \
        --model /path/to/minicrit-7b

Requires:
    pip install torch transformers anthropic
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ================================================================
# Configuration
# ================================================================

NUM_CANDIDATES = 4  # Generate this many critiques per input
TEMPERATURE_RANGE = (0.6, 1.0)  # Vary temperature for diversity

# ================================================================
# Data Classes
# ================================================================

@dataclass
class DPOPair:
    """A single DPO training pair."""
    prompt: str
    chosen: str
    rejected: str
    domain: str
    chosen_score: float
    rejected_score: float
    
    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "domain": self.domain,
            "chosen_score": self.chosen_score,
            "rejected_score": self.rejected_score,
            "timestamp": datetime.utcnow().isoformat(),
        }

# ================================================================
# Critique Generation
# ================================================================

class CritiqueGenerator:
    """Generate multiple critique candidates for DPO."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        print(f"Loading model: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on {self.device}")
    
    def generate_candidates(
        self,
        rationale: str,
        domain: str,
        num_candidates: int = NUM_CANDIDATES,
    ) -> List[Tuple[str, float]]:
        """Generate multiple critique candidates with varying temperatures."""
        
        prompt = f"### Domain: {domain}\n### Rationale:\n{rationale}\n\n### Critique:\n"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        candidates = []
        
        for i in range(num_candidates):
            # Vary temperature for diversity
            temp = random.uniform(*TEMPERATURE_RANGE)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            critique = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            
            candidates.append((critique, temp))
        
        return candidates

# ================================================================
# Critique Scoring
# ================================================================

def score_critique_rules(critique: str, rationale: str, domain: str) -> float:
    """Rule-based scoring of critique quality."""
    score = 0.5  # Base score
    
    critique_lower = critique.lower()
    
    # Positive signals
    if len(critique) > 100:  # Substantive
        score += 0.1
    if any(w in critique_lower for w in ["because", "since", "therefore", "specifically"]):
        score += 0.1  # Explains reasoning
    if any(w in critique_lower for w in ["however", "but", "although", "while"]):
        score += 0.05  # Nuanced
    if any(w in critique_lower for w in ["evidence", "data", "support"]):
        score += 0.1  # Evidence-based
    
    # Negative signals
    if len(critique) < 50:
        score -= 0.2  # Too short
    if critique.count("!") > 2:
        score -= 0.1  # Too emphatic
    if "I think" in critique or "I believe" in critique:
        score -= 0.1  # First person
    if critique_lower.startswith("this is"):
        score -= 0.05  # Weak opening
    
    # Domain-specific
    domain_terms = {
        "trading": ["risk", "return", "volatility", "position", "market"],
        "defense": ["threat", "asset", "mission", "operation"],
        "cybersecurity": ["vulnerability", "attack", "defense", "security"],
    }
    
    if domain in domain_terms:
        term_count = sum(1 for t in domain_terms[domain] if t in critique_lower)
        score += min(term_count * 0.05, 0.15)  # Domain expertise
    
    return max(0.0, min(1.0, score))


def score_critique_llm(
    critique: str,
    rationale: str,
    domain: str,
) -> float:
    """LLM-based scoring using Claude."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        
        prompt = f"""Rate this AI critique on a scale of 0-10.

Original rationale:
{rationale}

Domain: {domain}

Critique:
{critique}

Score based on:
- Accuracy (does it correctly identify issues or validate sound reasoning?)
- Specificity (does it point to concrete problems?)
- Helpfulness (would this help improve the reasoning?)
- Calibration (is confidence level appropriate?)

Respond with ONLY a number 0-10, nothing else."""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        
        score_text = response.content[0].text.strip()
        score = float(score_text) / 10.0
        return max(0.0, min(1.0, score))
        
    except Exception as e:
        print(f"LLM scoring error: {e}", file=sys.stderr)
        return score_critique_rules(critique, rationale, domain)


def score_critique(
    critique: str,
    rationale: str,
    domain: str,
    use_llm: bool = False,
) -> float:
    """Score a critique's quality."""
    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        return score_critique_llm(critique, rationale, domain)
    return score_critique_rules(critique, rationale, domain)

# ================================================================
# DPO Pair Creation
# ================================================================

def create_dpo_pairs(
    candidates: List[Tuple[str, float]],
    rationale: str,
    domain: str,
    use_llm: bool = False,
) -> List[DPOPair]:
    """Create DPO pairs from scored candidates."""
    
    # Score all candidates
    scored = []
    for critique, temp in candidates:
        score = score_critique(critique, rationale, domain, use_llm)
        scored.append((critique, score))
    
    # Sort by score
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Create pairs: best vs rest
    pairs = []
    best_critique, best_score = scored[0]
    
    prompt = f"### Domain: {domain}\n### Rationale:\n{rationale}\n\n### Critique:\n"
    
    for critique, score in scored[1:]:
        # Only pair if meaningful score difference
        if best_score - score >= 0.1:
            pairs.append(DPOPair(
                prompt=prompt,
                chosen=best_critique,
                rejected=critique,
                domain=domain,
                chosen_score=best_score,
                rejected_score=score,
            ))
    
    return pairs

# ================================================================
# Main Pipeline
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate DPO preference data")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", default="dpo_pairs.jsonl", help="Output file")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--device", default="auto", help="Device")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples")
    parser.add_argument("--llm-score", action="store_true", help="Use LLM for scoring")
    parser.add_argument("--candidates", type=int, default=4, help="Candidates per input")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 DPO Preference Data Generator")
    print("   Antagon Inc. | MiniCrit Training Pipeline")
    print("=" * 60)
    
    # Load generator
    generator = CritiqueGenerator(args.model, args.device)
    
    # Load input data
    examples = []
    with open(args.input, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    if args.limit:
        examples = examples[:args.limit]
    
    print(f"\nProcessing {len(examples)} examples...")
    print(f"Generating {args.candidates} candidates each")
    print(f"Scoring: {'LLM' if args.llm_score else 'Rules'}")
    
    # Generate pairs
    all_pairs = []
    
    for example in tqdm(examples, desc="Generating DPO pairs"):
        rationale = example.get("input", example.get("rationale", ""))
        domain = example.get("domain", "general")
        
        if not rationale:
            continue
        
        # Generate candidates
        candidates = generator.generate_candidates(
            rationale, domain, args.candidates
        )
        
        # Create pairs
        pairs = create_dpo_pairs(
            candidates, rationale, domain, args.llm_score
        )
        
        all_pairs.extend(pairs)
        
        # Rate limiting for LLM scoring
        if args.llm_score:
            time.sleep(0.5)
    
    # Save pairs
    with open(args.output, 'w') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair.to_dict()) + "\n")
    
    print(f"\n{'=' * 60}")
    print(f"✅ Complete!")
    print(f"   Input examples: {len(examples)}")
    print(f"   DPO pairs generated: {len(all_pairs)}")
    print(f"   Output: {args.output}")
    print("=" * 60)
    
    # Stats
    if all_pairs:
        avg_chosen = sum(p.chosen_score for p in all_pairs) / len(all_pairs)
        avg_rejected = sum(p.rejected_score for p in all_pairs) / len(all_pairs)
        print(f"\n📊 Score distribution:")
        print(f"   Avg chosen score: {avg_chosen:.3f}")
        print(f"   Avg rejected score: {avg_rejected:.3f}")
        print(f"   Avg margin: {avg_chosen - avg_rejected:.3f}")


if __name__ == "__main__":
    main()
