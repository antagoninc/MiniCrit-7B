#!/usr/bin/env python3
# ================================================================
# MiniCrit Hard Example Generator - Claude Sonnet
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# Co-Founder & CEO: William Alexander Ousley (Alex Ousley)
# Co-Founder & CTO: Jacqueline Villamor Ousley (Jacque Ousley) TS/SCI
# ================================================================
# WATERMARK Layer 1: Antagon Inc. Proprietary
# WATERMARK Layer 2: MiniCrit Hard Example Generator v2
# WATERMARK Layer 3: Claude Sonnet Pipeline
# WATERMARK Layer 4: Hash SHA256:HARDGEN_SONNET_2026
# WATERMARK Layer 5: Build 20260112
# ================================================================

"""
Generate high-quality hard training examples using Claude Sonnet.

Focus areas:
1. sophisticated_valid - Complex correct reasoning (prevents false positives)
2. subtle_flaw - Almost-correct with hidden issues
3. confident_wrong - Overconfidence detection

Usage:
    export ANTHROPIC_API_KEY=your-key
    python generate_hard_sonnet.py --count 5000 --output hard_examples_sonnet.jsonl

Cost estimate: ~$30 for 5K examples

Requirements:
    pip install anthropic
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime
from typing import Optional
import time

try:
    import anthropic
except ImportError:
    print("Error: pip install anthropic", file=sys.stderr)
    sys.exit(1)

# ================================================================
# Configuration
# ================================================================

MODEL = "claude-sonnet-4-20250514"

# Focus on highest-value example types
EXAMPLE_TYPES = {
    "sophisticated_valid": 0.35,  # 35% - prevents false positives
    "subtle_flaw": 0.30,  # 30% - catches hard errors
    "confident_wrong": 0.15,  # 15% - overconfidence detection
    "mixed_signals": 0.10,  # 10% - nuanced critique
    "missing_context": 0.10,  # 10% - context awareness
}

DOMAINS = {
    "trading": {
        "context": "quantitative trading, algorithmic strategies, risk management",
        "scenarios": [
            "momentum strategy evaluation",
            "risk-adjusted return analysis",
            "position sizing decision",
            "market regime assessment",
            "drawdown risk evaluation",
        ],
    },
    "defense": {
        "context": "military planning, threat assessment, resource allocation, mission planning",
        "scenarios": [
            "threat level assessment",
            "asset deployment recommendation",
            "mission risk evaluation",
            "resource prioritization",
            "operational timeline planning",
        ],
    },
    "risk_assessment": {
        "context": "enterprise risk, probability assessment, impact analysis",
        "scenarios": [
            "project risk scoring",
            "failure probability estimation",
            "mitigation strategy evaluation",
            "contingency planning",
            "risk aggregation analysis",
        ],
    },
    "planning": {
        "context": "project management, resource scheduling, logistics optimization",
        "scenarios": [
            "timeline feasibility assessment",
            "resource allocation decision",
            "dependency analysis",
            "critical path evaluation",
            "buffer estimation",
        ],
    },
    "cybersecurity": {
        "context": "security operations, threat detection, incident response, vulnerability assessment",
        "scenarios": [
            "threat severity assessment",
            "incident prioritization",
            "vulnerability risk scoring",
            "defense resource allocation",
            "attack vector analysis",
        ],
    },
}

# ================================================================
# Prompts
# ================================================================

SYSTEM_PROMPT = """You are an expert at creating training data for an AI critique model called MiniCrit.

MiniCrit validates AI reasoning before actions are taken. It needs to:
1. Correctly identify flawed reasoning
2. NOT flag valid reasoning as flawed (avoid false positives)
3. Provide specific, actionable critiques

Your job is to create challenging examples that will make MiniCrit better at these tasks.

Always output valid JSON with these exact fields:
{
    "rationale": "The AI reasoning being evaluated (2-4 sentences, realistic and domain-specific)",
    "critique": "The appropriate critique response (2-3 sentences)",
    "domain": "the domain",
    "data_type": "the example type",
    "is_valid": true or false,
    "flaw_type": "specific flaw name or null if valid"
}"""

EXAMPLE_PROMPTS = {
    "sophisticated_valid": """Create a SOPHISTICATED VALID reasoning example.

Domain: {domain}
Context: {context}
Scenario: {scenario}

Requirements:
- The rationale must be COMPLEX and CORRECT
- Use domain-specific terminology and metrics
- Include appropriate confidence levels and caveats
- The critique should CONFIRM validity and acknowledge strengths
- This tests that MiniCrit doesn't flag good reasoning

Make it realistic - the kind of reasoning a competent professional would produce.
The critique should say something like "This reasoning is sound because..." and explain why.

Output JSON only.""",
    "subtle_flaw": """Create a SUBTLE FLAW reasoning example.

Domain: {domain}
Context: {context}
Scenario: {scenario}

Requirements:
- The rationale should APPEAR sound at first glance
- Hide a subtle logical error, missing assumption, or overlooked risk
- The flaw should require careful analysis to detect
- NOT obvious errors like "stock went up so it will go up"
- The critique should identify the specific subtle flaw

Good subtle flaws:
- Survivorship bias (only considering successes)
- Base rate neglect (ignoring priors)
- Confusing correlation with causation
- Cherry-picking time periods
- Missing interaction effects
- Overfitting to historical data

Output JSON only.""",
    "confident_wrong": """Create a CONFIDENT BUT WRONG reasoning example.

Domain: {domain}
Context: {context}
Scenario: {scenario}

Requirements:
- Express HIGH confidence (90%+, "certainly", "definitely", "clear that")
- But the reasoning has a clear logical or factual flaw
- The overconfidence IS the primary problem
- The critique should call out both the error AND the overconfidence

The flaw can be:
- Ignoring obvious counterexamples
- Extrapolating from insufficient data
- Circular reasoning
- False dichotomy
- Hasty generalization

Output JSON only.""",
    "mixed_signals": """Create a MIXED SIGNALS reasoning example.

Domain: {domain}
Context: {context}
Scenario: {scenario}

Requirements:
- Include BOTH valid points AND flawed points
- Some aspects of the reasoning are correct
- Other aspects have problems
- The critique should acknowledge strengths while identifying weaknesses
- This teaches nuanced evaluation

Output JSON only.""",
    "missing_context": """Create a MISSING CONTEXT reasoning example.

Domain: {domain}
Context: {context}
Scenario: {scenario}

Requirements:
- The reasoning is CORRECT given the information provided
- But it fails to consider important context that would change the conclusion
- The critique should identify what context is missing and why it matters

Examples of missing context:
- Market conditions not considered
- Regulatory constraints ignored
- Resource limitations overlooked
- Stakeholder concerns not addressed
- Time constraints not factored

Output JSON only.""",
}

# ================================================================
# Generation
# ================================================================


def generate_example(client: anthropic.Anthropic, example_type: str, domain: str) -> Optional[dict]:
    """Generate a single example using Claude Sonnet."""

    domain_info = DOMAINS[domain]
    scenario = random.choice(domain_info["scenarios"])

    prompt = EXAMPLE_PROMPTS[example_type].format(
        domain=domain,
        context=domain_info["context"],
        scenario=scenario,
    )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()

        # Clean up markdown if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        # Parse JSON
        result = json.loads(text)

        # Add metadata
        result["timestamp"] = datetime.utcnow().isoformat()
        result["generator"] = "claude-sonnet"
        result["scenario"] = scenario

        return dict(result)

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"API error: {e}", file=sys.stderr)
        return None


def select_example_type() -> str:
    """Weighted random selection of example type."""
    r = random.random()
    cumulative: float = 0.0
    for example_type, weight in EXAMPLE_TYPES.items():
        cumulative += weight
        if r <= cumulative:
            return example_type
    return list(EXAMPLE_TYPES.keys())[-1]


def validate_example(example: dict) -> bool:
    """Validate generated example quality."""
    required = ["rationale", "critique", "domain", "data_type", "is_valid"]

    for field in required:
        if field not in example:
            return False

    # Check minimum lengths
    if len(example.get("rationale", "")) < 80:
        return False
    if len(example.get("critique", "")) < 50:
        return False

    return True


# ================================================================
# Main
# ================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate hard examples with Claude Sonnet")
    parser.add_argument("--output", default="hard_examples_sonnet.jsonl", help="Output file")
    parser.add_argument("--count", type=int, default=5000, help="Number of examples")
    parser.add_argument("--resume", action="store_true", help="Resume from existing file")

    args = parser.parse_args()

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic()

    print("=" * 60)
    print("🎯 MiniCrit Hard Example Generator (Claude Sonnet)")
    print("   Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3")
    print("=" * 60)
    print(f"\nTarget: {args.count} examples")
    print(f"Output: {args.output}")
    print(f"Estimated cost: ~${args.count * 0.006:.2f}")
    print(f"\nExample type distribution:")
    for t, w in EXAMPLE_TYPES.items():
        print(f"  {t}: {w*100:.0f}%")

    # Resume support
    generated = 0
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r") as f:
            generated = sum(1 for _ in f)
        print(f"\nResuming from {generated} existing examples")

    failed = 0
    domains = list(DOMAINS.keys())

    mode = "a" if args.resume else "w"
    with open(args.output, mode) as f:
        while generated < args.count:
            # Select type and domain
            example_type = select_example_type()
            domain = random.choice(domains)

            print(
                f"\r[{generated}/{args.count}] Generating {example_type} ({domain})... ",
                end="",
                flush=True,
            )

            example = generate_example(client, example_type, domain)

            if example and validate_example(example):
                f.write(json.dumps(example) + "\n")
                generated += 1

                # Checkpoint
                if generated % 100 == 0:
                    f.flush()
                    print(
                        f"\n✅ Checkpoint: {generated} generated, {failed} failed, ~${generated * 0.006:.2f} spent"
                    )
            else:
                failed += 1

            # Rate limiting (Sonnet allows ~50 req/min on standard tier)
            time.sleep(1.2)

    print(f"\n\n{'=' * 60}")
    print(f"✅ Complete!")
    print(f"   Generated: {generated}")
    print(f"   Failed: {failed}")
    print(f"   Estimated cost: ~${generated * 0.006:.2f}")
    print(f"   Output: {args.output}")
    print("=" * 60)

    # Summary stats
    type_counts = {}
    domain_counts = {}
    valid_counts = {"true": 0, "false": 0}

    with open(args.output, "r") as f:
        for line in f:
            ex = json.loads(line)
            t = ex.get("data_type", "unknown")
            d = ex.get("domain", "unknown")
            v = str(ex.get("is_valid", "unknown")).lower()
            type_counts[t] = type_counts.get(t, 0) + 1
            domain_counts[d] = domain_counts.get(d, 0) + 1
            if v in valid_counts:
                valid_counts[v] += 1

    print(f"\n📊 Distribution:")
    print(f"   By type: {type_counts}")
    print(f"   By domain: {domain_counts}")
    print(f"   Valid/Invalid: {valid_counts}")


if __name__ == "__main__":
    main()
