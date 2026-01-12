#!/usr/bin/env python3
# ================================================================
# MiniCrit CLI
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# Co-Founder & CEO: William Alexander Ousley (Alex Ousley)
# Co-Founder & CTO: Jacqueline Villamor Ousley (Jacque Ousley) TS/SCI
# ================================================================
# WATERMARK Layer 1: Antagon Inc. Proprietary
# WATERMARK Layer 2: MiniCrit CLI
# WATERMARK Layer 3: Command Line Interface
# WATERMARK Layer 4: Hash SHA256:CLI_2026
# WATERMARK Layer 5: Build 20260112
# ================================================================

"""
MiniCrit Command Line Interface

Usage:
    minicrit "Stock will rise because it rose yesterday"
    minicrit --domain trading "Buy signal based on MACD"
    minicrit --file rationales.txt --output results.json
    minicrit-server --port 8000
"""

import argparse
import json
import sys
from typing import Optional


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="minicrit",
        description="MiniCrit: Adversarial AI Validation",
        epilog="Antagon Inc. | https://antagon.ai",
    )
    
    parser.add_argument(
        "rationale",
        nargs="?",
        help="Rationale to validate",
    )
    parser.add_argument(
        "-d", "--domain",
        default="general",
        choices=[
            "trading", "finance", "defense", "cybersecurity",
            "medical", "risk_assessment", "planning", "general"
        ],
        help="Domain context (default: general)",
    )
    parser.add_argument(
        "-m", "--model",
        default="7b",
        choices=["7b", "1.5b"],
        help="Model size (default: 7b)",
    )
    parser.add_argument(
        "-f", "--file",
        help="File with rationales (one per line)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Show version",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)",
    )
    
    args = parser.parse_args()
    
    # Version
    if args.version:
        from minicrit import __version__
        print(f"minicrit {__version__}")
        return 0
    
    # Need rationale or file
    if not args.rationale and not args.file:
        parser.print_help()
        return 1
    
    # Import here to avoid slow startup for --help
    from minicrit import MiniCrit
    
    print("Loading MiniCrit...", file=sys.stderr)
    critic = MiniCrit(model=args.model, device=args.device)
    
    results = []
    
    # Single rationale
    if args.rationale:
        result = critic.validate(args.rationale, domain=args.domain)
        results.append(result)
    
    # File input
    if args.file:
        with open(args.file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    result = critic.validate(line, domain=args.domain)
                    results.append(result)
    
    # Output
    if args.json or args.output:
        output_data = [
            {
                "valid": r.valid,
                "severity": r.severity,
                "critique": r.critique,
                "confidence": r.confidence,
                "flags": r.flags,
                "domain": r.domain,
                "latency_ms": r.latency_ms,
            }
            for r in results
        ]
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Results saved to {args.output}", file=sys.stderr)
        else:
            print(json.dumps(output_data, indent=2))
    else:
        # Human-readable output
        for i, r in enumerate(results):
            if len(results) > 1:
                print(f"\n--- Result {i+1} ---")
            
            status = "✓ VALID" if r.valid else f"✗ {r.severity.upper()}"
            print(f"Status: {status}")
            print(f"Confidence: {r.confidence:.0%}")
            if r.flags:
                print(f"Flags: {', '.join(r.flags)}")
            print(f"Critique: {r.critique}")
            print(f"Latency: {r.latency_ms:.1f}ms")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
