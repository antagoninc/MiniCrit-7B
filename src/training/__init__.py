# ================================================================
# MiniCrit Training Utilities
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# Co-Founder & CEO: William Alexander Ousley (Alex Ousley)
# Co-Founder & CTO: Jacqueline Villamor Ousley (Jacque Ousley) TS/SCI
# ================================================================
# WATERMARK Layer 1: Antagon Inc. Proprietary
# WATERMARK Layer 2: MiniCrit Training Module
# WATERMARK Layer 3: DPO and Data Generation
# WATERMARK Layer 4: Hash SHA256:TRAIN_INIT_2026
# WATERMARK Layer 5: Build 20260112
# ================================================================

"""
MiniCrit Training Utilities

Scripts:
- generate_hard_examples.py: Generate challenging training data via Claude
- generate_dpo_data.py: Create preference pairs for DPO
- train_dpo.py: Direct Preference Optimization training

Usage:
    # Generate hard examples (~$30 for 5K)
    python -m src.training.generate_hard_examples --count 5000
    
    # Generate DPO pairs
    python -m src.training.generate_dpo_data --input eval.jsonl --model minicrit-7b
    
    # Run DPO training
    python -m src.training.train_dpo --model minicrit-7b --data dpo_pairs.jsonl
"""
