# MiniCrit Model Excellence Guide
## Making the Best Possible AI Validation Model

**Antagon Inc.** | CAGE: 17E75 | UEI: KBSGT7CZ4AH3

---

## 🎯 The Goal

Transform MiniCrit-7B from "good" to "best-in-class" adversarial AI validator.

**Current State:** Training at 64%, loss 0.440
**Target State:** Industry-leading critique quality with measurable superiority

---

## 📊 What "Best" Means (Metrics)

| Metric | Current Target | Excellence Target |
|--------|----------------|-------------------|
| False Positive Rate | <15% | <8% |
| Flaw Detection F1 | >0.80 | >0.90 |
| Operator Agreement | 95% | 98% |
| Sharpe Improvement | +0.28 | +0.40 |
| Inference Latency | <50ms | <30ms |

---

## 🛠️ Improvement Strategies

### Phase 1: RIGHT NOW (While Training Runs)

#### 1.1 Generate Harder Training Data

**Why:** Current data may be too easy. Model needs adversarial examples.

```bash
# Generate 10K hard examples with Ollama
python generate_hard_examples.py \
    --output hard_examples_10k.jsonl \
    --count 10000

# Or use Claude for higher quality (costs ~$5-10)
python generate_hard_examples.py \
    --output hard_examples_10k.jsonl \
    --count 10000 \
    --anthropic
```

**Hard Example Types:**
- Subtle flaws (almost correct)
- Sophisticated valid (complex but correct)
- Mixed signals (some good, some bad)
- Cognitive biases (anchoring, confirmation, survivorship)
- Overconfident wrong
- Uncertain correct

#### 1.2 Create Adversarial Test Suite

Test cases specifically designed to break the model:

```python
ADVERSARIAL_TESTS = [
    # Should NOT flag (false positive traps)
    "After analyzing 50 data points over 6 months with R²=0.89, controlling for seasonality and market conditions, I estimate 15% growth with 70% confidence.",
    
    # Should flag (subtle flaws)
    "Based on the last 3 successful trades using this pattern, I'm 95% confident it will work again.",  # Survivorship bias
    
    # Edge cases
    "Given the data, probability is either 0% or 100%, nothing in between.",  # False dichotomy
]
```

#### 1.3 Prepare DPO Data Pipeline

When training finishes, you'll want to immediately run DPO:

```bash
# Generate preference pairs from eval data
python generate_dpo_data.py \
    --input ~/Desktop/domain_data_clean/eval_holdout.jsonl \
    --output dpo_pairs.jsonl \
    --model /path/to/minicrit-7b \
    --candidates 4
```

---

### Phase 2: AFTER BASE TRAINING (Days 1-3)

#### 2.1 Comprehensive Benchmarking

```bash
# Full benchmark with LLM judge
export ANTHROPIC_API_KEY=your-key
python benchmark_models.py \
    --eval-data ~/Desktop/domain_data_clean/eval_holdout.jsonl \
    --model-1 wmaousley/MiniCrit-1.5B \
    --model-2 /path/to/minicrit-7b \
    --judge-sample 200
```

**Analyze:**
- Where does 7B beat 1.5B?
- Where does it fail?
- What patterns cause errors?

#### 2.2 Error Analysis

```python
# Categorize failures
failure_types = {
    "false_positive": [],      # Flagged valid reasoning
    "false_negative": [],      # Missed actual flaws
    "wrong_flaw_type": [],     # Identified flaw but wrong category
    "poor_explanation": [],    # Correct assessment, bad critique
}
```

#### 2.3 DPO Fine-tuning

Direct Preference Optimization improves without reward model:

```bash
# Generate preference data
python generate_dpo_data.py \
    --input training_data.jsonl \
    --output dpo_pairs.jsonl \
    --model minicrit-7b \
    --llm-score  # Use Claude for quality scoring

# Run DPO training (using TRL library)
python train_dpo.py \
    --model minicrit-7b \
    --data dpo_pairs.jsonl \
    --output minicrit-7b-dpo
```

---

### Phase 3: ITERATIVE IMPROVEMENT (Week 2+)

#### 3.1 Rejection Sampling Fine-tuning (RSF)

Generate many outputs, keep only the best:

```python
def rejection_sampling_finetune(model, data, n_samples=8):
    """
    1. For each input, generate n_samples outputs
    2. Score each with Claude or reward model
    3. Keep only top 25%
    4. Fine-tune on best outputs
    """
    best_outputs = []
    for example in data:
        candidates = [model.generate(example) for _ in range(n_samples)]
        scores = [score_critique(c) for c in candidates]
        best = candidates[np.argmax(scores)]
        best_outputs.append(best)
    
    return finetune(model, best_outputs)
```

#### 3.2 Domain-Specific LoRA Adapters

Train specialized adapters for each domain:

```
minicrit-7b-base
├── minicrit-7b-trading (LoRA)
├── minicrit-7b-defense (LoRA)
├── minicrit-7b-cyber (LoRA)
└── minicrit-7b-medical (LoRA)
```

**Benefits:**
- Better domain expertise
- Smaller deployment (swap adapters)
- Faster iteration per domain

#### 3.3 Constitutional AI Self-Improvement

Train model to critique and improve its own critiques:

```python
CONSTITUTION = """
A good critique should:
1. Be specific about what is wrong
2. Explain WHY it's problematic
3. Suggest how to fix it
4. Calibrate confidence appropriately
5. Acknowledge valid aspects

A good critique should NOT:
1. Be vague or generic
2. Flag valid reasoning as flawed
3. Be overconfident about uncertain issues
4. Miss obvious problems
5. Use first-person language
"""

# Self-improvement loop
critique_v1 = model.generate(rationale)
critique_v2 = model.generate(f"Improve this critique: {critique_v1}\nGuidelines: {CONSTITUTION}")
# Train on (critique_v1, critique_v2) pairs where v2 is better
```

---

### Phase 4: PRODUCTION OPTIMIZATION

#### 4.1 Quantization

Reduce model size while maintaining quality:

```bash
# 8-bit quantization
python quantize_model.py \
    --model minicrit-7b-dpo \
    --bits 8 \
    --output minicrit-7b-int8

# 4-bit for edge deployment
python quantize_model.py \
    --model minicrit-7b-dpo \
    --bits 4 \
    --output minicrit-7b-int4
```

| Quantization | Size | Speed | Quality Loss |
|--------------|------|-------|--------------|
| BF16 (current) | 14GB | 1x | 0% |
| INT8 | 7GB | 1.5x | <1% |
| INT4 | 3.5GB | 2x | 2-3% |

#### 4.2 Speculative Decoding

Use small model to draft, large model to verify:

```python
# MiniCrit-1.5B drafts tokens
# MiniCrit-7B verifies/corrects
# 2-3x speedup with same quality
```

#### 4.3 Batching & Caching

```python
# Batch similar requests
# Cache common critique patterns
# Pre-compute embeddings for frequent inputs
```

---

## 📋 Action Checklist

### Right Now ⏰
- [ ] Start hard example generation (10K examples)
- [ ] Create adversarial test suite (100 edge cases)
- [ ] Prepare DPO data generation script
- [ ] Set up Claude API for scoring

### When Training Completes 🎯
- [ ] Download 7B checkpoint from Vista
- [ ] Run comprehensive benchmark vs 1.5B
- [ ] Error analysis on failures
- [ ] Generate DPO preference pairs

### Week 2 🔄
- [ ] Run DPO fine-tuning
- [ ] Train domain-specific adapters
- [ ] Re-benchmark after DPO
- [ ] Begin rejection sampling

### Week 3+ 🚀
- [ ] Constitutional AI self-improvement
- [ ] Quantization for deployment
- [ ] A/B test in production
- [ ] Customer feedback integration

---

## 🎯 Priority Matrix

| Action | Impact | Effort | Priority |
|--------|--------|--------|----------|
| Hard example generation | High | Low | **DO NOW** |
| Benchmark on completion | High | Low | **DO NOW** |
| DPO fine-tuning | Very High | Medium | **Week 1** |
| Domain adapters | High | Medium | **Week 2** |
| Rejection sampling | Medium | Medium | **Week 2** |
| Quantization | Medium | Low | **Week 3** |
| Constitutional AI | Medium | High | **Later** |

---

## 📊 Expected Improvements

| Technique | False Positive Rate | F1 Score | Notes |
|-----------|---------------------|----------|-------|
| Base 7B | ~12% | ~0.82 | Current training |
| + Hard examples | ~10% | ~0.85 | Better edge cases |
| + DPO | ~8% | ~0.88 | Preference learning |
| + Domain adapters | ~6% | ~0.91 | Specialized models |
| + Rejection sampling | ~5% | ~0.93 | Best-of-N training |

---

## 🧪 Quick Experiments to Try

### 1. Temperature Sensitivity
Does lower temperature improve accuracy?
```python
for temp in [0.3, 0.5, 0.7, 0.9]:
    results = evaluate(model, temp=temp)
    print(f"Temp {temp}: FP={results.fp_rate:.2%}")
```

### 2. Prompt Engineering
Test different prompt formats:
```python
prompts = [
    "### Rationale:\n{input}\n\n### Critique:\n",
    "Analyze this reasoning for flaws:\n{input}\n\nCritique:",
    "You are an expert critic. Find issues in:\n{input}\n\nAnalysis:",
]
```

### 3. Chain-of-Thought Critiques
Force step-by-step reasoning:
```python
prompt = """
Rationale: {input}

Think step by step:
1. What claim is being made?
2. What evidence supports it?
3. What evidence is missing?
4. Are there logical errors?
5. What's the overall assessment?

Critique:
"""
```

---

## 🔑 Key Insights

1. **Data quality > quantity** - 10K hard examples may beat 100K easy ones
2. **DPO is highest ROI** - Significant improvement with moderate effort
3. **Domain matters** - Trading model should know trading terms
4. **Calibration is crucial** - Overconfident critiques destroy trust
5. **Human feedback wins** - Eventually incorporate customer corrections

---

## 📞 Resources

**Papers:**
- DPO: "Direct Preference Optimization" (Rafailov et al.)
- RLHF: "Training language models to follow instructions"
- Constitutional AI: "Constitutional AI: Harmlessness from AI Feedback"

**Tools:**
- TRL library for DPO: `pip install trl`
- vLLM for fast inference: `pip install vllm`
- bitsandbytes for quantization: `pip install bitsandbytes`

---

*Focus on DPO first - it's the highest impact improvement you can make after base training completes.*
