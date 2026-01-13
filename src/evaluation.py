"""Evaluation metrics and benchmarking for MiniCrit models.

This module provides evaluation functionality including ROUGE scores,
BERTScore, and custom metrics for critique quality assessment.

Example:
    >>> from src.evaluation import evaluate_model, compute_metrics
    >>> results = evaluate_model(model, tokenizer, test_data)
    >>> print(f"ROUGE-L: {results['rouge_l']:.4f}")

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results.

    Attributes:
        rouge1: ROUGE-1 F1 score (unigram overlap).
        rouge2: ROUGE-2 F1 score (bigram overlap).
        rougeL: ROUGE-L F1 score (longest common subsequence).
        bert_score_precision: BERTScore precision.
        bert_score_recall: BERTScore recall.
        bert_score_f1: BERTScore F1.
        avg_length: Average generated response length in words.
        num_samples: Number of samples evaluated.
        predictions: List of generated predictions.
        references: List of reference texts.
    """

    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    bert_score_precision: float = 0.0
    bert_score_recall: float = 0.0
    bert_score_f1: float = 0.0
    avg_length: float = 0.0
    num_samples: int = 0
    predictions: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "rouge1": self.rouge1,
            "rouge2": self.rouge2,
            "rougeL": self.rougeL,
            "bert_score_precision": self.bert_score_precision,
            "bert_score_recall": self.bert_score_recall,
            "bert_score_f1": self.bert_score_f1,
            "avg_length": self.avg_length,
            "num_samples": self.num_samples,
        }

    def __str__(self) -> str:
        """Format results as string."""
        return (
            f"Evaluation Results (n={self.num_samples}):\n"
            f"  ROUGE-1: {self.rouge1:.4f}\n"
            f"  ROUGE-2: {self.rouge2:.4f}\n"
            f"  ROUGE-L: {self.rougeL:.4f}\n"
            f"  BERTScore P/R/F1: {self.bert_score_precision:.4f} / "
            f"{self.bert_score_recall:.4f} / {self.bert_score_f1:.4f}\n"
            f"  Avg Length: {self.avg_length:.1f} words"
        )


def compute_rouge_scores(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute ROUGE scores between predictions and references.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
    n-gram overlap between generated and reference texts.

    Args:
        predictions: List of generated texts.
        references: List of reference texts.

    Returns:
        Dictionary with rouge1, rouge2, and rougeL F1 scores.

    Example:
        >>> scores = compute_rouge_scores(["generated text"], ["reference text"])
        >>> print(f"ROUGE-L: {scores['rougeL']:.4f}")
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.warning("rouge_score not installed. Install with: pip install rouge-score")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
        "rouge2": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
        "rougeL": sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
    }


def compute_bert_score(
    predictions: list[str],
    references: list[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 32,
) -> dict[str, float]:
    """Compute BERTScore between predictions and references.

    BERTScore uses contextual embeddings to measure semantic similarity
    between generated and reference texts, which is more robust than
    n-gram based metrics like ROUGE.

    Args:
        predictions: List of generated texts.
        references: List of reference texts.
        model_type: Pretrained model for BERTScore computation.
        batch_size: Batch size for processing.

    Returns:
        Dictionary with precision, recall, and f1 scores.

    Example:
        >>> scores = compute_bert_score(["generated"], ["reference"])
        >>> print(f"BERTScore F1: {scores['f1']:.4f}")
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        logger.warning("bert_score not installed. Install with: pip install bert-score")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    try:
        P, R, F1 = bert_score_fn(
            predictions,
            references,
            model_type=model_type,
            batch_size=batch_size,
            verbose=False,
        )

        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item(),
        }
    except Exception as e:
        logger.error(f"BERTScore computation failed: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def generate_critique(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    rationale: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """Generate a critique for a given rationale.

    Args:
        model: The fine-tuned model.
        tokenizer: Tokenizer for the model.
        rationale: Input rationale to critique.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        do_sample: Whether to use sampling (vs greedy).

    Returns:
        Generated critique text.

    Example:
        >>> critique = generate_critique(model, tokenizer, "AAPL is bullish...")
        >>> print(critique)
    """
    prompt = f"### Rationale:\n{rationale}\n\n### Critique:\n"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(  # type: ignore[operator]
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and extract just the critique part
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract critique after the prompt
    if "### Critique:" in full_output:
        critique = full_output.split("### Critique:")[-1].strip()
    else:
        critique = full_output[len(prompt) :].strip()

    return critique


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_data: list[dict[str, str]],
    max_samples: int | None = None,
    max_new_tokens: int = 256,
    compute_bertscore: bool = True,
) -> EvaluationResult:
    """Evaluate model on test data with multiple metrics.

    Generates critiques for test rationales and computes ROUGE and
    BERTScore metrics against reference critiques.

    Args:
        model: Fine-tuned model to evaluate.
        tokenizer: Tokenizer for the model.
        test_data: List of dicts with 'rationale' and 'critique' keys.
        max_samples: Maximum number of samples to evaluate (for speed).
        max_new_tokens: Maximum tokens to generate per sample.
        compute_bertscore: Whether to compute BERTScore (slower but better).

    Returns:
        EvaluationResult containing all metrics.

    Example:
        >>> test_data = [{"rationale": "...", "critique": "..."}]
        >>> results = evaluate_model(model, tokenizer, test_data)
        >>> print(results)
    """
    if max_samples:
        test_data = test_data[:max_samples]

    predictions = []
    references = []

    logger.info(f"Evaluating on {len(test_data)} samples...")

    model.eval()

    for i, sample in enumerate(test_data):
        rationale = sample.get("rationale") or sample.get("text") or sample.get("input")
        reference = sample.get("critique") or sample.get("rebuttal") or sample.get("output")

        if not rationale or not reference:
            continue

        critique = generate_critique(model, tokenizer, rationale, max_new_tokens=max_new_tokens)

        predictions.append(critique)
        references.append(reference)

        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(test_data)} samples")

    if not predictions:
        logger.error("No valid predictions generated")
        return EvaluationResult()

    # Compute metrics
    logger.info("Computing ROUGE scores...")
    rouge_scores = compute_rouge_scores(predictions, references)

    bert_scores = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if compute_bertscore:
        logger.info("Computing BERTScore (this may take a while)...")
        bert_scores = compute_bert_score(predictions, references)

    # Compute average length
    avg_length = sum(len(p.split()) for p in predictions) / len(predictions)

    result = EvaluationResult(
        rouge1=rouge_scores["rouge1"],
        rouge2=rouge_scores["rouge2"],
        rougeL=rouge_scores["rougeL"],
        bert_score_precision=bert_scores["precision"],
        bert_score_recall=bert_scores["recall"],
        bert_score_f1=bert_scores["f1"],
        avg_length=avg_length,
        num_samples=len(predictions),
        predictions=predictions,
        references=references,
    )

    logger.info(f"\n{result}")

    return result


def save_evaluation_results(
    results: EvaluationResult,
    output_path: str | Path,
    include_samples: bool = False,
) -> None:
    """Save evaluation results to JSON file.

    Args:
        results: Evaluation results to save.
        output_path: Path to save JSON file.
        include_samples: Whether to include predictions/references in output.

    Example:
        >>> save_evaluation_results(results, "eval_results.json")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = results.to_dict()

    if include_samples:
        data["predictions"] = results.predictions
        data["references"] = results.references

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved evaluation results to {output_path}")


def load_test_data(data_path: str | Path) -> list[dict[str, str]]:
    """Load test data from CSV or JSON file.

    Args:
        data_path: Path to test data file.

    Returns:
        List of dictionaries with rationale and critique fields.

    Example:
        >>> test_data = load_test_data("test.csv")
    """
    data_path = Path(data_path)

    if data_path.suffix == ".json":
        with open(data_path) as f:
            data = json.load(f)
        return data  # type: ignore[no-any-return]

    elif data_path.suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(data_path)

        # Find columns
        text_col = None
        for col in ["text", "rationale", "input", "prompt"]:
            if col in df.columns:
                text_col = col
                break

        rebuttal_col = None
        for col in ["rebuttal", "critique", "response", "output"]:
            if col in df.columns:
                rebuttal_col = col
                break

        if not text_col or not rebuttal_col:
            raise ValueError(f"Could not find required columns in {data_path}")

        return (  # type: ignore[no-any-return]
            df[[text_col, rebuttal_col]]
            .rename(columns={text_col: "rationale", rebuttal_col: "critique"})
            .to_dict("records")
        )

    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
