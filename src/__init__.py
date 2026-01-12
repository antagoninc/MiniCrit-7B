# ================================================================
# MiniCrit - Adversarial AI Validation
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# Co-Founder & CEO: William Alexander Ousley (Alex Ousley)
# Co-Founder & CTO: Jacqueline Villamor Ousley (Jacque Ousley) TS/SCI
# ================================================================
# WATERMARK Layer 1: Antagon Inc. Proprietary
# WATERMARK Layer 2: MiniCrit Package
# WATERMARK Layer 3: PyPI Distribution
# WATERMARK Layer 4: Hash SHA256:PYPI_INIT_2026
# WATERMARK Layer 5: Build 20260112
# ================================================================

"""
MiniCrit: Adversarial AI Validation for Autonomous Systems

Catch reasoning flaws before they become failures.

Quick Start:
    >>> from minicrit import MiniCrit
    >>> critic = MiniCrit()
    >>> result = critic.validate("Stock will rise because it rose yesterday")
    >>> print(result.valid)  # False
    >>> print(result.severity)  # "high"
    >>> print(result.critique)  # "This reasoning exhibits recency bias..."

With domain context:
    >>> result = critic.validate(
    ...     "Buy signal based on MACD crossover",
    ...     domain="trading"
    ... )

Available domains:
    trading, finance, defense, cybersecurity, medical,
    risk_assessment, planning, general

For more info: https://github.com/antagoninc/MiniCrit-7B
"""

__version__ = "0.1.0"
__author__ = "Antagon Inc."
__email__ = "founders@antagon.ai"
__license__ = "Apache-2.0"

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class Severity(str, Enum):
    """Critique severity levels."""
    PASS = "pass"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CritiqueResult:
    """Result of MiniCrit validation."""
    valid: bool
    severity: str
    critique: str
    confidence: float
    flags: List[str]
    domain: str
    latency_ms: float
    
    def __repr__(self):
        status = "✓ VALID" if self.valid else f"✗ {self.severity.upper()}"
        return f"CritiqueResult({status}, flags={self.flags})"


class MiniCrit:
    """
    MiniCrit adversarial AI validator.
    
    Args:
        model: HuggingFace model ID or local path
            - "wmaousley/MiniCrit-7B" (default, best quality)
            - "wmaousley/MiniCrit-1.5B" (faster, smaller)
        device: Device to run on ("auto", "cuda", "mps", "cpu")
        
    Example:
        >>> critic = MiniCrit()
        >>> result = critic.validate("Buy AAPL, 95% confident")
        >>> if not result.valid:
        ...     print(f"Warning: {result.critique}")
    """
    
    DOMAINS = [
        "trading", "finance", "defense", "cybersecurity", 
        "medical", "risk_assessment", "planning", "general"
    ]
    
    # Default models
    DEFAULT_7B_ADAPTER = "wmaousley/MiniCrit-7B"
    DEFAULT_7B_BASE = "Qwen/Qwen2-7B-Instruct"
    DEFAULT_1_5B_ADAPTER = "wmaousley/MiniCrit-1.5B"
    DEFAULT_1_5B_BASE = "Qwen/Qwen2-0.5B-Instruct"
    
    def __init__(
        self,
        model: str = "7b",
        device: str = "auto",
        load_on_init: bool = True,
    ):
        """Initialize MiniCrit."""
        self.device = device
        self._model = None
        self._tokenizer = None
        
        # Select model
        if model == "7b" or model == self.DEFAULT_7B_ADAPTER:
            self.adapter_id = self.DEFAULT_7B_ADAPTER
            self.base_model_id = self.DEFAULT_7B_BASE
        elif model == "1.5b" or model == self.DEFAULT_1_5B_ADAPTER:
            self.adapter_id = self.DEFAULT_1_5B_ADAPTER
            self.base_model_id = self.DEFAULT_1_5B_BASE
        else:
            # Custom model path
            self.adapter_id = model
            self.base_model_id = None  # Must be set manually
        
        if load_on_init:
            self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        # Determine device
        if self.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.device
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, 
            trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Load base model
        dtype = torch.float16 if device == "mps" else torch.bfloat16
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        base_model = base_model.to(device)
        
        # Apply LoRA adapter
        self._model = PeftModel.from_pretrained(base_model, self.adapter_id)
        self._model.eval()
        
        self._device = device
    
    def validate(
        self,
        rationale: str,
        domain: str = "general",
        context: Optional[str] = None,
    ) -> CritiqueResult:
        """
        Validate AI reasoning.
        
        Args:
            rationale: The reasoning to validate
            domain: Domain context (trading, defense, etc.)
            context: Additional context (optional)
            
        Returns:
            CritiqueResult with valid, severity, critique, flags
        """
        import torch
        import time
        
        if self._model is None:
            self._load_model()
        
        start_time = time.time()
        
        # Format prompt
        prompt_parts = [f"### Domain: {domain}\n"]
        if context:
            prompt_parts.append(f"### Context:\n{context}\n")
        prompt_parts.append(f"### Rationale:\n{rationale}\n\n### Critique:\n")
        prompt = "".join(prompt_parts)
        
        # Tokenize
        inputs = self._tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        
        # Decode
        generated = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Parse result
        flags = []
        severity = Severity.PASS
        critique_lower = generated.lower()
        
        # Detect severity
        if any(w in critique_lower for w in ["critical", "severe", "dangerous"]):
            severity = Severity.CRITICAL
        elif any(w in critique_lower for w in ["significant", "major", "serious", "flawed"]):
            severity = Severity.HIGH
        elif any(w in critique_lower for w in ["concern", "issue", "problem", "missing"]):
            severity = Severity.MEDIUM
        elif any(w in critique_lower for w in ["minor", "slight"]):
            severity = Severity.LOW
        
        # Detect flags
        if any(w in critique_lower for w in ["overconfident", "overconfidence"]):
            flags.append("overconfidence")
        if any(w in critique_lower for w in ["missing", "omit", "neglect"]):
            flags.append("missing_consideration")
        if any(w in critique_lower for w in ["contradict", "inconsistent"]):
            flags.append("logical_inconsistency")
        if any(w in critique_lower for w in ["evidence", "support", "unsupported"]):
            flags.append("insufficient_evidence")
        if any(w in critique_lower for w in ["risk", "danger", "threat"]):
            flags.append("unaddressed_risk")
        
        valid = severity in [Severity.PASS, Severity.LOW]
        confidence = 0.85 if len(generated) >= 20 else 0.5
        latency_ms = (time.time() - start_time) * 1000
        
        return CritiqueResult(
            valid=valid,
            severity=severity.value,
            critique=generated,
            confidence=confidence,
            flags=flags,
            domain=domain,
            latency_ms=round(latency_ms, 2),
        )
    
    def batch_validate(
        self,
        rationales: List[str],
        domain: str = "general",
    ) -> List[CritiqueResult]:
        """Validate multiple rationales."""
        return [self.validate(r, domain=domain) for r in rationales]


# Convenience function
def validate(
    rationale: str,
    domain: str = "general",
    model: str = "7b",
) -> CritiqueResult:
    """
    Quick validation without instantiating MiniCrit.
    
    Note: Creates new model instance each call. 
    For multiple calls, use MiniCrit() class instead.
    
    Example:
        >>> from minicrit import validate
        >>> result = validate("Stock will rise because it rose yesterday")
    """
    critic = MiniCrit(model=model)
    return critic.validate(rationale, domain=domain)


__all__ = [
    "MiniCrit",
    "CritiqueResult", 
    "Severity",
    "validate",
    "__version__",
]
