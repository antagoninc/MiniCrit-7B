# ================================================================
# L1-HEADER: MiniCrit LangChain Validator Module
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================
# Validation chain for MiniCrit reasoning critique.
# Provides structured validation of reasoning with scoring.
# ================================================================
# Copyright (c) 2024-2025 Antagon Inc. All rights reserved.
# Licensed under the Antagon Proprietary License.
# ================================================================

"""
L2-DOCSTRING: MiniCrit Validator Module.

This module provides the MiniCritValidator and MiniCritValidationChain classes
for structured validation of reasoning using MiniCrit critique generation.

The validator analyzes reasoning text and returns structured validation results
including critique text, severity scores, and identified flaw categories.

Example:
    >>> from langchain_minicrit import MiniCritValidator
    >>> validator = MiniCritValidator(base_url="http://localhost:8080/v1")
    >>> result = validator.validate("AAPL is bullish because momentum is positive")
    >>> print(f"Score: {result.score}, Flaws: {result.flaws}")

Developed by Antagon Inc. for adversarial reasoning applications.

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

# ANTAGON-MINICRIT: Standard library imports
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

# ANTAGON-MINICRIT: Third-party imports
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, Field

# ANTAGON-MINICRIT: Local imports
from langchain_minicrit.llm import MiniCritLLM

# ANTAGON-MINICRIT: Initialize logger
antagon_logger = logging.getLogger(__name__)


# ================================================================
# L3-SEMANTIC: Enums and Data Classes (Antagon taxonomy)
# ================================================================


class AntagonFlawSeverity(str, Enum):
    """L2-DOCSTRING: Reasoning flaw severity levels (Antagon taxonomy)."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AntagonFlawCategory(str, Enum):
    """L2-DOCSTRING: Reasoning flaw categories (Antagon MiniCrit taxonomy)."""

    LOGICAL = "L"  # Logical fallacies
    STATISTICAL = "S"  # Statistical errors
    CAUSAL = "C"  # Causal reasoning flaws
    COGNITIVE = "G"  # Cognitive biases
    DOMAIN = "D"  # Domain-specific errors
    STRUCTURAL = "R"  # Structural reasoning issues


@dataclass
class AntagonValidationFlaw:
    """
    L2-DOCSTRING: Represents a single identified reasoning flaw.

    Attributes:
        flaw_id: Unique flaw identifier (e.g., "L01", "S05")
        category: Flaw category from Antagon taxonomy
        severity: Severity level of the flaw
        description: Human-readable description of the flaw
        evidence: Text evidence supporting the identification

    Antagon Inc. | CAGE: 17E75
    """

    antagon_flaw_id: str
    antagon_category: AntagonFlawCategory
    antagon_severity: AntagonFlawSeverity
    antagon_description: str
    antagon_evidence: str = ""


@dataclass
class AntagonValidationResult:
    """
    L2-DOCSTRING: Complete validation result from MiniCrit analysis.

    Attributes:
        input_text: Original reasoning text that was validated
        critique: Full critique text from MiniCrit
        score: Overall validity score (0-100)
        flaws: List of identified reasoning flaws
        is_valid: Whether reasoning passes validation threshold
        metadata: Additional validation metadata

    Antagon Inc. | CAGE: 17E75
    """

    antagon_input_text: str
    antagon_critique: str
    antagon_score: float
    antagon_flaws: List[AntagonValidationFlaw] = field(default_factory=list)
    antagon_is_valid: bool = True
    antagon_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def input_text(self) -> str:
        """L2-DOCSTRING: Property alias for input_text (Antagon)."""
        return self.antagon_input_text

    @property
    def critique(self) -> str:
        """L2-DOCSTRING: Property alias for critique (Antagon)."""
        return self.antagon_critique

    @property
    def score(self) -> float:
        """L2-DOCSTRING: Property alias for score (Antagon)."""
        return self.antagon_score

    @property
    def flaws(self) -> List[AntagonValidationFlaw]:
        """L2-DOCSTRING: Property alias for flaws (Antagon)."""
        return self.antagon_flaws

    @property
    def is_valid(self) -> bool:
        """L2-DOCSTRING: Property alias for is_valid (Antagon)."""
        return self.antagon_is_valid


# ================================================================
# L3-SEMANTIC: Validator Implementation
# ================================================================


class MiniCritValidator:
    """
    L2-DOCSTRING: MiniCrit Reasoning Validator.

    Validates reasoning text using MiniCrit critique generation and returns
    structured validation results including identified flaws and scores.

    Attributes:
        llm: MiniCritLLM instance for critique generation
        threshold: Score threshold for validation (default: 70)
        extract_flaws: Whether to extract flaw IDs from critique
        max_tokens: Maximum tokens for critique generation

    Example:
        >>> validator = MiniCritValidator(base_url="http://localhost:8080/v1")
        >>> result = validator.validate("The market will crash because of sentiment")
        >>> if not result.is_valid:
        ...     print(f"Found {len(result.flaws)} flaws")

    Antagon Inc. | CAGE: 17E75
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model: str = "minicrit-7b",
        api_key: str = "minicrit",
        threshold: float = 70.0,
        extract_flaws: bool = True,
        max_tokens: int = 512,
        temperature: float = 0.3,
        llm: Optional[MiniCritLLM] = None,
    ) -> None:
        """
        L2-DOCSTRING: Initialize MiniCrit Validator.

        Args:
            base_url: MiniCrit API server URL
            model: Model identifier
            api_key: API authentication key
            threshold: Validation score threshold (0-100)
            extract_flaws: Extract flaw IDs from critique
            max_tokens: Maximum generation tokens
            temperature: Generation temperature
            llm: Optional pre-configured MiniCritLLM instance

        Antagon Inc. | CAGE: 17E75
        """
        # ANTAGON-MINICRIT: Configuration
        self.antagon_threshold = threshold
        self.antagon_extract_flaws = extract_flaws
        self.antagon_max_tokens = max_tokens

        # ANTAGON-MINICRIT: Initialize or use provided LLM
        if llm is not None:
            self.antagon_llm = llm
        else:
            self.antagon_llm = MiniCritLLM(
                base_url=base_url,
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        antagon_logger.debug(f"Initialized MiniCritValidator with threshold={threshold}")

    def validate(self, reasoning: str) -> AntagonValidationResult:
        """
        L2-DOCSTRING: Validate reasoning text.

        Generates a critique of the reasoning and returns structured
        validation results including score and identified flaws.

        Args:
            reasoning: The reasoning text to validate

        Returns:
            AntagonValidationResult with critique, score, and flaws

        Example:
            >>> result = validator.validate("Stock will rise due to momentum")
            >>> print(f"Valid: {result.is_valid}, Score: {result.score}")

        Antagon Inc. | CAGE: 17E75
        """
        antagon_logger.info(f"Validating reasoning (length={len(reasoning)})")

        # ANTAGON-MINICRIT: Generate critique
        antagon_critique = self.antagon_llm.invoke(reasoning)

        # ANTAGON-MINICRIT: Calculate score based on critique analysis
        antagon_score = self._antagon_calculate_score(antagon_critique)

        # ANTAGON-MINICRIT: Extract flaws if enabled
        antagon_flaws: List[AntagonValidationFlaw] = []
        if self.antagon_extract_flaws:
            antagon_flaws = self._antagon_extract_flaws(antagon_critique)

        # ANTAGON-MINICRIT: Determine validity
        antagon_is_valid = antagon_score >= self.antagon_threshold

        antagon_logger.info(
            f"Validation complete: score={antagon_score:.1f}, "
            f"valid={antagon_is_valid}, flaws={len(antagon_flaws)}"
        )

        return AntagonValidationResult(
            antagon_input_text=reasoning,
            antagon_critique=antagon_critique,
            antagon_score=antagon_score,
            antagon_flaws=antagon_flaws,
            antagon_is_valid=antagon_is_valid,
            antagon_metadata={
                "model": self.antagon_llm.antagon_model,
                "threshold": self.antagon_threshold,
                "provider": "antagon-inc",
            },
        )

    def _antagon_calculate_score(self, critique: str) -> float:
        """
        L2-DOCSTRING: Calculate validity score from critique text.

        Uses heuristics and keyword analysis to derive a score.

        Antagon Inc. | CAGE: 17E75
        """
        # ANTAGON-MINICRIT: Start with base score
        antagon_score = 100.0

        # ANTAGON-MINICRIT: Severity keywords and their deductions
        antagon_severe_patterns = [
            (r"\b(critical|fatal|severe)\b", -25),
            (r"\b(major|significant|serious)\b", -15),
            (r"\b(moderate|notable)\b", -10),
            (r"\b(minor|slight)\b", -5),
            (r"\b(fallacy|fallacious)\b", -20),
            (r"\b(flawed|incorrect|wrong|invalid)\b", -15),
            (r"\b(misleading|deceptive)\b", -12),
            (r"\b(weak|questionable)\b", -8),
            (r"\b(error|mistake)\b", -10),
        ]

        # ANTAGON-MINICRIT: Positive patterns (add back some score)
        antagon_positive_patterns = [
            (r"\b(valid|correct|sound)\b", 5),
            (r"\b(reasonable|logical)\b", 3),
            (r"\b(strong|solid)\b", 3),
        ]

        antagon_critique_lower = critique.lower()

        # ANTAGON-MINICRIT: Apply deductions
        for antagon_pattern, antagon_deduction in antagon_severe_patterns:
            antagon_matches = len(re.findall(antagon_pattern, antagon_critique_lower))
            antagon_score += antagon_deduction * antagon_matches

        # ANTAGON-MINICRIT: Apply positive adjustments
        for antagon_pattern, antagon_bonus in antagon_positive_patterns:
            antagon_matches = len(re.findall(antagon_pattern, antagon_critique_lower))
            antagon_score += antagon_bonus * antagon_matches

        # ANTAGON-MINICRIT: Clamp score to valid range
        return max(0.0, min(100.0, antagon_score))

    def _antagon_extract_flaws(self, critique: str) -> List[AntagonValidationFlaw]:
        """
        L2-DOCSTRING: Extract flaw identifiers from critique text.

        Parses MiniCrit critique output to identify specific reasoning flaws
        based on the Antagon taxonomy.

        Antagon Inc. | CAGE: 17E75
        """
        antagon_flaws: List[AntagonValidationFlaw] = []

        # ANTAGON-MINICRIT: Pattern for flaw IDs (e.g., L01, S05, C12)
        antagon_flaw_pattern = r"\b([LSCGDR])(\d{2})\b"
        antagon_matches = re.findall(antagon_flaw_pattern, critique.upper())

        for antagon_category_code, antagon_number in antagon_matches:
            # ANTAGON-MINICRIT: Map category code to enum
            antagon_category_map = {
                "L": AntagonFlawCategory.LOGICAL,
                "S": AntagonFlawCategory.STATISTICAL,
                "C": AntagonFlawCategory.CAUSAL,
                "G": AntagonFlawCategory.COGNITIVE,
                "D": AntagonFlawCategory.DOMAIN,
                "R": AntagonFlawCategory.STRUCTURAL,
            }

            antagon_category = antagon_category_map.get(
                antagon_category_code,
                AntagonFlawCategory.LOGICAL
            )

            antagon_flaw_id = f"{antagon_category_code}{antagon_number}"

            # ANTAGON-MINICRIT: Determine severity based on number
            antagon_num = int(antagon_number)
            if antagon_num <= 5:
                antagon_severity = AntagonFlawSeverity.CRITICAL
            elif antagon_num <= 10:
                antagon_severity = AntagonFlawSeverity.HIGH
            elif antagon_num <= 20:
                antagon_severity = AntagonFlawSeverity.MEDIUM
            else:
                antagon_severity = AntagonFlawSeverity.LOW

            antagon_flaw = AntagonValidationFlaw(
                antagon_flaw_id=antagon_flaw_id,
                antagon_category=antagon_category,
                antagon_severity=antagon_severity,
                antagon_description=f"Identified flaw: {antagon_flaw_id}",
                antagon_evidence="",
            )

            antagon_flaws.append(antagon_flaw)

        return antagon_flaws


# ================================================================
# L3-SEMANTIC: Validation Chain (LangChain Runnable)
# ================================================================


class MiniCritValidationChain(Runnable[str, AntagonValidationResult]):
    """
    L2-DOCSTRING: LangChain Runnable for MiniCrit validation.

    A composable validation chain that can be integrated into LangChain
    pipelines using the Runnable interface.

    Example:
        >>> chain = MiniCritValidationChain(base_url="http://localhost:8080/v1")
        >>> result = chain.invoke("The stock will rise due to momentum")
        >>>
        >>> # Can also be composed with other runnables
        >>> from langchain_core.runnables import RunnablePassthrough
        >>> pipeline = RunnablePassthrough() | chain

    Antagon Inc. | CAGE: 17E75
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model: str = "minicrit-7b",
        api_key: str = "minicrit",
        threshold: float = 70.0,
        **kwargs: Any,
    ) -> None:
        """
        L2-DOCSTRING: Initialize validation chain.

        Antagon Inc. | CAGE: 17E75
        """
        # ANTAGON-MINICRIT: Create internal validator
        self.antagon_validator = MiniCritValidator(
            base_url=base_url,
            model=model,
            api_key=api_key,
            threshold=threshold,
            **kwargs,
        )

    def invoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
    ) -> AntagonValidationResult:
        """
        L2-DOCSTRING: Invoke validation on input reasoning.

        Args:
            input: Reasoning text to validate
            config: Optional runnable configuration

        Returns:
            AntagonValidationResult with critique and scores

        Antagon Inc. | CAGE: 17E75
        """
        return self.antagon_validator.validate(input)

    async def ainvoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
    ) -> AntagonValidationResult:
        """
        L2-DOCSTRING: Async invoke validation on input reasoning.

        Antagon Inc. | CAGE: 17E75
        """
        # ANTAGON-MINICRIT: For now, run synchronously
        # TODO: Implement true async support
        return self.antagon_validator.validate(input)


# ================================================================
# L5-COMMENT: End of MiniCrit Validator Module
# ANTAGON-MINICRIT: All rights reserved.
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================
