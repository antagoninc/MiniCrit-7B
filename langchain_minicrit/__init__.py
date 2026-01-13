# ================================================================
# L1-HEADER: MiniCrit LangChain Integration Package
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================
# Official LangChain integration for MiniCrit adversarial reasoning.
# Provides LLM, Chat, Validator, and Callback components.
# ================================================================
# Copyright (c) 2024-2025 Antagon Inc. All rights reserved.
# Licensed under the Antagon Proprietary License.
# ================================================================

"""
L2-DOCSTRING: MiniCrit LangChain Integration Package.

This package provides official LangChain integration for the MiniCrit
adversarial reasoning system. It enables seamless integration of MiniCrit
critique generation into LangChain pipelines and applications.

Components:
    - MiniCritLLM: Base LLM wrapper for critique generation
    - MiniCritChat: Chat model for conversational critique
    - MiniCritValidator: Validation chain for reasoning critique
    - MiniCritCallbackHandler: Callback handler for critique events

Example:
    >>> from langchain_minicrit import MiniCritLLM, MiniCritValidator
    >>> llm = MiniCritLLM(base_url="http://localhost:8080/v1")
    >>> validator = MiniCritValidator(llm=llm)
    >>> result = validator.validate("AAPL is bullish because...")

Developed by Antagon Inc. as part of the MiniCrit adversarial reasoning system.

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

# ANTAGON-MINICRIT: Package version
__version__ = "0.1.0"
__author__ = "Antagon Inc."
__email__ = "engineering@antagon.dev"

# ANTAGON-MINICRIT: Import core components
from langchain_minicrit.llm import MiniCritLLM
from langchain_minicrit.chat import MiniCritChat
from langchain_minicrit.validator import MiniCritValidator, MiniCritValidationChain
from langchain_minicrit.callbacks import MiniCritCallbackHandler

# ANTAGON-MINICRIT: Public API
__all__ = [
    "MiniCritLLM",
    "MiniCritChat",
    "MiniCritValidator",
    "MiniCritValidationChain",
    "MiniCritCallbackHandler",
    "__version__",
]


# ================================================================
# L5-COMMENT: End of MiniCrit LangChain Package Init
# ANTAGON-MINICRIT: All rights reserved.
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================
