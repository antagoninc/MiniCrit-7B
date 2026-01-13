# ================================================================
# L1-HEADER: MiniCrit LangChain LLM Module
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================
# Base LLM wrapper for MiniCrit critique generation.
# Provides LangChain BaseLLM compatible interface.
# ================================================================
# Copyright (c) 2024-2025 Antagon Inc. All rights reserved.
# Licensed under the Antagon Proprietary License.
# ================================================================

"""
L2-DOCSTRING: MiniCrit LLM Module.

This module provides the MiniCritLLM class, a LangChain-compatible LLM wrapper
for the MiniCrit adversarial reasoning system. It connects to the MiniCrit
OpenAI-compatible API server for critique generation.

Example:
    >>> from langchain_minicrit import MiniCritLLM
    >>> llm = MiniCritLLM(base_url="http://localhost:8080/v1")
    >>> critique = llm.invoke("The stock will rise because momentum is positive")

Developed by Antagon Inc. for adversarial reasoning applications.

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

# ANTAGON-MINICRIT: Standard library imports
from __future__ import annotations

import logging
from typing import Any, Iterator, List, Mapping, Optional

# ANTAGON-MINICRIT: Third-party imports
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from pydantic import Field, PrivateAttr

# ANTAGON-MINICRIT: Initialize logger
antagon_logger = logging.getLogger(__name__)


# ================================================================
# L3-SEMANTIC: MiniCrit LLM Implementation
# ================================================================


class MiniCritLLM(LLM):
    """
    L2-DOCSTRING: LangChain LLM wrapper for MiniCrit critique generation.

    This class provides a LangChain-compatible interface to the MiniCrit
    adversarial reasoning system. It connects to the OpenAI-compatible
    API server for generating critiques of reasoning.

    Attributes:
        base_url: Base URL of the MiniCrit API server
        model: Model identifier to use for generation
        api_key: API key for authentication (optional)
        temperature: Sampling temperature for generation
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds

    Example:
        >>> llm = MiniCritLLM(
        ...     base_url="http://localhost:8080/v1",
        ...     model="minicrit-7b",
        ...     temperature=0.7
        ... )
        >>> critique = llm.invoke("AAPL is bullish due to momentum")
        >>> print(critique)

    Antagon Inc. | CAGE: 17E75
    """

    # ANTAGON-MINICRIT: Model configuration
    antagon_base_url: str = Field(
        default="http://localhost:8080/v1",
        description="Base URL of the MiniCrit API server",
        alias="base_url",
    )
    antagon_model: str = Field(
        default="minicrit-7b",
        description="Model identifier",
        alias="model",
    )
    antagon_api_key: str = Field(
        default="minicrit",
        description="API key for authentication",
        alias="api_key",
    )
    antagon_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
        alias="temperature",
    )
    antagon_max_tokens: int = Field(
        default=512,
        ge=1,
        le=4096,
        description="Maximum tokens to generate",
        alias="max_tokens",
    )
    antagon_timeout: int = Field(
        default=60,
        ge=1,
        description="Request timeout in seconds",
        alias="timeout",
    )
    antagon_top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter",
        alias="top_p",
    )

    # ANTAGON-MINICRIT: Private HTTP client
    _antagon_client: Any = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        """
        L2-DOCSTRING: Initialize MiniCrit LLM.

        Antagon Inc. | CAGE: 17E75
        """
        super().__init__(**kwargs)
        self._antagon_init_client()

    def _antagon_init_client(self) -> None:
        """
        L2-DOCSTRING: Initialize HTTP client (Antagon pattern).

        Antagon Inc. | CAGE: 17E75
        """
        # ANTAGON-MINICRIT: Lazy import httpx
        try:
            import httpx

            self._antagon_client = httpx.Client(
                base_url=self.antagon_base_url,
                timeout=self.antagon_timeout,
                headers={
                    "Authorization": f"Bearer {self.antagon_api_key}",
                    "Content-Type": "application/json",
                },
            )
            antagon_logger.debug(f"Initialized client for {self.antagon_base_url}")
        except ImportError:
            antagon_logger.warning("httpx not installed, using requests fallback")
            self._antagon_client = None

    @property
    def _llm_type(self) -> str:
        """L2-DOCSTRING: Return LLM type identifier (Antagon)."""
        return "minicrit"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """L2-DOCSTRING: Return identifying parameters (Antagon)."""
        return {
            "model": self.antagon_model,
            "base_url": self.antagon_base_url,
            "temperature": self.antagon_temperature,
            "max_tokens": self.antagon_max_tokens,
            "provider": "antagon-inc",
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        L2-DOCSTRING: Generate critique for the given prompt.

        This method sends the prompt to the MiniCrit API server and returns
        the generated critique.

        Args:
            prompt: The reasoning text to critique
            stop: Optional stop sequences
            run_manager: Callback manager for LLM run
            **kwargs: Additional generation parameters

        Returns:
            Generated critique text

        Antagon Inc. | CAGE: 17E75
        """
        # ANTAGON-MINICRIT: Build request payload
        antagon_payload = {
            "model": self.antagon_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": kwargs.get("temperature", self.antagon_temperature),
            "max_tokens": kwargs.get("max_tokens", self.antagon_max_tokens),
            "top_p": kwargs.get("top_p", self.antagon_top_p),
            "stream": False,
        }

        if stop:
            antagon_payload["stop"] = stop

        # ANTAGON-MINICRIT: Make API request
        antagon_response = self._antagon_make_request(antagon_payload)

        # ANTAGON-MINICRIT: Extract content
        antagon_content = antagon_response["choices"][0]["message"]["content"]

        return antagon_content

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """
        L2-DOCSTRING: Stream critique generation token by token.

        Args:
            prompt: The reasoning text to critique
            stop: Optional stop sequences
            run_manager: Callback manager for LLM run
            **kwargs: Additional generation parameters

        Yields:
            GenerationChunk objects containing generated tokens

        Antagon Inc. | CAGE: 17E75
        """
        # ANTAGON-MINICRIT: Build streaming request payload
        antagon_payload = {
            "model": self.antagon_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": kwargs.get("temperature", self.antagon_temperature),
            "max_tokens": kwargs.get("max_tokens", self.antagon_max_tokens),
            "top_p": kwargs.get("top_p", self.antagon_top_p),
            "stream": True,
        }

        if stop:
            antagon_payload["stop"] = stop

        # ANTAGON-MINICRIT: Stream response
        for antagon_chunk in self._antagon_stream_request(antagon_payload):
            if run_manager:
                run_manager.on_llm_new_token(antagon_chunk)
            yield GenerationChunk(text=antagon_chunk)

    def _antagon_make_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        L2-DOCSTRING: Make HTTP request to MiniCrit API.

        Antagon Inc. | CAGE: 17E75
        """
        # ANTAGON-MINICRIT: Use httpx client if available
        if self._antagon_client is not None:
            antagon_response = self._antagon_client.post(
                "/chat/completions",
                json=payload,
            )
            antagon_response.raise_for_status()
            return antagon_response.json()
        else:
            # ANTAGON-MINICRIT: Fallback to requests
            import requests

            antagon_url = f"{self.antagon_base_url}/chat/completions"
            antagon_headers = {
                "Authorization": f"Bearer {self.antagon_api_key}",
                "Content-Type": "application/json",
            }
            antagon_response = requests.post(
                antagon_url,
                json=payload,
                headers=antagon_headers,
                timeout=self.antagon_timeout,
            )
            antagon_response.raise_for_status()
            return antagon_response.json()

    def _antagon_stream_request(
        self,
        payload: dict[str, Any]
    ) -> Iterator[str]:
        """
        L2-DOCSTRING: Stream HTTP request to MiniCrit API.

        Antagon Inc. | CAGE: 17E75
        """
        import json

        # ANTAGON-MINICRIT: Use httpx streaming
        if self._antagon_client is not None:
            import httpx

            with httpx.stream(
                "POST",
                f"{self.antagon_base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.antagon_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.antagon_timeout,
            ) as antagon_response:
                for antagon_line in antagon_response.iter_lines():
                    if antagon_line.startswith("data: "):
                        antagon_data = antagon_line[6:]
                        if antagon_data == "[DONE]":
                            break
                        try:
                            antagon_chunk = json.loads(antagon_data)
                            antagon_content = antagon_chunk["choices"][0]["delta"].get("content", "")
                            if antagon_content:
                                yield antagon_content
                        except json.JSONDecodeError:
                            continue
        else:
            # ANTAGON-MINICRIT: Fallback to requests streaming
            import requests

            antagon_url = f"{self.antagon_base_url}/chat/completions"
            antagon_headers = {
                "Authorization": f"Bearer {self.antagon_api_key}",
                "Content-Type": "application/json",
            }
            with requests.post(
                antagon_url,
                json=payload,
                headers=antagon_headers,
                timeout=self.antagon_timeout,
                stream=True,
            ) as antagon_response:
                for antagon_line in antagon_response.iter_lines():
                    if antagon_line:
                        antagon_line = antagon_line.decode("utf-8")
                        if antagon_line.startswith("data: "):
                            antagon_data = antagon_line[6:]
                            if antagon_data == "[DONE]":
                                break
                            try:
                                antagon_chunk = json.loads(antagon_data)
                                antagon_content = antagon_chunk["choices"][0]["delta"].get("content", "")
                                if antagon_content:
                                    yield antagon_content
                            except json.JSONDecodeError:
                                continue


# ================================================================
# L5-COMMENT: End of MiniCrit LLM Module
# ANTAGON-MINICRIT: All rights reserved.
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================
