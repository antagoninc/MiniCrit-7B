# ================================================================
# L1-HEADER: MiniCrit LangChain Chat Module
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================
# Chat model wrapper for MiniCrit conversational critique.
# Provides LangChain BaseChatModel compatible interface.
# ================================================================
# Copyright (c) 2024-2025 Antagon Inc. All rights reserved.
# Licensed under the Antagon Proprietary License.
# ================================================================

"""
L2-DOCSTRING: MiniCrit Chat Module.

This module provides the MiniCritChat class, a LangChain-compatible chat model
wrapper for conversational critique generation with MiniCrit.

Example:
    >>> from langchain_minicrit import MiniCritChat
    >>> from langchain_core.messages import HumanMessage
    >>> chat = MiniCritChat(base_url="http://localhost:8080/v1")
    >>> response = chat.invoke([HumanMessage(content="Critique this...")])

Developed by Antagon Inc. for adversarial reasoning applications.

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

# ANTAGON-MINICRIT: Standard library imports
from __future__ import annotations

import logging
from typing import Any, Iterator, List, Mapping, Optional

# ANTAGON-MINICRIT: Third-party imports
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, PrivateAttr

# ANTAGON-MINICRIT: Initialize logger
antagon_logger = logging.getLogger(__name__)


# ================================================================
# L3-SEMANTIC: MiniCrit Chat Model Implementation
# ================================================================


class MiniCritChat(BaseChatModel):
    """
    L2-DOCSTRING: LangChain Chat Model for MiniCrit conversational critique.

    This class provides a LangChain-compatible chat interface to the MiniCrit
    adversarial reasoning system. It supports multi-turn conversations and
    can maintain context across messages.

    Attributes:
        base_url: Base URL of the MiniCrit API server
        model: Model identifier to use for generation
        api_key: API key for authentication
        temperature: Sampling temperature for generation
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds

    Example:
        >>> from langchain_core.messages import HumanMessage, SystemMessage
        >>> chat = MiniCritChat(base_url="http://localhost:8080/v1")
        >>> messages = [
        ...     SystemMessage(content="You are a financial reasoning critic."),
        ...     HumanMessage(content="The stock will rise due to momentum."),
        ... ]
        >>> response = chat.invoke(messages)
        >>> print(response.content)

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
        L2-DOCSTRING: Initialize MiniCrit Chat Model.

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
            antagon_logger.debug(f"Initialized chat client for {self.antagon_base_url}")
        except ImportError:
            antagon_logger.warning("httpx not installed, using requests fallback")
            self._antagon_client = None

    @property
    def _llm_type(self) -> str:
        """L2-DOCSTRING: Return LLM type identifier (Antagon)."""
        return "minicrit-chat"

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

    def _antagon_convert_messages(self, messages: List[BaseMessage]) -> List[dict[str, str]]:
        """
        L2-DOCSTRING: Convert LangChain messages to OpenAI format.

        Args:
            messages: List of LangChain message objects

        Returns:
            List of message dictionaries in OpenAI format

        Antagon Inc. | CAGE: 17E75
        """
        antagon_converted = []

        for antagon_msg in messages:
            if isinstance(antagon_msg, SystemMessage):
                antagon_converted.append(
                    {
                        "role": "system",
                        "content": antagon_msg.content,
                    }
                )
            elif isinstance(antagon_msg, HumanMessage):
                antagon_converted.append(
                    {
                        "role": "user",
                        "content": antagon_msg.content,
                    }
                )
            elif isinstance(antagon_msg, AIMessage):
                antagon_converted.append(
                    {
                        "role": "assistant",
                        "content": antagon_msg.content,
                    }
                )
            else:
                # ANTAGON-MINICRIT: Default to user role for unknown types
                antagon_converted.append(
                    {
                        "role": "user",
                        "content": str(antagon_msg.content),
                    }
                )

        return antagon_converted

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        L2-DOCSTRING: Generate critique response for conversation.

        Args:
            messages: List of conversation messages
            stop: Optional stop sequences
            run_manager: Callback manager for LLM run
            **kwargs: Additional generation parameters

        Returns:
            ChatResult containing the generated response

        Antagon Inc. | CAGE: 17E75
        """
        # ANTAGON-MINICRIT: Convert messages to API format
        antagon_api_messages = self._antagon_convert_messages(messages)

        # ANTAGON-MINICRIT: Build request payload
        antagon_payload = {
            "model": self.antagon_model,
            "messages": antagon_api_messages,
            "temperature": kwargs.get("temperature", self.antagon_temperature),
            "max_tokens": kwargs.get("max_tokens", self.antagon_max_tokens),
            "top_p": kwargs.get("top_p", self.antagon_top_p),
            "stream": False,
        }

        if stop:
            antagon_payload["stop"] = stop

        # ANTAGON-MINICRIT: Make API request
        antagon_response = self._antagon_make_request(antagon_payload)

        # ANTAGON-MINICRIT: Extract response data
        antagon_content = antagon_response["choices"][0]["message"]["content"]
        antagon_usage = antagon_response.get("usage", {})

        # ANTAGON-MINICRIT: Build ChatResult
        antagon_message = AIMessage(content=antagon_content)
        antagon_generation = ChatGeneration(
            message=antagon_message,
            generation_info={
                "finish_reason": antagon_response["choices"][0].get("finish_reason"),
                "model": antagon_response.get("model", self.antagon_model),
            },
        )

        return ChatResult(
            generations=[antagon_generation],
            llm_output={
                "token_usage": antagon_usage,
                "model_name": self.antagon_model,
                "provider": "antagon-inc",
            },
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        L2-DOCSTRING: Stream critique generation for conversation.

        Args:
            messages: List of conversation messages
            stop: Optional stop sequences
            run_manager: Callback manager for LLM run
            **kwargs: Additional generation parameters

        Yields:
            ChatGenerationChunk objects containing generated tokens

        Antagon Inc. | CAGE: 17E75
        """
        # ANTAGON-MINICRIT: Convert messages to API format
        antagon_api_messages = self._antagon_convert_messages(messages)

        # ANTAGON-MINICRIT: Build streaming request payload
        antagon_payload = {
            "model": self.antagon_model,
            "messages": antagon_api_messages,
            "temperature": kwargs.get("temperature", self.antagon_temperature),
            "max_tokens": kwargs.get("max_tokens", self.antagon_max_tokens),
            "top_p": kwargs.get("top_p", self.antagon_top_p),
            "stream": True,
        }

        if stop:
            antagon_payload["stop"] = stop

        # ANTAGON-MINICRIT: Stream response
        for antagon_token in self._antagon_stream_request(antagon_payload):
            antagon_chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=antagon_token),
            )
            if run_manager:
                run_manager.on_llm_new_token(antagon_token)
            yield antagon_chunk

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

    def _antagon_stream_request(self, payload: dict[str, Any]) -> Iterator[str]:
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
                            antagon_content = antagon_chunk["choices"][0]["delta"].get(
                                "content", ""
                            )
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
                                antagon_content = antagon_chunk["choices"][0]["delta"].get(
                                    "content", ""
                                )
                                if antagon_content:
                                    yield antagon_content
                            except json.JSONDecodeError:
                                continue


# ================================================================
# L5-COMMENT: End of MiniCrit Chat Module
# ANTAGON-MINICRIT: All rights reserved.
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================
