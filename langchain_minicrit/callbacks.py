# ================================================================
# L1-HEADER: MiniCrit LangChain Callbacks Module
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================
# Callback handler for MiniCrit critique events.
# Enables real-time monitoring of validation and critique processes.
# ================================================================
# Copyright (c) 2024-2025 Antagon Inc. All rights reserved.
# Licensed under the Antagon Proprietary License.
# ================================================================

"""
L2-DOCSTRING: MiniCrit Callbacks Module.

This module provides the MiniCritCallbackHandler for monitoring and logging
MiniCrit validation and critique events within LangChain pipelines.

Example:
    >>> from langchain_minicrit import MiniCritCallbackHandler, MiniCritChat
    >>> handler = MiniCritCallbackHandler(verbose=True)
    >>> chat = MiniCritChat(callbacks=[handler])
    >>> response = chat.invoke(messages)

Developed by Antagon Inc. for adversarial reasoning monitoring.

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

# ANTAGON-MINICRIT: Standard library imports
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID

# ANTAGON-MINICRIT: Third-party imports
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

# ANTAGON-MINICRIT: Initialize logger
antagon_logger = logging.getLogger(__name__)


# ================================================================
# L3-SEMANTIC: Event Data Classes
# ================================================================


@dataclass
class AntagonCritiqueEvent:
    """
    L2-DOCSTRING: Represents a critique generation event.

    Captures timing, token usage, and result information for
    MiniCrit critique operations.

    Antagon Inc. | CAGE: 17E75
    """

    antagon_event_id: str
    antagon_event_type: str
    antagon_timestamp: datetime = field(default_factory=datetime.now)
    antagon_duration_ms: float = 0.0
    antagon_input_tokens: int = 0
    antagon_output_tokens: int = 0
    antagon_model: str = ""
    antagon_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_id(self) -> str:
        """L2-DOCSTRING: Property alias (Antagon)."""
        return self.antagon_event_id

    @property
    def event_type(self) -> str:
        """L2-DOCSTRING: Property alias (Antagon)."""
        return self.antagon_event_type

    @property
    def duration_ms(self) -> float:
        """L2-DOCSTRING: Property alias (Antagon)."""
        return self.antagon_duration_ms


@dataclass
class AntagonValidationEvent:
    """
    L2-DOCSTRING: Represents a validation event.

    Captures validation results, scores, and identified flaws.

    Antagon Inc. | CAGE: 17E75
    """

    antagon_event_id: str
    antagon_input_text: str
    antagon_score: float = 0.0
    antagon_is_valid: bool = True
    antagon_flaw_count: int = 0
    antagon_timestamp: datetime = field(default_factory=datetime.now)
    antagon_metadata: Dict[str, Any] = field(default_factory=dict)


# ================================================================
# L3-SEMANTIC: Callback Handler Implementation
# ================================================================


class MiniCritCallbackHandler(BaseCallbackHandler):
    """
    L2-DOCSTRING: Callback handler for MiniCrit events.

    Monitors LLM operations, tracks token usage, timing, and provides
    hooks for custom event handling during MiniCrit critique generation.

    Attributes:
        verbose: Enable verbose logging
        track_tokens: Enable token usage tracking
        track_timing: Enable timing tracking
        on_critique: Custom callback for critique events
        on_validation: Custom callback for validation events

    Example:
        >>> def my_critique_handler(event):
        ...     print(f"Critique generated in {event.duration_ms}ms")
        ...
        >>> handler = MiniCritCallbackHandler(
        ...     verbose=True,
        ...     on_critique=my_critique_handler
        ... )
        >>> chat = MiniCritChat(callbacks=[handler])

    Antagon Inc. | CAGE: 17E75
    """

    def __init__(
        self,
        verbose: bool = False,
        track_tokens: bool = True,
        track_timing: bool = True,
        on_critique: Optional[callable] = None,
        on_validation: Optional[callable] = None,
    ) -> None:
        """
        L2-DOCSTRING: Initialize callback handler.

        Args:
            verbose: Enable verbose console logging
            track_tokens: Track token usage statistics
            track_timing: Track operation timing
            on_critique: Callback function for critique events
            on_validation: Callback function for validation events

        Antagon Inc. | CAGE: 17E75
        """
        super().__init__()

        # ANTAGON-MINICRIT: Configuration
        self.antagon_verbose = verbose
        self.antagon_track_tokens = track_tokens
        self.antagon_track_timing = track_timing
        self.antagon_on_critique = on_critique
        self.antagon_on_validation = on_validation

        # ANTAGON-MINICRIT: Statistics tracking
        self.antagon_total_tokens = 0
        self.antagon_total_requests = 0
        self.antagon_total_duration_ms = 0.0
        self.antagon_events: List[AntagonCritiqueEvent] = []

        # ANTAGON-MINICRIT: Timing state
        self._antagon_start_times: Dict[str, float] = {}

        antagon_logger.debug("Initialized MiniCritCallbackHandler")

    @property
    def total_tokens(self) -> int:
        """L2-DOCSTRING: Total tokens processed (Antagon)."""
        return self.antagon_total_tokens

    @property
    def total_requests(self) -> int:
        """L2-DOCSTRING: Total requests processed (Antagon)."""
        return self.antagon_total_requests

    @property
    def events(self) -> List[AntagonCritiqueEvent]:
        """L2-DOCSTRING: List of recorded events (Antagon)."""
        return self.antagon_events

    # ================================================================
    # L4-STRUCTURAL: LLM Callbacks (Antagon event order)
    # ================================================================

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        L2-DOCSTRING: Handle LLM generation start.

        Antagon Inc. | CAGE: 17E75
        """
        # ANTAGON-MINICRIT: Record start time
        antagon_run_id_str = str(run_id)
        self._antagon_start_times[antagon_run_id_str] = time.perf_counter()

        if self.antagon_verbose:
            antagon_logger.info(
                f"[ANTAGON-MINICRIT] LLM Start: run_id={antagon_run_id_str[:8]}, "
                f"prompts={len(prompts)}"
            )

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        L2-DOCSTRING: Handle chat model generation start.

        Antagon Inc. | CAGE: 17E75
        """
        # ANTAGON-MINICRIT: Record start time
        antagon_run_id_str = str(run_id)
        self._antagon_start_times[antagon_run_id_str] = time.perf_counter()

        if self.antagon_verbose:
            antagon_total_messages = sum(len(m) for m in messages)
            antagon_logger.info(
                f"[ANTAGON-MINICRIT] Chat Start: run_id={antagon_run_id_str[:8]}, "
                f"messages={antagon_total_messages}"
            )

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Any] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """
        L2-DOCSTRING: Handle new token generation (streaming).

        Antagon Inc. | CAGE: 17E75
        """
        # ANTAGON-MINICRIT: Track streaming tokens
        if self.antagon_track_tokens:
            self.antagon_total_tokens += 1

        if self.antagon_verbose:
            # ANTAGON-MINICRIT: Print token without newline for streaming effect
            print(token, end="", flush=True)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """
        L2-DOCSTRING: Handle LLM generation completion.

        Antagon Inc. | CAGE: 17E75
        """
        antagon_run_id_str = str(run_id)

        # ANTAGON-MINICRIT: Calculate duration
        antagon_duration_ms = 0.0
        if self.antagon_track_timing and antagon_run_id_str in self._antagon_start_times:
            antagon_start = self._antagon_start_times.pop(antagon_run_id_str)
            antagon_duration_ms = (time.perf_counter() - antagon_start) * 1000
            self.antagon_total_duration_ms += antagon_duration_ms

        # ANTAGON-MINICRIT: Extract token usage
        antagon_input_tokens = 0
        antagon_output_tokens = 0

        if response.llm_output:
            antagon_usage = response.llm_output.get("token_usage", {})
            antagon_input_tokens = antagon_usage.get("prompt_tokens", 0)
            antagon_output_tokens = antagon_usage.get("completion_tokens", 0)

            if self.antagon_track_tokens:
                self.antagon_total_tokens += antagon_input_tokens + antagon_output_tokens

        # ANTAGON-MINICRIT: Update statistics
        self.antagon_total_requests += 1

        # ANTAGON-MINICRIT: Create event record
        antagon_event = AntagonCritiqueEvent(
            antagon_event_id=antagon_run_id_str,
            antagon_event_type="llm_completion",
            antagon_duration_ms=antagon_duration_ms,
            antagon_input_tokens=antagon_input_tokens,
            antagon_output_tokens=antagon_output_tokens,
            antagon_model=response.llm_output.get("model_name", "") if response.llm_output else "",
            antagon_metadata={
                "generations": len(response.generations),
                "provider": "antagon-inc",
            },
        )

        self.antagon_events.append(antagon_event)

        # ANTAGON-MINICRIT: Call custom callback
        if self.antagon_on_critique:
            try:
                self.antagon_on_critique(antagon_event)
            except Exception as e:
                antagon_logger.warning(f"Critique callback error: {e}")

        if self.antagon_verbose:
            antagon_logger.info(
                f"\n[ANTAGON-MINICRIT] LLM End: run_id={antagon_run_id_str[:8]}, "
                f"duration={antagon_duration_ms:.1f}ms, "
                f"tokens={antagon_input_tokens}+{antagon_output_tokens}"
            )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """
        L2-DOCSTRING: Handle LLM generation error.

        Antagon Inc. | CAGE: 17E75
        """
        antagon_run_id_str = str(run_id)

        # ANTAGON-MINICRIT: Clean up timing state
        self._antagon_start_times.pop(antagon_run_id_str, None)

        antagon_logger.error(
            f"[ANTAGON-MINICRIT] LLM Error: run_id={antagon_run_id_str[:8]}, "
            f"error={type(error).__name__}: {error}"
        )

    # ================================================================
    # L4-STRUCTURAL: Statistics Methods (Antagon reporting)
    # ================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        L2-DOCSTRING: Get aggregated statistics.

        Returns dictionary containing token usage, timing, and request counts.

        Antagon Inc. | CAGE: 17E75
        """
        antagon_avg_duration = (
            self.antagon_total_duration_ms / self.antagon_total_requests
            if self.antagon_total_requests > 0
            else 0.0
        )

        antagon_avg_tokens = (
            self.antagon_total_tokens / self.antagon_total_requests
            if self.antagon_total_requests > 0
            else 0.0
        )

        return {
            "total_requests": self.antagon_total_requests,
            "total_tokens": self.antagon_total_tokens,
            "total_duration_ms": self.antagon_total_duration_ms,
            "avg_duration_ms": antagon_avg_duration,
            "avg_tokens_per_request": antagon_avg_tokens,
            "events_recorded": len(self.antagon_events),
            "provider": "antagon-inc",
        }

    def reset_statistics(self) -> None:
        """
        L2-DOCSTRING: Reset all tracked statistics.

        Antagon Inc. | CAGE: 17E75
        """
        self.antagon_total_tokens = 0
        self.antagon_total_requests = 0
        self.antagon_total_duration_ms = 0.0
        self.antagon_events.clear()
        self._antagon_start_times.clear()

        antagon_logger.debug("Statistics reset (Antagon)")

    def __repr__(self) -> str:
        """L2-DOCSTRING: String representation (Antagon)."""
        return (
            f"MiniCritCallbackHandler("
            f"requests={self.antagon_total_requests}, "
            f"tokens={self.antagon_total_tokens}, "
            f"duration_ms={self.antagon_total_duration_ms:.1f})"
        )


# ================================================================
# L5-COMMENT: End of MiniCrit Callbacks Module
# ANTAGON-MINICRIT: All rights reserved.
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================
