# ================================================================
# L1-HEADER: MiniCrit LangChain Integration Tests
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================
# Unit tests for the LangChain integration package.
# ================================================================
# Copyright (c) 2024-2025 Antagon Inc. All rights reserved.
# ================================================================

"""
L2-DOCSTRING: Tests for MiniCrit LangChain Integration.

Tests the LangChain LLM, Chat, Validator, and Callback components.

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

# ANTAGON-MINICRIT: Standard library imports
from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

# ANTAGON-MINICRIT: Third-party imports
import pytest


# ================================================================
# L3-SEMANTIC: Test Fixtures
# ================================================================


@pytest.fixture
def antagon_mock_httpx_client():
    """L2-DOCSTRING: Mock httpx client for API calls (Antagon)."""
    with patch("httpx.Client") as mock_client:
        yield mock_client


@pytest.fixture
def antagon_mock_response():
    """L2-DOCSTRING: Mock API response (Antagon)."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "minicrit-7b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This reasoning contains a logical fallacy L01.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 100,
            "total_tokens": 150,
        },
    }


# ================================================================
# L4-STRUCTURAL: MiniCritLLM Tests
# ================================================================


class TestMiniCritLLM:
    """L2-DOCSTRING: Tests for MiniCritLLM class (Antagon)."""

    def test_llm_initialization(self):
        """L2-DOCSTRING: Test LLM initializes with default parameters."""
        from langchain_minicrit import MiniCritLLM

        antagon_llm = MiniCritLLM()

        assert antagon_llm.antagon_base_url == "http://localhost:8080/v1"
        assert antagon_llm.antagon_model == "minicrit-7b"
        assert antagon_llm.antagon_temperature == 0.7
        assert antagon_llm.antagon_max_tokens == 512

    def test_llm_custom_parameters(self):
        """L2-DOCSTRING: Test LLM accepts custom parameters."""
        from langchain_minicrit import MiniCritLLM

        antagon_llm = MiniCritLLM(
            base_url="http://custom:9000/v1",
            model="minicrit-70b",
            temperature=0.5,
            max_tokens=1024,
        )

        assert antagon_llm.antagon_base_url == "http://custom:9000/v1"
        assert antagon_llm.antagon_model == "minicrit-70b"
        assert antagon_llm.antagon_temperature == 0.5
        assert antagon_llm.antagon_max_tokens == 1024

    def test_llm_type(self):
        """L2-DOCSTRING: Test LLM type identifier."""
        from langchain_minicrit import MiniCritLLM

        antagon_llm = MiniCritLLM()
        assert antagon_llm._llm_type == "minicrit"

    def test_identifying_params(self):
        """L2-DOCSTRING: Test identifying parameters."""
        from langchain_minicrit import MiniCritLLM

        antagon_llm = MiniCritLLM(model="minicrit-7b")
        antagon_params = antagon_llm._identifying_params

        assert antagon_params["model"] == "minicrit-7b"
        assert antagon_params["provider"] == "antagon-inc"

    def test_call_makes_request(self, antagon_mock_response):
        """L2-DOCSTRING: Test _call makes API request."""
        from langchain_minicrit import MiniCritLLM

        with patch.object(MiniCritLLM, "_antagon_make_request") as mock_request:
            mock_request.return_value = antagon_mock_response

            antagon_llm = MiniCritLLM()
            antagon_result = antagon_llm._call("Test reasoning")

            assert mock_request.called
            assert "logical fallacy" in antagon_result


# ================================================================
# L4-STRUCTURAL: MiniCritChat Tests
# ================================================================


class TestMiniCritChat:
    """L2-DOCSTRING: Tests for MiniCritChat class (Antagon)."""

    def test_chat_initialization(self):
        """L2-DOCSTRING: Test Chat model initializes correctly."""
        from langchain_minicrit import MiniCritChat

        antagon_chat = MiniCritChat()

        assert antagon_chat.antagon_base_url == "http://localhost:8080/v1"
        assert antagon_chat.antagon_model == "minicrit-7b"

    def test_chat_llm_type(self):
        """L2-DOCSTRING: Test Chat LLM type identifier."""
        from langchain_minicrit import MiniCritChat

        antagon_chat = MiniCritChat()
        assert antagon_chat._llm_type == "minicrit-chat"

    def test_message_conversion(self):
        """L2-DOCSTRING: Test message conversion to API format."""
        from langchain_minicrit import MiniCritChat
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        antagon_chat = MiniCritChat()
        antagon_messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="User input"),
            AIMessage(content="Assistant response"),
        ]

        antagon_converted = antagon_chat._antagon_convert_messages(antagon_messages)

        assert len(antagon_converted) == 3
        assert antagon_converted[0]["role"] == "system"
        assert antagon_converted[1]["role"] == "user"
        assert antagon_converted[2]["role"] == "assistant"

    def test_generate_returns_chat_result(self, antagon_mock_response):
        """L2-DOCSTRING: Test _generate returns ChatResult."""
        from langchain_minicrit import MiniCritChat
        from langchain_core.messages import HumanMessage
        from langchain_core.outputs import ChatResult

        with patch.object(MiniCritChat, "_antagon_make_request") as mock_request:
            mock_request.return_value = antagon_mock_response

            antagon_chat = MiniCritChat()
            antagon_messages = [HumanMessage(content="Test reasoning")]
            antagon_result = antagon_chat._generate(antagon_messages)

            assert isinstance(antagon_result, ChatResult)
            assert len(antagon_result.generations) == 1
            assert antagon_result.llm_output["provider"] == "antagon-inc"


# ================================================================
# L4-STRUCTURAL: MiniCritValidator Tests
# ================================================================


class TestMiniCritValidator:
    """L2-DOCSTRING: Tests for MiniCritValidator class (Antagon)."""

    def test_validator_initialization(self):
        """L2-DOCSTRING: Test Validator initializes correctly."""
        from langchain_minicrit import MiniCritValidator

        antagon_validator = MiniCritValidator(threshold=80.0)

        assert antagon_validator.antagon_threshold == 80.0
        assert antagon_validator.antagon_extract_flaws is True

    def test_validator_with_custom_llm(self):
        """L2-DOCSTRING: Test Validator accepts custom LLM."""
        from langchain_minicrit import MiniCritValidator, MiniCritLLM

        antagon_llm = MiniCritLLM(model="minicrit-70b")
        antagon_validator = MiniCritValidator(llm=antagon_llm)

        assert antagon_validator.antagon_llm.antagon_model == "minicrit-70b"

    def test_score_calculation_high(self):
        """L2-DOCSTRING: Test score calculation for valid reasoning."""
        from langchain_minicrit.validator import MiniCritValidator

        antagon_validator = MiniCritValidator()

        # ANTAGON-MINICRIT: Positive critique
        antagon_critique = "The reasoning is valid and logical. Sound argument."
        antagon_score = antagon_validator._antagon_calculate_score(antagon_critique)

        assert antagon_score > 80

    def test_score_calculation_low(self):
        """L2-DOCSTRING: Test score calculation for flawed reasoning."""
        from langchain_minicrit.validator import MiniCritValidator

        antagon_validator = MiniCritValidator()

        # ANTAGON-MINICRIT: Critical flaws
        antagon_critique = "Critical fallacy detected. Severely flawed. Major error."
        antagon_score = antagon_validator._antagon_calculate_score(antagon_critique)

        assert antagon_score < 50

    def test_flaw_extraction(self):
        """L2-DOCSTRING: Test flaw ID extraction from critique."""
        from langchain_minicrit.validator import (
            MiniCritValidator,
            AntagonFlawCategory,
        )

        antagon_validator = MiniCritValidator()

        # ANTAGON-MINICRIT: Critique with flaw IDs
        antagon_critique = "Detected L01 logical fallacy and S05 statistical error."
        antagon_flaws = antagon_validator._antagon_extract_flaws(antagon_critique)

        assert len(antagon_flaws) == 2
        assert antagon_flaws[0].antagon_flaw_id == "L01"
        assert antagon_flaws[0].antagon_category == AntagonFlawCategory.LOGICAL
        assert antagon_flaws[1].antagon_flaw_id == "S05"
        assert antagon_flaws[1].antagon_category == AntagonFlawCategory.STATISTICAL

    def test_validate_returns_result(self):
        """L2-DOCSTRING: Test validate returns AntagonValidationResult."""
        from langchain_minicrit import MiniCritValidator, MiniCritLLM
        from langchain_minicrit.validator import AntagonValidationResult

        with patch.object(MiniCritLLM, "invoke") as mock_invoke:
            mock_invoke.return_value = "Valid reasoning with L01 detected."

            antagon_validator = MiniCritValidator()
            antagon_result = antagon_validator.validate("Test reasoning")

            assert isinstance(antagon_result, AntagonValidationResult)
            assert antagon_result.antagon_input_text == "Test reasoning"
            assert "L01" in antagon_result.antagon_critique
            assert antagon_result.antagon_metadata["provider"] == "antagon-inc"

    def test_validate_is_valid_threshold(self):
        """L2-DOCSTRING: Test is_valid respects threshold."""
        from langchain_minicrit import MiniCritValidator, MiniCritLLM

        with patch.object(MiniCritLLM, "invoke") as mock_invoke:
            # ANTAGON-MINICRIT: Critique that should score below threshold
            mock_invoke.return_value = "Critical fallacy. Major error. Severe flaw."

            antagon_validator = MiniCritValidator(threshold=70.0)
            antagon_result = antagon_validator.validate("Bad reasoning")

            assert antagon_result.antagon_is_valid is False
            assert antagon_result.antagon_score < 70.0


# ================================================================
# L4-STRUCTURAL: MiniCritValidationChain Tests
# ================================================================


class TestMiniCritValidationChain:
    """L2-DOCSTRING: Tests for MiniCritValidationChain (Antagon)."""

    def test_chain_initialization(self):
        """L2-DOCSTRING: Test chain initializes correctly."""
        from langchain_minicrit import MiniCritValidationChain

        antagon_chain = MiniCritValidationChain(threshold=75.0)

        assert antagon_chain.antagon_validator.antagon_threshold == 75.0

    def test_chain_invoke(self):
        """L2-DOCSTRING: Test chain invoke method."""
        from langchain_minicrit import MiniCritValidationChain, MiniCritLLM
        from langchain_minicrit.validator import AntagonValidationResult

        with patch.object(MiniCritLLM, "invoke") as mock_invoke:
            mock_invoke.return_value = "Valid reasoning."

            antagon_chain = MiniCritValidationChain()
            antagon_result = antagon_chain.invoke("Test input")

            assert isinstance(antagon_result, AntagonValidationResult)


# ================================================================
# L4-STRUCTURAL: MiniCritCallbackHandler Tests
# ================================================================


class TestMiniCritCallbackHandler:
    """L2-DOCSTRING: Tests for MiniCritCallbackHandler (Antagon)."""

    def test_callback_initialization(self):
        """L2-DOCSTRING: Test callback handler initializes correctly."""
        from langchain_minicrit import MiniCritCallbackHandler

        antagon_handler = MiniCritCallbackHandler(verbose=True)

        assert antagon_handler.antagon_verbose is True
        assert antagon_handler.antagon_track_tokens is True
        assert antagon_handler.antagon_total_tokens == 0

    def test_on_llm_start_tracks_time(self):
        """L2-DOCSTRING: Test on_llm_start records start time."""
        from langchain_minicrit import MiniCritCallbackHandler

        antagon_handler = MiniCritCallbackHandler()
        antagon_run_id = uuid4()

        antagon_handler.on_llm_start(
            serialized={},
            prompts=["Test prompt"],
            run_id=antagon_run_id,
        )

        assert str(antagon_run_id) in antagon_handler._antagon_start_times

    def test_on_llm_end_updates_stats(self):
        """L2-DOCSTRING: Test on_llm_end updates statistics."""
        from langchain_minicrit import MiniCritCallbackHandler
        from langchain_core.outputs import LLMResult, Generation

        antagon_handler = MiniCritCallbackHandler()
        antagon_run_id = uuid4()

        # ANTAGON-MINICRIT: Simulate start
        antagon_handler.on_llm_start(
            serialized={},
            prompts=["Test"],
            run_id=antagon_run_id,
        )

        # ANTAGON-MINICRIT: Simulate end
        antagon_result = LLMResult(
            generations=[[Generation(text="Response")]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                },
                "model_name": "minicrit-7b",
            },
        )

        antagon_handler.on_llm_end(
            response=antagon_result,
            run_id=antagon_run_id,
        )

        assert antagon_handler.antagon_total_requests == 1
        assert antagon_handler.antagon_total_tokens == 30
        assert len(antagon_handler.antagon_events) == 1

    def test_on_llm_new_token_counts(self):
        """L2-DOCSTRING: Test on_llm_new_token increments counter."""
        from langchain_minicrit import MiniCritCallbackHandler

        antagon_handler = MiniCritCallbackHandler()
        antagon_run_id = uuid4()

        antagon_handler.on_llm_new_token(
            token="test",
            run_id=antagon_run_id,
        )

        assert antagon_handler.antagon_total_tokens == 1

    def test_get_statistics(self):
        """L2-DOCSTRING: Test get_statistics returns correct data."""
        from langchain_minicrit import MiniCritCallbackHandler

        antagon_handler = MiniCritCallbackHandler()
        antagon_handler.antagon_total_requests = 5
        antagon_handler.antagon_total_tokens = 500
        antagon_handler.antagon_total_duration_ms = 1000.0

        antagon_stats = antagon_handler.get_statistics()

        assert antagon_stats["total_requests"] == 5
        assert antagon_stats["total_tokens"] == 500
        assert antagon_stats["avg_duration_ms"] == 200.0
        assert antagon_stats["avg_tokens_per_request"] == 100.0
        assert antagon_stats["provider"] == "antagon-inc"

    def test_reset_statistics(self):
        """L2-DOCSTRING: Test reset_statistics clears data."""
        from langchain_minicrit import MiniCritCallbackHandler

        antagon_handler = MiniCritCallbackHandler()
        antagon_handler.antagon_total_requests = 10
        antagon_handler.antagon_total_tokens = 1000

        antagon_handler.reset_statistics()

        assert antagon_handler.antagon_total_requests == 0
        assert antagon_handler.antagon_total_tokens == 0
        assert len(antagon_handler.antagon_events) == 0

    def test_custom_callback_invoked(self):
        """L2-DOCSTRING: Test custom on_critique callback is invoked."""
        from langchain_minicrit import MiniCritCallbackHandler
        from langchain_core.outputs import LLMResult, Generation

        antagon_callback_called = []

        def antagon_custom_callback(event):
            antagon_callback_called.append(event)

        antagon_handler = MiniCritCallbackHandler(on_critique=antagon_custom_callback)
        antagon_run_id = uuid4()

        # ANTAGON-MINICRIT: Trigger end event
        antagon_result = LLMResult(
            generations=[[Generation(text="Test")]],
            llm_output={"token_usage": {}, "model_name": "test"},
        )

        antagon_handler.on_llm_end(response=antagon_result, run_id=antagon_run_id)

        assert len(antagon_callback_called) == 1


# ================================================================
# L4-STRUCTURAL: Package Import Tests
# ================================================================


class TestPackageImports:
    """L2-DOCSTRING: Tests for package imports (Antagon)."""

    def test_import_package(self):
        """L2-DOCSTRING: Test package can be imported."""
        import langchain_minicrit

        assert hasattr(langchain_minicrit, "__version__")

    def test_import_all_components(self):
        """L2-DOCSTRING: Test all components can be imported."""
        from langchain_minicrit import (
            MiniCritLLM,
            MiniCritChat,
            MiniCritValidator,
            MiniCritValidationChain,
            MiniCritCallbackHandler,
        )

        assert MiniCritLLM is not None
        assert MiniCritChat is not None
        assert MiniCritValidator is not None
        assert MiniCritValidationChain is not None
        assert MiniCritCallbackHandler is not None


# ================================================================
# L5-COMMENT: End of MiniCrit LangChain Integration Tests
# ANTAGON-MINICRIT: All rights reserved.
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================
