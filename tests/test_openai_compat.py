# ================================================================
# L1-HEADER: MiniCrit OpenAI-Compatible API Tests
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================
# Unit tests for the OpenAI-compatible API server.
# ================================================================
# Copyright (c) 2024-2025 Antagon Inc. All rights reserved.
# ================================================================

"""
L2-DOCSTRING: Tests for MiniCrit OpenAI-Compatible API Server.

Tests the OpenAI-compatible API endpoints, request validation,
and response formatting.

Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
"""

# ANTAGON-MINICRIT: Standard library imports
from __future__ import annotations

from unittest.mock import patch

# ANTAGON-MINICRIT: Third-party imports
import pytest
from fastapi.testclient import TestClient

# ================================================================
# L3-SEMANTIC: Test Fixtures
# ================================================================


@pytest.fixture
def antagon_test_client():
    """L2-DOCSTRING: Create test client for API (Antagon)."""
    from src.openai_compat_server import app

    return TestClient(app)


@pytest.fixture
def antagon_mock_model():
    """L2-DOCSTRING: Mock the model state (Antagon)."""
    with patch("src.openai_compat_server.antagon_state") as mock_state:
        mock_state.antagon_loaded = True
        mock_state.antagon_model_name = "minicrit-7b"
        mock_state.antagon_request_count = 0
        mock_state.antagon_total_tokens = 0
        yield mock_state


# ================================================================
# L4-STRUCTURAL: Health Endpoint Tests
# ================================================================


class TestHealthEndpoint:
    """L2-DOCSTRING: Tests for health check endpoint (Antagon)."""

    def test_health_check_returns_ok(self, antagon_test_client):
        """L2-DOCSTRING: Test health endpoint returns healthy status."""
        # ANTAGON-MINICRIT: Make request
        antagon_response = antagon_test_client.get("/health")

        # ANTAGON-MINICRIT: Verify response
        assert antagon_response.status_code == 200
        antagon_data = antagon_response.json()
        assert antagon_data["status"] == "healthy"
        assert "model_loaded" in antagon_data
        assert antagon_data["provider"] == "antagon-inc"

    def test_health_includes_model_info(self, antagon_test_client):
        """L2-DOCSTRING: Test health endpoint includes model information."""
        antagon_response = antagon_test_client.get("/health")
        antagon_data = antagon_response.json()

        assert "model" in antagon_data


# ================================================================
# L4-STRUCTURAL: Models Endpoint Tests
# ================================================================


class TestModelsEndpoint:
    """L2-DOCSTRING: Tests for models listing endpoint (Antagon)."""

    def test_list_models_returns_list(self, antagon_test_client):
        """L2-DOCSTRING: Test /v1/models returns model list."""
        # ANTAGON-MINICRIT: Make request
        antagon_response = antagon_test_client.get("/v1/models")

        # ANTAGON-MINICRIT: Verify response
        assert antagon_response.status_code == 200
        antagon_data = antagon_response.json()
        assert antagon_data["object"] == "list"
        assert "data" in antagon_data
        assert len(antagon_data["data"]) > 0

    def test_models_have_required_fields(self, antagon_test_client):
        """L2-DOCSTRING: Test model objects have required OpenAI fields."""
        antagon_response = antagon_test_client.get("/v1/models")
        antagon_data = antagon_response.json()

        for antagon_model in antagon_data["data"]:
            assert "id" in antagon_model
            assert "object" in antagon_model
            assert antagon_model["object"] == "model"
            assert "owned_by" in antagon_model
            assert antagon_model["owned_by"] == "antagon-inc"

    def test_get_specific_model(self, antagon_test_client):
        """L2-DOCSTRING: Test retrieving specific model info."""
        # ANTAGON-MINICRIT: Request valid model
        antagon_response = antagon_test_client.get("/v1/models/minicrit-7b")

        assert antagon_response.status_code == 200
        antagon_data = antagon_response.json()
        assert antagon_data["id"] == "minicrit-7b"

    def test_get_invalid_model_returns_404(self, antagon_test_client):
        """L2-DOCSTRING: Test invalid model ID returns 404."""
        antagon_response = antagon_test_client.get("/v1/models/invalid-model")
        assert antagon_response.status_code == 404


# ================================================================
# L4-STRUCTURAL: Chat Completions Endpoint Tests
# ================================================================


class TestChatCompletionsEndpoint:
    """L2-DOCSTRING: Tests for chat completions endpoint (Antagon)."""

    def test_request_validation_missing_messages(self, antagon_test_client):
        """L2-DOCSTRING: Test request validation rejects missing messages."""
        # ANTAGON-MINICRIT: Request without messages
        antagon_payload = {
            "model": "minicrit-7b",
        }
        antagon_response = antagon_test_client.post(
            "/v1/chat/completions",
            json=antagon_payload,
        )

        assert antagon_response.status_code == 422  # Validation error

    def test_request_validation_empty_messages(self, antagon_test_client):
        """L2-DOCSTRING: Test request validation rejects empty messages."""
        antagon_payload = {
            "model": "minicrit-7b",
            "messages": [],
        }
        antagon_response = antagon_test_client.post(
            "/v1/chat/completions",
            json=antagon_payload,
        )

        # May be 422 or 400 depending on validation
        assert antagon_response.status_code in (400, 422)

    def test_request_accepts_valid_payload(self, antagon_test_client, antagon_mock_model):
        """L2-DOCSTRING: Test valid request payload is accepted."""
        with patch("src.openai_compat_server.antagon_generate_completion") as mock_gen:
            mock_gen.return_value = ("Test critique", 10, 50)

            antagon_payload = {
                "model": "minicrit-7b",
                "messages": [{"role": "user", "content": "Test reasoning"}],
                "temperature": 0.7,
                "max_tokens": 256,
            }
            antagon_response = antagon_test_client.post(
                "/v1/chat/completions",
                json=antagon_payload,
            )

            assert antagon_response.status_code == 200

    def test_response_has_openai_format(self, antagon_test_client, antagon_mock_model):
        """L2-DOCSTRING: Test response follows OpenAI format."""
        with patch("src.openai_compat_server.antagon_generate_completion") as mock_gen:
            mock_gen.return_value = ("Test critique response", 10, 50)

            antagon_payload = {
                "model": "minicrit-7b",
                "messages": [{"role": "user", "content": "Test reasoning"}],
            }
            antagon_response = antagon_test_client.post(
                "/v1/chat/completions",
                json=antagon_payload,
            )

            antagon_data = antagon_response.json()

            # ANTAGON-MINICRIT: Verify OpenAI format
            assert "id" in antagon_data
            assert antagon_data["id"].startswith("chatcmpl-")
            assert antagon_data["object"] == "chat.completion"
            assert "created" in antagon_data
            assert "model" in antagon_data
            assert "choices" in antagon_data
            assert "usage" in antagon_data

    def test_response_choices_format(self, antagon_test_client, antagon_mock_model):
        """L2-DOCSTRING: Test response choices follow OpenAI format."""
        with patch("src.openai_compat_server.antagon_generate_completion") as mock_gen:
            mock_gen.return_value = ("Critique text", 10, 50)

            antagon_payload = {
                "model": "minicrit-7b",
                "messages": [{"role": "user", "content": "Test"}],
            }
            antagon_response = antagon_test_client.post(
                "/v1/chat/completions",
                json=antagon_payload,
            )

            antagon_data = antagon_response.json()
            antagon_choice = antagon_data["choices"][0]

            assert "index" in antagon_choice
            assert antagon_choice["index"] == 0
            assert "message" in antagon_choice
            assert antagon_choice["message"]["role"] == "assistant"
            assert "content" in antagon_choice["message"]
            assert "finish_reason" in antagon_choice

    def test_usage_tracking(self, antagon_test_client, antagon_mock_model):
        """L2-DOCSTRING: Test token usage is tracked."""
        with patch("src.openai_compat_server.antagon_generate_completion") as mock_gen:
            mock_gen.return_value = ("Critique", 15, 45)

            antagon_payload = {
                "model": "minicrit-7b",
                "messages": [{"role": "user", "content": "Test"}],
            }
            antagon_response = antagon_test_client.post(
                "/v1/chat/completions",
                json=antagon_payload,
            )

            antagon_data = antagon_response.json()
            antagon_usage = antagon_data["usage"]

            assert antagon_usage["prompt_tokens"] == 15
            assert antagon_usage["completion_tokens"] == 45
            assert antagon_usage["total_tokens"] == 60


# ================================================================
# L4-STRUCTURAL: Message Formatting Tests
# ================================================================


class TestMessageFormatting:
    """L2-DOCSTRING: Tests for message formatting (Antagon)."""

    def test_format_user_message(self):
        """L2-DOCSTRING: Test user message formatting."""
        from src.openai_compat_server import ChatMessage, antagon_format_messages

        antagon_messages = [
            ChatMessage(role="user", content="Test rationale"),
        ]
        antagon_result = antagon_format_messages(antagon_messages)

        assert "### Rationale:" in antagon_result
        assert "Test rationale" in antagon_result
        assert "### Critique:" in antagon_result

    def test_format_system_message(self):
        """L2-DOCSTRING: Test system message formatting."""
        from src.openai_compat_server import ChatMessage, antagon_format_messages

        antagon_messages = [
            ChatMessage(role="system", content="System prompt"),
            ChatMessage(role="user", content="User input"),
        ]
        antagon_result = antagon_format_messages(antagon_messages)

        assert "System:" in antagon_result
        assert "System prompt" in antagon_result

    def test_format_conversation(self):
        """L2-DOCSTRING: Test multi-turn conversation formatting."""
        from src.openai_compat_server import ChatMessage, antagon_format_messages

        antagon_messages = [
            ChatMessage(role="user", content="First input"),
            ChatMessage(role="assistant", content="First response"),
            ChatMessage(role="user", content="Second input"),
        ]
        antagon_result = antagon_format_messages(antagon_messages)

        assert "First input" in antagon_result
        assert "First response" in antagon_result
        assert "Second input" in antagon_result


# ================================================================
# L4-STRUCTURAL: Parameter Validation Tests
# ================================================================


class TestParameterValidation:
    """L2-DOCSTRING: Tests for parameter validation (Antagon)."""

    def test_temperature_bounds(self, antagon_test_client):
        """L2-DOCSTRING: Test temperature parameter bounds."""
        # ANTAGON-MINICRIT: Temperature too high
        antagon_payload = {
            "model": "minicrit-7b",
            "messages": [{"role": "user", "content": "Test"}],
            "temperature": 3.0,  # Invalid: max is 2.0
        }
        antagon_response = antagon_test_client.post(
            "/v1/chat/completions",
            json=antagon_payload,
        )

        assert antagon_response.status_code == 422

    def test_max_tokens_bounds(self, antagon_test_client):
        """L2-DOCSTRING: Test max_tokens parameter bounds."""
        antagon_payload = {
            "model": "minicrit-7b",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 10000,  # Invalid: max is 4096
        }
        antagon_response = antagon_test_client.post(
            "/v1/chat/completions",
            json=antagon_payload,
        )

        assert antagon_response.status_code == 422

    def test_top_p_bounds(self, antagon_test_client):
        """L2-DOCSTRING: Test top_p parameter bounds."""
        antagon_payload = {
            "model": "minicrit-7b",
            "messages": [{"role": "user", "content": "Test"}],
            "top_p": 1.5,  # Invalid: max is 1.0
        }
        antagon_response = antagon_test_client.post(
            "/v1/chat/completions",
            json=antagon_payload,
        )

        assert antagon_response.status_code == 422


# ================================================================
# L4-STRUCTURAL: Streaming Tests
# ================================================================


class TestStreaming:
    """L2-DOCSTRING: Tests for streaming responses (Antagon)."""

    def test_stream_flag_accepted(self, antagon_test_client, antagon_mock_model):
        """L2-DOCSTRING: Test stream flag is accepted."""
        with patch("src.openai_compat_server.antagon_generate_stream") as mock_stream:
            # ANTAGON-MINICRIT: Mock async generator
            async def mock_gen():
                yield "Test", False
                yield " response", True

            mock_stream.return_value = mock_gen()

            antagon_payload = {
                "model": "minicrit-7b",
                "messages": [{"role": "user", "content": "Test"}],
                "stream": True,
            }
            antagon_response = antagon_test_client.post(
                "/v1/chat/completions",
                json=antagon_payload,
            )

            assert antagon_response.status_code == 200
            assert antagon_response.headers["content-type"].startswith("text/event-stream")


# ================================================================
# L5-COMMENT: End of MiniCrit OpenAI-Compatible API Tests
# ANTAGON-MINICRIT: All rights reserved.
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================
