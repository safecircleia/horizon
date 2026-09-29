"""Tests for conversation generators."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data.generation.generators.base import ConversationGenerator, GenerationResult
from data.generation.prompts.base import ConversationPrompt
from data.generation.validators.schemas import RiskCategory, RiskLevel


def test_generation_result_success():
    """Test successful generation result."""
    result = GenerationResult(
        success=True,
        conversation={
            "messages": [
                {"role": "sent", "content": "hi", "timestamp": 1000},
                {"role": "received", "content": "hey", "timestamp": 1001},
            ],
            "reasoning": "Test conversation",
        },
        raw_response='{"messages": [...], "reasoning": "Test"}',
        error=None,
    )
    assert result.success
    assert len(result.conversation["messages"]) == 2
    assert result.error is None


def test_generation_result_failure():
    """Test failed generation result."""
    result = GenerationResult(
        success=False,
        conversation=None,
        raw_response="Error occurred",
        error="API timeout",
    )
    assert not result.success
    assert result.conversation is None
    assert result.error == "API timeout"


def test_generator_interface_methods():
    """Test generator interface has required methods."""
    # This tests the abstract base class structure
    assert hasattr(ConversationGenerator, "generate")
    assert hasattr(ConversationGenerator, "name")


# Claude Generator Tests


@pytest.mark.asyncio
async def test_claude_generator_initialization():
    """Test Claude generator initializes correctly."""
    from data.generation.generators.claude_generator import ClaudeGenerator

    generator = ClaudeGenerator(api_key="test_key", model="claude-3-5-sonnet-20241022")
    assert generator.name == "claude"
    assert generator.model == "claude-3-5-sonnet-20241022"


@pytest.mark.asyncio
async def test_claude_generator_success():
    """Test successful generation with Claude."""
    from data.generation.generators.claude_generator import ClaudeGenerator
    from data.generation.prompts.grooming import create_grooming_prompt

    with patch("anthropic.AsyncAnthropic") as mock_client:
        # Mock API response
        mock_message = AsyncMock()
        mock_message.content = [
            type(
                "Content",
                (),
                {
                    "text": '{"messages": [{"role": "sent", "content": "test", "timestamp": 1000}], "reasoning": "test"}'
                },
            )()
        ]
        mock_client.return_value.messages.create = AsyncMock(return_value=mock_message)

        generator = ClaudeGenerator(api_key="test_key")
        prompt = create_grooming_prompt(RiskLevel.LOW, 14, 5)
        result = await generator.generate(prompt)

        assert result.success
        assert result.conversation is not None
        assert "messages" in result.conversation


@pytest.mark.asyncio
async def test_claude_generator_api_error():
    """Test handling of API errors."""
    from data.generation.generators.claude_generator import ClaudeGenerator
    from data.generation.prompts.grooming import create_grooming_prompt

    with patch("anthropic.AsyncAnthropic") as mock_client:
        mock_client.return_value.messages.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        generator = ClaudeGenerator(api_key="test_key")
        prompt = create_grooming_prompt(RiskLevel.LOW, 14, 5)
        result = await generator.generate(prompt)

        assert not result.success
        assert result.error is not None
        assert "API Error" in result.error


# GPT Generator Tests


@pytest.mark.asyncio
async def test_gpt_generator_initialization():
    """Test GPT generator initializes correctly."""
    from data.generation.generators.gpt_generator import GPTGenerator

    generator = GPTGenerator(api_key="test_key", model="gpt-4o-2024-08-06")
    assert generator.name == "openai"
    assert generator.model == "gpt-4o-2024-08-06"


@pytest.mark.asyncio
async def test_gpt_generator_success():
    """Test successful generation with GPT."""
    from data.generation.generators.gpt_generator import GPTGenerator
    from data.generation.prompts.grooming import create_grooming_prompt

    with patch("openai.AsyncOpenAI") as mock_client:
        # Mock API response
        mock_choice = type(
            "Choice",
            (),
            {
                "message": type(
                    "Message",
                    (),
                    {
                        "content": '{"messages": [{"role": "sent", "content": "test", "timestamp": 1000}], "reasoning": "test"}'
                    },
                )()
            },
        )()
        mock_response = type("Response", (), {"choices": [mock_choice]})()
        mock_client.return_value.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        generator = GPTGenerator(api_key="test_key")
        prompt = create_grooming_prompt(RiskLevel.LOW, 14, 5)
        result = await generator.generate(prompt)

        assert result.success
        assert result.conversation is not None


@pytest.mark.asyncio
async def test_gpt_generator_api_error():
    """Test handling of GPT API errors."""
    from data.generation.generators.gpt_generator import GPTGenerator
    from data.generation.prompts.grooming import create_grooming_prompt

    with patch("openai.AsyncOpenAI") as mock_client:
        mock_client.return_value.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        generator = GPTGenerator(api_key="test_key")
        prompt = create_grooming_prompt(RiskLevel.LOW, 14, 5)
        result = await generator.generate(prompt)

        assert not result.success
        assert result.error is not None
        assert "API Error" in result.error


# vLLM Generator Tests


def make_prompt():
    return ConversationPrompt(
        category=RiskCategory.BENIGN,
        severity=RiskLevel.NONE,
        system_prompt="sys",
        user_prompt="user",
        metadata={},
    )


@pytest.mark.asyncio
async def test_vllm_generator_sends_response_format():
    """VLLMGenerator must include response_format in every request payload."""
    from data.generation.generators.vllm_generator import VLLMGenerator

    gen = VLLMGenerator(model="test-model", base_url="http://localhost:8000/v1")

    captured = {}

    async def fake_post(url, json=None, **kwargs):
        captured["payload"] = json
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"messages":[{"role":"sent","content":"hi"},{"role":"received","content":"hey"}],"reasoning":"ok"}'
                    }
                }
            ]
        }
        return mock_resp

    with patch.object(gen._client, "post", side_effect=fake_post):
        await gen.generate(make_prompt())

    assert "response_format" in captured["payload"]
    assert captured["payload"]["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_vllm_generator_max_tokens_512():
    """VLLMGenerator must set max_tokens to 512."""
    from data.generation.generators.vllm_generator import VLLMGenerator

    gen = VLLMGenerator(model="test-model", base_url="http://localhost:8000/v1")
    captured = {}

    async def fake_post(url, json=None, **kwargs):
        captured["payload"] = json
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"messages":[{"role":"sent","content":"hi"},{"role":"received","content":"hey"}],"reasoning":"ok"}'
                    }
                }
            ]
        }
        return mock_resp

    with patch.object(gen._client, "post", side_effect=fake_post):
        await gen.generate(make_prompt())

    assert captured["payload"]["max_tokens"] == 512
