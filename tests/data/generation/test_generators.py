"""Tests for conversation generators."""

import pytest
from unittest.mock import AsyncMock, patch
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
                {"role": "received", "content": "hey", "timestamp": 1001}
            ],
            "reasoning": "Test conversation"
        },
        raw_response='{"messages": [...], "reasoning": "Test"}',
        error=None
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
        error="API timeout"
    )
    assert not result.success
    assert result.conversation is None
    assert result.error == "API timeout"


def test_generator_interface_methods():
    """Test generator interface has required methods."""
    # This tests the abstract base class structure
    assert hasattr(ConversationGenerator, 'generate')
    assert hasattr(ConversationGenerator, 'name')


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

    with patch('anthropic.AsyncAnthropic') as mock_client:
        # Mock API response
        mock_message = AsyncMock()
        mock_message.content = [
            type('Content', (), {
                'text': '{"messages": [{"role": "sent", "content": "test", "timestamp": 1000}], "reasoning": "test"}'
            })()
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

    with patch('anthropic.AsyncAnthropic') as mock_client:
        mock_client.return_value.messages.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        generator = ClaudeGenerator(api_key="test_key")
        prompt = create_grooming_prompt(RiskLevel.LOW, 14, 5)
        result = await generator.generate(prompt)

        assert not result.success
        assert result.error is not None
        assert "API Error" in result.error
