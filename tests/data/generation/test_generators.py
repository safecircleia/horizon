"""Tests for conversation generators."""

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
