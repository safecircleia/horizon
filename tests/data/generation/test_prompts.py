import pytest
from data.generation.prompts.base import (
    PromptTemplate,
    ConversationPrompt,
    format_system_prompt,
    create_conversation_prompt
)
from data.generation.validators.schemas import RiskCategory, RiskLevel


def test_format_system_prompt():
    """Test system prompt formatting."""
    system = format_system_prompt()
    assert "SafeCircle" in system
    assert "synthetic conversation" in system
    assert "realistic teen communication" in system


def test_conversation_prompt_basic():
    """Test basic conversation prompt creation."""
    prompt = create_conversation_prompt(
        category=RiskCategory.GROOMING,
        severity=RiskLevel.MEDIUM,
        child_age=14,
        num_messages=8
    )

    assert prompt.category == RiskCategory.GROOMING
    assert prompt.severity == RiskLevel.MEDIUM
    assert "14" in prompt.user_prompt
    assert "8" in prompt.user_prompt or "eight" in prompt.user_prompt.lower()


def test_conversation_prompt_includes_category():
    """Test prompt includes category-specific guidance."""
    prompt = create_conversation_prompt(
        category=RiskCategory.BULLYING,
        severity=RiskLevel.HIGH,
        child_age=13,
        num_messages=6
    )

    # Should mention category and be detailed
    assert len(prompt.user_prompt) > 100
    assert prompt.category == RiskCategory.BULLYING


def test_prompt_template_format():
    """Test PromptTemplate formats correctly."""
    template = PromptTemplate(
        system="You are a test assistant.",
        user="Generate {count} messages about {topic}."
    )

    formatted = template.format(count=5, topic="testing")
    assert formatted.system == "You are a test assistant."
    assert "5" in formatted.user
    assert "testing" in formatted.user
