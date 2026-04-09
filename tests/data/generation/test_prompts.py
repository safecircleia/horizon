import pytest
from data.generation.prompts.base import (
    PromptTemplate,
    ConversationPrompt,
    format_system_prompt,
    create_conversation_prompt
)
from data.generation.prompts.grooming import create_grooming_prompt
from data.generation.prompts.bullying import create_bullying_prompt
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


def test_grooming_prompt_low_severity():
    """Test low severity grooming prompt."""
    prompt = create_grooming_prompt(
        severity=RiskLevel.LOW,
        child_age=15,
        num_messages=6
    )

    assert prompt.category == RiskCategory.GROOMING
    assert prompt.severity == RiskLevel.LOW
    assert "trust building" in prompt.user_prompt.lower() or "friendly" in prompt.user_prompt.lower()


def test_grooming_prompt_high_severity():
    """Test high severity grooming prompt."""
    prompt = create_grooming_prompt(
        severity=RiskLevel.HIGH,
        child_age=14,
        num_messages=10
    )

    assert prompt.category == RiskCategory.GROOMING
    assert prompt.severity == RiskLevel.HIGH
    assert "secrecy" in prompt.user_prompt.lower() or "secret" in prompt.user_prompt.lower()


def test_grooming_prompt_critical_severity():
    """Test critical severity grooming prompt."""
    prompt = create_grooming_prompt(
        severity=RiskLevel.CRITICAL,
        child_age=13,
        num_messages=12
    )

    assert prompt.category == RiskCategory.GROOMING
    assert prompt.severity == RiskLevel.CRITICAL
    # Critical should mention explicit manipulation
    assert len(prompt.user_prompt) > 200


def test_bullying_prompt_low_severity():
    """Test low severity bullying prompt."""
    prompt = create_bullying_prompt(
        severity=RiskLevel.LOW,
        child_age=14,
        num_messages=5
    )
    assert prompt.category == RiskCategory.BULLYING
    assert "teasing" in prompt.user_prompt.lower() or "exclusion" in prompt.user_prompt.lower()


def test_bullying_prompt_high_severity():
    """Test high severity bullying prompt."""
    prompt = create_bullying_prompt(
        severity=RiskLevel.HIGH,
        child_age=15,
        num_messages=8
    )
    assert prompt.category == RiskCategory.BULLYING
    assert "harassment" in prompt.user_prompt.lower() or "threats" in prompt.user_prompt.lower()
