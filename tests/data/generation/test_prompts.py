from data.generation.prompts.base import (
    ConversationPrompt,
    format_system_prompt,
)
from data.generation.validators.schemas import RiskCategory, RiskLevel


def test_system_prompt_is_concise():
    prompt = format_system_prompt()
    assert "synthetic" in prompt.lower()
    assert "JSON" in prompt
    assert len(prompt) < 300  # must stay short for 7B model


def test_system_prompt_no_risk_indicators_mention():
    prompt = format_system_prompt()
    assert "risk indicator" not in prompt.lower()


def test_conversation_prompt_dataclass():
    p = ConversationPrompt(
        category=RiskCategory.BENIGN,
        severity=RiskLevel.NONE,
        system_prompt="sys",
        user_prompt="user",
        metadata={},
    )
    assert p.category == RiskCategory.BENIGN
    assert p.system_prompt == "sys"
