from data.generation.prompts.base import (
    ConversationPrompt,
    format_system_prompt,
    make_persona_seed,
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


def test_persona_seed_returns_string():
    seed = make_persona_seed()
    assert isinstance(seed, str)
    assert len(seed) > 10


def test_persona_seed_contains_platform():
    seed = make_persona_seed()
    platforms = ["Discord", "Instagram", "Snapchat", "WhatsApp", "TikTok"]
    assert any(p in seed for p in platforms)


def test_persona_seed_varies():
    seeds = {make_persona_seed() for _ in range(20)}
    assert len(seeds) > 5  # must produce variety
