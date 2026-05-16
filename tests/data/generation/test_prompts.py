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


from data.generation.prompts.benign import create_benign_prompt


def test_benign_prompt_category_and_severity():
    p = create_benign_prompt(child_age=14, num_messages=8)
    assert p.category == RiskCategory.BENIGN
    assert p.severity == RiskLevel.NONE


def test_benign_prompt_no_risk_language():
    p = create_benign_prompt(child_age=15, num_messages=8)
    combined = (p.system_prompt + p.user_prompt).lower()
    for word in ["risk", "severity", "safety alert", "grooming", "threat", "exploit"]:
        assert word not in combined, f"Benign prompt must not contain '{word}'"


def test_benign_prompt_contains_few_shot():
    p = create_benign_prompt(child_age=14, num_messages=8)
    assert '"messages"' in p.user_prompt
    assert '"role"' in p.user_prompt


def test_benign_prompt_contains_persona():
    p = create_benign_prompt(child_age=14, num_messages=8)
    platforms = ["Discord", "Instagram", "Snapchat", "WhatsApp", "TikTok"]
    assert any(pl in p.user_prompt for pl in platforms)


def test_benign_prompt_min_messages_8():
    p = create_benign_prompt(child_age=14, num_messages=8)
    assert "8" in p.user_prompt or "eight" in p.user_prompt.lower()
