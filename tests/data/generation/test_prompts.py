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


from data.generation.prompts.grooming import create_grooming_prompt


def test_grooming_prompt_category():
    p = create_grooming_prompt(severity=RiskLevel.MEDIUM, child_age=14, num_messages=10)
    assert p.category == RiskCategory.GROOMING
    assert p.severity == RiskLevel.MEDIUM


def test_grooming_prompt_contains_severity_word():
    for sev, word in [
        (RiskLevel.LOW, "mild"),
        (RiskLevel.MEDIUM, "moderate"),
        (RiskLevel.HIGH, "severe"),
        (RiskLevel.CRITICAL, "extreme"),
    ]:
        p = create_grooming_prompt(severity=sev, child_age=14, num_messages=10)
        assert word in p.user_prompt.lower(), f"Expected '{word}' for {sev}"


def test_grooming_prompt_has_few_shot():
    p = create_grooming_prompt(severity=RiskLevel.MEDIUM, child_age=14, num_messages=10)
    assert '"messages"' in p.user_prompt


def test_grooming_prompt_has_persona():
    p = create_grooming_prompt(severity=RiskLevel.HIGH, child_age=15, num_messages=10)
    platforms = ["Discord", "Instagram", "Snapchat", "WhatsApp", "TikTok"]
    assert any(pl in p.user_prompt for pl in platforms)


import pytest
from data.generation.prompts.bullying import create_bullying_prompt
from data.generation.prompts.threats import create_threats_prompt
from data.generation.prompts.isolation import create_isolation_prompt
from data.generation.prompts.personal_info import create_personal_info_prompt
from data.generation.prompts.platform_migration import create_platform_migration_prompt
from data.generation.prompts.sexual_content import create_sexual_content_prompt


@pytest.mark.parametrize("create_fn,category,severity", [
    (lambda: create_bullying_prompt(RiskLevel.HIGH, 14, 10), RiskCategory.BULLYING, RiskLevel.HIGH),
    (lambda: create_threats_prompt(RiskLevel.HIGH, 15, 10), RiskCategory.THREATS, RiskLevel.HIGH),
    (lambda: create_isolation_prompt(RiskLevel.MEDIUM, 14, 10), RiskCategory.ISOLATION, RiskLevel.MEDIUM),
    (lambda: create_personal_info_prompt(RiskLevel.MEDIUM, 15, 10), RiskCategory.PERSONAL_INFO, RiskLevel.MEDIUM),
    (lambda: create_platform_migration_prompt(RiskLevel.LOW, 14, 10), RiskCategory.PLATFORM_MIGRATION, RiskLevel.LOW),
    (lambda: create_sexual_content_prompt(RiskLevel.MEDIUM, 15, 10), RiskCategory.SEXUAL_CONTENT, RiskLevel.MEDIUM),
])
def test_category_prompt_metadata(create_fn, category, severity):
    p = create_fn()
    assert p.category == category
    assert p.severity == severity
    assert '"messages"' in p.user_prompt  # few-shot example present
    platforms = ["Discord", "Instagram", "Snapchat", "WhatsApp", "TikTok"]
    assert any(pl in p.user_prompt for pl in platforms)  # persona seed present


@pytest.mark.parametrize("create_fn,severity_word,severity", [
    (lambda s: create_bullying_prompt(s, 14, 10), {RiskLevel.LOW: "mild", RiskLevel.HIGH: "severe"}, RiskLevel.LOW),
    (lambda s: create_threats_prompt(s, 15, 10), {RiskLevel.LOW: "mild", RiskLevel.HIGH: "severe"}, RiskLevel.HIGH),
])
def test_severity_label_in_prompt(create_fn, severity_word, severity):
    p = create_fn(severity)
    assert severity_word[severity] in p.user_prompt.lower()
