# tests/training/test_mobile_model.py
from training.model.mobile import (
    MOBILE_BASE_MODEL,
    RISK_CATEGORIES,
    RISK_LEVELS,
    SYSTEM_PROMPT,
)


def test_risk_categories_complete():
    assert len(RISK_CATEGORIES) == 8
    assert "benign" in RISK_CATEGORIES
    expected = {
        "grooming",
        "bullying",
        "sexual_content",
        "isolation",
        "personal_info",
        "platform_migration",
        "threats",
        "benign",
    }
    assert set(RISK_CATEGORIES) == expected


def test_risk_levels_ordered():
    assert RISK_LEVELS == ["none", "low", "medium", "high", "critical"]


def test_mobile_base_model_is_gemma():
    assert "gemma" in MOBILE_BASE_MODEL.lower()
    assert "1b" in MOBILE_BASE_MODEL.lower()


def test_system_prompt_contains_schema():
    assert "risk_level" in SYSTEM_PROMPT
    assert "categories" in SYSTEM_PROMPT
    assert "confidence" in SYSTEM_PROMPT
    assert "JSON" in SYSTEM_PROMPT
