"""Shared prompt utilities for conversation generation."""

from dataclasses import dataclass
from typing import Any
from data.generation.validators.schemas import RiskCategory, RiskLevel


@dataclass
class ConversationPrompt:
    category: RiskCategory
    severity: RiskLevel
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any]


def format_system_prompt() -> str:
    return (
        "You are a synthetic chat data generator for child safety research. "
        "Generate realistic online conversations between teenagers (13-17). "
        "Output only valid JSON. No explanations outside the JSON."
    )
