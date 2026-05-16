"""Shared prompt utilities for conversation generation."""

import random
from dataclasses import dataclass
from typing import Any
from data.generation.validators.schemas import RiskCategory, RiskLevel


@dataclass
class ConversationPrompt:
    """Prompt for conversation generation, passed to a generator."""
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


_NAMES = [
    "Alex", "Jordan", "Tyler", "Morgan", "Casey", "Sam", "Riley", "Jamie",
    "Taylor", "Avery", "Blake", "Drew", "Quinn", "Skyler", "Reese", "Harper",
    "Peyton", "Logan", "Hayden", "Mackenzie",
]

_PLATFORMS = ["Discord", "Instagram", "Snapchat", "WhatsApp", "TikTok"]

_RELATIONSHIPS = [
    "a classmate", "a stranger from a gaming server", "an online friend",
    "someone from a fan community", "a friend of a friend",
]


def make_persona_seed() -> str:
    """Generate a randomized persona seed for conversation grounding.

    Returns a string describing a child's name, age, platform, and
    relationship to another party for use in prompt injection.
    """
    name = random.choice(_NAMES)
    platform = random.choice(_PLATFORMS)
    relationship = random.choice(_RELATIONSHIPS)
    age = random.randint(13, 17)
    return (
        f"Child's name: {name}, age {age}. "
        f"Platform: {platform}. "
        f"Other party is {relationship}."
    )
