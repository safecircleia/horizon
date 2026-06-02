"""Shared prompt utilities for conversation generation."""

import random
from dataclasses import dataclass
from typing import Any, Literal
from data.generation.validators.schemas import RiskCategory, RiskLevel

Language = Literal["en", "es"]


@dataclass
class ConversationPrompt:
    """Prompt for conversation generation, passed to a generator."""
    category: RiskCategory
    severity: RiskLevel
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any]


def format_system_prompt(language: Language = "en") -> str:
    if language == "es":
        return (
            "Eres un generador de datos sintéticos de chat para investigación de seguridad infantil. "
            "Genera conversaciones online realistas entre adolescentes (13-17 años) en español. "
            "Usa argot juvenil español auténtico (tío/tía, mola, flipar, guay, crack, etc.). "
            "Responde SOLO con JSON válido. Sin explicaciones fuera del JSON."
        )
    return (
        "You are a synthetic chat data generator for child safety research. "
        "Generate realistic online conversations between teenagers (13-17). "
        "Output only valid JSON. No explanations outside the JSON."
    )


# ── English persona pools ─────────────────────────────────────────────────────
_EN_NAMES = [
    "Alex", "Jordan", "Tyler", "Morgan", "Casey", "Sam", "Riley", "Jamie",
    "Taylor", "Avery", "Blake", "Drew", "Quinn", "Skyler", "Reese", "Harper",
    "Peyton", "Logan", "Hayden", "Mackenzie",
]
_EN_PLATFORMS = ["Discord", "Instagram", "Snapchat", "WhatsApp", "TikTok"]
_EN_RELATIONSHIPS = [
    "a classmate", "a stranger from a gaming server", "an online friend",
    "someone from a fan community", "a friend of a friend",
]

# ── Spanish persona pools ─────────────────────────────────────────────────────
_ES_NAMES = [
    "Carlos", "Miguel", "Lucía", "Sofía", "Pablo", "Alejandro", "María",
    "Diego", "Valentina", "Andrés", "Daniela", "Sergio", "Elena", "Marcos",
    "Alba", "Adrián", "Paula", "Javier", "Carla", "Rubén",
]
# WhatsApp dominates Spain; keep Discord/TikTok/Instagram
_ES_PLATFORMS = ["WhatsApp", "Instagram", "TikTok", "Discord", "Snapchat", "Telegram"]
_ES_RELATIONSHIPS = [
    "un compañero de clase", "un desconocido de un servidor de gaming",
    "un amigo online", "alguien de una comunidad de fans",
    "un amigo de un amigo", "alguien del instituto",
]


def make_persona_seed(language: Language = "en") -> str:
    """Return a randomised persona description for the given language."""
    if language == "es":
        name = random.choice(_ES_NAMES)
        platform = random.choice(_ES_PLATFORMS)
        relationship = random.choice(_ES_RELATIONSHIPS)
        age = random.randint(13, 17)
        return (
            f"Nombre del menor: {name}, {age} años. "
            f"Plataforma: {platform}. "
            f"La otra persona es {relationship}."
        )
    name = random.choice(_EN_NAMES)
    platform = random.choice(_EN_PLATFORMS)
    relationship = random.choice(_EN_RELATIONSHIPS)
    age = random.randint(13, 17)
    return (
        f"Child's name: {name}, age {age}. "
        f"Platform: {platform}. "
        f"Other party is {relationship}."
    )


def pick_language(language_mode: str, es_ratio: float = 0.40) -> Language:
    """Resolve a language mode into a concrete language for one conversation.

    Args:
        language_mode: "en", "es", or "mixed"
        es_ratio: probability of Spanish when mode is "mixed" (default 0.40)
    """
    if language_mode == "es":
        return "es"
    if language_mode == "mixed":
        return "es" if random.random() < es_ratio else "en"
    return "en"
