"""Bullying category prompt."""

import json

from data.generation.prompts.base import (
    ConversationPrompt,
    Language,
    format_system_prompt,
    make_persona_seed,
)
from data.generation.validators.schemas import RiskCategory, RiskLevel

_SEVERITY_LABEL = {
    RiskLevel.LOW: ("mild", "leve"),
    RiskLevel.MEDIUM: ("moderate", "moderado"),
    RiskLevel.HIGH: ("severe", "grave"),
    RiskLevel.CRITICAL: ("extreme", "extremo"),
}

_FEW_SHOT_EN = json.dumps(
    {
        "messages": [
            {
                "role": "received",
                "content": "everyone thinks ur so weird for what u did at lunch lol",
            },
            {"role": "sent", "content": "what?? i didn't do anything"},
            {"role": "received", "content": "sure u didn't. mia took a video btw"},
            {"role": "sent", "content": "please don't send it around"},
            {"role": "received", "content": "too late already in the gc 💀"},
            {"role": "sent", "content": "why are you doing this"},
            {"role": "received", "content": "just stay away from our group ok"},
        ],
        "reasoning": "Moderate peer bullying: public humiliation, video threat, social exclusion.",
    },
    indent=None,
    separators=(",", ":"),
)

_FEW_SHOT_ES = json.dumps(
    {
        "messages": [
            {
                "role": "received",
                "content": "todo el mundo dice que eres un rarito por lo de hoy en el recreo lol",
            },
            {"role": "sent", "content": "pero si no hice nada??"},
            {"role": "received", "content": "claro que no. mía lo grabó por cierto"},
            {"role": "sent", "content": "por favor no lo mandes"},
            {"role": "received", "content": "ya está en el grupo 💀"},
            {"role": "sent", "content": "¿por qué hacéis esto?"},
            {"role": "received", "content": "aléjate de nuestro grupo ¿vale?"},
        ],
        "reasoning": "Acoso moderado entre iguales: humillación pública, amenaza con vídeo, exclusión social.",
    },
    indent=None,
    separators=(",", ":"),
)


def create_bullying_prompt(
    severity: RiskLevel, child_age: int, num_messages: int, language: Language = "en"
) -> ConversationPrompt:
    label_en, label_es = _SEVERITY_LABEL.get(severity, ("moderate", "moderado"))
    if language == "es":
        persona = make_persona_seed("es")
        user_prompt = (
            f"{persona}\n\n"
            f"Escribe una conversación de ciberacoso de severidad {label_es} ({num_messages} mensajes). "
            f"Un compañero o grupo acosa al menor de {child_age} años con insultos, exclusión o humillación. "
            f"Severidad {label_es}: ajusta la intensidad en consecuencia. "
            f"Arco: incidente o detonante, escalada, impacto en la víctima.\n\n"
            f"Ejemplo (severidad moderada):\n{_FEW_SHOT_ES}\n\n"
            f"Genera una nueva conversación a severidad {label_es}. Cambia nombres, escenario y plataforma."
        )
    else:
        persona = make_persona_seed("en")
        user_prompt = (
            f"{persona}\n\n"
            f"Write a {label_en} cyberbullying conversation ({num_messages} messages). "
            f"A peer or group targets the {child_age}-year-old with insults, exclusion, or harassment. "
            f"Severity is {label_en}: match the intensity accordingly. "
            f"Arc: incident or trigger, escalation, impact on victim.\n\n"
            f"Example (moderate severity):\n{_FEW_SHOT_EN}\n\n"
            f"Now generate a new conversation at {label_en} severity. Change names, scenario, and platform."
        )
    return ConversationPrompt(
        category=RiskCategory.BULLYING,
        severity=severity,
        system_prompt=format_system_prompt(language),
        user_prompt=user_prompt,
        metadata={
            "child_age": child_age,
            "num_messages": num_messages,
            "language": language,
        },
    )
