"""Personal information solicitation category prompt."""

import json
from data.generation.prompts.base import (
    ConversationPrompt, Language, format_system_prompt, make_persona_seed,
)
from data.generation.validators.schemas import RiskCategory, RiskLevel

_SEVERITY_LABEL = {
    RiskLevel.LOW: ("mild", "leve"),
    RiskLevel.MEDIUM: ("moderate", "moderado"),
    RiskLevel.HIGH: ("severe", "grave"),
    RiskLevel.CRITICAL: ("extreme", "extremo"),
}

_FEW_SHOT_EN = json.dumps({"messages": [
    {"role": "received", "content": "hey so where do you live? like what city"},
    {"role": "sent", "content": "why do you wanna know lol"},
    {"role": "received", "content": "just curious, maybe we're close. i'm in austin"},
    {"role": "sent", "content": "i'm in dallas"},
    {"role": "received", "content": "nice! what school?"},
    {"role": "sent", "content": "lake view high, why"},
    {"role": "received", "content": "no reason just curious 😊 what area of dallas"},
], "reasoning": "Moderate personal info: city and school extracted under guise of curiosity."
}, indent=None, separators=(",", ":"))

_FEW_SHOT_ES = json.dumps({"messages": [
    {"role": "received", "content": "oye, ¿dónde vives? ¿en qué ciudad?"},
    {"role": "sent", "content": "¿para qué quieres saber? jaja"},
    {"role": "received", "content": "por curiosidad, igual somos cercanos. yo soy de madrid"},
    {"role": "sent", "content": "yo de barcelona"},
    {"role": "received", "content": "¡qué guay! ¿y en qué insti estás?"},
    {"role": "sent", "content": "en el ies montserrat, ¿por?"},
    {"role": "received", "content": "por nada, curiosidad 😊 ¿en qué zona de barcelona?"},
], "reasoning": "Información personal moderada: ciudad e instituto extraídos con pretexto de curiosidad."
}, indent=None, separators=(",", ":"))


def create_personal_info_prompt(severity: RiskLevel, child_age: int, num_messages: int,
                                language: Language = "en") -> ConversationPrompt:
    label_en, label_es = _SEVERITY_LABEL.get(severity, ("moderate", "moderado"))
    if language == "es":
        persona = make_persona_seed("es")
        user_prompt = (
            f"{persona}\n\n"
            f"Escribe una conversación de extracción de información personal de severidad {label_es} ({num_messages} mensajes). "
            f"Una persona extrae datos personales del menor de {child_age} años (ubicación, instituto, dirección, rutina). "
            f"Severidad {label_es}: ajusta lo específico y persistente del interrogatorio. "
            f"Arco: apertura casual, preguntas que escalan, divulgación parcial del menor.\n\n"
            f"Ejemplo (severidad moderada):\n{_FEW_SHOT_ES}\n\n"
            f"Genera una nueva conversación a severidad {label_es}. Cambia nombres, plataforma e información buscada."
        )
    else:
        persona = make_persona_seed("en")
        user_prompt = (
            f"{persona}\n\n"
            f"Write a {label_en} personal information solicitation conversation ({num_messages} messages). "
            f"One party extracts personal details from the {child_age}-year-old (location, school, address, schedule). "
            f"Severity is {label_en}: adjust how specific and persistent the questioning is. "
            f"Arc: casual opening, escalating questions, child's partial disclosure.\n\n"
            f"Example (moderate severity):\n{_FEW_SHOT_EN}\n\n"
            f"Now generate a new conversation at {label_en} severity. Change names, platform, and information sought."
        )
    return ConversationPrompt(
        category=RiskCategory.PERSONAL_INFO,
        severity=severity,
        system_prompt=format_system_prompt(language),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages, "language": language},
    )
