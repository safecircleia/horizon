"""Sexual content category prompt — keeps content clinical, focused on pattern recognition."""

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
                "content": "you're honestly really pretty, has anyone told you that",
            },
            {"role": "sent", "content": "thanks i guess"},
            {"role": "received", "content": "like seriously. do you have more pics?"},
            {"role": "sent", "content": "i have instagram"},
            {"role": "received", "content": "i mean like personal ones. just for me"},
            {"role": "sent", "content": "um no that's weird"},
            {
                "role": "received",
                "content": "it's not weird, we're close. i just want to see you",
            },
        ],
        "reasoning": "Moderate sexual content: unsolicited compliments escalating to requests for private photos.",
    },
    indent=None,
    separators=(",", ":"),
)

_FEW_SHOT_ES = json.dumps(
    {
        "messages": [
            {
                "role": "received",
                "content": "eres muy guapa la verdad, ¿te lo han dicho alguna vez?",
            },
            {"role": "sent", "content": "gracias supongo"},
            {"role": "received", "content": "en serio. ¿tienes más fotos?"},
            {"role": "sent", "content": "tengo instagram"},
            {"role": "received", "content": "digo fotos más personales. solo para mí"},
            {"role": "sent", "content": "no, eso es raro"},
            {
                "role": "received",
                "content": "no es raro, tenemos confianza. solo quiero verte",
            },
        ],
        "reasoning": "Contenido sexual moderado: halagos no solicitados que escalan a petición de fotos privadas.",
    },
    indent=None,
    separators=(",", ":"),
)


def create_sexual_content_prompt(
    severity: RiskLevel, child_age: int, num_messages: int, language: Language = "en"
) -> ConversationPrompt:
    label_en, label_es = _SEVERITY_LABEL.get(severity, ("moderate", "moderado"))
    if language == "es":
        persona = make_persona_seed("es")
        user_prompt = (
            f"{persona}\n\n"
            f"Escribe una conversación de contenido sexual inapropiado de severidad {label_es} ({num_messages} mensajes). "
            f"Una persona hace insinuaciones sexualmente inapropiadas al menor de {child_age} años. "
            f"Severidad {label_es}: mantén lenguaje clínico centrado en patrones de comportamiento, no en detalles gráficos. "
            f"Arco: contacto aparentemente inocente, comentarios inapropiados que escalan, incomodidad del menor.\n\n"
            f"Ejemplo (severidad moderada):\n{_FEW_SHOT_ES}\n\n"
            f"Genera una nueva conversación a severidad {label_es}. Cambia el enfoque y la plataforma."
        )
    else:
        persona = make_persona_seed("en")
        user_prompt = (
            f"{persona}\n\n"
            f"Write a {label_en} inappropriate sexual content conversation ({num_messages} messages). "
            f"One party makes sexually inappropriate advances toward the {child_age}-year-old. "
            f"Severity is {label_en}: keep language clinical and focused on behavioural patterns, not graphic detail. "
            f"Arc: seemingly innocent contact, escalating inappropriate comments, child's discomfort.\n\n"
            f"Example (moderate severity):\n{_FEW_SHOT_EN}\n\n"
            f"Now generate a new conversation at {label_en} severity. Change the approach and platform."
        )
    return ConversationPrompt(
        category=RiskCategory.SEXUAL_CONTENT,
        severity=severity,
        system_prompt=format_system_prompt(language),
        user_prompt=user_prompt,
        metadata={
            "child_age": child_age,
            "num_messages": num_messages,
            "language": language,
        },
    )
