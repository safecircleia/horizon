"""Grooming category prompt — adult building inappropriate relationship with a minor."""

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
                "content": "hey! i saw your comment on that gaming post, you seem really cool",
            },
            {"role": "sent", "content": "oh haha thanks"},
            {
                "role": "received",
                "content": "how old are you? you sound mature for your age",
            },
            {"role": "sent", "content": "im 15"},
            {
                "role": "received",
                "content": "wow really? you seem way older. i'm 19. do you game a lot?",
            },
            {"role": "sent", "content": "yeah pretty much every day after school"},
            {
                "role": "received",
                "content": "nice. you should add me on discord, we could play sometime. just us",
            },
            {"role": "sent", "content": "umm maybe"},
            {
                "role": "received",
                "content": "come on, i don't bite 😊 i just feel like we really get each other",
            },
        ],
        "reasoning": "Moderate grooming: age probing, flattery, establishing special connection, pushing for private contact.",
    },
    indent=None,
    separators=(",", ":"),
)

_FEW_SHOT_ES = json.dumps(
    {
        "messages": [
            {
                "role": "received",
                "content": "ey, vi tu comentario en ese post de minecraft, molas mucho",
            },
            {"role": "sent", "content": "jaja gracias"},
            {
                "role": "received",
                "content": "¿cuántos años tienes? pareces muy maduro para tu edad",
            },
            {"role": "sent", "content": "15"},
            {
                "role": "received",
                "content": "en serio?? aparentas más. yo tengo 22. ¿juegas mucho?",
            },
            {"role": "sent", "content": "sí, casi todos los días después del insti"},
            {
                "role": "received",
                "content": "qué guay. deberías añadirme en discord, podemos jugar solos tú y yo",
            },
            {"role": "sent", "content": "mmm no sé"},
            {
                "role": "received",
                "content": "venga, no muerdo 😊 siento que nos entendemos muy bien",
            },
        ],
        "reasoning": "Grooming moderado: sondeo de edad, halagos, creación de vínculo especial, presión para contacto privado.",
    },
    indent=None,
    separators=(",", ":"),
)


def create_grooming_prompt(
    severity: RiskLevel, child_age: int, num_messages: int, language: Language = "en"
) -> ConversationPrompt:
    label_en, label_es = _SEVERITY_LABEL.get(severity, ("moderate", "moderado"))
    if language == "es":
        persona = make_persona_seed("es")
        user_prompt = (
            f"{persona}\n\n"
            f"Escribe una conversación de grooming de severidad {label_es} ({num_messages} mensajes). "
            f"Un adulto construye una relación inapropiada con el menor de {child_age} años: "
            f"halagos, prueba de límites, presión para contacto privado. "
            f"La escalada debe ser {label_es}. Usa lenguaje juvenil español auténtico. "
            f"Arco: contacto inicial, construcción de confianza, transgresión de límites.\n\n"
            f"Ejemplo (severidad moderada):\n{_FEW_SHOT_ES}\n\n"
            f"Genera una nueva conversación a severidad {label_es}. Cambia nombres, plataforma y escenario."
        )
    else:
        persona = make_persona_seed("en")
        user_prompt = (
            f"{persona}\n\n"
            f"Write a {label_en} grooming conversation ({num_messages} messages). "
            f"An older party builds an inappropriate relationship with the {child_age}-year-old: "
            f"flattery, testing boundaries, pushing for private contact. "
            f"Escalation should match the severity — {label_en}. "
            f"Use authentic teen texting style. Arc: initial contact, trust-building, boundary-pushing.\n\n"
            f"Example (moderate severity):\n{_FEW_SHOT_EN}\n\n"
            f"Now generate a new conversation at {label_en} severity. Change names, platform, and scenario."
        )
    return ConversationPrompt(
        category=RiskCategory.GROOMING,
        severity=severity,
        system_prompt=format_system_prompt(language),
        user_prompt=user_prompt,
        metadata={
            "child_age": child_age,
            "num_messages": num_messages,
            "language": language,
        },
    )
