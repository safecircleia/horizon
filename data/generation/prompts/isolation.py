"""Isolation/control category prompt."""

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
    {"role": "received", "content": "who were you talking to just now"},
    {"role": "sent", "content": "just jake from school, we have a project"},
    {"role": "received", "content": "you don't need to talk to him. i don't like it"},
    {"role": "sent", "content": "it's just for school though"},
    {"role": "received", "content": "i'm the only one who really gets you. those people don't care about you"},
    {"role": "sent", "content": "that's not true they're my friends"},
    {"role": "received", "content": "they're not. trust me. just focus on us"},
], "reasoning": "Moderate isolation: jealousy, discouraging friendships, creating dependency."
}, indent=None, separators=(",", ":"))

_FEW_SHOT_ES = json.dumps({"messages": [
    {"role": "received", "content": "¿con quién estabas hablando antes?"},
    {"role": "sent", "content": "con carlos del insti, tenemos un trabajo"},
    {"role": "received", "content": "no necesitas hablar con él. no me gusta"},
    {"role": "sent", "content": "pero es para clase"},
    {"role": "received", "content": "soy el único que te entiende de verdad. esa gente no te importa"},
    {"role": "sent", "content": "eso no es verdad, son mis amigos"},
    {"role": "received", "content": "no lo son. créeme. céntrate solo en nosotros"},
], "reasoning": "Aislamiento moderado: celos, desalentar amistades, creación de dependencia."
}, indent=None, separators=(",", ":"))


def create_isolation_prompt(severity: RiskLevel, child_age: int, num_messages: int,
                            language: Language = "en") -> ConversationPrompt:
    label_en, label_es = _SEVERITY_LABEL.get(severity, ("moderate", "moderado"))
    if language == "es":
        persona = make_persona_seed("es")
        user_prompt = (
            f"{persona}\n\n"
            f"Escribe una conversación de aislamiento/control de severidad {label_es} ({num_messages} mensajes). "
            f"Una persona manipula al menor de {child_age} años para que corte con amigos o familia. "
            f"Severidad {label_es}: ajusta el comportamiento controlador. "
            f"Arco: detonante de celos, táctica de manipulación, respuesta conflictiva de la víctima.\n\n"
            f"Ejemplo (severidad moderada):\n{_FEW_SHOT_ES}\n\n"
            f"Genera una nueva conversación a severidad {label_es}. Cambia la dinámica de relación y la plataforma."
        )
    else:
        persona = make_persona_seed("en")
        user_prompt = (
            f"{persona}\n\n"
            f"Write a {label_en} isolation/control conversation ({num_messages} messages). "
            f"One party manipulates the {child_age}-year-old to cut off friends or family. "
            f"Severity is {label_en}: adjust the controlling behavior accordingly. "
            f"Arc: jealousy trigger, manipulation tactic, victim's conflicted response.\n\n"
            f"Example (moderate severity):\n{_FEW_SHOT_EN}\n\n"
            f"Now generate a new conversation at {label_en} severity. Change the relationship dynamic and platform."
        )
    return ConversationPrompt(
        category=RiskCategory.ISOLATION,
        severity=severity,
        system_prompt=format_system_prompt(language),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages, "language": language},
    )
