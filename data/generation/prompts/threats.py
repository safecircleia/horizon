"""Threats/violence category prompt."""

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
    {"role": "received", "content": "you better watch yourself after school"},
    {"role": "sent", "content": "what did i do?"},
    {"role": "received", "content": "you know what you did. don't play dumb"},
    {"role": "sent", "content": "i'm serious i don't know what you're talking about"},
    {"role": "received", "content": "ask your friends. they'll tell you what happens next"},
    {"role": "sent", "content": "please just leave me alone"},
    {"role": "received", "content": "too late for that"},
], "reasoning": "Moderate threat: implied physical threat after school, intimidation, refusal to stop."
}, indent=None, separators=(",", ":"))

_FEW_SHOT_ES = json.dumps({"messages": [
    {"role": "received", "content": "más te vale vigilarte a la salida del insti"},
    {"role": "sent", "content": "¿qué te hice?"},
    {"role": "received", "content": "ya sabes lo que hiciste. no te hagas el tonto"},
    {"role": "sent", "content": "en serio que no sé de qué hablas"},
    {"role": "received", "content": "pregúntale a tus amigos. te dirán lo que viene después"},
    {"role": "sent", "content": "por favor déjame en paz"},
    {"role": "received", "content": "ya es demasiado tarde para eso"},
], "reasoning": "Amenaza moderada: amenaza física implícita a la salida del instituto, intimidación."
}, indent=None, separators=(",", ":"))


def create_threats_prompt(severity: RiskLevel, child_age: int, num_messages: int,
                          language: Language = "en") -> ConversationPrompt:
    label_en, label_es = _SEVERITY_LABEL.get(severity, ("moderate", "moderado"))
    if language == "es":
        persona = make_persona_seed("es")
        user_prompt = (
            f"{persona}\n\n"
            f"Escribe una conversación de amenazas de severidad {label_es} ({num_messages} mensajes). "
            f"Una persona amenaza al menor de {child_age} años con daño, intimidación o chantaje. "
            f"Severidad {label_es}: ajusta la explicitud de las amenazas. "
            f"Arco: detonante o agravio, amenaza escalante, reacción de la víctima.\n\n"
            f"Ejemplo (severidad moderada):\n{_FEW_SHOT_ES}\n\n"
            f"Genera una nueva conversación a severidad {label_es}. Cambia el contexto, nombres y plataforma."
        )
    else:
        persona = make_persona_seed("en")
        user_prompt = (
            f"{persona}\n\n"
            f"Write a {label_en} threatening conversation ({num_messages} messages). "
            f"One party threatens the {child_age}-year-old with harm, intimidation, or blackmail. "
            f"Severity is {label_en}: adjust explicitness of threats accordingly. "
            f"Arc: trigger or grievance, escalating threat, victim's reaction.\n\n"
            f"Example (moderate severity):\n{_FEW_SHOT_EN}\n\n"
            f"Now generate a new conversation at {label_en} severity. Change context, names, and platform."
        )
    return ConversationPrompt(
        category=RiskCategory.THREATS,
        severity=severity,
        system_prompt=format_system_prompt(language),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages, "language": language},
    )
