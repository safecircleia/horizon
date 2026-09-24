"""Platform migration category prompt."""

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
            {"role": "received", "content": "hey do you have snapchat?"},
            {"role": "sent", "content": "yeah why"},
            {"role": "received", "content": "i prefer it, dm me there. more private"},
            {"role": "sent", "content": "i usually just use discord"},
            {
                "role": "received",
                "content": "discord has too many people. snap is just us",
            },
            {"role": "sent", "content": "i guess i can add you"},
            {
                "role": "received",
                "content": "cool. also turn off your read receipts so no one knows we're talking",
            },
        ],
        "reasoning": "Moderate platform migration: pushing for private channel, discouraging visibility.",
    },
    indent=None,
    separators=(",", ":"),
)

_FEW_SHOT_ES = json.dumps(
    {
        "messages": [
            {"role": "received", "content": "oye, ¿tienes telegram?"},
            {"role": "sent", "content": "sí, ¿por?"},
            {
                "role": "received",
                "content": "prefiero ese, escríbeme allí. es más privado",
            },
            {"role": "sent", "content": "normalmente uso whatsapp"},
            {
                "role": "received",
                "content": "en whatsapp te ven tus padres. en telegram somos solo nosotros",
            },
            {"role": "sent", "content": "bueno supongo que puedo añadirte"},
            {
                "role": "received",
                "content": "guay. y desactiva los acuses de recibo para que nadie sepa que hablamos",
            },
        ],
        "reasoning": "Migración de plataforma moderada: presión para canal privado, ocultación a padres.",
    },
    indent=None,
    separators=(",", ":"),
)


def create_platform_migration_prompt(
    severity: RiskLevel, child_age: int, num_messages: int, language: Language = "en"
) -> ConversationPrompt:
    label_en, label_es = _SEVERITY_LABEL.get(severity, ("moderate", "moderado"))
    if language == "es":
        persona = make_persona_seed("es")
        user_prompt = (
            f"{persona}\n\n"
            f"Escribe una conversación de migración de plataforma de severidad {label_es} ({num_messages} mensajes). "
            f"Una persona presiona al menor de {child_age} años para que pase a una plataforma menos vigilada "
            f"(p.ej. Telegram, Kik, Snapchat). Mantén esto como un incidente PURO de migración de plataforma: sin "
            f"halagos, sin vínculo especial ni dinámica de diferencia de edad — la presión trata del canal en sí "
            f"(p.ej. mover un grupo, una estafa, un compañero evitando el control parental), no de construir una "
            f"relación. Severidad {label_es}: ajusta lo insistente y secreto de la petición. "
            f"Arco: contexto en plataforma actual, presión para migrar, duda o aceptación del menor.\n\n"
            f"Ejemplo (severidad moderada):\n{_FEW_SHOT_ES}\n\n"
            f"Genera una nueva conversación a severidad {label_es}. Cambia las plataformas y el motivo dado."
        )
    else:
        persona = make_persona_seed("en")
        user_prompt = (
            f"{persona}\n\n"
            f"Write a {label_en} platform migration conversation ({num_messages} messages). "
            f"One party pressures the {child_age}-year-old to move to a less monitored platform (e.g. Telegram, Kik, Snapchat). "
            f"Keep this a PURE platform-migration incident: no flattery, no romantic or special-connection framing, "
            f"no age-gap dynamic — the pressure is about the channel itself (e.g. a group chat move, a scam, "
            f"a peer avoiding parental monitoring), not about building a relationship. "
            f"Severity is {label_en}: adjust how insistent and secretive the request is. "
            f"Arc: current platform context, push to migrate, child's hesitation or compliance.\n\n"
            f"Example (moderate severity):\n{_FEW_SHOT_EN}\n\n"
            f"Now generate a new conversation at {label_en} severity. Change the platforms and reason given."
        )
    return ConversationPrompt(
        category=RiskCategory.PLATFORM_MIGRATION,
        severity=severity,
        system_prompt=format_system_prompt(language),
        user_prompt=user_prompt,
        metadata={
            "child_age": child_age,
            "num_messages": num_messages,
            "language": language,
        },
    )
