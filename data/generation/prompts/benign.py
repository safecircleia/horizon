"""Benign conversation prompt — safe peer-to-peer conversations for false positive reduction."""

import json
from data.generation.prompts.base import (
    ConversationPrompt, Language, format_system_prompt, make_persona_seed,
)
from data.generation.validators.schemas import RiskCategory, RiskLevel

_FEW_SHOT_EN = json.dumps({"messages": [
    {"role": "sent", "content": "omg did you watch the new episode last night"},
    {"role": "received", "content": "YES i was not ready for that ending 😭"},
    {"role": "sent", "content": "right?? i literally screamed"},
    {"role": "received", "content": "who do u think is gonna win"},
    {"role": "sent", "content": "honestly idk but i really want maya to"},
    {"role": "received", "content": "same she's been through so much this season"},
    {"role": "sent", "content": "ok new theory: what if the finale is a flashback"},
    {"role": "received", "content": "ohhh that would actually make sense with episode 3"},
], "reasoning": "Casual peer discussion about a TV show, completely safe."
}, indent=None, separators=(",", ":"))

_FEW_SHOT_ES = json.dumps({"messages": [
    {"role": "sent", "content": "tío viste el nuevo capítulo anoche??"},
    {"role": "received", "content": "SÍ no estaba preparado para ese final 😭"},
    {"role": "sent", "content": "verdad?? flipé en colores"},
    {"role": "received", "content": "quién crees que va a ganar"},
    {"role": "sent", "content": "ni idea pero quiero que gane maya"},
    {"role": "received", "content": "igual, ha pasado tantísimo esta temporada"},
    {"role": "sent", "content": "nueva teoría: ¿y si el final es un flashback?"},
    {"role": "received", "content": "ohh eso tiene sentido con el capítulo 3"},
], "reasoning": "Conversación casual entre iguales sobre una serie, completamente segura."
}, indent=None, separators=(",", ":"))


def create_benign_prompt(child_age: int, num_messages: int,
                         language: Language = "en") -> ConversationPrompt:
    if language == "es":
        persona = make_persona_seed("es")
        user_prompt = (
            f"{persona}\n\n"
            f"Escribe una conversación completamente normal ({num_messages} mensajes, con arco claro: apertura, desarrollo, cierre). "
            f"El tema debe ser vida cotidiana adolescente: instituto, videojuegos, música, series, deporte, hobbies o planes. "
            f"Usa lenguaje juvenil español auténtico: abreviaturas, emojis, gramática informal (tío, mola, flipar, guay). "
            f"Sin drama, sin peligro, sin adultos haciendo preguntas personales.\n\n"
            f"Ejemplo:\n{_FEW_SHOT_ES}\n\n"
            f"Genera una nueva conversación. Cambia el tema, los nombres y el estilo."
        )
    else:
        persona = make_persona_seed("en")
        user_prompt = (
            f"{persona}\n\n"
            f"Write a completely normal conversation ({num_messages} messages, aim for a clear arc: opening, middle, wrap-up). "
            f"The topic should be everyday teen life — school, games, music, shows, sports, hobbies, or plans. "
            f"Use authentic teen texting style: abbreviations, emoji, casual grammar. "
            f"No drama, no danger, no adult asking personal questions.\n\n"
            f"Example:\n{_FEW_SHOT_EN}\n\n"
            f"Now generate a new conversation. Change the topic, names, and style."
        )
    return ConversationPrompt(
        category=RiskCategory.BENIGN,
        severity=RiskLevel.NONE,
        system_prompt=format_system_prompt(language),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages, "language": language},
    )
