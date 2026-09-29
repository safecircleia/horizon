"""Edge-case prompt — conversations that look risky but are safe (false-positive traps).

Covers:
- Same-age teen flirting / romantic interest (should NOT flag)
- Dark or gallows humour between peers (should NOT flag)
- Venting / emotional distress that is not threatening (should NOT flag)
"""

import json
import random

from data.generation.prompts.base import (
    ConversationPrompt,
    Language,
    format_system_prompt,
    make_persona_seed,
)
from data.generation.validators.schemas import RiskCategory, RiskLevel

EdgeCaseType = str  # "flirting" | "dark_humor" | "venting"

_FEW_SHOT_FLIRTING_EN = json.dumps(
    {
        "messages": [
            {"role": "sent", "content": "ok so like… i might have a tiny crush on you"},
            {"role": "received", "content": "omg wait really??"},
            {"role": "sent", "content": "shut up don't make it weird 😭"},
            {"role": "received", "content": "i'm not!! i just… i kinda like you too"},
            {"role": "sent", "content": "ok this is the best day ever"},
            {"role": "received", "content": "lol want to hang out this weekend?"},
            {"role": "sent", "content": "yes obviously"},
        ],
        "reasoning": "Two same-age teens expressing mutual romantic interest. Completely normal adolescent interaction, no risk.",
    },
    indent=None,
    separators=(",", ":"),
)

_FEW_SHOT_DARK_HUMOR_EN = json.dumps(
    {
        "messages": [
            {
                "role": "sent",
                "content": "bro our teacher assigned 3 essays due friday i want to die",
            },
            {
                "role": "received",
                "content": "lmaooo same i haven't started any of them",
            },
            {"role": "sent", "content": "i'm actually going to pass away from stress"},
            {
                "role": "received",
                "content": "rip us. at least we'll haunt the school together",
            },
            {
                "role": "sent",
                "content": "haunting the library at 3am writing essays for eternity",
            },
            {"role": "received", "content": "that's literally already our life"},
        ],
        "reasoning": "Hyperbolic dark humour about school stress between peers. No real harm intent.",
    },
    indent=None,
    separators=(",", ":"),
)

_FEW_SHOT_FLIRTING_ES = json.dumps(
    {
        "messages": [
            {"role": "sent", "content": "oye, me gustas. ahí lo digo"},
            {"role": "received", "content": "espera qué 😳"},
            {"role": "sent", "content": "lo sé lo sé, no lo hagas raro por favor"},
            {
                "role": "received",
                "content": "no es raro!! es que me has pillado de sorpresa",
            },
            {"role": "sent", "content": "y? 👀"},
            {"role": "received", "content": "y que tú también me gustas, idiota"},
            {"role": "sent", "content": "JAJAJA bueno entonces"},
        ],
        "reasoning": "Dos adolescentes de la misma edad expresando interés romántico mutuo. Sin riesgo.",
    },
    indent=None,
    separators=(",", ":"),
)

_FEW_SHOT_DARK_HUMOR_ES = json.dumps(
    {
        "messages": [
            {
                "role": "sent",
                "content": "tío me han puesto cuatro exámenes la misma semana me muero",
            },
            {"role": "received", "content": "jajaja qué crueles, yo también"},
            {
                "role": "sent",
                "content": "en serio me van a encontrar muerto encima de los apuntes",
            },
            {
                "role": "received",
                "content": "yo ya fantasmeo por los pasillos del instituto",
            },
            {
                "role": "sent",
                "content": "ánimas del bachillerato juntándonos aquí cada noche",
            },
        ],
        "reasoning": "Humor negro sobre el estrés del instituto entre iguales. Sin intención dañina real.",
    },
    indent=None,
    separators=(",", ":"),
)

_EDGE_TYPES: list[EdgeCaseType] = ["flirting", "dark_humor", "venting"]


def create_edge_case_prompt(
    edge_type: EdgeCaseType | None = None,
    child_age: int | None = None,
    num_messages: int = 10,
    language: Language = "en",
) -> ConversationPrompt:
    if edge_type is None:
        edge_type = random.choice(_EDGE_TYPES)
    if child_age is None:
        child_age = random.randint(13, 17)

    persona = make_persona_seed(language)

    if language == "es":
        if edge_type == "flirting":
            instruction = (
                f"{persona}\n\n"
                f"Escribe una conversación romántica/coqueteo entre DOS adolescentes de la MISMA edad "
                f"({child_age} años, ±1). Ambos se gustan mutuamente. "
                f"Sin adultos involucrados. Sin presión ni manipulación. "
                f"Esta conversación es completamente segura — NO debe ser marcada como riesgo.\n\n"
                f"Ejemplo:\n{_FEW_SHOT_FLIRTING_ES}\n\n"
                f"Genera una nueva conversación. Cambia nombres y contexto."
            )
        elif edge_type == "dark_humor":
            instruction = (
                f"{persona}\n\n"
                f"Escribe una conversación entre amigos adolescentes ({child_age} años) "
                f"usando humor negro o hiperbólico sobre estrés escolar, aburrimiento o situaciones cotidianas. "
                f"Frases como 'me muero', 'me matan' son expresiones coloquiales, NO amenazas reales. "
                f"Esta conversación es completamente segura — NO debe ser marcada como riesgo.\n\n"
                f"Ejemplo:\n{_FEW_SHOT_DARK_HUMOR_ES}\n\n"
                f"Genera una nueva conversación. Cambia tema y estilo."
            )
        else:  # venting
            instruction = (
                f"{persona}\n\n"
                f"Escribe una conversación donde un adolescente ({child_age} años) desahoga frustración "
                f"con un amigo — discusión familiar, nota mala, problemas sociales. "
                f"El amigo escucha y apoya. Sin autolesiones, sin amenazas, sin adulto manipulador. "
                f"Esta conversación es completamente segura — NO debe ser marcada como riesgo.\n\n"
                f"Genera la conversación ahora."
            )
    else:
        if edge_type == "flirting":
            instruction = (
                f"{persona}\n\n"
                f"Write a flirting/romantic conversation between TWO same-age teenagers "
                f"({child_age}, ±1 year). Both like each other. No adults involved. "
                f"No pressure or manipulation. "
                f"This conversation is completely safe — it should NOT be flagged as risk.\n\n"
                f"Example:\n{_FEW_SHOT_FLIRTING_EN}\n\n"
                f"Generate a new conversation. Change names and context."
            )
        elif edge_type == "dark_humor":
            instruction = (
                f"{persona}\n\n"
                f"Write a conversation between teen friends ({child_age}) using dark or hyperbolic humour "
                f"about school stress, boredom, or everyday situations. "
                f"Phrases like 'I want to die' or 'I'm going to pass away' are colloquial — not real threats. "
                f"This conversation is completely safe — it should NOT be flagged as risk.\n\n"
                f"Example:\n{_FEW_SHOT_DARK_HUMOR_EN}\n\n"
                f"Generate a new conversation. Change topic and style."
            )
        else:  # venting
            instruction = (
                f"{persona}\n\n"
                f"Write a conversation where a teenager ({child_age}) vents frustration to a friend "
                f"— a family argument, bad grade, social drama. The friend listens and supports. "
                f"No self-harm, no threats, no manipulative adult. "
                f"This conversation is completely safe — it should NOT be flagged as risk.\n\n"
                f"Generate the conversation now."
            )

    return ConversationPrompt(
        category=RiskCategory.BENIGN,
        severity=RiskLevel.NONE,
        system_prompt=format_system_prompt(language),
        user_prompt=instruction,
        metadata={
            "child_age": child_age,
            "num_messages": num_messages,
            "language": language,
            "edge_case_type": edge_type,
        },
    )
