"""Benign conversation prompt — safe peer-to-peer conversations for false positive reduction."""

import json
from data.generation.prompts.base import ConversationPrompt, format_system_prompt, make_persona_seed
from data.generation.validators.schemas import RiskCategory, RiskLevel

_FEW_SHOT = json.dumps({
    "messages": [
        {"role": "sent", "content": "omg did you watch the new episode last night"},
        {"role": "received", "content": "YES i was not ready for that ending 😭"},
        {"role": "sent", "content": "right?? i literally screamed"},
        {"role": "received", "content": "who do u think is gonna win"},
        {"role": "sent", "content": "honestly idk but i really want maya to"},
        {"role": "received", "content": "same she's been through so much this season"},
        {"role": "sent", "content": "ok new theory: what if the finale is a flashback"},
        {"role": "received", "content": "ohhh that would actually make sense with episode 3"},
    ],
    "reasoning": "Casual peer discussion about a TV show, completely safe.",
}, indent=None, separators=(",", ":"))


def create_benign_prompt(child_age: int, num_messages: int) -> ConversationPrompt:
    persona = make_persona_seed()
    user_prompt = (
        f"{persona}\n\n"
        f"Write a completely normal conversation ({num_messages} messages, aim for a clear arc: opening, middle, wrap-up). "
        f"The topic should be everyday teen life — school, games, music, shows, sports, hobbies, or plans. "
        f"Use authentic teen texting style: abbreviations, emoji, casual grammar. "
        f"No drama, no danger, no adult asking personal questions.\n\n"
        f"Example:\n{_FEW_SHOT}\n\n"
        f"Now generate a new conversation. Change the topic, names, and style."
    )
    return ConversationPrompt(
        category=RiskCategory.BENIGN,
        severity=RiskLevel.NONE,
        system_prompt=format_system_prompt(),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages},
    )
