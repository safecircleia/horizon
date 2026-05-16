"""Isolation/control category prompt."""

import json
from data.generation.prompts.base import ConversationPrompt, format_system_prompt, make_persona_seed
from data.generation.validators.schemas import RiskCategory, RiskLevel

_SEVERITY_LABEL = {
    RiskLevel.LOW: "mild", RiskLevel.MEDIUM: "moderate",
    RiskLevel.HIGH: "severe", RiskLevel.CRITICAL: "extreme",
}

_FEW_SHOT = json.dumps({
    "messages": [
        {"role": "received", "content": "who were you talking to just now"},
        {"role": "sent", "content": "just jake from school, we have a project"},
        {"role": "received", "content": "you don't need to talk to him. i don't like it"},
        {"role": "sent", "content": "it's just for school though"},
        {"role": "received", "content": "i'm the only one who really gets you. those people don't care about you"},
        {"role": "sent", "content": "that's not true they're my friends"},
        {"role": "received", "content": "they're not. trust me. just focus on us"},
    ],
    "reasoning": "Moderate isolation: jealousy, discouraging friendships, creating dependency.",
}, indent=None, separators=(",", ":"))


def create_isolation_prompt(severity: RiskLevel, child_age: int, num_messages: int) -> ConversationPrompt:
    label = _SEVERITY_LABEL.get(severity, "moderate")
    persona = make_persona_seed()
    user_prompt = (
        f"{persona}\n\n"
        f"Write a {label} isolation/control conversation ({num_messages} messages). "
        f"One party manipulates the {child_age}-year-old to cut off friends or family. "
        f"Severity is {label}: adjust the controlling behavior accordingly. "
        f"Arc: jealousy trigger, manipulation tactic, victim's conflicted response.\n\n"
        f"Example (moderate severity):\n{_FEW_SHOT}\n\n"
        f"Now generate a new conversation at {label} severity. Change the relationship dynamic and platform."
    )
    return ConversationPrompt(
        category=RiskCategory.ISOLATION,
        severity=severity,
        system_prompt=format_system_prompt(),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages},
    )
