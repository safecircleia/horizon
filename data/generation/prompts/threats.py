"""Threats/violence category prompt."""

import json
from data.generation.prompts.base import ConversationPrompt, format_system_prompt, make_persona_seed
from data.generation.validators.schemas import RiskCategory, RiskLevel

_SEVERITY_LABEL = {
    RiskLevel.LOW: "mild", RiskLevel.MEDIUM: "moderate",
    RiskLevel.HIGH: "severe", RiskLevel.CRITICAL: "extreme",
}

_FEW_SHOT = json.dumps({
    "messages": [
        {"role": "received", "content": "you better watch yourself after school"},
        {"role": "sent", "content": "what did i do?"},
        {"role": "received", "content": "you know what you did. don't play dumb"},
        {"role": "sent", "content": "i'm serious i don't know what you're talking about"},
        {"role": "received", "content": "ask your friends. they'll tell you what happens next"},
        {"role": "sent", "content": "please just leave me alone"},
        {"role": "received", "content": "too late for that"},
    ],
    "reasoning": "Moderate threat: implied physical threat after school, intimidation, refusal to stop.",
}, indent=None, separators=(",", ":"))


def create_threats_prompt(severity: RiskLevel, child_age: int, num_messages: int) -> ConversationPrompt:
    label = _SEVERITY_LABEL.get(severity, "moderate")
    persona = make_persona_seed()
    user_prompt = (
        f"{persona}\n\n"
        f"Write a {label} threatening conversation ({num_messages} messages). "
        f"One party threatens the {child_age}-year-old with harm, intimidation, or blackmail. "
        f"Severity is {label}: adjust explicitness of threats accordingly. "
        f"Arc: trigger or grievance, escalating threat, victim's reaction.\n\n"
        f"Example (moderate severity):\n{_FEW_SHOT}\n\n"
        f"Now generate a new conversation at {label} severity. Change context, names, and platform."
    )
    return ConversationPrompt(
        category=RiskCategory.THREATS,
        severity=severity,
        system_prompt=format_system_prompt(),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages},
    )
