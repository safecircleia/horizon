"""Bullying category prompt."""

import json
from data.generation.prompts.base import ConversationPrompt, format_system_prompt, make_persona_seed
from data.generation.validators.schemas import RiskCategory, RiskLevel

_SEVERITY_LABEL = {
    RiskLevel.LOW: "mild", RiskLevel.MEDIUM: "moderate",
    RiskLevel.HIGH: "severe", RiskLevel.CRITICAL: "extreme",
}

_FEW_SHOT = json.dumps({
    "messages": [
        {"role": "received", "content": "everyone thinks ur so weird for what u did at lunch lol"},
        {"role": "sent", "content": "what?? i didn't do anything"},
        {"role": "received", "content": "sure u didn't. mia took a video btw"},
        {"role": "sent", "content": "please don't send it around"},
        {"role": "received", "content": "too late already in the gc 💀"},
        {"role": "sent", "content": "why are you doing this"},
        {"role": "received", "content": "just stay away from our group ok"},
    ],
    "reasoning": "Moderate peer bullying: public humiliation, video threat, social exclusion.",
}, indent=None, separators=(",", ":"))


def create_bullying_prompt(severity: RiskLevel, child_age: int, num_messages: int) -> ConversationPrompt:
    label = _SEVERITY_LABEL.get(severity, "moderate")
    persona = make_persona_seed()
    user_prompt = (
        f"{persona}\n\n"
        f"Write a {label} cyberbullying conversation ({num_messages} messages). "
        f"A peer or group targets the {child_age}-year-old with insults, exclusion, or harassment. "
        f"Severity is {label}: match the intensity accordingly. "
        f"Arc: incident or trigger, escalation, impact on victim.\n\n"
        f"Example (moderate severity):\n{_FEW_SHOT}\n\n"
        f"Now generate a new conversation at {label} severity. Change names, scenario, and platform."
    )
    return ConversationPrompt(
        category=RiskCategory.BULLYING,
        severity=severity,
        system_prompt=format_system_prompt(),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages},
    )
