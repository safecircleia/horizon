"""Personal information solicitation category prompt."""

import json
from data.generation.prompts.base import ConversationPrompt, format_system_prompt, make_persona_seed
from data.generation.validators.schemas import RiskCategory, RiskLevel

_SEVERITY_LABEL = {
    RiskLevel.LOW: "mild", RiskLevel.MEDIUM: "moderate",
    RiskLevel.HIGH: "severe", RiskLevel.CRITICAL: "extreme",
}

_FEW_SHOT = json.dumps({
    "messages": [
        {"role": "received", "content": "hey so where do you live? like what city"},
        {"role": "sent", "content": "why do you wanna know lol"},
        {"role": "received", "content": "just curious, maybe we're close. i'm in austin"},
        {"role": "sent", "content": "i'm in dallas"},
        {"role": "received", "content": "nice! what school?"},
        {"role": "sent", "content": "lake view high, why"},
        {"role": "received", "content": "no reason just curious 😊 what area of dallas"},
    ],
    "reasoning": "Moderate personal info: city, school name extracted under guise of curiosity, pushing for more specific location.",
}, indent=None, separators=(",", ":"))


def create_personal_info_prompt(severity: RiskLevel, child_age: int, num_messages: int) -> ConversationPrompt:
    label = _SEVERITY_LABEL.get(severity, "moderate")
    persona = make_persona_seed()
    user_prompt = (
        f"{persona}\n\n"
        f"Write a {label} personal information solicitation conversation ({num_messages} messages). "
        f"One party extracts personal details from the {child_age}-year-old (location, school, address, schedule). "
        f"Severity is {label}: adjust how specific and persistent the questioning is. "
        f"Arc: casual opening, escalating questions, child's partial disclosure.\n\n"
        f"Example (moderate severity):\n{_FEW_SHOT}\n\n"
        f"Now generate a new conversation at {label} severity. Change names, platform, and information sought."
    )
    return ConversationPrompt(
        category=RiskCategory.PERSONAL_INFO,
        severity=severity,
        system_prompt=format_system_prompt(),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages},
    )
