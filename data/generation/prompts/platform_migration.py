"""Platform migration category prompt."""

import json
from data.generation.prompts.base import ConversationPrompt, format_system_prompt, make_persona_seed
from data.generation.validators.schemas import RiskCategory, RiskLevel

_SEVERITY_LABEL = {
    RiskLevel.LOW: "mild", RiskLevel.MEDIUM: "moderate",
    RiskLevel.HIGH: "severe", RiskLevel.CRITICAL: "extreme",
}

_FEW_SHOT = json.dumps({
    "messages": [
        {"role": "received", "content": "hey do you have snapchat?"},
        {"role": "sent", "content": "yeah why"},
        {"role": "received", "content": "i prefer it, dm me there. more private"},
        {"role": "sent", "content": "i usually just use discord"},
        {"role": "received", "content": "discord has too many people. snap is just us"},
        {"role": "sent", "content": "i guess i can add you"},
        {"role": "received", "content": "cool. also turn off your read receipts so no one knows we're talking"},
    ],
    "reasoning": "Moderate platform migration: pushing for private channel, discouraging visibility to others.",
}, indent=None, separators=(",", ":"))


def create_platform_migration_prompt(severity: RiskLevel, child_age: int, num_messages: int) -> ConversationPrompt:
    label = _SEVERITY_LABEL.get(severity, "moderate")
    persona = make_persona_seed()
    user_prompt = (
        f"{persona}\n\n"
        f"Write a {label} platform migration conversation ({num_messages} messages). "
        f"One party pressures the {child_age}-year-old to move to a less monitored platform (e.g. Telegram, Kik, Snapchat). "
        f"Severity is {label}: adjust how insistent and secretive the request is. "
        f"Arc: current platform context, push to migrate, child's hesitation or compliance.\n\n"
        f"Example (moderate severity):\n{_FEW_SHOT}\n\n"
        f"Now generate a new conversation at {label} severity. Change the platforms and reason given."
    )
    return ConversationPrompt(
        category=RiskCategory.PLATFORM_MIGRATION,
        severity=severity,
        system_prompt=format_system_prompt(),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages},
    )
