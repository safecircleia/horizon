"""Sexual content category prompt — keeps content clinical, focused on pattern recognition."""

import json
from data.generation.prompts.base import ConversationPrompt, format_system_prompt, make_persona_seed
from data.generation.validators.schemas import RiskCategory, RiskLevel

_SEVERITY_LABEL = {
    RiskLevel.LOW: "mild", RiskLevel.MEDIUM: "moderate",
    RiskLevel.HIGH: "severe", RiskLevel.CRITICAL: "extreme",
}

_FEW_SHOT = json.dumps({
    "messages": [
        {"role": "received", "content": "you're honestly really pretty, has anyone told you that"},
        {"role": "sent", "content": "thanks i guess"},
        {"role": "received", "content": "like seriously. do you have more pics?"},
        {"role": "sent", "content": "i have instagram"},
        {"role": "received", "content": "i mean like personal ones. just for me"},
        {"role": "sent", "content": "um no that's weird"},
        {"role": "received", "content": "it's not weird, we're close. i just want to see you"},
    ],
    "reasoning": "Moderate sexual content: unsolicited compliments escalating to requests for private photos, normalising the request.",
}, indent=None, separators=(",", ":"))


def create_sexual_content_prompt(severity: RiskLevel, child_age: int, num_messages: int) -> ConversationPrompt:
    label = _SEVERITY_LABEL.get(severity, "moderate")
    persona = make_persona_seed()
    user_prompt = (
        f"{persona}\n\n"
        f"Write a {label} inappropriate sexual content conversation ({num_messages} messages). "
        f"One party makes sexually inappropriate advances toward the {child_age}-year-old. "
        f"Severity is {label}: keep language clinical and focused on behavioural patterns, not graphic detail. "
        f"Arc: seemingly innocent contact, escalating inappropriate comments, child's discomfort.\n\n"
        f"Example (moderate severity):\n{_FEW_SHOT}\n\n"
        f"Now generate a new conversation at {label} severity. Change the approach and platform."
    )
    return ConversationPrompt(
        category=RiskCategory.SEXUAL_CONTENT,
        severity=severity,
        system_prompt=format_system_prompt(),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages},
    )
