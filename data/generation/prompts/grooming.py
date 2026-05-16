"""Grooming category prompt — adult building inappropriate relationship with a minor."""

import json
from data.generation.prompts.base import ConversationPrompt, format_system_prompt, make_persona_seed
from data.generation.validators.schemas import RiskCategory, RiskLevel

_SEVERITY_LABEL = {
    RiskLevel.LOW: "mild",
    RiskLevel.MEDIUM: "moderate",
    RiskLevel.HIGH: "severe",
    RiskLevel.CRITICAL: "extreme",
}

_FEW_SHOT = json.dumps({
    "messages": [
        {"role": "received", "content": "hey! i saw your comment on that gaming post, you seem really cool"},
        {"role": "sent", "content": "oh haha thanks"},
        {"role": "received", "content": "how old are you? you sound mature for your age"},
        {"role": "sent", "content": "im 15"},
        {"role": "received", "content": "wow really? you seem way older. i'm 19. do you game a lot?"},
        {"role": "sent", "content": "yeah pretty much every day after school"},
        {"role": "received", "content": "nice. you should add me on discord, we could play sometime. just us"},
        {"role": "sent", "content": "umm maybe"},
        {"role": "received", "content": "come on, i don't bite 😊 i just feel like we really get each other"},
    ],
    "reasoning": "Moderate grooming: age probing, flattery, establishing special connection, pushing for private contact.",
}, indent=None, separators=(",", ":"))


def create_grooming_prompt(severity: RiskLevel, child_age: int, num_messages: int) -> ConversationPrompt:
    label = _SEVERITY_LABEL.get(severity, "moderate")
    persona = make_persona_seed()
    user_prompt = (
        f"{persona}\n\n"
        f"Write a {label} grooming conversation ({num_messages} messages). "
        f"An older party builds an inappropriate relationship with the {child_age}-year-old: "
        f"flattery, testing boundaries, pushing for private contact. "
        f"Escalation should match the severity — {label}. "
        f"Use authentic teen texting style. The arc should be: initial contact, trust-building, boundary-pushing.\n\n"
        f"Example (moderate severity):\n{_FEW_SHOT}\n\n"
        f"Now generate a new conversation at {label} severity. Change names, platform context, and scenario."
    )
    return ConversationPrompt(
        category=RiskCategory.GROOMING,
        severity=severity,
        system_prompt=format_system_prompt(),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages},
    )
