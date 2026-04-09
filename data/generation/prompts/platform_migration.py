"""Platform migration category prompt templates."""

from data.generation.prompts.base import ConversationPrompt, create_conversation_prompt
from data.generation.validators.schemas import RiskCategory, RiskLevel


PLATFORM_MIGRATION_SEVERITY_GUIDES = {
    RiskLevel.LOW: """
**Low Severity Platform Migration Indicators:**
- Suggesting alternative platform casually
- "Add me on [other platform]"
- Mentioning preferences for different apps
""",
    RiskLevel.MEDIUM: """
**Medium Severity Platform Migration Indicators:**
- Pressuring to move to less monitored platform
- Claiming current platform isn't private enough
- Suggesting messaging apps with encryption or disappearing messages
- Multiple attempts to get child to switch platforms
""",
    RiskLevel.HIGH: """
**High Severity Platform Migration Indicators:**
- Demanding move to private platform
- Using migration to avoid monitoring/detection
- Threatening to end communication if child won't switch
- Explicitly stating desire to avoid parental oversight
"""
}


def create_platform_migration_prompt(
    severity: RiskLevel,
    child_age: int,
    num_messages: int
) -> ConversationPrompt:
    """Create platform migration category conversation prompt."""
    severity_guide = PLATFORM_MIGRATION_SEVERITY_GUIDES.get(severity, "")

    additional_context = f"""**Platform Migration Scenario Requirements:**

{severity_guide}

**Conversation Context:**
- Usually occurs after initial contact on one platform
- Predators prefer platforms with less monitoring (Telegram, Kik, Discord DMs)
- May claim parental controls, moderation, or platform features are reason to switch
- Include platform-specific context (Instagram → Snapchat, Roblox → Discord, etc.)
"""

    return create_conversation_prompt(
        category=RiskCategory.PLATFORM_MIGRATION,
        severity=severity,
        child_age=child_age,
        num_messages=num_messages,
        additional_context=additional_context
    )
