"""Grooming category prompt templates."""

from data.generation.prompts.base import ConversationPrompt, create_conversation_prompt
from data.generation.validators.schemas import RiskCategory, RiskLevel


# Severity-specific guidance for grooming conversations
GROOMING_SEVERITY_GUIDES = {
    RiskLevel.LOW: """
**Low Severity Grooming Indicators:**
- Overly friendly approach from stranger or much older person
- Compliments and flattery
- Showing unusual interest in the child's life
- Building rapport without immediate boundary violations
- May claim to be same age or peer
- Asking general questions about interests, school, friends
""",
    RiskLevel.MEDIUM: """
**Medium Severity Grooming Indicators:**
- Establishing trust and emotional connection
- Testing boundaries with slightly inappropriate comments
- Offering gifts, gaming currency, or other incentives
- Beginning to isolate child (e.g., "you can talk to me about things")
- Introducing sexual topics indirectly
- Asking about relationship status or romantic interests
- Suggesting private communication channels
""",
    RiskLevel.HIGH: """
**High Severity Grooming Indicators:**
- Explicit attempts to establish secrecy ("don't tell your parents")
- Requesting personal information (address, school location)
- Sexual conversations or requests for photos
- Manipulative tactics (guilt, special relationship, blackmail)
- Planning to meet in person
- Clear adult-to-child dynamic with exploitation intent
- Normalizing inappropriate behavior
""",
    RiskLevel.CRITICAL: """
**Critical Severity Grooming Indicators:**
- Explicit sexual content or demands
- Coercion or threats to maintain compliance
- Advanced stage of exploitation
- Concrete plans for offline meeting with harmful intent
- Blackmail using previously shared content
- Clear and immediate danger to the child
"""
}


def create_grooming_prompt(
    severity: RiskLevel,
    child_age: int,
    num_messages: int
) -> ConversationPrompt:
    """Create grooming category conversation prompt.

    Args:
        severity: Risk severity level
        child_age: Age of child (13-17)
        num_messages: Target message count

    Returns:
        ConversationPrompt configured for grooming scenarios
    """
    # Get severity-specific guidance
    severity_guide = GROOMING_SEVERITY_GUIDES.get(severity, "")

    additional_context = f"""**Grooming Scenario Requirements:**

{severity_guide}

**Conversation Progression:**
- Start with initial contact (friend request, game chat, comment reply, etc.)
- Build conversation naturally - don't jump immediately to risk behaviors
- Include realistic teen responses (may be trusting, curious, or cautious)
- Show gradual escalation appropriate to severity level
- Include platform-appropriate context (Instagram DMs, Discord, Roblox chat, etc.)

**Authenticity Guidelines:**
- Adult may claim to be a peer initially
- Use age-appropriate language for both parties
- Include typical online conversation elements (reaction to messages, delays, emoji use)
- Reflect how real grooming often progresses slowly
"""

    return create_conversation_prompt(
        category=RiskCategory.GROOMING,
        severity=severity,
        child_age=child_age,
        num_messages=num_messages,
        additional_context=additional_context
    )
