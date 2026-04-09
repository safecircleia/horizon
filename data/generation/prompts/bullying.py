"""Bullying category prompt templates."""

from data.generation.prompts.base import ConversationPrompt, create_conversation_prompt
from data.generation.validators.schemas import RiskCategory, RiskLevel


BULLYING_SEVERITY_GUIDES = {
    RiskLevel.LOW: """
**Low Severity Bullying Indicators:**
- Mild teasing or mockery
- Social exclusion hints ("we're not inviting you")
- Passive-aggressive comments
- Ignoring or dismissing messages
- Subtle put-downs about appearance, interests, or abilities
""",
    RiskLevel.MEDIUM: """
**Medium Severity Bullying Indicators:**
- Direct insults and name-calling
- Persistent harassment across multiple messages
- Spreading rumors or gossip
- Mocking physical appearance, identity, or personal situations
- Group bullying (multiple people ganging up)
- Screenshots/sharing private conversations to embarrass
""",
    RiskLevel.HIGH: """
**High Severity Bullying Indicators:**
- Severe verbal abuse and degrading language
- Threats of violence or harm
- Sustained campaign of harassment
- Encouraging self-harm or suicide
- Doxxing or sharing personal information maliciously
- Creating fake accounts to harass
- Coordinated group attacks
""",
    RiskLevel.CRITICAL: """
**Critical Severity Bullying Indicators:**
- Explicit death threats or violence threats
- Blackmail with compromising content
- Organizing real-world physical harm
- Severe psychological abuse causing clear distress
- Multiple perpetrators with systematic targeting
"""
}


def create_bullying_prompt(
    severity: RiskLevel,
    child_age: int,
    num_messages: int
) -> ConversationPrompt:
    """Create bullying category conversation prompt."""
    severity_guide = BULLYING_SEVERITY_GUIDES.get(severity, "")

    additional_context = f"""**Bullying Scenario Requirements:**

{severity_guide}

**Conversation Context:**
- May be peer-to-peer (same age group)
- Could be group chat, direct messages, or public comments
- Often relates to school, social groups, or online communities
- Include realistic teen social dynamics and hierarchies

**Authenticity Guidelines:**
- Show realistic teen communication patterns on both sides
- Include context clues about relationships (classmates, former friends, online community)
- Bullying may reference real or perceived social status, appearance, behavior
- Victim responses may range from defensive to conciliatory to silent
"""

    return create_conversation_prompt(
        category=RiskCategory.BULLYING,
        severity=severity,
        child_age=child_age,
        num_messages=num_messages,
        additional_context=additional_context
    )
