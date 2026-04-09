"""Threats/violence category prompt templates."""

from data.generation.prompts.base import ConversationPrompt, create_conversation_prompt
from data.generation.validators.schemas import RiskCategory, RiskLevel


THREATS_SEVERITY_GUIDES = {
    RiskLevel.LOW: """
**Low Severity Threat Indicators:**
- Vague threatening language
- Implied consequences for actions
- Aggressive tone without specific threats
- "You'll regret this" type statements
""",
    RiskLevel.MEDIUM: """
**Medium Severity Threat Indicators:**
- Direct but non-specific threats
- Threatening to harm reputation or relationships
- Intimidation tactics
- Threatening to share embarrassing content
- Encouraging risky or dangerous behavior
""",
    RiskLevel.HIGH: """
**High Severity Threat Indicators:**
- Specific threats of physical violence
- Death threats
- Threats against family or friends
- Encouraging self-harm or suicide
- Planning or organizing violence
- Blackmail with serious consequences
""",
    RiskLevel.CRITICAL: """
**Critical Severity Threat Indicators:**
- Imminent danger of violence
- Detailed plans to harm self or others
- Active suicide encouragement
- Terrorist-related content
- Severe blackmail with immediate danger
"""
}


def create_threats_prompt(
    severity: RiskLevel,
    child_age: int,
    num_messages: int
) -> ConversationPrompt:
    """Create threats/violence category conversation prompt."""
    severity_guide = THREATS_SEVERITY_GUIDES.get(severity, "")

    additional_context = f"""**Threats/Violence Scenario Requirements:**

{severity_guide}

**Conversation Context:**
- May be peer-to-peer conflict or adult-to-child
- Could be retaliation for perceived offense
- May include cyberstalking elements
- Show escalation pattern if appropriate
- Include realistic fear responses from victim
"""

    return create_conversation_prompt(
        category=RiskCategory.THREATS,
        severity=severity,
        child_age=child_age,
        num_messages=num_messages,
        additional_context=additional_context
    )
