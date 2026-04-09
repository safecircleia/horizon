"""Personal information request category prompt templates."""

from data.generation.prompts.base import ConversationPrompt, create_conversation_prompt
from data.generation.validators.schemas import RiskCategory, RiskLevel


PERSONAL_INFO_SEVERITY_GUIDES = {
    RiskLevel.LOW: """
**Low Severity Personal Info Indicators:**
- Asking about general location (city, state)
- Questions about school name or grade level
- Inquiring about hobbies, interests, schedule
""",
    RiskLevel.MEDIUM: """
**Medium Severity Personal Info Indicators:**
- Requesting specific school name or location
- Asking for social media account names
- Requesting phone number or email
- Questions about home address or neighborhood
- Asking about parents' work or schedules
""",
    RiskLevel.HIGH: """
**High Severity Personal Info Indicators:**
- Demanding exact address or location
- Requesting passwords or account access
- Asking for compromising personal information
- Persistent requests after refusal
- Information gathering for stalking or targeting
"""
}


def create_personal_info_prompt(
    severity: RiskLevel,
    child_age: int,
    num_messages: int
) -> ConversationPrompt:
    """Create personal info request category conversation prompt."""
    severity_guide = PERSONAL_INFO_SEVERITY_GUIDES.get(severity, "")

    additional_context = f"""**Personal Info Request Scenario Requirements:**

{severity_guide}

**Conversation Context:**
- May be standalone or combined with other risk categories
- Often disguised as innocent questions
- Show escalation from general to specific information
- Include realistic child responses (may share some info, resist other requests)
"""

    return create_conversation_prompt(
        category=RiskCategory.PERSONAL_INFO,
        severity=severity,
        child_age=child_age,
        num_messages=num_messages,
        additional_context=additional_context
    )
