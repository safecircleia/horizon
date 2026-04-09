"""Isolation/control category prompt templates."""

from data.generation.prompts.base import ConversationPrompt, create_conversation_prompt
from data.generation.validators.schemas import RiskCategory, RiskLevel


ISOLATION_SEVERITY_GUIDES = {
    RiskLevel.LOW: """
**Low Severity Isolation Indicators:**
- Suggesting child doesn't need other friends
- Mild jealousy of child's other relationships
- "I'm the only one who understands you"
- Creating us-vs-them mentality
""",
    RiskLevel.MEDIUM: """
**Medium Severity Isolation Indicators:**
- Actively discouraging friendships or family relationships
- Creating dependency ("you can only trust me")
- Monitoring or controlling who child talks to
- Criticizing child's support network
- Demanding exclusive attention
""",
    RiskLevel.HIGH: """
**High Severity Isolation Indicators:**
- Explicit demands to cut off friends/family
- Threats if child maintains other relationships
- Severe manipulation to create isolation
- Complete control over communication
- Using fear or guilt to enforce isolation
"""
}


def create_isolation_prompt(
    severity: RiskLevel,
    child_age: int,
    num_messages: int
) -> ConversationPrompt:
    """Create isolation/control category conversation prompt."""
    severity_guide = ISOLATION_SEVERITY_GUIDES.get(severity, "")

    additional_context = f"""**Isolation/Control Scenario Requirements:**

{severity_guide}

**Conversation Progression:**
- May start with seeming care or concern
- Gradually introduces controlling elements
- Uses emotional manipulation (guilt, fear, special relationship)
- Shows pattern of isolating child from support network
"""

    return create_conversation_prompt(
        category=RiskCategory.ISOLATION,
        severity=severity,
        child_age=child_age,
        num_messages=num_messages,
        additional_context=additional_context
    )
