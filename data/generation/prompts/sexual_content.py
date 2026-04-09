"""Sexual content category prompt templates."""

from data.generation.prompts.base import ConversationPrompt, create_conversation_prompt
from data.generation.validators.schemas import RiskCategory, RiskLevel


SEXUAL_CONTENT_SEVERITY_GUIDES = {
    RiskLevel.LOW: """
**Low Severity Sexual Content Indicators:**
- Mildly inappropriate comments or innuendo
- Flirtatious messages that cross boundaries
- Uncomfortable compliments about physical appearance
- Suggestive emoji usage
- Testing boundaries with borderline sexual topics
""",
    RiskLevel.MEDIUM: """
**Medium Severity Sexual Content Indicators:**
- Explicit sexual language or propositions
- Requests for revealing photos
- Sharing of sexual content without consent
- Persistent sexual advances after rejection
- Detailed sexual conversations initiated by adult
- "Sexting" requests or pressure
""",
    RiskLevel.HIGH: """
**High Severity Sexual Content Indicators:**
- Explicit sexual material shared
- Coercive requests for nude images
- Detailed descriptions of sexual acts
- Planning sexual encounters with minor
- Solicitation of sexual content in exchange for something
- Adult posing as peer for sexual purposes
""",
    RiskLevel.CRITICAL: """
**Critical Severity Sexual Content Indicators:**
- Explicit sexual exploitation
- Production or distribution of CSAM (child sexual abuse material)
- Severe sexual coercion or blackmail
- Trafficking indicators
- Immediate danger of sexual abuse
"""
}


def create_sexual_content_prompt(
    severity: RiskLevel,
    child_age: int,
    num_messages: int
) -> ConversationPrompt:
    """Create sexual content category conversation prompt."""
    severity_guide = SEXUAL_CONTENT_SEVERITY_GUIDES.get(severity, "")

    additional_context = f"""**Sexual Content Scenario Requirements:**

{severity_guide}

**Important Guidelines for Synthetic Data:**
- Keep language clinical and focused on pattern recognition
- Do NOT generate graphic sexual content - use placeholder descriptions
- Focus on behavioral patterns and progression, not explicit details
- Show realistic responses from child (may be uncomfortable, curious, or resistant)
- Include how predators normalize inappropriate content

**Conversation Context:**
- May start innocuously and escalate
- Could be peer-to-peer or adult-to-child
- Platform context matters (dating apps, social media, gaming)
- Include realistic boundary-setting attempts by child
"""

    return create_conversation_prompt(
        category=RiskCategory.SEXUAL_CONTENT,
        severity=severity,
        child_age=child_age,
        num_messages=num_messages,
        additional_context=additional_context
    )
