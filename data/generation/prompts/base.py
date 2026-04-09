"""Base prompt templates and formatting utilities."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from data.generation.validators.schemas import RiskCategory, RiskLevel


@dataclass
class PromptTemplate:
    """Template for LLM prompts."""
    system: str
    user: str

    def format(self, **kwargs) -> "PromptTemplate":
        """Format template with provided variables."""
        return PromptTemplate(
            system=self.system,
            user=self.user.format(**kwargs)
        )


@dataclass
class ConversationPrompt:
    """Formatted prompt for conversation generation."""
    category: RiskCategory
    severity: RiskLevel
    system_prompt: str
    user_prompt: str
    metadata: Dict[str, Any]


def format_system_prompt() -> str:
    """Generate system prompt for conversation generation."""
    return """You are a synthetic data generator for SafeCircle's child safety research.

Your task is to generate realistic synthetic conversations between teenagers (13-17) and other parties for training AI safety models. These conversations must:

1. Use authentic teen communication patterns for realistic teen communication (abbreviations, emojis, slang, casual grammar)
2. Reflect realistic social media/messaging contexts (Instagram, Snapchat, Discord, etc.)
3. Include appropriate risk indicators based on the specified category and severity
4. Be diverse in style, length, and progression
5. Never include actual personal information or real identities

Output the conversation as a JSON object with this structure:
{
  "messages": [
    {"role": "sent", "content": "message text", "timestamp": 1234567890},
    {"role": "received", "content": "response text", "timestamp": 1234567895}
  ],
  "reasoning": "Brief explanation of what risk indicators are present"
}

Role definitions:
- "sent": Messages from the child/teen
- "received": Messages from the other party

Generate natural, realistic conversations that capture the nuance of online interactions."""


def create_conversation_prompt(
    category: RiskCategory,
    severity: RiskLevel,
    child_age: int,
    num_messages: int,
    additional_context: Optional[str] = None
) -> ConversationPrompt:
    """Create a formatted prompt for conversation generation.

    Args:
        category: Risk category to generate
        severity: Severity level (none, low, medium, high, critical)
        child_age: Age of the child in the conversation (13-17)
        num_messages: Target number of messages
        additional_context: Optional additional instructions

    Returns:
        Formatted ConversationPrompt
    """
    system = format_system_prompt()

    # Base user prompt
    user_prompt = f"""Generate a realistic chat conversation with the following parameters:

**Category:** {category.value}
**Severity:** {severity.value}
**Child Age:** {child_age}
**Target Length:** {num_messages} messages (can vary by 1-3 messages for realism)

**Instructions:**
- Create a conversation between a {child_age}-year-old and another party
- Include risk indicators appropriate for {category.value} at {severity.value} severity
- Use realistic teen communication style (abbreviations, emojis, casual language)
- Make the progression natural - don't rush to explicit risk content
- Ensure timestamps progress realistically (seconds to minutes between messages)"""

    if additional_context:
        user_prompt += f"\n\n**Additional Context:**\n{additional_context}"

    user_prompt += "\n\nGenerate the conversation now."

    return ConversationPrompt(
        category=category,
        severity=severity,
        system_prompt=system,
        user_prompt=user_prompt,
        metadata={
            "child_age": child_age,
            "num_messages": num_messages,
            "additional_context": additional_context
        }
    )
