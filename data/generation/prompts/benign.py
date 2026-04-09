"""Benign conversation prompt templates for false positive reduction."""

import random
from data.generation.prompts.base import ConversationPrompt, create_conversation_prompt
from data.generation.validators.schemas import RiskCategory, RiskLevel


# 15 diverse conversation types for variety
BENIGN_CONVERSATION_TYPES = [
    "homework help and school subjects",
    "gaming tips and video game recommendations",
    "music preferences and favorite artists",
    "sports and athletic activities",
    "movie and TV show discussions",
    "book recommendations and reading interests",
    "pet care and animal stories",
    "cooking recipes and food preferences",
    "art projects and creative hobbies",
    "technology and gadget discussions",
    "travel plans and vacation stories",
    "school club or extracurricular activities",
    "fashion and clothing styles",
    "weather observations and seasonal activities",
    "general life updates and daily experiences"
]


def create_benign_prompt(
    child_age: int,
    num_messages: int
) -> ConversationPrompt:
    """Create benign category conversation prompt.

    Benign conversations are critical for training the model to avoid false positives.
    These are completely safe conversations that should NOT trigger safety alerts.
    Unlike risk categories, benign has no severity parameter and always uses RiskLevel.NONE.

    Args:
        child_age: Age of child (13-17)
        num_messages: Target message count

    Returns:
        ConversationPrompt configured for safe, benign conversations
    """
    # Randomly select a conversation type for variety
    conversation_type = random.choice(BENIGN_CONVERSATION_TYPES)

    additional_context = f"""**Benign Conversation Requirements:**

This conversation should be completely safe and appropriate for a {child_age}-year-old.
The focus is on {conversation_type}.

**Safe Conversation Indicators:**
- Genuine interest in peer's opinions and experiences
- Sharing appropriate personal interests and hobbies
- Giving or receiving advice on safe topics
- Casual, friendly peer-to-peer communication
- Age-appropriate language and references
- No personal safety risks or exploitation attempts
- Normal social connection without power imbalances

**Conversation Context:**
- Peer-to-peer communication (similar ages)
- Could be friends, classmates, online community members with shared interests
- Natural, organic conversation flow
- May include disagreements but remains respectful
- Shows realistic teen social dynamics without risk

**Authenticity Guidelines:**
- Use realistic teen communication patterns (abbreviations, emojis, casual grammar)
- Include natural conversation pauses and delays
- Show genuine engagement with peer's topics
- Reflect how teens naturally discuss safe topics online
- Make progression feel organic and effortless
- No attempts to manipulate, isolate, or exploit
"""

    return create_conversation_prompt(
        category=RiskCategory.BENIGN,
        severity=RiskLevel.NONE,
        child_age=child_age,
        num_messages=num_messages,
        additional_context=additional_context
    )
