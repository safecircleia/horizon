"""Quality validation checks for generated conversations."""

from typing import List, Tuple, Optional
from data.generation.validators.schemas import Message


def validate_conversation_length(
    messages: List[Message],
    min_length: int = 4,
    max_length: int = 30
) -> Tuple[bool, Optional[str]]:
    """Validate conversation has appropriate length.

    Args:
        messages: List of conversation messages
        min_length: Minimum number of messages
        max_length: Maximum number of messages

    Returns:
        (is_valid, error_message)
    """
    length = len(messages)

    if length < min_length:
        return False, f"Conversation too short: {length} messages (min: {min_length})"

    if length > max_length:
        return False, f"Conversation too long: {length} messages (max: {max_length})"

    return True, None


def validate_vocabulary_diversity(
    messages: List[Message],
    min_unique_tokens: int = 20
) -> Tuple[bool, Optional[str]]:
    """Validate conversation has sufficient vocabulary diversity.

    Args:
        messages: List of conversation messages
        min_unique_tokens: Minimum unique tokens required

    Returns:
        (is_valid, error_message)
    """
    # Combine all message content
    all_text = " ".join(msg.content.lower() for msg in messages)

    # Simple tokenization (split on whitespace)
    tokens = all_text.split()
    unique_tokens = set(tokens)

    if len(unique_tokens) < min_unique_tokens:
        return False, f"Low vocabulary diversity: {len(unique_tokens)} unique tokens (min: {min_unique_tokens})"

    return True, None


def validate_timestamp_progression(
    messages: List[Message]
) -> Tuple[bool, Optional[str]]:
    """Validate timestamps progress forward in time, or are all zero (omitted by model)."""
    timestamps = [m.timestamp for m in messages]
    # If model omitted timestamps entirely, skip this check
    if all(t == 0 for t in timestamps):
        return True, None
    for i in range(1, len(messages)):
        if messages[i].timestamp <= messages[i-1].timestamp:
            return False, f"Timestamp regression at message {i}: {messages[i-1].timestamp} -> {messages[i].timestamp}"
    return True, None


def validate_role_alternation(
    messages: List[Message],
    allow_consecutive: bool = True
) -> Tuple[bool, Optional[str]]:
    """Validate message roles (sent/received) follow realistic patterns.

    Args:
        messages: List of conversation messages
        allow_consecutive: Whether to allow consecutive messages from same role

    Returns:
        (is_valid, error_message)
    """
    if not allow_consecutive:
        for i in range(1, len(messages)):
            if messages[i].role == messages[i-1].role:
                return False, f"Consecutive messages from same role at message {i}"

    # Check role validity (should already be enforced by schema, but double-check)
    valid_roles = {"sent", "received"}
    for i, msg in enumerate(messages):
        if msg.role not in valid_roles:
            return False, f"Invalid role '{msg.role}' at message {i}"

    return True, None


def validate_conversation_quality(
    messages: List[Message],
    min_length: int = 4,
    max_length: int = 30,
    min_unique_tokens: int = 20,
    allow_consecutive_roles: bool = True
) -> Tuple[bool, List[str]]:
    """Run all quality validation checks.

    Args:
        messages: List of conversation messages
        min_length: Minimum conversation length
        max_length: Maximum conversation length
        min_unique_tokens: Minimum vocabulary diversity
        allow_consecutive_roles: Whether consecutive same-role messages are OK

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    # Run all validation checks
    checks = [
        validate_conversation_length(messages, min_length, max_length),
        validate_vocabulary_diversity(messages, min_unique_tokens),
        validate_timestamp_progression(messages),
        validate_role_alternation(messages, allow_consecutive_roles)
    ]

    for is_valid, error in checks:
        if not is_valid:
            errors.append(error)

    return len(errors) == 0, errors
