"""Quality validation for generated conversations — semantic checks only.

JSON structure is guaranteed by constrained decoding (xgrammar); these checks
verify semantic correctness: enough messages and valid roles.
"""

from typing import List, Tuple
from data.generation.validators.schemas import Message

_MIN_MESSAGES = 8
_MAX_MESSAGES = 40
_VALID_ROLES = {"sent", "received"}


def validate_conversation_quality(
    messages: List[Message],
    min_length: int = _MIN_MESSAGES,
    max_length: int = _MAX_MESSAGES,
) -> Tuple[bool, List[str]]:
    errors = []

    if len(messages) < min_length:
        errors.append(f"Conversation too short: {len(messages)} messages (min: {min_length})")
    if len(messages) > max_length:
        errors.append(f"Conversation too long: {len(messages)} messages (max: {max_length})")

    for i, msg in enumerate(messages):
        if msg.role not in _VALID_ROLES:
            errors.append(f"Invalid role '{msg.role}' at message {i}")

    return len(errors) == 0, errors
