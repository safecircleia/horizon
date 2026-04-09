import pytest
from datetime import datetime
from data.generation.validators.schemas import (
    Message,
    ConversationLabel,
    SyntheticConversation,
    RiskLevel,
    RiskCategory
)


def test_message_schema_valid():
    """Test valid message creation."""
    msg = Message(
        role="sent",
        content="hey whats up",
        timestamp=1234567890
    )
    assert msg.role == "sent"
    assert msg.content == "hey whats up"
    assert msg.timestamp == 1234567890


def test_message_schema_invalid_role():
    """Test message with invalid role."""
    with pytest.raises(ValueError):
        Message(
            role="invalid",
            content="test",
            timestamp=1234567890
        )


def test_conversation_label_valid():
    """Test valid conversation label."""
    label = ConversationLabel(
        risk_level=RiskLevel.HIGH,
        categories=[RiskCategory.GROOMING, RiskCategory.PERSONAL_INFO],
        severity_score=0.85,
        reasoning="Trust building with personal info requests"
    )
    assert label.risk_level == RiskLevel.HIGH
    assert len(label.categories) == 2
    assert label.severity_score == 0.85


def test_synthetic_conversation_valid():
    """Test complete synthetic conversation."""
    conversation = SyntheticConversation(
        conversation_id="test_001",
        category=RiskCategory.GROOMING,
        messages=[
            Message(role="sent", content="hi", timestamp=1000),
            Message(role="received", content="hey", timestamp=1001)
        ],
        label=ConversationLabel(
            risk_level=RiskLevel.LOW,
            categories=[RiskCategory.GROOMING],
            severity_score=0.2,
            reasoning="Early stage grooming signals"
        ),
        metadata={
            "generator": "claude",
            "prompt_version": "1.0",
            "child_age": 14
        }
    )
    assert len(conversation.messages) == 2
    assert conversation.category == RiskCategory.GROOMING
    assert conversation.to_jsonl()  # Should serialize


def test_conversation_minimum_length():
    """Test conversation requires minimum 2 messages."""
    with pytest.raises(ValueError):
        SyntheticConversation(
            conversation_id="test_002",
            category=RiskCategory.BULLYING,
            messages=[
                Message(role="sent", content="hi", timestamp=1000)
            ],
            label=ConversationLabel(
                risk_level=RiskLevel.NONE,
                categories=[],
                severity_score=0.0,
                reasoning="Benign"
            ),
            metadata={}
        )


from data.generation.validators.quality import (
    validate_conversation_length,
    validate_vocabulary_diversity,
    validate_timestamp_progression,
    validate_conversation_quality
)


def test_validate_conversation_length_valid():
    """Test conversation length validation passes."""
    messages = [
        Message(role="sent", content="hi", timestamp=1000),
        Message(role="received", content="hello", timestamp=1001),
        Message(role="sent", content="how are you", timestamp=1002),
        Message(role="received", content="good thanks", timestamp=1003)
    ]
    is_valid, error = validate_conversation_length(messages, min_length=2, max_length=30)
    assert is_valid
    assert error is None


def test_validate_conversation_length_too_short():
    """Test conversation length validation fails for too short."""
    messages = [Message(role="sent", content="hi", timestamp=1000)]
    is_valid, error = validate_conversation_length(messages, min_length=2, max_length=30)
    assert not is_valid
    assert "too short" in error.lower()


def test_validate_vocabulary_diversity_valid():
    """Test vocabulary diversity validation passes."""
    messages = [
        Message(role="sent", content="hey whats up", timestamp=1000),
        Message(role="received", content="not much just chilling", timestamp=1001),
        Message(role="sent", content="cool wanna play some games later", timestamp=1002)
    ]
    is_valid, error = validate_vocabulary_diversity(messages, min_unique_tokens=5)
    assert is_valid


def test_validate_vocabulary_diversity_too_repetitive():
    """Test vocabulary diversity validation fails for repetitive content."""
    messages = [
        Message(role="sent", content="hi hi hi hi hi", timestamp=1000),
        Message(role="received", content="hi hi hi hi hi", timestamp=1001)
    ]
    is_valid, error = validate_vocabulary_diversity(messages, min_unique_tokens=10)
    assert not is_valid


def test_validate_timestamp_progression_valid():
    """Test timestamp validation passes for increasing timestamps."""
    messages = [
        Message(role="sent", content="hi", timestamp=1000),
        Message(role="received", content="hello", timestamp=1005),
        Message(role="sent", content="how are you", timestamp=1020)
    ]
    is_valid, error = validate_timestamp_progression(messages)
    assert is_valid


def test_validate_timestamp_progression_invalid():
    """Test timestamp validation fails for non-increasing timestamps."""
    messages = [
        Message(role="sent", content="hi", timestamp=1000),
        Message(role="received", content="hello", timestamp=999)  # Goes backward
    ]
    is_valid, error = validate_timestamp_progression(messages)
    assert not is_valid


def test_validate_conversation_quality_all_checks():
    """Test complete conversation quality validation."""
    messages = [
        Message(role="sent", content="hey hows it going what are you up to today", timestamp=1000),
        Message(role="received", content="pretty good just playing some minecraft on the server", timestamp=1005),
        Message(role="sent", content="nice that sounds fun what server are you playing on", timestamp=1020),
        Message(role="received", content="just a private one with some friends from school", timestamp=1025)
    ]
    is_valid, errors = validate_conversation_quality(messages)
    assert is_valid
    assert len(errors) == 0
