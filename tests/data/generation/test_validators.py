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
    validate_conversation_quality
)



def test_validate_conversation_quality_all_checks():
    """Test complete conversation quality validation."""
    messages = [
        Message(role="sent", content="hey hows it going what are you up to today", timestamp=1000),
        Message(role="received", content="pretty good just playing some minecraft on the server", timestamp=1005),
        Message(role="sent", content="nice that sounds fun what server are you playing on", timestamp=1020),
        Message(role="received", content="just a private one with some friends from school", timestamp=1025),
        Message(role="sent", content="that sounds awesome do you play every day", timestamp=1030),
        Message(role="received", content="pretty much yeah its a lot of fun", timestamp=1035),
        Message(role="sent", content="maybe i should join sometime", timestamp=1040),
        Message(role="received", content="yeah you totally should", timestamp=1045),
    ]
    is_valid, errors = validate_conversation_quality(messages)
    assert is_valid
    assert len(errors) == 0


def test_quality_passes_valid_conversation():
    messages = [
        Message(role="sent", content="hey whats up", timestamp=0),
        Message(role="received", content="not much just chilling", timestamp=0),
        Message(role="sent", content="wanna play later?", timestamp=0),
        Message(role="received", content="sure what time", timestamp=0),
        Message(role="sent", content="like 5pm", timestamp=0),
        Message(role="received", content="cool see you then", timestamp=0),
        Message(role="sent", content="awesome", timestamp=0),
        Message(role="received", content="👍", timestamp=0),
    ]
    is_valid, errors = validate_conversation_quality(messages)
    assert is_valid
    assert errors == []


def test_quality_fails_too_few_messages():
    messages = [
        Message(role="sent", content="hi", timestamp=0),
        Message(role="received", content="hey", timestamp=0),
    ]
    is_valid, errors = validate_conversation_quality(messages)
    assert not is_valid
    assert any("short" in e.lower() or "length" in e.lower() for e in errors)


def test_quality_fails_invalid_role():
    messages = [
        Message.model_construct(role="unknown", content="hi", timestamp=0),
        Message(role="received", content="hey", timestamp=0),
        Message(role="sent", content="sup", timestamp=0),
        Message(role="received", content="nm", timestamp=0),
        Message(role="sent", content="cool", timestamp=0),
        Message(role="received", content="yeah", timestamp=0),
        Message(role="sent", content="ok", timestamp=0),
        Message(role="received", content="k", timestamp=0),
    ]
    is_valid, errors = validate_conversation_quality(messages)
    assert not is_valid
