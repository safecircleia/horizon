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
