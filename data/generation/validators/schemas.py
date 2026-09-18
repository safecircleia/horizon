"""Pydantic models for synthetic conversation validation."""

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    """Risk severity levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(str, Enum):
    """Risk categories for child safety."""

    GROOMING = "grooming"
    BULLYING = "bullying"
    SEXUAL_CONTENT = "sexual_content"
    ISOLATION = "isolation"
    PERSONAL_INFO = "personal_info"
    PLATFORM_MIGRATION = "platform_migration"
    THREATS = "threats"
    BENIGN = "benign"


class GroomingStage(str, Enum):
    """Grooming progression stages (only meaningful when category=grooming)."""

    TARGETING = "targeting"
    TRUST_BUILDING = "trust_building"
    ISOLATION = "isolation"
    DESENSITIZATION = "desensitization"


class Message(BaseModel):
    """Single message in a conversation."""

    role: str = Field(..., description="Message sender role: 'sent' or 'received'")
    content: str = Field(..., min_length=1, description="Message text content")
    timestamp: int = Field(default=0, description="Unix timestamp")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ["sent", "received"]:
            raise ValueError("Role must be 'sent' or 'received'")
        return v


class ConversationLabel(BaseModel):
    """Ground truth labels for a conversation."""

    risk_level: RiskLevel
    categories: list[RiskCategory] = Field(default_factory=list)
    severity_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(
        ..., min_length=10, description="Why this label was assigned"
    )
    grooming_stage: GroomingStage | None = Field(
        default=None,
        description="Grooming progression stage; only set when category includes grooming",
    )
    context_dependent: bool = Field(
        default=False,
        description="True when risk only becomes apparent across multiple messages in sequence",
    )

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, v: list[RiskCategory], info) -> list[RiskCategory]:
        # Benign conversations should have no other categories
        if RiskCategory.BENIGN in v and len(v) > 1:
            raise ValueError("Benign category cannot be combined with risk categories")
        return v


class SyntheticConversation(BaseModel):
    """Complete synthetic conversation with labels."""

    conversation_id: str = Field(..., min_length=1)
    category: RiskCategory = Field(..., description="Primary category for generation")
    messages: list[Message] = Field(..., min_length=2)
    label: ConversationLabel
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[Message]) -> list[Message]:
        if len(v) < 2:
            raise ValueError("Conversation must have at least 2 messages")
        return v

    def to_jsonl(self) -> str:
        """Serialize to JSONL format."""
        return json.dumps(self.model_dump(), separators=(",", ":"))

    @classmethod
    def from_jsonl(cls, line: str) -> "SyntheticConversation":
        """Deserialize from JSONL format."""
        data = json.loads(line)
        return cls(**data)
