# Phase 1: Foundation - Data Generation Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build data generation pipeline to create 10K synthetic conversations for risk detection training

**Architecture:** Prompt template system + LLM API wrappers (Claude/GPT-4) + quality validators + JSONL output pipeline. Each risk category gets dedicated prompt templates with severity variants. Validators check conversation quality, diversity, and label consistency.

**Tech Stack:** Python 3.10+, anthropic SDK, openai SDK, jsonlines, pydantic (validation), pytest

---

## File Structure

**New files to create:**
```
data/generation/
├── prompts/
│   ├── __init__.py
│   ├── base.py                    # Base prompt templates and formatting
│   ├── grooming.py               # Grooming category prompts
│   ├── bullying.py               # Bullying category prompts
│   ├── sexual_content.py         # Sexual content prompts
│   ├── isolation.py              # Isolation/control prompts
│   ├── personal_info.py          # Personal info request prompts
│   ├── platform_migration.py     # Platform migration prompts
│   ├── threats.py                # Threats/violence prompts
│   └── benign.py                 # Benign conversation prompts
├── generators/
│   ├── __init__.py
│   ├── base.py                   # Base generator interface
│   ├── claude_generator.py       # Claude API generator
│   └── gpt_generator.py          # GPT-4 API generator
├── validators/
│   ├── __init__.py
│   ├── schemas.py                # Pydantic models for output validation
│   ├── quality.py                # Quality checks (length, diversity, etc.)
│   └── label_validator.py        # Label consistency validation
├── scripts/
│   ├── __init__.py
│   ├── generate.py               # Main generation entry point
│   └── stats.py                  # Dataset statistics reporting
└── config.yaml                    # Generation configuration

data/raw/                          # Generated conversations (JSONL)
data/processed/                    # Formatted training data

tests/data/generation/
├── test_prompts.py
├── test_generators.py
└── test_validators.py
```

---

## Task 1: Project Dependencies & Configuration

**Files:**
- Modify: `requirements.txt`
- Create: `data/generation/config.yaml`
- Create: `.env.example`

- [ ] **Step 1: Add data generation dependencies to requirements.txt**

```txt
# Existing dependencies remain...

# Data generation
anthropic>=0.25.0
openai>=1.35.0
jsonlines>=4.0.0
pydantic>=2.7.0
pyyaml>=6.0.1
python-dotenv>=1.0.0
tqdm>=4.66.0

# Testing
pytest>=8.2.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 2: Create environment variable template**

Create `.env.example`:

```bash
# API Keys for data generation
ANTHROPIC_API_KEY=your_claude_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Generation settings
DEFAULT_GENERATOR=claude  # or 'openai'
MAX_RETRIES=3
RATE_LIMIT_DELAY=1.0
```

- [ ] **Step 3: Create generation configuration file**

Create `data/generation/config.yaml`:

```yaml
# Data Generation Configuration

# Target dataset size
dataset_targets:
  grooming: 1000
  bullying: 1000
  sexual_content: 1000
  isolation: 1000
  personal_info: 1000
  platform_migration: 1000
  threats: 1000
  benign: 2000

# Severity distribution (for risk categories)
severity_distribution:
  low: 0.25
  medium: 0.35
  high: 0.25
  critical: 0.15

# Generation parameters
generation:
  model_claude: "claude-3-5-sonnet-20241022"
  model_openai: "gpt-4o-2024-08-06"
  temperature: 0.9
  max_tokens: 2000
  batch_size: 10
  retry_attempts: 3

# Quality thresholds
quality:
  min_conversation_length: 4  # minimum messages
  max_conversation_length: 30  # maximum messages
  min_unique_tokens: 20  # vocabulary diversity
  max_repetition_ratio: 0.3  # avoid copy-paste patterns
```

- [ ] **Step 4: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

- [ ] **Step 5: Commit configuration**

```bash
git add requirements.txt data/generation/config.yaml .env.example
git commit -m "feat: add data generation dependencies and config"
```

---

## Task 2: Data Schemas & Validation Models

**Files:**
- Create: `data/generation/validators/__init__.py`
- Create: `data/generation/validators/schemas.py`
- Create: `tests/data/generation/test_validators.py`

- [ ] **Step 1: Write test for conversation schema validation**

Create `tests/data/generation/test_validators.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/generation/test_validators.py -v`
Expected: FAIL with "No module named 'data.generation.validators.schemas'"

- [ ] **Step 3: Create validator package init**

Create `data/generation/validators/__init__.py`:

```python
"""Data validation components for synthetic conversation generation."""

from data.generation.validators.schemas import (
    Message,
    ConversationLabel,
    SyntheticConversation,
    RiskLevel,
    RiskCategory
)

__all__ = [
    "Message",
    "ConversationLabel",
    "SyntheticConversation",
    "RiskLevel",
    "RiskCategory",
]
```

- [ ] **Step 4: Implement schemas with Pydantic**

Create `data/generation/validators/schemas.py`:

```python
"""Pydantic models for synthetic conversation validation."""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
import json


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


class Message(BaseModel):
    """Single message in a conversation."""
    role: str = Field(..., description="Message sender role: 'sent' or 'received'")
    content: str = Field(..., min_length=1, description="Message text content")
    timestamp: int = Field(..., gt=0, description="Unix timestamp")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ["sent", "received"]:
            raise ValueError("Role must be 'sent' or 'received'")
        return v


class ConversationLabel(BaseModel):
    """Ground truth labels for a conversation."""
    risk_level: RiskLevel
    categories: List[RiskCategory] = Field(default_factory=list)
    severity_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=10, description="Why this label was assigned")

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, v: List[RiskCategory], info) -> List[RiskCategory]:
        # Benign conversations should have no other categories
        if RiskCategory.BENIGN in v and len(v) > 1:
            raise ValueError("Benign category cannot be combined with risk categories")
        return v


class SyntheticConversation(BaseModel):
    """Complete synthetic conversation with labels."""
    conversation_id: str = Field(..., min_length=1)
    category: RiskCategory = Field(..., description="Primary category for generation")
    messages: List[Message] = Field(..., min_length=2)
    label: ConversationLabel
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List[Message]) -> List[Message]:
        if len(v) < 2:
            raise ValueError("Conversation must have at least 2 messages")
        return v

    def to_jsonl(self) -> str:
        """Serialize to JSONL format."""
        return json.dumps(self.model_dump(), separators=(',', ':'))

    @classmethod
    def from_jsonl(cls, line: str) -> "SyntheticConversation":
        """Deserialize from JSONL format."""
        data = json.loads(line)
        return cls(**data)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/data/generation/test_validators.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit schemas**

```bash
git add data/generation/validators/ tests/data/generation/test_validators.py
git commit -m "feat: add pydantic schemas for conversation validation"
```

---

## Task 3: Base Prompt Template System

**Files:**
- Create: `data/generation/prompts/__init__.py`
- Create: `data/generation/prompts/base.py`
- Create: `tests/data/generation/test_prompts.py`

- [ ] **Step 1: Write tests for prompt template system**

Create `tests/data/generation/test_prompts.py`:

```python
import pytest
from data.generation.prompts.base import (
    PromptTemplate,
    ConversationPrompt,
    format_system_prompt,
    create_conversation_prompt
)
from data.generation.validators.schemas import RiskCategory, RiskLevel


def test_format_system_prompt():
    """Test system prompt formatting."""
    system = format_system_prompt()
    assert "SafeCircle" in system
    assert "synthetic conversation" in system
    assert "realistic teen communication" in system


def test_conversation_prompt_basic():
    """Test basic conversation prompt creation."""
    prompt = create_conversation_prompt(
        category=RiskCategory.GROOMING,
        severity=RiskLevel.MEDIUM,
        child_age=14,
        num_messages=8
    )
    
    assert prompt.category == RiskCategory.GROOMING
    assert prompt.severity == RiskLevel.MEDIUM
    assert "14" in prompt.user_prompt
    assert "8" in prompt.user_prompt or "eight" in prompt.user_prompt.lower()


def test_conversation_prompt_includes_category():
    """Test prompt includes category-specific guidance."""
    prompt = create_conversation_prompt(
        category=RiskCategory.BULLYING,
        severity=RiskLevel.HIGH,
        child_age=13,
        num_messages=6
    )
    
    # Should mention category and be detailed
    assert len(prompt.user_prompt) > 100
    assert prompt.category == RiskCategory.BULLYING


def test_prompt_template_format():
    """Test PromptTemplate formats correctly."""
    template = PromptTemplate(
        system="You are a test assistant.",
        user="Generate {count} messages about {topic}."
    )
    
    formatted = template.format(count=5, topic="testing")
    assert formatted.system == "You are a test assistant."
    assert "5" in formatted.user
    assert "testing" in formatted.user
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/generation/test_prompts.py -v`
Expected: FAIL with "No module named 'data.generation.prompts.base'"

- [ ] **Step 3: Create prompts package init**

Create `data/generation/prompts/__init__.py`:

```python
"""Prompt templates for synthetic conversation generation."""

from data.generation.prompts.base import (
    PromptTemplate,
    ConversationPrompt,
    format_system_prompt,
    create_conversation_prompt
)

__all__ = [
    "PromptTemplate",
    "ConversationPrompt",
    "format_system_prompt",
    "create_conversation_prompt",
]
```

- [ ] **Step 4: Implement base prompt system**

Create `data/generation/prompts/base.py`:

```python
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

Your task is to generate realistic chat conversations between teenagers (13-17) and other parties for training AI safety models. These conversations must:

1. Use authentic teen communication patterns (abbreviations, emojis, slang, casual grammar)
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/data/generation/test_prompts.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit base prompt system**

```bash
git add data/generation/prompts/ tests/data/generation/test_prompts.py
git commit -m "feat: add base prompt template system"
```

---

## Task 4: Category-Specific Prompts (Grooming)

**Files:**
- Create: `data/generation/prompts/grooming.py`
- Modify: `tests/data/generation/test_prompts.py`

- [ ] **Step 1: Write tests for grooming prompts**

Add to `tests/data/generation/test_prompts.py`:

```python
from data.generation.prompts.grooming import create_grooming_prompt


def test_grooming_prompt_low_severity():
    """Test low severity grooming prompt."""
    prompt = create_grooming_prompt(
        severity=RiskLevel.LOW,
        child_age=15,
        num_messages=6
    )
    
    assert prompt.category == RiskCategory.GROOMING
    assert prompt.severity == RiskLevel.LOW
    assert "trust building" in prompt.user_prompt.lower() or "friendly" in prompt.user_prompt.lower()


def test_grooming_prompt_high_severity():
    """Test high severity grooming prompt."""
    prompt = create_grooming_prompt(
        severity=RiskLevel.HIGH,
        child_age=14,
        num_messages=10
    )
    
    assert prompt.category == RiskCategory.GROOMING
    assert prompt.severity == RiskLevel.HIGH
    assert "secrecy" in prompt.user_prompt.lower() or "secret" in prompt.user_prompt.lower()


def test_grooming_prompt_critical_severity():
    """Test critical severity grooming prompt."""
    prompt = create_grooming_prompt(
        severity=RiskLevel.CRITICAL,
        child_age=13,
        num_messages=12
    )
    
    assert prompt.category == RiskCategory.GROOMING
    assert prompt.severity == RiskLevel.CRITICAL
    # Critical should mention explicit manipulation
    assert len(prompt.user_prompt) > 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/generation/test_prompts.py::test_grooming_prompt_low_severity -v`
Expected: FAIL with "cannot import name 'create_grooming_prompt'"

- [ ] **Step 3: Implement grooming-specific prompts**

Create `data/generation/prompts/grooming.py`:

```python
"""Grooming category prompt templates."""

from data.generation.prompts.base import ConversationPrompt, create_conversation_prompt
from data.generation.validators.schemas import RiskCategory, RiskLevel


# Severity-specific guidance for grooming conversations
GROOMING_SEVERITY_GUIDES = {
    RiskLevel.LOW: """
**Low Severity Grooming Indicators:**
- Overly friendly approach from stranger or much older person
- Compliments and flattery
- Showing unusual interest in the child's life
- Building rapport without immediate boundary violations
- May claim to be same age or peer
- Asking general questions about interests, school, friends
""",
    RiskLevel.MEDIUM: """
**Medium Severity Grooming Indicators:**
- Establishing trust and emotional connection
- Testing boundaries with slightly inappropriate comments
- Offering gifts, gaming currency, or other incentives
- Beginning to isolate child (e.g., "you can talk to me about things")
- Introducing sexual topics indirectly
- Asking about relationship status or romantic interests
- Suggesting private communication channels
""",
    RiskLevel.HIGH: """
**High Severity Grooming Indicators:**
- Explicit attempts to establish secrecy ("don't tell your parents")
- Requesting personal information (address, school location)
- Sexual conversations or requests for photos
- Manipulative tactics (guilt, special relationship, blackmail)
- Planning to meet in person
- Clear adult-to-child dynamic with exploitation intent
- Normalizing inappropriate behavior
""",
    RiskLevel.CRITICAL: """
**Critical Severity Grooming Indicators:**
- Explicit sexual content or demands
- Coercion or threats to maintain compliance
- Advanced stage of exploitation
- Concrete plans for offline meeting with harmful intent
- Blackmail using previously shared content
- Clear and immediate danger to the child
"""
}


def create_grooming_prompt(
    severity: RiskLevel,
    child_age: int,
    num_messages: int
) -> ConversationPrompt:
    """Create grooming category conversation prompt.
    
    Args:
        severity: Risk severity level
        child_age: Age of child (13-17)
        num_messages: Target message count
    
    Returns:
        ConversationPrompt configured for grooming scenarios
    """
    # Get severity-specific guidance
    severity_guide = GROOMING_SEVERITY_GUIDES.get(severity, "")
    
    additional_context = f"""**Grooming Scenario Requirements:**

{severity_guide}

**Conversation Progression:**
- Start with initial contact (friend request, game chat, comment reply, etc.)
- Build conversation naturally - don't jump immediately to risk behaviors
- Include realistic teen responses (may be trusting, curious, or cautious)
- Show gradual escalation appropriate to severity level
- Include platform-appropriate context (Instagram DMs, Discord, Roblox chat, etc.)

**Authenticity Guidelines:**
- Adult may claim to be a peer initially
- Use age-appropriate language for both parties
- Include typical online conversation elements (reaction to messages, delays, emoji use)
- Reflect how real grooming often progresses slowly
"""

    return create_conversation_prompt(
        category=RiskCategory.GROOMING,
        severity=severity,
        child_age=child_age,
        num_messages=num_messages,
        additional_context=additional_context
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/generation/test_prompts.py -k grooming -v`
Expected: All grooming tests PASS

- [ ] **Step 5: Commit grooming prompts**

```bash
git add data/generation/prompts/grooming.py tests/data/generation/test_prompts.py
git commit -m "feat: add grooming category prompts"
```

---

## Task 5: Category-Specific Prompts (Bullying)

**Files:**
- Create: `data/generation/prompts/bullying.py`
- Modify: `tests/data/generation/test_prompts.py`

- [ ] **Step 1: Write tests for bullying prompts**

Add to `tests/data/generation/test_prompts.py`:

```python
from data.generation.prompts.bullying import create_bullying_prompt


def test_bullying_prompt_low_severity():
    """Test low severity bullying prompt."""
    prompt = create_bullying_prompt(
        severity=RiskLevel.LOW,
        child_age=14,
        num_messages=5
    )
    assert prompt.category == RiskCategory.BULLYING
    assert "teasing" in prompt.user_prompt.lower() or "exclusion" in prompt.user_prompt.lower()


def test_bullying_prompt_high_severity():
    """Test high severity bullying prompt."""
    prompt = create_bullying_prompt(
        severity=RiskLevel.HIGH,
        child_age=15,
        num_messages=8
    )
    assert prompt.category == RiskCategory.BULLYING
    assert "harassment" in prompt.user_prompt.lower() or "threats" in prompt.user_prompt.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/generation/test_prompts.py -k bullying -v`
Expected: FAIL with "cannot import name 'create_bullying_prompt'"

- [ ] **Step 3: Implement bullying prompts**

Create `data/generation/prompts/bullying.py`:

```python
"""Bullying category prompt templates."""

from data.generation.prompts.base import ConversationPrompt, create_conversation_prompt
from data.generation.validators.schemas import RiskCategory, RiskLevel


BULLYING_SEVERITY_GUIDES = {
    RiskLevel.LOW: """
**Low Severity Bullying Indicators:**
- Mild teasing or mockery
- Social exclusion hints ("we're not inviting you")
- Passive-aggressive comments
- Ignoring or dismissing messages
- Subtle put-downs about appearance, interests, or abilities
""",
    RiskLevel.MEDIUM: """
**Medium Severity Bullying Indicators:**
- Direct insults and name-calling
- Persistent harassment across multiple messages
- Spreading rumors or gossip
- Mocking physical appearance, identity, or personal situations
- Group bullying (multiple people ganging up)
- Screenshots/sharing private conversations to embarrass
""",
    RiskLevel.HIGH: """
**High Severity Bullying Indicators:**
- Severe verbal abuse and degrading language
- Threats of violence or harm
- Sustained campaign of harassment
- Encouraging self-harm or suicide
- Doxxing or sharing personal information maliciously
- Creating fake accounts to harass
- Coordinated group attacks
""",
    RiskLevel.CRITICAL: """
**Critical Severity Bullying Indicators:**
- Explicit death threats or violence threats
- Blackmail with compromising content
- Organizing real-world physical harm
- Severe psychological abuse causing clear distress
- Multiple perpetrators with systematic targeting
"""
}


def create_bullying_prompt(
    severity: RiskLevel,
    child_age: int,
    num_messages: int
) -> ConversationPrompt:
    """Create bullying category conversation prompt."""
    severity_guide = BULLYING_SEVERITY_GUIDES.get(severity, "")
    
    additional_context = f"""**Bullying Scenario Requirements:**

{severity_guide}

**Conversation Context:**
- May be peer-to-peer (same age group)
- Could be group chat, direct messages, or public comments
- Often relates to school, social groups, or online communities
- Include realistic teen social dynamics and hierarchies

**Authenticity Guidelines:**
- Show realistic teen communication patterns on both sides
- Include context clues about relationships (classmates, former friends, online community)
- Bullying may reference real or perceived social status, appearance, behavior
- Victim responses may range from defensive to conciliatory to silent
"""

    return create_conversation_prompt(
        category=RiskCategory.BULLYING,
        severity=severity,
        child_age=child_age,
        num_messages=num_messages,
        additional_context=additional_context
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/generation/test_prompts.py -k bullying -v`
Expected: All bullying tests PASS

- [ ] **Step 5: Commit bullying prompts**

```bash
git add data/generation/prompts/bullying.py tests/data/generation/test_prompts.py
git commit -m "feat: add bullying category prompts"
```

---

## Task 6: Category-Specific Prompts (Sexual Content)

**Files:**
- Create: `data/generation/prompts/sexual_content.py`
- Modify: `tests/data/generation/test_prompts.py`

- [ ] **Step 1: Write tests for sexual content prompts**

Add to `tests/data/generation/test_prompts.py`:

```python
from data.generation.prompts.sexual_content import create_sexual_content_prompt


def test_sexual_content_prompt_medium_severity():
    """Test medium severity sexual content prompt."""
    prompt = create_sexual_content_prompt(
        severity=RiskLevel.MEDIUM,
        child_age=15,
        num_messages=6
    )
    assert prompt.category == RiskCategory.SEXUAL_CONTENT
    assert "inappropriate" in prompt.user_prompt.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/generation/test_prompts.py -k sexual_content -v`
Expected: FAIL with "cannot import name 'create_sexual_content_prompt'"

- [ ] **Step 3: Implement sexual content prompts**

Create `data/generation/prompts/sexual_content.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/generation/test_prompts.py -k sexual_content -v`
Expected: All sexual content tests PASS

- [ ] **Step 5: Commit sexual content prompts**

```bash
git add data/generation/prompts/sexual_content.py tests/data/generation/test_prompts.py
git commit -m "feat: add sexual content category prompts"
```

---

## Task 7: Remaining Category Prompts (Isolation, Personal Info, Platform Migration, Threats)

**Files:**
- Create: `data/generation/prompts/isolation.py`
- Create: `data/generation/prompts/personal_info.py`
- Create: `data/generation/prompts/platform_migration.py`
- Create: `data/generation/prompts/threats.py`
- Modify: `tests/data/generation/test_prompts.py`

- [ ] **Step 1: Write tests for remaining categories**

Add to `tests/data/generation/test_prompts.py`:

```python
from data.generation.prompts.isolation import create_isolation_prompt
from data.generation.prompts.personal_info import create_personal_info_prompt
from data.generation.prompts.platform_migration import create_platform_migration_prompt
from data.generation.prompts.threats import create_threats_prompt


def test_isolation_prompt():
    """Test isolation/control prompt."""
    prompt = create_isolation_prompt(RiskLevel.MEDIUM, 14, 7)
    assert prompt.category == RiskCategory.ISOLATION


def test_personal_info_prompt():
    """Test personal info request prompt."""
    prompt = create_personal_info_prompt(RiskLevel.MEDIUM, 15, 6)
    assert prompt.category == RiskCategory.PERSONAL_INFO


def test_platform_migration_prompt():
    """Test platform migration prompt."""
    prompt = create_platform_migration_prompt(RiskLevel.MEDIUM, 14, 5)
    assert prompt.category == RiskCategory.PLATFORM_MIGRATION


def test_threats_prompt():
    """Test threats/violence prompt."""
    prompt = create_threats_prompt(RiskLevel.HIGH, 15, 8)
    assert prompt.category == RiskCategory.THREATS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/generation/test_prompts.py -k "isolation or personal_info or platform_migration or threats" -v`
Expected: FAIL with import errors

- [ ] **Step 3: Implement isolation prompts**

Create `data/generation/prompts/isolation.py`:

```python
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
```

- [ ] **Step 4: Implement personal info prompts**

Create `data/generation/prompts/personal_info.py`:

```python
"""Personal information request category prompt templates."""

from data.generation.prompts.base import ConversationPrompt, create_conversation_prompt
from data.generation.validators.schemas import RiskCategory, RiskLevel


PERSONAL_INFO_SEVERITY_GUIDES = {
    RiskLevel.LOW: """
**Low Severity Personal Info Indicators:**
- Asking about general location (city, state)
- Questions about school name or grade level
- Inquiring about hobbies, interests, schedule
""",
    RiskLevel.MEDIUM: """
**Medium Severity Personal Info Indicators:**
- Requesting specific school name or location
- Asking for social media account names
- Requesting phone number or email
- Questions about home address or neighborhood
- Asking about parents' work or schedules
""",
    RiskLevel.HIGH: """
**High Severity Personal Info Indicators:**
- Demanding exact address or location
- Requesting passwords or account access
- Asking for compromising personal information
- Persistent requests after refusal
- Information gathering for stalking or targeting
"""
}


def create_personal_info_prompt(
    severity: RiskLevel,
    child_age: int,
    num_messages: int
) -> ConversationPrompt:
    """Create personal info request category conversation prompt."""
    severity_guide = PERSONAL_INFO_SEVERITY_GUIDES.get(severity, "")
    
    additional_context = f"""**Personal Info Request Scenario Requirements:**

{severity_guide}

**Conversation Context:**
- May be standalone or combined with other risk categories
- Often disguised as innocent questions
- Show escalation from general to specific information
- Include realistic child responses (may share some info, resist other requests)
"""

    return create_conversation_prompt(
        category=RiskCategory.PERSONAL_INFO,
        severity=severity,
        child_age=child_age,
        num_messages=num_messages,
        additional_context=additional_context
    )
```

- [ ] **Step 5: Implement platform migration prompts**

Create `data/generation/prompts/platform_migration.py`:

```python
"""Platform migration category prompt templates."""

from data.generation.prompts.base import ConversationPrompt, create_conversation_prompt
from data.generation.validators.schemas import RiskCategory, RiskLevel


PLATFORM_MIGRATION_SEVERITY_GUIDES = {
    RiskLevel.LOW: """
**Low Severity Platform Migration Indicators:**
- Suggesting alternative platform casually
- "Add me on [other platform]"
- Mentioning preferences for different apps
""",
    RiskLevel.MEDIUM: """
**Medium Severity Platform Migration Indicators:**
- Pressuring to move to less monitored platform
- Claiming current platform isn't private enough
- Suggesting messaging apps with encryption or disappearing messages
- Multiple attempts to get child to switch platforms
""",
    RiskLevel.HIGH: """
**High Severity Platform Migration Indicators:**
- Demanding move to private platform
- Using migration to avoid monitoring/detection
- Threatening to end communication if child won't switch
- Explicitly stating desire to avoid parental oversight
"""
}


def create_platform_migration_prompt(
    severity: RiskLevel,
    child_age: int,
    num_messages: int
) -> ConversationPrompt:
    """Create platform migration category conversation prompt."""
    severity_guide = PLATFORM_MIGRATION_SEVERITY_GUIDES.get(severity, "")
    
    additional_context = f"""**Platform Migration Scenario Requirements:**

{severity_guide}

**Conversation Context:**
- Usually occurs after initial contact on one platform
- Predators prefer platforms with less monitoring (Telegram, Kik, Discord DMs)
- May claim parental controls, moderation, or platform features are reason to switch
- Include platform-specific context (Instagram → Snapchat, Roblox → Discord, etc.)
"""

    return create_conversation_prompt(
        category=RiskCategory.PLATFORM_MIGRATION,
        severity=severity,
        child_age=child_age,
        num_messages=num_messages,
        additional_context=additional_context
    )
```

- [ ] **Step 6: Implement threats prompts**

Create `data/generation/prompts/threats.py`:

```python
"""Threats/violence category prompt templates."""

from data.generation.prompts.base import ConversationPrompt, create_conversation_prompt
from data.generation.validators.schemas import RiskCategory, RiskLevel


THREATS_SEVERITY_GUIDES = {
    RiskLevel.LOW: """
**Low Severity Threat Indicators:**
- Vague threatening language
- Implied consequences for actions
- Aggressive tone without specific threats
- "You'll regret this" type statements
""",
    RiskLevel.MEDIUM: """
**Medium Severity Threat Indicators:**
- Direct but non-specific threats
- Threatening to harm reputation or relationships
- Intimidation tactics
- Threatening to share embarrassing content
- Encouraging risky or dangerous behavior
""",
    RiskLevel.HIGH: """
**High Severity Threat Indicators:**
- Specific threats of physical violence
- Death threats
- Threats against family or friends
- Encouraging self-harm or suicide
- Planning or organizing violence
- Blackmail with serious consequences
""",
    RiskLevel.CRITICAL: """
**Critical Severity Threat Indicators:**
- Imminent danger of violence
- Detailed plans to harm self or others
- Active suicide encouragement
- Terrorist-related content
- Severe blackmail with immediate danger
"""
}


def create_threats_prompt(
    severity: RiskLevel,
    child_age: int,
    num_messages: int
) -> ConversationPrompt:
    """Create threats/violence category conversation prompt."""
    severity_guide = THREATS_SEVERITY_GUIDES.get(severity, "")
    
    additional_context = f"""**Threats/Violence Scenario Requirements:**

{severity_guide}

**Conversation Context:**
- May be peer-to-peer conflict or adult-to-child
- Could be retaliation for perceived offense
- May include cyberstalking elements
- Show escalation pattern if appropriate
- Include realistic fear responses from victim
"""

    return create_conversation_prompt(
        category=RiskCategory.THREATS,
        severity=severity,
        child_age=child_age,
        num_messages=num_messages,
        additional_context=additional_context
    )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/data/generation/test_prompts.py -k "isolation or personal_info or platform_migration or threats" -v`
Expected: All tests PASS

- [ ] **Step 8: Commit remaining category prompts**

```bash
git add data/generation/prompts/isolation.py \
    data/generation/prompts/personal_info.py \
    data/generation/prompts/platform_migration.py \
    data/generation/prompts/threats.py \
    tests/data/generation/test_prompts.py
git commit -m "feat: add isolation, personal info, platform migration, and threats prompts"
```

---

## Task 8: Benign Conversation Prompts

**Files:**
- Create: `data/generation/prompts/benign.py`
- Modify: `tests/data/generation/test_prompts.py`

- [ ] **Step 1: Write tests for benign prompts**

Add to `tests/data/generation/test_prompts.py`:

```python
from data.generation.prompts.benign import create_benign_prompt


def test_benign_prompt():
    """Test benign conversation prompt."""
    prompt = create_benign_prompt(child_age=15, num_messages=8)
    assert prompt.category == RiskCategory.BENIGN
    assert prompt.severity == RiskLevel.NONE
    assert "safe" in prompt.user_prompt.lower() or "benign" in prompt.user_prompt.lower()


def test_benign_prompt_varied_contexts():
    """Test benign prompts have varied contexts."""
    prompts = [create_benign_prompt(14, 6) for _ in range(3)]
    # Each should be valid
    for p in prompts:
        assert p.category == RiskCategory.BENIGN
        assert len(p.user_prompt) > 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/generation/test_prompts.py -k benign -v`
Expected: FAIL with "cannot import name 'create_benign_prompt'"

- [ ] **Step 3: Implement benign conversation prompts**

Create `data/generation/prompts/benign.py`:

```python
"""Benign conversation category prompt templates."""

from data.generation.prompts.base import ConversationPrompt, create_conversation_prompt
from data.generation.validators.schemas import RiskCategory, RiskLevel
import random


# Types of benign conversations to generate variety
BENIGN_CONVERSATION_TYPES = [
    "Friends discussing school, homework, or classes",
    "Peers talking about video games or gaming strategies",
    "Classmates planning a group project or study session",
    "Friends discussing movies, TV shows, or music",
    "Siblings or relatives having casual conversation",
    "Peers discussing sports, hobbies, or extracurricular activities",
    "Friends making plans to hang out (with appropriate context)",
    "Peers sharing memes, jokes, or funny content",
    "Classmates discussing social events (school dance, party, etc.)",
    "Friends supporting each other through normal teen challenges",
    "Peers discussing social media posts or trends",
    "Friends talking about fashion, style, or personal interests",
    "Classmates coordinating for a school event",
    "Peers discussing pets, animals, or nature",
    "Friends having lighthearted banter and conversation"
]


def create_benign_prompt(
    child_age: int,
    num_messages: int
) -> ConversationPrompt:
    """Create benign conversation prompt.
    
    Benign conversations are crucial for training the model to avoid false positives.
    They should feel realistic and natural, covering typical teen interactions.
    
    Args:
        child_age: Age of child participants (13-17)
        num_messages: Target message count
    
    Returns:
        ConversationPrompt for benign scenario
    """
    # Randomly select conversation type for variety
    conversation_type = random.choice(BENIGN_CONVERSATION_TYPES)
    
    additional_context = f"""**Benign Conversation Requirements:**

**Scenario Type:** {conversation_type}

**Key Requirements:**
- NO risk indicators of any kind
- Natural, authentic teen communication
- Appropriate topics and language
- May include slang, abbreviations, emojis (normal teen usage)
- Positive or neutral emotional tone
- Realistic context (platforms, references, timing)

**Common False Positive Traps to Avoid:**
- Don't include ANY of these even innocuously:
  - Requests for personal info (addresses, passwords, etc.)
  - Secrecy requests
  - Platform switching
  - Boundary testing
  - Inappropriate compliments or flirtation
  - Manipulation tactics
  - Threats or aggressive language

**Examples of Safe Content:**
- Discussing homework assignments
- Making weekend plans with context showing parental awareness
- Gaming strategies and achievements
- Sharing funny content
- Supporting friend through normal stress (test anxiety, friend drama)
- Planning group activities
- Discussing interests and hobbies

**Important:** This is training data for false positive reduction. The conversation must be completely benign while still feeling realistic.
"""

    return create_conversation_prompt(
        category=RiskCategory.BENIGN,
        severity=RiskLevel.NONE,
        child_age=child_age,
        num_messages=num_messages,
        additional_context=additional_context
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/generation/test_prompts.py -k benign -v`
Expected: All benign tests PASS

- [ ] **Step 5: Commit benign prompts**

```bash
git add data/generation/prompts/benign.py tests/data/generation/test_prompts.py
git commit -m "feat: add benign conversation prompts"
```

---

## Task 9: LLM Generator Interface

**Files:**
- Create: `data/generation/generators/__init__.py`
- Create: `data/generation/generators/base.py`
- Create: `tests/data/generation/test_generators.py`

- [ ] **Step 1: Write tests for generator interface**

Create `tests/data/generation/test_generators.py`:

```python
import pytest
from data.generation.generators.base import ConversationGenerator, GenerationResult
from data.generation.prompts.base import ConversationPrompt
from data.generation.validators.schemas import RiskCategory, RiskLevel


def test_generation_result_success():
    """Test successful generation result."""
    result = GenerationResult(
        success=True,
        conversation={
            "messages": [
                {"role": "sent", "content": "hi", "timestamp": 1000},
                {"role": "received", "content": "hey", "timestamp": 1001}
            ],
            "reasoning": "Test conversation"
        },
        raw_response='{"messages": [...], "reasoning": "Test"}',
        error=None
    )
    assert result.success
    assert len(result.conversation["messages"]) == 2
    assert result.error is None


def test_generation_result_failure():
    """Test failed generation result."""
    result = GenerationResult(
        success=False,
        conversation=None,
        raw_response="Error occurred",
        error="API timeout"
    )
    assert not result.success
    assert result.conversation is None
    assert result.error == "API timeout"


def test_generator_interface_methods():
    """Test generator interface has required methods."""
    # This tests the abstract base class structure
    assert hasattr(ConversationGenerator, 'generate')
    assert hasattr(ConversationGenerator, 'name')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/generation/test_generators.py -v`
Expected: FAIL with "No module named 'data.generation.generators.base'"

- [ ] **Step 3: Create generators package init**

Create `data/generation/generators/__init__.py`:

```python
"""LLM-based conversation generators."""

from data.generation.generators.base import (
    ConversationGenerator,
    GenerationResult
)

__all__ = [
    "ConversationGenerator",
    "GenerationResult",
]
```

- [ ] **Step 4: Implement base generator interface**

Create `data/generation/generators/base.py`:

```python
"""Base interface for conversation generators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from data.generation.prompts.base import ConversationPrompt


@dataclass
class GenerationResult:
    """Result from a conversation generation attempt."""
    success: bool
    conversation: Optional[Dict[str, Any]]
    raw_response: str
    error: Optional[str] = None


class ConversationGenerator(ABC):
    """Abstract base class for conversation generators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Generator name (e.g., 'claude', 'openai')."""
        pass

    @abstractmethod
    async def generate(self, prompt: ConversationPrompt) -> GenerationResult:
        """Generate a conversation from the given prompt.
        
        Args:
            prompt: ConversationPrompt with category, severity, and instructions
        
        Returns:
            GenerationResult with success status and conversation data
        """
        pass

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response, handling common formatting issues.
        
        Args:
            response: Raw LLM response text
        
        Returns:
            Parsed JSON dict or None if parsing fails
        """
        import json
        import re
        
        # Try direct JSON parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find any JSON object in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/data/generation/test_generators.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit base generator interface**

```bash
git add data/generation/generators/ tests/data/generation/test_generators.py
git commit -m "feat: add base generator interface"
```

---

## Task 10: Claude Generator Implementation

**Files:**
- Create: `data/generation/generators/claude_generator.py`
- Modify: `tests/data/generation/test_generators.py`

- [ ] **Step 1: Write tests for Claude generator**

Add to `tests/data/generation/test_generators.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch
from data.generation.generators.claude_generator import ClaudeGenerator
from data.generation.prompts.grooming import create_grooming_prompt


@pytest.mark.asyncio
async def test_claude_generator_initialization():
    """Test Claude generator initializes correctly."""
    generator = ClaudeGenerator(api_key="test_key", model="claude-3-5-sonnet-20241022")
    assert generator.name == "claude"
    assert generator.model == "claude-3-5-sonnet-20241022"


@pytest.mark.asyncio
async def test_claude_generator_success():
    """Test successful generation with Claude."""
    with patch('anthropic.AsyncAnthropic') as mock_client:
        # Mock API response
        mock_message = AsyncMock()
        mock_message.content = [
            type('Content', (), {
                'text': '{"messages": [{"role": "sent", "content": "test", "timestamp": 1000}], "reasoning": "test"}'
            })()
        ]
        mock_client.return_value.messages.create = AsyncMock(return_value=mock_message)
        
        generator = ClaudeGenerator(api_key="test_key")
        prompt = create_grooming_prompt(RiskLevel.LOW, 14, 5)
        result = await generator.generate(prompt)
        
        assert result.success
        assert result.conversation is not None
        assert "messages" in result.conversation


@pytest.mark.asyncio
async def test_claude_generator_api_error():
    """Test handling of API errors."""
    with patch('anthropic.AsyncAnthropic') as mock_client:
        mock_client.return_value.messages.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        generator = ClaudeGenerator(api_key="test_key")
        prompt = create_grooming_prompt(RiskLevel.LOW, 14, 5)
        result = await generator.generate(prompt)
        
        assert not result.success
        assert result.error is not None
        assert "API Error" in result.error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/generation/test_generators.py -k claude -v`
Expected: FAIL with "cannot import name 'ClaudeGenerator'"

- [ ] **Step 3: Implement Claude generator**

Create `data/generation/generators/claude_generator.py`:

```python
"""Claude-based conversation generator."""

import os
from typing import Optional
import anthropic
from data.generation.generators.base import ConversationGenerator, GenerationResult
from data.generation.prompts.base import ConversationPrompt


class ClaudeGenerator(ConversationGenerator):
    """Generate conversations using Claude API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.9,
        max_tokens: int = 2000
    ):
        """Initialize Claude generator.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "claude"

    async def generate(self, prompt: ConversationPrompt) -> GenerationResult:
        """Generate conversation using Claude API."""
        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=prompt.system_prompt,
                messages=[
                    {"role": "user", "content": prompt.user_prompt}
                ]
            )
            
            # Extract text from response
            response_text = message.content[0].text
            
            # Parse JSON conversation
            conversation = self._parse_json_response(response_text)
            
            if conversation is None:
                return GenerationResult(
                    success=False,
                    conversation=None,
                    raw_response=response_text,
                    error="Failed to parse JSON from response"
                )
            
            # Validate required fields
            if "messages" not in conversation:
                return GenerationResult(
                    success=False,
                    conversation=None,
                    raw_response=response_text,
                    error="Response missing 'messages' field"
                )
            
            return GenerationResult(
                success=True,
                conversation=conversation,
                raw_response=response_text,
                error=None
            )
            
        except Exception as e:
            return GenerationResult(
                success=False,
                conversation=None,
                raw_response="",
                error=f"API Error: {str(e)}"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/generation/test_generators.py -k claude -v`
Expected: All Claude tests PASS

- [ ] **Step 5: Commit Claude generator**

```bash
git add data/generation/generators/claude_generator.py tests/data/generation/test_generators.py
git commit -m "feat: add Claude conversation generator"
```

---

## Task 11: OpenAI Generator Implementation

**Files:**
- Create: `data/generation/generators/gpt_generator.py`
- Modify: `tests/data/generation/test_generators.py`

- [ ] **Step 1: Write tests for GPT generator**

Add to `tests/data/generation/test_generators.py`:

```python
from data.generation.generators.gpt_generator import GPTGenerator


@pytest.mark.asyncio
async def test_gpt_generator_initialization():
    """Test GPT generator initializes correctly."""
    generator = GPTGenerator(api_key="test_key", model="gpt-4o-2024-08-06")
    assert generator.name == "openai"
    assert generator.model == "gpt-4o-2024-08-06"


@pytest.mark.asyncio
async def test_gpt_generator_success():
    """Test successful generation with GPT."""
    with patch('openai.AsyncOpenAI') as mock_client:
        # Mock API response
        mock_choice = type('Choice', (), {
            'message': type('Message', (), {
                'content': '{"messages": [{"role": "sent", "content": "test", "timestamp": 1000}], "reasoning": "test"}'
            })()
        })()
        mock_response = type('Response', (), {'choices': [mock_choice]})()
        mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
        
        generator = GPTGenerator(api_key="test_key")
        prompt = create_grooming_prompt(RiskLevel.LOW, 14, 5)
        result = await generator.generate(prompt)
        
        assert result.success
        assert result.conversation is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/generation/test_generators.py -k gpt -v`
Expected: FAIL with "cannot import name 'GPTGenerator'"

- [ ] **Step 3: Implement GPT generator**

Create `data/generation/generators/gpt_generator.py`:

```python
"""OpenAI GPT-based conversation generator."""

import os
from typing import Optional
import openai
from data.generation.generators.base import ConversationGenerator, GenerationResult
from data.generation.prompts.base import ConversationPrompt


class GPTGenerator(ConversationGenerator):
    """Generate conversations using OpenAI GPT API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.9,
        max_tokens: int = 2000
    ):
        """Initialize GPT generator.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: GPT model to use
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = openai.AsyncOpenAI(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "openai"

    async def generate(self, prompt: ConversationPrompt) -> GenerationResult:
        """Generate conversation using OpenAI API."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": prompt.system_prompt},
                    {"role": "user", "content": prompt.user_prompt}
                ],
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            # Extract text from response
            response_text = response.choices[0].message.content
            
            # Parse JSON conversation
            conversation = self._parse_json_response(response_text)
            
            if conversation is None:
                return GenerationResult(
                    success=False,
                    conversation=None,
                    raw_response=response_text,
                    error="Failed to parse JSON from response"
                )
            
            # Validate required fields
            if "messages" not in conversation:
                return GenerationResult(
                    success=False,
                    conversation=None,
                    raw_response=response_text,
                    error="Response missing 'messages' field"
                )
            
            return GenerationResult(
                success=True,
                conversation=conversation,
                raw_response=response_text,
                error=None
            )
            
        except Exception as e:
            return GenerationResult(
                success=False,
                conversation=None,
                raw_response="",
                error=f"API Error: {str(e)}"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/generation/test_generators.py -k gpt -v`
Expected: All GPT tests PASS

- [ ] **Step 5: Commit GPT generator**

```bash
git add data/generation/generators/gpt_generator.py tests/data/generation/test_generators.py
git commit -m "feat: add OpenAI GPT conversation generator"
```

---

## Task 12: Quality Validators

**Files:**
- Create: `data/generation/validators/quality.py`
- Modify: `tests/data/generation/test_validators.py`

- [ ] **Step 1: Write tests for quality validators**

Add to `tests/data/generation/test_validators.py`:

```python
from data.generation.validators.quality import (
    validate_conversation_length,
    validate_vocabulary_diversity,
    validate_timestamp_progression,
    validate_conversation_quality
)
from data.generation.validators.schemas import Message


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
        Message(role="sent", content="hey hows it going", timestamp=1000),
        Message(role="received", content="pretty good just playing some minecraft", timestamp=1005),
        Message(role="sent", content="nice what server", timestamp=1020),
        Message(role="received", content="just a private one with friends", timestamp=1025)
    ]
    is_valid, errors = validate_conversation_quality(messages)
    assert is_valid
    assert len(errors) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/data/generation/test_validators.py -k quality -v`
Expected: FAIL with "cannot import name 'validate_conversation_length'"

- [ ] **Step 3: Implement quality validators**

Create `data/generation/validators/quality.py`:

```python
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
    """Validate timestamps progress forward in time.
    
    Args:
        messages: List of conversation messages
    
    Returns:
        (is_valid, error_message)
    """
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/generation/test_validators.py -k quality -v`
Expected: All quality validation tests PASS

- [ ] **Step 5: Commit quality validators**

```bash
git add data/generation/validators/quality.py tests/data/generation/test_validators.py
git commit -m "feat: add quality validation checks"
```

---

## Task 13: Main Generation Script

**Files:**
- Create: `data/generation/scripts/generate.py`
- Create: `.env` (gitignored, for local API keys)

- [ ] **Step 1: Create environment file for API keys**

Create `.env` (add to .gitignore if not already):

```bash
# Copy from .env.example and add your actual keys
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
DEFAULT_GENERATOR=claude
```

Run: `echo ".env" >> .gitignore` (if not already there)

- [ ] **Step 2: Implement main generation script**

Create `data/generation/scripts/generate.py`:

```python
#!/usr/bin/env python3
"""Main script for generating synthetic conversations."""

import asyncio
import argparse
import os
import sys
from pathlib import Path
from typing import List
import yaml
import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.generation.validators.schemas import (
    RiskCategory,
    RiskLevel,
    SyntheticConversation,
    ConversationLabel,
    Message
)
from data.generation.generators.claude_generator import ClaudeGenerator
from data.generation.generators.gpt_generator import GPTGenerator
from data.generation.validators.quality import validate_conversation_quality

# Import all prompt creators
from data.generation.prompts.grooming import create_grooming_prompt
from data.generation.prompts.bullying import create_bullying_prompt
from data.generation.prompts.sexual_content import create_sexual_content_prompt
from data.generation.prompts.isolation import create_isolation_prompt
from data.generation.prompts.personal_info import create_personal_info_prompt
from data.generation.prompts.platform_migration import create_platform_migration_prompt
from data.generation.prompts.threats import create_threats_prompt
from data.generation.prompts.benign import create_benign_prompt


# Map categories to prompt creators
PROMPT_CREATORS = {
    RiskCategory.GROOMING: create_grooming_prompt,
    RiskCategory.BULLYING: create_bullying_prompt,
    RiskCategory.SEXUAL_CONTENT: create_sexual_content_prompt,
    RiskCategory.ISOLATION: create_isolation_prompt,
    RiskCategory.PERSONAL_INFO: create_personal_info_prompt,
    RiskCategory.PLATFORM_MIGRATION: create_platform_migration_prompt,
    RiskCategory.THREATS: create_threats_prompt,
    RiskCategory.BENIGN: create_benign_prompt,
}


def load_config(config_path: str = "data/generation/config.yaml") -> dict:
    """Load generation configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_generator(generator_type: str, config: dict):
    """Initialize the appropriate generator."""
    if generator_type == "claude":
        return ClaudeGenerator(
            model=config["generation"]["model_claude"],
            temperature=config["generation"]["temperature"],
            max_tokens=config["generation"]["max_tokens"]
        )
    elif generator_type == "openai":
        return GPTGenerator(
            model=config["generation"]["model_openai"],
            temperature=config["generation"]["temperature"],
            max_tokens=config["generation"]["max_tokens"]
        )
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


def create_prompt_for_category(category: RiskCategory, severity: RiskLevel, config: dict):
    """Create prompt for given category and severity."""
    # Random child age (13-17)
    child_age = random.randint(13, 17)
    
    # Random message count (varied for diversity)
    num_messages = random.randint(5, 15)
    
    prompt_creator = PROMPT_CREATORS[category]
    
    # Benign conversations don't have severity
    if category == RiskCategory.BENIGN:
        return prompt_creator(child_age=child_age, num_messages=num_messages)
    else:
        return prompt_creator(severity=severity, child_age=child_age, num_messages=num_messages)


def convert_to_synthetic_conversation(
    conversation_data: dict,
    category: RiskCategory,
    severity: RiskLevel,
    conversation_id: str,
    generator_name: str
) -> SyntheticConversation:
    """Convert generated conversation data to SyntheticConversation object."""
    # Extract messages
    messages = [
        Message(**msg_data) for msg_data in conversation_data["messages"]
    ]
    
    # Create label
    reasoning = conversation_data.get("reasoning", "Generated conversation")
    
    # Determine severity score based on level
    severity_scores = {
        RiskLevel.NONE: 0.0,
        RiskLevel.LOW: 0.3,
        RiskLevel.MEDIUM: 0.5,
        RiskLevel.HIGH: 0.75,
        RiskLevel.CRITICAL: 0.95
    }
    severity_score = severity_scores.get(severity, 0.5)
    
    # Determine categories
    if category == RiskCategory.BENIGN:
        categories = [RiskCategory.BENIGN]
    else:
        categories = [category]
    
    label = ConversationLabel(
        risk_level=severity,
        categories=categories,
        severity_score=severity_score,
        reasoning=reasoning
    )
    
    return SyntheticConversation(
        conversation_id=conversation_id,
        category=category,
        messages=messages,
        label=label,
        metadata={
            "generator": generator_name,
            "prompt_version": "1.0"
        }
    )


async def generate_conversations(
    category: RiskCategory,
    count: int,
    generator,
    config: dict,
    output_file: Path
) -> tuple[int, int]:
    """Generate conversations for a specific category.
    
    Returns:
        (successful_count, failed_count)
    """
    successful = 0
    failed = 0
    
    # Determine severity distribution
    if category == RiskCategory.BENIGN:
        severities = [RiskLevel.NONE] * count
    else:
        severity_dist = config["severity_distribution"]
        severities = (
            [RiskLevel.LOW] * int(count * severity_dist["low"]) +
            [RiskLevel.MEDIUM] * int(count * severity_dist["medium"]) +
            [RiskLevel.HIGH] * int(count * severity_dist["high"]) +
            [RiskLevel.CRITICAL] * int(count * severity_dist["critical"])
        )
        # Fill to exact count
        while len(severities) < count:
            severities.append(random.choice([RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]))
        severities = severities[:count]
        random.shuffle(severities)
    
    # Open output file in append mode
    with jsonlines.open(output_file, mode='a') as writer:
        for i, severity in enumerate(tqdm(severities, desc=f"{category.value}")):
            conversation_id = f"{category.value}_{severity.value}_{i:04d}"
            
            try:
                # Create prompt
                prompt = create_prompt_for_category(category, severity, config)
                
                # Generate conversation
                result = await generator.generate(prompt)
                
                if not result.success:
                    print(f"\nGeneration failed for {conversation_id}: {result.error}")
                    failed += 1
                    continue
                
                # Validate quality
                messages = [Message(**msg) for msg in result.conversation["messages"]]
                is_valid, errors = validate_conversation_quality(
                    messages,
                    min_length=config["quality"]["min_conversation_length"],
                    max_length=config["quality"]["max_conversation_length"],
                    min_unique_tokens=config["quality"]["min_unique_tokens"]
                )
                
                if not is_valid:
                    print(f"\nQuality check failed for {conversation_id}: {errors}")
                    failed += 1
                    continue
                
                # Convert to SyntheticConversation
                conversation = convert_to_synthetic_conversation(
                    result.conversation,
                    category,
                    severity,
                    conversation_id,
                    generator.name
                )
                
                # Write to JSONL
                writer.write(conversation.model_dump())
                successful += 1
                
            except Exception as e:
                print(f"\nError generating {conversation_id}: {e}")
                failed += 1
                continue
    
    return successful, failed


async def main():
    """Main generation orchestration."""
    parser = argparse.ArgumentParser(description="Generate synthetic conversations")
    parser.add_argument(
        "--category",
        type=str,
        choices=[c.value for c in RiskCategory],
        help="Generate only this category"
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Override count for category"
    )
    parser.add_argument(
        "--generator",
        type=str,
        choices=["claude", "openai"],
        help="Which generator to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/conversations.jsonl",
        help="Output file path"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data/generation/config.yaml",
        help="Config file path"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize generator
    generator_type = args.generator or os.getenv("DEFAULT_GENERATOR", "claude")
    generator = get_generator(generator_type, config)
    
    # Prepare output directory
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine what to generate
    if args.category:
        # Single category
        category = RiskCategory(args.category)
        count = args.count or config["dataset_targets"][args.category]
        categories_to_generate = [(category, count)]
    else:
        # All categories
        categories_to_generate = [
            (RiskCategory(cat), count)
            for cat, count in config["dataset_targets"].items()
        ]
    
    # Generate conversations
    total_successful = 0
    total_failed = 0
    
    for category, count in categories_to_generate:
        print(f"\n{'='*60}")
        print(f"Generating {count} conversations for {category.value}")
        print(f"{'='*60}")
        
        successful, failed = await generate_conversations(
            category, count, generator, config, output_file
        )
        
        total_successful += successful
        total_failed += failed
        
        print(f"  ✓ Success: {successful}")
        print(f"  ✗ Failed: {failed}")
    
    print(f"\n{'='*60}")
    print(f"Generation Complete")
    print(f"{'='*60}")
    print(f"Total successful: {total_successful}")
    print(f"Total failed: {total_failed}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Make script executable**

Run: `chmod +x data/generation/scripts/generate.py`

- [ ] **Step 4: Test script help text**

Run: `python data/generation/scripts/generate.py --help`
Expected: Shows help text with all arguments

- [ ] **Step 5: Commit generation script**

```bash
git add data/generation/scripts/generate.py .gitignore
git commit -m "feat: add main conversation generation script"
```

---

## Task 14: Makefile Targets for Data Generation

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add data generation targets to Makefile**

Add to `Makefile`:

```makefile
# Data Generation
.PHONY: generate-data generate-all validate-data clean-data

generate-data:
	@echo "Generating $(COUNT) $(CATEGORY) conversations..."
	python data/generation/scripts/generate.py \
		--category $(CATEGORY) \
		--count $(COUNT) \
		--generator $(or $(GENERATOR),claude)

generate-all:
	@echo "Generating complete dataset..."
	python data/generation/scripts/generate.py

validate-data:
	@echo "Validating generated conversations..."
	python data/generation/scripts/stats.py --validate

stats:
	@echo "Generating dataset statistics..."
	python data/generation/scripts/stats.py

clean-data:
	@echo "Cleaning generated data..."
	rm -rf data/raw/*.jsonl
	@echo "Data cleaned."

# Quick test generation (100 samples)
generate-test:
	@echo "Generating test dataset (100 samples)..."
	python data/generation/scripts/generate.py --count 100 --category benign
```

- [ ] **Step 2: Test makefile target**

Run: `make generate-data CATEGORY=benign COUNT=5 GENERATOR=claude`
Expected: Should attempt to run generation script (may fail if no API key)

- [ ] **Step 3: Commit Makefile updates**

```bash
git add Makefile
git commit -m "feat: add data generation make targets"
```

---

## Task 15: Dataset Statistics & Reporting

**Files:**
- Create: `data/generation/scripts/stats.py`

- [ ] **Step 1: Implement statistics script**

Create `data/generation/scripts/stats.py`:

```python
#!/usr/bin/env python3
"""Dataset statistics and validation reporting."""

import argparse
import sys
from pathlib import Path
from collections import defaultdict, Counter
import jsonlines
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.generation.validators.schemas import SyntheticConversation


def load_conversations(file_path: Path) -> List[SyntheticConversation]:
    """Load conversations from JSONL file."""
    conversations = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            try:
                conv = SyntheticConversation(**obj)
                conversations.append(conv)
            except Exception as e:
                print(f"Warning: Failed to parse conversation: {e}")
    return conversations


def generate_statistics(conversations: List[SyntheticConversation]) -> Dict:
    """Generate comprehensive dataset statistics."""
    stats = {
        "total_conversations": len(conversations),
        "by_category": defaultdict(int),
        "by_severity": defaultdict(int),
        "by_generator": defaultdict(int),
        "message_lengths": [],
        "conversation_lengths": [],
        "vocabulary_sizes": []
    }
    
    for conv in conversations:
        # Category counts
        stats["by_category"][conv.category.value] += 1
        
        # Severity counts
        stats["by_severity"][conv.label.risk_level.value] += 1
        
        # Generator counts
        generator = conv.metadata.get("generator", "unknown")
        stats["by_generator"][generator] += 1
        
        # Message statistics
        num_messages = len(conv.messages)
        stats["conversation_lengths"].append(num_messages)
        
        # Calculate vocabulary
        all_text = " ".join(msg.content.lower() for msg in conv.messages)
        unique_tokens = len(set(all_text.split()))
        stats["vocabulary_sizes"].append(unique_tokens)
        
        # Individual message lengths
        for msg in conv.messages:
            stats["message_lengths"].append(len(msg.content))
    
    return stats


def print_statistics(stats: Dict):
    """Print formatted statistics report."""
    print("\n" + "="*60)
    print("DATASET STATISTICS REPORT")
    print("="*60)
    
    print(f"\nTotal Conversations: {stats['total_conversations']}")
    
    print("\n📊 By Category:")
    for category, count in sorted(stats["by_category"].items()):
        pct = (count / stats['total_conversations']) * 100
        print(f"  {category:25s}: {count:5d} ({pct:5.1f}%)")
    
    print("\n📊 By Severity:")
    for severity, count in sorted(stats["by_severity"].items()):
        pct = (count / stats['total_conversations']) * 100
        print(f"  {severity:25s}: {count:5d} ({pct:5.1f}%)")
    
    print("\n📊 By Generator:")
    for generator, count in sorted(stats["by_generator"].items()):
        pct = (count / stats['total_conversations']) * 100
        print(f"  {generator:25s}: {count:5d} ({pct:5.1f}%)")
    
    if stats["conversation_lengths"]:
        conv_lengths = stats["conversation_lengths"]
        print("\n📏 Conversation Lengths:")
        print(f"  Min: {min(conv_lengths)} messages")
        print(f"  Max: {max(conv_lengths)} messages")
        print(f"  Avg: {sum(conv_lengths)/len(conv_lengths):.1f} messages")
    
    if stats["vocabulary_sizes"]:
        vocab = stats["vocabulary_sizes"]
        print("\n📚 Vocabulary Diversity:")
        print(f"  Min unique tokens: {min(vocab)}")
        print(f"  Max unique tokens: {max(vocab)}")
        print(f"  Avg unique tokens: {sum(vocab)/len(vocab):.1f}")
    
    if stats["message_lengths"]:
        msg_lengths = stats["message_lengths"]
        print("\n💬 Message Lengths:")
        print(f"  Min: {min(msg_lengths)} chars")
        print(f"  Max: {max(msg_lengths)} chars")
        print(f"  Avg: {sum(msg_lengths)/len(msg_lengths):.1f} chars")
    
    print("\n" + "="*60)


def validate_dataset(conversations: List[SyntheticConversation]) -> bool:
    """Validate dataset meets quality requirements."""
    print("\n" + "="*60)
    print("DATASET VALIDATION")
    print("="*60)
    
    errors = []
    warnings = []
    
    # Check minimum size
    if len(conversations) < 1000:
        warnings.append(f"Dataset size ({len(conversations)}) below recommended minimum (1000)")
    
    # Check category balance
    category_counts = Counter(conv.category.value for conv in conversations)
    if len(category_counts) < 8:
        errors.append(f"Missing categories. Found: {list(category_counts.keys())}")
    
    # Check for empty conversations
    empty_convs = [conv for conv in conversations if len(conv.messages) < 2]
    if empty_convs:
        errors.append(f"Found {len(empty_convs)} conversations with < 2 messages")
    
    # Print results
    if errors:
        print("\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not errors and not warnings:
        print("\n✅ All validation checks passed!")
    
    print("\n" + "="*60)
    
    return len(errors) == 0


def main():
    """Main statistics and validation script."""
    parser = argparse.ArgumentParser(description="Dataset statistics and validation")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/conversations.jsonl",
        help="Input JSONL file"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks"
    )
    
    args = parser.parse_args()
    
    # Load conversations
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"Loading conversations from {input_file}...")
    conversations = load_conversations(input_file)
    
    if not conversations:
        print("Error: No conversations loaded")
        sys.exit(1)
    
    # Generate and print statistics
    stats = generate_statistics(conversations)
    print_statistics(stats)
    
    # Run validation if requested
    if args.validate:
        is_valid = validate_dataset(conversations)
        sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make stats script executable**

Run: `chmod +x data/generation/scripts/stats.py`

- [ ] **Step 3: Test stats script (will show error since no data yet, that's expected)**

Run: `python data/generation/scripts/stats.py`
Expected: Error message about missing data file (normal - no data generated yet)

- [ ] **Step 4: Commit stats script**

```bash
git add data/generation/scripts/stats.py
git commit -m "feat: add dataset statistics and validation script"
```

---

## Final Summary & Next Steps

**Phase 1 Implementation Complete! 🎉**

**What we built:**
- Complete data generation pipeline with prompt templates for all 7 risk categories + benign
- Claude and OpenAI generator implementations
- Pydantic schemas for data validation
- Quality validators (length, diversity, timestamps)
- Main generation script with async support
- Statistics and reporting tools
- Makefile targets for easy usage

**Repository structure created:**
```
data/generation/
├── prompts/          # ✓ All 8 categories
├── generators/       # ✓ Claude + GPT
├── validators/       # ✓ Schemas + quality checks
└── scripts/          # ✓ Generation + stats
```

**Ready to use:**
```bash
# Generate 1000 benign conversations
make generate-data CATEGORY=benign COUNT=1000

# Generate full dataset (10K total)
make generate-all

# View statistics
make stats

# Validate quality
make validate-data
```

**Next Phase:** Training Infrastructure (Weeks 3-4)

