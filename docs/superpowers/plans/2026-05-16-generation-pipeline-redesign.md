# Generation Pipeline Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace unconstrained JSON generation with vLLM constrained decoding (xgrammar) + simplified per-category prompts with persona seeds and few-shot examples to eliminate the ~100% benign failure rate and improve all category success rates.

**Architecture:** vLLM's `response_format` with a flat `ConversationOut` JSON Schema guarantees structurally valid output from every request. Each category gets a self-contained prompt function with one baked-in few-shot example and a randomised persona seed (name, platform, relationship). The pipeline assembles labels after generation rather than relying on the model for metadata.

**Tech Stack:** Python 3.13, vLLM (xgrammar constrained decoding), httpx async, pydantic v2, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `data/generation/prompts/base.py` | Modify | Shared system prompt string + `ConversationPrompt` dataclass only |
| `data/generation/prompts/benign.py` | Rewrite | Self-contained benign prompt with persona seed + few-shot example |
| `data/generation/prompts/grooming.py` | Rewrite | Self-contained grooming prompt with persona seed + few-shot example |
| `data/generation/prompts/bullying.py` | Rewrite | Self-contained bullying prompt with persona seed + few-shot example |
| `data/generation/prompts/sexual_content.py` | Rewrite | Self-contained sexual_content prompt with persona seed + few-shot example |
| `data/generation/prompts/isolation.py` | Rewrite | Self-contained isolation prompt with persona seed + few-shot example |
| `data/generation/prompts/personal_info.py` | Rewrite | Self-contained personal_info prompt with persona seed + few-shot example |
| `data/generation/prompts/platform_migration.py` | Rewrite | Self-contained platform_migration prompt with persona seed + few-shot example |
| `data/generation/prompts/threats.py` | Rewrite | Self-contained threats prompt with persona seed + few-shot example |
| `data/generation/generators/vllm_generator.py` | Modify | Add `response_format` + `max_tokens=512` to every request |
| `data/generation/scripts/generate.py` | Modify | Route all 8 categories to their prompt function; remove JSON parse retry |
| `data/generation/validators/quality.py` | Modify | Remove JSON parse checks; keep message count + role validity only |
| `tests/data/generation/test_prompts.py` | Rewrite | Update all prompt tests to match new signatures and content |
| `tests/data/generation/test_validators.py` | Modify | Update quality validator tests to match stripped-down checks |

---

## Task 1: Strip base.py to shared utilities only

**Files:**
- Modify: `data/generation/prompts/base.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/data/generation/test_prompts.py` (replace file contents entirely — the existing tests will be rewritten):

```python
import pytest
import json
from data.generation.prompts.base import (
    ConversationPrompt,
    format_system_prompt,
)
from data.generation.validators.schemas import RiskCategory, RiskLevel


def test_system_prompt_is_concise():
    prompt = format_system_prompt()
    assert "synthetic" in prompt.lower()
    assert "JSON" in prompt
    assert len(prompt) < 300  # must stay short for 7B model


def test_system_prompt_no_risk_indicators_mention():
    prompt = format_system_prompt()
    assert "risk indicator" not in prompt.lower()


def test_conversation_prompt_dataclass():
    p = ConversationPrompt(
        category=RiskCategory.BENIGN,
        severity=RiskLevel.NONE,
        system_prompt="sys",
        user_prompt="user",
        metadata={},
    )
    assert p.category == RiskCategory.BENIGN
    assert p.system_prompt == "sys"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/tomas/Documents/Github/SafeCircle/horizon
python -m pytest tests/data/generation/test_prompts.py::test_system_prompt_is_concise tests/data/generation/test_prompts.py::test_system_prompt_no_risk_indicators_mention tests/data/generation/test_prompts.py::test_conversation_prompt_dataclass -v
```

Expected: at least `test_system_prompt_is_concise` and `test_system_prompt_no_risk_indicators_mention` FAIL (current prompt is long and mentions risk indicators).

- [ ] **Step 3: Rewrite base.py**

Replace `data/generation/prompts/base.py` entirely:

```python
"""Shared prompt utilities for conversation generation."""

from dataclasses import dataclass
from typing import Any
from data.generation.validators.schemas import RiskCategory, RiskLevel


@dataclass
class ConversationPrompt:
    category: RiskCategory
    severity: RiskLevel
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any]


def format_system_prompt() -> str:
    return (
        "You are a synthetic chat data generator for child safety research. "
        "Generate realistic online conversations between teenagers (13-17). "
        "Output only valid JSON. No explanations outside the JSON."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/data/generation/test_prompts.py::test_system_prompt_is_concise tests/data/generation/test_prompts.py::test_system_prompt_no_risk_indicators_mention tests/data/generation/test_prompts.py::test_conversation_prompt_dataclass -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add data/generation/prompts/base.py tests/data/generation/test_prompts.py
git commit -m "refactor: strip base.py to shared utilities only"
```

---

## Task 2: Define shared persona seed helper

**Files:**
- Modify: `data/generation/prompts/base.py`
- Modify: `tests/data/generation/test_prompts.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/data/generation/test_prompts.py`:

```python
from data.generation.prompts.base import make_persona_seed


def test_persona_seed_returns_string():
    seed = make_persona_seed()
    assert isinstance(seed, str)
    assert len(seed) > 10


def test_persona_seed_contains_platform():
    seed = make_persona_seed()
    platforms = ["Discord", "Instagram", "Snapchat", "WhatsApp", "TikTok"]
    assert any(p in seed for p in platforms)


def test_persona_seed_varies():
    seeds = {make_persona_seed() for _ in range(20)}
    assert len(seeds) > 5  # must produce variety
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/data/generation/test_prompts.py::test_persona_seed_returns_string tests/data/generation/test_prompts.py::test_persona_seed_contains_platform tests/data/generation/test_prompts.py::test_persona_seed_varies -v
```

Expected: FAIL with ImportError on `make_persona_seed`.

- [ ] **Step 3: Implement make_persona_seed**

Append to `data/generation/prompts/base.py`:

```python
import random

_NAMES = [
    "Alex", "Jordan", "Tyler", "Morgan", "Casey", "Sam", "Riley", "Jamie",
    "Taylor", "Avery", "Blake", "Drew", "Quinn", "Skyler", "Reese", "Harper",
    "Peyton", "Logan", "Hayden", "Mackenzie",
]

_PLATFORMS = ["Discord", "Instagram", "Snapchat", "WhatsApp", "TikTok"]

_RELATIONSHIPS = [
    "a classmate", "a stranger from a gaming server", "an online friend",
    "someone from a fan community", "a friend of a friend",
]


def make_persona_seed() -> str:
    name = random.choice(_NAMES)
    platform = random.choice(_PLATFORMS)
    relationship = random.choice(_RELATIONSHIPS)
    age = random.randint(13, 17)
    return (
        f"Child's name: {name}, age {age}. "
        f"Platform: {platform}. "
        f"Other party is {relationship}."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/data/generation/test_prompts.py::test_persona_seed_returns_string tests/data/generation/test_prompts.py::test_persona_seed_contains_platform tests/data/generation/test_prompts.py::test_persona_seed_varies -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add data/generation/prompts/base.py tests/data/generation/test_prompts.py
git commit -m "feat: add make_persona_seed helper for conversation grounding"
```

---

## Task 3: Rewrite benign prompt

**Files:**
- Rewrite: `data/generation/prompts/benign.py`
- Modify: `tests/data/generation/test_prompts.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/generation/test_prompts.py`:

```python
from data.generation.prompts.benign import create_benign_prompt


def test_benign_prompt_category_and_severity():
    p = create_benign_prompt(child_age=14, num_messages=8)
    assert p.category == RiskCategory.BENIGN
    assert p.severity == RiskLevel.NONE


def test_benign_prompt_no_risk_language():
    p = create_benign_prompt(child_age=15, num_messages=8)
    combined = (p.system_prompt + p.user_prompt).lower()
    for word in ["risk", "severity", "safety alert", "grooming", "threat", "exploit"]:
        assert word not in combined, f"Benign prompt must not contain '{word}'"


def test_benign_prompt_contains_few_shot():
    p = create_benign_prompt(child_age=14, num_messages=8)
    assert '"messages"' in p.user_prompt
    assert '"role"' in p.user_prompt


def test_benign_prompt_contains_persona():
    p = create_benign_prompt(child_age=14, num_messages=8)
    platforms = ["Discord", "Instagram", "Snapchat", "WhatsApp", "TikTok"]
    assert any(pl in p.user_prompt for pl in platforms)


def test_benign_prompt_min_messages_8():
    p = create_benign_prompt(child_age=14, num_messages=8)
    assert "8" in p.user_prompt or "eight" in p.user_prompt.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/data/generation/test_prompts.py::test_benign_prompt_category_and_severity tests/data/generation/test_prompts.py::test_benign_prompt_no_risk_language tests/data/generation/test_prompts.py::test_benign_prompt_contains_few_shot tests/data/generation/test_prompts.py::test_benign_prompt_contains_persona tests/data/generation/test_prompts.py::test_benign_prompt_min_messages_8 -v
```

Expected: `test_benign_prompt_no_risk_language` and `test_benign_prompt_contains_few_shot` FAIL.

- [ ] **Step 3: Rewrite benign.py**

Replace `data/generation/prompts/benign.py` entirely:

```python
"""Benign conversation prompt — safe peer-to-peer conversations for false positive reduction."""

import json
from data.generation.prompts.base import ConversationPrompt, format_system_prompt, make_persona_seed
from data.generation.validators.schemas import RiskCategory, RiskLevel

_FEW_SHOT = json.dumps({
    "messages": [
        {"role": "sent", "content": "omg did you watch the new episode last night"},
        {"role": "received", "content": "YES i was not ready for that ending 😭"},
        {"role": "sent", "content": "right?? i literally screamed"},
        {"role": "received", "content": "who do u think is gonna win"},
        {"role": "sent", "content": "honestly idk but i really want maya to"},
        {"role": "received", "content": "same she's been through so much this season"},
        {"role": "sent", "content": "ok new theory: what if the finale is a flashback"},
        {"role": "received", "content": "ohhh that would actually make sense with episode 3"},
    ],
    "reasoning": "Casual peer discussion about a TV show, no risk indicators.",
}, indent=None, separators=(",", ":"))


def create_benign_prompt(child_age: int, num_messages: int) -> ConversationPrompt:
    persona = make_persona_seed()
    user_prompt = (
        f"{persona}\n\n"
        f"Write a completely normal conversation ({num_messages} messages, aim for a clear arc: opening, middle, wrap-up). "
        f"The topic should be everyday teen life — school, games, music, shows, sports, hobbies, or plans. "
        f"Use authentic teen texting style: abbreviations, emoji, casual grammar. "
        f"No drama, no danger, no adult asking personal questions.\n\n"
        f"Example:\n{_FEW_SHOT}\n\n"
        f"Now generate a new conversation. Change the topic, names, and style."
    )
    return ConversationPrompt(
        category=RiskCategory.BENIGN,
        severity=RiskLevel.NONE,
        system_prompt=format_system_prompt(),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages},
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/data/generation/test_prompts.py::test_benign_prompt_category_and_severity tests/data/generation/test_prompts.py::test_benign_prompt_no_risk_language tests/data/generation/test_prompts.py::test_benign_prompt_contains_few_shot tests/data/generation/test_prompts.py::test_benign_prompt_contains_persona tests/data/generation/test_prompts.py::test_benign_prompt_min_messages_8 -v
```

Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add data/generation/prompts/benign.py tests/data/generation/test_prompts.py
git commit -m "feat: rewrite benign prompt with few-shot example and persona seed"
```

---

## Task 4: Rewrite grooming prompt

**Files:**
- Rewrite: `data/generation/prompts/grooming.py`
- Modify: `tests/data/generation/test_prompts.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/generation/test_prompts.py`:

```python
from data.generation.prompts.grooming import create_grooming_prompt


def test_grooming_prompt_category():
    p = create_grooming_prompt(severity=RiskLevel.MEDIUM, child_age=14, num_messages=10)
    assert p.category == RiskCategory.GROOMING
    assert p.severity == RiskLevel.MEDIUM


def test_grooming_prompt_contains_severity_word():
    for sev, word in [
        (RiskLevel.LOW, "mild"),
        (RiskLevel.MEDIUM, "moderate"),
        (RiskLevel.HIGH, "severe"),
        (RiskLevel.CRITICAL, "extreme"),
    ]:
        p = create_grooming_prompt(severity=sev, child_age=14, num_messages=10)
        assert word in p.user_prompt.lower(), f"Expected '{word}' for {sev}"


def test_grooming_prompt_has_few_shot():
    p = create_grooming_prompt(severity=RiskLevel.MEDIUM, child_age=14, num_messages=10)
    assert '"messages"' in p.user_prompt


def test_grooming_prompt_has_persona():
    p = create_grooming_prompt(severity=RiskLevel.HIGH, child_age=15, num_messages=10)
    platforms = ["Discord", "Instagram", "Snapchat", "WhatsApp", "TikTok"]
    assert any(pl in p.user_prompt for pl in platforms)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/data/generation/test_prompts.py::test_grooming_prompt_category tests/data/generation/test_prompts.py::test_grooming_prompt_contains_severity_word tests/data/generation/test_prompts.py::test_grooming_prompt_has_few_shot tests/data/generation/test_prompts.py::test_grooming_prompt_has_persona -v
```

Expected: `test_grooming_prompt_contains_severity_word` and `test_grooming_prompt_has_few_shot` FAIL.

- [ ] **Step 3: Rewrite grooming.py**

Replace `data/generation/prompts/grooming.py` entirely:

```python
"""Grooming category prompt — adult building inappropriate relationship with a minor."""

import json
from data.generation.prompts.base import ConversationPrompt, format_system_prompt, make_persona_seed
from data.generation.validators.schemas import RiskCategory, RiskLevel

_SEVERITY_LABEL = {
    RiskLevel.LOW: "mild",
    RiskLevel.MEDIUM: "moderate",
    RiskLevel.HIGH: "severe",
    RiskLevel.CRITICAL: "extreme",
}

_FEW_SHOT = json.dumps({
    "messages": [
        {"role": "received", "content": "hey! i saw your comment on that gaming post, you seem really cool"},
        {"role": "sent", "content": "oh haha thanks"},
        {"role": "received", "content": "how old are you? you sound mature for your age"},
        {"role": "sent", "content": "im 15"},
        {"role": "received", "content": "wow really? you seem way older. i'm 19. do you game a lot?"},
        {"role": "sent", "content": "yeah pretty much every day after school"},
        {"role": "received", "content": "nice. you should add me on discord, we could play sometime. just us"},
        {"role": "sent", "content": "umm maybe"},
        {"role": "received", "content": "come on, i don't bite 😊 i just feel like we really get each other"},
    ],
    "reasoning": "Moderate grooming: age probing, flattery, establishing special connection, pushing for private contact.",
}, indent=None, separators=(",", ":"))


def create_grooming_prompt(severity: RiskLevel, child_age: int, num_messages: int) -> ConversationPrompt:
    label = _SEVERITY_LABEL.get(severity, "moderate")
    persona = make_persona_seed()
    user_prompt = (
        f"{persona}\n\n"
        f"Write a {label} grooming conversation ({num_messages} messages). "
        f"An older party builds an inappropriate relationship with the {child_age}-year-old: "
        f"flattery, testing boundaries, pushing for private contact. "
        f"Escalation should match the severity — {label}. "
        f"Use authentic teen texting style. The arc should be: initial contact, trust-building, boundary-pushing.\n\n"
        f"Example (moderate severity):\n{_FEW_SHOT}\n\n"
        f"Now generate a new conversation at {label} severity. Change names, platform context, and scenario."
    )
    return ConversationPrompt(
        category=RiskCategory.GROOMING,
        severity=severity,
        system_prompt=format_system_prompt(),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages},
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/data/generation/test_prompts.py::test_grooming_prompt_category tests/data/generation/test_prompts.py::test_grooming_prompt_contains_severity_word tests/data/generation/test_prompts.py::test_grooming_prompt_has_few_shot tests/data/generation/test_prompts.py::test_grooming_prompt_has_persona -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add data/generation/prompts/grooming.py tests/data/generation/test_prompts.py
git commit -m "feat: rewrite grooming prompt with few-shot example and persona seed"
```

---

## Task 5: Rewrite bullying, threats, isolation, personal_info, platform_migration, sexual_content prompts

**Files:**
- Rewrite: `data/generation/prompts/bullying.py`
- Rewrite: `data/generation/prompts/threats.py`
- Rewrite: `data/generation/prompts/isolation.py`
- Rewrite: `data/generation/prompts/personal_info.py`
- Rewrite: `data/generation/prompts/platform_migration.py`
- Rewrite: `data/generation/prompts/sexual_content.py`
- Modify: `tests/data/generation/test_prompts.py`

- [ ] **Step 1: Write failing tests for all 6 categories**

Append to `tests/data/generation/test_prompts.py`:

```python
from data.generation.prompts.bullying import create_bullying_prompt
from data.generation.prompts.threats import create_threats_prompt
from data.generation.prompts.isolation import create_isolation_prompt
from data.generation.prompts.personal_info import create_personal_info_prompt
from data.generation.prompts.platform_migration import create_platform_migration_prompt
from data.generation.prompts.sexual_content import create_sexual_content_prompt


@pytest.mark.parametrize("create_fn,category,severity", [
    (lambda: create_bullying_prompt(RiskLevel.HIGH, 14, 10), RiskCategory.BULLYING, RiskLevel.HIGH),
    (lambda: create_threats_prompt(RiskLevel.HIGH, 15, 10), RiskCategory.THREATS, RiskLevel.HIGH),
    (lambda: create_isolation_prompt(RiskLevel.MEDIUM, 14, 10), RiskCategory.ISOLATION, RiskLevel.MEDIUM),
    (lambda: create_personal_info_prompt(RiskLevel.MEDIUM, 15, 10), RiskCategory.PERSONAL_INFO, RiskLevel.MEDIUM),
    (lambda: create_platform_migration_prompt(RiskLevel.LOW, 14, 10), RiskCategory.PLATFORM_MIGRATION, RiskLevel.LOW),
    (lambda: create_sexual_content_prompt(RiskLevel.MEDIUM, 15, 10), RiskCategory.SEXUAL_CONTENT, RiskLevel.MEDIUM),
])
def test_category_prompt_metadata(create_fn, category, severity):
    p = create_fn()
    assert p.category == category
    assert p.severity == severity
    assert '"messages"' in p.user_prompt  # few-shot example present
    platforms = ["Discord", "Instagram", "Snapchat", "WhatsApp", "TikTok"]
    assert any(pl in p.user_prompt for pl in platforms)  # persona seed present


@pytest.mark.parametrize("create_fn,severity_word,severity", [
    (lambda s: create_bullying_prompt(s, 14, 10), {RiskLevel.LOW: "mild", RiskLevel.HIGH: "severe"}, RiskLevel.LOW),
    (lambda s: create_threats_prompt(s, 15, 10), {RiskLevel.LOW: "mild", RiskLevel.HIGH: "severe"}, RiskLevel.HIGH),
])
def test_severity_label_in_prompt(create_fn, severity_word, severity):
    p = create_fn(severity)
    assert severity_word[severity] in p.user_prompt.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/data/generation/test_prompts.py::test_category_prompt_metadata tests/data/generation/test_prompts.py::test_severity_label_in_prompt -v
```

Expected: FAIL (few-shot and persona not in current prompts).

- [ ] **Step 3: Rewrite bullying.py**

Replace `data/generation/prompts/bullying.py`:

```python
"""Bullying category prompt."""

import json
from data.generation.prompts.base import ConversationPrompt, format_system_prompt, make_persona_seed
from data.generation.validators.schemas import RiskCategory, RiskLevel

_SEVERITY_LABEL = {
    RiskLevel.LOW: "mild", RiskLevel.MEDIUM: "moderate",
    RiskLevel.HIGH: "severe", RiskLevel.CRITICAL: "extreme",
}

_FEW_SHOT = json.dumps({
    "messages": [
        {"role": "received", "content": "everyone thinks ur so weird for what u did at lunch lol"},
        {"role": "sent", "content": "what?? i didn't do anything"},
        {"role": "received", "content": "sure u didn't. mia took a video btw"},
        {"role": "sent", "content": "please don't send it around"},
        {"role": "received", "content": "too late already in the gc 💀"},
        {"role": "sent", "content": "why are you doing this"},
        {"role": "received", "content": "just stay away from our group ok"},
    ],
    "reasoning": "Moderate peer bullying: public humiliation, video threat, social exclusion.",
}, indent=None, separators=(",", ":"))


def create_bullying_prompt(severity: RiskLevel, child_age: int, num_messages: int) -> ConversationPrompt:
    label = _SEVERITY_LABEL.get(severity, "moderate")
    persona = make_persona_seed()
    user_prompt = (
        f"{persona}\n\n"
        f"Write a {label} cyberbullying conversation ({num_messages} messages). "
        f"A peer or group targets the {child_age}-year-old with insults, exclusion, or harassment. "
        f"Severity is {label}: match the intensity accordingly. "
        f"Arc: incident or trigger, escalation, impact on victim.\n\n"
        f"Example (moderate severity):\n{_FEW_SHOT}\n\n"
        f"Now generate a new conversation at {label} severity. Change names, scenario, and platform."
    )
    return ConversationPrompt(
        category=RiskCategory.BULLYING,
        severity=severity,
        system_prompt=format_system_prompt(),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages},
    )
```

- [ ] **Step 4: Rewrite threats.py**

Replace `data/generation/prompts/threats.py`:

```python
"""Threats/violence category prompt."""

import json
from data.generation.prompts.base import ConversationPrompt, format_system_prompt, make_persona_seed
from data.generation.validators.schemas import RiskCategory, RiskLevel

_SEVERITY_LABEL = {
    RiskLevel.LOW: "mild", RiskLevel.MEDIUM: "moderate",
    RiskLevel.HIGH: "severe", RiskLevel.CRITICAL: "extreme",
}

_FEW_SHOT = json.dumps({
    "messages": [
        {"role": "received", "content": "you better watch yourself after school"},
        {"role": "sent", "content": "what did i do?"},
        {"role": "received", "content": "you know what you did. don't play dumb"},
        {"role": "sent", "content": "i'm serious i don't know what you're talking about"},
        {"role": "received", "content": "ask your friends. they'll tell you what happens next"},
        {"role": "sent", "content": "please just leave me alone"},
        {"role": "received", "content": "too late for that"},
    ],
    "reasoning": "Moderate threat: implied physical threat after school, intimidation, refusal to stop.",
}, indent=None, separators=(",", ":"))


def create_threats_prompt(severity: RiskLevel, child_age: int, num_messages: int) -> ConversationPrompt:
    label = _SEVERITY_LABEL.get(severity, "moderate")
    persona = make_persona_seed()
    user_prompt = (
        f"{persona}\n\n"
        f"Write a {label} threatening conversation ({num_messages} messages). "
        f"One party threatens the {child_age}-year-old with harm, intimidation, or blackmail. "
        f"Severity is {label}: adjust explicitness of threats accordingly. "
        f"Arc: trigger or grievance, escalating threat, victim's reaction.\n\n"
        f"Example (moderate severity):\n{_FEW_SHOT}\n\n"
        f"Now generate a new conversation at {label} severity. Change context, names, and platform."
    )
    return ConversationPrompt(
        category=RiskCategory.THREATS,
        severity=severity,
        system_prompt=format_system_prompt(),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages},
    )
```

- [ ] **Step 5: Rewrite isolation.py**

Replace `data/generation/prompts/isolation.py`:

```python
"""Isolation/control category prompt."""

import json
from data.generation.prompts.base import ConversationPrompt, format_system_prompt, make_persona_seed
from data.generation.validators.schemas import RiskCategory, RiskLevel

_SEVERITY_LABEL = {
    RiskLevel.LOW: "mild", RiskLevel.MEDIUM: "moderate",
    RiskLevel.HIGH: "severe", RiskLevel.CRITICAL: "extreme",
}

_FEW_SHOT = json.dumps({
    "messages": [
        {"role": "received", "content": "who were you talking to just now"},
        {"role": "sent", "content": "just jake from school, we have a project"},
        {"role": "received", "content": "you don't need to talk to him. i don't like it"},
        {"role": "sent", "content": "it's just for school though"},
        {"role": "received", "content": "i'm the only one who really gets you. those people don't care about you"},
        {"role": "sent", "content": "that's not true they're my friends"},
        {"role": "received", "content": "they're not. trust me. just focus on us"},
    ],
    "reasoning": "Moderate isolation: jealousy, discouraging friendships, creating dependency.",
}, indent=None, separators=(",", ":"))


def create_isolation_prompt(severity: RiskLevel, child_age: int, num_messages: int) -> ConversationPrompt:
    label = _SEVERITY_LABEL.get(severity, "moderate")
    persona = make_persona_seed()
    user_prompt = (
        f"{persona}\n\n"
        f"Write a {label} isolation/control conversation ({num_messages} messages). "
        f"One party manipulates the {child_age}-year-old to cut off friends or family. "
        f"Severity is {label}: adjust the controlling behavior accordingly. "
        f"Arc: jealousy trigger, manipulation tactic, victim's conflicted response.\n\n"
        f"Example (moderate severity):\n{_FEW_SHOT}\n\n"
        f"Now generate a new conversation at {label} severity. Change the relationship dynamic and platform."
    )
    return ConversationPrompt(
        category=RiskCategory.ISOLATION,
        severity=severity,
        system_prompt=format_system_prompt(),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages},
    )
```

- [ ] **Step 6: Rewrite personal_info.py**

Replace `data/generation/prompts/personal_info.py`:

```python
"""Personal information solicitation category prompt."""

import json
from data.generation.prompts.base import ConversationPrompt, format_system_prompt, make_persona_seed
from data.generation.validators.schemas import RiskCategory, RiskLevel

_SEVERITY_LABEL = {
    RiskLevel.LOW: "mild", RiskLevel.MEDIUM: "moderate",
    RiskLevel.HIGH: "severe", RiskLevel.CRITICAL: "extreme",
}

_FEW_SHOT = json.dumps({
    "messages": [
        {"role": "received", "content": "hey so where do you live? like what city"},
        {"role": "sent", "content": "why do you wanna know lol"},
        {"role": "received", "content": "just curious, maybe we're close. i'm in austin"},
        {"role": "sent", "content": "i'm in dallas"},
        {"role": "received", "content": "nice! what school?"},
        {"role": "sent", "content": "lake view high, why"},
        {"role": "received", "content": "no reason just curious 😊 what area of dallas"},
    ],
    "reasoning": "Moderate personal info: city, school name extracted under guise of curiosity, pushing for more specific location.",
}, indent=None, separators=(",", ":"))


def create_personal_info_prompt(severity: RiskLevel, child_age: int, num_messages: int) -> ConversationPrompt:
    label = _SEVERITY_LABEL.get(severity, "moderate")
    persona = make_persona_seed()
    user_prompt = (
        f"{persona}\n\n"
        f"Write a {label} personal information solicitation conversation ({num_messages} messages). "
        f"One party extracts personal details from the {child_age}-year-old (location, school, address, schedule). "
        f"Severity is {label}: adjust how specific and persistent the questioning is. "
        f"Arc: casual opening, escalating questions, child's partial disclosure.\n\n"
        f"Example (moderate severity):\n{_FEW_SHOT}\n\n"
        f"Now generate a new conversation at {label} severity. Change names, platform, and information sought."
    )
    return ConversationPrompt(
        category=RiskCategory.PERSONAL_INFO,
        severity=severity,
        system_prompt=format_system_prompt(),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages},
    )
```

- [ ] **Step 7: Rewrite platform_migration.py**

Replace `data/generation/prompts/platform_migration.py`:

```python
"""Platform migration category prompt."""

import json
from data.generation.prompts.base import ConversationPrompt, format_system_prompt, make_persona_seed
from data.generation.validators.schemas import RiskCategory, RiskLevel

_SEVERITY_LABEL = {
    RiskLevel.LOW: "mild", RiskLevel.MEDIUM: "moderate",
    RiskLevel.HIGH: "severe", RiskLevel.CRITICAL: "extreme",
}

_FEW_SHOT = json.dumps({
    "messages": [
        {"role": "received", "content": "hey do you have snapchat?"},
        {"role": "sent", "content": "yeah why"},
        {"role": "received", "content": "i prefer it, dm me there. more private"},
        {"role": "sent", "content": "i usually just use discord"},
        {"role": "received", "content": "discord has too many people. snap is just us"},
        {"role": "sent", "content": "i guess i can add you"},
        {"role": "received", "content": "cool. also turn off your read receipts so no one knows we're talking"},
    ],
    "reasoning": "Moderate platform migration: pushing for private channel, discouraging visibility to others.",
}, indent=None, separators=(",", ":"))


def create_platform_migration_prompt(severity: RiskLevel, child_age: int, num_messages: int) -> ConversationPrompt:
    label = _SEVERITY_LABEL.get(severity, "moderate")
    persona = make_persona_seed()
    user_prompt = (
        f"{persona}\n\n"
        f"Write a {label} platform migration conversation ({num_messages} messages). "
        f"One party pressures the {child_age}-year-old to move to a less monitored platform (e.g. Telegram, Kik, Snapchat). "
        f"Severity is {label}: adjust how insistent and secretive the request is. "
        f"Arc: current platform context, push to migrate, child's hesitation or compliance.\n\n"
        f"Example (moderate severity):\n{_FEW_SHOT}\n\n"
        f"Now generate a new conversation at {label} severity. Change the platforms and reason given."
    )
    return ConversationPrompt(
        category=RiskCategory.PLATFORM_MIGRATION,
        severity=severity,
        system_prompt=format_system_prompt(),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages},
    )
```

- [ ] **Step 8: Rewrite sexual_content.py**

Replace `data/generation/prompts/sexual_content.py`:

```python
"""Sexual content category prompt — keeps content clinical, focused on pattern recognition."""

import json
from data.generation.prompts.base import ConversationPrompt, format_system_prompt, make_persona_seed
from data.generation.validators.schemas import RiskCategory, RiskLevel

_SEVERITY_LABEL = {
    RiskLevel.LOW: "mild", RiskLevel.MEDIUM: "moderate",
    RiskLevel.HIGH: "severe", RiskLevel.CRITICAL: "extreme",
}

_FEW_SHOT = json.dumps({
    "messages": [
        {"role": "received", "content": "you're honestly really pretty, has anyone told you that"},
        {"role": "sent", "content": "thanks i guess"},
        {"role": "received", "content": "like seriously. do you have more pics?"},
        {"role": "sent", "content": "i have instagram"},
        {"role": "received", "content": "i mean like personal ones. just for me"},
        {"role": "sent", "content": "um no that's weird"},
        {"role": "received", "content": "it's not weird, we're close. i just want to see you"},
    ],
    "reasoning": "Moderate sexual content: unsolicited compliments escalating to requests for private photos, normalising the request.",
}, indent=None, separators=(",", ":"))


def create_sexual_content_prompt(severity: RiskLevel, child_age: int, num_messages: int) -> ConversationPrompt:
    label = _SEVERITY_LABEL.get(severity, "moderate")
    persona = make_persona_seed()
    user_prompt = (
        f"{persona}\n\n"
        f"Write a {label} inappropriate sexual content conversation ({num_messages} messages). "
        f"One party makes sexually inappropriate advances toward the {child_age}-year-old. "
        f"Severity is {label}: keep language clinical and focused on behavioural patterns, not graphic detail. "
        f"Arc: seemingly innocent contact, escalating inappropriate comments, child's discomfort.\n\n"
        f"Example (moderate severity):\n{_FEW_SHOT}\n\n"
        f"Now generate a new conversation at {label} severity. Change the approach and platform."
    )
    return ConversationPrompt(
        category=RiskCategory.SEXUAL_CONTENT,
        severity=severity,
        system_prompt=format_system_prompt(),
        user_prompt=user_prompt,
        metadata={"child_age": child_age, "num_messages": num_messages},
    )
```

- [ ] **Step 9: Run all category prompt tests**

```bash
python -m pytest tests/data/generation/test_prompts.py::test_category_prompt_metadata tests/data/generation/test_prompts.py::test_severity_label_in_prompt -v
```

Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add data/generation/prompts/bullying.py data/generation/prompts/threats.py \
        data/generation/prompts/isolation.py data/generation/prompts/personal_info.py \
        data/generation/prompts/platform_migration.py data/generation/prompts/sexual_content.py \
        tests/data/generation/test_prompts.py
git commit -m "feat: rewrite all 6 risk category prompts with few-shot examples and persona seeds"
```

---

## Task 6: Add constrained decoding to VLLMGenerator

**Files:**
- Modify: `data/generation/generators/vllm_generator.py`
- Modify: `tests/data/generation/test_generators.py`

- [ ] **Step 1: Read the current test file to understand what exists**

```bash
cat tests/data/generation/test_generators.py
```

- [ ] **Step 2: Write the failing test**

Append to `tests/data/generation/test_generators.py`:

```python
import json
from unittest.mock import AsyncMock, MagicMock, patch
from data.generation.generators.vllm_generator import VLLMGenerator
from data.generation.prompts.base import ConversationPrompt
from data.generation.validators.schemas import RiskCategory, RiskLevel


def make_prompt():
    return ConversationPrompt(
        category=RiskCategory.BENIGN,
        severity=RiskLevel.NONE,
        system_prompt="sys",
        user_prompt="user",
        metadata={},
    )


@pytest.mark.asyncio
async def test_vllm_generator_sends_response_format():
    """VLLMGenerator must include response_format in every request payload."""
    gen = VLLMGenerator(model="test-model", base_url="http://localhost:8000/v1")

    captured = {}

    async def fake_post(url, json=None, **kwargs):
        captured["payload"] = json
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"messages":[{"role":"sent","content":"hi"},{"role":"received","content":"hey"}],"reasoning":"ok"}'}}]
        }
        return mock_resp

    with patch.object(gen._client, "post", side_effect=fake_post):
        await gen.generate(make_prompt())

    assert "response_format" in captured["payload"]
    assert captured["payload"]["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_vllm_generator_max_tokens_512():
    gen = VLLMGenerator(model="test-model", base_url="http://localhost:8000/v1")
    captured = {}

    async def fake_post(url, json=None, **kwargs):
        captured["payload"] = json
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"messages":[{"role":"sent","content":"hi"},{"role":"received","content":"hey"}],"reasoning":"ok"}'}}]
        }
        return mock_resp

    with patch.object(gen._client, "post", side_effect=fake_post):
        await gen.generate(make_prompt())

    assert captured["payload"]["max_tokens"] == 512
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
python -m pytest tests/data/generation/test_generators.py::test_vllm_generator_sends_response_format tests/data/generation/test_generators.py::test_vllm_generator_max_tokens_512 -v
```

Expected: FAIL (no `response_format` in current payload).

- [ ] **Step 4: Update VLLMGenerator**

Replace `data/generation/generators/vllm_generator.py` entirely:

```python
"""Local vLLM generator with constrained JSON decoding (xgrammar)."""

import os
from typing import Optional

import httpx
from pydantic import BaseModel
from typing import Literal

from data.generation.generators.base import ConversationGenerator, GenerationResult
from data.generation.prompts.base import ConversationPrompt


class _MessageOut(BaseModel):
    role: Literal["sent", "received"]
    content: str


class _ConversationOut(BaseModel):
    messages: list[_MessageOut]
    reasoning: str


_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "ConversationOut",
        "schema": _ConversationOut.model_json_schema(),
        "strict": True,
    },
}

_DEFAULT_BASE_URL = "http://localhost:8000/v1"
_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class VLLMGenerator(ConversationGenerator):
    """Generate conversations using a local vLLM server with constrained JSON decoding."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        base_url: Optional[str] = None,
        temperature: float = 0.9,
        max_tokens: int = 512,
        timeout: float = 120.0,
        max_connections: int = 200,
    ):
        self.model = model
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL", _DEFAULT_BASE_URL)).rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._endpoint = f"{self.base_url}/chat/completions"
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        )
        self._client = httpx.AsyncClient(timeout=timeout, limits=limits)

    @property
    def name(self) -> str:
        return "vllm"

    async def generate(self, prompt: ConversationPrompt) -> GenerationResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": _RESPONSE_FORMAT,
        }

        try:
            response = await self._client.post(self._endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            conversation = self._parse_json_response(text)

            if conversation is None:
                return GenerationResult(
                    success=False,
                    conversation=None,
                    raw_response=text,
                    error="Failed to parse JSON from response",
                )

            if "messages" not in conversation:
                return GenerationResult(
                    success=False,
                    conversation=None,
                    raw_response=text,
                    error="Response missing 'messages' field",
                )

            return GenerationResult(success=True, conversation=conversation, raw_response=text)

        except httpx.HTTPStatusError as e:
            return GenerationResult(
                success=False,
                conversation=None,
                raw_response=e.response.text,
                error=f"vLLM HTTP {e.response.status_code}: {e.response.text[:200]}",
            )
        except Exception as e:
            return GenerationResult(
                success=False,
                conversation=None,
                raw_response="",
                error=str(e),
            )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/data/generation/test_generators.py::test_vllm_generator_sends_response_format tests/data/generation/test_generators.py::test_vllm_generator_max_tokens_512 -v
```

Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add data/generation/generators/vllm_generator.py tests/data/generation/test_generators.py
git commit -m "feat: add constrained JSON decoding (xgrammar) to VLLMGenerator"
```

---

## Task 7: Route all categories in generate.py + strip JSON retry

**Files:**
- Modify: `data/generation/scripts/generate.py`

- [ ] **Step 1: Update generate.py**

In `data/generation/scripts/generate.py`, replace the prompt dispatch block and imports:

Find and replace the import section near the top (around line 41-43):

```python
# OLD — remove these two lines:
from data.generation.prompts.base import create_conversation_prompt
from data.generation.prompts.benign import create_benign_prompt
```

Replace with:

```python
from data.generation.prompts.benign import create_benign_prompt
from data.generation.prompts.bullying import create_bullying_prompt
from data.generation.prompts.grooming import create_grooming_prompt
from data.generation.prompts.isolation import create_isolation_prompt
from data.generation.prompts.personal_info import create_personal_info_prompt
from data.generation.prompts.platform_migration import create_platform_migration_prompt
from data.generation.prompts.sexual_content import create_sexual_content_prompt
from data.generation.prompts.threats import create_threats_prompt
```

Then replace the prompt dispatch block inside `generate_conversation` (currently lines ~167-177):

```python
# OLD — remove:
        if category == RiskCategory.BENIGN:
            prompt = create_benign_prompt(child_age=child_age, num_messages=num_messages)
        else:
            prompt = create_conversation_prompt(
                category=category,
                severity=severity,
                child_age=child_age,
                num_messages=num_messages,
            )
```

Replace with:

```python
        _PROMPT_BUILDERS = {
            RiskCategory.BENIGN: lambda: create_benign_prompt(child_age, num_messages),
            RiskCategory.BULLYING: lambda: create_bullying_prompt(severity, child_age, num_messages),
            RiskCategory.GROOMING: lambda: create_grooming_prompt(severity, child_age, num_messages),
            RiskCategory.ISOLATION: lambda: create_isolation_prompt(severity, child_age, num_messages),
            RiskCategory.PERSONAL_INFO: lambda: create_personal_info_prompt(severity, child_age, num_messages),
            RiskCategory.PLATFORM_MIGRATION: lambda: create_platform_migration_prompt(severity, child_age, num_messages),
            RiskCategory.SEXUAL_CONTENT: lambda: create_sexual_content_prompt(severity, child_age, num_messages),
            RiskCategory.THREATS: lambda: create_threats_prompt(severity, child_age, num_messages),
        }
        prompt = _PROMPT_BUILDERS[category]()
```

- [ ] **Step 2: Run the full prompt test suite to confirm nothing is broken**

```bash
python -m pytest tests/data/generation/test_prompts.py -v
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add data/generation/scripts/generate.py
git commit -m "feat: route all 8 categories to their own prompt builder in generate.py"
```

---

## Task 8: Strip quality validator to semantic-only checks

**Files:**
- Modify: `data/generation/validators/quality.py`
- Modify: `tests/data/generation/test_validators.py`

- [ ] **Step 1: Write new validator tests**

Add to `tests/data/generation/test_validators.py`:

```python
from data.generation.validators.quality import validate_conversation_quality


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
```

- [ ] **Step 2: Run tests to verify they fail or pass as expected**

```bash
python -m pytest tests/data/generation/test_validators.py::test_quality_passes_valid_conversation tests/data/generation/test_validators.py::test_quality_fails_too_few_messages tests/data/generation/test_validators.py::test_quality_fails_invalid_role -v
```

- [ ] **Step 3: Rewrite quality.py**

Replace `data/generation/validators/quality.py` entirely:

```python
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
```

- [ ] **Step 4: Run all validator tests**

```bash
python -m pytest tests/data/generation/test_validators.py -v
```

Expected: all PASS. (Some old tests that tested timestamp/vocabulary validation will now need to be removed — delete any tests for `validate_timestamp_progression` and `validate_vocabulary_diversity` since those functions no longer exist.)

- [ ] **Step 5: Commit**

```bash
git add data/generation/validators/quality.py tests/data/generation/test_validators.py
git commit -m "refactor: strip quality validator to message count + role checks only"
```

---

## Task 9: Full test suite + smoke test

- [ ] **Step 1: Run the full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS. Fix any import errors from removed functions (`create_conversation_prompt` is no longer exported from `base.py` — update any test that imports it).

- [ ] **Step 2: Smoke test the generation pipeline (single conversation per category)**

```bash
cd /home/tomas/Documents/Github/SafeCircle/horizon
python -c "
import asyncio
from data.generation.scripts.generate import create_generator, generate_conversation, load_config
from data.generation.validators.schemas import RiskCategory, RiskLevel

async def main():
    config = load_config('data/generation/config.yaml')
    gen = create_generator('vllm', config)
    for cat in RiskCategory:
        sev = RiskLevel.NONE if cat == RiskCategory.BENIGN else RiskLevel.MEDIUM
        result = await generate_conversation(gen, cat, sev, config, retry_attempts=1)
        status = 'OK' if result else 'FAIL'
        print(f'{cat.value:<22} {status}')

asyncio.run(main())
"
```

Expected: all 8 categories print `OK`.

- [ ] **Step 3: Commit final state and push**

```bash
git add -A
git commit -m "test: verify all 8 categories generate successfully with new pipeline"
git push origin main
```

---

## Task 10: Update generate_all.py concurrency config and run

- [ ] **Step 1: Verify config.yaml min_conversation_length matches new validator**

Open `data/generation/config.yaml` and ensure:

```yaml
quality:
  min_conversation_length: 8
  max_conversation_length: 40
```

- [ ] **Step 2: Restart the full generation run**

```bash
python -m data.generation.scripts.generate_all \
    --generator vllm \
    --count 200000 \
    --concurrency 150 \
    --resume
```

Watch for benign failure count — should be < 5% within the first 60 seconds. If any category still shows > 20% failure rate after 2 minutes, check `debug_benign.py` logic or run a single-category test run:

```bash
python -m data.generation.scripts.generate_all \
    --generator vllm \
    --count 100 \
    --concurrency 10 \
    --categories benign
```

- [ ] **Step 3: Commit any final config changes**

```bash
git add data/generation/config.yaml
git commit -m "config: update min_conversation_length to 8 to match new validator"
```
