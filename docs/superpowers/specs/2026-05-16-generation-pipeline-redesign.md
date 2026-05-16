# Generation Pipeline Redesign

**Date:** 2026-05-16
**Status:** Approved
**Context:** Benign category at 0 successful generations (~15k+ failures per run); threats/platform_migration also failing at high rates. Root cause: unconstrained JSON generation from Qwen2.5-7B-Instruct at high concurrency produces malformed or semantically wrong output.

---

## Problem

The current pipeline asks the model to produce JSON via a system prompt instruction and then manually parses whatever comes back. With a 7B model at concurrency 150:
- Malformed JSON is common (truncation, extra text, wrong structure)
- Overly verbose per-category prompts (6-8 bullet severity guides) overwhelm the 7B context
- Benign prompt passed through the generic base builder which injected "include risk indicators" framing
- All 8 categories use the same generic prompt builder; category-specific prompt files exist but most are never called

---

## Approach

Constrained decoding via vLLM's `response_format` with a flat JSON Schema (xgrammar backend). Combined with simplified per-category prompts with few-shot examples.

---

## Schema

The model only outputs:

```python
class MessageOut(BaseModel):
    role: Literal["sent", "received"]
    content: str

class ConversationOut(BaseModel):
    messages: list[MessageOut]   # 5–15 items
    reasoning: str               # ≤ 120 chars summary
```

Timestamps removed (7B model ignores them). Labels, metadata, and severity scores are assembled by the pipeline after generation, not produced by the model.

---

## Prompt Design

**System prompt** (shared across all categories):
```
You are a synthetic chat data generator for child safety research.
Generate realistic online conversations between teenagers (13-17).
Output only valid JSON. No explanations outside the JSON.
```

**User prompt per category:**
- 4–5 constraints maximum
- Severity expressed as a single adjective ("mild", "moderate", "severe") — no bullet-point severity guides
- One hardcoded few-shot example baked into each prompt
- Benign prompt never uses the words "risk", "safety", or "severity"
- Each category is fully self-contained — no shared `additional_context` injection
- Every prompt includes a **persona seed** (randomised name, platform — Discord/Instagram/Snapchat/WhatsApp, relationship type — classmate/stranger/online friend) so conversations feel grounded, not generic
- Minimum message count raised to **8** (from 5); prompts explicitly ask for a conversation with a clear arc: opening → development → resolution or escalation

---

## Pipeline Changes

| Component | Change |
|---|---|
| `vllm_generator.py` | Add `response_format` with `ConversationOut` JSON Schema; set `max_tokens=512` |
| `prompts/base.py` | Strip to shared utilities only (system prompt string, `ConversationPrompt` dataclass) |
| `prompts/{category}.py` × 8 | Rewrite each with simplified prompt + baked-in few-shot example |
| `generate.py` | Route all 8 categories to their own prompt function; remove JSON parse retry logic |
| `validators/quality.py` | Remove JSON parsing checks; keep only message count and role validity checks |

**No changes to:** `generate_all.py`, `schemas.py`, `write_conversations()`, resume logic, progress bars.

---

## Files Modified

- `data/generation/generators/vllm_generator.py`
- `data/generation/prompts/base.py`
- `data/generation/prompts/benign.py`
- `data/generation/prompts/grooming.py`
- `data/generation/prompts/bullying.py`
- `data/generation/prompts/sexual_content.py`
- `data/generation/prompts/isolation.py`
- `data/generation/prompts/personal_info.py`
- `data/generation/prompts/platform_migration.py`
- `data/generation/prompts/threats.py`
- `data/generation/scripts/generate.py`
- `data/generation/validators/quality.py`

---

## Key Constraints

- vLLM must be ≥ 0.4.0 for `response_format` with `json_schema` type (xgrammar default)
- Always pass `max_tokens=512` to mitigate Qwen2.5 non-termination bug with constrained decoding
- Schema must stay flat (≤ 2 nesting levels, ≤ 6 fields) for reliable 7B compliance
- One few-shot example per category prompt — do not add more (increases prompt length, reduces throughput)
- Persona seeds (name, platform, relationship) are randomised at call time in the prompt builder, not hardcoded

---

## Success Criteria

- Benign failure rate drops from ~100% to <5%
- All categories sustain ≥ 3 successful generations/sec at concurrency 150
- JSON parse failures → 0 (guaranteed by xgrammar)
- Output conversations pass semantic review: benign convos contain no risk content; risk convos contain appropriate indicators at the specified severity
