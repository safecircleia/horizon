"""Rule-based risk detector — Level 0-3 catch rate (no LLM required).

Implements a keyword/pattern matching pre-filter that catches obvious risk
signals before LLM inference. Used to measure the "Level 0-3 catch rate"
benchmark target: ≥ 80% of risk conversations should be flaggable by rules
alone, providing a fast-path that reduces LLM calls in production.

Levels:
  0 — explicit grooming keywords (highest confidence)
  1 — sexual content signals
  2 — threat / self-harm signals
  3 — isolation / platform-migration signals
"""

import re
from typing import Optional

# fmt: off
_PATTERNS: list[tuple[int, str, list[str]]] = [
    # (level, category, patterns)
    (0, "grooming", [
        r"\bage\b.{0,20}\b(you|ur|your)\b",
        r"\bhow old are you\b",
        r"\bsend (me |a )?(pic|photo|selfie|nude|nudes)\b",
        r"\bkeep (it|this|us) (a )?(secret|between us)\b",
        r"\bdon'?t tell (your|ur) (mom|dad|parent|family)\b",
        r"\bspecial (friend|bond|connection|relationship)\b",
        r"\bmature for (your|ur) age\b",
        r"\badd me on (discord|snapchat|telegram|whatsapp|signal)\b",
        r"\bjust (the )?two of us\b",
        r"\bi love you\b.{0,40}\b(meet|see you|in person)\b",
    ]),
    (1, "sexual_content", [
        r"\bsend (nudes?|naked|explicit)\b",
        r"\bsexting\b",
        r"\bturn(ed)? (me |you )?(on|off)\b",
        r"\btouch (yourself|myself|me)\b",
        r"\bfap\b|\bjerk off\b|\bfinger\b.{0,10}\byourself\b",
        r"\bshow (me )?(your )?(body|ass|tits|dick|cock|pussy)\b",
        r"\bvideo call.{0,20}\bnaked\b",
    ]),
    (2, "threats", [
        r"\bi('ll| will| am going to) (kill|hurt|stab|shoot|beat)\b",
        r"\bkill (your|ur)self\b",
        r"\bkys\b",
        r"\bi know where (you|u) live\b",
        r"\bgoing to (find|hurt|get) you\b",
        r"\bwatch (your|ur) back\b",
        r"\bpost (your|ur) (nudes?|photos?|pics?)\b.{0,30}\bif\b",
    ]),
    (3, "isolation", [
        r"\bdon'?t (talk to|trust|tell) (your|ur|them|him|her|anyone)\b",
        r"\bthey (don'?t|do not) (care|understand|get) (about )?you\b",
        r"\bonly (i|me) (care|understand|get) you\b",
        r"\bcome (meet|see) me\b.{0,30}\balone\b",
        r"\bmove to (another|different|private) (app|platform|chat)\b",
        r"\bdelete (this|our) (chat|messages|conversation)\b",
    ]),
]
# fmt: on

_COMPILED: list[tuple[int, str, list[re.Pattern]]] = [
    (level, cat, [re.compile(p, re.IGNORECASE) for p in patterns])
    for level, cat, patterns in _PATTERNS
]


def _conversation_text(conversation: dict) -> str:
    """Extract all message content from a conversation dict."""
    msgs = conversation.get("messages", [])
    if msgs and isinstance(msgs[0], dict) and "role" in msgs[0]:
        # Training/eval format: [{role, content}, ...]
        return " ".join(
            m["content"] for m in msgs if m.get("role") not in ("system",)
        )
    # Raw generation format: [{role: "sent"|"received", content}, ...]
    return " ".join(m.get("content", "") for m in msgs)


def rule_based_detect(conversation: dict) -> Optional[tuple[int, str]]:
    """Return (level, category) for the first rule that fires, or None."""
    text = _conversation_text(conversation)
    for level, category, patterns in _COMPILED:
        for pat in patterns:
            if pat.search(text):
                return level, category
    return None


def compute_rule_catch_rate(examples: list[dict], labels: list[dict]) -> dict:
    """Measure rule catch rate on a labelled set.

    Args:
        examples: list of conversation dicts (messages-format or raw)
        labels: list of label dicts with at least {"risk_level": str}

    Returns dict with counts and catch_rate (float 0-1).
    """
    total_risk = 0
    caught = 0
    false_positives = 0
    total_benign = 0

    for ex, label in zip(examples, labels):
        is_risk = label.get("risk_level", "none") != "none"
        hit = rule_based_detect(ex) is not None

        if is_risk:
            total_risk += 1
            if hit:
                caught += 1
        else:
            total_benign += 1
            if hit:
                false_positives += 1

    catch_rate = caught / total_risk if total_risk else 0.0
    fp_rate = false_positives / total_benign if total_benign else 0.0

    return {
        "total_risk": total_risk,
        "total_benign": total_benign,
        "caught": caught,
        "false_positives": false_positives,
        "catch_rate": round(catch_rate, 4),
        "fp_rate": round(fp_rate, 4),
    }
