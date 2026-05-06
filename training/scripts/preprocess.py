#!/usr/bin/env python3
"""Preprocess raw JSONL conversations into instruction-tuned training format.

Converts SyntheticConversation JSONL → Llama 3.1 chat format JSONL
ready for supervised fine-tuning.

Usage:
    python -m training.scripts.preprocess --input data/raw/ --output data/processed/
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterator

SYSTEM_PROMPT = (
    "You are SafeCircle's risk detection model. "
    "Analyze conversations for child safety risks. "
    "Output JSON with: risk_level (none/low/medium/high/critical), "
    "categories (array), confidence (0-1), "
    "matched_terms (array), reasoning (brief)."
)


def conversation_to_text(messages: list[dict]) -> str:
    """Format message list into readable conversation text."""
    lines = []
    for msg in messages:
        role = "Child" if msg["role"] == "sent" else "Other"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def build_target(label: dict, category: str) -> dict:
    """Build the expected JSON output from a conversation label."""
    risk_level = label["risk_level"]
    categories = [c for c in label.get("categories", []) if c != "benign"]

    # Infer confidence from severity score
    severity_score = label.get("severity_score", 0.5)
    confidence = round(0.7 + severity_score * 0.25, 2) if risk_level != "none" else round(0.85 + severity_score * 0.1, 2)
    confidence = min(confidence, 0.99)

    return {
        "risk_level": risk_level,
        "categories": categories if categories else [],
        "confidence": confidence,
        "matched_terms": [],
        "reasoning": label.get("reasoning", ""),
    }


def format_training_example(raw: dict) -> dict:
    """Convert a SyntheticConversation dict to a training example."""
    conversation_text = conversation_to_text(raw["messages"])
    target = build_target(raw["label"], raw["category"])

    return {
        "conversation_id": raw["conversation_id"],
        "text": (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{SYSTEM_PROMPT}\n"
            f"<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"Analyze this conversation:\n{conversation_text}\n"
            f"<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            f"{json.dumps(target)}\n"
            f"<|eot_id|>"
        ),
        "label": raw["label"],
        "category": raw["category"],
    }


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Preprocess conversations for training")
    parser.add_argument("--input", default="data/raw", help="Input directory with JSONL files")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--split", type=float, default=0.9, help="Train/eval split ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = list(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"Error: no JSONL files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    examples = []
    parse_errors = 0
    for path in jsonl_files:
        for raw in iter_jsonl(path):
            try:
                examples.append(format_training_example(raw))
            except (KeyError, TypeError):
                parse_errors += 1

    if not examples:
        print("Error: no valid examples parsed", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    random.shuffle(examples)
    split_idx = int(len(examples) * args.split)
    train, eval_ = examples[:split_idx], examples[split_idx:]

    for name, subset in [("train", train), ("eval", eval_)]:
        out_path = output_dir / f"{name}.jsonl"
        with open(out_path, "w") as f:
            for ex in subset:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(subset)} examples → {out_path}")

    if parse_errors:
        print(f"Warning: {parse_errors} records skipped due to parse errors")

    print(f"\nTotal: {len(examples)} examples ({len(train)} train / {len(eval_)} eval)")


if __name__ == "__main__":
    main()
