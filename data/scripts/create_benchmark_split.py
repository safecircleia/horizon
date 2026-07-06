#!/usr/bin/env python3
"""Carve a held-out benchmark split from the processed dataset.

Samples are stratified by risk_level and never overlap with the training set.
The benchmark set is stored at data/evaluation/benchmark.jsonl and is excluded
from public repos (already in .gitignore via data/evaluation/).

Usage:
    python -m data.scripts.create_benchmark_split
    python -m data.scripts.create_benchmark_split --size 1000 --seed 42
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

EVAL_JSONL = Path("data/processed/eval.jsonl")
OUT_PATH = Path("data/evaluation/benchmark.jsonl")
DEFAULT_SIZE = 500


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(EVAL_JSONL))
    parser.add_argument("--output", default=str(OUT_PATH))
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Source not found: {source}")
        raise SystemExit(1)

    with open(source) as f:
        all_examples = [json.loads(line) for line in f if line.strip()]

    # Stratify by risk_level for a representative benchmark
    buckets: dict[str, list] = defaultdict(list)
    for ex in all_examples:
        label = ex.get("label", {})
        if isinstance(label, str):
            label = json.loads(label)
        level = label.get("risk_level", "none")
        buckets[level].append(ex)

    rng = random.Random(args.seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    levels = sorted(buckets.keys())
    per_level = max(1, args.size // len(levels))
    selected = []
    for level in levels:
        selected.extend(buckets[level][:per_level])

    # Top up to target size from remainder, stable order
    remainder = [ex for level in levels for ex in buckets[level][per_level:]]
    rng.shuffle(remainder)
    selected.extend(remainder[: max(0, args.size - len(selected))])
    selected = selected[: args.size]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for ex in selected:
            f.write(json.dumps(ex) + "\n")

    by_level = defaultdict(int)
    for ex in selected:
        label = ex.get("label", {})
        if isinstance(label, str):
            label = json.loads(label)
        by_level[label.get("risk_level", "none")] += 1

    print(f"Wrote {len(selected)} examples to {out}")
    for level, count in sorted(by_level.items()):
        print(f"  {level:10s}: {count}")
    print("\nNote: benchmark.jsonl is not included in the public repo (.gitignore).")
    print("      All data is synthetic — no real children's data.")


if __name__ == "__main__":
    main()
