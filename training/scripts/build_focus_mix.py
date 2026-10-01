"""Build a grooming-enriched train file for a short continued fine-tune (issue #8).

The mobile model files subtle/low-severity grooming under sexual_content, so we
oversample grooming (low/medium severity most), keep sexual_content as the
contrast class, and replay a fraction of everything else so nothing regresses.

Usage: python -m training.scripts.build_focus_mix [--input IN] [--output OUT]
"""

import argparse
import json
import random

REPLAY_FRACTION = 0.2  # share of non-grooming/non-sexual examples kept
SUBTLE = {"low", "medium"}


def copies(label: dict, rng: random.Random) -> int:
    """How many times to emit an example with this label."""
    cats = label.get("categories") or []
    cat = cats[0] if label.get("risk_level", "none") != "none" and cats else "benign"
    if cat == "grooming":
        return 3 if label.get("risk_level") in SUBTLE else 1
    if cat == "sexual_content":
        return 1
    return 1 if rng.random() < REPLAY_FRACTION else 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/train.jsonl")
    p.add_argument("--output", default="data/processed/train_focus.jsonl")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    kept = []
    with open(args.input) as f:
        for line in f:
            if not line.strip():
                continue
            label = json.loads(json.loads(line)["messages"][-1]["content"])
            kept.extend([line] * copies(label, rng))
    rng.shuffle(kept)
    with open(args.output, "w") as f:
        f.writelines(kept)
    print(f"Wrote {len(kept)} examples to {args.output}")


if __name__ == "__main__":
    main()
