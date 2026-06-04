"""Update the dataset card (README.md) on HuggingFace Hub."""

from huggingface_hub import HfApi

README = """\
---
license: other
license_name: safecircle-research
license_link: https://safecircle.tech/licenses/research
language:
- en
task_categories:
- text-classification
task_ids:
- multi-label-classification
- intent-classification
tags:
- child-safety
- content-moderation
- risk-detection
- synthetic
- conversational
pretty_name: Horizon Child Safety Risk Detection Dataset
size_categories:
- 1M<n<10M
configs:
- config_name: processed
  data_files:
  - split: train
    path: processed/train-*.parquet
  - split: eval
    path: processed/eval-*.parquet
- config_name: raw
  data_files:
  - split: grooming
    path: raw/grooming-*
  - split: bullying
    path: raw/bullying-*
  - split: sexual_content
    path: raw/sexual_content-*
  - split: isolation
    path: raw/isolation-*
  - split: personal_info
    path: raw/personal_info-*
  - split: platform_migration
    path: raw/platform_migration-*
  - split: threats
    path: raw/threats-*
  - split: benign
    path: raw/benign-*
---

# Horizon Child Safety Risk Detection Dataset

**SafeCircle / Project Horizon** — Synthetic training data for fine-tuning child safety risk detection models.

> ⚠️ **Private dataset.** Access is restricted to SafeCircle team members and authorized research collaborators. All conversations are **fully synthetic** — no real child messages were used.

## Dataset Summary

1,600,000 synthetic conversations spanning seven child safety risk categories plus a benign baseline. Designed to train the Horizon model: a lightweight mobile classifier that outputs structured risk assessments.

| Config | Splits | Examples |
|---|---|---|
| `raw` | 8 per-category splits | 1,600,000 |
| `processed` | train / eval | TBD |

## Risk Categories

| Category | Examples | Description |
|---|---|---|
| `grooming` | 200,000 | Trust-building, boundary testing, secrecy establishment |
| `bullying` | 200,000 | Harassment, cyberbullying, peer intimidation |
| `sexual_content` | 200,000 | Explicit messages, inappropriate requests |
| `isolation` | 200,000 | Controlling behavior, network isolation tactics |
| `personal_info` | 200,000 | Solicitation of identifying information |
| `platform_migration` | 200,000 | Moving to less monitored platforms |
| `threats` | 200,000 | Violent threats, dangerous challenges |
| `benign` | 200,000 | Normal teen conversations (negative baseline) |

## Dataset Structure

### `raw` config

Each example represents one synthetic conversation with full metadata:

```python
{
  "conversation_id": "5b882378-e811-4db2-8399-141c75681eb5",
  "category": "grooming",
  "messages": [
    {"role": "sent", "content": "Hey, u up? 😊", "timestamp": 0},
    {"role": "received", "content": "Yh, just got home.", "timestamp": 0},
    ...
  ],
  "label": {
    "risk_level": "medium",        # none | low | medium | high | critical
    "categories": ["grooming"],
    "severity_score": 0.5,         # 0.0–1.0
    "reasoning": "..."
  },
  "metadata": {
    "generator": "vllm",
    "child_age": 14,
    "generated_at": "2026-05-17T09:01:02Z",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "attempt": 1
  }
}
```

### `processed` config

Instruction-formatted examples ready for supervised fine-tuning.

## Usage

```python
from datasets import load_dataset

# Per-category analysis (raw splits)
raw = load_dataset("safecircleai/horizon-training-data", name="raw", token="hf_...")
grooming = raw["grooming"]   # 200,000 examples
benign   = raw["benign"]     # 200,000 examples
```

## Generation

All conversations were generated synthetically using **Qwen/Qwen2.5-7B-Instruct** served via vLLM with **xgrammar constrained decoding** (JSON Schema enforcement). This guarantees every output is structurally valid JSON, eliminating parse failures at scale.

Key generation design decisions:
- **Persona seeds** — each conversation is grounded with a randomised name, platform (Discord/Instagram/Snapchat/WhatsApp/TikTok), and relationship type (classmate/online friend/stranger etc.)
- **Few-shot examples** — one baked-in example per category guides the 7B model toward the correct format and tone
- **Severity levels** — risk categories are generated across low/medium/high/critical severities (weighted distribution)
- **Narrative arc** — prompts explicitly request opening → development → resolution/escalation structure with 8–15 messages
- **Authentic style** — abbreviations, emoji, and casual grammar matching real teen communication patterns

No real conversations, personal data, or real child identities were used at any stage.

## Intended Use

- Fine-tuning child safety classifiers for the SafeCircle platform
- Research into automated risk detection in youth communications
- Benchmarking content moderation models on child safety tasks

**Out-of-scope uses:** This dataset must not be used to generate, facilitate, or normalize harmful content targeting minors.

## Ethical Considerations

- All data is synthetic — generated by AI, with no real children involved
- Dataset is private to prevent misuse
- The `benign` split (200k examples) is intentionally large to reduce false-positive rates in production

## License

Proprietary — SafeCircle Research License. Contact [team@safecircle.tech](mailto:team@safecircle.tech) for access and usage terms.
"""


def main() -> None:
    api = HfApi()
    repo_id = "safecircleai/horizon-training-data"

    api.upload_file(
        path_or_fileobj=README.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="docs: update dataset card — 200k per category, Qwen2.5-7B, xgrammar",
    )
    print(f"Dataset card updated: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
