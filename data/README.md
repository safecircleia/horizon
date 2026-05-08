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
- 10K<n<100K
configs:
- config_name: raw
  data_files:
  - split: grooming
    path: data/grooming/*
  - split: bullying
    path: data/bullying/*
  - split: sexual_content
    path: data/sexual_content/*
  - split: isolation
    path: data/isolation/*
  - split: personal_info
    path: data/personal_info/*
  - split: platform_migration
    path: data/platform_migration/*
  - split: threats
    path: data/threats/*
  - split: benign
    path: data/benign/*
- config_name: processed
  data_files:
  - split: train
    path: data/train/*
  - split: eval
    path: data/eval/*
---

# Horizon Child Safety Risk Detection Dataset

**SafeCircle / Project Horizon** — Synthetic training data for fine-tuning child safety risk detection models.

> ⚠️ **Private dataset.** Access is restricted to SafeCircle team members and authorized research collaborators. All conversations are **fully synthetic** — no real child messages were used.

## Dataset Summary

50,000 synthetic conversations spanning seven child safety risk categories plus a benign baseline. Designed to train the Horizon model: a fine-tuned Llama 3.1 8B that outputs structured JSON risk assessments.

| Config | Splits | Examples |
|---|---|---|
| `raw` | 8 per-category splits | 50,000 |
| `processed` | train / eval | 45,000 / 5,000 |

## Risk Categories

| Category | Examples | Description |
|---|---|---|
| `grooming` | 8,000 | Trust-building, boundary testing, secrecy establishment |
| `bullying` | 7,000 | Harassment, cyberbullying, peer intimidation |
| `sexual_content` | 7,000 | Explicit messages, inappropriate requests |
| `isolation` | 5,000 | Controlling behavior, network isolation tactics |
| `personal_info` | 5,000 | Solicitation of identifying information |
| `platform_migration` | 3,000 | Moving to less monitored platforms |
| `threats` | 5,000 | Violent threats, dangerous challenges |
| `benign` | 10,000 | Normal teen conversations (negative baseline) |

## Dataset Structure

### `raw` config

Each example represents one synthetic conversation with full metadata:

```python
{
  "conversation_id": "5b882378-e811-4db2-8399-141c75681eb5",
  "category": "grooming",
  "messages": [
    {"role": "sent", "content": "Hey, u up? 😊", "timestamp": 1609459200},
    {"role": "received", "content": "Yh, just got home.", "timestamp": 1609459210},
    ...
  ],
  "label": {
    "risk_level": "medium",        # none | low | medium | high | critical
    "categories": ["grooming"],
    "severity_score": 0.5,         # 0.0–1.0
    "reasoning": "..."
  },
  "metadata": {
    "generator": "bedrock",
    "child_age": 14,
    "generated_at": "2026-05-05T09:01:02Z",
    "model": "eu.amazon.nova-micro-v1:0",
    "attempt": 1
  }
}
```

### `processed` config

Llama 3.1 instruction-formatted examples ready for supervised fine-tuning:

```python
{
  "conversation_id": "...",
  "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n...<|eot_id|>...",
  "label": { "risk_level": "high", "categories": [...], ... },
  "category": "bullying"
}
```

The `text` field contains the full prompt+completion in Llama 3.1 chat format, with the model's JSON risk assessment as the assistant turn.

## Usage

```python
from datasets import load_dataset

# Fine-tuning (processed splits)
ds = load_dataset("safecircleai/horizon-training-data", name="processed", token="hf_...")
train_ds = ds["train"]   # 45,000 examples
eval_ds  = ds["eval"]    # 5,000 examples

# Per-category analysis (raw splits)
raw = load_dataset("safecircleai/horizon-training-data", name="raw", token="hf_...")
grooming = raw["grooming"]   # 8,000 examples

# Stream without downloading
ds = load_dataset("safecircleai/horizon-training-data", name="processed",
                  token="hf_...", streaming=True)
```

Or via the Horizon Makefile:

```bash
export HF_TOKEN=hf_...
make download-data            # all data
make download-data SPLIT=processed  # processed only
```

## Generation

All conversations were generated synthetically using Amazon Bedrock (Nova Micro) via the Horizon data generation pipeline. Generation prompts were designed to:

- Reflect realistic teen communication patterns (abbreviations, emoji, informal language)
- Cover a range of severity levels within each risk category
- Vary child ages (12–17) and conversational contexts
- Include escalation patterns that mirror real grooming/abuse progression

No real conversations, personal data, or real child identities were used at any stage.

## Model Output Format

The Horizon model produces structured JSON for each conversation:

```json
{
  "risk_level": "high",
  "categories": ["grooming", "personal_info"],
  "confidence": 0.87,
  "matched_terms": ["our secret", "don't tell", "send photo"],
  "reasoning": "Adult establishing secrecy while requesting personal media"
}
```

## Intended Use

- Fine-tuning child safety classifiers for the SafeCircle platform
- Research into automated risk detection in youth communications
- Benchmarking content moderation models on child safety tasks

**Out-of-scope uses:** This dataset must not be used to generate, facilitate, or normalize harmful content targeting minors.

## Ethical Considerations

- All data is synthetic — generated by AI, reviewed for quality, with no real children involved
- Dataset is private to prevent misuse
- Risk labels were generated by the same model that produced conversations; human review is recommended before production use
- The `benign` split is intentionally large (20%) to reduce false-positive rates

## Citation

```bibtex
@dataset{safecircle_horizon_2026,
  author    = {SafeCircle Team},
  title     = {Horizon Child Safety Risk Detection Dataset},
  year      = {2026},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/datasets/safecircleai/horizon-training-data}
}
```

## License

Proprietary — SafeCircle Research License. Contact [team@safecircle.tech](mailto:team@safecircle.tech) for access and usage terms.
