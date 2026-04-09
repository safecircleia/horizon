# Data Generation Scripts

This directory contains the main conversation generation scripts for the SafeCircle dataset.

## Main Script: generate.py

The primary entry point for generating synthetic conversations.

### Quick Start

1. Set up your API keys in `.env`:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

2. Generate conversations:
   ```bash
   # Using Claude (default)
   PYTHONPATH=. .venv/bin/python data/generation/scripts/generate.py \
     --category grooming \
     --count 100 \
     --output data/raw/grooming.jsonl

   # Using OpenAI
   PYTHONPATH=. .venv/bin/python data/generation/scripts/generate.py \
     --category benign \
     --count 200 \
     --generator openai
   ```

### CLI Arguments

- `--category`: Risk category to generate (required)
  - Choices: `grooming`, `bullying`, `sexual_content`, `isolation`, `personal_info`, `platform_migration`, `threats`, `benign`
  
- `--count`: Number of conversations to generate (required)

- `--generator`: LLM to use (default: `claude`)
  - Choices: `claude`, `openai`

- `--output`: Output JSONL file path (default: `data/raw/{category}.jsonl`)

- `--config`: Path to config.yaml (default: `data/generation/config.yaml`)

### Examples

```bash
# Generate 100 grooming conversations
PYTHONPATH=. .venv/bin/python data/generation/scripts/generate.py \
  --category grooming --count 100

# Generate 200 benign conversations with custom output
PYTHONPATH=. .venv/bin/python data/generation/scripts/generate.py \
  --category benign --count 200 --output my_benign.jsonl

# Use OpenAI with custom config
PYTHONPATH=. .venv/bin/python data/generation/scripts/generate.py \
  --category bullying --count 50 \
  --generator openai --config custom_config.yaml
```

### Configuration

The script loads configuration from `data/generation/config.yaml` which includes:

- **Dataset targets**: Number of conversations per category
- **Severity distribution**: Probability weights for risk levels
- **Generation parameters**: Model names, temperature, tokens, batch size
- **Quality thresholds**: Minimum/maximum conversation length, vocabulary diversity

### Output Format

Generated conversations are written to JSONL (JSON Lines) format, with each line containing a complete `SyntheticConversation` object:

```json
{
  "conversation_id": "uuid-here",
  "category": "grooming",
  "messages": [
    {"role": "sent", "content": "hey", "timestamp": 1234567890},
    {"role": "received", "content": "hi there", "timestamp": 1234567895}
  ],
  "label": {
    "risk_level": "medium",
    "categories": ["grooming"],
    "severity_score": 0.5,
    "reasoning": "..."
  },
  "metadata": {
    "generator": "claude",
    "child_age": 15,
    "generated_at": "2024-01-15T10:30:00",
    "model": "claude-3-5-sonnet-20241022",
    "attempt": 1
  }
}
```

### Quality Validation

Each generated conversation is validated for:

- **Length**: 4-30 messages (configurable)
- **Vocabulary diversity**: Minimum 20 unique tokens
- **Timestamp progression**: Monotonically increasing timestamps
- **Role alternation**: Valid "sent"/"received" patterns

Conversations that fail validation are automatically regenerated (up to 3 attempts).

### Summary Statistics

After generation completes, the script prints summary statistics:

- Total conversations generated
- Average message count
- Generation time and rate
- Severity distribution
- Generators used

Example output:
```
============================================================
GENERATION SUMMARY: grooming
============================================================
Total Conversations: 100
Average Messages: 8.3
Time Elapsed: 245.2s
Rate: 0.41 conv/s

Severity Distribution:
        low:   25 ( 25.0%)
     medium:   35 ( 35.0%)
       high:   25 ( 25.0%)
   critical:   15 ( 15.0%)

Generators Used:
      claude:  100 (100.0%)
============================================================
```

## Environment Variables

Required API keys (set in `.env`):

- `ANTHROPIC_API_KEY`: For Claude generator
- `OPENAI_API_KEY`: For OpenAI/GPT generator

## Error Handling

The script includes robust error handling:

- Retries failed API calls (3 attempts default)
- Validates JSON parsing from LLM responses
- Checks required fields and schema compliance
- Reports failures but continues generation
- Returns appropriate exit codes (0 for success, 1 for failures)

## Notes

- Progress bars show real-time generation status
- Failed generations (after all retries) are logged but don't stop the process
- Random parameters (child age, message count, severity) ensure diversity
- PYTHONPATH must include project root for imports to work
