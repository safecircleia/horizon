#!/usr/bin/env python3
"""Main conversation generation script.

This script orchestrates the entire data generation pipeline:
1. Loads configuration from config.yaml
2. Initializes the appropriate LLM generator
3. Generates synthetic conversations with quality validation
4. Writes results to JSONL format
5. Reports statistics

Usage:
    python generate.py --category grooming --count 100 --output data/raw/grooming.jsonl
    python generate.py --category benign --count 200 --generator openai
    python generate.py --config custom_config.yaml
"""

import argparse
import asyncio
import datetime
import os
import random
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import jsonlines

# Third-party imports
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from data.generation.generators import (
    BedrockGenerator,
    ClaudeGenerator,
    ConversationGenerator,
    GPTGenerator,
    VLLMGenerator,
)
from data.generation.prompts.base import pick_language
from data.generation.prompts.benign import create_benign_prompt
from data.generation.prompts.bullying import create_bullying_prompt
from data.generation.prompts.grooming import create_grooming_prompt
from data.generation.prompts.isolation import create_isolation_prompt
from data.generation.prompts.personal_info import create_personal_info_prompt
from data.generation.prompts.platform_migration import create_platform_migration_prompt
from data.generation.prompts.sexual_content import create_sexual_content_prompt
from data.generation.prompts.threats import create_threats_prompt
from data.generation.validators.quality import validate_conversation_quality

# Local imports
from data.generation.validators.schemas import (
    ConversationLabel,
    Message,
    RiskCategory,
    RiskLevel,
    SyntheticConversation,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def create_generator(
    generator_type: str, config: Dict[str, Any]
) -> ConversationGenerator:
    """Create and initialize the appropriate generator.

    Args:
        generator_type: Either 'claude' or 'openai'
        config: Configuration dictionary

    Returns:
        Initialized ConversationGenerator
    """
    gen_config = config.get("generation", {})

    if generator_type == "claude":
        return ClaudeGenerator(
            model=gen_config.get("model_claude", "claude-3-5-sonnet-20241022"),
            temperature=gen_config.get("temperature", 0.9),
            max_tokens=gen_config.get("max_tokens", 2000),
        )
    elif generator_type == "openai":
        return GPTGenerator(
            model=gen_config.get("model_openai", "gpt-4o-2024-08-06"),
            temperature=gen_config.get("temperature", 0.9),
            max_tokens=gen_config.get("max_tokens", 2000),
        )
    elif generator_type == "bedrock":
        return BedrockGenerator(
            model=gen_config.get(
                "model_bedrock", "eu.anthropic.claude-3-5-sonnet-20241022-v2:0"
            ),
            region=gen_config.get("bedrock_region", "eu-west-3"),
            temperature=gen_config.get("temperature", 0.9),
            max_tokens=gen_config.get("max_tokens", 2000),
        )
    elif generator_type == "vllm":
        return VLLMGenerator(
            model=gen_config.get("model_vllm", "Qwen/Qwen2.5-72B-Instruct-AWQ"),
            base_url=gen_config.get("vllm_base_url", "http://localhost:8000/v1"),
            temperature=gen_config.get("temperature", 0.9),
            max_tokens=gen_config.get("max_tokens", 2000),
        )
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


def select_severity(
    category: RiskCategory, severity_distribution: Dict[str, float]
) -> RiskLevel:
    """Select a random severity level based on distribution.

    Args:
        category: Risk category (benign always gets NONE)
        severity_distribution: Distribution weights for each severity

    Returns:
        Selected RiskLevel
    """
    if category == RiskCategory.BENIGN:
        return RiskLevel.NONE

    # Sample from distribution
    levels = list(severity_distribution.keys())
    weights = list(severity_distribution.values())

    selected = random.choices(levels, weights=weights, k=1)[0]
    return RiskLevel(selected)


async def generate_conversation(
    generator: ConversationGenerator,
    category: RiskCategory,
    severity: RiskLevel,
    config: Dict[str, Any],
    retry_attempts: int = 3,
    language_mode: str = "mixed",
) -> Optional[SyntheticConversation]:
    """Generate a single conversation with quality validation.

    Args:
        generator: LLM generator instance
        category: Risk category to generate
        severity: Severity level
        config: Configuration dictionary
        retry_attempts: Number of retry attempts on failure

    Returns:
        Valid SyntheticConversation or None if all attempts fail
    """
    quality_config = config.get("quality", {})

    for attempt in range(retry_attempts):
        # Random parameters
        child_age = random.randint(13, 17)
        num_messages = random.randint(8, 15)
        lang = pick_language(language_mode)

        # Create prompt
        _PROMPT_BUILDERS = {
            RiskCategory.BENIGN: lambda: create_benign_prompt(child_age, num_messages, lang),
            RiskCategory.BULLYING: lambda: create_bullying_prompt(severity, child_age, num_messages, lang),
            RiskCategory.GROOMING: lambda: create_grooming_prompt(severity, child_age, num_messages, lang),
            RiskCategory.ISOLATION: lambda: create_isolation_prompt(severity, child_age, num_messages, lang),
            RiskCategory.PERSONAL_INFO: lambda: create_personal_info_prompt(severity, child_age, num_messages, lang),
            RiskCategory.PLATFORM_MIGRATION: lambda: create_platform_migration_prompt(severity, child_age, num_messages, lang),
            RiskCategory.SEXUAL_CONTENT: lambda: create_sexual_content_prompt(severity, child_age, num_messages, lang),
            RiskCategory.THREATS: lambda: create_threats_prompt(severity, child_age, num_messages, lang),
        }
        prompt = _PROMPT_BUILDERS[category]()

        # Generate conversation
        result = await generator.generate(prompt)

        if not result.success:
            continue

        try:
            # Parse messages
            raw_messages = result.conversation.get("messages", [])
            messages = [Message(**msg) for msg in raw_messages]

            # Validate quality
            is_valid, errors = validate_conversation_quality(
                messages,
                min_length=quality_config.get("min_conversation_length", 8),
                max_length=quality_config.get("max_conversation_length", 40),
            )

            if not is_valid:
                continue

            # Calculate severity score
            severity_scores = {
                RiskLevel.NONE: 0.0,
                RiskLevel.LOW: 0.25,
                RiskLevel.MEDIUM: 0.5,
                RiskLevel.HIGH: 0.75,
                RiskLevel.CRITICAL: 0.95,
            }

            # Create label
            label = ConversationLabel(
                risk_level=severity,
                categories=[category]
                if category != RiskCategory.BENIGN
                else [RiskCategory.BENIGN],
                severity_score=severity_scores[severity],
                reasoning=result.conversation.get(
                    "reasoning", "Generated conversation"
                ),
            )

            # Create synthetic conversation
            conversation = SyntheticConversation(
                conversation_id=str(uuid.uuid4()),
                category=category,
                messages=messages,
                label=label,
                metadata={
                    "generator": generator.name,
                    "child_age": child_age,
                    "generated_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    "model": generator.model,
                    "attempt": attempt + 1,
                    "language": lang,
                },
            )

            return conversation

        except Exception:
            continue

    return None


async def generate_batch(
    generator: ConversationGenerator,
    category: RiskCategory,
    count: int,
    config: Dict[str, Any],
    concurrency: int = 10,
    on_progress: Optional[Callable[[int, int], None]] = None,
    tqdm_position: int = 0,
    language_mode: str = "mixed",
) -> List[SyntheticConversation]:
    """Generate a batch of conversations with a live worker pool.

    on_progress: optional callback(n_success, n_failed) after each result.
                 When provided, tqdm is suppressed — the caller owns the display.
    """
    severity_dist = config.get("severity_distribution", {})
    conversations: List[SyntheticConversation] = []
    failed = 0
    semaphore = asyncio.Semaphore(concurrency)
    queue: asyncio.Queue = asyncio.Queue()

    # Pre-fill with extra work to absorb failures, then refill as needed
    for _ in range(int(count * 1.3) + concurrency):
        await queue.put(select_severity(category, severity_dist))

    async def worker(pbar=None):
        nonlocal failed
        while True:
            try:
                severity = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            async with semaphore:
                result = await generate_conversation(generator, category, severity, config,
                                                     language_mode=language_mode)
            queue.task_done()
            if result is not None:
                conversations.append(result)
            else:
                failed += 1
            if on_progress is not None:
                on_progress(len(conversations), failed)
            elif pbar is not None and result is not None:
                pbar.update(1)
            if len(conversations) >= count:
                return
            if queue.empty():
                await queue.put(select_severity(category, severity_dist))

    if on_progress is not None:
        await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(concurrency)])
    else:
        with tqdm(total=count, desc=f"{category.value:<20}", unit="conv",
                  position=tqdm_position, leave=True) as pbar:
            await asyncio.gather(*[asyncio.create_task(worker(pbar)) for _ in range(concurrency)])

    return conversations[:count]


def count_existing(output_path: str) -> int:
    """Count valid lines already written to a JSONL file."""
    p = Path(output_path)
    if not p.exists():
        return 0
    count = 0
    with open(p) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def write_conversations(
    conversations: List[SyntheticConversation],
    output_path: str,
    append: bool = False,
) -> None:
    """Write conversations to JSONL file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with jsonlines.open(output_file, mode=mode) as writer:
        for conv in conversations:
            writer.write(conv.model_dump())


def print_summary(
    conversations: List[SyntheticConversation],
    category: RiskCategory,
    elapsed_time: float,
) -> None:
    """Print generation summary statistics.

    Args:
        conversations: Generated conversations
        category: Category that was generated
        elapsed_time: Time taken in seconds
    """
    if not conversations:
        print("\nNo conversations generated!")
        return

    # Calculate statistics
    total = len(conversations)
    avg_messages = sum(len(c.messages) for c in conversations) / total

    severity_counts = {}
    for conv in conversations:
        severity = conv.label.risk_level.value
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    generators = {}
    for conv in conversations:
        gen = conv.metadata.get("generator", "unknown")
        generators[gen] = generators.get(gen, 0) + 1

    # Print summary
    print("\n" + "=" * 60)
    print(f"GENERATION SUMMARY: {category.value}")
    print("=" * 60)
    print(f"Total Conversations: {total}")
    print(f"Average Messages: {avg_messages:.1f}")
    print(f"Time Elapsed: {elapsed_time:.1f}s")
    print(f"Rate: {total / elapsed_time:.2f} conv/s")
    print()

    print("Severity Distribution:")
    for severity, count in sorted(severity_counts.items()):
        percentage = (count / total) * 100
        print(f"  {severity:>10}: {count:>4} ({percentage:>5.1f}%)")
    print()

    print("Generators Used:")
    for gen, count in sorted(generators.items()):
        percentage = (count / total) * 100
        print(f"  {gen:>10}: {count:>4} ({percentage:>5.1f}%)")
    print("=" * 60)


async def main():
    """Main entry point for conversation generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic conversations for SafeCircle dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 grooming conversations using Claude
  python generate.py --category grooming --count 100 --output data/raw/grooming.jsonl

  # Generate 200 benign conversations using OpenAI
  python generate.py --category benign --count 200 --generator openai

  # Use custom config file
  python generate.py --config custom_config.yaml --category bullying --count 50
        """,
    )

    parser.add_argument(
        "--category",
        type=str,
        required=True,
        choices=[cat.value for cat in RiskCategory],
        help="Risk category to generate",
    )

    parser.add_argument(
        "--count", type=int, required=True, help="Number of conversations to generate"
    )

    parser.add_argument(
        "--generator",
        type=str,
        default="claude",
        choices=["claude", "openai", "bedrock", "vllm"],
        help="LLM generator to use (default: claude)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file path (default: data/raw/{category}.jsonl)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="data/generation/config.yaml",
        help="Path to config.yaml (default: data/generation/config.yaml)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-generated conversations and append only what is missing",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="mixed",
        choices=["en", "es", "mixed"],
        help="Language for generated conversations: en, es, or mixed (40%% Spanish, default: mixed)",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of concurrent workers (overrides config.yaml value)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Validate API keys
    generator_type = args.generator
    if generator_type == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not found in environment", file=sys.stderr)
        print("Please set it in .env file or export it", file=sys.stderr)
        sys.exit(1)
    elif generator_type == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment", file=sys.stderr)
        print("Please set it in .env file or export it", file=sys.stderr)
        sys.exit(1)
    elif generator_type == "bedrock" and not os.getenv("BEDROCK_API_KEY"):
        print("Error: BEDROCK_API_KEY not found in environment", file=sys.stderr)
        print("Please set it in .env file or export it", file=sys.stderr)
        sys.exit(1)
    # vllm: no API key needed — local server

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Create generator
    try:
        generator = create_generator(generator_type, config)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Prepare output path
    category = RiskCategory(args.category)
    output_path = args.output or f"data/raw/{category.value}.jsonl"

    # Resume: check how many already exist
    already_have = 0
    if args.resume:
        already_have = count_existing(output_path)
        if already_have >= args.count:
            print(
                f"Already have {already_have}/{args.count} conversations in {output_path} — skipping."
            )
            sys.exit(0)

    remaining = args.count - already_have

    # Print configuration
    print("=" * 60)
    print("SafeCircle Data Generation")
    print("=" * 60)
    print(f"Category: {category.value}")
    print(
        f"Target: {args.count}  |  Already done: {already_have}  |  Remaining: {remaining}"
    )
    print(f"Generator: {generator.name} ({generator.model})")
    print(f"Output: {output_path}")
    print(f"Config: {args.config}")
    print("=" * 60)
    print()

    # Generate conversations
    start_time = datetime.datetime.now()

    concurrency = args.concurrency or config.get("generation", {}).get("concurrency", 10)
    conversations = await generate_batch(
        generator,
        category,
        remaining,
        config,
        concurrency=concurrency,
        language_mode=args.language,
    )

    elapsed = (datetime.datetime.now() - start_time).total_seconds()

    # Write to file (append if resuming)
    if conversations:
        write_conversations(
            conversations, output_path, append=args.resume and already_have > 0
        )
        print(f"\nWrote {len(conversations)} conversations to {output_path}")

    # Print summary
    print_summary(conversations, category, elapsed)

    # Exit with appropriate code
    sys.exit(0 if len(conversations) == remaining else 1)


if __name__ == "__main__":
    asyncio.run(main())
