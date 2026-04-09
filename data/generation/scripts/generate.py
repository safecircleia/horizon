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

import asyncio
import argparse
import os
import sys
import random
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Third-party imports
import yaml
import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

# Local imports
from data.generation.validators.schemas import (
    RiskCategory,
    RiskLevel,
    Message,
    ConversationLabel,
    SyntheticConversation
)
from data.generation.validators.quality import validate_conversation_quality
from data.generation.generators import (
    ConversationGenerator,
    ClaudeGenerator,
    GPTGenerator
)
from data.generation.prompts.base import create_conversation_prompt


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

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def create_generator(
    generator_type: str,
    config: Dict[str, Any]
) -> ConversationGenerator:
    """Create and initialize the appropriate generator.

    Args:
        generator_type: Either 'claude' or 'openai'
        config: Configuration dictionary

    Returns:
        Initialized ConversationGenerator
    """
    gen_config = config.get('generation', {})

    if generator_type == 'claude':
        return ClaudeGenerator(
            model=gen_config.get('model_claude', 'claude-3-5-sonnet-20241022'),
            temperature=gen_config.get('temperature', 0.9),
            max_tokens=gen_config.get('max_tokens', 2000)
        )
    elif generator_type == 'openai':
        return GPTGenerator(
            model=gen_config.get('model_openai', 'gpt-4o-2024-08-06'),
            temperature=gen_config.get('temperature', 0.9),
            max_tokens=gen_config.get('max_tokens', 2000)
        )
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


def select_severity(
    category: RiskCategory,
    severity_distribution: Dict[str, float]
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
    retry_attempts: int = 3
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
    quality_config = config.get('quality', {})

    for attempt in range(retry_attempts):
        # Random parameters
        child_age = random.randint(13, 17)
        num_messages = random.randint(5, 15)

        # Create prompt
        prompt = create_conversation_prompt(
            category=category,
            severity=severity,
            child_age=child_age,
            num_messages=num_messages
        )

        # Generate conversation
        result = await generator.generate(prompt)

        if not result.success:
            continue

        try:
            # Parse messages
            raw_messages = result.conversation.get('messages', [])
            messages = [Message(**msg) for msg in raw_messages]

            # Validate quality
            is_valid, errors = validate_conversation_quality(
                messages,
                min_length=quality_config.get('min_conversation_length', 4),
                max_length=quality_config.get('max_conversation_length', 30),
                min_unique_tokens=quality_config.get('min_unique_tokens', 20),
                allow_consecutive_roles=True
            )

            if not is_valid:
                continue

            # Calculate severity score
            severity_scores = {
                RiskLevel.NONE: 0.0,
                RiskLevel.LOW: 0.25,
                RiskLevel.MEDIUM: 0.5,
                RiskLevel.HIGH: 0.75,
                RiskLevel.CRITICAL: 0.95
            }

            # Create label
            label = ConversationLabel(
                risk_level=severity,
                categories=[category] if category != RiskCategory.BENIGN else [RiskCategory.BENIGN],
                severity_score=severity_scores[severity],
                reasoning=result.conversation.get('reasoning', 'Generated conversation')
            )

            # Create synthetic conversation
            conversation = SyntheticConversation(
                conversation_id=str(uuid.uuid4()),
                category=category,
                messages=messages,
                label=label,
                metadata={
                    'generator': generator.name,
                    'child_age': child_age,
                    'generated_at': datetime.utcnow().isoformat(),
                    'model': generator.model,
                    'attempt': attempt + 1
                }
            )

            return conversation

        except Exception as e:
            continue

    return None


async def generate_batch(
    generator: ConversationGenerator,
    category: RiskCategory,
    count: int,
    config: Dict[str, Any]
) -> List[SyntheticConversation]:
    """Generate a batch of conversations.

    Args:
        generator: LLM generator instance
        category: Risk category to generate
        count: Number of conversations to generate
        config: Configuration dictionary

    Returns:
        List of successfully generated conversations
    """
    severity_dist = config.get('severity_distribution', {})
    conversations = []

    with tqdm(total=count, desc=f"Generating {category.value}", unit="conv") as pbar:
        while len(conversations) < count:
            # Select severity
            severity = select_severity(category, severity_dist)

            # Generate conversation
            conversation = await generate_conversation(
                generator,
                category,
                severity,
                config
            )

            if conversation:
                conversations.append(conversation)
                pbar.update(1)
            else:
                # Failed after all retries - this counts as a failure but we continue
                pbar.write(f"Warning: Failed to generate valid conversation after retries")

    return conversations


def write_conversations(
    conversations: List[SyntheticConversation],
    output_path: str
) -> None:
    """Write conversations to JSONL file.

    Args:
        conversations: List of conversations to write
        output_path: Output file path
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(output_file, mode='w') as writer:
        for conv in conversations:
            writer.write(conv.model_dump())


def print_summary(
    conversations: List[SyntheticConversation],
    category: RiskCategory,
    elapsed_time: float
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
        gen = conv.metadata.get('generator', 'unknown')
        generators[gen] = generators.get(gen, 0) + 1

    # Print summary
    print("\n" + "="*60)
    print(f"GENERATION SUMMARY: {category.value}")
    print("="*60)
    print(f"Total Conversations: {total}")
    print(f"Average Messages: {avg_messages:.1f}")
    print(f"Time Elapsed: {elapsed_time:.1f}s")
    print(f"Rate: {total/elapsed_time:.2f} conv/s")
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
    print("="*60)


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
        """
    )

    parser.add_argument(
        '--category',
        type=str,
        required=True,
        choices=[cat.value for cat in RiskCategory],
        help='Risk category to generate'
    )

    parser.add_argument(
        '--count',
        type=int,
        required=True,
        help='Number of conversations to generate'
    )

    parser.add_argument(
        '--generator',
        type=str,
        default='claude',
        choices=['claude', 'openai'],
        help='LLM generator to use (default: claude)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSONL file path (default: data/raw/{category}.jsonl)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='data/generation/config.yaml',
        help='Path to config.yaml (default: data/generation/config.yaml)'
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Validate API keys
    generator_type = args.generator
    if generator_type == 'claude' and not os.getenv('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY not found in environment", file=sys.stderr)
        print("Please set it in .env file or export it", file=sys.stderr)
        sys.exit(1)
    elif generator_type == 'openai' and not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment", file=sys.stderr)
        print("Please set it in .env file or export it", file=sys.stderr)
        sys.exit(1)

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

    # Print configuration
    print("="*60)
    print("SafeCircle Data Generation")
    print("="*60)
    print(f"Category: {category.value}")
    print(f"Count: {args.count}")
    print(f"Generator: {generator.name} ({generator.model})")
    print(f"Output: {output_path}")
    print(f"Config: {args.config}")
    print("="*60)
    print()

    # Generate conversations
    start_time = datetime.now()

    conversations = await generate_batch(
        generator,
        category,
        args.count,
        config
    )

    elapsed = (datetime.now() - start_time).total_seconds()

    # Write to file
    if conversations:
        write_conversations(conversations, output_path)
        print(f"\nWrote {len(conversations)} conversations to {output_path}")

    # Print summary
    print_summary(conversations, category, elapsed)

    # Exit with appropriate code
    sys.exit(0 if len(conversations) == args.count else 1)


if __name__ == '__main__':
    asyncio.run(main())
