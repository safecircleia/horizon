#!/usr/bin/env python3
"""Generate all risk categories concurrently with one progress bar per category.

Each category gets a fixed tqdm row — no overlapping output.

Usage:
    # vLLM (local H100, recommended for large runs)
    python -m data.generation.scripts.generate_all \
        --generator vllm --count 200000 --concurrency 150

    # Cloud generators
    python -m data.generation.scripts.generate_all \
        --generator bedrock --count 5000

    # Resume partial run
    python -m data.generation.scripts.generate_all \
        --generator vllm --count 200000 --resume

    # Single category
    python -m data.generation.scripts.generate_all \
        --generator vllm --count 50000 --categories grooming bullying
"""

import argparse
import asyncio
import datetime
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from data.generation.scripts.generate import (
    count_existing,
    create_generator,
    generate_batch,
    load_config,
    write_conversations,
)
from data.generation.validators.schemas import RiskCategory

ALL_CATEGORIES = [c.value for c in RiskCategory]


async def run_category(
    category_str: str,
    count: int,
    generator_type: str,
    config: dict,
    output_dir: Path,
    resume: bool,
    position: int,
) -> tuple[str, int, float]:
    """Run generation for one category; returns (category, n_written, elapsed)."""
    category = RiskCategory(category_str)
    output_path = output_dir / f"{category_str}.jsonl"
    concurrency = config.get("generation", {}).get("concurrency", 10)

    already_have = count_existing(str(output_path)) if resume else 0
    remaining = max(0, count - already_have)

    if remaining == 0:
        return category_str, 0, 0.0

    generator = create_generator(generator_type, config)
    start = datetime.datetime.now()

    conversations = await generate_batch(
        generator,
        category,
        remaining,
        config,
        concurrency=concurrency,
        tqdm_position=position,
    )

    elapsed = (datetime.datetime.now() - start).total_seconds()

    if conversations:
        write_conversations(conversations, str(output_path), append=resume and already_have > 0)

    return category_str, len(conversations), elapsed


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all categories concurrently")
    parser.add_argument("--generator", default="vllm",
                        choices=["claude", "openai", "bedrock", "vllm"])
    parser.add_argument("--count", type=int, default=200_000,
                        help="Target conversations per category (default: 200000)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Async workers per category (overrides config)")
    parser.add_argument("--categories", nargs="+", default=ALL_CATEGORIES,
                        choices=ALL_CATEGORIES, metavar="CATEGORY")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--config", default="data/generation/config.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-generated conversations")
    args = parser.parse_args()

    load_dotenv()

    config = load_config(args.config)
    if args.concurrency is not None:
        config.setdefault("generation", {})["concurrency"] = args.concurrency

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generator : {args.generator}")
    print(f"Categories: {', '.join(args.categories)}")
    print(f"Target    : {args.count:,} conversations each")
    print(f"Concurrency: {config.get('generation', {}).get('concurrency', 10)} workers/category")
    print(f"Output    : {output_dir}/")
    print()

    tasks = [
        run_category(cat, args.count, args.generator, config,
                     output_dir, args.resume, pos)
        for pos, cat in enumerate(args.categories)
    ]

    results = await asyncio.gather(*tasks)

    # Print final summary below all bars (add blank lines to clear bar area)
    print("\n" * len(args.categories))
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    total_written = 0
    for cat, n, elapsed in results:
        rate = n / elapsed if elapsed > 0 else 0
        print(f"  {cat:<22} {n:>7,} conv  {rate:>6.1f} conv/s")
        total_written += n
    print("-" * 60)
    print(f"  {'TOTAL':<22} {total_written:>7,} conv")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
