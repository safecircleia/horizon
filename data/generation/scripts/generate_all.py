#!/usr/bin/env python3
"""TUI for generating all risk categories concurrently.

Each category gets its own progress bar, updating live in a Rich panel.
A summary table prints on completion.

Usage:
    python -m data.generation.scripts.generate_all --generator vllm --count 200000
    python -m data.generation.scripts.generate_all --generator bedrock --count 5000
    python -m data.generation.scripts.generate_all --generator vllm --count 200000 --resume
    python -m data.generation.scripts.generate_all --generator vllm --categories grooming bullying
"""

import argparse
import asyncio
import datetime
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from data.generation.scripts.generate import (
    count_existing,
    create_generator,
    generate_batch,
    load_config,
    write_conversations,
)
from data.generation.validators.schemas import RiskCategory

ALL_CATEGORIES = [c.value for c in RiskCategory]

CATEGORY_COLORS = {
    "grooming":           "red",
    "bullying":           "orange3",
    "sexual_content":     "bright_red",
    "isolation":          "yellow",
    "personal_info":      "cyan",
    "platform_migration": "blue",
    "threats":            "magenta",
    "benign":             "green",
}


def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=35),
        MofNCompleteColumn(),
        TextColumn("[dim]•"),
        TextColumn("[green]{task.fields[rate]:.1f}[/green][dim] conv/s"),
        TextColumn("[dim]•"),
        TimeElapsedColumn(),
        TextColumn("[dim]eta"),
        TimeRemainingColumn(),
        refresh_per_second=4,
        expand=False,
    )


def make_header(generator: str, count: int, concurrency: int, categories: list[str]) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim")
    t.add_column()
    t.add_row("generator",   f"[bold cyan]{generator}[/]")
    t.add_row("target",      f"[bold]{count:,}[/] conversations per category")
    t.add_row("concurrency", f"{concurrency} workers/category")
    t.add_row("categories",  ", ".join(f"[{CATEGORY_COLORS.get(c, 'white')}]{c}[/]" for c in categories))
    return Panel(t, title="[bold]Horizon Dataset Generator[/]", border_style="bright_blue")


def make_summary(results: list[tuple[str, int, int, float]]) -> Panel:
    """results: list of (category, target, written, elapsed_s)"""
    t = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    t.add_column("Category",    style="bold")
    t.add_column("Written",     justify="right")
    t.add_column("Target",      justify="right", style="dim")
    t.add_column("Rate",        justify="right")
    t.add_column("Time",        justify="right", style="dim")

    total = 0
    for cat, target, n, elapsed in results:
        color = CATEGORY_COLORS.get(cat, "white")
        rate = n / elapsed if elapsed > 0 else 0
        mins, secs = divmod(int(elapsed), 60)
        t.add_row(
            f"[{color}]{cat}[/]",
            f"[green]{n:,}[/]",
            f"{target:,}",
            f"{rate:.1f} conv/s",
            f"{mins}m {secs:02d}s",
        )
        total += n

    t.add_section()
    t.add_row("[bold]TOTAL[/]", f"[bold green]{total:,}[/]", "", "", "")
    return Panel(t, title="[bold green]Generation Complete[/]", border_style="green")


async def run_category(
    category_str: str,
    target: int,
    generator_type: str,
    config: dict,
    output_dir: Path,
    resume: bool,
    progress: Progress,
    task_id: TaskID,
    start_time: float,
) -> tuple[str, int, int, float]:
    """Run one category; returns (category, target, n_written, elapsed_s)."""
    category = RiskCategory(category_str)
    output_path = output_dir / f"{category_str}.jsonl"
    concurrency = config.get("generation", {}).get("concurrency", 10)

    already_have = count_existing(str(output_path)) if resume else 0
    remaining = max(0, target - already_have)

    progress.update(task_id, completed=already_have)

    if remaining == 0:
        progress.update(task_id, description=f"[dim]{category_str:<20}[/] [green]done[/]")
        return category_str, target, already_have, 0.0

    generator = create_generator(generator_type, config)
    cat_start = time.monotonic()

    def on_progress(n: int) -> None:
        elapsed = time.monotonic() - cat_start
        rate = n / elapsed if elapsed > 0 else 0.0
        progress.update(task_id, completed=already_have + n, rate=rate)

    conversations = await generate_batch(
        generator, category, remaining, config,
        concurrency=concurrency,
        on_progress=on_progress,
    )

    elapsed = time.monotonic() - cat_start

    if conversations:
        write_conversations(conversations, str(output_path), append=resume and already_have > 0)

    color = CATEGORY_COLORS.get(category_str, "white")
    progress.update(
        task_id,
        completed=already_have + len(conversations),
        description=f"[{color}]{category_str:<20}[/] [green]✓[/]",
    )
    return category_str, target, already_have + len(conversations), elapsed


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all categories with live TUI")
    parser.add_argument("--generator", default="vllm",
                        choices=["claude", "openai", "bedrock", "vllm"])
    parser.add_argument("--count", type=int, default=200_000,
                        help="Target conversations per category (default: 200000)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Async workers per category (overrides config)")
    parser.add_argument("--categories", nargs="+", default=ALL_CATEGORIES,
                        choices=ALL_CATEGORIES, metavar="CATEGORY")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--config", default="data/generation/config.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    if args.concurrency is not None:
        config.setdefault("generation", {})["concurrency"] = args.concurrency

    concurrency = config.get("generation", {}).get("concurrency", 10)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    console = Console()
    progress = make_progress()

    # Register one task per category
    task_ids: dict[str, TaskID] = {}
    for cat in args.categories:
        color = CATEGORY_COLORS.get(cat, "white")
        task_ids[cat] = progress.add_task(
            f"[{color}]{cat:<20}[/]",
            total=args.count,
            rate=0.0,
        )

    layout = Layout()
    layout.split_column(
        Layout(make_header(args.generator, args.count, concurrency, args.categories), size=7),
        Layout(Panel(progress, border_style="bright_blue"), name="bars"),
    )

    global_start = time.monotonic()

    with Live(layout, console=console, refresh_per_second=4, screen=False):
        tasks = [
            run_category(
                cat, args.count, args.generator, config,
                output_dir, args.resume,
                progress, task_ids[cat], global_start,
            )
            for cat in args.categories
        ]
        results = await asyncio.gather(*tasks)

    console.print(make_summary(list(results)))


if __name__ == "__main__":
    asyncio.run(main())
