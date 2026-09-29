#!/usr/bin/env python3
"""Generate all risk categories concurrently with live progress bars.

Usage:
    python -m data.generation.scripts.generate_all --generator vllm --count 200000
    python -m data.generation.scripts.generate_all --generator vllm --count 200000 --resume
    python -m data.generation.scripts.generate_all --generator vllm --categories grooming bullying
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from data.generation.scripts.generate import (
    count_existing,
    create_generator,
    generate_batch,
    load_config,
    write_conversations,
)
from data.generation.validators.schemas import RiskCategory

ALL_CATEGORIES = [c.value for c in RiskCategory]


async def probe_vllm(base_url: str, model: str, console: Console) -> bool:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{base_url}/models")
            r.raise_for_status()
            available = [m["id"] for m in r.json().get("data", [])]
    except Exception as e:
        console.print(f"[red]ERROR:[/] Cannot reach vLLM at {base_url} — {e}")
        return False
    if model not in available:
        console.print(
            f"[red]ERROR:[/] Model '{model}' not loaded. Available: {available}"
        )
        return False
    console.print(f"[green]✓[/] vLLM OK — {model}")
    return True


async def run_category(
    cat: str,
    target: int,
    already: int,
    generator,
    config: dict,
    output_dir: Path,
    resume: bool,
    progress: Progress,
    task_id,
) -> tuple[str, int]:
    remaining = max(0, target - already)
    if remaining == 0:
        progress.update(task_id, completed=target)
        return cat, target

    concurrency = config.get("generation", {}).get("concurrency", 10)
    start = time.monotonic()

    def on_progress(n_done: int, n_failed: int) -> None:
        rate = n_done / max(time.monotonic() - start, 0.1)
        progress.update(task_id, completed=already + n_done, rate=rate, failed=n_failed)

    convs = await generate_batch(
        generator,
        RiskCategory(cat),
        remaining,
        config,
        concurrency=concurrency,
        on_progress=on_progress,
    )

    if convs:
        write_conversations(
            convs, str(output_dir / f"{cat}.jsonl"), append=resume and already > 0
        )

    total = already + len(convs)
    progress.update(task_id, completed=total)
    return cat, total


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator", default="vllm", choices=["claude", "openai", "bedrock", "vllm"]
    )
    parser.add_argument("--count", type=int, default=200_000)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument(
        "--categories", nargs="+", default=ALL_CATEGORIES, choices=ALL_CATEGORIES
    )
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--config", default="data/generation/config.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    if args.concurrency is not None:
        config.setdefault("generation", {})["concurrency"] = args.concurrency

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    console = Console()

    if args.generator == "vllm":
        gen_cfg = config.get("generation", {})
        base_url = gen_cfg.get("vllm_base_url", "http://localhost:8000/v1").rstrip("/")
        model = gen_cfg.get("model_vllm", "Qwen/Qwen2.5-72B-Instruct-AWQ")
        if not await probe_vllm(base_url, model, console):
            sys.exit(1)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description:<22}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("[green]{task.fields[rate]:.0f}/s[/]"),
        TextColumn("[red]✗{task.fields[failed]}[/]"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
    )

    jobs = []
    for cat in args.categories:
        already = count_existing(str(output_dir / f"{cat}.jsonl")) if args.resume else 0
        task_id = progress.add_task(
            cat, total=args.count, completed=already, rate=0.0, failed=0
        )
        jobs.append((cat, args.count, already, task_id))

    console.print(
        f"[bold]Horizon Generator[/]  generator=[cyan]{args.generator}[/]  "
        f"target=[bold]{args.count:,}[/]  "
        f"concurrency=[bold]{config['generation'].get('concurrency', 10)}[/]\n"
    )

    generator = create_generator(args.generator, config)

    with Live(progress, refresh_per_second=2, console=console):
        coros = [
            run_category(
                cat,
                target,
                already,
                generator,
                config,
                output_dir,
                args.resume,
                progress,
                task_id,
            )
            for cat, target, already, task_id in jobs
        ]
        results = await asyncio.gather(*coros)

    console.print("\n[bold green]Done![/]")
    total = 0
    for cat, n in results:
        console.print(f"  {cat:<22} [green]{n:,}[/]")
        total += n
    console.print(f"\n  [bold]total  {total:,}[/]")


if __name__ == "__main__":
    asyncio.run(main())
