#!/usr/bin/env python3
"""Generate all risk categories concurrently with a simple live progress display.

Usage:
    python -m data.generation.scripts.generate_all --generator vllm --count 200000
    python -m data.generation.scripts.generate_all --generator vllm --count 200000 --resume
    python -m data.generation.scripts.generate_all --generator bedrock --count 5000
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
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from data.generation.scripts.generate import count_existing, create_generator, generate_batch, load_config, write_conversations
from data.generation.validators.schemas import RiskCategory

ALL_CATEGORIES = [c.value for c in RiskCategory]


async def probe_vllm(base_url: str, model: str, console: Console) -> bool:
    """Check vLLM is reachable and the model is loaded. Prints a clear error if not."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{base_url}/models")
            r.raise_for_status()
            available = [m["id"] for m in r.json().get("data", [])]
    except Exception as e:
        console.print(f"[bold red]ERROR:[/] Cannot reach vLLM at {base_url}\n  {e}")
        return False

    if model not in available:
        console.print(
            f"[bold red]ERROR:[/] Model [bold]{model}[/] not found on vLLM server.\n"
            f"  Available: {available}\n"
            f"  Fix: update [bold]model_vllm[/] in data/generation/config.yaml to match."
        )
        return False

    console.print(f"[green]✓[/] vLLM OK — model [bold]{model}[/] at {base_url}")
    return True


async def run_category(cat: str, target: int, remaining: int, generator_type: str,
                       config: dict, output_dir: Path, resume: bool,
                       progress: Progress, task_id: int) -> tuple[str, int]:
    already = target - remaining
    start = time.monotonic()
    last_n = [0]

    def on_progress(n_done: int, n_failed: int) -> None:
        if n_done - last_n[0] >= 5:
            elapsed = time.monotonic() - start
            rate = n_done / elapsed if elapsed > 0 else 0
            progress.update(task_id, completed=already + n_done, rate=rate)
            last_n[0] = n_done

    generator = create_generator(generator_type, config)
    concurrency = config.get("generation", {}).get("concurrency", 10)

    convs = await generate_batch(
        generator, RiskCategory(cat), remaining, config,
        concurrency=concurrency, on_progress=on_progress,
    )

    output_path = output_dir / f"{cat}.jsonl"
    if convs:
        write_conversations(convs, str(output_path), append=resume and already > 0)

    progress.update(task_id, completed=already + len(convs))
    return cat, already + len(convs)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", default="vllm", choices=["claude", "openai", "bedrock", "vllm"])
    parser.add_argument("--count", type=int, default=200_000)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--categories", nargs="+", default=ALL_CATEGORIES, choices=ALL_CATEGORIES)
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

    # Probe vLLM before starting so failures are visible immediately
    if args.generator == "vllm":
        gen_cfg = config.get("generation", {})
        base_url = gen_cfg.get("vllm_base_url", "http://localhost:8000/v1").rstrip("/")
        model = gen_cfg.get("model_vllm", "Qwen/Qwen2.5-72B-Instruct-AWQ")
        ok = await probe_vllm(base_url, model, console)
        if not ok:
            sys.exit(1)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description:<22}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("[green]{task.fields[rate]:.0f}/s[/]"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
    )

    tasks = []
    for cat in args.categories:
        already = count_existing(str(output_dir / f"{cat}.jsonl")) if args.resume else 0
        remaining = max(0, args.count - already)
        task_id = progress.add_task(cat, total=args.count, completed=already, rate=0.0)
        tasks.append((cat, args.count, remaining, task_id))

    console.print(f"[bold]Horizon Generator[/]  generator=[cyan]{args.generator}[/]  "
                  f"target=[bold]{args.count:,}[/] per category  output=[dim]{output_dir}[/]\n")

    with Live(progress, refresh_per_second=2):
        coros = [
            run_category(cat, target, remaining, args.generator, config,
                         output_dir, args.resume, progress, task_id)
            for cat, target, remaining, task_id in tasks
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

    console.print("\n[bold green]Done![/]")
    total = 0
    for r in results:
        if isinstance(r, Exception):
            console.print(f"  [red]error:[/] {r}")
        else:
            cat, n = r
            console.print(f"  {cat:<22} [green]{n:,}[/]")
            total += n
    console.print(f"\n  [bold]total  {total:,}[/]")


if __name__ == "__main__":
    asyncio.run(main())
