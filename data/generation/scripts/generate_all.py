#!/usr/bin/env python3
"""Textual TUI for generating all risk categories concurrently.

Keyboard shortcuts:
  p / space   Pause / resume all workers
  q / ctrl+c  Quit (saves progress, workers finish current request)
  r           Restart failed categories
  l           Toggle error log panel
  s           Save current counts to disk immediately

Usage:
    python -m data.generation.scripts.generate_all --generator vllm --count 200000
    python -m data.generation.scripts.generate_all --generator bedrock --count 5000
    python -m data.generation.scripts.generate_all --generator vllm --count 200000 --resume
    python -m data.generation.scripts.generate_all --generator vllm --categories grooming bullying
"""

import argparse
import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    Log,
    ProgressBar,
    Static,
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

CATEGORY_COLORS = {
    "grooming":           "red",
    "bullying":           "dark_orange",
    "sexual_content":     "red1",
    "isolation":          "yellow",
    "personal_info":      "cyan",
    "platform_migration": "cornflower_blue",
    "threats":            "magenta",
    "benign":             "green",
}


@dataclass
class CatState:
    name: str
    target: int
    completed: int = 0
    failed: int = 0
    status: str = "waiting"   # waiting | running | paused | done | error
    rate: float = 0.0
    start_time: float = 0.0
    _rate_window: deque = field(default_factory=lambda: deque(maxlen=30))

    def tick(self, completed: int, failed: int) -> None:
        now = time.monotonic()
        self._rate_window.append((now, completed))
        self.completed = completed
        self.failed = failed
        if len(self._rate_window) >= 2:
            t0, c0 = self._rate_window[0]
            t1, c1 = self._rate_window[-1]
            dt = t1 - t0
            self.rate = (c1 - c0) / dt if dt > 0 else 0.0

    @property
    def pct(self) -> float:
        return min(self.completed / self.target, 1.0) if self.target else 0.0

    @property
    def eta_s(self) -> float:
        if self.rate <= 0:
            return -1
        remaining = self.target - self.completed
        return remaining / self.rate


class GeneratorApp(App):
    CSS = """
    Screen { background: $surface; }

    #title {
        content-align: center middle;
        background: $accent;
        color: $text;
        height: 3;
        text-style: bold;
    }

    #config-bar {
        height: 3;
        background: $panel;
        padding: 0 2;
        content-align: left middle;
    }

    #table-container {
        height: 1fr;
        border: solid $accent;
        padding: 0 1;
    }

    #log-panel {
        height: 10;
        border: solid $warning;
        display: none;
    }

    #log-panel.visible {
        display: block;
    }

    DataTable {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("p,space", "toggle_pause", "Pause/Resume", show=True),
        Binding("r",        "restart_failed", "Restart failed", show=True),
        Binding("s",        "save_now",       "Save now",       show=True),
        Binding("l",        "toggle_log",     "Toggle log",     show=True),
        Binding("q",        "quit",           "Quit",           show=True),
    ]

    paused: reactive[bool] = reactive(False)

    def __init__(
        self,
        generator_type: str,
        count: int,
        categories: list[str],
        config: dict,
        output_dir: Path,
        resume: bool,
    ):
        super().__init__()
        self.generator_type = generator_type
        self.count = count
        self.categories = categories
        self.config = config
        self.output_dir = output_dir
        self.resume = resume
        self.states: dict[str, CatState] = {}
        self._pause_events: dict[str, asyncio.Event] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._conversations: dict[str, list] = {c: [] for c in categories}

        concurrency = config.get("generation", {}).get("concurrency", 10)
        self._concurrency = concurrency

    def compose(self) -> ComposeResult:
        concurrency = self.config.get("generation", {}).get("concurrency", 10)
        yield Header(show_clock=True)
        yield Static(
            f"  generator=[bold cyan]{self.generator_type}[/]  "
            f"target=[bold]{self.count:,}[/] per category  "
            f"concurrency=[bold]{concurrency}[/]  "
            f"output=[dim]{self.output_dir}[/]",
            id="config-bar",
        )
        with Vertical(id="table-container"):
            yield DataTable(id="cattable", show_cursor=False)
        yield Log(id="log-panel", max_lines=200, highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#cattable", DataTable)
        table.add_columns("Category", "Progress", "Done", "Target", "Rate", "ETA", "Errors", "Status")

        for cat in self.categories:
            already = count_existing(str(self.output_dir / f"{cat}.jsonl")) if self.resume else 0
            state = CatState(name=cat, target=self.count, completed=already)
            self.states[cat] = state
            self._pause_events[cat] = asyncio.Event()
            self._pause_events[cat].set()  # not paused initially
            table.add_row(
                cat, "0%", str(already), str(self.count),
                "—", "—", "0", "waiting",
                key=cat,
            )

        self._start_all()
        self.set_interval(0.25, self._refresh_table)

    def _start_all(self) -> None:
        for cat in self.categories:
            if cat not in self._tasks or self._tasks[cat].done():
                self._tasks[cat] = asyncio.get_event_loop().create_task(
                    self._run_category(cat)
                )

    async def _run_category(self, cat: str) -> None:
        state = self.states[cat]
        pause_ev = self._pause_events[cat]
        output_path = self.output_dir / f"{cat}.jsonl"
        already = state.completed

        remaining = max(0, self.count - already)
        if remaining == 0:
            state.status = "done"
            return

        state.status = "running"
        state.start_time = time.monotonic()

        try:
            generator = create_generator(self.generator_type, self.config)
        except Exception as e:
            state.status = "error"
            self._log(f"[red]{cat}[/]: failed to create generator — {e}")
            return

        def on_progress(n_done: int, n_failed: int) -> None:
            state.tick(already + n_done, n_failed)

        # Wrap generate_batch to respect pause
        original_generate = generator.generate

        async def pauseable_generate(prompt):
            await pause_ev.wait()
            return await original_generate(prompt)

        generator.generate = pauseable_generate

        try:
            convs = await generate_batch(
                generator,
                RiskCategory(cat),
                remaining,
                self.config,
                concurrency=self._concurrency,
                on_progress=on_progress,
            )
            self._conversations[cat].extend(convs)
            write_conversations(
                convs, str(output_path),
                append=self.resume and already > 0
            )
            state.status = "done"
            state.completed = already + len(convs)
        except asyncio.CancelledError:
            state.status = "paused"
            # flush what we have
            if self._conversations[cat]:
                write_conversations(
                    self._conversations[cat], str(output_path),
                    append=self.resume and already > 0
                )
        except Exception as e:
            state.status = "error"
            self._log(f"[red]{cat}[/]: {e}")

    def _refresh_table(self) -> None:
        table = self.query_one("#cattable", DataTable)
        for cat, state in self.states.items():
            pct = f"{state.pct * 100:.1f}%"
            rate = f"{state.rate:.1f}/s" if state.rate > 0 else "—"
            if state.eta_s > 0:
                m, s = divmod(int(state.eta_s), 60)
                eta = f"{m}m{s:02d}s" if m else f"{s}s"
            else:
                eta = "—"

            color = CATEGORY_COLORS.get(cat, "white")
            status_styled = {
                "waiting": "[dim]waiting[/]",
                "running": "[green]running[/]",
                "paused":  "[yellow]paused[/]",
                "done":    "[bold green]done ✓[/]",
                "error":   "[bold red]error ✗[/]",
            }.get(state.status, state.status)

            table.update_cell(cat, "Progress", pct)
            table.update_cell(cat, "Done",     f"{state.completed:,}")
            table.update_cell(cat, "Rate",     rate)
            table.update_cell(cat, "ETA",      eta)
            table.update_cell(cat, "Errors",   str(state.failed))
            table.update_cell(cat, "Status",   status_styled)

    def _log(self, msg: str) -> None:
        log = self.query_one("#log-panel", Log)
        log.write_line(msg)
        # Auto-show log on first error
        if "red" in msg and "visible" not in log.classes:
            log.add_class("visible")

    # ── Actions ──────────────────────────────────────────────────────────────

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        for cat, ev in self._pause_events.items():
            if self.paused:
                ev.clear()
                self.states[cat].status = "paused" if self.states[cat].status == "running" else self.states[cat].status
            else:
                ev.set()
                self.states[cat].status = "running" if self.states[cat].status == "paused" else self.states[cat].status
        self._log(f"[yellow]{'Paused' if self.paused else 'Resumed'} all workers[/]")

    def action_restart_failed(self) -> None:
        restarted = 0
        for cat, state in self.states.items():
            if state.status == "error":
                state.status = "waiting"
                state.failed = 0
                self._pause_events[cat].set()
                self._tasks[cat] = asyncio.get_event_loop().create_task(
                    self._run_category(cat)
                )
                restarted += 1
        if restarted:
            self._log(f"[cyan]Restarted {restarted} failed categories[/]")

    def action_save_now(self) -> None:
        saved = 0
        for cat, convs in self._conversations.items():
            if convs:
                output_path = self.output_dir / f"{cat}.jsonl"
                write_conversations(convs, str(output_path), append=True)
                self._conversations[cat] = []
                saved += len(convs)
        self._log(f"[green]Saved {saved:,} conversations to disk[/]")

    def action_toggle_log(self) -> None:
        log = self.query_one("#log-panel", Log)
        if "visible" in log.classes:
            log.remove_class("visible")
        else:
            log.add_class("visible")

    def action_quit(self) -> None:
        self.action_save_now()
        self.exit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset generation TUI")
    parser.add_argument("--generator", default="vllm",
                        choices=["claude", "openai", "bedrock", "vllm"])
    parser.add_argument("--count", type=int, default=200_000)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--categories", nargs="+", default=ALL_CATEGORIES,
                        choices=ALL_CATEGORIES, metavar="CATEGORY")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--config", default="data/generation/config.yaml")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_dotenv()

    config = load_config(args.config)
    if args.concurrency is not None:
        config.setdefault("generation", {})["concurrency"] = args.concurrency

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    app = GeneratorApp(
        generator_type=args.generator,
        count=args.count,
        categories=args.categories,
        config=config,
        output_dir=output_dir,
        resume=args.resume,
    )
    app.run()
