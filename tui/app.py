"""Horizon TUI — manage training, jobs, evaluation, and deployment."""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Log,
    Static,
)

from .slurm import squeue, scancel, tail_log
from . import actions


# ── Sidebar: running jobs ────────────────────────────────────────────────────

class JobsSidebar(Vertical):
    DEFAULT_CSS = """
    JobsSidebar {
        width: 38;
        border-left: solid $accent;
        padding: 0 1;
    }
    JobsSidebar .title {
        text-style: bold;
        padding: 1 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("SLURM Jobs", classes="title")
        yield DataTable(id="jobs-table")

    def on_mount(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        table.add_columns("ID", "Name", "State", "Time")
        table.cursor_type = "row"
        self.refresh_jobs()

    def refresh_jobs(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        table.clear()
        import os
        jobs = squeue(user=os.environ.get("USER"))
        for job in jobs:
            table.add_row(job.job_id, job.name[:14], job.state[:7], job.time)


# ── Main menu ────────────────────────────────────────────────────────────────

MENU_ITEMS = [
    ("train", "Submit Training Job"),
    ("merge", "Merge LoRA Adapter"),
    ("export", "Export to LiteRT-LM"),
    ("evaluate", "Run Evaluation"),
    ("upload", "Upload Models (HF + R2)"),
    ("jobs", "View SLURM Jobs"),
    ("cleanup", "Cleanup Experiments"),
    ("maintenance", "Maintenance"),
]


class MainMenu(ListView):
    DEFAULT_CSS = """
    MainMenu {
        width: 1fr;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        for key, label in MENU_ITEMS:
            yield ListItem(Label(f"  {label}"), id=f"menu-{key}")


# ── Action panel (right of menu, left of sidebar) ────────────────────────────

class ActionPanel(VerticalScroll):
    DEFAULT_CSS = """
    ActionPanel {
        width: 2fr;
        border-left: solid $surface;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Select an action from the menu.", id="panel-content")
        yield Log(id="panel-log", auto_scroll=True)

    def on_mount(self) -> None:
        self.query_one("#panel-log", Log).display = False

    def show_content(self, text: str) -> None:
        self.query_one("#panel-content", Static).update(text)
        self.query_one("#panel-log", Log).display = False

    def show_log(self, text: str) -> None:
        log = self.query_one("#panel-log", Log)
        log.display = True
        log.clear()
        log.write(text)


# ── App ──────────────────────────────────────────────────────────────────────

class HorizonApp(App):
    CSS = """
    Screen {
        layout: horizontal;
    }
    #main-area {
        width: 1fr;
    }
    """

    TITLE = "Horizon"
    SUB_TITLE = "SafeCircle Model Management"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh_jobs", "Refresh Jobs"),
        Binding("escape", "back", "Back"),
    ]

    current_view = reactive("menu")
    _input_buffer: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="main-area"):
                yield MainMenu(id="main-menu")
                yield ActionPanel(id="action-panel")
            yield JobsSidebar(id="jobs-sidebar")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#action-panel").display = False
        self.set_interval(15, self._auto_refresh_jobs)

    def _auto_refresh_jobs(self) -> None:
        self.query_one(JobsSidebar).refresh_jobs()

    def action_refresh_jobs(self) -> None:
        self.query_one(JobsSidebar).refresh_jobs()
        self.notify("Jobs refreshed")

    def action_back(self) -> None:
        self.query_one("#main-menu").display = True
        self.query_one("#action-panel").display = False
        self.current_view = "menu"

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        action = item_id.replace("menu-", "")
        self._handle_action(action)

    def _handle_action(self, action: str) -> None:
        panel = self.query_one(ActionPanel)
        self.query_one("#main-menu").display = False
        panel.display = True
        self.current_view = action

        if action == "train":
            self._show_train(panel)
        elif action == "merge":
            self._show_merge(panel)
        elif action == "export":
            self._show_export(panel)
        elif action == "evaluate":
            self._show_evaluate(panel)
        elif action == "upload":
            self._show_upload(panel)
        elif action == "jobs":
            self._show_jobs(panel)
        elif action == "cleanup":
            self._show_cleanup(panel)
        elif action == "maintenance":
            self._show_maintenance(panel)

    def _show_train(self, panel: ActionPanel) -> None:
        lines = ["[b]Submit Training Job[/b]\n"]
        for i, name in enumerate(actions.TRAIN_CONFIGS, 1):
            lines.append(f"  [{i}] {name}")
        lines.append("\nPress a number key to submit.")
        panel.show_content("\n".join(lines))
        self._train_mode = True

    def _show_merge(self, panel: ActionPanel) -> None:
        experiments = actions.list_experiments()
        if not experiments:
            panel.show_content("[b]Merge LoRA[/b]\n\nNo experiments found.")
            return
        lines = ["[b]Merge LoRA Adapter[/b]\n"]
        finals = [e for e in experiments if (Path(e["path"]) / "final").exists()]
        if not finals:
            lines.append("No experiments with a /final checkpoint found.")
        else:
            for i, exp in enumerate(finals, 1):
                lines.append(f"  [{i}] {exp['name']} ({exp['size']})")
            lines.append("\nPress a number key to submit merge job.")
        panel.show_content("\n".join(lines))
        self._merge_candidates = finals

    def _show_export(self, panel: ActionPanel) -> None:
        lines = [
            "[b]Export to LiteRT-LM[/b]\n",
            "  [1] Edge 2B (Gemma 4 E2B)",
            "  [2] Edge 4B (Gemma 4 E4B)",
            "  [3] Mobile (Gemma 3 1B)",
            "\nPress a number key to submit export job.",
        ]
        panel.show_content("\n".join(lines))
        self._export_mode = True

    def _show_evaluate(self, panel: ActionPanel) -> None:
        experiments = actions.list_experiments()
        finals = [e for e in experiments if (Path(e["path"]) / "final").exists()]
        if not finals:
            panel.show_content("[b]Evaluate[/b]\n\nNo experiments with /final checkpoint found.")
            return
        lines = ["[b]Run Evaluation[/b]\n"]
        for i, exp in enumerate(finals, 1):
            lines.append(f"  [{i}] {exp['name']} ({exp['size']})")
        lines.append("\nPress a number key to submit evaluation job.")
        panel.show_content("\n".join(lines))
        self._eval_candidates = finals

    def _show_upload(self, panel: ActionPanel) -> None:
        lines = [
            "[b]Upload Models (HF + R2)[/b]\n",
            "  [1] edge-2b",
            "  [2] edge-4b",
            "  [3] mobile",
            "  [4] full",
            "  [5] gguf",
            "  [6] all",
            "\nPress a number key. You'll be prompted for the version.",
        ]
        panel.show_content("\n".join(lines))
        self._upload_mode = True

    def _show_jobs(self, panel: ActionPanel) -> None:
        import os
        jobs = squeue(user=os.environ.get("USER"))
        if not jobs:
            panel.show_content("[b]SLURM Jobs[/b]\n\nNo running jobs.")
            return
        lines = ["[b]SLURM Jobs[/b]\n"]
        for i, job in enumerate(jobs, 1):
            lines.append(f"  [{i}] {job.job_id} | {job.name} | {job.state} | {job.time} | {job.partition}")
        lines.append("\nPress a number to view log, or 'c' + number to cancel.")
        panel.show_content("\n".join(lines))
        self._jobs_list = jobs

    def _show_cleanup(self, panel: ActionPanel) -> None:
        experiments = actions.list_experiments()
        models = actions.list_models()
        lines = ["[b]Cleanup[/b]\n", "[u]Experiments:[/u]"]
        for i, e in enumerate(experiments, 1):
            lines.append(f"  [{i}] {e['name']} — {e['size']}")
        lines.append(f"\n[u]Models:[/u]")
        for m in models:
            lines.append(f"  {m['name']} — {m['size']}")
        lines.append("\nPress a number to delete an experiment.")
        panel.show_content("\n".join(lines))
        self._cleanup_experiments = experiments

    def _show_maintenance(self, panel: ActionPanel) -> None:
        lines = [
            "[b]Maintenance[/b]\n",
            "  [1] Update dependencies (uv pip install -r requirements.txt)",
            "  [2] Clear tokenized cache",
            "\nPress a number to run.",
        ]
        panel.show_content("\n".join(lines))
        self._maintenance_mode = True

    def on_key(self, event) -> None:
        if self.current_view == "menu":
            return

        key = event.key
        panel = self.query_one(ActionPanel)

        # Buffer digits, execute on Enter
        if key.isdigit():
            self._input_buffer += key
            self.sub_title = f"Selection: {self._input_buffer} (Enter to confirm)"
            return

        if key == "backspace" and self._input_buffer:
            self._input_buffer = self._input_buffer[:-1]
            self.sub_title = f"Selection: {self._input_buffer}" if self._input_buffer else "SafeCircle Model Management"
            return

        if key != "enter" or not self._input_buffer:
            return

        idx = int(self._input_buffer) - 1
        self._input_buffer = ""
        self.sub_title = "SafeCircle Model Management"

        if self.current_view == "train":
            configs = list(actions.TRAIN_CONFIGS.keys())
            if 0 <= idx < len(configs):
                ok, msg = actions.submit_training(configs[idx])
                self.notify(f"{'Submitted' if ok else 'Failed'}: {msg}")
                self.action_refresh_jobs()

        elif self.current_view == "merge":
            candidates = getattr(self, "_merge_candidates", [])
            if 0 <= idx < len(candidates):
                exp = candidates[idx]
                checkpoint = f"experiments/{exp['name']}/final"
                if "edge-2b" in exp["name"]:
                    output = "models/horizon-edge-2b-merged"
                elif "edge-4b" in exp["name"]:
                    output = "models/horizon-edge-4b-merged"
                elif "mobile" in exp["name"]:
                    output = "models/horizon-mobile-merged"
                else:
                    output = "models/horizon-full-merged"
                ok, msg = actions.submit_merge(checkpoint, output)
                self.notify(f"{'Submitted' if ok else 'Failed'}: {msg}")
                self.action_refresh_jobs()

        elif self.current_view == "export":
            export_map = {0: "e2b", 1: "e4b"}
            if idx in export_map:
                ok, msg = actions.submit_export_edge(export_map[idx])
                self.notify(f"{'Submitted' if ok else 'Failed'}: {msg}")
                self.action_refresh_jobs()
            elif idx == 2:
                panel.show_content("Mobile export requires a checkpoint path.\nUse: sbatch --export=CHECKPOINT=... slurm/export_litert.sbatch")

        elif self.current_view == "evaluate":
            candidates = getattr(self, "_eval_candidates", [])
            if 0 <= idx < len(candidates):
                checkpoint = f"experiments/{candidates[idx]['name']}/final"
                ok, msg = actions.submit_evaluate(checkpoint)
                self.notify(f"{'Submitted' if ok else 'Failed'}: {msg}")
                self.action_refresh_jobs()

        elif self.current_view == "jobs":
            jobs_list = getattr(self, "_jobs_list", [])
            if 0 <= idx < len(jobs_list):
                log_text = tail_log(jobs_list[idx].job_id, lines=80)
                panel.show_log(log_text)

        elif self.current_view == "cleanup":
            experiments = getattr(self, "_cleanup_experiments", [])
            if 0 <= idx < len(experiments):
                ok, msg = actions.delete_experiment(experiments[idx]["path"])
                self.notify(msg)
                self._show_cleanup(panel)

        elif self.current_view == "maintenance":
            if idx == 0:
                self.notify("Updating dependencies...")
                ok, msg = actions.update_deps()
                panel.show_log(msg)
            elif idx == 1:
                ok, msg = actions.clear_tokenized_cache()
                self.notify(msg)


def main():
    app = HorizonApp()
    app.run()


if __name__ == "__main__":
    main()
