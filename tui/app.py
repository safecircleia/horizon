"""Horizon TUI — manage training, jobs, evaluation, and deployment."""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Log,
    Static,
)

from .slurm import squeue, scancel, tail_log, tail_err_log, sinfo, sacct_recent, format_node_gpu, format_node_memory
from . import actions


# ── Job focus modal ──────────────────────────────────────────────────────────

class JobFocusModal(ModalScreen[bool]):
    """Ask user if they want to focus on the newly submitted job."""

    DEFAULT_CSS = """
    JobFocusModal {
        align: center middle;
    }
    #modal-box {
        width: 50;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #modal-buttons {
        layout: horizontal;
        height: auto;
        padding-top: 1;
    }
    #modal-buttons Button {
        margin-right: 2;
    }
    """

    def __init__(self, job_id: str, job_name: str) -> None:
        super().__init__()
        self.job_id = job_id
        self.job_name = job_name

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Static(f"[b]Job submitted:[/b] {self.job_id} ({self.job_name})")
            yield Static("\nFocus this job to watch its output?")
            with Horizontal(id="modal-buttons"):
                yield Button("Yes, watch", id="btn-yes", variant="primary")
                yield Button("No, back to menu", id="btn-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-yes")


class JobActionsModal(ModalScreen[str]):
    """Show actions for a recent/completed job: view stdout, stderr, details."""

    DEFAULT_CSS = """
    JobActionsModal {
        align: center middle;
    }
    #job-modal-box {
        width: 60;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #job-modal-box ListView {
        height: auto;
        max-height: 12;
    }
    """

    def __init__(self, job_id: str, job_name: str, job_state: str) -> None:
        super().__init__()
        self.job_id = job_id
        self.job_name = job_name
        self.job_state = job_state

    def compose(self) -> ComposeResult:
        with Vertical(id="job-modal-box"):
            yield Static(f"[b]Job {self.job_id}[/b] — {self.job_name} [{self.job_state}]")
            yield ListView(
                ListItem(Label("  Live tail stdout"), id="act-live-stdout"),
                ListItem(Label("  Live tail stderr"), id="act-live-stderr"),
                ListItem(Label("  View stdout (snapshot)"), id="act-stdout"),
                ListItem(Label("  View stderr (snapshot)"), id="act-stderr"),
                ListItem(Label("  View both (stdout + stderr)"), id="act-both"),
                ListItem(Label("  Close"), id="act-close"),
            )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.id or "act-close")


# ── Sidebar: running jobs ────────────────────────────────────────────────────

class JobsSidebar(VerticalScroll):
    DEFAULT_CSS = """
    JobsSidebar {
        width: 42;
        border-left: solid $accent;
        padding: 0 1;
    }
    JobsSidebar .title {
        text-style: bold;
        padding: 1 0;
    }
    """

    jobs: list = []

    def compose(self) -> ComposeResult:
        yield Label("SLURM Jobs (Enter to inspect)", classes="title")
        yield DataTable(id="jobs-table")
        yield Label("Cluster Nodes", classes="title")
        yield DataTable(id="nodes-table")

    def on_mount(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        table.add_columns("ID", "Name", "State", "Time")
        table.cursor_type = "row"

        nodes_table = self.query_one("#nodes-table", DataTable)
        nodes_table.add_columns("Node", "GPU", "Mem", "State")
        nodes_table.cursor_type = "row"

        self.refresh_jobs()
        self.refresh_nodes()

    def refresh_jobs(self) -> None:
        table = self.query_one("#jobs-table", DataTable)
        table.clear()
        import os
        self.jobs = squeue(user=os.environ.get("USER"))
        for job in self.jobs:
            table.add_row(job.job_id, job.name[:14], job.state[:7], job.time)

    def get_job_at_cursor(self):
        table = self.query_one("#jobs-table", DataTable)
        if table.cursor_row is not None and 0 <= table.cursor_row < len(self.jobs):
            return self.jobs[table.cursor_row]
        return None

    def refresh_nodes(self) -> None:
        table = self.query_one("#nodes-table", DataTable)
        table.clear()
        nodes = sinfo()
        for node in nodes:
            table.add_row(
                node.name.replace("slurm-", ""),
                format_node_gpu(node.gres),
                format_node_memory(node.memory),
                node.state[:7],
            )


# ── Recent jobs bar (bottom) ──────────────────────────────────────────────────

class RecentJobsBar(Vertical):
    DEFAULT_CSS = """
    RecentJobsBar {
        height: auto;
        max-height: 10;
        border-top: solid $accent;
        padding: 0 1;
    }
    RecentJobsBar .title {
        text-style: bold;
        padding: 0 0 0 0;
    }
    """

    jobs: list = []

    def compose(self) -> ComposeResult:
        yield Label("Recent Jobs (Enter to inspect)", classes="title")
        yield DataTable(id="recent-table")

    def on_mount(self) -> None:
        table = self.query_one("#recent-table", DataTable)
        table.add_columns("ID", "Name", "State", "Exit", "Elapsed", "Ended")
        table.cursor_type = "row"
        self.refresh_recent()

    def refresh_recent(self) -> None:
        table = self.query_one("#recent-table", DataTable)
        table.clear()
        self.jobs = sacct_recent(8)
        for job in self.jobs:
            state_display = job.state
            if "FAIL" in job.state or "OUT_OF_MEMORY" in job.state:
                state_display = f"[red]{job.state}[/red]"
            elif "COMPLETED" in job.state:
                state_display = f"[green]{job.state}[/green]"
            elif "TIMEOUT" in job.state or "CANCELLED" in job.state:
                state_display = f"[yellow]{job.state}[/yellow]"
            table.add_row(
                job.job_id,
                job.name[:16],
                state_display,
                job.exit_code,
                job.elapsed,
                job.end_time[-8:] if len(job.end_time) > 8 else job.end_time,
            )

    def get_job_at_cursor(self):
        """Return the RecentJob at the current cursor row, or None."""
        table = self.query_one("#recent-table", DataTable)
        if table.cursor_row is not None and 0 <= table.cursor_row < len(self.jobs):
            return self.jobs[table.cursor_row]
        return None


# ── Main menu items ──────────────────────────────────────────────────────────

MENU_ITEMS = [
    ("train", "Submit Training Job"),
    ("merge", "Merge LoRA Adapter"),
    ("export", "Export to LiteRT-LM"),
    ("evaluate", "Run Evaluation"),
    ("test", "Test LiteRT-LM Model"),
    ("upload", "Upload Models (HF + R2)"),
    ("jobs", "View SLURM Jobs"),
    ("cleanup", "Cleanup Experiments"),
    ("maintenance", "Maintenance"),
]


# ── Action panel with a sub-ListView ─────────────────────────────────────────

class ActionPanel(Vertical):
    DEFAULT_CSS = """
    ActionPanel {
        width: 2fr;
        border-left: solid $surface;
        padding: 1 2;
    }
    #panel-title {
        text-style: bold;
        padding-bottom: 1;
        height: auto;
    }
    #sub-menu {
        height: auto;
        max-height: 8;
    }
    #panel-log {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="panel-title")
        yield ListView(id="sub-menu")
        yield Log(id="panel-log", auto_scroll=True)

    def on_mount(self) -> None:
        self.query_one("#panel-log", Log).display = False

    _submenu_counter: int = 0

    def show_submenu(self, title: str, items: list[tuple[str, str]]) -> None:
        """Show a navigable submenu. items: [(id, label), ...]"""
        self.query_one("#panel-title", Static).update(f"[b]{title}[/b]")
        lv = self.query_one("#sub-menu", ListView)
        lv.clear()
        # Use a counter suffix to guarantee unique widget IDs across re-renders
        ActionPanel._submenu_counter += 1
        suffix = ActionPanel._submenu_counter
        for item_id, label in items:
            lv.append(ListItem(Label(label), id=f"{item_id}--{suffix}"))
        lv.display = True
        lv.focus()
        self.query_one("#panel-log", Log).display = False

    def show_log(self, text: str) -> None:
        log = self.query_one("#panel-log", Log)
        log.display = True
        log.clear()
        log.write(text)

    def show_message(self, title: str, message: str) -> None:
        self.query_one("#panel-title", Static).update(f"[b]{title}[/b]")
        lv = self.query_one("#sub-menu", ListView)
        lv.clear()
        lv.append(ListItem(Label(message), id="msg"))
        lv.display = True
        self.query_one("#panel-log", Log).display = False


# ── App ──────────────────────────────────────────────────────────────────────

class HorizonApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #top-area {
        layout: horizontal;
        height: 1fr;
    }
    #main-area {
        width: 1fr;
    }
    #main-menu {
        padding: 1 2;
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

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top-area"):
            with Vertical(id="main-area"):
                yield ListView(
                    *[ListItem(Label(f"  {label}"), id=f"menu-{key}") for key, label in MENU_ITEMS],
                    id="main-menu",
                )
                yield ActionPanel(id="action-panel")
            yield JobsSidebar(id="jobs-sidebar")
        yield RecentJobsBar(id="recent-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#action-panel").display = False
        self.set_interval(5, self._do_refresh)

    def _do_refresh(self) -> None:
        sidebar = self.query_one("#jobs-sidebar", JobsSidebar)
        sidebar.refresh_jobs()
        sidebar.refresh_nodes()
        self.query_one("#recent-bar", RecentJobsBar).refresh_recent()

    def action_refresh_jobs(self) -> None:
        self._do_refresh()
        self.notify("Refreshed")

    def action_back(self) -> None:
        self._stop_live_tail()
        self.query_one("#main-menu").display = True
        self.query_one("#action-panel").display = False
        self.query_one("#main-menu", ListView).focus()
        self.current_view = "menu"

    # ── Jobs table interaction (sidebar + bottom bar) ──────────────────────

    def _handle_sidebar_job_selected(self, event: DataTable.RowSelected) -> None:
        """When a running job in the sidebar is selected, open actions modal."""
        sidebar = self.query_one(JobsSidebar)
        job = sidebar.get_job_at_cursor()
        if not job:
            return

        def on_action(action: str) -> None:
            if action == "act-live-stdout":
                self._focus_job(job.job_id, "stdout")
            elif action == "act-live-stderr":
                self._focus_job(job.job_id, "stderr")
            elif action == "act-stdout":
                panel = self.query_one(ActionPanel)
                self.query_one("#main-menu").display = False
                panel.display = True
                self.current_view = "job-inspect"
                panel.show_submenu(f"Job {job.job_id} — stdout", [("job-refresh", "Refresh")])
                panel.show_log(tail_log(job.job_id, lines=100))
            elif action == "act-stderr":
                panel = self.query_one(ActionPanel)
                self.query_one("#main-menu").display = False
                panel.display = True
                self.current_view = "job-inspect"
                panel.show_submenu(f"Job {job.job_id} — stderr", [("job-refresh", "Refresh")])
                panel.show_log(tail_err_log(job.job_id, lines=100))
            elif action == "act-both":
                panel = self.query_one(ActionPanel)
                self.query_one("#main-menu").display = False
                panel.display = True
                self.current_view = "job-inspect"
                stdout = tail_log(job.job_id, lines=60)
                stderr = tail_err_log(job.job_id, lines=40)
                combined = f"{'═'*40} STDOUT {'═'*40}\n{stdout}\n{'═'*40} STDERR {'═'*40}\n{stderr}"
                panel.show_submenu(f"Job {job.job_id} — stdout + stderr", [("job-refresh", "Refresh")])
                panel.show_log(combined)

        self.push_screen(
            JobActionsModal(job.job_id, job.name, job.state),
            on_action,
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in both the sidebar jobs table and recent jobs table."""
        if event.data_table.id == "jobs-table":
            self._handle_sidebar_job_selected(event)
            return
        if event.data_table.id != "recent-table":
            return
        recent_bar = self.query_one(RecentJobsBar)
        job = recent_bar.get_job_at_cursor()
        if not job:
            return

        def on_action(action: str) -> None:
            if action == "act-live-stdout":
                self._focus_job(job.job_id, "stdout")
            elif action == "act-live-stderr":
                self._focus_job(job.job_id, "stderr")
            elif action in ("act-stdout", "act-stderr", "act-both"):
                panel = self.query_one(ActionPanel)
                self.query_one("#main-menu").display = False
                panel.display = True
                self.current_view = "job-inspect"
                if action == "act-stdout":
                    panel.show_submenu(f"Job {job.job_id} — stdout", [("job-refresh", "Refresh")])
                    panel.show_log(tail_log(job.job_id, lines=100))
                elif action == "act-stderr":
                    panel.show_submenu(f"Job {job.job_id} — stderr", [("job-refresh", "Refresh")])
                    panel.show_log(tail_err_log(job.job_id, lines=100))
                elif action == "act-both":
                    stdout = tail_log(job.job_id, lines=60)
                    stderr = tail_err_log(job.job_id, lines=40)
                    combined = f"{'═'*40} STDOUT {'═'*40}\n{stdout}\n{'═'*40} STDERR {'═'*40}\n{stderr}"
                    panel.show_submenu(f"Job {job.job_id} — stdout + stderr", [("job-refresh", "Refresh")])
                    panel.show_log(combined)

        self.push_screen(JobActionsModal(job.job_id, job.name, job.state), on_action)

    # ── Submit job + offer focus ─────────────────────────────────────────────

    def _submit_and_offer_focus(self, ok: bool, msg: str, job_name: str) -> None:
        """After submitting a SLURM job, show a popup asking to focus it."""
        self.action_refresh_jobs()
        if not ok:
            self.notify(f"Failed: {msg}")
            return
        job_id = msg  # sbatch returns job_id on success

        def on_dismiss(focus: bool) -> None:
            if focus:
                self._focus_job(job_id)

        self.push_screen(JobFocusModal(job_id, job_name), on_dismiss)

    def _focus_job(self, job_id: str, log_type: str = "stdout") -> None:
        """Switch to live-tailing log view for a specific job."""
        panel = self.query_one(ActionPanel)
        self.query_one("#main-menu").display = False
        panel.display = True
        self.current_view = "job-live"
        panel.show_submenu(f"Job {job_id} — Live ({log_type})", [
            ("live-stop", "Stop tailing"),
            ("live-switch-stdout", "Switch to stdout"),
            ("live-switch-stderr", "Switch to stderr"),
        ])
        self._focused_job_id = job_id
        self._focused_log_type = log_type
        self._tail_last_size = 0
        self._refresh_live_log()
        # Start live tail timer (every 2s)
        self._stop_live_tail()
        self._live_tail_timer = self.set_interval(2, self._refresh_live_log)

    def _refresh_live_log(self) -> None:
        """Fetch latest log content and append new lines to the panel."""
        job_id = getattr(self, "_focused_job_id", None)
        if not job_id:
            return
        log_type = getattr(self, "_focused_log_type", "stdout")
        if log_type == "stderr":
            text = tail_err_log(job_id, lines=200)
        else:
            text = tail_log(job_id, lines=200)
        # Only update if content changed
        new_size = len(text)
        if new_size != self._tail_last_size:
            self._tail_last_size = new_size
            panel = self.query_one(ActionPanel)
            log_widget = panel.query_one("#panel-log", Log)
            log_widget.display = True
            log_widget.clear()
            log_widget.write(text)

    def _stop_live_tail(self) -> None:
        timer = getattr(self, "_live_tail_timer", None)
        if timer:
            timer.stop()
            self._live_tail_timer = None

    # ── Route selections from both menus ─────────────────────────────────────

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""

        # Main menu selection
        if item_id.startswith("menu-"):
            action = item_id.replace("menu-", "")
            self._open_action(action)
            return

        # Strip the unique suffix added by show_submenu (e.g. "train-0--3" -> "train-0")
        if "--" in item_id:
            item_id = item_id.rsplit("--", 1)[0]

        # Submenu selection — dispatch based on current_view
        self._handle_submenu(item_id)

    def _open_action(self, action: str) -> None:
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
        elif action == "test":
            self._show_test(panel)
        elif action == "upload":
            self._show_upload(panel)
        elif action == "jobs":
            self._show_jobs(panel)
        elif action == "cleanup":
            self._show_cleanup(panel)
        elif action == "maintenance":
            self._show_maintenance(panel)

    # ── Submenu builders ─────────────────────────────────────────────────────

    def _show_train(self, panel: ActionPanel) -> None:
        items = [(f"train-{i}", name) for i, name in enumerate(actions.TRAIN_CONFIGS)]
        panel.show_submenu("Submit Training Job", items)

    def _show_merge(self, panel: ActionPanel) -> None:
        experiments = actions.list_experiments()
        finals = [e for e in experiments if (Path(e["path"]) / "final").exists()]
        if not finals:
            panel.show_message("Merge LoRA", "No experiments with a /final checkpoint found.")
            return
        items = [(f"merge-{i}", f"{e['name']} ({e['size']})") for i, e in enumerate(finals)]
        panel.show_submenu("Merge LoRA Adapter", items)
        self._merge_candidates = finals

    def _show_export(self, panel: ActionPanel) -> None:
        items = [
            ("export-e2b", "Edge 2B (Gemma 4 E2B)"),
            ("export-e4b", "Edge 4B (Gemma 4 E4B)"),
            ("export-mobile", "Mobile (Gemma 3 1B)"),
        ]
        panel.show_submenu("Export to LiteRT-LM", items)

    def _show_evaluate(self, panel: ActionPanel) -> None:
        experiments = actions.list_experiments()
        finals = [e for e in experiments if (Path(e["path"]) / "final").exists()]
        if not finals:
            panel.show_message("Evaluate", "No experiments with /final checkpoint found.")
            return
        items = [(f"eval-{i}", f"{e['name']} ({e['size']})") for i, e in enumerate(finals)]
        panel.show_submenu("Run Evaluation", items)
        self._eval_candidates = finals

    def _show_test(self, panel: ActionPanel) -> None:
        # Find .litertlm files in models/
        models_dir = actions.PROJECT_ROOT / "models"
        litertlm_files = sorted(models_dir.rglob("*.litertlm")) if models_dir.exists() else []
        if not litertlm_files:
            items = [("test-default", "Run test (default model path)")]
        else:
            items = [(f"test-{i}", str(f.relative_to(actions.PROJECT_ROOT))) for i, f in enumerate(litertlm_files)]
        panel.show_submenu("Test LiteRT-LM Model", items)
        self._test_models = litertlm_files

    def _show_upload(self, panel: ActionPanel) -> None:
        items = [
            ("upload-edge-2b", "edge-2b"),
            ("upload-edge-4b", "edge-4b"),
            ("upload-mobile", "mobile"),
            ("upload-full", "full"),
            ("upload-gguf", "gguf"),
            ("upload-all", "all"),
        ]
        panel.show_submenu("Upload Models (HF + R2)", items)

    def _show_jobs(self, panel: ActionPanel) -> None:
        import os
        jobs = squeue(user=os.environ.get("USER"))
        if not jobs:
            panel.show_message("SLURM Jobs", "No running jobs.")
            return
        items = [
            (f"job-{i}", f"{j.job_id} | {j.name} | {j.state} | {j.time} | {j.partition}")
            for i, j in enumerate(jobs)
        ]
        items.append(("job-cancel-header", "── Cancel a job ──"))
        for i, j in enumerate(jobs):
            items.append((f"cancel-{i}", f"Cancel: {j.job_id} {j.name}"))
        panel.show_submenu("SLURM Jobs", items)
        self._jobs_list = jobs

    def _show_cleanup(self, panel: ActionPanel) -> None:
        experiments = actions.list_experiments()
        items = [(f"delete-{i}", f"{e['name']} — {e['size']}") for i, e in enumerate(experiments)]
        if not items:
            panel.show_message("Cleanup", "No experiments to clean up.")
            return
        panel.show_submenu("Cleanup — select to delete", items)
        self._cleanup_experiments = experiments

    def _show_maintenance(self, panel: ActionPanel) -> None:
        items = [
            ("maint-deps", "Update dependencies (uv pip install -r requirements.txt)"),
            ("maint-cache", "Clear tokenized cache"),
        ]
        panel.show_submenu("Maintenance", items)

    # ── Submenu action dispatch ──────────────────────────────────────────────

    def _handle_submenu(self, item_id: str) -> None:
        panel = self.query_one(ActionPanel)

        # Live tail controls
        if item_id == "job-refresh":
            self._refresh_live_log()
            return
        if item_id == "live-stop":
            self._stop_live_tail()
            self.notify("Stopped tailing")
            return
        if item_id == "live-switch-stdout":
            self._focused_log_type = "stdout"
            self._tail_last_size = 0
            self._refresh_live_log()
            return
        if item_id == "live-switch-stderr":
            self._focused_log_type = "stderr"
            self._tail_last_size = 0
            self._refresh_live_log()
            return

        # Train
        if item_id.startswith("train-"):
            idx = int(item_id.split("-", 1)[1])
            configs = list(actions.TRAIN_CONFIGS.keys())
            if 0 <= idx < len(configs):
                ok, msg = actions.submit_training(configs[idx])
                self._submit_and_offer_focus(ok, msg, configs[idx])

        # Merge
        elif item_id.startswith("merge-"):
            idx = int(item_id.split("-", 1)[1])
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
                self._submit_and_offer_focus(ok, msg, f"merge {exp['name']}")

        # Export
        elif item_id == "export-e2b":
            ok, msg = actions.submit_export_edge("e2b")
            self._submit_and_offer_focus(ok, msg, "export edge-2b")
        elif item_id == "export-e4b":
            ok, msg = actions.submit_export_edge("e4b")
            self._submit_and_offer_focus(ok, msg, "export edge-4b")
        elif item_id == "export-mobile":
            panel.show_message("Export Mobile", "Use: sbatch --export=CHECKPOINT=... slurm/export_litert.sbatch")

        # Evaluate
        elif item_id.startswith("eval-"):
            idx = int(item_id.split("-", 1)[1])
            candidates = getattr(self, "_eval_candidates", [])
            if 0 <= idx < len(candidates):
                checkpoint = f"experiments/{candidates[idx]['name']}/final"
                ok, msg = actions.submit_evaluate(checkpoint)
                self._submit_and_offer_focus(ok, msg, f"eval {candidates[idx]['name']}")

        # Test LiteRT-LM
        elif item_id == "test-default":
            ok, msg = actions.submit_test_litert()
            self._submit_and_offer_focus(ok, msg, "test litert")
        elif item_id.startswith("test-"):
            idx = int(item_id.split("-", 1)[1])
            test_models = getattr(self, "_test_models", [])
            if 0 <= idx < len(test_models):
                model_path = str(test_models[idx].relative_to(actions.PROJECT_ROOT))
                ok, msg = actions.submit_test_litert(model_path)
                self._submit_and_offer_focus(ok, msg, f"test {test_models[idx].name}")

        # Upload
        elif item_id.startswith("upload-"):
            what = item_id.replace("upload-", "")
            ok, msg = actions.run_upload(what, "1.0.0")
            self.notify("Upload complete" if ok else f"Upload failed: {msg[:80]}")
            panel.show_log(msg)

        # Jobs — view log
        elif item_id.startswith("job-") and not item_id.startswith("job-cancel"):
            idx = int(item_id.split("-", 1)[1])
            jobs_list = getattr(self, "_jobs_list", [])
            if 0 <= idx < len(jobs_list):
                log_text = tail_log(jobs_list[idx].job_id, lines=80)
                panel.show_log(log_text)

        # Jobs — cancel
        elif item_id.startswith("cancel-"):
            idx = int(item_id.split("-", 1)[1])
            jobs_list = getattr(self, "_jobs_list", [])
            if 0 <= idx < len(jobs_list):
                ok, msg = scancel(jobs_list[idx].job_id)
                self.notify(msg)
                self.action_refresh_jobs()
                self._show_jobs(panel)

        # Cleanup
        elif item_id.startswith("delete-"):
            idx = int(item_id.split("-", 1)[1])
            experiments = getattr(self, "_cleanup_experiments", [])
            if 0 <= idx < len(experiments):
                ok, msg = actions.delete_experiment(experiments[idx]["path"])
                self.notify(msg)
                self._show_cleanup(panel)

        # Maintenance
        elif item_id == "maint-deps":
            self.notify("Updating dependencies...")
            ok, msg = actions.update_deps()
            panel.show_log(msg)
        elif item_id == "maint-cache":
            ok, msg = actions.clear_tokenized_cache()
            self.notify(msg)


def main():
    app = HorizonApp()
    app.run()


if __name__ == "__main__":
    main()
