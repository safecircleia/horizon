"""Horizon TUI — manage training, jobs, evaluation, and deployment.

All job submissions go to the remote SLURM cluster via SSH.
The workflow enforces:  commit → push → cluster git pull → sbatch
"""

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
    Input,
    Label,
    ListItem,
    ListView,
    Log,
    Static,
)

from . import actions
from .config import get_cluster_config
from .remote_actions import (
    remote_submit_benchmark,
    remote_submit_evaluate,
    remote_submit_export_edge,
    remote_submit_export_edge_web,
    remote_submit_merge,
    remote_submit_profile_mobile,
    remote_submit_test_litert,
    remote_submit_training,
)
from .remote_slurm import (
    ssh_sacct_recent,
    ssh_scancel,
    ssh_sinfo,
    ssh_squeue,
    ssh_tail_err_log,
    ssh_tail_log,
)
from .slurm import (
    format_node_gpu,
    format_node_memory,
)
from .ssh import git_commit_and_push, git_local_status

# ── Commit guard modal ───────────────────────────────────────────────────────


class CommitGuardModal(ModalScreen[tuple[bool, str]]):
    """Shown when there are uncommitted changes before a job submission.

    Returns (proceed, commit_message).  If proceed is False the user cancelled.
    """

    DEFAULT_CSS = """
    CommitGuardModal {
        align: center middle;
    }
    #cg-box {
        width: 70;
        height: auto;
        border: thick $warning;
        background: $surface;
        padding: 1 2;
    }
    #cg-dirty {
        color: $warning;
        margin-bottom: 1;
    }
    #cg-input {
        margin-bottom: 1;
    }
    #cg-buttons {
        layout: horizontal;
        height: auto;
    }
    #cg-buttons Button {
        margin-right: 2;
    }
    """

    def __init__(self, dirty_files: list[str]) -> None:
        super().__init__()
        self.dirty_files = dirty_files

    def compose(self) -> ComposeResult:
        preview = "\n".join(f"  {ln}" for ln in self.dirty_files[:8])
        if len(self.dirty_files) > 8:
            preview += f"\n  … and {len(self.dirty_files) - 8} more"
        with Vertical(id="cg-box"):
            yield Static("[b]Uncommitted changes detected[/b]")
            yield Static(
                f"The following files must be committed before submitting:\n{preview}",
                id="cg-dirty",
            )
            yield Static("Commit message:")
            yield Input(
                placeholder="chore: pre-job commit",
                id="cg-input",
            )
            with Horizontal(id="cg-buttons"):
                yield Button("Commit & Submit", id="btn-commit", variant="primary")
                yield Button("Cancel", id="btn-cancel", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.dismiss((False, ""))
        else:
            msg = self.query_one("#cg-input", Input).value.strip()
            if not msg:
                msg = "chore: pre-job commit"
            self.dismiss((True, msg))


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
            yield Static(
                f"[b]Job submitted on cluster:[/b] {self.job_id} ({self.job_name})"
            )
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
        is_running = self.job_state in ("RUNNING", "PENDING", "COMPLETING")
        with Vertical(id="job-modal-box"):
            yield Static(
                f"[b]Job {self.job_id}[/b] — {self.job_name} [{self.job_state}]"
            )
            items = [
                ListItem(Label("  Live tail stdout"), id="act-live-stdout"),
                ListItem(Label("  Live tail stderr"), id="act-live-stderr"),
                ListItem(Label("  View stdout (snapshot)"), id="act-stdout"),
                ListItem(Label("  View stderr (snapshot)"), id="act-stderr"),
                ListItem(Label("  View both (stdout + stderr)"), id="act-both"),
            ]
            if is_running:
                items += [
                    ListItem(Label("  ─────────────────────────"), id="act-sep"),
                    ListItem(Label("  [red]Cancel job[/red]"), id="act-cancel"),
                ]
            items.append(ListItem(Label("  Close"), id="act-close"))
            yield ListView(*items)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(event.item.id or "act-close")


# ── Sidebar: remote running jobs ─────────────────────────────────────────────


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
        yield Label("Cluster Jobs (Enter to inspect)", classes="title")
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
        self.jobs = ssh_squeue()
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
        nodes = ssh_sinfo()
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
        yield Label("Recent Cluster Jobs (Enter to inspect)", classes="title")
        yield DataTable(id="recent-table")

    def on_mount(self) -> None:
        table = self.query_one("#recent-table", DataTable)
        table.add_columns("ID", "Name", "State", "Exit", "Elapsed", "Ended")
        table.cursor_type = "row"
        self.refresh_recent()

    def refresh_recent(self) -> None:
        table = self.query_one("#recent-table", DataTable)
        table.clear()
        raw = ssh_sacct_recent(8)
        self.jobs = raw
        for job in raw:
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
    ("benchmark", "Run Accuracy Benchmark"),
    ("test", "Test LiteRT-LM Model"),
    ("upload", "Upload Models (HF + R2)"),
    ("jobs", "View Cluster Jobs"),
    ("cluster", "Cluster (SSH)"),
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
        border-bottom: solid $accent;
        margin-bottom: 1;
    }
    #panel-log {
        height: 1fr;
        border: solid $surface-lighten-2;
        padding: 0 1;
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
        Binding("r", "refresh_jobs", "Refresh"),
        Binding("escape", "back", "Back"),
    ]

    current_view = reactive("menu")

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top-area"):
            with Vertical(id="main-area"):
                yield ListView(
                    *[
                        ListItem(Label(f"  {label}"), id=f"menu-{key}")
                        for key, label in MENU_ITEMS
                    ],
                    id="main-menu",
                )
                yield ActionPanel(id="action-panel")
            yield JobsSidebar(id="jobs-sidebar")
        yield RecentJobsBar(id="recent-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#action-panel").display = False
        self.set_interval(10, self._do_refresh)

    def _do_refresh(self) -> None:
        sidebar = self.query_one("#jobs-sidebar", JobsSidebar)
        sidebar.refresh_jobs()
        sidebar.refresh_nodes()
        self.query_one("#recent-bar", RecentJobsBar).refresh_recent()

    def action_refresh_jobs(self) -> None:
        self._do_refresh()
        self.notify("Refreshed cluster status")

    def action_back(self) -> None:
        self._stop_live_tail()
        self.query_one("#main-menu").display = True
        self.query_one("#action-panel").display = False
        self.query_one("#main-menu", ListView).focus()
        self.current_view = "menu"

    # ── Git commit guard ──────────────────────────────────────────────────────

    def _check_and_submit(
        self,
        submit_fn,
        job_name: str,
    ) -> None:
        """Check git status; if dirty show CommitGuardModal, then call submit_fn.

        submit_fn is a zero-arg callable that returns (ok, msg).
        """
        status = git_local_status()
        if not status["clean"]:

            def on_guard(result: tuple[bool, str]) -> None:
                proceed, commit_msg = result
                if not proceed:
                    self.notify("Submission cancelled.")
                    return
                # Commit locally first
                self.notify("Committing changes…")
                ok, out = git_commit_and_push(commit_msg)
                if not ok:
                    self.notify(f"Commit failed: {out[:80]}")
                    panel = self.query_one(ActionPanel)
                    panel.show_log(f"git commit/push failed:\n{out}")
                    return
                self.notify("Committed & pushed. Submitting job…")
                ok, msg = submit_fn()
                self._submit_and_offer_focus(ok, msg, job_name)

            self.push_screen(
                CommitGuardModal(status["status_lines"]),
                on_guard,
            )
        elif status["ahead"] > 0:
            # Committed but not pushed
            self.notify(f"Pushing {status['ahead']} unpushed commit(s)…")
            from .ssh import git_push_existing

            ok, out = git_push_existing()
            if not ok:
                self.notify(f"Push failed: {out[:80]}")
                return
            ok, msg = submit_fn()
            self._submit_and_offer_focus(ok, msg, job_name)
        else:
            # Clean and up-to-date
            ok, msg = submit_fn()
            self._submit_and_offer_focus(ok, msg, job_name)

    # ── Jobs table interaction (sidebar + bottom bar) ──────────────────────

    def _handle_sidebar_job_selected(self, event: DataTable.RowSelected) -> None:
        sidebar = self.query_one(JobsSidebar)
        job = sidebar.get_job_at_cursor()
        if not job:
            return

        def on_action(action: str) -> None:
            if action == "act-live-stdout":
                self._focus_job(job.job_id, "stdout")
            elif action == "act-live-stderr":
                self._focus_job(job.job_id, "stderr")
            elif action == "act-cancel":
                ok, msg = ssh_scancel(job.job_id)
                self.notify(msg)
                self._do_refresh()
            elif action == "act-stdout":
                panel = self.query_one(ActionPanel)
                self.query_one("#main-menu").display = False
                panel.display = True
                self.current_view = "job-inspect"
                panel.show_submenu(
                    f"Job {job.job_id} — stdout", [("job-refresh", "Refresh")]
                )
                panel.show_log(ssh_tail_log(job.job_id, lines=100))
            elif action == "act-stderr":
                panel = self.query_one(ActionPanel)
                self.query_one("#main-menu").display = False
                panel.display = True
                self.current_view = "job-inspect"
                panel.show_submenu(
                    f"Job {job.job_id} — stderr", [("job-refresh", "Refresh")]
                )
                panel.show_log(ssh_tail_err_log(job.job_id, lines=100))
            elif action == "act-both":
                panel = self.query_one(ActionPanel)
                self.query_one("#main-menu").display = False
                panel.display = True
                self.current_view = "job-inspect"
                stdout = ssh_tail_log(job.job_id, lines=60)
                stderr = ssh_tail_err_log(job.job_id, lines=40)
                combined = f"{'═' * 40} STDOUT {'═' * 40}\n{stdout}\n{'═' * 40} STDERR {'═' * 40}\n{stderr}"
                panel.show_submenu(
                    f"Job {job.job_id} — stdout + stderr", [("job-refresh", "Refresh")]
                )
                panel.show_log(combined)

        self.push_screen(
            JobActionsModal(job.job_id, job.name, job.state),
            on_action,
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "jobs-table":
            self._handle_sidebar_job_selected(event)
            return
        if event.data_table.id != "recent-table":
            return
        recent_bar = self.query_one(RecentJobsBar)
        job = recent_bar.get_job_at_cursor()
        if not job:
            return

        # For recently completed jobs we have a RecentJob (no .state sentinel)
        state = job.state if hasattr(job, "state") else "COMPLETED"

        def on_action(action: str) -> None:
            if action == "act-live-stdout":
                self._focus_job(job.job_id, "stdout")
            elif action == "act-live-stderr":
                self._focus_job(job.job_id, "stderr")
            elif action == "act-cancel":
                ok, msg = ssh_scancel(job.job_id)
                self.notify(msg)
                self._do_refresh()
            elif action in ("act-stdout", "act-stderr", "act-both"):
                panel = self.query_one(ActionPanel)
                self.query_one("#main-menu").display = False
                panel.display = True
                self.current_view = "job-inspect"
                if action == "act-stdout":
                    panel.show_submenu(
                        f"Job {job.job_id} — stdout", [("job-refresh", "Refresh")]
                    )
                    panel.show_log(ssh_tail_log(job.job_id, lines=100))
                elif action == "act-stderr":
                    panel.show_submenu(
                        f"Job {job.job_id} — stderr", [("job-refresh", "Refresh")]
                    )
                    panel.show_log(ssh_tail_err_log(job.job_id, lines=100))
                elif action == "act-both":
                    stdout = ssh_tail_log(job.job_id, lines=60)
                    stderr = ssh_tail_err_log(job.job_id, lines=40)
                    combined = f"{'═' * 40} STDOUT {'═' * 40}\n{stdout}\n{'═' * 40} STDERR {'═' * 40}\n{stderr}"
                    panel.show_submenu(
                        f"Job {job.job_id} — stdout + stderr",
                        [("job-refresh", "Refresh")],
                    )
                    panel.show_log(combined)

        self.push_screen(JobActionsModal(job.job_id, job.name, state), on_action)

    # ── Submit job + offer focus ─────────────────────────────────────────────

    def _submit_and_offer_focus(self, ok: bool, msg: str, job_name: str) -> None:
        self.action_refresh_jobs()
        if not ok:
            self.notify(f"Failed: {msg}")
            panel = self.query_one(ActionPanel)
            panel.show_log(f"Submission failed:\n{msg}")
            return
        job_id = msg

        def on_dismiss(focus: bool) -> None:
            if focus:
                self._focus_job(job_id)

        self.push_screen(JobFocusModal(job_id, job_name), on_dismiss)

    def _focus_job(self, job_id: str, log_type: str = "stdout") -> None:
        panel = self.query_one(ActionPanel)
        self.query_one("#main-menu").display = False
        panel.display = True
        self.current_view = "job-live"
        panel.show_submenu(
            f"Job {job_id} — Live ({log_type})",
            [
                ("live-stop", "Stop tailing"),
                ("live-switch-stdout", "Switch to stdout"),
                ("live-switch-stderr", "Switch to stderr"),
            ],
        )
        self._focused_job_id = job_id
        self._focused_log_type = log_type
        self._tail_last_size = 0
        self._refresh_live_log()
        self._stop_live_tail()
        self._live_tail_timer = self.set_interval(3, self._refresh_live_log)

    def _refresh_live_log(self) -> None:
        job_id = getattr(self, "_focused_job_id", None)
        if not job_id:
            return
        log_type = getattr(self, "_focused_log_type", "stdout")
        if log_type == "stderr":
            text = ssh_tail_err_log(job_id, lines=200)
        else:
            text = ssh_tail_log(job_id, lines=200)
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

        if item_id.startswith("menu-"):
            action = item_id.replace("menu-", "")
            self._open_action(action)
            return

        if "--" in item_id:
            item_id = item_id.rsplit("--", 1)[0]

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
        elif action == "benchmark":
            self._show_benchmark(panel)
        elif action == "test":
            self._show_test(panel)
        elif action == "upload":
            self._show_upload(panel)
        elif action == "jobs":
            self._show_jobs(panel)
        elif action == "cluster":
            self._show_cluster(panel)
        elif action == "cleanup":
            self._show_cleanup(panel)
        elif action == "maintenance":
            self._show_maintenance(panel)

    # ── Submenu builders ─────────────────────────────────────────────────────

    def _show_train(self, panel: ActionPanel) -> None:
        items = [(f"train-{i}", name) for i, name in enumerate(actions.TRAIN_CONFIGS)]
        panel.show_submenu("Submit Training Job (remote)", items)

    def _show_merge(self, panel: ActionPanel) -> None:
        experiments = actions.list_experiments()
        finals = [e for e in experiments if (Path(e["path"]) / "final").exists()]
        if not finals:
            panel.show_message(
                "Merge LoRA", "No experiments with a /final checkpoint found."
            )
            return
        items = [
            (f"merge-{i}", f"{e['name']} ({e['size']})") for i, e in enumerate(finals)
        ]
        panel.show_submenu("Merge LoRA Adapter (remote)", items)
        self._merge_candidates = finals

    def _show_export(self, panel: ActionPanel) -> None:
        items = [
            ("export-e2b", "Edge 2B — all variants"),
            ("export-e4b", "Edge 4B — all variants"),
            ("export-e2b-web", "Edge 2B — Web (WebGPU) only"),
            ("export-e4b-web", "Edge 4B — Web (WebGPU) only"),
            ("export-mobile", "Mobile (Gemma 3 1B)"),
        ]
        panel.show_submenu("Export to LiteRT-LM (remote)", items)

    def _show_evaluate(self, panel: ActionPanel) -> None:
        experiments = actions.list_experiments()
        finals = [e for e in experiments if (Path(e["path"]) / "final").exists()]
        if not finals:
            panel.show_message(
                "Evaluate", "No experiments with /final checkpoint found."
            )
            return
        items = [
            (f"eval-{i}", f"{e['name']} ({e['size']})") for i, e in enumerate(finals)
        ]
        panel.show_submenu("Run Evaluation (remote)", items)
        self._eval_candidates = finals

    def _show_test(self, panel: ActionPanel) -> None:
        models_dir = actions.PROJECT_ROOT / "models"
        litertlm_files = (
            sorted(models_dir.rglob("*.litertlm")) if models_dir.exists() else []
        )
        if not litertlm_files:
            items = [("test-default", "Run test (default model path)")]
        else:
            items = [
                (f"test-{i}", str(f.relative_to(actions.PROJECT_ROOT)))
                for i, f in enumerate(litertlm_files)
            ]
        panel.show_submenu("Test LiteRT-LM Model (remote)", items)
        self._test_models = litertlm_files

    def _show_upload(self, panel: ActionPanel) -> None:
        items = [
            ("upload-edge-2b", "edge-2b (all variants)"),
            ("upload-edge-4b", "edge-4b (all variants)"),
            ("upload-edge-2b-web", "edge-2b-web (WebGPU only)"),
            ("upload-edge-4b-web", "edge-4b-web (WebGPU only)"),
            ("upload-mobile", "mobile"),
            ("upload-full", "full"),
            ("upload-gguf", "gguf"),
            ("upload-all", "all"),
        ]
        panel.show_submenu("Upload Models (HF + R2)", items)

    def _show_jobs(self, panel: ActionPanel) -> None:
        jobs = ssh_squeue()
        if not jobs:
            panel.show_message("Cluster Jobs", "No running jobs on cluster.")
            return
        items = [
            (
                f"job-{i}",
                f"{j.job_id} | {j.name} | {j.state} | {j.time} | {j.partition}",
            )
            for i, j in enumerate(jobs)
        ]
        items.append(("job-cancel-header", "── Cancel a job ──"))
        for i, j in enumerate(jobs):
            items.append((f"cancel-{i}", f"Cancel: {j.job_id} {j.name}"))
        panel.show_submenu("Cluster Jobs", items)
        self._jobs_list = jobs

    def _show_cluster(self, panel: ActionPanel) -> None:
        cfg = get_cluster_config()
        status = git_local_status()
        dirty_marker = " ⚠ uncommitted" if not status["clean"] else ""
        ahead_marker = f" ↑{status['ahead']} unpushed" if status["ahead"] > 0 else ""
        items = [
            ("cluster-git-status", f"Git status{dirty_marker}{ahead_marker}"),
            ("cluster-git-commit", "Commit & push changes"),
            ("cluster-status", f"Check connection ({cfg.remote})"),
            ("cluster-sync", "Sync repo to cluster (git pull)"),
            ("cluster-setup-venv", "Setup venv on cluster (uv)"),
            ("cluster-jobs", "View remote SLURM jobs"),
            ("cluster-recent", "View recently completed jobs"),
            ("cluster-nodes", "View cluster nodes"),
            ("cluster-train", "Submit remote training job"),
        ]
        panel.show_submenu(f"Cluster SSH — {cfg.remote}", items)

    def _show_cleanup(self, panel: ActionPanel) -> None:
        experiments = actions.list_experiments()
        items = [
            (f"delete-{i}", f"{e['name']} — {e['size']}")
            for i, e in enumerate(experiments)
        ]
        if not items:
            panel.show_message("Cleanup", "No experiments to clean up.")
            return
        panel.show_submenu("Cleanup — select to delete", items)
        self._cleanup_experiments = experiments

    def _show_benchmark(self, panel: ActionPanel) -> None:
        experiments = actions.list_experiments()
        finals = [e for e in experiments if (Path(e["path"]) / "final").exists()]
        items = []
        if finals:
            for i, e in enumerate(finals):
                items.append((f"bench-{i}", f"{e['name']} ({e['size']})"))
                items.append(
                    (f"bench-baseline-{i}", f"{e['name']} — run & save as baseline")
                )
        else:
            items.append(("bench-default", "Run benchmark (default checkpoint)"))
        items.append(("bench-create-split", "Create / refresh benchmark split"))
        items.append(("bench-view-report", "View last benchmark report"))
        items.append(("bench-profile", "Profile mobile model (RAM / battery)"))
        panel.show_submenu("Run Accuracy Benchmark (remote)", items)
        self._bench_candidates = finals

    def _show_maintenance(self, panel: ActionPanel) -> None:
        items = [
            ("maint-deps", "Update dependencies (uv pip install -r requirements.txt)"),
            ("maint-cache", "Clear tokenized cache"),
        ]
        panel.show_submenu("Maintenance", items)

    # ── Submenu action dispatch ──────────────────────────────────────────────

    def _handle_submenu(self, item_id: str) -> None:  # noqa: C901
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

        # ── Training (remote) ──────────────────────────────────────────────
        if item_id.startswith("train-"):
            idx = int(item_id.split("-", 1)[1])
            configs = list(actions.TRAIN_CONFIGS.keys())
            if 0 <= idx < len(configs):
                config_name = configs[idx]
                self._check_and_submit(
                    lambda cn=config_name: remote_submit_training(cn),
                    config_name,
                )

        # ── Merge (remote) ─────────────────────────────────────────────────
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
                self._check_and_submit(
                    lambda c=checkpoint, o=output: remote_submit_merge(c, o),
                    f"merge {exp['name']}",
                )

        # ── Export (remote) ────────────────────────────────────────────────
        elif item_id == "export-e2b":
            self._check_and_submit(
                lambda: remote_submit_export_edge("e2b"), "export edge-2b"
            )
        elif item_id == "export-e4b":
            self._check_and_submit(
                lambda: remote_submit_export_edge("e4b"), "export edge-4b"
            )
        elif item_id == "export-e2b-web":
            self._check_and_submit(
                lambda: remote_submit_export_edge_web("e2b"), "export edge-2b web"
            )
        elif item_id == "export-e4b-web":
            self._check_and_submit(
                lambda: remote_submit_export_edge_web("e4b"), "export edge-4b web"
            )
        elif item_id == "export-mobile":
            panel.show_message(
                "Export Mobile",
                "Choose a checkpoint via Merge LoRA first, then this runs on cluster.",
            )

        # ── Evaluate (remote) ──────────────────────────────────────────────
        elif item_id.startswith("eval-"):
            idx = int(item_id.split("-", 1)[1])
            candidates = getattr(self, "_eval_candidates", [])
            if 0 <= idx < len(candidates):
                checkpoint = f"experiments/{candidates[idx]['name']}/final"
                self._check_and_submit(
                    lambda c=checkpoint: remote_submit_evaluate(c),
                    f"eval {candidates[idx]['name']}",
                )

        # ── Benchmark (remote) ─────────────────────────────────────────────
        elif item_id.startswith("bench-baseline-"):
            idx = int(item_id.split("-", 2)[2])
            candidates = getattr(self, "_bench_candidates", [])
            if 0 <= idx < len(candidates):
                checkpoint = f"experiments/{candidates[idx]['name']}/final"
                self._check_and_submit(
                    lambda c=checkpoint: remote_submit_benchmark(c, save_baseline=True),
                    f"benchmark {candidates[idx]['name']}",
                )
        elif (
            item_id.startswith("bench-")
            and not item_id.startswith("bench-create")
            and not item_id.startswith("bench-view")
            and not item_id.startswith("bench-default")
            and not item_id.startswith("bench-profile")
        ):
            idx = int(item_id.split("-", 1)[1])
            candidates = getattr(self, "_bench_candidates", [])
            if 0 <= idx < len(candidates):
                checkpoint = f"experiments/{candidates[idx]['name']}/final"
                self._check_and_submit(
                    lambda c=checkpoint: remote_submit_benchmark(c),
                    f"benchmark {candidates[idx]['name']}",
                )
        elif item_id == "bench-default":
            self._check_and_submit(
                lambda: remote_submit_benchmark("models/horizon-edge-2b"), "benchmark"
            )
        elif item_id == "bench-create-split":
            self.notify("Creating benchmark split…")
            ok, msg = actions.create_benchmark_split()
            self.notify("Benchmark split created" if ok else f"Failed: {msg[:60]}")
            panel.show_log(msg)
        elif item_id == "bench-profile":
            models_dir = actions.PROJECT_ROOT / "models"
            litertlm_files = (
                sorted(models_dir.rglob("*.litertlm")) if models_dir.exists() else []
            )
            if not litertlm_files:
                panel.show_message(
                    "Profile Mobile",
                    "No .litertlm models found in models/. Export the mobile model first.",
                )
            else:
                model_path = str(litertlm_files[0].relative_to(actions.PROJECT_ROOT))
                self._check_and_submit(
                    lambda mp=model_path: remote_submit_profile_mobile(mp),
                    f"profile {litertlm_files[0].name}",
                )
        elif item_id == "bench-view-report":
            report = actions.load_latest_benchmark_report()
            if report is None:
                panel.show_message(
                    "Benchmark Report", "No report found. Run a benchmark first."
                )
            else:
                s = report.get("summary", {})
                lat = report.get("latency", {})
                lines = [
                    f"Checkpoint: {report.get('checkpoint', 'unknown')}",
                    "",
                    f"  Recall:     {s.get('recall', 'N/A')}  [target ≥ 0.97]",
                    f"  FPR:        {s.get('fpr', 'N/A')}  [target ≤ 0.03]",
                    f"  Precision:  {s.get('precision', 'N/A')}  [target ≥ 0.95]",
                    f"  Binary F1:  {s.get('f1', 'N/A')}  [target ≥ 0.96]",
                    f"  Macro F1:   {s.get('macro_f1', 'N/A')}",
                    f"  Rule catch: {s.get('rule_catch_rate', 'N/A')}  [target ≥ 0.80]",
                ]
                if lat:
                    lines += [
                        "",
                        f"  Latency  P50={lat.get('p50_ms', '?')}ms  P95={lat.get('p95_ms', '?')}ms  P99={lat.get('p99_ms', '?')}ms",
                    ]
                panel.show_submenu(
                    "Last Benchmark Report", [("bench-view-report", "Refresh")]
                )
                panel.show_log("\n".join(lines))

        # ── Test LiteRT-LM (remote) ────────────────────────────────────────
        elif item_id == "test-default":
            self._check_and_submit(lambda: remote_submit_test_litert(), "test litert")
        elif item_id.startswith("test-"):
            idx = int(item_id.split("-", 1)[1])
            test_models = getattr(self, "_test_models", [])
            if 0 <= idx < len(test_models):
                model_path = str(test_models[idx].relative_to(actions.PROJECT_ROOT))
                self._check_and_submit(
                    lambda mp=model_path: remote_submit_test_litert(mp),
                    f"test {test_models[idx].name}",
                )

        # ── Upload (local) ────────────────────────────────────────────────
        elif item_id.startswith("upload-"):
            what = item_id.replace("upload-", "")
            ok, msg = actions.run_upload(what, "1.0.0")
            self.notify("Upload complete" if ok else f"Upload failed: {msg[:80]}")
            panel.show_log(msg)

        # ── Jobs — view log ────────────────────────────────────────────────
        elif item_id.startswith("job-") and not item_id.startswith("job-cancel"):
            idx = int(item_id.split("-", 1)[1])
            jobs_list = getattr(self, "_jobs_list", [])
            if 0 <= idx < len(jobs_list):
                log_text = ssh_tail_log(jobs_list[idx].job_id, lines=80)
                panel.show_log(log_text)

        # ── Jobs — cancel ──────────────────────────────────────────────────
        elif item_id.startswith("cancel-"):
            idx = int(item_id.split("-", 1)[1])
            jobs_list = getattr(self, "_jobs_list", [])
            if 0 <= idx < len(jobs_list):
                ok, msg = ssh_scancel(jobs_list[idx].job_id)
                self.notify(msg)
                self.action_refresh_jobs()
                self._show_jobs(panel)

        # ── Cleanup ────────────────────────────────────────────────────────
        elif item_id.startswith("delete-"):
            idx = int(item_id.split("-", 1)[1])
            experiments = getattr(self, "_cleanup_experiments", [])
            if 0 <= idx < len(experiments):
                ok, msg = actions.delete_experiment(experiments[idx]["path"])
                self.notify(msg)
                self._show_cleanup(panel)

        # ── Cluster (SSH) menu ─────────────────────────────────────────────
        elif item_id == "cluster-git-status":
            status = git_local_status()
            lines = [
                f"Branch: {status['branch']}",
                f"Clean:  {status['clean']}",
                f"Ahead:  {status['ahead']} unpushed commit(s)",
                "",
            ]
            if status["status_lines"]:
                lines.append("Changed files:")
                lines += [f"  {ln}" for ln in status["status_lines"]]
            else:
                lines.append("Working tree is clean.")
            panel.show_submenu("Git Status", [("cluster-git-status", "Refresh")])
            panel.show_log("\n".join(lines))

        elif item_id == "cluster-git-commit":
            status = git_local_status()
            if status["clean"] and status["ahead"] == 0:
                panel.show_log("Nothing to commit — working tree clean and up-to-date.")
                return

            def on_guard(result: tuple[bool, str]) -> None:
                proceed, commit_msg = result
                if not proceed:
                    return
                ok, out = git_commit_and_push(commit_msg)
                panel.show_log(out)
                self.notify("Committed & pushed" if ok else f"Failed: {out[:60]}")

            self.push_screen(
                CommitGuardModal(status["status_lines"]),
                on_guard,
            )

        elif item_id == "cluster-status":
            self.notify("Checking cluster connection…")
            from .ssh import check_connection

            ok, msg = check_connection()
            panel.show_log(f"{'✓' if ok else '✗'} {msg}")
            self.notify(msg[:80])

        elif item_id == "cluster-sync":
            self.notify("Syncing repo to cluster…")
            from .ssh import sync_repo

            ok, msg = sync_repo()
            panel.show_log(msg)
            self.notify("Repo synced" if ok else f"Sync failed: {msg[:60]}")

        elif item_id == "cluster-setup-venv":
            self.notify("Setting up venv on cluster (may take a minute)…")
            from .ssh import ensure_venv

            ok, msg = ensure_venv()
            panel.show_log(msg)
            self.notify("venv ready" if ok else f"venv setup failed: {msg[:60]}")

        elif item_id == "cluster-jobs":
            jobs = ssh_squeue()
            if not jobs:
                panel.show_message("Remote SLURM Jobs", "No jobs running on cluster.")
            else:
                lines = [
                    f"{'ID':>8}  {'Name':<24} {'State':<12} {'Time':<8} {'Partition':<12} Node",
                    "-" * 76,
                ]
                for j in jobs:
                    lines.append(
                        f"{j.job_id:>8}  {j.name:<24} {j.state:<12} {j.time:<8} {j.partition:<12} {j.node}"
                    )
                panel.show_submenu(
                    "Remote SLURM Jobs",
                    [
                        ("cluster-jobs", "Refresh"),
                        ("cluster-jobs-cancel", "Cancel a job…"),
                    ],
                )
                panel.show_log("\n".join(lines))
                self._cluster_jobs_list = jobs

        elif item_id == "cluster-recent":
            recent = ssh_sacct_recent(15)
            if not recent:
                panel.show_message(
                    "Recent Cluster Jobs", "No recent jobs found (last 7 days)."
                )
            else:
                lines = [
                    f"{'ID':>8}  {'Name':<20} {'State':<14} {'Exit':>4} {'Elapsed':<10} Ended",
                    "-" * 72,
                ]
                for j in recent:
                    lines.append(
                        f"{j.job_id:>8}  {j.name:<20} {j.state:<14} {j.exit_code:>4} "
                        f"{j.elapsed:<10} {j.end_time[-8:] if len(j.end_time) > 8 else j.end_time}"
                    )
                panel.show_submenu(
                    "Recent Cluster Jobs", [("cluster-recent", "Refresh")]
                )
                panel.show_log("\n".join(lines))

        elif item_id == "cluster-nodes":
            nodes = ssh_sinfo()
            if not nodes:
                panel.show_message("Cluster Nodes", "No node info available.")
            else:
                lines = [
                    f"{'Node':<16} {'Partition':<12} {'State':<10} {'GPUs':<16} {'CPUs':>5} {'Mem':>8}",
                    "-" * 72,
                ]
                for n in nodes:
                    lines.append(
                        f"{n.name:<16} {n.partition:<12} {n.state:<10} "
                        f"{format_node_gpu(n.gres):<16} {n.cpus:>5} {format_node_memory(n.memory):>8}"
                    )
                panel.show_submenu("Cluster Nodes", [("cluster-nodes", "Refresh")])
                panel.show_log("\n".join(lines))

        elif item_id == "cluster-train":
            configs = list(actions.TRAIN_CONFIGS.keys())
            items = [(f"cluster-train-{i}", name) for i, name in enumerate(configs)]
            panel.show_submenu("Submit Remote Training Job", items)

        elif item_id.startswith("cluster-train-"):
            idx = int(item_id.split("-", 2)[2])
            configs = list(actions.TRAIN_CONFIGS.keys())
            if 0 <= idx < len(configs):
                config_name = configs[idx]
                self._check_and_submit(
                    lambda cn=config_name: remote_submit_training(cn),
                    config_name,
                )

        elif item_id == "cluster-log":
            jobs = getattr(self, "_cluster_jobs_list", [])
            if not jobs:
                panel.show_message(
                    "Remote Log", "No remote jobs cached. Refresh first."
                )
            else:
                items = [
                    (f"cluster-log-{i}", f"{j.job_id} — {j.name}")
                    for i, j in enumerate(jobs)
                ]
                panel.show_submenu("View Remote Log", items)

        elif item_id.startswith("cluster-log-"):
            idx = int(item_id.split("-", 2)[2])
            jobs = getattr(self, "_cluster_jobs_list", [])
            if 0 <= idx < len(jobs):
                log = ssh_tail_log(jobs[idx].job_id)
                panel.show_log(log)

        # ── Maintenance ────────────────────────────────────────────────────
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
