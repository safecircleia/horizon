"""On-device memory management for 24/7 LiteRT-LM model execution.

Implements the memory management requirements from issue #8:
    - Lazy loading: model loaded on first inference request
    - Auto-unload: model unloaded after configurable inactivity (default 5 min)
    - Fast reload: target < 2 s reload time
    - RSS monitoring: continuous peak-RSS tracking with configurable budget
    - Memory pressure response: unload model when system memory is low

This module is designed to be embedded in the Android/iOS client SDK.
On a development host it uses psutil as a proxy for Android memory APIs.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelState(Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"


@dataclass
class MemoryBudget:
    """Memory budget constraints from issue #8."""

    max_loaded_ram_mb: float = 100.0  # < 100 MB when model loaded
    max_total_ram_mb: float = 150.0  # < 150 MB total process
    low_memory_threshold_mb: float = 500.0  # Trigger unload when system avail < this
    critical_memory_threshold_mb: float = 200.0  # Force-kill inference below this


@dataclass
class MemoryStats:
    """Snapshot of current memory usage."""

    process_rss_mb: float = 0.0
    system_available_mb: float = 0.0
    system_total_mb: float = 0.0
    peak_rss_mb: float = 0.0
    model_state: ModelState = ModelState.UNLOADED
    timestamp: float = field(default_factory=time.time)

    @property
    def system_used_pct(self) -> float:
        if self.system_total_mb == 0:
            return 0.0
        return (
            (self.system_total_mb - self.system_available_mb)
            / self.system_total_mb
            * 100
        )

    def exceeds_budget(self, budget: MemoryBudget) -> bool:
        return self.process_rss_mb > budget.max_total_ram_mb

    def system_memory_low(self, budget: MemoryBudget) -> bool:
        return self.system_available_mb < budget.low_memory_threshold_mb

    def system_memory_critical(self, budget: MemoryBudget) -> bool:
        return self.system_available_mb < budget.critical_memory_threshold_mb


def get_memory_stats() -> MemoryStats:
    """Get current memory stats using psutil (dev host proxy)."""
    try:
        import os

        import psutil

        proc = psutil.Process(os.getpid())
        mem_info = proc.memory_info()
        vm = psutil.virtual_memory()

        return MemoryStats(
            process_rss_mb=mem_info.rss / 1024 / 1024,
            system_available_mb=vm.available / 1024 / 1024,
            system_total_mb=vm.total / 1024 / 1024,
        )
    except ImportError:
        return MemoryStats()


class ModelLifecycleManager:
    """Manages the lifecycle of a LiteRT-LM model for 24/7 on-device execution.

    Features:
        - Lazy loading on first inference
        - Automatic unloading after inactivity timeout
        - Memory pressure monitoring
        - Thread-safe state transitions
        - Reload time tracking
    """

    def __init__(
        self,
        model_path: str,
        idle_timeout_s: float = 300.0,  # 5 minutes
        memory_budget: MemoryBudget | None = None,
        monitor_interval_s: float = 10.0,
        on_unload: Callable[[], None] | None = None,
        on_load: Callable[[], None] | None = None,
    ):
        self.model_path = model_path
        self.idle_timeout_s = idle_timeout_s
        self.budget = memory_budget or MemoryBudget()
        self.monitor_interval_s = monitor_interval_s
        self._on_unload = on_unload
        self._on_load = on_load

        self._state = ModelState.UNLOADED
        self._lock = threading.RLock()
        self._last_inference_time: float = 0.0
        self._last_load_time: float = 0.0
        self._load_duration_s: float = 0.0
        self._peak_rss_mb: float = 0.0
        self._inference_count: int = 0

        # Background monitor
        self._monitor_thread: threading.Thread | None = None
        self._monitor_stop = threading.Event()

        # Subprocess handle (for litert-lm process model)
        self._process: subprocess.Popen | None = None  # type: ignore[type-arg]

    @property
    def state(self) -> ModelState:
        return self._state

    @property
    def peak_rss_mb(self) -> float:
        return self._peak_rss_mb

    @property
    def load_duration_s(self) -> float:
        return self._load_duration_s

    @property
    def inference_count(self) -> int:
        return self._inference_count

    def start(self) -> None:
        """Start the background memory monitor."""
        if self._monitor_thread is not None:
            return
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="mem-monitor"
        )
        self._monitor_thread.start()
        logger.info("Memory monitor started (interval=%.1fs)", self.monitor_interval_s)

    def stop(self) -> None:
        """Stop the monitor and unload the model."""
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        self._unload()

    def ensure_loaded(self) -> bool:
        """Ensure the model is loaded.  Returns True if ready."""
        with self._lock:
            if self._state == ModelState.LOADED:
                self._last_inference_time = time.time()
                return True
            if self._state in (ModelState.LOADING, ModelState.UNLOADING):
                return False
            return self._load()

    def record_inference(self) -> None:
        """Record that an inference just completed."""
        with self._lock:
            self._last_inference_time = time.time()
            self._inference_count += 1

            # Update peak RSS
            stats = get_memory_stats()
            if stats.process_rss_mb > self._peak_rss_mb:
                self._peak_rss_mb = stats.process_rss_mb

    def get_stats(self) -> dict:
        """Return current lifecycle stats."""
        stats = get_memory_stats()
        stats.model_state = self._state
        stats.peak_rss_mb = self._peak_rss_mb
        return {
            "state": self._state.value,
            "inference_count": self._inference_count,
            "peak_rss_mb": round(self._peak_rss_mb, 1),
            "current_rss_mb": round(stats.process_rss_mb, 1),
            "system_available_mb": round(stats.system_available_mb, 1),
            "load_duration_s": round(self._load_duration_s, 3),
            "idle_timeout_s": self.idle_timeout_s,
            "idle_s": (
                round(time.time() - self._last_inference_time, 1)
                if self._last_inference_time
                else 0
            ),
            "budget": {
                "max_loaded_ram_mb": self.budget.max_loaded_ram_mb,
                "max_total_ram_mb": self.budget.max_total_ram_mb,
            },
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> bool:
        """Load the model.  Must be called under self._lock."""
        if not Path(self.model_path).exists():
            logger.error("Model file not found: %s", self.model_path)
            return False

        self._state = ModelState.LOADING
        t0 = time.time()

        try:
            # For the LiteRT-LM CLI model, "loading" means verifying the file
            # is accessible and pre-warming (actual inference subprocess is
            # spawned per-call; this validates the model can be read).
            model_size_mb = Path(self.model_path).stat().st_size / 1024 / 1024
            logger.info(
                "Loading model: %s (%.1f MB)",
                Path(self.model_path).name,
                model_size_mb,
            )

            self._load_duration_s = time.time() - t0
            self._state = ModelState.LOADED
            self._last_inference_time = time.time()
            self._last_load_time = time.time()

            if self._on_load:
                self._on_load()

            logger.info("Model loaded in %.3fs", self._load_duration_s)
            return True

        except Exception:
            logger.exception("Failed to load model")
            self._state = ModelState.UNLOADED
            return False

    def _unload(self) -> None:
        """Unload the model to free memory."""
        with self._lock:
            if self._state != ModelState.LOADED:
                return

            self._state = ModelState.UNLOADING
            logger.info(
                "Unloading model (was idle %.1fs)",
                time.time() - self._last_inference_time,
            )

            # Terminate any running subprocess
            if self._process is not None:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=5.0)
                except Exception:
                    pass
                self._process = None

            self._state = ModelState.UNLOADED

            if self._on_unload:
                self._on_unload()

            logger.info("Model unloaded")

    def _monitor_loop(self) -> None:
        """Background loop: check idle timeout and memory pressure."""
        while not self._monitor_stop.wait(timeout=self.monitor_interval_s):
            try:
                self._check_idle_timeout()
                self._check_memory_pressure()
            except Exception:
                logger.exception("Monitor loop error")

    def _check_idle_timeout(self) -> None:
        """Unload the model if it's been idle too long."""
        if self._state != ModelState.LOADED:
            return
        if self._last_inference_time == 0:
            return
        idle_s = time.time() - self._last_inference_time
        if idle_s >= self.idle_timeout_s:
            logger.info(
                "Idle timeout reached (%.1fs >= %.1fs)", idle_s, self.idle_timeout_s
            )
            self._unload()

    def _check_memory_pressure(self) -> None:
        """Respond to system memory pressure."""
        if self._state != ModelState.LOADED:
            return

        stats = get_memory_stats()

        # Critical: system memory dangerously low
        if stats.system_memory_critical(self.budget):
            logger.warning(
                "CRITICAL memory pressure (avail=%.0f MB < %.0f MB) — force-unloading",
                stats.system_available_mb,
                self.budget.critical_memory_threshold_mb,
            )
            self._unload()
            return

        # Low: system memory below threshold
        if stats.system_memory_low(self.budget):
            logger.warning(
                "Low memory pressure (avail=%.0f MB < %.0f MB) — unloading model",
                stats.system_available_mb,
                self.budget.low_memory_threshold_mb,
            )
            self._unload()
            return

        # Process exceeds budget
        if stats.exceeds_budget(self.budget):
            logger.warning(
                "Process RSS (%.0f MB) exceeds budget (%.0f MB) — unloading",
                stats.process_rss_mb,
                self.budget.max_total_ram_mb,
            )
            self._unload()
