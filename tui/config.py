"""Horizon TUI configuration — cluster SSH settings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

_CONFIG_PATH = Path.home() / ".config" / "horizon" / "config.json"

# ── Defaults ─────────────────────────────────────────────────────────────────

CLUSTER_HOST = "155.54.210.99"
CLUSTER_USER = "jtpalma"
# Path on the cluster where the repo lives (or will be cloned to)
CLUSTER_REPO_PATH = "/slurm/home/jtpalma/safecircle/horizon"
GITHUB_REPO = "https://github.com/safecircleia/horizon.git"


@dataclass
class ClusterConfig:
    host: str = CLUSTER_HOST
    user: str = CLUSTER_USER
    repo_path: str = CLUSTER_REPO_PATH
    github_repo: str = GITHUB_REPO

    @property
    def remote(self) -> str:
        return f"{self.user}@{self.host}"

    def save(self) -> None:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> ClusterConfig:
        if _CONFIG_PATH.exists():
            data = json.loads(_CONFIG_PATH.read_text())
            return cls(
                **{k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            )
        return cls()


# Singleton — shared across the TUI session
_cfg: ClusterConfig | None = None


def get_cluster_config() -> ClusterConfig:
    global _cfg
    if _cfg is None:
        _cfg = ClusterConfig.load()
    return _cfg
