"""On-device adaptive execution configuration.

Defines model tiers, device capability detection, and runtime parameters
for 24/7 on-device execution per issue #8 constraints.

Model tiers:
    Standard  (~50 MB, INT8) — for 6 GB+ RAM devices
    Lite      (~25 MB, INT4) — for 4-6 GB RAM devices

The runtime selects the appropriate tier at startup based on available
device RAM, and adjusts execution behaviour based on battery level and
thermal state.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Model tier definitions
# ---------------------------------------------------------------------------

ModelTier = Literal["standard", "lite"]


@dataclasses.dataclass(frozen=True)
class TierConfig:
    """Immutable configuration for one model tier."""

    name: ModelTier
    description: str
    model_filename: str
    quantization: str
    # Size budgets
    max_disk_mb: float
    max_loaded_ram_mb: float
    # Inference
    max_latency_ms: float
    kv_cache_tokens: int
    prefill_lengths: tuple[int, ...]

    def model_path(self, models_dir: str = "models") -> Path:
        subdir = f"mobile-{self.name}"
        return Path(models_dir) / subdir / self.model_filename


TIER_STANDARD = TierConfig(
    name="standard",
    description="INT8 model for devices with >= 6 GB RAM",
    model_filename="horizon-mobile-int8_q8_ekv1280.litertlm",
    quantization="dynamic_int8",
    max_disk_mb=50.0,
    max_loaded_ram_mb=100.0,
    max_latency_ms=500.0,
    kv_cache_tokens=1280,
    prefill_lengths=(8, 64, 128, 256, 512),
)

TIER_LITE = TierConfig(
    name="lite",
    description="INT4 model for devices with 4-6 GB RAM",
    model_filename="horizon-mobile-int4_q4_block128_ekv1280.litertlm",
    quantization="dynamic_int4_block128",
    max_disk_mb=30.0,
    max_loaded_ram_mb=60.0,
    max_latency_ms=500.0,
    kv_cache_tokens=1280,
    prefill_lengths=(8, 64, 128, 256, 512),
)

TIERS: dict[ModelTier, TierConfig] = {
    "standard": TIER_STANDARD,
    "lite": TIER_LITE,
}


# ---------------------------------------------------------------------------
# Device capability detection
# ---------------------------------------------------------------------------


def _detect_thermal_state() -> Literal["nominal", "warm", "critical"]:
    """Best-effort CPU thermal state from Linux sensors (dev host proxy).

    Android/iOS would use their platform thermal APIs; psutil.sensors_temperatures()
    is Linux-only and returns nothing on macOS/Windows and most CI runners, so
    unsupported platforms fall back to "nominal".
    """
    try:
        import psutil

        temps = psutil.sensors_temperatures()
    except Exception:
        return "nominal"
    if not temps:
        return "nominal"
    max_temp = max(
        (reading.current for readings in temps.values() for reading in readings),
        default=0.0,
    )
    if max_temp >= 85.0:
        return "critical"
    if max_temp >= 70.0:
        return "warm"
    return "nominal"


@dataclasses.dataclass
class DeviceCapabilities:
    """Runtime snapshot of device hardware capabilities."""

    total_ram_mb: float
    available_ram_mb: float
    battery_pct: float = 100.0
    is_charging: bool = True
    thermal_state: Literal["nominal", "warm", "critical"] = "nominal"
    cpu_cores: int = 4

    def select_tier(self) -> ModelTier:
        """Select the best model tier for this device."""
        if self.total_ram_mb >= 6000:
            return "standard"
        return "lite"

    @classmethod
    def detect(cls) -> DeviceCapabilities:
        """Detect device capabilities on the current host (best-effort).

        On Android, this would use android.os.ActivityManager and
        BatteryManager. On a dev host, we use psutil as a rough proxy.
        """
        try:
            import psutil

            vm = psutil.virtual_memory()
            battery = psutil.sensors_battery()
            return cls(
                total_ram_mb=vm.total / 1024 / 1024,
                available_ram_mb=vm.available / 1024 / 1024,
                battery_pct=battery.percent if battery else 100.0,
                is_charging=(battery.power_plugged or False) if battery else True,
                thermal_state=_detect_thermal_state(),
                cpu_cores=psutil.cpu_count(logical=True) or 4,
            )
        except (ImportError, Exception):
            # Fallback: assume a 4 GB device, fully charged
            return cls(total_ram_mb=4096, available_ram_mb=2048)


# ---------------------------------------------------------------------------
# Runtime execution policy
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ExecutionPolicy:
    """Adaptive runtime policy that adjusts inference behaviour."""

    tier: TierConfig
    # Message batching: accumulate N messages before invoking the LLM
    batch_size: int = 1
    # Inactivity timeout before unloading the model (seconds)
    idle_unload_s: int = 300  # 5 minutes
    # Maximum reload time budget (seconds)
    max_reload_s: float = 2.0
    # Whether to skip LLM for messages caught by the rule pre-filter
    use_rule_prefilter: bool = True
    # Minimum battery level to continue running (%)
    min_battery_pct: float = 15.0
    # Thermal throttle: increase batch window when warm, pause when critical
    thermal_batch_multiplier: float = 1.0

    @classmethod
    def from_device(cls, capabilities: DeviceCapabilities) -> ExecutionPolicy:
        """Create a policy tuned for the detected device."""
        tier_name = capabilities.select_tier()
        tier = TIERS[tier_name]

        policy = cls(tier=tier)

        # Battery-aware adjustments
        if capabilities.battery_pct < 20 and not capabilities.is_charging:
            policy.batch_size = 5  # batch more to reduce wake-ups
            policy.idle_unload_s = 120  # unload sooner
        elif capabilities.battery_pct < 50 and not capabilities.is_charging:
            policy.batch_size = 3

        # Thermal adjustments
        if capabilities.thermal_state == "warm":
            policy.thermal_batch_multiplier = 2.0
            policy.batch_size = max(policy.batch_size, 3)
        elif capabilities.thermal_state == "critical":
            policy.thermal_batch_multiplier = 5.0
            policy.batch_size = max(policy.batch_size, 10)

        return policy

    def effective_batch_size(self) -> int:
        return max(1, int(self.batch_size * self.thermal_batch_multiplier))

    def should_run(
        self,
        battery_pct: float,
        is_charging: bool,
        *,
        safety_escalation: bool = False,
    ) -> bool:
        """Whether inference should proceed given current battery state.

        High-confidence rule-prefilter escalations (Level 0-3 hits) must
        still run below min_battery_pct — the battery policy throttles
        routine LLM calls, not safety-critical ones.
        """
        if is_charging or safety_escalation:
            return True
        return battery_pct >= self.min_battery_pct

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.name,
            "batch_size": self.batch_size,
            "effective_batch_size": self.effective_batch_size(),
            "idle_unload_s": self.idle_unload_s,
            "use_rule_prefilter": self.use_rule_prefilter,
            "min_battery_pct": self.min_battery_pct,
            "thermal_batch_multiplier": self.thermal_batch_multiplier,
        }
