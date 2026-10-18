"""Shared, conservative RAM+swap admission checks for experiment schedulers."""
from pathlib import Path
from .scheduling_config import MEMORY_RESUME_MARGIN_GIB


def memory_status(path="/proc/meminfo"):
    """Return GiB of RAM, swap use and conservative launch headroom."""
    try:
        fields = {}
        for line in Path(path).read_text().splitlines():
            name, rest = line.split(":", 1)
            if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                fields[name] = int(rest.split()[0]) / 1024**2
        total = fields["MemTotal"]
        available = fields["MemAvailable"]
        swap = fields["SwapTotal"] - fields["SwapFree"]
        if total <= 0 or not 0 <= available <= total or not 0 <= swap <= fields["SwapTotal"]:
            return None
        return {"total_gib": total, "available_gib": available, "swap_used_gib": swap,
                "ram_pressure_gib": total - available, "headroom_gib": available - swap}
    except (OSError, ValueError, KeyError, IndexError):
        return None


def memory_admission(snapshot, reserve, was_blocked=False, resume_margin=MEMORY_RESUME_MARGIN_GIB):
    """Pause below reserve, resume above reserve+margin; fail closed on errors."""
    if reserve < 0 or resume_margin < 0:
        raise ValueError("Memory reserve and resume margin must be nonnegative")
    if snapshot is None:
        return False
    threshold = reserve + (resume_margin if was_blocked else 0.0)
    return snapshot["headroom_gib"] >= threshold
