"""OOM prevention helpers for the declass-process pipeline.

Rationale: the KH-9 PC per-segment path has a pathological peak-memory
profile. On 2026-04-21 an overnight run on D3C1213-200346A003 crashed a
MacBook Pro (MacBookPro18,4) with a jetsam kill at 02:36 followed by a
kernel watchdog panic at 08:57. Memory counters at jetsam time:
``uncompressed = 9,457,439 pages × 16 KB ≈ 144 GB`` demand, ``free =
4,101 pages = 64 MB`` — i.e. the OS was swap-thrashing the Apple
silicon SSD compressor until the kernel couldn't service its watchdog.

The pipeline itself doesn't monitor system memory. This module adds:

1. ``check_memory_or_warn()`` — one-shot health check at stage start.
   Warns (and optionally aborts) when less than a configured free-RAM
   floor is available before the pipeline commits to a long-running
   memory-heavy stage.

2. ``apply_process_memory_cap()`` — soft ``RLIMIT_AS`` cap. When set,
   Python raises ``MemoryError`` instead of the OS killing the process
   silently. The cap is preferred to a hard hang because a MemoryError
   unwinds Python's stack, prints a traceback, and exits cleanly.

3. ``log_memory_pressure()`` — on-demand debug snapshot of system +
   process memory, callable between heavy stages for diagnostic logs.

All helpers degrade to no-ops if ``psutil`` isn't available — the
module imports are deferred and guarded.
"""

from __future__ import annotations

import os
import sys


# Defaults tuned for the KH-9 PC per-segment path. Override via env vars
# when running on machines with different RAM budgets.
_DEFAULT_MIN_FREE_GB = 12.0         # warn below this
_DEFAULT_ABORT_FLOOR_GB = 4.0       # refuse to start below this
_DEFAULT_SOFT_CAP_GB = 0.0          # 0 = disabled; set explicitly to opt in


def _read_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(
            f"  [memory_guard] ignoring non-numeric {name}={raw!r}, "
            f"using default {default} GB",
            flush=True,
        )
        return default


def _available_memory_gb() -> float | None:
    """Return system memory available for use, in GB, or None if psutil
    is missing."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        return float(psutil.virtual_memory().available) / (1024.0 ** 3)
    except Exception:
        return None


def _total_memory_gb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        return float(psutil.virtual_memory().total) / (1024.0 ** 3)
    except Exception:
        return None


def check_memory_or_warn(
    stage_label: str,
    min_free_gb: float | None = None,
    abort_floor_gb: float | None = None,
) -> None:
    """Warn when available RAM is below ``min_free_gb``; sys.exit(1)
    when below ``abort_floor_gb``. Called at the start of memory-heavy
    stages so a trivially under-provisioned machine bails before it
    starts thrashing.

    Override defaults via ``DECLASS_MIN_FREE_GB`` /
    ``DECLASS_ABORT_FLOOR_GB`` env vars.
    """
    min_free = (
        float(min_free_gb) if min_free_gb is not None
        else _read_float_env("DECLASS_MIN_FREE_GB", _DEFAULT_MIN_FREE_GB)
    )
    abort_floor = (
        float(abort_floor_gb) if abort_floor_gb is not None
        else _read_float_env(
            "DECLASS_ABORT_FLOOR_GB", _DEFAULT_ABORT_FLOOR_GB
        )
    )
    avail = _available_memory_gb()
    total = _total_memory_gb()
    if avail is None:
        print(
            f"  [memory_guard] {stage_label}: psutil unavailable — "
            f"skipping memory check",
            flush=True,
        )
        return
    total_str = f"{total:.1f}" if total is not None else "?"
    if avail < abort_floor:
        print(
            f"  [memory_guard] {stage_label}: ABORT — only "
            f"{avail:.1f} GB free of {total_str} GB total "
            f"(floor {abort_floor:.1f} GB). The KH-9 PC per-segment path "
            f"peaked at 144 GB uncompressed demand in the 2026-04-21 "
            f"crash. Close browsers/IDEs or run on a bigger box. To "
            f"override, set DECLASS_ABORT_FLOOR_GB=0.",
            flush=True,
        )
        sys.exit(1)
    if avail < min_free:
        print(
            f"  [memory_guard] {stage_label}: WARNING — only "
            f"{avail:.1f} GB free of {total_str} GB total "
            f"(recommended {min_free:.1f} GB). Expect swap thrashing "
            f"during 14-param fits + Phase 3 TPS warp. To silence, set "
            f"DECLASS_MIN_FREE_GB lower.",
            flush=True,
        )
    else:
        print(
            f"  [memory_guard] {stage_label}: {avail:.1f} GB free of "
            f"{total_str} GB total (ok)",
            flush=True,
        )


def apply_process_memory_cap(soft_cap_gb: float | None = None) -> None:
    """Install a soft RLIMIT_AS cap on this Python process.

    When the process's virtual-address allocations exceed the cap,
    Python raises ``MemoryError``. That's recoverable — the pipeline can
    unwind, drop caches, and fail one scene cleanly — whereas letting
    the OS jetsam-kill the process loses work and, on Apple silicon,
    can kernel-panic the whole machine.

    ``soft_cap_gb`` overrides the env default. Pass 0 (or leave
    ``DECLASS_MEMORY_CAP_GB`` unset) to disable.
    """
    cap_gb = (
        float(soft_cap_gb) if soft_cap_gb is not None
        else _read_float_env("DECLASS_MEMORY_CAP_GB", _DEFAULT_SOFT_CAP_GB)
    )
    if cap_gb <= 0:
        return
    try:
        import resource
    except ImportError:
        print(
            "  [memory_guard] resource module unavailable — "
            "cannot install RLIMIT_AS cap",
            flush=True,
        )
        return
    cap_bytes = int(cap_gb * (1024 ** 3))
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_hard = hard if hard != resource.RLIM_INFINITY else cap_bytes
        resource.setrlimit(resource.RLIMIT_AS, (cap_bytes, new_hard))
        print(
            f"  [memory_guard] RLIMIT_AS soft cap set to {cap_gb:.1f} GB "
            f"(Python MemoryError will raise before OS jetsam kills us)",
            flush=True,
        )
    except (ValueError, OSError) as exc:
        print(
            f"  [memory_guard] could not set RLIMIT_AS to {cap_gb:.1f} "
            f"GB: {exc} — continuing without cap",
            flush=True,
        )


def log_memory_pressure(tag: str) -> None:
    """Print a snapshot of system + process memory, for diagnostic logs
    between heavy stages. No-op when psutil is missing.
    """
    try:
        import psutil
    except ImportError:
        return
    try:
        sys_mem = psutil.virtual_memory()
        proc = psutil.Process()
        rss_gb = proc.memory_info().rss / (1024.0 ** 3)
        print(
            f"  [memory_guard/{tag}] sys_free={sys_mem.available / 1024**3:.1f}"
            f" GB sys_total={sys_mem.total / 1024**3:.1f} GB "
            f"proc_rss={rss_gb:.2f} GB sys_pct={sys_mem.percent:.0f}%",
            flush=True,
        )
    except Exception:
        pass
