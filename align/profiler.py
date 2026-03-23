"""Lightweight hierarchical pipeline profiler.

Captures wall time, CPU time, I/O wait (derived), memory, and GPU memory
per section.  Context-manager based, negligible overhead, always on.
"""

import resource
import time
from contextlib import contextmanager
from typing import Any


def _gpu_mem_mb() -> float | None:
    """Current MPS/CUDA allocated memory in MB, or None."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory() / (1024 * 1024)
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return None


def _gpu_peak_mb() -> float | None:
    """Driver-level peak GPU memory in MB, or None."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return torch.mps.driver_allocated_memory() / (1024 * 1024)
    except Exception:
        pass
    return None


class PipelineProfiler:
    """Hierarchical timer with resource tracking."""

    def __init__(self):
        self._stack: list[str] = []           # current nesting path
        self._entries: list[dict] = []        # flat list in entry order
        self._t0_global = time.perf_counter()

    @contextmanager
    def section(self, name: str):
        """Time a named section.  Nests automatically via the call stack."""
        self._stack.append(name)
        path = ".".join(self._stack)
        depth = len(self._stack)

        # Reserve slot at entry time so ordering is top-down
        idx = len(self._entries)
        self._entries.append(None)  # placeholder

        t0 = time.perf_counter()
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        gpu0 = _gpu_mem_mb()

        try:
            yield
        finally:
            t1 = time.perf_counter()
            ru1 = resource.getrusage(resource.RUSAGE_SELF)
            gpu1 = _gpu_mem_mb()

            wall = t1 - t0
            cpu_user = ru1.ru_utime - ru0.ru_utime
            cpu_sys = ru1.ru_stime - ru0.ru_stime
            cpu = cpu_user + cpu_sys
            io_wait = max(0.0, wall - cpu)

            # macOS ru_maxrss is in bytes
            import sys as _sys
            rss_divisor = 1024 * 1024 if _sys.platform == "darwin" else 1024
            peak_rss_mb = ru1.ru_maxrss / rss_divisor

            entry = {
                "path": path,
                "name": name,
                "depth": depth,
                "wall_s": round(wall, 2),
                "cpu_s": round(cpu, 2),
                "cpu_user_s": round(cpu_user, 2),
                "cpu_sys_s": round(cpu_sys, 2),
                "io_wait_s": round(io_wait, 2),
                "peak_rss_mb": round(peak_rss_mb, 1),
                "io_read_blocks": ru1.ru_inblock - ru0.ru_inblock,
                "io_write_blocks": ru1.ru_oublock - ru0.ru_oublock,
                "gpu_mem_mb": round(gpu1 - gpu0, 1) if gpu0 is not None and gpu1 is not None else None,
                "gpu_peak_mb": round(_gpu_peak_mb(), 1) if _gpu_peak_mb() is not None else None,
            }
            self._entries[idx] = entry

            # Top-level sections print backwards-compatible timing line
            if depth == 1:
                print(f"  [{name}] {wall:.1f}s", flush=True)

            self._stack.pop()

    def to_dict(self) -> dict:
        """Return nested dict suitable for JSON serialisation."""
        root: dict[str, Any] = {"sections": {}}

        for entry in self._entries:
            if entry is None:
                continue
            parts = entry["path"].split(".")
            node = root["sections"]
            for i, part in enumerate(parts):
                if part not in node:
                    node[part] = {}
                if i < len(parts) - 1:
                    node[part].setdefault("children", {})
                    node = node[part]["children"]
                else:
                    # Leaf — write metrics (children may already exist)
                    children = node[part].get("children")
                    node[part] = {
                        k: v for k, v in entry.items()
                        if k not in ("path", "name", "depth")
                    }
                    if children:
                        node[part]["children"] = children

        root["total_wall_s"] = round(time.perf_counter() - self._t0_global, 2)
        return root

    def print_waterfall(self):
        """Print indented timing tree to stdout."""
        completed = [e for e in self._entries if e is not None]
        if not completed:
            return
        print("\n=== Timing Profile ===", flush=True)
        hdr = f"{'Section':<44s} {'Wall':>8s} {'CPU':>8s} {'IO Wait':>8s} {'GPU Mem':>8s} {'RSS':>8s}"
        print(hdr, flush=True)
        print("-" * len(hdr), flush=True)

        for entry in completed:
            indent = "  " * (entry["depth"] - 1)
            label = f"{indent}{entry['name']}"
            wall = f"{entry['wall_s']:.1f}s"
            cpu = f"{entry['cpu_s']:.1f}s"
            io_w = f"{entry['io_wait_s']:.1f}s"
            gpu = f"{entry['gpu_mem_mb']:.0f}MB" if entry["gpu_mem_mb"] is not None else ""
            rss = f"{entry['peak_rss_mb'] / 1024:.1f}GB" if entry["peak_rss_mb"] > 0 else ""
            print(f"{label:<44s} {wall:>8s} {cpu:>8s} {io_w:>8s} {gpu:>8s} {rss:>8s}",
                  flush=True)


class _NullProfiler:
    """No-op fallback when profiler is not passed."""

    @contextmanager
    def section(self, name: str):
        yield
