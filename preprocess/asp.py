"""Ames Stereo Pipeline (ASP) tool discovery.

Centralizes ASP binary discovery for image_mosaic (sub-frame stitching).
"""

import glob
import os
import shutil
import subprocess


# ---------------------------------------------------------------------------
# ASP binary discovery
# ---------------------------------------------------------------------------

_KNOWN_ASP_DIRS = [
    os.path.expanduser("~/tools/StereoPipeline-3.6.0-2025-12-26-arm64-OSX/bin"),
    os.path.expanduser("~/tools/StereoPipeline/bin"),
    "/usr/local/bin",
    "/opt/StereoPipeline/bin",
]

_asp_cache: dict[str, str | None] = {}


def find_asp_tool(name: str) -> str | None:
    """Find an ASP binary by name.

    Checks PATH first, then known install locations. Caches results.
    """
    if name in _asp_cache:
        return _asp_cache[name]

    # Check PATH
    path = shutil.which(name)
    if path is not None:
        _asp_cache[name] = path
        return path

    # Check known ASP install directories
    for asp_dir in _KNOWN_ASP_DIRS:
        candidate = os.path.join(asp_dir, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            _asp_cache[name] = candidate
            return candidate

    # Also check for glob pattern (version-varying install dirs)
    for pattern in [os.path.expanduser("~/tools/StereoPipeline-*/bin")]:
        for d in sorted(glob.glob(pattern), reverse=True):  # newest first
            candidate = os.path.join(d, name)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                _asp_cache[name] = candidate
                return candidate

    _asp_cache[name] = None
    return None
