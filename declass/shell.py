"""Shared subprocess helper for GDAL and other CLI tools."""

import subprocess


def run_gdal_cmd(cmd, check=True):
    """Run a shell command, raising on failure if *check* is True."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result
