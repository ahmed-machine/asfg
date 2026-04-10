"""Preprocessing pipeline for USGS declassified satellite imagery."""

import subprocess
import time

def run_gdal_cmd(cmd, check=True, retries=3):
    """Run a shell command, retrying on failure, raising if *check* is True."""
    for attempt in range(retries):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result
        # If it failed, wait and retry
        if attempt < retries - 1:
            print(f"Warning: Command failed (attempt {attempt+1}/{retries}): {' '.join(cmd)}\nRetrying in 5 seconds...")
            time.sleep(5)
    
    # If we exhausted all retries
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed after {retries} attempts: {' '.join(cmd)}\n{result.stderr}")
    return result
