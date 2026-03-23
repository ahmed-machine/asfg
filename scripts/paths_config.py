"""Load machine-specific paths from data/local_paths.yaml."""

import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = PROJECT_ROOT / "data" / "local_paths.yaml"
_EXAMPLE_PATH = PROJECT_ROOT / "data" / "local_paths.yaml.example"


def load_paths() -> dict:
    """Load and expand paths from data/local_paths.yaml.

    Returns dict with keys 'targets' and 'references', each a dict
    of name -> expanded absolute path.
    """
    if not _CONFIG_PATH.exists():
        print(
            f"ERROR: {_CONFIG_PATH} not found.\n"
            f"Copy the example and fill in your paths:\n"
            f"  cp {_EXAMPLE_PATH} {_CONFIG_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)

    if not raw:
        print(f"ERROR: {_CONFIG_PATH} is empty.", file=sys.stderr)
        sys.exit(1)

    result = {}
    for section in ("targets", "references"):
        entries = raw.get(section, {})
        if not isinstance(entries, dict):
            print(f"ERROR: '{section}' in {_CONFIG_PATH} must be a mapping.", file=sys.stderr)
            sys.exit(1)
        result[section] = {k: os.path.expanduser(v) for k, v in entries.items()}

    return result


def get_target(name: str) -> str:
    """Load paths and return a specific target."""
    paths = load_paths()
    if name not in paths["targets"]:
        available = list(paths["targets"].keys())
        print(f"ERROR: Target '{name}' not in local_paths.yaml. Available: {available}", file=sys.stderr)
        sys.exit(1)
    return paths["targets"][name]


def get_reference(name: str) -> str:
    """Load paths and return a specific reference."""
    paths = load_paths()
    if name not in paths["references"]:
        available = list(paths["references"].keys())
        print(f"ERROR: Reference '{name}' not in local_paths.yaml. Available: {available}", file=sys.stderr)
        sys.exit(1)
    return paths["references"][name]
