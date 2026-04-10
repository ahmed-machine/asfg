#!/usr/bin/env python3
"""Download fine-tuned SSD weights for RoMa and SEA-RAFT."""

import argparse
import hashlib
import os
import sys
from pathlib import Path

WEIGHTS = {
    "roma_ssd": {
        "filename": "roma_ssd.pth",
        "url": None,  # TODO: set when hosted (HuggingFace, S3, etc.)
        "sha256": None,
    },
    "searaft_ssd": {
        "filename": "searaft_ssd.pth",
        "url": None,  # TODO: set when hosted
        "sha256": None,
    },
}

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "declass-process")
# __file__ lives at scripts/experimental/ssd/download_weights.py; weights dir is
# align/weights/ at the repo root, so three ".." to reach it.
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "align", "weights")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_weight(name, info, force=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    cache_path = os.path.join(CACHE_DIR, info["filename"])
    dest_path = os.path.join(WEIGHTS_DIR, info["filename"])

    # Already present locally?
    if os.path.exists(dest_path) and not force:
        print(f"  {name}: {dest_path} exists (use --force to re-download)")
        return True

    if os.path.exists(cache_path) and not force:
        print(f"  {name}: found in cache")
        _link(cache_path, dest_path)
        return True

    if info["url"] is None:
        print(f"  {name}: no download URL configured — train with scripts/experimental/ssd/finetune.py")
        return False

    # Download
    print(f"  {name}: downloading from {info['url']}...")
    import urllib.request
    tmp_path = cache_path + ".tmp"
    try:
        urllib.request.urlretrieve(info["url"], tmp_path)
    except Exception as e:
        print(f"  {name}: download failed: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False

    # Verify checksum
    if info["sha256"]:
        actual = _sha256(tmp_path)
        if actual != info["sha256"]:
            print(f"  {name}: checksum mismatch (expected {info['sha256'][:12]}..., got {actual[:12]}...)")
            os.remove(tmp_path)
            return False

    os.rename(tmp_path, cache_path)
    _link(cache_path, dest_path)
    print(f"  {name}: saved to {dest_path}")
    return True


def _link(src, dst):
    """Symlink src -> dst, replacing dst if it exists."""
    dst = os.path.abspath(dst)
    src = os.path.abspath(src)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument("names", nargs="*", default=list(WEIGHTS.keys()),
                        help=f"Weights to download (default: all). Choices: {list(WEIGHTS.keys())}")
    args = parser.parse_args()

    print("Downloading SSD weights...")
    ok = True
    for name in args.names:
        if name not in WEIGHTS:
            print(f"  Unknown weight: {name}")
            ok = False
            continue
        if not download_weight(name, WEIGHTS[name], force=args.force):
            ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
