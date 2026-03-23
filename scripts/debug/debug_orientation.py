#!/usr/bin/env python3
"""Fast orientation detection debug tool.

Tests orientation detection on downsampled imagery for rapid iteration.
Runs in seconds per entity, not minutes.

Usage:
    python3 scripts/debug/debug_orientation.py
    python3 scripts/debug/debug_orientation.py --entity DS1052-1073DA157
"""

import argparse
import csv
import os
import sys
import time

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paths_config import get_reference

REFERENCE = get_reference("kh9_dzb1212")
OUTPUT_DIR = PROJECT_ROOT / "output" / "multi-test"

TEST_ENTITIES = {
    "DS1052-1073DA157": {"dataset": "corona2", "date": "1969/09/27", "camera": "KH-4"},
    "DS1104-1057DA025": {"dataset": "corona2", "date": "1968/08/11", "camera": "KH-4"},
    "DZB00403600089H015001": {"dataset": "declassii", "date": "1967/02/08", "camera": "KH-7"},
    "DZB00403600089H016001": {"dataset": "declassii", "date": "1967/02/08", "camera": "KH-7"},
    "D3C1213-200346A003": {"dataset": "declassiii", "date": "1977/08/27", "camera": "KH-9"},
}

# Expected results based on user feedback + reference-based verification.
# User originally reported DS1052/DS1104 as "upside down" but reference
# matching shows 0 or 90 is correct -- the issue was sub-frame stitching,
# not rotation.
EXPECTED = {
    "DS1052-1073DA157": None,      # reference says 90; user issue was stitching
    "DS1104-1057DA025": None,      # reference says 0; user issue was stitching
    "DZB00403600089H015001": None,  # correct (accept 0 or 90)
    "DZB00403600089H016001": None,  # correct (accept 0 or 90)
    "D3C1213-200346A003": 180,     # was upside down, reference confirms 180
}

CSV_DIR = PROJECT_ROOT / "data" / "available"
CSVS = [
    CSV_DIR / "corona2_69b17d89ee62ff28.csv",
    CSV_DIR / "declassii_69b17e00f346b5bd.csv",
    CSV_DIR / "declassiii_69b17de7ab15497e.csv",
]


def parse_corners_from_csv(entity_id):
    """Parse corner coordinates from CSV for a given entity."""
    for csvf in CSVS:
        with open(csvf, encoding="latin-1") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Entity ID"] == entity_id:
                    # Handle typo in some CSVs: "NW Cormer" vs "NW Corner"
                    nw_lat_key = "NW Cormer Lat dec" if "NW Cormer Lat dec" in row else "NW Corner Lat dec"
                    return {
                        "NW": (float(row[nw_lat_key]), float(row["NW Corner Long dec"])),
                        "NE": (float(row["NE Corner Lat dec"]), float(row["NE Corner Long dec"])),
                        "SE": (float(row["SE Corner Lat dec"]), float(row["SE Corner Long dec"])),
                        "SW": (float(row["SW Corner Lat dec"]), float(row["SW Corner Long dec"])),
                    }
    return None


def find_test_image(entity_id, info):
    """Find the raw/stitched image for an entity (pre-georef)."""
    # For KH-9: use stitched output if available, otherwise look at extracted frames
    stitched = OUTPUT_DIR / "stitched" / f"{entity_id}_stitched.tif"
    if stitched.exists():
        return str(stitched)

    # For single-frame: use extracted or downloaded
    extracted_dir = OUTPUT_DIR / "extracted" / entity_id
    if extracted_dir.exists():
        frames = sorted(
            str(extracted_dir / f) for f in os.listdir(str(extracted_dir))
            if f.lower().endswith(".tif") and "_rot" not in f and "_sub" not in f
        )
        if frames:
            return frames[0]

    ext = ".tgz" if info["camera"] == "KH-9" else ".tif"
    dl = OUTPUT_DIR / "downloads" / f"{entity_id}{ext}"
    if dl.exists() and ext == ".tif":
        return str(dl)

    return None


def main():
    parser = argparse.ArgumentParser(description="Fast orientation debug")
    parser.add_argument("--entity", nargs="+", default=None)
    args = parser.parse_args()

    from preprocess.orientation import (
        detect_orientation_against_reference,
        detect_orientation,
    )
    from preprocess.catalog import identify_camera

    entities = args.entity or list(TEST_ENTITIES.keys())
    entities = [e for e in entities if e in TEST_ENTITIES]

    print(f"Reference: {REFERENCE}")
    print(f"{'Entity':<30} {'Cam':<5} {'RefRot':>7} {'Inliers':>8} {'MetaRot':>8} "
          f"{'Expect':>7} {'Result':>8} {'Time':>6}")
    print("-" * 90)

    total_time = 0
    pass_count = 0
    fail_count = 0

    for eid in entities:
        info = TEST_ENTITIES[eid]
        corners = parse_corners_from_csv(eid)
        if corners is None:
            print(f"{eid:<30} {'':5} {'no CSV':>7}")
            fail_count += 1
            continue

        image_path = find_test_image(eid, info)
        if image_path is None:
            print(f"{eid:<30} {'':5} {'no img':>7}")
            fail_count += 1
            continue

        camera = identify_camera(eid)

        # Metadata-based detection (old method)
        meta_rot, _ = detect_orientation(image_path, corners, camera)

        # Reference-based detection (new method)
        t0 = time.time()
        ref_rot, _, n_inliers = detect_orientation_against_reference(
            image_path, corners, REFERENCE,
        )
        dt = time.time() - t0
        total_time += dt

        expected = EXPECTED.get(eid)
        if expected is None:
            # KH-7: we just need it to produce a valid result
            verdict = "OK"
            pass_count += 1
        elif ref_rot == expected:
            verdict = "PASS"
            pass_count += 1
        else:
            verdict = "FAIL"
            fail_count += 1

        exp_str = str(expected) if expected is not None else "any"
        print(f"{eid:<30} {info['camera']:<5} {ref_rot:>6} {n_inliers:>7} "
              f"{meta_rot:>7} {exp_str:>7} {verdict:>8} {dt:>5.1f}s")

    print("-" * 90)
    print(f"Total: {len(entities)}, Pass: {pass_count}, Fail: {fail_count}, "
          f"Time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
