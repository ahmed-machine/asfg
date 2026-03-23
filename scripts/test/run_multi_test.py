#!/usr/bin/env python3
"""
Multi-image test runner for the alignment pipeline.

Runs the full pipeline (extract, stitch, georef, align) on a set of
test images from different camera systems, then evaluates results.

Usage:
    python3 scripts/test/run_multi_test.py [--skip-download] [--skip-georef] [--entity ENTITY_ID ...]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paths_config import get_reference

REFERENCE = get_reference("kh9_dzb1212")
ANCHORS = str(PROJECT_ROOT / "data" / "bahrain_anchor_gcps.json")
OUTPUT_DIR = PROJECT_ROOT / "output" / "multi-test"

# Test entities: 2 per camera system
TEST_ENTITIES = {
    # KH-4 (CORONA) — single TIF, no stitching
    "DS1052-1073DA157": {"dataset": "corona2", "date": "1969/09/27", "camera": "KH-4"},
    "DS1104-1057DA025": {"dataset": "corona2", "date": "1968/08/11", "camera": "KH-4"},
    # KH-7 (GAMBIT) — single TIF, no stitching
    "DZB00403600089H015001": {"dataset": "declassii", "date": "1967/02/08", "camera": "KH-7"},
    "DZB00403600089H016001": {"dataset": "declassii", "date": "1967/02/08", "camera": "KH-7"},
    # KH-9 (HEXAGON) — .tgz archive, needs stitching
    "D3C1213-200346A003": {"dataset": "declassiii", "date": "1977/08/27", "camera": "KH-9"},
    "D3C1217-100109A007": {"dataset": "declassiii", "date": "1982/05/23", "camera": "KH-9"},
}

CSV_DIR = PROJECT_ROOT / "data" / "available"
CSVS = [
    str(CSV_DIR / "corona2_69b17d89ee62ff28.csv"),
    str(CSV_DIR / "declassii_69b17e00f346b5bd.csv"),
    str(CSV_DIR / "declassiii_69b17de7ab15497e.csv"),
]


def check_downloads():
    """Check which entities have been downloaded."""
    dl_dir = OUTPUT_DIR / "downloads"
    status = {}
    for eid, info in TEST_ENTITIES.items():
        ext = ".tgz" if info["camera"] == "KH-9" else ".tif"
        path = dl_dir / f"{eid}{ext}"
        status[eid] = {
            "downloaded": path.exists() and path.stat().st_size > 0,
            "path": str(path),
            "size_mb": round(path.stat().st_size / 1024 / 1024, 1) if path.exists() else 0,
        }
    return status


def run_georef(entities=None):
    """Run the process.py pipeline for extract/stitch/georef (skip align)."""
    entity_args = entities or list(TEST_ENTITIES.keys())
    cmd = [
        sys.executable, str(PROJECT_ROOT / "process.py"),
        "--csv", *CSVS,
        "--entities", *entity_args,
        "--reference", REFERENCE,
        "--output-dir", str(OUTPUT_DIR),
        "--skip-download",
        "--skip-align", "--skip-mosaic",
    ]

    print(f"Running georef for {len(entity_args)} entities...")
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=str(PROJECT_ROOT))
    return result.returncode


def run_alignment(entity_id: str, timeout: int = 6000):
    """Run auto-align on a single georeferenced image."""
    georef_path = OUTPUT_DIR / "georef" / f"{entity_id}_georef.tif"
    if not georef_path.exists():
        print(f"  ERROR: No georef output for {entity_id}")
        return None

    aligned_path = OUTPUT_DIR / "aligned" / f"{entity_id}_aligned.tif"
    diag_dir = OUTPUT_DIR / "diagnostics" / entity_id
    qa_path = diag_dir / "qa.json"

    os.makedirs(str(diag_dir), exist_ok=True)
    os.makedirs(str(OUTPUT_DIR / "aligned"), exist_ok=True)

    cmd = [
        sys.executable, str(PROJECT_ROOT / "auto-align.py"),
        str(georef_path),
        "-r", REFERENCE,
        "--anchors", ANCHORS,
        "-y", "--best",
        "--diagnostics-dir", str(diag_dir) + "/",
        "--qa-json", str(qa_path),
        "-o", str(aligned_path),
    ]

    info = TEST_ENTITIES[entity_id]
    print(f"\n{'=' * 60}")
    print(f"Aligning: {entity_id} ({info['camera']}, {info['date']})")
    print(f"{'=' * 60}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    start = time.time()
    log_lines = []

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, cwd=str(PROJECT_ROOT), bufsize=1,
        )

        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                decoded = line.decode("utf-8", errors="replace")
                sys.stdout.write(decoded)
                sys.stdout.flush()
                log_lines.append(decoded)

            elapsed = time.time() - start
            if elapsed > timeout:
                print(f"\n=== TIMEOUT after {elapsed:.0f}s ===")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break

        stderr_data = proc.stderr.read()
        exit_code = proc.returncode

    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    wall_clock = time.time() - start
    log_text = "".join(log_lines)

    # Save log
    log_path = diag_dir / "run.log"
    log_path.write_text(log_text)

    if stderr_data:
        stderr_path = diag_dir / "stderr.log"
        stderr_path.write_bytes(stderr_data)

    # Load QA results
    result = {
        "entity_id": entity_id,
        "camera": info["camera"],
        "date": info["date"],
        "exit_code": exit_code,
        "wall_clock_s": round(wall_clock, 1),
    }

    if qa_path.exists():
        try:
            qa = json.loads(qa_path.read_text())
            for report in qa.get("reports", []):
                if report.get("candidate") == "grid":
                    im = report.get("image_metrics", {})
                    result["score"] = round(im.get("score", 0), 1)
                    result["accepted"] = report.get("accepted", False)
                    result["patch_med"] = round(im.get("patch_med", 0))
                    result["stable_iou"] = round(im.get("stable_iou", 0), 3)
                    result["cv_mean_m"] = round(report.get("cv_mean_m", 0), 1) if report.get("cv_mean_m") is not None else None
        except Exception as e:
            result["error"] = str(e)

    return result


def run_stitch_qa(entity_id: str):
    """Run quality checks on stitched/georef output for all camera systems."""
    info = TEST_ENTITIES[entity_id]
    camera = info["camera"]

    stitched_path = OUTPUT_DIR / "stitched" / f"{entity_id}_stitched.tif"
    georef_path = OUTPUT_DIR / "georef" / f"{entity_id}_georef.tif"

    # Check if stitched output exists (may have been cleaned up)
    check_path = georef_path if georef_path.exists() else stitched_path
    if not check_path.exists():
        return {"entity_id": entity_id, "stitch_qa": "no output found"}

    # Check image dimensions and properties
    result = subprocess.run(
        ["gdalinfo", "-json", str(check_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {"entity_id": entity_id, "stitch_qa": f"gdalinfo failed: {result.stderr[:200]}"}

    info_data = json.loads(result.stdout)
    width, height = info_data["size"]

    qa_result = {
        "entity_id": entity_id,
        "camera": camera,
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 2) if height > 0 else 0,
        "bands": len(info_data.get("bands", [])),
    }

    # Check for obvious problems
    issues = []

    # Camera-aware aspect ratio thresholds
    aspect = width / height if height > 0 else 0
    min_aspect = {"KH-4": 3.0, "KH-7": 0.5, "KH-9": 2.0}.get(camera, 1.0)
    if aspect < min_aspect:
        issues.append(f"suspicious aspect ratio {aspect:.2f} (expected >{min_aspect} for {camera})")

    # Detect orientation status
    is_portrait = height > width
    qa_result["orientation"] = "portrait" if is_portrait else "landscape"
    if is_portrait and camera != "KH-7":
        issues.append(f"portrait orientation (aspect={aspect:.2f})")

    # Check for sub-frame seams in the raw extracted image
    extracted_dir = OUTPUT_DIR / "extracted" / entity_id
    raw_frames = []
    if extracted_dir.exists():
        raw_frames = sorted(
            str(extracted_dir / f) for f in os.listdir(str(extracted_dir))
            if f.lower().endswith(".tif") and "_rot" not in f and "_sub" not in f
        )
    if len(raw_frames) == 1:
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from preprocess.stitch import detect_subframe_seams
            seams = detect_subframe_seams(raw_frames[0])
            qa_result["subframe_seams_detected"] = len(seams)
            if seams:
                qa_result["subframe_seam_positions"] = seams
        except Exception as e:
            qa_result["subframe_detection_error"] = str(e)

    # Check for black borders (possible stitching alignment issue)
    # Read a few sample rows and check for all-zero edges
    try:
        from osgeo import gdal
        ds = gdal.Open(str(check_path))
        if ds:
            band = ds.GetRasterBand(1)
            import numpy as np

            # Check left and right 1% of image
            edge_w = max(10, width // 100)

            # Left edge
            left = band.ReadAsArray(0, height // 2, edge_w, min(100, height))
            if left is not None:
                left_zeros = np.mean(left == 0) * 100
                if left_zeros > 80:
                    issues.append(f"left edge {left_zeros:.0f}% black")

            # Right edge
            right = band.ReadAsArray(width - edge_w, height // 2, edge_w, min(100, height))
            if right is not None:
                right_zeros = np.mean(right == 0) * 100
                if right_zeros > 80:
                    issues.append(f"right edge {right_zeros:.0f}% black")

            # Check for seam artifacts at expected stitch boundaries
            # A bad stitch shows as a vertical band of brightness discontinuity
            mid_strip = band.ReadAsArray(0, height // 2, width, 1)
            if mid_strip is not None:
                mid_strip = mid_strip.astype(float).flatten()
                # Compute gradient
                grad = np.abs(np.diff(mid_strip))
                # Large gradients at positions far from edges might indicate seam
                inner = grad[width // 10: -width // 10]
                if len(inner) > 0:
                    p99 = np.percentile(inner, 99)
                    spikes = np.where(inner > max(p99, 50))[0]
                    if len(spikes) > 0:
                        # Check if spikes are clustered (indicating a seam)
                        for spike_pos in spikes:
                            actual_pos = spike_pos + width // 10
                            pct = actual_pos / width * 100
                            if 10 < pct < 90:  # Only flag interior spikes
                                # Check if it's a sustained jump (seam) vs single pixel noise
                                local = mid_strip[max(0, actual_pos - 5):actual_pos + 5]
                                jump = abs(float(mid_strip[actual_pos + 1]) - float(mid_strip[actual_pos]))
                                if jump > 30:
                                    issues.append(f"possible seam at {pct:.1f}% (jump={jump:.0f})")
                                    break

            ds = None
    except Exception as e:
        issues.append(f"pixel check failed: {e}")

    # Post-georef orientation verification against reference
    if georef_path.exists() and os.path.exists(REFERENCE):
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from preprocess.orientation import verify_orientation_against_reference
            post_georef_rotation = verify_orientation_against_reference(
                str(georef_path), REFERENCE
            )
            qa_result["post_georef_orientation_check"] = post_georef_rotation
            if post_georef_rotation != 0:
                issues.append(f"post-georef orientation off by {post_georef_rotation}°")
            else:
                qa_result["orientation_verified"] = True
        except Exception as e:
            qa_result["orientation_check_error"] = str(e)

    qa_result["issues"] = issues
    qa_result["passed"] = len(issues) == 0
    return qa_result


def print_summary(results, stitch_results):
    """Print a summary table of all results."""
    print("\n" + "=" * 80)
    print("MULTI-IMAGE TEST SUMMARY")
    print("=" * 80)

    # Stitch QA
    print("\n--- Stitch QA ---")
    for sr in stitch_results:
        if sr.get("stitch_qa") == "n/a (single frame)":
            continue
        eid = sr["entity_id"]
        passed = sr.get("passed", False)
        issues = sr.get("issues", [])
        status = "PASS" if passed else "FAIL"
        print(f"  {eid}: {status}")
        if issues:
            for issue in issues:
                print(f"    - {issue}")

    # Alignment results
    print("\n--- Alignment Results ---")
    print(f"  {'Entity':<30} {'Camera':<6} {'Date':<12} {'Score':>6} {'Patch':>6} {'Acc':>4} {'Time':>6}")
    print(f"  {'-'*30} {'-'*6} {'-'*12} {'-'*6} {'-'*6} {'-'*4} {'-'*6}")

    for r in results:
        if r is None:
            continue
        eid = r["entity_id"]
        cam = r["camera"]
        date = r["date"]
        score = r.get("score", "ERR")
        patch = r.get("patch_med", "ERR")
        acc = "yes" if r.get("accepted") else "no"
        wc = f"{r['wall_clock_s']:.0f}s"
        ec = r.get("exit_code", -1)

        if ec != 0:
            print(f"  {eid:<30} {cam:<6} {date:<12} {'FAIL':>6} {'':>6} {'':>4} {wc:>6}")
        else:
            print(f"  {eid:<30} {cam:<6} {date:<12} {score:>6} {patch:>5}m {acc:>4} {wc:>6}")

    # Summary stats
    successful = [r for r in results if r and r.get("exit_code") == 0]
    accepted = [r for r in successful if r.get("accepted")]
    print(f"\n  Total: {len(results)}, Successful: {len(successful)}, Accepted: {len(accepted)}")

    # Save summary
    summary = {
        "stitch_qa": stitch_results,
        "alignment": [r for r in results if r is not None],
        "total": len(results),
        "successful": len(successful),
        "accepted": len(accepted),
    }
    summary_path = OUTPUT_DIR / "multi_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-image alignment test")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip USGS download (use existing files)")
    parser.add_argument("--skip-georef", action="store_true",
                        help="Skip georef (use existing georef outputs)")
    parser.add_argument("--skip-align", action="store_true",
                        help="Only run stitch QA, skip alignment")
    parser.add_argument("--entity", nargs="+", default=None,
                        help="Only process specific entity IDs")
    parser.add_argument("--timeout", type=int, default=6000,
                        help="Per-image alignment timeout in seconds (default: 6000)")
    args = parser.parse_args()

    entities = args.entity or list(TEST_ENTITIES.keys())
    entities = [e for e in entities if e in TEST_ENTITIES]

    if not entities:
        print("No valid entities specified")
        sys.exit(1)

    print(f"Testing {len(entities)} entities:")
    for eid in entities:
        info = TEST_ENTITIES[eid]
        print(f"  {eid} ({info['camera']}, {info['date']})")

    # Check downloads
    dl_status = check_downloads()
    missing = [eid for eid in entities if not dl_status[eid]["downloaded"]]
    if missing:
        print(f"\nMissing downloads: {missing}")
        if not args.skip_download:
            print("Run downloads first or use --skip-download")
            sys.exit(1)
        else:
            print("Skipping missing downloads, will process what's available")
            entities = [e for e in entities if dl_status[e]["downloaded"]]

    # Georef
    if not args.skip_georef:
        print("\n--- Running extract/stitch/georef ---")
        rc = run_georef(entities)
        if rc != 0:
            print(f"WARNING: georef exited with code {rc}")

    # Stitch QA
    print("\n--- Running stitch QA ---")
    stitch_results = []
    for eid in entities:
        sr = run_stitch_qa(eid)
        stitch_results.append(sr)
        if sr.get("issues"):
            print(f"  {eid}: {len(sr['issues'])} issue(s)")
            for issue in sr["issues"]:
                print(f"    - {issue}")
        elif sr.get("passed"):
            print(f"  {eid}: PASS")

    if args.skip_align:
        print_summary([], stitch_results)
        return

    # Alignment
    print("\n--- Running alignment ---")
    results = []
    for eid in entities:
        georef_path = OUTPUT_DIR / "georef" / f"{eid}_georef.tif"
        if not georef_path.exists():
            print(f"  Skipping {eid} — no georef output")
            continue
        result = run_alignment(eid, timeout=args.timeout)
        results.append(result)

    print_summary(results, stitch_results)


if __name__ == "__main__":
    main()
