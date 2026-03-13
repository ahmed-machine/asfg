#!/usr/bin/env python3
"""
Automated pipeline for USGS declassified satellite imagery:
  1. Catalog  — Parse CSVs, identify camera systems, group into strips
  2. Download — Fetch .tgz/.tif from USGS M2M API
  3. Extract  — Unpack .tgz archives (KH-9 multi-frame strips)
  4. Stitch   — VRT-based frame stitching into panoramic strips (KH-9)
  5. Georef   — Rough georeferencing using CSV corner coordinates
  6. Align    — Generate strip manifests, run auto-align.py
  7. Mosaic   — Assemble aligned outputs by date/mission

All stages are idempotent — re-running skips completed work.
"""

import argparse
import json
import os
import re
import subprocess
import sys

from declass.catalog import parse_csvs, group_into_strips, filter_scenes
from declass.camera import identify_camera
from declass.usgs import download_scenes, fetch_corners_batch
from declass.extract import extract_archive, list_frames
from declass.stitch import stitch_frames, detect_strip_direction, compute_subset_corners
from declass.georef import georef_with_corners
from declass.mosaic import build_all_mosaics, build_mosaic
from declass.reference import fetch_sentinel2_reference


def load_progress(output_dir: str) -> dict:
    """Load processing progress from progress.json."""
    path = os.path.join(output_dir, "progress.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed": {}, "failed": {}}


def save_progress(output_dir: str, progress: dict):
    """Save processing progress to progress.json."""
    path = os.path.join(output_dir, "progress.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


def ensure_dirs(output_dir: str):
    """Create output directory structure."""
    for subdir in ["downloads", "extracted", "stitched", "georef",
                   "aligned", "mosaic", "manifests", "diagnostics"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)


def process_scene(scene, output_dir: str, file_map: dict, reference: str,
                  progress: dict, dry_run: bool = False) -> bool:
    """Process a single scene through extract → stitch → georef.

    Returns True on success, False on failure.
    """
    eid = scene.entity_id
    camera = scene.camera_system

    # Check if already completed
    georef_path = os.path.join(output_dir, "georef", f"{eid}_georef.tif")
    if os.path.exists(georef_path):
        print(f"  [skip] Already georeferenced: {eid}")
        progress["completed"][eid] = {"stage": "georef"}
        return True

    if dry_run:
        print(f"  [dry-run] Would process: {eid} ({camera.name})")
        return True

    # Check for downloaded file
    file_path = file_map.get(eid)
    if not file_path or not os.path.exists(file_path):
        print(f"  WARNING: No downloaded file for {eid}, skipping")
        progress["failed"][eid] = "no_download"
        return False

    try:
        # Step 3: Extract
        print(f"\n  --- Extract: {eid} ---")
        entity_dir = extract_archive(file_path, output_dir, eid)
        frames = list_frames(entity_dir)

        if not frames:
            print(f"  WARNING: No frames found for {eid}")
            progress["failed"][eid] = "no_frames"
            return False

        # Step 4: Stitch (KH-9 only — multi-frame strips)
        corners = scene.corners
        if camera.needs_stitching and len(frames) > 1:
            print(f"\n  --- Stitch: {eid} ({len(frames)} frames) ---")

            # Detect strip direction from corner coordinates
            is_reversed = detect_strip_direction(corners)
            direction = "reversed" if is_reversed else "forward"
            print(f"  Strip direction: {direction}")

            stitched_path = os.path.join(output_dir, "stitched", f"{eid}_stitched.tif")
            stitch_frames(frames, stitched_path, output_dir)
            input_for_georef = stitched_path

            # No frame subset selection — stitch all frames.
            # Corner coordinates from CSV cover the full strip.
        else:
            input_for_georef = frames[0]

        # Step 5: Georef
        print(f"\n  --- Georef: {eid} ---")
        georef_with_corners(input_for_georef, georef_path, corners)

        # Clean up stitched intermediate to save disk
        stitched_path = os.path.join(output_dir, "stitched", f"{eid}_stitched.tif")
        if os.path.exists(stitched_path):
            os.remove(stitched_path)
            print(f"  Removed intermediate: {os.path.basename(stitched_path)}")

        progress["completed"][eid] = {"stage": "georef"}
        return True

    except Exception as e:
        print(f"  ERROR processing {eid}: {e}")
        progress["failed"][eid] = str(e)
        return False


def generate_manifest(scenes: list, output_dir: str, reference: str) -> str:
    """Generate a strip manifest JSON for auto-align.py.

    Returns path to the manifest file.
    """
    manifests_dir = os.path.join(output_dir, "manifests")
    os.makedirs(manifests_dir, exist_ok=True)

    jobs = []
    for scene in scenes:
        eid = scene.entity_id
        georef_path = os.path.join(output_dir, "georef", f"{eid}_georef.tif")
        if not os.path.exists(georef_path):
            continue

        aligned_path = os.path.join(output_dir, "aligned", f"{eid}_aligned.tif")
        if os.path.exists(aligned_path):
            print(f"  [skip] Already aligned: {eid}")
            continue

        diag_dir = os.path.join(output_dir, "diagnostics", eid)

        jobs.append({
            "input": os.path.abspath(georef_path),
            "output": os.path.abspath(aligned_path),
            "diagnostics_dir": os.path.abspath(diag_dir),
        })

    if not jobs:
        print("  No scenes to align (all done or no georef outputs)")
        return None

    manifest = {
        "shared": {
            "reference": os.path.abspath(reference),
            "device": "auto",
            "global_search": True,
        },
        "jobs": jobs,
    }

    manifest_path = os.path.join(manifests_dir, "alignment_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Generated manifest with {len(jobs)} jobs: {manifest_path}")
    return manifest_path


def run_alignment(manifest_path: str):
    """Run auto-align.py with a strip manifest."""
    if not manifest_path:
        return

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto-align.py")
    cmd = [sys.executable, script, "--strip-manifest", manifest_path]
    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  WARNING: Alignment exited with code {result.returncode}")


def scan_frames_dir(frames_dir: str) -> dict:
    """Scan a directory of pre-downloaded sub-frame TIFFs and group by entity.

    Expects filenames like D3C1213-200346F002_a.tif through _g.tif.

    Returns:
        Dict mapping entity_id -> sorted list of frame paths.
    """
    entities = {}
    pattern = re.compile(r'^(.+)_([a-z])\.tif$', re.IGNORECASE)

    for fname in sorted(os.listdir(frames_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        entity_id = m.group(1)
        entities.setdefault(entity_id, []).append(
            os.path.join(frames_dir, fname)
        )

    # Sort frames within each entity (already sorted by filename, but be explicit)
    for eid in entities:
        entities[eid].sort()

    return entities


def process_frames_dir(frames_dir: str, output_dir: str, crop_bbox: tuple = None,
                       dry_run: bool = False):
    """Process pre-downloaded sub-frame TIFFs: stitch, georef, and optionally crop.

    Args:
        frames_dir: Directory containing {entity}_{frame}.tif files.
        output_dir: Output directory for stitched/georef/cropped results.
        crop_bbox: Optional (west, south, east, north) to clip output.
        dry_run: If True, just show what would be done.
    """
    # Create output subdirectories
    for subdir in ["stitched", "georef", "cropped", "mosaic"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    # Step 1: Scan and group
    print("\n" + "=" * 60)
    print("Step 1: Scan frames directory")
    print("=" * 60)

    entities = scan_frames_dir(frames_dir)
    if not entities:
        print(f"  ERROR: No frame TIFFs found in {frames_dir}")
        sys.exit(1)

    print(f"  Found {len(entities)} entities:")
    for eid, frames in sorted(entities.items()):
        print(f"    {eid}: {len(frames)} sub-frames ({os.path.basename(frames[0])} .. {os.path.basename(frames[-1])})")

    # Detect dataset from entity prefix
    first_eid = next(iter(entities))
    camera = identify_camera(first_eid)
    dataset = camera.ee_dataset
    print(f"  Camera system: {camera.name}, dataset: {dataset}")

    if dry_run:
        print("\n[dry-run] Would fetch metadata, stitch, georef, and crop. Exiting.")
        return

    # Step 2: Fetch corner coordinates from M2M API
    print("\n" + "=" * 60)
    print("Step 2: Fetch corner coordinates from USGS M2M API")
    print("=" * 60)

    entity_ids = sorted(entities.keys())
    corners_map = fetch_corners_batch(dataset, entity_ids)

    # Step 3: Stitch sub-frames per entity
    print("\n" + "=" * 60)
    print("Step 3: Stitch sub-frames")
    print("=" * 60)

    stitched_paths = {}
    for eid in entity_ids:
        frames = entities[eid]
        corners = corners_map[eid]

        print(f"\n{'─' * 50}")
        print(f"Stitching: {eid} ({len(frames)} sub-frames)")
        print(f"{'─' * 50}")

        stitched_path = os.path.join(output_dir, "stitched", f"{eid}_stitched.tif")

        if len(frames) == 1:
            # Single frame, just copy
            import shutil
            shutil.copy2(frames[0], stitched_path)
            print(f"  Single frame, copied")
        else:
            is_reversed = detect_strip_direction(corners)
            if is_reversed:
                print(f"  Strip direction: reversed (east→west)")
            else:
                print(f"  Strip direction: forward (west→east)")
            stitch_frames(frames, stitched_path, output_dir)

        stitched_paths[eid] = stitched_path

    # Step 4: Georeference
    print("\n" + "=" * 60)
    print("Step 4: Georeference with M2M corner coordinates")
    print("=" * 60)

    georef_paths = {}
    for eid in entity_ids:
        stitched_path = stitched_paths[eid]
        corners = corners_map[eid]
        georef_path = os.path.join(output_dir, "georef", f"{eid}_georef.tif")

        print(f"\n  Georeferencing: {eid}")
        georef_with_corners(stitched_path, georef_path, corners)
        georef_paths[eid] = georef_path

    # Clean up stitched intermediates
    for eid, stitched_path in stitched_paths.items():
        if os.path.exists(stitched_path):
            os.remove(stitched_path)
            print(f"  Removed intermediate: {os.path.basename(stitched_path)}")

    # Step 5: Crop (optional)
    if crop_bbox:
        print("\n" + "=" * 60)
        print(f"Step 5: Crop to bbox ({crop_bbox[0]},{crop_bbox[1]},{crop_bbox[2]},{crop_bbox[3]})")
        print("=" * 60)

        west, south, east, north = crop_bbox
        for eid in entity_ids:
            georef_path = georef_paths[eid]
            cropped_path = os.path.join(output_dir, "cropped", f"{eid}_cropped.tif")

            if os.path.exists(cropped_path):
                print(f"  [skip] Already cropped: {eid}")
                continue

            cmd = [
                "gdalwarp",
                "-te", str(west), str(south), str(east), str(north),
                "-te_srs", "EPSG:4326",
                "-co", "COMPRESS=LZW",
                "-co", "PREDICTOR=2",
                "-co", "TILED=YES",
                "-co", "BIGTIFF=IF_SAFER",
                georef_path,
                cropped_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  WARNING: Crop failed for {eid}: {result.stderr}")
                # Remove empty/failed output
                if os.path.exists(cropped_path):
                    os.remove(cropped_path)
            else:
                # Check if the cropped file has any data (entity might not overlap bbox)
                info_result = subprocess.run(
                    ["gdalinfo", "-json", cropped_path],
                    capture_output=True, text=True
                )
                if info_result.returncode == 0:
                    info = json.loads(info_result.stdout)
                    w, h = info["size"]
                    if w > 0 and h > 0:
                        print(f"  Cropped: {eid} ({w}x{h})")
                    else:
                        os.remove(cropped_path)
                        print(f"  [skip] {eid} — no overlap with crop bbox")

    # Step 6: Mosaic
    print("\n" + "=" * 60)
    print("Step 6: Mosaic")
    print("=" * 60)

    # Use cropped files if crop was applied, otherwise georef files
    if crop_bbox:
        mosaic_inputs = sorted(
            os.path.join(output_dir, "cropped", f)
            for f in os.listdir(os.path.join(output_dir, "cropped"))
            if f.endswith(".tif")
        )
    else:
        mosaic_inputs = [georef_paths[eid] for eid in entity_ids]

    mosaic_path = os.path.join(output_dir, "mosaic", "mosaic.tif")
    result = build_mosaic(mosaic_inputs, mosaic_path)
    if result:
        print(f"  Mosaic output: {result}")

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline complete")
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print(f"  Entities processed: {len(entity_ids)}")
    if crop_bbox:
        cropped_dir = os.path.join(output_dir, "cropped")
        cropped_count = len([f for f in os.listdir(cropped_dir) if f.endswith(".tif")])
        print(f"  Cropped outputs: {cropped_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Automated pipeline for USGS declassified satellite imagery"
    )
    parser.add_argument(
        "--csv", nargs="+", required=False,
        help="Path(s) to USGS EarthExplorer CSV catalog files",
    )
    parser.add_argument(
        "--reference", "-r", default=None,
        help="Path to a correctly-aligned reference GeoTIFF",
    )
    parser.add_argument(
        "--auto-reference", action="store_true",
        help="Auto-fetch a Sentinel-2 reference image",
    )
    parser.add_argument(
        "--output-dir", "-o", required=False, default="output",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--entities", nargs="+", default=None,
        help="Process only these specific entity IDs",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip USGS download (use existing files in downloads/)",
    )
    parser.add_argument(
        "--skip-align", action="store_true",
        help="Skip alignment step (georef only)",
    )
    parser.add_argument(
        "--skip-mosaic", action="store_true",
        help="Skip mosaic assembly step",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without processing",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Resume from an existing output directory (reads progress.json)",
    )
    parser.add_argument(
        "--frames-dir", default=None,
        help="Directory containing pre-downloaded {entity}_{frame}.tif files (bypasses catalog/download)",
    )
    parser.add_argument(
        "--crop-bbox", default=None,
        help="Crop output to WEST,SOUTH,EAST,NORTH bbox in decimal degrees (e.g. 50.15,25.55,50.90,26.40)",
    )
    args = parser.parse_args()

    # Handle --frames-dir mode (bypass catalog/download)
    if args.frames_dir:
        crop_bbox = None
        if args.crop_bbox:
            parts = args.crop_bbox.split(",")
            if len(parts) != 4:
                parser.error("--crop-bbox must be WEST,SOUTH,EAST,NORTH (4 comma-separated values)")
            crop_bbox = tuple(float(x) for x in parts)

        process_frames_dir(
            frames_dir=args.frames_dir,
            output_dir=args.output_dir,
            crop_bbox=crop_bbox,
            dry_run=args.dry_run,
        )
        return

    # Handle --resume
    if args.resume:
        args.output_dir = args.resume
        progress = load_progress(args.output_dir)
        # Try to find CSVs and reference from previous manifest
        manifest_path = os.path.join(args.output_dir, "manifests", "alignment_manifest.json")
        if os.path.exists(manifest_path) and not args.reference:
            with open(manifest_path) as f:
                prev = json.load(f)
            args.reference = prev.get("shared", {}).get("reference")
        if not args.csv:
            # Look for CSVs in data/available/
            csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "available")
            if os.path.exists(csv_dir):
                args.csv = [
                    os.path.join(csv_dir, f)
                    for f in os.listdir(csv_dir) if f.endswith(".csv")
                ]
    else:
        progress = load_progress(args.output_dir)

    if not args.csv:
        parser.error("--csv is required (or use --resume with an existing output directory)")

    # Validate reference
    if not args.reference and not args.auto_reference and not args.skip_align:
        parser.error("--reference or --auto-reference required (or use --skip-align)")

    output_dir = args.output_dir
    ensure_dirs(output_dir)

    # === Stage 1: Catalog ===
    print("\n" + "=" * 60)
    print("Stage 1: Parse catalog CSVs")
    print("=" * 60)

    scenes = parse_csvs(args.csv)

    # Filter by entity IDs if specified
    if args.entities:
        scenes = filter_scenes(scenes, entity_ids=args.entities, download_only=False)
        print(f"  Filtered to {len(scenes)} scenes matching --entities")

    # Filter to downloadable only
    downloadable = [s for s in scenes if s.download_available]
    print(f"  Total: {len(scenes)} scenes, {len(downloadable)} downloadable")

    # Group into strips
    strips = group_into_strips(downloadable)
    print(f"  Grouped into {len(strips)} strips:")
    for strip in strips:
        print(f"    {strip.camera_system.name} {strip.date} {strip.mission} "
              f"cam={strip.camera_designation} ({len(strip.scenes)} scenes)")

    if args.dry_run:
        print("\n[dry-run] Would process the above scenes. Exiting.")
        return

    # === Stage 2: Download ===
    print("\n" + "=" * 60)
    print("Stage 2: Download imagery")
    print("=" * 60)

    file_map = download_scenes(downloadable, output_dir, skip_download=args.skip_download)
    found = sum(1 for v in file_map.values() if v)
    print(f"  Files available: {found}/{len(downloadable)}")
    save_progress(output_dir, progress)

    # === Stage 3-5: Extract, Stitch, Georef ===
    print("\n" + "=" * 60)
    print("Stage 3-5: Extract, Stitch, Georef")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    for scene in downloadable:
        print(f"\n{'─' * 50}")
        print(f"Processing: {scene.entity_id} ({scene.camera_system.name}, {scene.acquisition_date})")
        print(f"{'─' * 50}")

        ok = process_scene(scene, output_dir, file_map, args.reference, progress)
        if ok:
            success_count += 1
        else:
            fail_count += 1
        save_progress(output_dir, progress)

    print(f"\n  Georef complete: {success_count} succeeded, {fail_count} failed")

    # === Auto-reference ===
    reference = args.reference
    if args.auto_reference and not reference:
        print("\n" + "=" * 60)
        print("Auto-reference: Fetching Sentinel-2 composite")
        print("=" * 60)

        # Compute overall bbox from all scenes
        all_lats = []
        all_lons = []
        for s in downloadable:
            for corner in s.corners.values():
                all_lats.append(corner[0])
                all_lons.append(corner[1])

        bbox = (min(all_lons), min(all_lats), max(all_lons), max(all_lats))
        ref_path = os.path.join(output_dir, "reference", "sentinel2_reference.tif")
        reference = fetch_sentinel2_reference(bbox, ref_path)

    # === Stage 6: Align ===
    if not args.skip_align and reference:
        print("\n" + "=" * 60)
        print("Stage 6: Alignment")
        print("=" * 60)

        manifest_path = generate_manifest(downloadable, output_dir, reference)
        run_alignment(manifest_path)

    # === Stage 7: Mosaic ===
    if not args.skip_mosaic and not args.skip_align and reference:
        print("\n" + "=" * 60)
        print("Stage 7: Mosaic assembly")
        print("=" * 60)

        aligned_dir = os.path.join(output_dir, "aligned")
        mosaic_dir = os.path.join(output_dir, "mosaic")
        mosaics = build_all_mosaics(downloadable, aligned_dir, mosaic_dir)
        print(f"  Built {len(mosaics)} mosaics")

    # === Summary ===
    print("\n" + "=" * 60)
    print("Pipeline complete")
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print(f"  Scenes processed: {success_count}/{len(downloadable)}")
    if progress["failed"]:
        print(f"  Failed:")
        for eid, err in progress["failed"].items():
            print(f"    {eid}: {err}")


if __name__ == "__main__":
    main()
