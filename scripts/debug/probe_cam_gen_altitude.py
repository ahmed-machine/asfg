"""Quick sanity check for the ASP cam_gen altitude derivation.

Runs ``cam_gen_opticalbar_per_subframe`` on each sub-frame of a scene and
prints the derived altitude, ECEF iC magnitude, focal length, and mean
corner lat/lon. Used to validate Bahrain KH-9 D3C1213 (expected altitude
~170 km per segment, matching the nominal orbital altitude) before
re-running the full per-segment pipeline.

Usage:
    python3 scripts/debug/probe_cam_gen_altitude.py <segments_dir> [scene_id]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys


def _find_scene_id(segments_dir: str) -> str | None:
    matches = glob.glob(os.path.join(segments_dir, "*_seg00_rot180_clean.tif"))
    if not matches:
        matches = glob.glob(os.path.join(segments_dir, "*_seg00_rot180.tif"))
    if not matches:
        return None
    base = os.path.basename(matches[0])
    for suffix in ("_seg00_rot180_clean.tif", "_seg00_rot180.tif"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return None


def _list_segments(segments_dir: str, scene_id: str) -> list[tuple[int, str]]:
    """(index, path) for each sub-frame; prefer the cleaned rot180 files."""
    out: list[tuple[int, str]] = []
    i = 0
    while True:
        clean = os.path.join(
            segments_dir, f"{scene_id}_seg{i:02d}_rot180_clean.tif"
        )
        raw = os.path.join(segments_dir, f"{scene_id}_seg{i:02d}_rot180.tif")
        if os.path.isfile(clean):
            out.append((i, clean))
        elif os.path.isfile(raw):
            out.append((i, raw))
        else:
            break
        i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("segments_dir", help="Directory containing *_seg*_rot180*.tif")
    ap.add_argument("scene_id", nargs="?", help="Scene id (auto-detected if omitted)")
    ap.add_argument("--profile", default="kh9",
                    help="Profile name (kh9 or kh4; default kh9)")
    ap.add_argument("--dem", default=None,
                    help="DEM path (defaults to output/dem.tif if present)")
    ap.add_argument("--allow-no-dem", action="store_true",
                    help="Proceed without a DEM. DEFAULT IS TO REFUSE: cam_gen "
                         "altitude without --refine-camera is unreliable and "
                         "can return unphysical values (e.g. 51 km) that look "
                         "like valid 4-corner fits. Only use for exploration.")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    sys.path.insert(0, project_root)

    from align.params import load_profile
    from preprocess.camera_model import (
        cam_gen_opticalbar_per_subframe,
    )

    profile = load_profile(args.profile)
    camera_params = profile.camera.to_dict() if hasattr(profile.camera, "to_dict") else vars(profile.camera)

    scene_id = args.scene_id or _find_scene_id(args.segments_dir)
    if scene_id is None:
        print(f"ERROR: could not auto-detect scene id in {args.segments_dir}")
        return 1

    segs = _list_segments(args.segments_dir, scene_id)
    if not segs:
        print(f"ERROR: no seg*_rot180 files for {scene_id}")
        return 1

    # Load the real USGS 4-corner bbox for this scene from the declassified
    # catalog CSVs. cam_gen is calibrated to these; interpolated sub-frame
    # corners are unreliable (they collide with our 180° rotation for Aft
    # sub-frames and give degenerate altitudes).
    strip_corners = None
    import csv
    for cat in glob.glob(os.path.join(project_root, "data", "available", "*.csv")):
        try:
            with open(cat, encoding="latin1") as fh:
                for row in csv.DictReader(fh):
                    if row.get("Entity ID") == scene_id:
                        strip_corners = {
                            "NW": (float(row["NW Corner Lat dec"]),
                                   float(row["NW Corner Long dec"])),
                            "NE": (float(row["NE Corner Lat dec"]),
                                   float(row["NE Corner Long dec"])),
                            "SE": (float(row["SE Corner Lat dec"]),
                                   float(row["SE Corner Long dec"])),
                            "SW": (float(row["SW Corner Lat dec"]),
                                   float(row["SW Corner Long dec"])),
                        }
                        break
            if strip_corners:
                break
        except Exception:
            continue
    if strip_corners is None:
        print(f"ERROR: could not find {scene_id} in data/available/*.csv")
        return 1

    n = len(segs)
    print(f"Scene {scene_id}: {n} sub-frames; profile={args.profile}")
    print(f"Strip corners (USGS): {strip_corners}")

    dem = args.dem
    if dem is None:
        for candidate in ("output/dem.tif", "output/dem.vrt"):
            if os.path.isfile(os.path.join(project_root, candidate)):
                dem = os.path.join(project_root, candidate)
                break
    if dem:
        print(f"Using DEM: {dem}")
    elif args.allow_no_dem:
        print(
            "WARNING: no DEM supplied and --allow-no-dem set. "
            "cam_gen runs without --refine-camera; altitude is unreliable "
            "and results should be labelled invalid_for_comparison.",
            file=sys.stderr,
        )
    else:
        print(
            "ERROR: no DEM found (tried --dem, output/dem.tif, output/dem.vrt). "
            "cam_gen without a DEM cannot refine altitude and has been observed "
            "to return 51 km altitudes on Bahrain that look like valid 4-corner "
            "fits but poison downstream per-segment fits. "
            "Pass --allow-no-dem to bypass this check (results will be printed "
            "with an invalid_for_comparison marker).",
            file=sys.stderr,
        )
        return 2

    # cam_gen needs the image whose 4 corners match the supplied
    # --lon-lat-values. That's the STITCHED frame, not a sub-frame —
    # stitched covers the full USGS ground bbox; sub-frames cover a
    # fraction of it. Passing a sub-frame with the full-strip corners
    # makes cam_gen solve for a camera close enough to the ground to
    # pack the whole bbox into one sub-frame image (observed 51 km).
    stitched_candidates = [
        os.path.join(args.segments_dir, f"{scene_id}_cropped.tif"),
        os.path.join(args.segments_dir, f"{scene_id}.tif"),
    ]
    stitched = next((p for p in stitched_candidates if os.path.isfile(p)), None)
    if stitched is None:
        print(f"ERROR: no stitched frame in {args.segments_dir}; "
              f"tried {stitched_candidates}")
        return 1

    info = cam_gen_opticalbar_per_subframe(
        sub_path=stitched,
        corners=strip_corners,
        camera_params=camera_params,
        dem_path=dem,
        output_tsai=os.path.join(args.segments_dir, f"{scene_id}_probe.tsai"),
        overwrite=True,
    )
    if info is None:
        print("ERROR: cam_gen failed (ASP not found or subprocess error)")
        return 1

    iC = info["iC"]
    iC_mag_km = ((float(iC[0]) ** 2 + float(iC[1]) ** 2 + float(iC[2]) ** 2) ** 0.5
                 / 1000)
    lat_deg = math.degrees(info["lat_rad"])
    lon_deg = math.degrees(info["lon_rad"])
    print()
    print(f"altitude_m = {info['altitude_m']:,.0f}")
    print(f"f          = {info['focal_length']:.4f} m  (nominal {camera_params.get('focal_length')})")
    print(f"iC_mag_km  = {iC_mag_km:,.1f}")
    print(f"lat/lon    = {lat_deg:.3f}, {lon_deg:.3f}")
    in_orbit = 140_000 <= info["altitude_m"] <= 280_000
    print(f"sanity     = {'OK (in KH orbit range)' if in_orbit else 'REJECT (outside 140–280 km)'}")
    if not dem:
        print("status     = invalid_for_comparison (ran without DEM)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
