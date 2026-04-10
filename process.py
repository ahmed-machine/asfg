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
import copy
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

from preprocess.catalog import parse_csvs, group_into_strips, filter_scenes, identify_camera, select_best_mission_coverage
from preprocess.usgs import download_scenes, fetch_corners_batch, extract_archive, list_frames
from preprocess.stitch import stitch_frames, stitch_with_asp, detect_subframe_seams, split_at_seams
from preprocess.georef import georef_with_corners, coarse_align_and_crop
from preprocess.mosaic import build_all_mosaics, build_mosaic
from preprocess.georef import fetch_sentinel2_reference, build_composite_reference
from preprocess.orientation import swap_corners_180, detect_orientation, verify_orientation_against_reference
from preprocess.auto_anchors import generate_auto_anchors
from preprocess.experimental.match_ip import generate_strip_matches
from preprocess.camera_model import generate_camera, mapproject_image
from preprocess.dem import fetch_and_prepare_dem
from align.experimental.bundle_adjust import run_strip_bundle_adjustment
from align.params import load_profile
from align.models import ModelCache, get_torch_device
import paths
from paths import georef_metadata_path, ensure_pipeline_dirs


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


def _load_scene_metadata(cache_dir: str, entity_id: str) -> dict | None:
    path = georef_metadata_path(cache_dir, entity_id)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _save_scene_metadata(cache_dir: str, entity_id: str, payload: dict) -> str:
    path = georef_metadata_path(cache_dir, entity_id)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _default_scene_metadata(scene, cache_dir: str) -> dict:
    eid = scene.entity_id
    return {
        "entity_id": eid,
        "camera_designation": _camera_designation(scene),
        "profile": _profile_name_for_scene(scene),
        "gcp_corners": {k: list(v) for k, v in scene.corners.items()},
        "georef_path": os.path.abspath(paths.georef_path(cache_dir, eid)),
        "stitched_path": os.path.abspath(paths.stitched_path(cache_dir, eid)),
        "asp_camera_path": None,
        "asp_ortho_path": None,
        "primary_input_kind": None,
        "primary_input_path": None,
        "alignment_crop_path": None,
    }


def _merge_scene_metadata(cache_dir: str, scene, **updates) -> dict:
    metadata = _load_scene_metadata(cache_dir, scene.entity_id) or _default_scene_metadata(scene, cache_dir)
    metadata.update(updates)
    _save_scene_metadata(cache_dir, scene.entity_id, metadata)
    return metadata


def _camera_designation(scene) -> str:
    entity_id = scene.entity_id
    if scene.camera_system.entity_prefix == "D3C":
        parts = entity_id.split("-")
        if len(parts) == 2:
            for ch in parts[1]:
                if ch.isalpha():
                    return ch.upper()
    if scene.camera_system.entity_prefix == "DS1":
        parts = entity_id.split("-")
        if len(parts) == 2:
            suffix = parts[1]
            for i, ch in enumerate(suffix):
                if ch == "D" and i + 1 < len(suffix) and suffix[i + 1] in ("A", "F"):
                    return suffix[i + 1]
        ct = (scene.camera_type or "").strip().lower()
        if "aft" in ct:
            return "A"
        if "forward" in ct or "fore" in ct:
            return "F"
    if scene.camera_system.entity_prefix == "DZB":
        return "H"
    return (scene.camera_type or "X")[:1].upper()


def _profile_name_for_scene(scene) -> str:
    prefix = scene.camera_system.entity_prefix
    if prefix == "DS1":
        return "kh4"
    if prefix == "DZB":
        return "kh7"
    if prefix == "D3C":
        return "kh9"
    raise ValueError(f"Unsupported camera system for {scene.entity_id}")


def _camera_params_for_scene(scene) -> dict | None:
    profile = load_profile(_profile_name_for_scene(scene))
    if not profile.camera.is_panoramic:
        return None
    params = copy.deepcopy(profile.camera.to_dict())
    designation = _camera_designation(scene)
    if designation == "A":
        params["forward_tilt"] = -abs(params.get("forward_tilt", 0.0))
    elif designation == "F":
        params["forward_tilt"] = abs(params.get("forward_tilt", 0.0))
    return params


def _corners_from_metadata(scene, metadata: dict | None) -> dict:
    if metadata and isinstance(metadata.get("gcp_corners"), dict):
        return metadata["gcp_corners"]
    return {k: list(v) for k, v in scene.corners.items()}


def _bbox_from_corners(corners: dict) -> tuple[float, float, float, float]:
    lats = [float(v[0]) for v in corners.values()]
    lons = [float(v[1]) for v in corners.values()]
    return (min(lons), min(lats), max(lons), max(lats))


def _reference_resolution(reference_path: str) -> float | None:
    """Return the reference image's ground resolution in METRES.

    ASP mapproject is invoked with t_srs=EPSG:3857 (projected, metres), so
    the --tr value must be in metres. If the reference is stored in a
    geographic CRS (degrees), convert the degree-based pixel size to
    approximate metres at the image's centre latitude. Returning raw
    degrees here causes mapproject to reject the --tr value as "likely in
    degrees, while metres are expected".
    """
    if not reference_path or not os.path.exists(reference_path):
        return None
    import math
    import rasterio
    with rasterio.open(reference_path) as src:
        px_x = abs(src.transform.a)
        px_y = abs(src.transform.e)
        if src.crs and src.crs.is_geographic:
            lat_c = (src.bounds.top + src.bounds.bottom) * 0.5
            # WGS84 mean radius -> 1° latitude ~= 111.32 km; longitude
            # scales by cos(latitude). Use the smaller of the two axes so
            # we don't under-sample the image.
            m_per_deg_lat = 111320.0
            m_per_deg_lon = 111320.0 * math.cos(math.radians(lat_c))
            return float(min(px_x * m_per_deg_lon, px_y * m_per_deg_lat))
        return float(min(px_x, px_y))


def _path_is_stale(path: str | None, *dependencies: str | None) -> bool:
    if not path or not os.path.exists(path):
        return True
    path_mtime = os.path.getmtime(path)
    for dep in dependencies:
        if dep and os.path.exists(dep) and os.path.getmtime(dep) > path_mtime + 1e-6:
            return True
    return False


def _alignment_crop_path(cache_dir: str, entity_id: str, input_kind: str) -> str:
    crop_dir = "ortho" if input_kind == "asp_ortho" else "georef"
    safe_kind = re.sub(r"[^a-z0-9_]+", "_", input_kind.lower())
    return os.path.join(cache_dir, crop_dir, f"{entity_id}_{safe_kind}_cropped.tif")


def _per_segment_sub_frames(scene, cache_dir: str) -> list[str] | None:
    """Locate the extracted sub-frames for a scene in stitch order.

    KH-4 sub-frames are delivered (a, b, c, d) but stitched in reverse
    so the scan axis runs left-to-right; KH-7/KH-9 are stitched in
    delivery order. The per-segment processor needs them in stitch order
    (left-to-right along the scan axis), so we mirror what
    ``stitch_with_asp`` does.
    """
    frames = list_frames(paths.extracted_dir(cache_dir, scene.entity_id))
    if not frames:
        return None
    cam_name = (scene.camera_system.name or "").upper().replace("-", "")
    if cam_name.startswith("KH4"):
        return list(reversed(frames))
    return frames


def _maybe_generate_asp_ortho(scene, cache_dir: str, stitched_path: str,
                              corners: dict, reference: str | None) -> str | None:
    camera_params = _camera_params_for_scene(scene)
    if camera_params is None:
        return None
    if not os.path.exists(stitched_path):
        return None

    bbox = _bbox_from_corners(corners)
    dem_path = fetch_and_prepare_dem(
        west=bbox[0] - 0.1, south=bbox[1] - 0.1,
        east=bbox[2] + 0.1, north=bbox[3] + 0.1,
    )

    # 2OC §3.1: per-sub-image processing if the profile asks for it. Each
    # sub-frame gets its own cam_gen + mapproject, and the resulting
    # orthos are mosaicked via gdalbuildvrt. Falls back to the stitched
    # path if sub-frames aren't on disk (e.g. archive not extracted).
    profile = load_profile(_profile_name_for_scene(scene))
    use_per_segment = bool(getattr(profile.camera, "per_segment_ortho", False))
    if use_per_segment:
        sub_frames = _per_segment_sub_frames(scene, cache_dir)
        if sub_frames and len(sub_frames) > 1:
            from preprocess.camera_model import opticalbar_per_segment_precorrect
            seg_dir = paths.ortho_segments_dir(cache_dir, scene.entity_id)
            print(f"  [per_segment_ortho] {len(sub_frames)} sub-frames found for {scene.entity_id}")
            vrt_path = opticalbar_per_segment_precorrect(
                sub_frames=sub_frames,
                camera_params=camera_params,
                strip_corners=corners,
                output_dir=seg_dir,
                dem_path=dem_path,
                resolution=_reference_resolution(reference),
                t_srs="EPSG:3857",
                scene_id=scene.entity_id,
            )
            if vrt_path:
                return vrt_path
            print(f"  [per_segment_ortho] failed; falling back to stitched cam_gen")
        else:
            print(f"  [per_segment_ortho] no usable sub-frames for {scene.entity_id}; falling back to stitched cam_gen")

    cam_path = generate_camera(stitched_path, camera_params, corners, dem_path=dem_path)
    if cam_path is None:
        return None

    return mapproject_image(
        stitched_path,
        cam_path,
        dem_path=dem_path,
        output_path=paths.ortho_path(cache_dir, scene.entity_id),
        resolution=_reference_resolution(reference),
        t_srs="EPSG:3857",
    )


def _ensure_scene_asp_ortho(scene, cache_dir: str, reference: str | None,
                            metadata: dict | None = None, file_map: dict | None = None) -> dict:
    metadata = metadata or (_load_scene_metadata(cache_dir, scene.entity_id) or _default_scene_metadata(scene, cache_dir))
    if not reference or not os.path.exists(reference):
        return metadata
    if _camera_params_for_scene(scene) is None:
        return metadata

    asp_ortho_path = metadata.get("asp_ortho_path")
    if asp_ortho_path and os.path.exists(asp_ortho_path) and asp_ortho_path.endswith("_ortho_ba.tif"):
        return metadata

    georef_path = metadata.get("georef_path") or paths.georef_path(cache_dir, scene.entity_id)
    stitched_path = metadata.get("stitched_path") or paths.stitched_path(cache_dir, scene.entity_id)
    if not _path_is_stale(asp_ortho_path, stitched_path, georef_path):
        return metadata

    if not os.path.exists(stitched_path):
        file_path = (file_map or {}).get(scene.entity_id)
        if not file_path or not os.path.exists(file_path):
            return metadata
        entity_dir = extract_archive(file_path, cache_dir, scene.entity_id)
        frames = list_frames(entity_dir)
        if not frames:
            return metadata
        stitched_path = _stitch_if_needed(
            frames,
            scene.entity_id,
            scene.camera_system,
            paths.stitched_path(cache_dir, scene.entity_id),
            cache_dir,
        )

    if asp_ortho_path and os.path.exists(asp_ortho_path):
        os.remove(asp_ortho_path)

    corners = _corners_from_metadata(scene, metadata)
    asp_ortho_path = _maybe_generate_asp_ortho(scene, cache_dir, stitched_path, corners, reference)
    if not asp_ortho_path:
        return metadata

    asp_camera_path = paths.ba_camera_path(stitched_path)
    return _merge_scene_metadata(
        cache_dir,
        scene,
        stitched_path=os.path.abspath(stitched_path),
        asp_camera_path=os.path.abspath(asp_camera_path) if os.path.exists(asp_camera_path) else None,
        asp_ortho_path=os.path.abspath(asp_ortho_path),
    )


def _preferred_alignment_input_info(cache_dir: str, entity_id: str) -> tuple[str | None, str | None]:
    metadata = _load_scene_metadata(cache_dir, entity_id)
    if metadata:
        ortho_path = metadata.get("asp_ortho_path")
        if ortho_path and os.path.exists(ortho_path):
            return ("asp_ortho", ortho_path)
        georef_path = metadata.get("georef_path")
        if georef_path and os.path.exists(georef_path):
            return ("georef", georef_path)
    georef_path = paths.georef_path(cache_dir, entity_id)
    if os.path.exists(georef_path):
        return ("georef", georef_path)
    return (None, None)


def _preferred_alignment_input(cache_dir: str, entity_id: str) -> str | None:
    _, path = _preferred_alignment_input_info(cache_dir, entity_id)
    return path


def _check_duplicate_scans(frame_a: str, frame_b: str) -> bool:
    """Check if two frames are duplicate scans of the same film frame.

    Uses phase correlation at low resolution.  If the horizontal shift is
    less than 5% of image width, frames are considered duplicates.
    Tests both normal and 180°-flipped orientations of B.
    """
    import numpy as np
    from osgeo import gdal
    gdal.UseExceptions()

    ds_a = gdal.Open(frame_a)
    ds_b = gdal.Open(frame_b)
    if ds_a is None or ds_b is None:
        return False

    w_a, h_a = ds_a.RasterXSize, ds_a.RasterYSize
    w_b, h_b = ds_b.RasterXSize, ds_b.RasterYSize

    # Different sizes → not duplicates
    if abs(w_a - w_b) > 100 or abs(h_a - h_b) > 100:
        ds_a = ds_b = None
        return False

    # Read at ~2% scale
    scale = 0.02
    ow, oh = max(64, int(w_a * scale)), max(64, int(h_a * scale))

    a = ds_a.GetRasterBand(1).ReadAsArray(buf_xsize=ow, buf_ysize=oh).astype(np.float32)
    b = ds_b.GetRasterBand(1).ReadAsArray(buf_xsize=ow, buf_ysize=oh).astype(np.float32)
    ds_a = ds_b = None

    def _phase_corr_shift(img1, img2):
        """Return (dx, dy) pixel shift via phase correlation."""
        # Mask out black borders — crop to rows/cols where both have data
        valid1 = img1 > 10
        valid2 = img2 > 10
        both_valid = valid1 & valid2
        rows = np.any(both_valid, axis=1)
        cols = np.any(both_valid, axis=0)
        if not rows.any() or not cols.any():
            return 0, 0, 0.0
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        i1 = img1[r0:r1+1, c0:c1+1]
        i2 = img2[r0:r1+1, c0:c1+1]
        if i1.shape[0] < 16 or i1.shape[1] < 16:
            return 0, 0, 0.0

        A = np.fft.fft2(i1)
        B = np.fft.fft2(i2)
        cross = A * np.conj(B)
        cross /= np.abs(cross) + 1e-8
        corr = np.fft.ifft2(cross).real
        peak = np.unravel_index(np.argmax(corr), corr.shape)
        dy, dx = peak[0], peak[1]
        if dy > corr.shape[0] // 2:
            dy -= corr.shape[0]
        if dx > corr.shape[1] // 2:
            dx -= corr.shape[1]
        return dx, dy, float(corr.max())

    # Test normal orientation
    dx, dy, score = _phase_corr_shift(a, b)
    dx_fullres = abs(dx / scale)
    threshold = w_a * 0.05  # 5% of width

    if dx_fullres < threshold and score > 0.02:
        print(f"    Duplicate check: dx={dx_fullres:.0f}px "
              f"({100*dx_fullres/w_a:.1f}%), score={score:.3f} → duplicate")
        return True

    # Test B flipped 180°
    b_flip = b[::-1, ::-1]
    dx2, dy2, score2 = _phase_corr_shift(a, b_flip)
    dx2_fullres = abs(dx2 / scale)

    if dx2_fullres < threshold and score2 > 0.02:
        print(f"    Duplicate check (180°): dx={dx2_fullres:.0f}px "
              f"({100*dx2_fullres/w_a:.1f}%), score={score2:.3f} → duplicate")
        return True

    print(f"    Duplicate check: dx={dx_fullres:.0f}px, "
          f"dx_180={dx2_fullres:.0f}px → not duplicate (stitching)")
    return False


def _get_image_width(path: str) -> int:
    """Get image width via GDAL."""
    result = subprocess.run(["gdalinfo", "-json", path], capture_output=True, text=True)
    if result.returncode != 0:
        return 0
    info = json.loads(result.stdout)
    return info["size"][0]


def _stitch_if_needed(frames: list[str], eid: str, camera,
                      stitched_path: str, output_dir: str) -> str:
    """Stitch frames or handle single-frame seam detection.

    Returns the path to use for orientation detection (stitched or original).
    """
    if camera.needs_stitching and len(frames) > 1:
        print(f"\n  --- Stitch: {eid} ({len(frames)} frames) ---")
        # Prefer ASP's image_mosaic when available (handles KH-4/7/9 correctly)
        asp_result = stitch_with_asp(frames, stitched_path, camera.name, eid)
        if asp_result is None:
            print("  ASP not available, using built-in stitcher")
            stitch_frames(frames, stitched_path, output_dir, preserve_order=True)
        return stitched_path

    if len(frames) == 1 and camera.needs_stitching:
        print(f"\n  --- Sub-frame detection: {eid} ---")
        seams = detect_subframe_seams(frames[0])
        if seams:
            info_result = subprocess.run(
                ["gdalinfo", "-json", frames[0]],
                capture_output=True, text=True,
            )
            info = json.loads(info_result.stdout)
            img_w, img_h = info["size"]
            is_portrait = img_h > img_w

            sub_frames = split_at_seams(frames[0], seams,
                                        os.path.dirname(frames[0]),
                                        is_portrait=is_portrait)
            if len(sub_frames) > 1:
                print(f"\n  --- Re-stitch: {eid} ({len(sub_frames)} sub-frames) ---")
                stitch_frames(sub_frames, stitched_path, output_dir,
                              preserve_order=True)
                return stitched_path

    return frames[0]


def extract_stitch_georef_scene(scene, output_dir: str, file_map: dict, reference: str,
                                progress: dict, dry_run: bool = False,
                                cache_dir: str | None = None,
                                preserve_stitched: bool = False) -> bool:
    """Run the extract → stitch → georef cascade on a single scene.

    Preprocessing outputs (extracted, stitched, georef) go under *cache_dir*
    so they can be reused across test runs.

    Returns True on success, False on failure.
    """
    cd = cache_dir or output_dir
    eid = scene.entity_id
    camera = scene.camera_system

    # Check if already completed
    georef_path = paths.georef_path(cd, eid)
    if os.path.exists(georef_path):
        metadata = _merge_scene_metadata(
            cd,
            scene,
            georef_path=os.path.abspath(georef_path),
        )
        metadata = _ensure_scene_asp_ortho(
            scene,
            cd,
            reference,
            metadata=metadata,
            file_map=file_map,
        )
        print(f"  [skip] Already georeferenced: {eid}")
        progress["completed"][eid] = {"stage": "georef"}
        progress["failed"].pop(eid, None)
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
        entity_dir = extract_archive(file_path, cd, eid)
        frames = list_frames(entity_dir)

        if not frames:
            print(f"  WARNING: No frames found for {eid}")
            progress["failed"][eid] = "no_frames"
            return False

        # Step 4: Stitch in raw film coordinates (always horizontal)
        corners = scene.corners
        stitched_path = paths.stitched_path(cd, eid)
        input_for_orient = _stitch_if_needed(frames, eid, camera, stitched_path, cd)

        # Step 4b: Orientation detection — returns GCP corners, no pixel rotation
        print(f"\n  --- Orientation: {eid} ---")
        rotation, gcp_corners = detect_orientation(
            input_for_orient, corners, camera, reference_path=reference)

        if rotation != 0:
            print(f"  Orientation: {rotation} CW (via GCP assignment, no image rotation)")

        # Step 5: Georef — GDAL handles rotation via affine warp from GCP corners
        # KH-4 panoramic cameras need intermediate GCPs to model the non-uniform
        # GSD across the 70° scan arc.
        is_panoramic = camera.program == "CORONA"  # KH-4/4A/4B
        print(f"\n  --- Georef: {eid} ---")
        georef_with_corners(input_for_orient, georef_path, gcp_corners,
                            panoramic=is_panoramic)

        # Step 5b: Post-georef verification — auto-correct if 180° flip needed
        if reference and os.path.exists(reference):
            print(f"\n  --- Post-georef orientation check: {eid} ---")
            correction = verify_orientation_against_reference(georef_path, reference)
            if correction == 180:
                print(f"  Auto-correcting: re-georeferencing with 180° flipped corners")
                flipped_corners = swap_corners_180(gcp_corners)
                os.remove(georef_path)
                georef_with_corners(input_for_orient, georef_path, flipped_corners,
                                    panoramic=is_panoramic)
                gcp_corners = flipped_corners

        asp_ortho_path = None
        if reference and os.path.exists(reference):
            print(f"\n  --- ASP orthorectify: {eid} ---")
            try:
                asp_ortho_path = _maybe_generate_asp_ortho(
                    scene, cd, input_for_orient, gcp_corners, reference)
                if asp_ortho_path:
                    print(f"  ASP ortho ready: {os.path.basename(asp_ortho_path)}")
            except Exception as e:
                print(f"  WARNING: ASP orthorectification failed for {eid}: {e}")

        asp_camera_path = paths.ba_camera_path(input_for_orient)
        _merge_scene_metadata(
            cd,
            scene,
            gcp_corners={k: list(v) for k, v in gcp_corners.items()},
            georef_path=os.path.abspath(georef_path),
            stitched_path=os.path.abspath(input_for_orient),
            asp_camera_path=os.path.abspath(asp_camera_path) if os.path.exists(asp_camera_path) else None,
            asp_ortho_path=os.path.abspath(asp_ortho_path) if asp_ortho_path else None,
        )

        # Clean up stitched intermediates to save disk
        stitched_int = paths.stitched_path(cd, eid)
        if os.path.exists(stitched_int) and not preserve_stitched:
            os.remove(stitched_int)
            print(f"  Removed intermediate: {os.path.basename(stitched_int)}")

        progress["completed"][eid] = {"stage": "georef"}
        progress["failed"].pop(eid, None)
        return True

    except Exception as e:
        print(f"  ERROR processing {eid}: {e}")
        progress["failed"][eid] = str(e)
        return False


def _bbox_overlap_fraction(bbox_a: tuple, bbox_b: tuple) -> float:
    """Fraction of bbox_b covered by its intersection with bbox_a."""
    inter_w = max(0, min(bbox_a[2], bbox_b[2]) - max(bbox_a[0], bbox_b[0]))
    inter_h = max(0, min(bbox_a[3], bbox_b[3]) - max(bbox_a[1], bbox_b[1]))
    b_area = max(1e-10, (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]))
    return (inter_w * inter_h) / b_area


def _reference_bounds_wgs(reference_path: str) -> tuple:
    """Get reference image bounds in EPSG:4326."""
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(reference_path) as src:
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds)


def generate_manifest(scenes: list, output_dir: str, reference: str,
                      cache_dir: str | None = None,
                      device: str = "auto") -> str:
    """Generate a strip manifest JSON for auto-align.py.

    Each frame gets per-frame metadata priors (from USGS corners) and a
    reference_window constraining the global search to the expected overlap
    region.  Frames with no expected overlap with the reference are skipped.

    Georeferenced inputs are read from *cache_dir* (shared preprocessing).
    Alignment outputs go to *output_dir* (per-run).

    Returns path to the manifest file.
    """
    cd = cache_dir or output_dir
    os.makedirs(paths.manifests_dir(output_dir), exist_ok=True)

    # Read reference extent once for overlap checks and reference windowing
    ref_bbox = _reference_bounds_wgs(reference)
    margin_deg = 0.18  # ~20 km at equatorial latitudes

    jobs = []
    skipped = 0
    anchor_cache = ModelCache(get_torch_device(device))
    try:
        for scene in scenes:
            eid = scene.entity_id
            input_kind, input_path = _preferred_alignment_input_info(cd, eid)
            if not input_path:
                continue

            aligned_path = paths.aligned_path(output_dir, eid)
            if os.path.exists(aligned_path):
                print(f"  [skip] Already aligned: {eid}")
                continue

            # Check expected overlap between frame and reference
            metadata = _load_scene_metadata(cd, eid)
            corners = _corners_from_metadata(scene, metadata)
            frame_bbox = _bbox_from_corners(corners)
            overlap = _bbox_overlap_fraction(frame_bbox, ref_bbox)
            if overlap < 0.01:
                print(f"  [skip] {eid} — <1% expected overlap with reference "
                      f"({overlap*100:.1f}%), skipping alignment")
                skipped += 1
                continue

            # Coarse-align and crop wide strips to reference bbox before alignment.
            # This finds the actual content overlap (accounting for USGS corner
            # offset), shifts the image, and crops to the reference footprint +
            # margin so the alignment pipeline doesn't search 200km of ocean.
            input_for_align = input_path
            cropped_path = _alignment_crop_path(cd, eid, input_kind or "georef")
            if _path_is_stale(cropped_path, input_path):
                crop_result = coarse_align_and_crop(
                    input_path, reference, cropped_path,
                    target_bbox_wgs=frame_bbox,
                )
                if crop_result:
                    input_for_align = crop_result
            else:
                input_for_align = cropped_path

            _merge_scene_metadata(
                cd,
                scene,
                primary_input_kind=input_kind,
                primary_input_path=os.path.abspath(input_path),
                alignment_crop_path=os.path.abspath(input_for_align),
            )

            diag_dir = paths.scene_diagnostics_dir(output_dir, eid)
            qa_json = os.path.join(diag_dir, "qa.json")

            # Write per-frame metadata prior from USGS corners
            prior_data = {
                "source": f"usgs_corners_{eid}",
                "confidence": 0.35,
                "west": frame_bbox[0],
                "south": frame_bbox[1],
                "east": frame_bbox[2],
                "north": frame_bbox[3],
                "crs": "EPSG:4326",
                "center_lon": (frame_bbox[0] + frame_bbox[2]) / 2,
                "center_lat": (frame_bbox[1] + frame_bbox[3]) / 2,
                "corners": corners,
            }
            prior_path = paths.scene_prior_path(output_dir, eid)
            with open(prior_path, "w") as f:
                json.dump(prior_data, f, indent=2)

            # Compute per-frame reference window: intersection of frame bbox with
            # reference bbox, expanded by ~20 km margin, clamped to reference.
            win_left = max(ref_bbox[0], min(frame_bbox[0], ref_bbox[2]) - margin_deg)
            win_bottom = max(ref_bbox[1], min(frame_bbox[1], ref_bbox[3]) - margin_deg)
            win_right = min(ref_bbox[2], max(frame_bbox[2], ref_bbox[0]) + margin_deg)
            win_top = min(ref_bbox[3], max(frame_bbox[3], ref_bbox[1]) + margin_deg)
            window_str = f"{win_left},{win_bottom},{win_right},{win_top}"

            # Generate automatic anchor GCPs from coarse RoMa matching
            anchors_path = paths.scene_anchors_path(output_dir, eid)
            if not os.path.exists(anchors_path):
                try:
                    anchors_path = generate_auto_anchors(
                        input_for_align,
                        reference,
                        frame_bbox,
                        anchors_path,
                        model_cache=anchor_cache,
                        device_override=device,
                    )
                except Exception as e:
                    print(f"  [auto_anchors] Failed for {eid}: {e}")
                    anchors_path = None

            job_dict = {
                "input": os.path.abspath(input_for_align),
                "output": os.path.abspath(aligned_path),
                "metadata_priors": [os.path.abspath(prior_path)],
                "reference_window": window_str,
                "diagnostics_dir": os.path.abspath(diag_dir),
                "qa_json": os.path.abspath(qa_json),
            }
            if anchors_path:
                job_dict["anchors"] = os.path.abspath(anchors_path)

            jobs.append((overlap, job_dict))
    finally:
        anchor_cache.close()

    if skipped:
        print(f"  Skipped {skipped} scenes with no reference overlap")

    if not jobs:
        print("  No scenes to align (all done or no georef outputs)")
        return None

    # Sort by reference overlap (highest first) so the best-overlapping
    # frame aligns first and can anchor adjacent frames.
    jobs.sort(key=lambda item: item[0], reverse=True)
    sorted_jobs = [job for _, job in jobs]

    manifest = {
        "shared": {
            "reference": os.path.abspath(reference),
            "device": device,
            "global_search": True,
            "allow_abstain": True,
        },
        "jobs": sorted_jobs,
    }

    manifest_path = paths.alignment_manifest_path(output_dir)
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
                       dry_run: bool = False, reference: str = None):
    """Process pre-downloaded sub-frame TIFFs: stitch, georef, and optionally crop.

    Args:
        frames_dir: Directory containing {entity}_{frame}.tif files.
        output_dir: Output directory for stitched/georef/cropped results.
        crop_bbox: Optional (west, south, east, north) to clip output.
        dry_run: If True, just show what would be done.
        reference: Optional path to a georeferenced reference image for orientation.
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

        stitched_path = paths.stitched_path(output_dir, eid)

        if len(frames) == 1:
            # Single frame — check for sub-frame seams
            seams = detect_subframe_seams(frames[0])
            if seams:
                info_result = subprocess.run(
                    ["gdalinfo", "-json", frames[0]],
                    capture_output=True, text=True,
                )
                info_data = json.loads(info_result.stdout)
                img_w, img_h = info_data["size"]
                is_portrait = img_h > img_w

                sub_frames = split_at_seams(frames[0], seams,
                                            os.path.dirname(frames[0]),
                                            is_portrait=is_portrait)
                if len(sub_frames) > 1:
                    print(f"  Re-stitching {len(sub_frames)} sub-frames")
                    stitch_frames(sub_frames, stitched_path, output_dir,
                                  preserve_order=True)
                else:
                    import shutil
                    shutil.copy2(frames[0], stitched_path)
                    print(f"  Single frame, copied")
            else:
                import shutil
                shutil.copy2(frames[0], stitched_path)
                print(f"  Single frame, copied")
        else:
            # Multi-frame strip: alphabetical ordering is reliable for all camera types
            keep_order = True
            stitch_frames(frames, stitched_path, output_dir,
                          preserve_order=keep_order)

        stitched_paths[eid] = stitched_path

    # Step 3b: Orientation detection and correction
    print("\n" + "=" * 60)
    print("Step 3b: Orientation detection")
    print("=" * 60)

    for eid in entity_ids:
        stitched_path = stitched_paths[eid]
        corners = corners_map[eid]

        rotation, gcp_corners = detect_orientation(
            stitched_path, corners, camera, reference_path=reference
        )
        corners_map[eid] = gcp_corners  # GCP mapping for georef (no pixel rotation)
        if rotation != 0:
            print(f"  Orientation: {rotation} CW (via GCP assignment, no image rotation)")

    # Step 4: Georeference with corner coordinates
    print("\n" + "=" * 60)
    print("Step 4: Georeference with M2M corner coordinates")
    print("=" * 60)

    georef_paths = {}
    for eid in entity_ids:
        stitched_path = stitched_paths[eid]
        corners = corners_map[eid]
        georef_path = paths.georef_path(output_dir, eid)

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
            cropped_path = paths.cropped_path(output_dir, eid)

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
        crop_root = paths.cropped_dir(output_dir)
        mosaic_inputs = sorted(
            os.path.join(crop_root, f)
            for f in os.listdir(crop_root)
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
        crop_root = paths.cropped_dir(output_dir)
        cropped_count = len([f for f in os.listdir(crop_root) if f.endswith(".tif")])
        print(f"  Cropped outputs: {cropped_count}")


def _parse_boundary(boundary_str: str) -> tuple:
    """Parse a boundary argument into (west, south, east, north) bbox.

    Accepts either a GeoJSON file path or a WEST,SOUTH,EAST,NORTH string.
    """
    # Try as GeoJSON file
    if os.path.isfile(boundary_str):
        with open(boundary_str) as f:
            geojson = json.load(f)
        # Extract coordinates from first feature or top-level geometry
        geom = geojson
        if "features" in geom:
            geom = geom["features"][0]["geometry"]
        elif "geometry" in geom:
            geom = geom["geometry"]
        coords = geom["coordinates"]
        # Flatten nested coordinate arrays
        if isinstance(coords[0][0], list):
            flat = [pt for ring in coords for pt in ring]
        else:
            flat = coords
        lons = [pt[0] for pt in flat]
        lats = [pt[1] for pt in flat]
        return (min(lons), min(lats), max(lons), max(lats))

    # Try as comma-separated bbox
    parts = boundary_str.split(",")
    if len(parts) == 4:
        return tuple(float(x) for x in parts)

    raise ValueError(f"Cannot parse boundary: {boundary_str} "
                     f"(expected GeoJSON file or WEST,SOUTH,EAST,NORTH)")


def evaluate_mosaic_quality(mosaic_path: str, reference_path: str,
                            target_bbox: tuple = None) -> dict:
    """Run alignment QA on final mosaic vs reference.

    Returns dict with QA metrics and coverage fraction.
    """
    import rasterio
    from align.geo import compute_overlap_or_none, get_metric_crs
    from align.qa import evaluate_alignment_quality_paths

    result = {}

    with rasterio.open(mosaic_path) as src_m, rasterio.open(reference_path) as src_r:
        work_crs = get_metric_crs(src_m, src_r)
        overlap = compute_overlap_or_none(src_m, src_r, work_crs)

    if overlap is None:
        return {"error": "no_overlap"}

    metrics = evaluate_alignment_quality_paths(
        mosaic_path, reference_path, overlap, work_crs)
    if metrics:
        result["qa"] = metrics

    # Compute actual coverage: fraction of target bbox with valid mosaic pixels
    if target_bbox:
        from osgeo import gdal
        gdal.UseExceptions()
        ds = gdal.Open(mosaic_path)
        if ds:
            gt = ds.GetGeoTransform()
            w, h = ds.RasterXSize, ds.RasterYSize
            # Read alpha band at low res
            alpha_band = ds.GetRasterBand(ds.RasterCount)
            scale = max(1, max(w, h) // 1000)
            alpha = alpha_band.ReadAsArray(
                buf_xsize=w // scale, buf_ysize=h // scale)
            valid_frac = float((alpha > 0).sum()) / max(alpha.size, 1)
            result["actual_coverage"] = round(valid_frac, 3)
            ds = None

    return result


# ---------------------------------------------------------------------------
# Pipeline context + stage functions
# ---------------------------------------------------------------------------

@dataclass
class PipelineContext:
    """Shared state carried across stage functions.

    ``cache_dir`` holds shared preprocessing outputs (downloads, extracted,
    stitched, georef, ortho, ba); ``output_dir`` holds per-run outputs
    (aligned, mosaic, manifests, diagnostics, match_files, reference).
    """

    args: argparse.Namespace
    output_dir: str
    cache_dir: str
    progress: dict
    reference: Optional[str] = None            # composite when built, primary otherwise
    primary_reference: Optional[str] = None    # always the original reference (used for alignment)
    target_bbox: Optional[tuple] = None
    selection_meta: Optional[dict] = None
    scenes: list = field(default_factory=list)
    downloadable: list = field(default_factory=list)
    strips: list = field(default_factory=list)
    file_map: dict = field(default_factory=dict)
    success_count: int = 0
    fail_count: int = 0
    mosaics: list = field(default_factory=list)
    mosaic_qa: dict = field(default_factory=dict)
    crop_bbox: Optional[tuple] = None


def _print_stage_banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automated pipeline for USGS declassified satellite imagery"
    )
    parser.add_argument("--csv", nargs="+", required=False,
                        help="Path(s) to USGS EarthExplorer CSV catalog files")
    parser.add_argument("--reference", "-r", default=None,
                        help="Path to a correctly-aligned reference GeoTIFF")
    parser.add_argument("--auto-reference", action="store_true",
                        help="Auto-fetch a Sentinel-2 reference image")
    parser.add_argument("--output-dir", "-o", default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--cache-dir", default=None,
                        help="Shared preprocessing directory for downloads/extracted/georef. "
                             "Reused across runs. Defaults to --output-dir.")
    parser.add_argument("--entities", nargs="+", default=None,
                        help="Process only these specific entity IDs")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip USGS download (use existing files in downloads/)")
    parser.add_argument("--skip-align", action="store_true",
                        help="Skip alignment step (georef only)")
    parser.add_argument("--skip-mosaic", action="store_true",
                        help="Skip mosaic assembly step")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without processing")
    parser.add_argument("--resume", default=None,
                        help="Resume from an existing output directory (reads progress.json)")
    parser.add_argument("--frames-dir", default=None,
                        help="Directory containing pre-downloaded {entity}_{frame}.tif files "
                             "(bypasses catalog/download)")
    parser.add_argument("--crop-bbox", default=None,
                        help="Crop output to WEST,SOUTH,EAST,NORTH bbox in decimal degrees "
                             "(e.g. 50.15,25.55,50.90,26.40)")
    parser.add_argument("--boundary", default=None,
                        help="GeoJSON file or WEST,SOUTH,EAST,NORTH bbox for automatic scene "
                             "selection by coverage")
    parser.add_argument("--prefer-camera", default=None,
                        help="Preferred camera designation (A=Aft, F=Forward) for scene selection")
    parser.add_argument("--bundle-adjust", action="store_true",
                        help="Enable strip bundle adjustment with RoMa tie points "
                             "(experimental, requires ASP + --experimental)")
    parser.add_argument("--experimental", action="store_true",
                        help="Opt in to experimental features (currently: --bundle-adjust). "
                             "In-progress research threads are gated behind this flag so "
                             "production runs can't stumble into unvalidated code paths.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                        help="Torch device override for auto-anchor generation and alignment")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete intermediate files (extracted/, georef/) after mosaic")
    return parser


def _parse_crop_bbox(arg: str | None) -> tuple | None:
    if not arg:
        return None
    parts = arg.split(",")
    if len(parts) != 4:
        return None
    return tuple(float(x) for x in parts)


def _resume_from_prior_run(args: argparse.Namespace) -> None:
    """Populate ``args`` with CSVs and reference recovered from a prior output dir."""
    args.output_dir = args.resume
    manifest_path = paths.alignment_manifest_path(args.output_dir)
    if os.path.exists(manifest_path) and not args.reference:
        with open(manifest_path) as f:
            prev = json.load(f)
        args.reference = prev.get("shared", {}).get("reference")
    if not args.csv:
        csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "available")
        if os.path.exists(csv_dir):
            args.csv = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")]


def _init_context(args: argparse.Namespace) -> PipelineContext:
    if args.resume:
        _resume_from_prior_run(args)
    output_dir = args.output_dir
    cache_dir = args.cache_dir or output_dir
    progress = load_progress(cache_dir)
    ensure_pipeline_dirs(output_dir, cache_dir)
    return PipelineContext(
        args=args,
        output_dir=output_dir,
        cache_dir=cache_dir,
        progress=progress,
        crop_bbox=_parse_crop_bbox(args.crop_bbox),
    )


def stage_parse_catalog(ctx: PipelineContext) -> None:
    """Parse USGS CSV catalogs, apply filters, pick the best mission+date group."""
    _print_stage_banner("Stage 1: Parse catalog CSVs")
    args = ctx.args

    ctx.scenes = parse_csvs(args.csv)

    if args.boundary:
        ctx.target_bbox = _parse_boundary(args.boundary)
        print(f"  Target bbox: {ctx.target_bbox}")

    if args.entities:
        ctx.scenes = filter_scenes(ctx.scenes, entity_ids=args.entities, download_only=False)
        print(f"  Filtered to {len(ctx.scenes)} scenes matching --entities")

    ctx.downloadable = [s for s in ctx.scenes if s.download_available]
    print(f"  Total: {len(ctx.scenes)} scenes, {len(ctx.downloadable)} downloadable")

    if ctx.target_bbox and not args.entities:
        ctx.downloadable, _coverage, ctx.selection_meta = select_best_mission_coverage(
            ctx.downloadable, ctx.target_bbox, prefer_camera=args.prefer_camera,
        )
        if not ctx.downloadable:
            raise SystemExit("  ERROR: No scenes provide coverage of the target bbox")
        if ctx.crop_bbox is None:
            ctx.crop_bbox = tuple(ctx.target_bbox)

    ctx.strips = group_into_strips(ctx.downloadable)
    print(f"  Grouped into {len(ctx.strips)} strips:")
    for strip in ctx.strips:
        print(f"    {strip.camera_system.name} {strip.date} {strip.mission} "
              f"cam={strip.camera_designation} ({len(strip.scenes)} scenes)")


def stage_fetch_reference(ctx: PipelineContext) -> None:
    """Resolve ``ctx.reference`` from --reference or --auto-reference."""
    args = ctx.args
    ctx.reference = args.reference
    if ctx.reference or not args.auto_reference:
        return

    _print_stage_banner("Auto-reference: Fetching Sentinel-2 composite")
    all_lats = []
    all_lons = []
    for scene in ctx.downloadable:
        for corner in scene.corners.values():
            all_lats.append(corner[0])
            all_lons.append(corner[1])
    bbox = (min(all_lons), min(all_lats), max(all_lons), max(all_lats))
    ref_path = os.path.join(ctx.output_dir, "reference", "sentinel2_reference.tif")
    ctx.reference = fetch_sentinel2_reference(bbox, ref_path)
    args.reference = ctx.reference


def stage_download_imagery(ctx: PipelineContext) -> None:
    _print_stage_banner("Stage 2: Download imagery")
    ctx.file_map = download_scenes(
        ctx.downloadable, ctx.cache_dir, skip_download=ctx.args.skip_download,
    )
    found = sum(1 for v in ctx.file_map.values() if v)
    print(f"  Files available: {found}/{len(ctx.downloadable)}")
    save_progress(ctx.cache_dir, ctx.progress)


def stage_preprocess_scenes(ctx: PipelineContext) -> None:
    """Run extract → stitch → georef on every downloadable scene."""
    _print_stage_banner("Stage 3-5: Extract, Stitch, Georef")
    preserve_stitched = bool(ctx.args.bundle_adjust)

    for scene in ctx.downloadable:
        print(f"\n{'─' * 50}")
        print(f"Processing: {scene.entity_id} ({scene.camera_system.name}, {scene.acquisition_date})")
        print(f"{'─' * 50}")
        ok = extract_stitch_georef_scene(
            scene, ctx.output_dir, ctx.file_map, ctx.reference, ctx.progress,
            cache_dir=ctx.cache_dir, preserve_stitched=preserve_stitched,
        )
        if ok:
            ctx.success_count += 1
        else:
            ctx.fail_count += 1
        save_progress(ctx.cache_dir, ctx.progress)

    print(f"\n  Georef complete: {ctx.success_count} succeeded, {ctx.fail_count} failed")


def stage_build_composite_reference(ctx: PipelineContext) -> None:
    """Build a Sentinel-2 fill for mosaic QA; keep primary reference for alignment."""
    ctx.primary_reference = ctx.reference
    if not (ctx.target_bbox and ctx.reference and not ctx.args.skip_align):
        return

    composite_dir = os.path.join(ctx.output_dir, "reference")
    os.makedirs(composite_dir, exist_ok=True)
    composite_path = os.path.join(composite_dir, "composite_reference.tif")
    try:
        ctx.reference = build_composite_reference(ctx.reference, ctx.target_bbox, composite_path)
    except Exception as e:
        print(f"  WARNING: Composite reference failed: {e}")
        print(f"  Continuing with primary reference only")


def _collect_strip_frames(strip, cache_dir: str):
    """Return (scenes, frames, corners) for every strip member that has a stitched frame on disk."""
    strip_scenes: list = []
    strip_frames: list = []
    strip_corners: list = []
    for scene in strip.scenes:
        metadata = _load_scene_metadata(cache_dir, scene.entity_id)
        if not metadata:
            continue
        stitched = metadata.get("stitched_path")
        corners = metadata.get("gcp_corners")
        if stitched and os.path.exists(stitched) and corners:
            strip_scenes.append(scene)
            strip_frames.append(stitched)
            strip_corners.append(corners)
    return strip_scenes, strip_frames, strip_corners


def _record_ba_ortho_metadata(scene, cache_dir: str, frame_path: str, remapped: str) -> None:
    metadata = _load_scene_metadata(cache_dir, scene.entity_id) or {}
    metadata["asp_ortho_path"] = os.path.abspath(remapped)
    metadata.setdefault("georef_path", paths.georef_path(cache_dir, scene.entity_id))
    metadata.setdefault("stitched_path", os.path.abspath(frame_path))
    metadata.setdefault("entity_id", scene.entity_id)
    metadata.setdefault("camera_designation", _camera_designation(scene))
    metadata.setdefault("profile", _profile_name_for_scene(scene))
    metadata.setdefault("gcp_corners", {k: list(v) for k, v in scene.corners.items()})
    _save_scene_metadata(cache_dir, scene.entity_id, metadata)


def _bundle_adjust_strip(strip, ctx: PipelineContext) -> None:
    """Run RoMa → ASP bundle_adjust → mapproject on one strip."""
    strip_scenes, strip_frames, strip_corners = _collect_strip_frames(strip, ctx.cache_dir)
    if len(strip_frames) < 2:
        return

    match_prefix = generate_strip_matches(
        strip_frames, strip_corners, paths.match_files_dir(ctx.output_dir),
        match_prefix=f"roma_{strip.mission}_{strip.date}".replace("/", "_"),
    )
    if not match_prefix:
        return

    strip._match_prefix = match_prefix
    print(f"  Strip {strip.mission} {strip.date}: match prefix → {match_prefix}")

    camera_params = _camera_params_for_scene(strip_scenes[0])
    if camera_params is None:
        return
    profile = load_profile(_profile_name_for_scene(strip_scenes[0]))
    solve_intrinsics = bool(getattr(profile.camera, "bundle_adjust_solve_intrinsics", False))
    bbox = strip.bbox
    dem_path = fetch_and_prepare_dem(
        west=bbox[0] - 0.1, south=bbox[1] - 0.1,
        east=bbox[2] + 0.1, north=bbox[3] + 0.1,
    )
    strip_key = f"{strip.mission}_{strip.date}_{strip.camera_designation}".replace("/", "_")
    adjusted = run_strip_bundle_adjustment(
        strip_frames, camera_params, strip_corners,
        dem_path=dem_path,
        output_dir=paths.bundle_adjust_dir(ctx.cache_dir, strip_key),
        match_prefix=match_prefix,
        solve_intrinsics=solve_intrinsics,
    )
    if not adjusted:
        return

    for scene, frame_path, camera_path in zip(strip_scenes, strip_frames, adjusted):
        remapped = mapproject_image(
            frame_path, camera_path,
            dem_path=dem_path,
            output_path=paths.ortho_path(ctx.cache_dir, scene.entity_id, bundle_adjusted=True),
            resolution=_reference_resolution(ctx.primary_reference),
            t_srs="EPSG:3857",
        )
        if remapped:
            _record_ba_ortho_metadata(scene, ctx.cache_dir, frame_path, remapped)


def stage_bundle_adjust_strips(ctx: PipelineContext) -> None:
    """Experimental: inter-frame RoMa tie points + ASP bundle_adjust (2OC P2)."""
    _print_stage_banner("Stage 5b: Inter-frame match generation (RoMa → ASP)")
    for strip in ctx.strips:
        if len(strip.scenes) < 2:
            continue
        try:
            _bundle_adjust_strip(strip, ctx)
        except Exception as e:
            print(f"  WARNING: bundle adjustment failed for strip "
                  f"{strip.mission} {strip.date}: {e}")


def stage_align_scenes(ctx: PipelineContext) -> None:
    _print_stage_banner("Stage 6: Alignment")
    manifest_path = generate_manifest(
        ctx.downloadable, ctx.output_dir, ctx.primary_reference,
        cache_dir=ctx.cache_dir, device=ctx.args.device,
    )
    run_alignment(manifest_path)


def stage_assemble_mosaics(ctx: PipelineContext) -> None:
    _print_stage_banner("Stage 7: Mosaic assembly")
    ctx.mosaics = build_all_mosaics(
        ctx.downloadable,
        paths.aligned_dir(ctx.output_dir),
        os.path.join(ctx.output_dir, "mosaic"),
        diagnostics_dir=paths.diagnostics_dir(ctx.output_dir),
    )
    print(f"  Built {len(ctx.mosaics)} mosaics")


def _gdalwarp_crop_to_bbox(src: str, dst: str, bbox: tuple) -> bool:
    west, south, east, north = bbox
    cmd = [
        "gdalwarp",
        "-te", str(west), str(south), str(east), str(north),
        "-te_srs", "EPSG:4326",
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=IF_SAFER",
        src, dst,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True
    print(f"  WARNING: Crop failed: {result.stderr}")
    return False


def stage_crop_mosaics(ctx: PipelineContext) -> None:
    if not (ctx.crop_bbox and ctx.mosaics):
        return
    west, south, east, north = ctx.crop_bbox
    _print_stage_banner(f"Stage 8: Crop mosaic to bbox "
                        f"({west:.2f},{south:.2f},{east:.2f},{north:.2f})")
    cropped: list = []
    for mosaic_path in ctx.mosaics:
        cropped_path = mosaic_path.replace(".tif", "_cropped.tif")
        if os.path.exists(cropped_path):
            print(f"  [skip] Already cropped: {os.path.basename(cropped_path)}")
            cropped.append(cropped_path)
            continue
        if _gdalwarp_crop_to_bbox(mosaic_path, cropped_path, ctx.crop_bbox):
            print(f"  Cropped: {os.path.basename(cropped_path)}")
            cropped.append(cropped_path)
    ctx.mosaics = cropped


def _evaluate_and_log_mosaic(mosaic_path: str, ctx: PipelineContext) -> dict:
    print(f"  Evaluating: {os.path.basename(mosaic_path)}")
    try:
        qa_result = evaluate_mosaic_quality(
            mosaic_path, ctx.reference, target_bbox=ctx.target_bbox,
        )
    except Exception as e:
        print(f"    ERROR: QA failed: {e}")
        return {"error": str(e)}

    if "qa" in qa_result:
        qa = qa_result["qa"]
        print(f"    score={qa.get('score', '?')}, "
              f"grid_score={qa.get('grid_score', '?')}, "
              f"patch_med={qa.get('patch_med', '?')}")
    if "actual_coverage" in qa_result:
        print(f"    actual_coverage={qa_result['actual_coverage'] * 100:.1f}%")
    if "error" in qa_result:
        print(f"    ERROR: {qa_result['error']}")

    qa_json_path = mosaic_path.replace(".tif", "_qa.json")
    with open(qa_json_path, "w") as f:
        json.dump(qa_result, f, indent=2, default=str)
    print(f"    Written: {os.path.basename(qa_json_path)}")
    return qa_result


def stage_score_mosaics(ctx: PipelineContext) -> None:
    if not (ctx.mosaics and ctx.reference):
        return
    _print_stage_banner("Stage 9: Mosaic QA")
    for mosaic_path in ctx.mosaics:
        ctx.mosaic_qa[mosaic_path] = _evaluate_and_log_mosaic(mosaic_path, ctx)


def stage_cleanup_intermediates(ctx: PipelineContext) -> None:
    # Only clean preprocessing dirs when they're co-located with output
    # (not when using a shared --cache-dir that other runs depend on).
    if ctx.cache_dir != ctx.output_dir:
        return
    _print_stage_banner("Cleanup: Removing intermediate files")
    for subdir in ("extracted", "georef"):
        dir_path = os.path.join(ctx.output_dir, subdir)
        if not os.path.exists(dir_path):
            continue
        size_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fnames in os.walk(dir_path)
            for f in fnames
        ) / (1024 * 1024)
        shutil.rmtree(dir_path)
        print(f"  Removed {subdir}/ ({size_mb:.0f} MB)")


def _print_pipeline_summary(ctx: PipelineContext) -> None:
    _print_stage_banner("Pipeline complete")
    print(f"  Output directory: {ctx.output_dir}")
    print(f"  Scenes processed: {ctx.success_count}/{len(ctx.downloadable)}")
    if ctx.selection_meta:
        print(f"  Mission: {ctx.selection_meta.get('mission')} "
              f"({ctx.selection_meta.get('date')}) "
              f"cam={ctx.selection_meta.get('camera_designation')}")
        print(f"  Predicted coverage: {ctx.selection_meta.get('predicted_coverage', 0) * 100:.1f}%")
    for mosaic_path, qa in ctx.mosaic_qa.items():
        if "qa" in qa:
            print(f"  Mosaic QA score: {qa['qa'].get('score', '?')}")
        if "actual_coverage" in qa:
            print(f"  Actual coverage: {qa['actual_coverage'] * 100:.1f}%")
    if ctx.progress["failed"]:
        print(f"  Failed:")
        for eid, err in ctx.progress["failed"].items():
            print(f"    {eid}: {err}")


def _run_frames_dir_mode(args: argparse.Namespace) -> None:
    crop_bbox = _parse_crop_bbox(args.crop_bbox)
    if args.crop_bbox and crop_bbox is None:
        raise SystemExit("--crop-bbox must be WEST,SOUTH,EAST,NORTH (4 comma-separated values)")
    process_frames_dir(
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        crop_bbox=crop_bbox,
        dry_run=args.dry_run,
        reference=args.reference,
    )


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.bundle_adjust and not args.experimental:
        parser.error(
            "--bundle-adjust is an experimental feature (in-progress research, "
            "downstream camera-model improvement not yet quantified). "
            "Pass --experimental alongside --bundle-adjust to acknowledge."
        )
    if args.frames_dir:
        return  # frames-dir mode has its own arg shape
    if not args.csv and not args.resume:
        parser.error("--csv is required (or use --resume with an existing output directory)")
    if not args.reference and not args.auto_reference and not args.skip_align:
        parser.error("--reference or --auto-reference required (or use --skip-align)")


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    if args.frames_dir:
        return _run_frames_dir_mode(args)

    ctx = _init_context(args)
    stage_parse_catalog(ctx)
    stage_fetch_reference(ctx)
    if args.dry_run:
        print("\n[dry-run] Would process the above scenes. Exiting.")
        return
    stage_download_imagery(ctx)
    stage_preprocess_scenes(ctx)
    stage_build_composite_reference(ctx)
    if args.bundle_adjust and not args.skip_align:
        stage_bundle_adjust_strips(ctx)
    if not args.skip_align and ctx.primary_reference:
        stage_align_scenes(ctx)
    if not args.skip_mosaic and not args.skip_align and ctx.reference:
        stage_assemble_mosaics(ctx)
    stage_crop_mosaics(ctx)
    stage_score_mosaics(ctx)
    if args.cleanup and ctx.mosaics:
        stage_cleanup_intermediates(ctx)
    _print_pipeline_summary(ctx)


if __name__ == "__main__":
    main()
