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
from preprocess.camera_model import generate_camera, interpolate_camera_pose, mapproject_image
from preprocess.mission_altitude import altitude_m_at, parse_entity_id
from preprocess.dem import fetch_and_prepare_dem
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


def _parse_acquisition_date(raw: str):
    """Convert USGS ``acquisition_date`` ('YYYY/MM/DD') to :class:`datetime.date`.

    Returns ``None`` when the string is missing or unparseable; downstream
    code then falls back to catalog-nominal or 170 km.
    """
    from datetime import date, datetime
    if not raw:
        return None
    raw = raw.strip()
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _altitude_seed_for_scene(scene) -> float | None:
    """Best-effort per-mission altitude seed for cam_gen Zs0.

    Returns ``None`` when mission catalog / TLE are unavailable so the
    caller falls through to the 170 km nominal baked into ``cam_gen``.
    """
    mref = parse_entity_id(scene.entity_id) if scene.entity_id else None
    if mref is None:
        return None
    acq = _parse_acquisition_date(scene.acquisition_date)
    ctr = getattr(scene, "center", None)
    lat = lon = None
    if isinstance(ctr, (list, tuple)) and len(ctr) >= 2:
        lat = float(ctr[0])
        lon = float(ctr[1])
    res = altitude_m_at(mref.mission_id, acq, lat, lon)
    return float(res.altitude_m) if res is not None else None


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
        # Outcome of the most recent _coarse_align_ortho_to_sidecar call.
        # "ok" → sidecar written; "abstained" → coarse_align_and_crop
        # could not pin position; None → never attempted (no reference).
        # Consumed by generate_manifest's hard-fail skip on profiles
        # whose USGS corners are flagged unreliable.
        "coarse_align_status": None,
        # Reference-sensitive cache identity for the georef/ortho state.
        # Older caches lack this key and are treated as untrusted for
        # unreliable-corner profiles until preprocessing refreshes them.
        "georef_cache_key": None,
        "georef_cache_reuse_status": None,
        "georef_cache_reuse_reason": None,
    }


def _merge_scene_metadata(cache_dir: str, scene, **updates) -> dict:
    metadata = _load_scene_metadata(cache_dir, scene.entity_id) or _default_scene_metadata(scene, cache_dir)
    metadata.update(updates)
    _save_scene_metadata(cache_dir, scene.entity_id, metadata)
    return metadata


def _reference_cache_identity(reference: str | None) -> dict | None:
    if not reference or not os.path.exists(reference):
        return None
    try:
        st = os.stat(reference)
    except OSError:
        return None
    return {
        "basename": os.path.basename(reference),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _georef_output_identity(path: str | None) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return {
        "basename": os.path.basename(path),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _scene_corner_cache_identity(scene) -> dict:
    corners = getattr(scene, "corners", {}) or {}
    return {
        str(k).upper(): [round(float(v[0]), 8), round(float(v[1]), 8)]
        for k, v in sorted(corners.items())
        if isinstance(v, (list, tuple)) and len(v) >= 2
    }


def _build_georef_cache_key(scene, reference: str | None,
                            georef_path: str | None) -> dict:
    return {
        "schema_version": 1,
        "entity_id": scene.entity_id,
        "profile": _profile_name_for_scene(scene),
        "camera_designation": _camera_designation(scene),
        "corners": _scene_corner_cache_identity(scene),
        "reference": _reference_cache_identity(reference),
        "georef": _georef_output_identity(georef_path),
    }


def _georef_cache_reject(cache_dir: str, scene, georef_path: str,
                         reason: str, cache_key: dict | None = None) -> bool:
    """Mark a georef cache miss and remove the stale completion sentinel."""
    print(f"  [cache] Rejecting georef cache for {scene.entity_id}: {reason}")
    metadata = _load_scene_metadata(cache_dir, scene.entity_id)
    if metadata is not None:
        metadata["georef_cache_key"] = cache_key
        metadata["georef_cache_reuse_status"] = "rejected"
        metadata["georef_cache_reuse_reason"] = reason
        _save_scene_metadata(cache_dir, scene.entity_id, metadata)
    try:
        if os.path.exists(georef_path):
            os.remove(georef_path)
    except OSError as exc:
        print(f"  [cache] WARNING: could not remove stale georef {georef_path}: {exc}")
    return False


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
    """Map a scene to its profile YAML stem.

    DS1 entities (CORONA) split by mission-ID: DS1001-DS1052 → kh4a
    (Itek J-1, oscillating scanner, scan-head-translation IMC), DS1101-
    DS1117 → kh4b (Itek J-3, rotating scanner, lens-rotation IMC). The
    J-1/J-3 difference in IMC direction + scan kinematics is real, even
    though CoSP (Ghuffar 2022) absorbs it in pose + focal-length free
    parameters. Mission-ID ranges sourced from
    `preprocess/mission_altitude.py::_series_from_mission_id`.
    """
    prefix = scene.camera_system.entity_prefix
    if prefix == "DS1":
        # Parse mission ID from entity_id (format: DS<4-digit mission>-<frame>).
        from preprocess.mission_altitude import parse_entity_id
        ref = parse_entity_id(scene.entity_id)
        if ref is not None:
            if ref.system == "KH-4A":
                return "kh4a"
            if ref.system == "KH-4B":
                return "kh4b"
            # KH-4 (missions 9001-9099) — treat as KH-4A-adjacent. No
            # separate profile; route to kh4a for now since the Itek lens
            # is closer to J-1 than J-3, and no catalog rows ever exist
            # (the original KH-4 is pre-1963 and unscanned).
            if ref.system == "KH-4":
                return "kh4a"
        # Fallback: unknown DS1 mission → kh4b (preserves legacy behaviour
        # for catalog rows that predate parse_entity_id's coverage).
        return "kh4b"
    if prefix == "DZB":
        # DZB4xxx → KH-7 (GAMBIT-1 strip camera).
        # DZB12xx → KH-9 Mapping Camera (Hexagon frame camera, Declass-II).
        # See `preprocess/mission_altitude.py::_series_from_mission_id` for ranges.
        from preprocess.mission_altitude import parse_entity_id
        ref = parse_entity_id(scene.entity_id)
        if ref is not None:
            if ref.system == "KH-7":
                return "kh7"
            if ref.system == "KH-9":
                return "kh9_mc"
        # Unknown DZB mission → KH-7 legacy behaviour.
        return "kh7"
    if prefix == "D3C":
        return "kh9"
    raise ValueError(f"Unsupported camera system for {scene.entity_id}")


def _camera_params_for_scene(scene) -> dict | None:
    """Return a camera-params dict suitable for `preprocess.camera_model.generate_camera`.

    Returns None when the profile declares no known geometry (e.g. the KH-8
    stub). Panoramic (KH-4, KH-9 PC), pinhole / frame (KH-9 MC), and strip /
    linescan (KH-7) all return populated dicts; ``generate_camera`` dispatches
    on the ``type`` field to pick the ASP cam_gen model.
    """
    profile = load_profile(_profile_name_for_scene(scene))
    if not profile.camera.is_known_geometry:
        return None
    params = copy.deepcopy(profile.camera.to_dict())
    if profile.camera.is_panoramic:
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


def _native_ortho_resolution_m(scene, camera_params: dict | None) -> float | None:
    """Compute the scene's native GSD (ground-sampling distance) in metres.

    Returns ``altitude × pixel_pitch / focal_length`` with altitude from
    (in precedence order): TLE-derived per-scene altitude, profile
    ``nominal_altitude_km``, or None when unavailable. The native GSD
    is what the film actually resolves — orthos rendered at this pitch
    preserve full detail (the user complaint that "all outputs are
    downscaled from the original" comes from mapproject inheriting the
    basemap's coarser pixel size).
    """
    if camera_params is None:
        return None
    f = camera_params.get("focal_length")
    p = camera_params.get("pixel_pitch")
    if not f or not p or f <= 0 or p <= 0:
        return None
    # Prefer per-scene altitude from TLE / catalog.
    alt = _altitude_seed_for_scene(scene) if scene is not None else None
    if alt is None or alt <= 0:
        nom_km = camera_params.get("nominal_altitude_km")
        if nom_km:
            alt = float(nom_km) * 1000.0
    if alt is None or alt <= 0:
        return None
    return float(alt) * float(p) / float(f)


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


def _ortho_has_content(ortho_path: str, *,
                       min_valid_fraction: float = 0.001) -> bool:
    """Return True iff the ortho TIFF at ``ortho_path`` has at least
    ``min_valid_fraction`` of valid (>0) pixels. Reads a decimated
    overview to keep the check cheap (under ~50 ms even on a 3 GB raster).

    Used as a defensive gate after ``mapproject_image`` because
    ASP/mapproject can silently emit an all-nodata sparse-tiled raster
    when projection geometry fails (camera entirely off-DEM, numerical
    blow-up, killed mid-write). Detected case: KH-4B DS1104-1057DA024
    in v13/v14/v17/v19 — 49 MB output with 0 valid pixels.
    """
    try:
        import rasterio
    except Exception:
        return True   # if rasterio is unavailable assume content
    if not os.path.exists(ortho_path):
        return False
    try:
        with rasterio.open(ortho_path) as src:
            decim = 256
            arr = src.read(
                1,
                out_shape=(max(1, src.height // decim),
                           max(1, src.width // decim)),
            )
        valid = float((arr > 0).mean())
        return valid >= min_valid_fraction
    except Exception:
        # Treat read failures as broken — better safe than letting a
        # corrupt ortho through.
        return False


def _primary_input_update(georef_path: str | None, asp_ortho_path: str | None) -> dict:
    if asp_ortho_path and os.path.exists(asp_ortho_path):
        return {
            "primary_input_kind": "asp_ortho",
            "primary_input_path": os.path.abspath(asp_ortho_path),
        }
    if georef_path and os.path.exists(georef_path):
        return {
            "primary_input_kind": "georef",
            "primary_input_path": os.path.abspath(georef_path),
        }
    return {
        "primary_input_kind": None,
        "primary_input_path": None,
    }


def _alignment_crop_path(cache_dir: str, entity_id: str, input_kind: str) -> str:
    crop_dir = "ortho" if input_kind == "asp_ortho" else "georef"
    safe_kind = re.sub(r"[^a-z0-9_]+", "_", input_kind.lower())
    return os.path.join(cache_dir, crop_dir, f"{entity_id}_{safe_kind}_cropped.tif")


def _coarse_shift_from_geotransforms(source_path: str, shifted_path: str) -> tuple[float, float] | None:
    """Return the sidecar geotransform delta in approximate metres."""
    try:
        import rasterio
        with rasterio.open(source_path) as src, rasterio.open(shifted_path) as shifted:
            dx = float(shifted.transform.c - src.transform.c)
            dy = float(shifted.transform.f - src.transform.f)
            crs = shifted.crs or src.crs
            if crs is not None and crs.is_geographic:
                dx *= 111000.0
                dy *= 111000.0
            return dx, dy
    except Exception:
        return None


def _entity_frame_key(entity_id: str) -> tuple[str, int] | None:
    match = re.match(r"^(?P<prefix>.*?)(?P<frame>\d{3})$", entity_id or "")
    if not match:
        return None
    return match.group("prefix"), int(match.group("frame"))


def _validated_neighbour_coarse_shifts(cache_dir: str, entity_id: str,
                                       reference: str) -> list[tuple[float, float]]:
    """Read validated sibling coarse shifts for strip-coherence priors."""
    key = _entity_frame_key(entity_id)
    if key is None:
        return []
    prefix, frame = key
    ortho_dir = os.path.join(cache_dir, "ortho")
    try:
        names = os.listdir(ortho_dir)
    except OSError:
        return []
    shifts: list[tuple[float, float]] = []
    for name in names:
        if not name.endswith("_ortho.coarse.json"):
            continue
        other_eid = name[:-len("_ortho.coarse.json")]
        other_key = _entity_frame_key(other_eid)
        if other_key is None:
            continue
        other_prefix, other_frame = other_key
        if other_prefix != prefix or other_frame == frame:
            continue
        path = os.path.join(ortho_dir, name)
        try:
            with open(path) as f:
                prov = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        try:
            if int(prov.get("schema_version", 0)) < 2:
                continue
        except (TypeError, ValueError):
            continue
        if not bool(prov.get("coarse_validated", False)):
            continue
        # The shift corrects USGS-corner error in the target's geotransform
        # and is invariant to which correctly-georeferenced reference was
        # used to measure it; accept sibling priors across references so a
        # frame validated against (e.g.) the 1976 KH-9 reference can still
        # seed neighbours run against ESRI WorldImagery. Drift between
        # modern references is sub-pixel; the per-axis gate downstream
        # rejects any prior that disagrees with the local NCC peak by more
        # than the strip-coherence bound.
        try:
            shifts.append((float(prov["coarse_dx_m"]), float(prov["coarse_dy_m"])))
        except (KeyError, TypeError, ValueError):
            continue
    return shifts


def _bracketing_stitched_camera_neighbors(cache_dir: str, entity_id: str) -> tuple[list[str], float, list[str]] | None:
    """Return lower/upper stitched .tsai neighbors and interpolation alpha."""
    key = _entity_frame_key(entity_id)
    if key is None:
        return None
    prefix, frame = key
    stitched_dir = os.path.join(cache_dir, "stitched")
    try:
        names = os.listdir(stitched_dir)
    except OSError:
        return None
    candidates: list[tuple[int, str, str]] = []
    for name in names:
        if not name.endswith("_stitched.tsai") or name.endswith(".interp.tsai"):
            continue
        other_eid = name[:-len("_stitched.tsai")]
        other_key = _entity_frame_key(other_eid)
        if other_key is None:
            continue
        other_prefix, other_frame = other_key
        if other_prefix != prefix or other_frame == frame:
            continue
        candidates.append((other_frame, other_eid, os.path.join(stitched_dir, name)))
    lower = [item for item in candidates if item[0] < frame]
    upper = [item for item in candidates if item[0] > frame]
    if not lower or not upper:
        return None
    left = max(lower, key=lambda item: item[0])
    right = min(upper, key=lambda item: item[0])
    span = max(1, right[0] - left[0])
    alpha = (frame - left[0]) / span
    return [left[2], right[2]], float(alpha), [left[1], right[1]]


def _write_coarse_ortho_provenance(provenance_path: str, source_path: str,
                                   reference: str,
                                   coarse_details: dict | None = None) -> None:
    data = {
        "schema_version": 2,
        "reference_basename": os.path.basename(reference),
        "reference_mtime_ns": os.stat(reference).st_mtime_ns,
        "reference_size": os.path.getsize(reference),
        "source_path": os.path.abspath(source_path),
        "source_mtime_ns": os.stat(source_path).st_mtime_ns,
        "source_size": os.path.getsize(source_path),
    }
    if coarse_details:
        data.update({
            "coarse_dx_m": float(coarse_details.get("dx_m", 0.0)),
            "coarse_dy_m": float(coarse_details.get("dy_m", 0.0)),
            "coarse_n_matches": int(coarse_details.get("n_matches", 0)),
            "coarse_agreement": float(coarse_details.get("agreement", 0.0)),
            "coarse_validated": bool(coarse_details.get("validated", False)),
        })
    os.makedirs(os.path.dirname(provenance_path), exist_ok=True)
    tmp = provenance_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, provenance_path)


def _coarse_sidecar_provenance_matches(provenance_path: str, source_path: str,
                                       reference: str) -> bool:
    """True iff the sidecar provenance records the given reference + source."""
    if not os.path.exists(provenance_path):
        return False
    try:
        with open(provenance_path) as f:
            prov = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    try:
        try:
            if int(prov.get("schema_version", 0)) < 2:
                return False
        except (TypeError, ValueError):
            return False
        if prov.get("reference_basename") != os.path.basename(reference):
            return False
        if prov.get("reference_size") != os.path.getsize(reference):
            return False
        if prov.get("reference_mtime_ns") != os.stat(reference).st_mtime_ns:
            return False
        if prov.get("source_size") != os.path.getsize(source_path):
            return False
        if prov.get("source_mtime_ns") != os.stat(source_path).st_mtime_ns:
            return False
    except OSError:
        return False
    return True


def _record_coarse_align_status(cache_dir: str, entity_id: str,
                                status: str) -> None:
    """Persist the coarse-align outcome on the scene metadata so the
    align-loop can hard-fail unreliable-corners profiles whose preprocess
    coarse-align abstained. Tolerant of missing metadata (we only stamp
    the field if a metadata file exists; first-time scenes get the field
    written by the success path).
    """
    metadata = _load_scene_metadata(cache_dir, entity_id)
    if metadata is None:
        return
    if metadata.get("coarse_align_status") == status:
        return
    metadata["coarse_align_status"] = status
    _save_scene_metadata(cache_dir, entity_id, metadata)


def _coarse_align_ortho_to_sidecar(source_path: str, reference: str, bbox,
                                   label: str, cache_dir: str,
                                   entity_id: str, *,
                                   model_cache=None,
                                   params=None) -> str | None:
    """Coarse-align ``source_path`` against ``reference`` by SHIFTING the
    geotransform and write the shifted copy to a reference-specific
    sidecar. Do NOT mutate ``source_path`` in place; the canonical ortho
    (and per-segment VRT) stays reference-neutral so switching
    ``--reference`` doesn't poison the shared preprocessing cache.

    Returns the sidecar path on success, ``None`` on failure. The sidecar
    is at ``paths.ortho_coarse_path`` with a companion provenance JSON that
    records which (source, reference) pair produced it; runs with a
    different reference rebuild the sidecar rather than inherit a stale
    shift.

    Side effect: stamps ``coarse_align_status`` ("ok" | "abstained") on
    the scene metadata so ``generate_manifest`` can hard-fail entities on
    profiles flagged ``usgs_corners_reliable: false``.
    """
    if not reference or not os.path.exists(reference):
        return None
    if not os.path.exists(source_path):
        return None
    sidecar = paths.ortho_coarse_path(cache_dir, entity_id)
    provenance = paths.ortho_coarse_provenance_path(cache_dir, entity_id)
    if (os.path.exists(sidecar)
            and _coarse_sidecar_provenance_matches(provenance, source_path, reference)):
        print(f"  [coarse_crop] {label} sidecar up-to-date: {sidecar}")
        _record_coarse_align_status(cache_dir, entity_id, "ok")
        return sidecar
    # Lazy-load profile params from scene metadata when callers didn't
    # pass ``params``. Used by the stacked NCC fallback in
    # preprocess.coarse_align_ncc_stack to read camera knobs.
    if params is None:
        meta = _load_scene_metadata(cache_dir, entity_id) or {}
        profile_name = meta.get("profile")
        if profile_name:
            try:
                params = load_profile(profile_name)
            except Exception:
                params = None
    try:
        neighbour_shifts = _validated_neighbour_coarse_shifts(
            cache_dir, entity_id, reference)
        if neighbour_shifts:
            med_dx = sorted(s[0] for s in neighbour_shifts)[len(neighbour_shifts) // 2]
            med_dy = sorted(s[1] for s in neighbour_shifts)[len(neighbour_shifts) // 2]
            print(f"  [coarse_crop] validated strip prior from "
                  f"{len(neighbour_shifts)} neighbour(s): "
                  f"dx≈{med_dx:+.0f}m, dy≈{med_dy:+.0f}m")
        coarse_result = coarse_align_and_crop(
            source_path, reference, sidecar,
            target_bbox_wgs=bbox, crop=False,
            model_cache=model_cache,
            params=params,
            neighbour_shifts_m=neighbour_shifts,
            return_details=True,
        )
    except Exception as exc:
        print(f"  [coarse_crop] {label} coarse-align failed: {exc}")
        _record_coarse_align_status(cache_dir, entity_id, "abstained")
        return None
    coarse_details = None
    if isinstance(coarse_result, tuple):
        coarse_result, coarse_details = coarse_result
    if not coarse_result or not os.path.exists(coarse_result):
        _record_coarse_align_status(cache_dir, entity_id, "abstained")
        return None
    if coarse_details is None:
        shift = _coarse_shift_from_geotransforms(source_path, coarse_result)
        if shift is not None:
            coarse_details = {
                "dx_m": shift[0],
                "dy_m": shift[1],
                "n_matches": 0,
                "agreement": 0.0,
                "validated": False,
            }
    try:
        _write_coarse_ortho_provenance(
            provenance, source_path, reference, coarse_details)
    except OSError as exc:
        print(f"  [coarse_crop] provenance write failed ({exc}); discarding sidecar")
        try:
            os.remove(sidecar)
        except OSError:
            pass
        _record_coarse_align_status(cache_dir, entity_id, "abstained")
        return None
    print(f"  [coarse_crop] {label} coarse-shifted to sidecar "
          f"(canonical preserved): {sidecar}")
    _record_coarse_align_status(cache_dir, entity_id, "ok")
    return sidecar


def _coarse_ortho_for_reference(cache_dir: str, entity_id: str,
                                source_path: str | None,
                                reference: str | None) -> str | None:
    """Return the coarse-align sidecar path iff its provenance matches the
    given ``(source_path, reference)``. ``source_path`` is the canonical
    ortho or per-segment VRT that was shifted to produce the sidecar.
    """
    if not reference or not os.path.exists(reference):
        return None
    if not source_path or not os.path.exists(source_path):
        return None
    sidecar = paths.ortho_coarse_path(cache_dir, entity_id)
    if not os.path.exists(sidecar):
        return None
    provenance = paths.ortho_coarse_provenance_path(cache_dir, entity_id)
    if _coarse_sidecar_provenance_matches(provenance, source_path, reference):
        return sidecar
    return None


def _build_stitched_ortho(scene, cache_dir: str, stitched_path: str,
                          camera_params: dict, corners: dict,
                          reference: str | None, bbox,
                          dem_path: str, *,
                          model_cache=None, params=None) -> str | None:
    """Whole-strip cam_gen + mapproject. Returns the (reference-neutral)
    canonical ortho path and — when a reference is given — also writes a
    coarse-align sidecar at ``paths.ortho_coarse_path`` so alignment can
    pick it up without mutating the canonical cache."""
    if not os.path.exists(stitched_path):
        entity_dir = paths.extracted_dir(cache_dir, scene.entity_id)
        frames = list_frames(entity_dir)
        if frames:
            stitched_path = _stitch_if_needed(
                frames, scene.entity_id, scene.camera_system, stitched_path, cache_dir,
            )
    if not os.path.exists(stitched_path):
        return None
    cam_path = generate_camera(
        stitched_path, camera_params, corners, dem_path=dem_path,
        altitude_m=_altitude_seed_for_scene(scene),
    )
    if cam_path is None:
        return None
    # Render at native GSD (altitude × pixel_pitch / focal_length) by
    # default so orthos preserve the original film resolution. Users
    # can force reference-resolution rendering via
    # DECLASS_REFERENCE_ORTHO=1 when alignment runtime is tight.
    _native = _native_ortho_resolution_m(scene, camera_params)
    _ref = _reference_resolution(reference)
    force_ref = os.environ.get("DECLASS_REFERENCE_ORTHO", "").lower() in {"1", "true", "yes"}
    ortho_res = (_ref if force_ref and _ref else _native) or _ref
    stitched_ortho = mapproject_image(
        stitched_path,
        cam_path,
        dem_path=dem_path,
        output_path=paths.ortho_path(cache_dir, scene.entity_id),
        resolution=ortho_res,
        t_srs="EPSG:3857",
    )
    if not stitched_ortho:
        return stitched_ortho
    # Validity gate: mapproject can silently write an all-nodata raster
    # (sparse-tiled GeoTIFF) when the camera pose projects entirely
    # outside the DEM, the projection geometry blows up numerically, or
    # the run was killed mid-write. The downstream coarse-align trips
    # its "Insufficient content for matching" early gate and the entity
    # is dropped without any clear failure signal. Validate the ortho
    # actually has content; if not, remove the empty file so the next
    # run regenerates from scratch.
    if not _ortho_has_content(stitched_ortho, min_valid_fraction=0.001):
        print(f"  [ortho] mapproject output {os.path.basename(stitched_ortho)} "
              f"has < 0.1% valid pixels; removing as broken")
        try:
            os.remove(stitched_ortho)
        except OSError:
            pass
        neighbour_info = _bracketing_stitched_camera_neighbors(
            cache_dir, scene.entity_id)
        if neighbour_info is None:
            return None
        neighbour_tsais, alpha, neighbour_ids = neighbour_info
        interp_cam = os.path.splitext(stitched_path)[0] + ".interp.tsai"
        interp_path = interpolate_camera_pose(
            neighbour_tsais, interp_cam, alpha=alpha, base_tsai_path=cam_path)
        if not interp_path:
            return None
        print(f"  [ortho] retrying mapproject with interpolated pose from "
              f"neighbours {neighbour_ids}")
        stitched_ortho = mapproject_image(
            stitched_path,
            interp_path,
            dem_path=dem_path,
            output_path=paths.ortho_path(cache_dir, scene.entity_id),
            resolution=ortho_res,
            t_srs="EPSG:3857",
        )
        if (not stitched_ortho
                or not _ortho_has_content(stitched_ortho, min_valid_fraction=0.001)):
            if stitched_ortho:
                print(f"  [ortho] interpolated mapproject output "
                      f"{os.path.basename(stitched_ortho)} still has "
                      f"< 0.1% valid pixels; removing as broken")
                try:
                    os.remove(stitched_ortho)
                except OSError:
                    pass
            return None
    # cam_gen's seed from the USGS 4-corner centroid is often 10–30 km
    # off on KH-4/4B. The shift lands in a reference-specific sidecar
    # (paths.ortho_coarse_path); the canonical stays reference-neutral.
    _coarse_align_ortho_to_sidecar(
        stitched_ortho, reference, bbox, "stitched ortho",
        cache_dir, scene.entity_id,
        model_cache=model_cache,
        params=params,
    )
    return stitched_ortho


def _maybe_generate_asp_ortho(scene, cache_dir: str, stitched_path: str,
                              corners: dict, reference: str | None,
                              *,
                              model_cache=None) -> str | None:
    camera_params = _camera_params_for_scene(scene)
    if camera_params is None:
        return None

    bbox = _bbox_from_corners(corners)
    dem_path = fetch_and_prepare_dem(
        west=bbox[0] - 0.1, south=bbox[1] - 0.1,
        east=bbox[2] + 0.1, north=bbox[3] + 0.1,
    )

    # Per-segment ortho is hard-disabled (see align/params.py); always
    # build the whole-strip cam_gen + mapproject ortho.
    profile = load_profile(_profile_name_for_scene(scene))
    profile_name = getattr(getattr(profile, "meta", None), "name", "?")
    print(f"  [ortho] strategy=stitched (profile={profile_name})")

    return _build_stitched_ortho(
        scene, cache_dir, stitched_path, camera_params, corners, reference, bbox, dem_path,
        model_cache=model_cache, params=profile,
    )


def _ensure_scene_asp_ortho(scene, cache_dir: str, reference: str | None,
                            metadata: dict | None = None, file_map: dict | None = None,
                            *,
                            model_cache=None) -> dict:
    metadata = metadata or (_load_scene_metadata(cache_dir, scene.entity_id) or _default_scene_metadata(scene, cache_dir))
    if not reference or not os.path.exists(reference):
        return metadata
    if _camera_params_for_scene(scene) is None:
        return metadata

    asp_ortho_path = metadata.get("asp_ortho_path")

    georef_path = metadata.get("georef_path") or paths.georef_path(cache_dir, scene.entity_id)
    stitched_path = metadata.get("stitched_path") or paths.stitched_path(cache_dir, scene.entity_id)
    if not _path_is_stale(asp_ortho_path, stitched_path, georef_path):
        # Canonical is fresh, but the *sidecar* may be stale (e.g. the
        # caller switched ``--reference`` since the last run, or the
        # sidecar was never built because the previous run predates this
        # logic). Refresh the sidecar so its provenance always tracks the
        # current reference and ``coarse_align_status`` lands in metadata
        # for generate_manifest's hard-fail abstain skip to consult.
        if asp_ortho_path:
            corners = _corners_from_metadata(scene, metadata)
            try:
                _scene_profile = load_profile(_profile_name_for_scene(scene))
            except Exception:
                _scene_profile = None
            _coarse_align_ortho_to_sidecar(
                asp_ortho_path, reference, _bbox_from_corners(corners),
                "stitched ortho", cache_dir, scene.entity_id,
                model_cache=model_cache,
                params=_scene_profile,
            )
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
    asp_ortho_path = _maybe_generate_asp_ortho(
        scene,
        cache_dir,
        stitched_path,
        corners,
        reference,
        model_cache=model_cache,
    )
    if not asp_ortho_path:
        return metadata

    asp_camera_path = paths.ba_camera_path(stitched_path)
    return _merge_scene_metadata(
        cache_dir,
        scene,
        stitched_path=os.path.abspath(stitched_path),
        asp_camera_path=os.path.abspath(asp_camera_path) if os.path.exists(asp_camera_path) else None,
        asp_ortho_path=os.path.abspath(asp_ortho_path),
        **_primary_input_update(georef_path, asp_ortho_path),
    )


def _preferred_alignment_input_info(cache_dir: str, entity_id: str,
                                    reference: str | None = None
                                    ) -> tuple[str | None, str | None]:
    """Return ``(kind, path)`` for the best alignment input.

    When ``reference`` is given and a coarse-align sidecar exists whose
    provenance matches ``(primary_source, reference)``, that sidecar is
    preferred over the reference-neutral canonical. The canonical path
    itself is never mutated, so switching reference only invalidates the
    sidecar — the canonical stays shared across runs.
    """
    metadata = _load_scene_metadata(cache_dir, entity_id)
    primary_kind: str | None = None
    primary_path: str | None = None
    if metadata:
        primary_kind = metadata.get("primary_input_kind")
        primary_path = metadata.get("primary_input_path")
    if not (primary_kind and primary_path and os.path.exists(primary_path)):
        if metadata:
            ortho_path = metadata.get("asp_ortho_path")
            if ortho_path and os.path.exists(ortho_path):
                primary_kind, primary_path = "asp_ortho", ortho_path
            else:
                georef_path = metadata.get("georef_path")
                if georef_path and os.path.exists(georef_path):
                    primary_kind, primary_path = "georef", georef_path
        if not (primary_kind and primary_path):
            georef_path = paths.georef_path(cache_dir, entity_id)
            if os.path.exists(georef_path):
                primary_kind, primary_path = "georef", georef_path
    if not (primary_kind and primary_path and os.path.exists(primary_path)):
        return (None, None)
    if primary_kind == "asp_ortho" and reference:
        sidecar = _coarse_ortho_for_reference(
            cache_dir, entity_id, primary_path, reference,
        )
        if sidecar:
            return ("asp_ortho_coarse", sidecar)
    return (primary_kind, primary_path)


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


def _reuse_georef_cache(scene, cd: str, reference: str, file_map: dict,
                        georef_path: str,
                        progress: dict, eid: str,
                        *,
                        model_cache=None) -> bool:
    """Short-circuit when georef already exists: refresh metadata and skip.

    When metadata is missing (cold cache), write fresh metadata and proceed
    with ortho generation. When metadata exists but the cache key drifted,
    reject (deletes the georef so the next call rebuilds from scratch).
    """
    metadata = _load_scene_metadata(cd, eid)
    cache_key = _build_georef_cache_key(scene, reference, georef_path)

    if metadata is not None and metadata.get("georef_cache_key") != cache_key:
        return _georef_cache_reject(
            cd, scene, georef_path, "cache_key_changed", cache_key)

    if metadata is not None:
        try:
            profile = load_profile(_profile_name_for_scene(scene))
            unreliable_corners = not bool(profile.camera.usgs_corners_reliable)
        except Exception:
            unreliable_corners = True
        if unreliable_corners and reference and os.path.exists(reference):
            coarse_status = metadata.get("coarse_align_status")
            if coarse_status != "ok":
                return _georef_cache_reject(
                    cd, scene, georef_path,
                    f"coarse_align_status_not_ok:{coarse_status}", cache_key)
            primary_kind = metadata.get("primary_input_kind")
            primary_path = metadata.get("primary_input_path")
            if primary_kind == "asp_ortho" and primary_path:
                sidecar = _coarse_ortho_for_reference(
                    cd, eid, primary_path, reference)
                if sidecar is None:
                    return _georef_cache_reject(
                        cd, scene, georef_path,
                        "coarse_sidecar_provenance_mismatch", cache_key)

    metadata = _merge_scene_metadata(
        cd, scene,
        georef_path=os.path.abspath(georef_path),
        georef_cache_key=cache_key,
        georef_cache_reuse_status="accepted",
        georef_cache_reuse_reason="cache_valid",
    )
    metadata = _ensure_scene_asp_ortho(
        scene, cd, reference,
        metadata=metadata, file_map=file_map,
        model_cache=model_cache,
    )
    _merge_scene_metadata(
        cd, scene,
        georef_cache_key=cache_key,
        georef_cache_reuse_status="accepted",
        georef_cache_reuse_reason="cache_valid",
        **_primary_input_update(
            metadata.get("georef_path"),
            metadata.get("asp_ortho_path"),
        ),
    )
    print(f"  [skip] Already georeferenced: {eid}")
    progress["completed"][eid] = {"stage": "georef"}
    progress["failed"].pop(eid, None)
    return True


def _run_scene_standard_path(scene, cd: str, eid: str, camera, frames, corners,
                             reference: str,
                             georef_path: str, preserve_stitched: bool,
                             *,
                             model_cache=None) -> None:
    """Standard path: stitch → orient → georef → verify → ortho."""
    stitched_path = paths.stitched_path(cd, eid)
    input_for_orient = _stitch_if_needed(frames, eid, camera, stitched_path, cd)

    print(f"\n  --- Orientation: {eid} ---")
    rotation, gcp_corners = detect_orientation(
        input_for_orient, corners, camera, reference_path=reference)
    if rotation != 0:
        print(f"  Orientation: {rotation} CW (via GCP assignment, no image rotation)")

    is_panoramic = camera.program == "CORONA"
    print(f"\n  --- Georef: {eid} ---")
    georef_with_corners(input_for_orient, georef_path, gcp_corners,
                        panoramic=is_panoramic)

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
                scene, cd, input_for_orient, gcp_corners, reference,
                model_cache=model_cache,
            )
            if asp_ortho_path:
                print(f"  ASP ortho ready: {os.path.basename(asp_ortho_path)}")
        except Exception as e:
            print(f"  WARNING: ASP orthorectification failed for {eid}: {e}")

    asp_camera_path = paths.ba_camera_path(input_for_orient)
    _merge_scene_metadata(
        cd, scene,
        gcp_corners={k: list(v) for k, v in gcp_corners.items()},
        georef_path=os.path.abspath(georef_path),
        stitched_path=os.path.abspath(input_for_orient),
        asp_camera_path=os.path.abspath(asp_camera_path) if os.path.exists(asp_camera_path) else None,
        asp_ortho_path=os.path.abspath(asp_ortho_path) if asp_ortho_path else None,
        georef_cache_key=_build_georef_cache_key(scene, reference, georef_path),
        georef_cache_reuse_status="refreshed",
        georef_cache_reuse_reason="processed",
        **_primary_input_update(georef_path, asp_ortho_path),
    )

    stitched_int = paths.stitched_path(cd, eid)
    if os.path.exists(stitched_int) and not preserve_stitched:
        os.remove(stitched_int)
        print(f"  Removed intermediate: {os.path.basename(stitched_int)}")


def extract_stitch_georef_scene(scene, output_dir: str, file_map: dict, reference: str,
                                progress: dict, dry_run: bool = False,
                                cache_dir: str | None = None,
                                preserve_stitched: bool = False,
                                *,
                                model_cache=None) -> bool:
    """Run the extract → stitch → georef cascade on a single scene.

    Preprocessing outputs (extracted, stitched, georef) go under *cache_dir*
    so they can be reused across test runs.

    Returns True on success, False on failure.
    """
    cd = cache_dir or output_dir
    eid = scene.entity_id
    camera = scene.camera_system

    georef_path = paths.georef_path(cd, eid)
    if os.path.exists(georef_path):
        if _reuse_georef_cache(scene, cd, reference, file_map,
                               georef_path, progress, eid,
                               model_cache=model_cache):
            return True
        # Cache rejected (deleted georef); fall through to fresh extraction.

    if dry_run:
        print(f"  [dry-run] Would process: {eid} ({camera.name})")
        return True

    file_path = file_map.get(eid)
    if not file_path or not os.path.exists(file_path):
        print(f"  WARNING: No downloaded file for {eid}, skipping")
        progress["failed"][eid] = "no_download"
        return False

    try:
        print(f"\n  --- Extract: {eid} ---")
        entity_dir = extract_archive(file_path, cd, eid)
        frames = list_frames(entity_dir)

        if not frames:
            print(f"  WARNING: No frames found for {eid}")
            progress["failed"][eid] = "no_frames"
            return False

        corners = scene.corners
        _run_scene_standard_path(
            scene, cd, eid, camera, frames, corners, reference,
            georef_path, preserve_stitched,
            model_cache=model_cache,
        )

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
                      device: str = "auto",
                      anchors_override: str | None = None,
                      secondary_references: list[str] | None = None) -> str:
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
            input_kind, input_path = _preferred_alignment_input_info(
                cd, eid, reference=reference,
            )
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

            # Hard-fail when preprocess coarse-align abstained AND the
            # camera profile flags USGS corners as unreliable. Aligning
            # against an ortho positioned only by USGS corners (10–30 km
            # off on KH-4 / KH-7) is worse than not aligning: ROMA finds
            # an internally-consistent fit on the wrong shoreline. Two
            # ways to bypass:
            #   * DECLASS_ALLOW_UNCOARSE_ALIGN=1 — diagnostic escape hatch.
            #   * --anchors-override <PATH> — hand-curated anchors give us
            #     the geographic prior coarse-align couldn't, so the
            #     anchor stage can pin position from the lat/lon side.
            coarse_status = (metadata or {}).get("coarse_align_status")
            allow_uncoarse = os.environ.get(
                "DECLASS_ALLOW_UNCOARSE_ALIGN", "").lower() in {"1", "true", "yes"}
            override_provides_prior = bool(
                anchors_override and os.path.exists(anchors_override))
            if (coarse_status == "abstained" and not allow_uncoarse
                    and not override_provides_prior):
                profile = load_profile(_profile_name_for_scene(scene))
                if not profile.camera.usgs_corners_reliable:
                    print(f"  [skip] {eid} — coarse-align abstained and "
                          f"profile '{profile.meta.name}' flags USGS corners "
                          f"as unreliable; set DECLASS_ALLOW_UNCOARSE_ALIGN=1 "
                          f"or pass --anchors-override to override")
                    skipped += 1
                    continue
            if (coarse_status == "abstained"
                    and override_provides_prior):
                print(f"  [info] {eid} — coarse-align abstained but "
                      f"--anchors-override supplies hand-curated GCPs; "
                      f"proceeding with alignment")

            # Pre-alignment coarse crop is disabled by default: the USGS-
            # corner bbox can be 20+ km off, which causes the crop to
            # discard valid data. auto-align runs its own coarse→fine
            # offset detection on the full ortho. Set
            # DECLASS_PRE_ALIGN_CROP=1 to opt back into the crop for
            # tight-budget workflows where discarding edge data is
            # acceptable.
            input_for_align = input_path
            if os.environ.get("DECLASS_PRE_ALIGN_CROP", "").lower() in {"1", "true", "yes"}:
                cropped_path = _alignment_crop_path(cd, eid, input_kind or "georef")
                if _path_is_stale(cropped_path, input_path):
                    try:
                        _crop_profile = load_profile(_profile_name_for_scene(scene))
                    except Exception:
                        _crop_profile = None
                    crop_result = coarse_align_and_crop(
                        input_path, reference, cropped_path,
                        target_bbox_wgs=frame_bbox,
                        params=_crop_profile,
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

            # Write per-frame metadata prior from USGS corners.
            # ``acquisition_date`` is embedded so step_setup's era-gap
            # inference can compute the year-gap to the reference for
            # the QA scorer's cross-temporal corrections.
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
                "primary_input_kind": input_kind,
                "acquisition_date": getattr(scene, "acquisition_date", None),
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

            # Generate automatic anchor GCPs from coarse RoMa matching.
            # When --anchors-override is set, drop the hand-curated file at
            # the canonical scene path; the existing skip-if-exists guard
            # then preserves it instead of regenerating. This lets
            # run_test.py pin the iteration loop on its known-good anchor
            # set without forking the production code path.
            anchors_path = paths.scene_anchors_path(output_dir, eid)
            if anchors_override and os.path.exists(anchors_override):
                os.makedirs(os.path.dirname(anchors_path), exist_ok=True)
                if not os.path.exists(anchors_path) or _path_is_stale(
                        anchors_path, anchors_override):
                    import shutil as _sh
                    _sh.copy2(anchors_override, anchors_path)
                    print(f"  [auto_anchors] Using --anchors-override "
                          f"({os.path.basename(anchors_override)}) for {eid}")
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
                "profile": _profile_name_for_scene(scene),
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

    shared: dict = {
        "reference": os.path.abspath(reference),
        "device": device,
        "global_search": True,
        "allow_abstain": True,
    }
    if secondary_references:
        # Absolute paths so auto-align in a different cwd resolves them.
        shared["secondary_references"] = [
            os.path.abspath(p) for p in secondary_references if p
        ]
    manifest = {
        "shared": shared,
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
    # Shared ELoFTR ModelCache for the preprocess coarse-align step. Lazy-
    # created in stage_preprocess_scenes the first time a scene is processed
    # under a reference, then disposed at the end of the stage. Threaded
    # through extract_stitch_georef_scene → _run_scene_* → _maybe_generate_
    # asp_ortho → _coarse_align_ortho_to_sidecar → coarse_align_and_crop so
    # batches of scenes amortise the ~5 s ELoFTR weights load instead of
    # paying it per scene.
    eloftr_cache: Optional[object] = None


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
    parser.add_argument("--secondary-reference", action="append", default=[],
                        metavar="PATH",
                        help="Additional reference(s) tried when the primary "
                             "alignment doesn't accept (e.g. a wider modern "
                             "basemap when the era-matched primary leaves "
                             "edge frames unmatched). Repeatable.")
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
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                        help="Torch device override for auto-anchor generation and alignment")
    parser.add_argument("--anchors-override", default=None,
                        help="Path to a hand-curated anchor JSON (same format as the "
                             "auto-anchors output). When set, the file is copied to each "
                             "scene's scene_anchors_path BEFORE generate_manifest's "
                             "auto-anchor branch runs, so the auto-anchor stage skips "
                             "regeneration. Used by scripts/test/run_test.py to pin the "
                             "iteration loop on a known-good anchor set.")
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

    # Lazy-create the ELoFTR ModelCache shared across all scenes' coarse-
    # align calls. Skipped entirely when no reference is set (the coarse
    # stage is a no-op without one). Disposed in finally so MPS / CUDA
    # memory is released at end-of-stage even on exceptions.
    if ctx.eloftr_cache is None and ctx.reference and os.path.exists(ctx.reference):
        try:
            from align.models import ModelCache, get_torch_device
            ctx.eloftr_cache = ModelCache(get_torch_device(ctx.args.device))
        except Exception as exc:
            print(f"  [coarse_crop] ELoFTR ModelCache unavailable ({exc}); "
                  f"each scene will lazy-load")
            ctx.eloftr_cache = None

    try:
        for scene in ctx.downloadable:
            print(f"\n{'─' * 50}")
            print(f"Processing: {scene.entity_id} ({scene.camera_system.name}, {scene.acquisition_date})")
            print(f"{'─' * 50}")
            ok = extract_stitch_georef_scene(
                scene, ctx.output_dir, ctx.file_map, ctx.reference, ctx.progress,
                cache_dir=ctx.cache_dir,
                model_cache=ctx.eloftr_cache,
            )
            if ok:
                ctx.success_count += 1
            else:
                ctx.fail_count += 1
            save_progress(ctx.cache_dir, ctx.progress)
    finally:
        if ctx.eloftr_cache is not None:
            try:
                ctx.eloftr_cache.close()
            except Exception:
                pass
            ctx.eloftr_cache = None

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


def stage_align_scenes(ctx: PipelineContext) -> None:
    _print_stage_banner("Stage 6: Alignment")
    secondary_refs = list(getattr(ctx.args, "secondary_reference", None) or [])
    manifest_path = generate_manifest(
        ctx.downloadable, ctx.output_dir, ctx.primary_reference,
        cache_dir=ctx.cache_dir, device=ctx.args.device,
        anchors_override=getattr(ctx.args, "anchors_override", None),
        secondary_references=secondary_refs,
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

    # OOM guard — warn / abort on under-provisioned machines and optionally
    # install a soft RLIMIT_AS cap. The KH-9 PC per-segment path crashed a
    # MacBookPro18,4 overnight with jetsam→kernel panic (2026-04-21) after
    # hitting 144 GB uncompressed demand. See align/memory_guard.py for
    # tunables (DECLASS_MIN_FREE_GB, DECLASS_ABORT_FLOOR_GB,
    # DECLASS_MEMORY_CAP_GB).
    from align.memory_guard import apply_process_memory_cap, check_memory_or_warn
    apply_process_memory_cap()
    check_memory_or_warn("startup")

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
