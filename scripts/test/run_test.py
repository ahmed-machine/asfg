#!/usr/bin/env python3
"""
Test runner for the declass alignment pipeline — scene-parameterised.

Runs the pipeline, captures stdout/stderr, parses logs into a structured
summary.json for automated analysis. The scene (entity id + reference +
catalog) comes from a scripts/test/e2e_configs/<scene>.yaml so the same
runner drives both the KH-9 PC Bahrain test and the KH-4B Bahrain test.

Usage:
    python3 scripts/test/run_test.py [--scene NAME] [--version N] [--timeout 2400]
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paths_config import get_target, get_reference

ANCHORS = str(PROJECT_ROOT / "data" / "bahrain_anchor_gcps.json")

# Shared preprocessing cache (downloads/extracted/stitched/georef/ortho).
# Re-used across run_test.py and run_e2e_test.py so frames don't re-download.
CACHE_DIR = PROJECT_ROOT / "output"
E2E_CONFIG_DIR = PROJECT_ROOT / "scripts" / "test" / "e2e_configs"
DEFAULT_SCENE = "bahrain_kh9_pc_1977"


def _load_scene_config(scene_name: str) -> dict:
    """Resolve scripts/test/e2e_configs/<scene_name>.yaml to a dict with
    resolved absolute paths.

    Returns keys: ``name``, ``entity_id``, ``reference_path``,
    ``boundary_path``, ``catalogs`` (list of absolute paths), and the
    optional ``prefer_camera``. ``reference_name`` in the YAML is looked
    up via :func:`scripts.paths_config.get_reference`.
    """
    import yaml

    cfg_path = E2E_CONFIG_DIR / f"{scene_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Scene config not found: {cfg_path}. "
            f"Available: {[p.stem for p in E2E_CONFIG_DIR.glob('*.yaml')]}"
        )
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh) or {}

    entity_id = cfg.get("entity_id")
    if not entity_id:
        raise ValueError(
            f"{cfg_path}: missing required field 'entity_id'"
        )
    reference_name = cfg.get("reference_name") or cfg.get("reference")
    if not reference_name:
        raise ValueError(
            f"{cfg_path}: missing required field 'reference_name'"
        )
    boundary_rel = cfg.get("boundary")
    if not boundary_rel:
        raise ValueError(f"{cfg_path}: missing required field 'boundary'")
    catalogs_rel = cfg.get("catalogs") or []
    if not catalogs_rel:
        raise ValueError(f"{cfg_path}: missing required field 'catalogs'")

    return {
        "name": cfg.get("name", scene_name),
        "entity_id": str(entity_id),
        "reference_path": get_reference(reference_name),
        "boundary_path": str(PROJECT_ROOT / boundary_rel),
        "catalogs": [str(PROJECT_ROOT / c) for c in catalogs_rel],
        "prefer_camera": cfg.get("prefer_camera"),
    }


def ensure_target_built(scene_cfg: dict, preprocess_matcher: str = "roma") -> str:
    """Materialize the per-segment orthorectified target for ``scene_cfg`` and
    return its path. Runs ``process.py --skip-align --skip-mosaic`` against the
    shared output/ cache, so downloads/extracted/georef stages short-circuit on
    mtime checks. Set DECLASS_USE_PRESTITCHED=1 to fall back to the legacy
    pre-stitched Dropbox TIFF for A/B comparison (only valid for the KH-9 PC
    Bahrain scene — the legacy target was hardcoded to that entity).
    """
    entity_id = scene_cfg["entity_id"]
    ref_path = scene_cfg["reference_path"]
    boundary_path = scene_cfg["boundary_path"]
    catalogs = scene_cfg["catalogs"]

    if os.environ.get("DECLASS_USE_PRESTITCHED"):
        legacy = get_target("bahrain_1977")
        print(f"  [ensure_target_built] DECLASS_USE_PRESTITCHED set — using {legacy}")
        return legacy

    # Candidate output paths (see paths.py :: ortho_path, ortho_segments_dir
    # and preprocess/camera_model.py for the per-segment output).
    ortho_tif = CACHE_DIR / "ortho" / f"{entity_id}_ortho.tif"
    ortho_seg_dir = CACHE_DIR / "ortho" / f"{entity_id}_segments"
    selected_matcher = str(preprocess_matcher).strip().lower() or "roma"
    # Prefer materialized TIF over VRT (VRT chains cause GDAL version issues).
    existing_tifs = sorted(ortho_seg_dir.glob("*_per_segment.tif")) if ortho_seg_dir.exists() else []
    existing_vrts = sorted(ortho_seg_dir.glob("*_per_segment.vrt")) if ortho_seg_dir.exists() and not existing_tifs else []

    def _resolve_existing() -> str | None:
        if existing_tifs:
            return str(existing_tifs[-1])
        if existing_vrts:
            return str(existing_vrts[-1])
        if ortho_tif.exists():
            return str(ortho_tif)
        return None

    metadata_path = CACHE_DIR / "georef" / f"{entity_id}_metadata.json"
    metadata = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except Exception:
            metadata = None
    cached_matcher = str((metadata or {}).get("preprocess_matcher", "roma")).strip().lower() or "roma"

    # Fast path: a per-segment mosaic already exists. Prefer materialized
    # TIF over VRT. Crop to the reference overlap before returning.
    per_seg = existing_tifs or existing_vrts
    if per_seg and cached_matcher == selected_matcher:
        print(f"  [ensure_target_built] Using cached per-segment mosaic: {per_seg[-1]}")
        cropped = _crop_ortho_to_reference(str(per_seg[-1]), ref_path, ortho_seg_dir,
                                           entity_id=entity_id)
        print(f"  [ensure_target_built] Cropped target: {cropped}")
        return cropped
    if per_seg and cached_matcher != selected_matcher:
        print(
            f"  [ensure_target_built] Cached per-segment target uses matcher "
            f"{cached_matcher}; rebuilding with {selected_matcher}"
        )

    # Stale single-image ortho sitting in the cache would short-circuit
    # process.py's _path_is_stale check and block per-segment regeneration.
    # Remove it so process.py regenerates using the freshly-enabled
    # per_segment_ortho profile flag.
    if ortho_tif.exists():
        print(f"  [ensure_target_built] Removing stale single-image ortho to force per-segment rebuild: {ortho_tif}")
        ortho_tif.unlink()

    # Also clear the asp_ortho_path key from the scene metadata so
    # _ensure_scene_asp_ortho doesn't short-circuit on the old single-image
    # path. Keep the rest of the georef metadata intact.
    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text())
        except Exception:
            meta = None
        if isinstance(meta, dict) and "asp_ortho_path" in meta:
            meta.pop("asp_ortho_path", None)
            metadata_path.write_text(json.dumps(meta, indent=2))
            print(f"  [ensure_target_built] Cleared asp_ortho_path from {metadata_path}")

    cmd = [
        sys.executable, str(PROJECT_ROOT / "process.py"),
        "--csv", *catalogs,
        "--reference", ref_path,
        "--boundary", boundary_path,
        "--entities", entity_id,
        "--cache-dir", str(CACHE_DIR),
        "--output-dir", str(CACHE_DIR / "preprocess_run"),
        "--skip-align",
        "--skip-mosaic",
        "--preprocess-matcher", preprocess_matcher,
    ]
    if scene_cfg.get("prefer_camera"):
        cmd.extend(["--prefer-camera", scene_cfg["prefer_camera"]])

    print("=== run_test.py: Preprocessing target via process.py ===")
    print(f"  cmd: {' '.join(cmd[:6])}...")
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

    # Re-scan after preprocessing.
    existing_tifs = sorted(ortho_seg_dir.glob("*_per_segment.tif")) if ortho_seg_dir.exists() else []
    existing_vrts = sorted(ortho_seg_dir.glob("*_per_segment.vrt")) if ortho_seg_dir.exists() and not existing_tifs else []
    resolved = _resolve_existing()
    if not resolved:
        raise RuntimeError(
            f"Preprocessing did not produce an ortho under {ortho_seg_dir} or at {ortho_tif}"
        )

    # The per-segment mosaic spans the full ~200 km KH-9 strip, but the
    # reference only covers ~60 km (Bahrain). Crop to the reference's
    # footprint plus ~5 km margin so coarse offset detection and feature
    # matching don't waste effort on regions with no reference coverage.
    cropped = _crop_ortho_to_reference(resolved, ref_path, ortho_seg_dir,
                                       entity_id=entity_id)
    print(f"  [ensure_target_built] Cropped target: {cropped}")
    return cropped


def _crop_ortho_to_reference(ortho_path: str, reference_path: str,
                             output_dir, *, entity_id: str) -> str:
    """Crop a full-strip ortho to its overlap with the reference, plus margin.

    KH-9 panoramic strips are ~200 km long and cross the reference image
    diagonally, so each segment's y-range covers only a fraction of the
    reference's y-range. The correct crop is the intersection of:

      1. the reference image's bounds (plus ~5 km margin), and
      2. the union of segments that actually overlap the reference

    Cropping to the reference bounds alone leaves huge empty regions north
    and south of the actual strip data (which breaks coarse offset
    detection — see v175). Cropping to the segment union alone may leak
    strip content well outside Bahrain. The intersection gives us a tight
    bbox matching v173's pre-stitched extent.
    """
    import rasterio
    from rasterio.warp import transform_bounds

    MARGIN_M = 5000.0
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cropped_tif = out_dir / f"{entity_id}_cropped.tif"

    # Determine target CRS from the ortho.
    with rasterio.open(ortho_path) as ortho_src:
        target_crs = ortho_src.crs.to_string() if ortho_src.crs else "EPSG:3857"

    # Reference bounds in the ortho's CRS.
    with rasterio.open(reference_path) as ref_src:
        ref_bounds_native = ref_src.bounds
        ref_crs = ref_src.crs.to_string() if ref_src.crs else "EPSG:4326"
    ref_left, ref_bottom, ref_right, ref_top = transform_bounds(
        ref_crs, target_crs,
        ref_bounds_native.left, ref_bounds_native.bottom,
        ref_bounds_native.right, ref_bounds_native.top,
    )

    # Enumerate sub-segment tifs (siblings of the ortho) and compute the
    # union of segment bounds that intersect the reference bounds.
    seg_dir = Path(ortho_path).parent
    seg_tifs = sorted(seg_dir.glob("*_seg*_ortho.tif"))
    if not seg_tifs:
        with rasterio.open(ortho_path) as src:
            src_bounds = src.bounds
        seg_union = (src_bounds.left, src_bounds.bottom,
                     src_bounds.right, src_bounds.top)
    else:
        lefts, bottoms, rights, tops = [], [], [], []
        for tif in seg_tifs:
            with rasterio.open(tif) as s:
                b = s.bounds
            if (b.right < ref_left or b.left > ref_right or
                    b.top < ref_bottom or b.bottom > ref_top):
                continue
            lefts.append(b.left)
            bottoms.append(b.bottom)
            rights.append(b.right)
            tops.append(b.top)
        if not lefts:
            print(f"  [crop] No segments overlap reference — using full extent")
            with rasterio.open(ortho_path) as src:
                src_bounds = src.bounds
            seg_union = (src_bounds.left, src_bounds.bottom,
                         src_bounds.right, src_bounds.top)
        else:
            seg_union = (min(lefts), min(bottoms), max(rights), max(tops))

    # Intersection of reference bounds with the overlapping-segments union.
    west = max(seg_union[0], ref_left) - MARGIN_M
    south = max(seg_union[1], ref_bottom) - MARGIN_M
    east = min(seg_union[2], ref_right) + MARGIN_M
    north = min(seg_union[3], ref_top) + MARGIN_M

    if east <= west or north <= south:
        print(f"  [crop] Empty intersection (ref={ref_left:.0f},{ref_bottom:.0f},"
              f"{ref_right:.0f},{ref_top:.0f} segs={seg_union}) — using as-is")
        return ortho_path

    # Reuse existing cropped TIF if it's newer than the source.
    try:
        if cropped_tif.exists() and cropped_tif.stat().st_mtime > Path(ortho_path).stat().st_mtime:
            return str(cropped_tif)
    except OSError:
        pass

    # Materialize crop as a real GeoTIFF (not a VRT).  VRT chains cause
    # rasterio/GDAL version mismatch failures when the CLI writer (ASP
    # GDAL 3.8) differs from the Python reader (homebrew GDAL 3.11).
    gdal_translate = "/opt/homebrew/bin/gdal_translate" if os.path.isfile(
        "/opt/homebrew/bin/gdal_translate") else "gdal_translate"
    cmd = [
        gdal_translate,
        "-projwin", f"{west}", f"{north}", f"{east}", f"{south}",
        "-co", "COMPRESS=LZW", "-co", "TILED=YES", "-co", "BIGTIFF=IF_SAFER",
        ortho_path, str(cropped_tif),
    ]
    print(f"  [crop] gdal_translate -projwin {west:.1f} {north:.1f} {east:.1f} {south:.1f} "
          f"(width={east-west:.0f}m height={north-south:.0f}m)")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not cropped_tif.exists():
        print(f"  [crop] gdal_translate failed: {result.stderr[:400]}")
        return ortho_path

    # Diagnostic: check that the crop has meaningful content.
    try:
        with rasterio.open(str(cropped_tif)) as src:
            w, h = src.width, src.height
            cx, cy = w // 2, h // 2
            sz = min(128, w, h)
            window = rasterio.windows.Window(cx - sz // 2, cy - sz // 2, sz, sz)
            patch = src.read(1, window=window)
            nd = src.nodata if src.nodata is not None else -32768
            valid_frac = float((patch != nd).sum()) / max(1, patch.size)
            print(f"  [crop] Cropped TIF: {w}x{h}px, centre valid={valid_frac:.1%}")
            if valid_frac < 0.05:
                print(f"  [crop] WARNING: <5% valid pixels in centre — "
                      f"possible orientation mismatch or wrong crop region")
    except Exception as e:
        print(f"  [crop] Content check skipped: {e}")

    return str(cropped_tif)


def _diagnostics_root_for_scene(scene_name: str) -> Path:
    """Return the per-scene diagnostics directory.

    For the default KH-9 PC Bahrain scene, we keep the historical
    ``diagnostics/run_v*/`` layout (to preserve comparison with the many
    pre-parameterisation runs sitting in that folder). Any other scene
    gets its own subdirectory.
    """
    if scene_name == DEFAULT_SCENE:
        return PROJECT_ROOT / "diagnostics"
    return PROJECT_ROOT / "diagnostics" / scene_name


def detect_next_version(scene_name: str = DEFAULT_SCENE):
    """Scan diagnostics/{scene}/run_v*/ and return the next version number."""
    diag_dir = _diagnostics_root_for_scene(scene_name)
    existing = glob(str(diag_dir / "run_v*"))
    versions = []
    for d in existing:
        m = re.search(r"run_v(\d+)$", d)
        if m:
            versions.append(int(m.group(1)))
    return max(versions) + 1 if versions else 1


def snapshot_git_state(out_path):
    """Save git commit hash and diff stat to a file."""
    lines = []
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        lines.append(f"commit: {head.stdout.strip()}")
    except Exception:
        lines.append("commit: unknown")
    try:
        diff = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        lines.append(f"\ngit diff --stat:\n{diff.stdout}")
    except Exception:
        pass
    try:
        diff_staged = subprocess.run(
            ["git", "diff", "--staged", "--stat"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        if diff_staged.stdout.strip():
            lines.append(f"\ngit diff --staged --stat:\n{diff_staged.stdout}")
    except Exception:
        pass
    Path(out_path).write_text("\n".join(lines))


def parse_log(log_text):
    """Parse pipeline log into structured data. Returns dict of extracted fields."""
    result = {
        "step_timings": {},
        "coarse_offset": {},
        "anchors": {"located": 0, "total": 0, "rejected": 0},
        "anchor_details": [],
        "auto_gcps": {"ransac_survivors": 0, "roma_coverage": 0},
        "gcp_selection": {"count": 0, "coverage": 0.0},
        "grid_optimizer": {},
        "flow": {"reliable_pct": None, "mean_correction_m": None, "bias_dx_m": None, "bias_dy_m": None, "applied": None},
        "reclamation": {},
        "icp": {},
        "grid_qa": {},
        "tps_qa": {},
        "cv": {"fit_m": None, "cv_m": None},
        "grid_independent": {},
        "tps_independent": {},
        "selected": None,
        "warnings": [],
        "errors": [],
    }

    # Step timings: [step_name] Ns
    for m in re.finditer(r"\[(\w+)\]\s+([\d.]+)s", log_text):
        step, secs = m.group(1), float(m.group(2))
        # Skip grid optimizer iteration lines (they contain extra fields)
        if re.match(r"L\d+", step):
            continue
        result["step_timings"][step] = secs

    # TOTAL timing
    m = re.search(r"\[TOTAL\]\s+([\d.]+)s", log_text)
    if m:
        result["step_timings"]["TOTAL"] = float(m.group(1))

    # Coarse offset (second pass after translation, or only pass)
    # Find the last occurrence of coarse estimate
    for m in re.finditer(r"Coarse estimate:\s*dx=([+\-]?\d+)m,\s*dy=([+\-]?\d+)m", log_text):
        result["coarse_offset"] = {"dx_m": int(m.group(1)), "dy_m": int(m.group(2))}

    # Anchor located
    anchor_names_seen = set()
    for m in re.finditer(r"Located:\s+(.+?)\s+\((.+?)\)\s*\n\s*displacement:\s*dE=([+\-]?[\d.]+)m,\s*dN=([+\-]?[\d.]+)m", log_text):
        name = m.group(1)
        if name not in anchor_names_seen:
            anchor_names_seen.add(name)
            result["anchor_details"].append({
                "name": name,
                "status": "located",
                "method": m.group(2),
                "dE": float(m.group(3)),
                "dN": float(m.group(4)),
            })

    # Anchor recovered via Pass 1.5
    for m in re.finditer(r"(\S.*?):\s+RECOVERED via Pass 1\.5.*?disp:\s*dE=([+\-]?[\d.]+)m,\s*dN=([+\-]?[\d.]+)m", log_text):
        name = m.group(1).strip()
        if name not in anchor_names_seen:
            anchor_names_seen.add(name)
            result["anchor_details"].append({
                "name": name,
                "status": "located",
                "method": "Pass 1.5 recovery",
                "dE": float(m.group(2)),
                "dN": float(m.group(3)),
            })

    # Anchor skipped
    for m in re.finditer(r"Skipped:\s+(.+?)\s+\((.+?)\)", log_text):
        name = m.group(1)
        if name not in anchor_names_seen:
            anchor_names_seen.add(name)
            result["anchor_details"].append({
                "name": name,
                "status": "skipped",
                "reason": m.group(2),
            })

    # Anchor rejection
    m = re.search(r"Anchor QA: rejected (\d+) displacement outlier", log_text)
    if m:
        result["anchors"]["rejected"] = int(m.group(1))

    # Anchor count
    m = re.search(r"(\d+) anchor GCPs located", log_text)
    if m:
        result["anchors"]["located"] = int(m.group(1))

    # Total anchors loaded
    m = re.search(r"Loaded (\d+) anchors", log_text)
    if m:
        result["anchors"]["total"] = int(m.group(1))

    # RoMa coverage
    m = re.search(r"RoMa coverage:\s+(\d+)\s+matches", log_text)
    if m:
        result["auto_gcps"]["roma_coverage"] = int(m.group(1))

    # RANSAC
    m = re.search(r"Geographic RANSAC.*?:\s+(\d+)\s*->\s*(\d+)", log_text)
    if m:
        result["auto_gcps"]["ransac_survivors"] = int(m.group(2))

    # GCP selection
    m = re.search(r"Selected (\d+) GCPs.*?coverage:\s*([\d.]+)%?\)?", log_text)
    if m:
        result["gcp_selection"]["count"] = int(m.group(1))
        cov = float(m.group(2))
        result["gcp_selection"]["coverage"] = cov / 100.0 if cov > 1 else cov

    # Grid optimizer iterations (capture last iteration per level)
    for m in re.finditer(r"\[L(\d+).*?\]\s+Iter\s+(\d+)/(\d+)\s*\|\s*total=([\d.]+)m\s+data=([\d.]+)\s+cham=([\d.]+)", log_text):
        level = int(m.group(1))
        iter_num = int(m.group(2))
        max_iters = int(m.group(3))
        result["grid_optimizer"][f"L{level}_final_total"] = float(m.group(4))
        result["grid_optimizer"][f"L{level}_final_data"] = float(m.group(5))
        result["grid_optimizer"][f"L{level}_final_chamfer"] = float(m.group(6))
        result["grid_optimizer"][f"L{level}_final_iter"] = iter_num
        result["grid_optimizer"][f"L{level}_max_iters"] = max_iters
        result["grid_optimizer"][f"L{level}_converged"] = iter_num < max_iters

    # Grid optimizer iter 1 losses (hierarchical)
    for m in re.finditer(r"\[L(\d+).*?\]\s+Iter 1:\s+data=([\d.]+)\s+cham=([\d.]+)", log_text):
        level = int(m.group(1))
        result["grid_optimizer"][f"L{level}_init_data"] = float(m.group(2))
        result["grid_optimizer"][f"L{level}_init_chamfer"] = float(m.group(3))

    # Grid optimizer iter 1 losses (single-level)
    m = re.search(r"Iter 1 weighted contributions \(m\):\s*\n\s*data=([\d.]+)\s+chamfer=([\d.]+)", log_text)
    if m:
        result["grid_optimizer"]["init_data"] = float(m.group(1))
        result["grid_optimizer"]["init_chamfer"] = float(m.group(2))

    # Flow refinement
    m = re.search(r"\[FlowRefine\]\s+(\d+)%\s+reliable.*?mean correction\s+([\d.]+)m.*?max\s+([\d.]+)m", log_text)
    if m:
        result["flow"]["reliable_pct"] = int(m.group(1))
        result["flow"]["mean_correction_m"] = float(m.group(2))

    # Flow bias
    m = re.search(r"\[FlowRefine\].*?median bias:\s*dx=([+\-]?[\d.]+)m,\s*dy=([+\-]?[\d.]+)m", log_text)
    if m:
        result["flow"]["bias_dx_m"] = float(m.group(1))
        result["flow"]["bias_dy_m"] = float(m.group(2))

    # Flow applied/skipped
    if re.search(r"\[FlowRefine\]\s+Only \d+% reliable.*skipping", log_text):
        result["flow"]["applied"] = False
    elif re.search(r"\[FlowRefine\]\s+Post-refinement complete", log_text):
        result["flow"]["applied"] = True

    # Fold check (final)
    m = re.search(r"WARNING: Final warp has ([\d.]+)% folds.*Falling back to pure affine", log_text)
    if m:
        result["grid_optimizer"]["fold_frac"] = float(m.group(1)) / 100.0
        result["grid_optimizer"]["fold_fallback"] = True
    elif re.search(r"WARNING:.*Falling back to pure affine warp", log_text):
        result["grid_optimizer"]["fold_fallback"] = True
    else:
        result["grid_optimizer"]["fold_fallback"] = False
        # Capture fold fraction from per-level or final check
        fold_fracs = []
        for fm in re.finditer(r"Fold check:\s*([\d.]+)%\s*(?:folded|folds|\(ok\))", log_text):
            fold_fracs.append(float(fm.group(1)) / 100.0)
        if re.search(r"Fold check:\s*clean", log_text):
            fold_fracs.append(0.0)
        if fold_fracs:
            result["grid_optimizer"]["fold_frac"] = fold_fracs[-1]

    # Reclamation
    m = re.search(r"\[Reclamation\]\s*Raw XOR:\s*([\d.]+)%.*?cleaning:\s*([\d.]+)%\s*\((\d+)\s*blobs?,\s*(\d+)\s*large\)", log_text)
    if m:
        result["reclamation"] = {
            "raw_pct": float(m.group(1)),
            "cleaned_pct": float(m.group(2)),
            "n_blobs": int(m.group(3)),
            "n_large": int(m.group(4)),
        }

    # ICP
    m = re.search(r"\[ICP\]\s*Coastline correction:\s*dx=([+\-]?[\d.]+)m,\s*dy=([+\-]?[\d.]+)m", log_text)
    if m:
        result["icp"] = {"applied": True, "dx_m": float(m.group(1)), "dy_m": float(m.group(2))}
    elif re.search(r"\[ICP\]\s*Too few", log_text):
        result["icp"] = {"applied": False, "reason": "too_few_points"}
    elif re.search(r"\[ICP\]\s*Correction too large", log_text):
        result["icp"] = {"applied": False, "reason": "correction_too_large"}
    elif re.search(r"\[ICP\]\s*Negligible", log_text):
        result["icp"] = {"applied": False, "reason": "negligible"}

    # Grid QA — handles both old format (west=42m) and new format (west=n/a, grid=18/24)
    def _parse_qa_line(prefix, text):
        m = re.search(prefix + r"\s*west=(\d+|n/a)m?\s+center=(\d+|n/a)m?\s+east=(\d+|n/a)m?\s+"
                       r"(?:north=([+\-]?\d+)m?\s+)?patch=(\d+)m\s+"
                       r"(?:grid=(\d+)/(\d+)\s+)?"
                       r"stable_iou=([\d.]+)\s+score=(\d+)", text)
        if not m:
            return None
        def _int_or_none(s):
            return int(s) if s and s != "n/a" else None
        qa = {
            "west": _int_or_none(m.group(1)),
            "center": _int_or_none(m.group(2)),
            "east": _int_or_none(m.group(3)),
            "north": _int_or_none(m.group(4)),
            "patch_med": int(m.group(5)),
            "stable_iou": float(m.group(8)),
            "score": int(m.group(9)),
        }
        if m.group(6) is not None:
            qa["grid_valid"] = int(m.group(6))
            qa["grid_total"] = int(m.group(7))
        return qa

    grid_qa = _parse_qa_line(r"Grid warp QA:", log_text)
    if grid_qa:
        result["grid_qa"] = grid_qa

    tps_qa = _parse_qa_line(r"TPS fallback QA:", log_text)
    if tps_qa:
        result["tps_qa"] = tps_qa

    # Cross-validation
    m = re.search(r"Cross-validation:\s*fit=([\d.]+)m,\s*CV=([\d.]+)m", log_text)
    if m:
        result["cv"] = {"fit_m": float(m.group(1)), "cv_m": float(m.group(2))}

    # Grid independent QA
    m = re.search(r"Grid independent QA:\s*total=([\d.]+).*?confidence=([\d.]+).*?accepted=(True|False)", log_text)
    if m:
        result["grid_independent"] = {
            "total": float(m.group(1)),
            "confidence": float(m.group(2)),
            "accepted": m.group(3) == "True",
        }

    # TPS independent QA
    m = re.search(r"TPS independent QA:\s*total=([\d.]+).*?confidence=([\d.]+).*?accepted=(True|False)", log_text)
    if m:
        result["tps_independent"] = {
            "total": float(m.group(1)),
            "confidence": float(m.group(2)),
            "accepted": m.group(3) == "True",
        }

    # Selected candidate
    m = re.search(r"Grid optimizer wins|Selected candidate.*?accepted=(True|False)", log_text)
    if m:
        # Determine from the QA lines which was selected
        pass
    m = re.search(r"(Grid optimizer|TPS fallback) wins", log_text)
    if m:
        result["selected"] = "grid" if "Grid" in m.group(1) else "tps"

    # Warnings
    if re.search(r"cross_validation_high", log_text):
        result["warnings"].append("cross_validation_high")
    for m in re.finditer(r"WARNING:\s+(.+)", log_text):
        result["warnings"].append(m.group(1).strip())

    return result


def _run_ground_truth_eval(output_path):
    """Run ground-truth evaluation if a GT reference is configured.

    Checks data/local_paths.yaml for a 'ground_truths' section. Returns
    a dict of GT metrics or None if no GT is configured/available.
    """
    try:
        gt_config_path = PROJECT_ROOT / "data" / "local_paths.yaml"
        if not gt_config_path.exists():
            return None

        import yaml
        with open(gt_config_path) as f:
            paths_cfg = yaml.safe_load(f) or {}

        ground_truths = paths_cfg.get("ground_truths", {})
        if not ground_truths:
            return None

        # Use the first configured GT (typically the primary test case)
        gt_name = next(iter(ground_truths))
        gt_path_raw = ground_truths[gt_name]
        gt_path = os.path.expanduser(gt_path_raw)
        if not os.path.exists(gt_path):
            print(f"  GT reference not found: {gt_path}")
            return None

        from scripts.test.eval_ground_truth import evaluate_ground_truth
        print(f"\n=== Ground-truth evaluation ({gt_name}) ===")
        result = evaluate_ground_truth(
            str(output_path), gt_path, eval_res=8.0)

        if "error" in result:
            print(f"  GT eval error: {result['error']}")
            return None

        # Return compact summary for summary.json
        return {
            "gt_name": gt_name,
            "oracle_median_m": result.get("oracle_median_m"),
            "oracle_mean_m": result.get("oracle_mean_m"),
            "oracle_p90_m": result.get("oracle_p90_m"),
            "oracle_patch_count": result.get("oracle_patch_count"),
            "grid_median_m": result.get("grid", {}).get("grid_median_m"),
            "coastal_median_m": result.get("coastal", {}).get("median_m"),
            "inland_median_m": result.get("inland", {}).get("median_m"),
        }
    except Exception as e:
        print(f"  GT evaluation failed: {e}")
        return None


def build_summary(version, run_dir, log_text, exit_code, wall_clock_s, qa_path):
    """Build the summary.json structure from parsed log and qa.json."""
    parsed = parse_log(log_text)

    # Get git info
    git_commit = "unknown"
    git_dirty = False
    code_state_path = run_dir / "code_state.txt"
    if code_state_path.exists():
        cs = code_state_path.read_text()
        m = re.search(r"commit:\s*(\w+)", cs)
        if m:
            git_commit = m.group(1)[:7]
        git_dirty = "changed" in cs or "modified" in cs or bool(re.search(r"\d+ file", cs))

    # Load qa.json if it exists
    qa_data = {}
    if qa_path.exists():
        try:
            qa_data = json.loads(qa_path.read_text())
        except Exception:
            parsed["errors"].append(f"Failed to parse {qa_path}")

    # Extract grid and tps reports from qa.json
    grid_report = {}
    tps_report = {}
    selected = parsed.get("selected")
    for report in qa_data.get("reports", []):
        if report.get("candidate") == "grid":
            grid_report = report
        elif report.get("candidate") == "tps":
            tps_report = report
    if not selected:
        selected = qa_data.get("selected_candidate", "grid")

    def report_to_summary(report):
        if not report:
            return None
        im = report.get("image_metrics", {})
        s = {
            "score": round(im.get("score", 0), 1),
            "total": round(report.get("total_score", 0), 1),
            "accepted": report.get("accepted", False),
            "west": round(im["west"]) if im.get("west") is not None else None,
            "center": round(im["center"]) if im.get("center") is not None else None,
            "east": round(im["east"]) if im.get("east") is not None else None,
            "north": round(im["north_shift"]) if im.get("north_shift") is not None else None,
            "patch_med": round(im.get("patch_med", 0)),
            "patch_p90": round(im.get("patch_p90", 0), 1) if im.get("patch_p90") is not None else None,
            "patch_count": im.get("patch_count"),
            "stable_iou": round(im.get("stable_iou", 0), 3),
            "shore_iou": round(im.get("shore_iou", 0), 3),
            "grid_score": round(im.get("grid_score", 0), 1) if im.get("grid_score") is not None else None,
            "grid_coverage": im.get("grid_coverage"),
            "quality_grade": report.get("quality_grade", "?"),
            "stable_boundary_m": round(im["stable_boundary_m"], 1) if im.get("stable_boundary_m") is not None else None,
            "shore_boundary_m": round(im["shore_boundary_m"], 1) if im.get("shore_boundary_m") is not None else None,
        }
        if im.get("grid"):
            s["grid"] = {
                "valid_count": im["grid"].get("valid_count"),
                "total_count": im["grid"].get("total_count"),
            }
        if im.get("score_breakdown"):
            s["score_breakdown"] = im["score_breakdown"]
        return s

    grid_summary = report_to_summary(grid_report)
    tps_summary = report_to_summary(tps_report)

    # Fallback: use parsed log scores when qa.json is missing/incomplete
    if grid_summary is None and parsed.get("grid_qa"):
        gq = parsed["grid_qa"]
        grid_summary = {
            "score": gq["score"], "total": gq["score"],
            "accepted": True, "west": gq.get("west"), "center": gq.get("center"),
            "east": gq.get("east"), "north": gq.get("north"),
            "patch_med": gq["patch_med"], "patch_p90": None,
            "patch_count": None, "stable_iou": gq["stable_iou"],
            "shore_iou": None,
        }
    if tps_summary is None and parsed.get("tps_qa"):
        tq = parsed["tps_qa"]
        tps_summary = {
            "score": tq["score"], "total": tq["score"],
            "accepted": True, "west": tq.get("west"), "center": tq.get("center"),
            "east": tq.get("east"), "north": tq.get("north"),
            "patch_med": tq["patch_med"], "patch_p90": None,
            "patch_count": None, "stable_iou": tq["stable_iou"],
            "shore_iou": None,
        }

    # Rejection reasons from qa.json
    for report in qa_data.get("reports", []):
        for reason in report.get("reasons", []):
            if reason not in parsed["warnings"]:
                parsed["warnings"].append(reason)

    # CV and GCP info from qa.json
    cv_mean_m = None
    gcp_count = None
    coverage = None
    if grid_report:
        cv_mean_m = grid_report.get("cv_mean_m")
        coverage = grid_report.get("coverage")
    gcp_count = qa_data.get("metadata", {}).get("gcp_count")

    # Override from parsed log if available
    if parsed["gcp_selection"]["count"]:
        gcp_count = parsed["gcp_selection"]["count"]
    if parsed["gcp_selection"]["coverage"]:
        coverage = parsed["gcp_selection"]["coverage"]

    summary = {
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "exit_code": exit_code,
        "wall_clock_s": round(wall_clock_s, 1),
        "grid": grid_summary,
        "tps": tps_summary,
        "selected": selected,
        "cv_mean_m": round(cv_mean_m, 1) if cv_mean_m is not None else None,
        "gcp_count": gcp_count,
        "coverage": round(coverage, 3) if coverage is not None else None,
        "anchors": parsed["anchors"],
        "auto_gcps": parsed["auto_gcps"],
        "flow": parsed["flow"],
        "grid_optimizer": parsed["grid_optimizer"],
        "reclamation": parsed["reclamation"],
        "icp": parsed["icp"],
        "step_timings": parsed["step_timings"],
        "anchor_details": parsed["anchor_details"],
        "warnings": parsed["warnings"],
        "errors": parsed["errors"],
    }

    # Load hierarchical profile if available
    profile_path = run_dir / "profile.json"
    if profile_path.exists():
        try:
            summary["profile"] = json.loads(profile_path.read_text())
        except Exception:
            pass

    return summary


def run_pipeline(version, timeout, scene_cfg: dict):
    """Run the alignment pipeline and capture everything."""
    run_dir = _diagnostics_root_for_scene(scene_cfg["name"]) / f"run_v{version}"
    run_dir.mkdir(parents=True, exist_ok=True)

    qa_path = run_dir / "qa.json"
    log_path = run_dir / "run.log"
    stderr_path = run_dir / "stderr.log"
    summary_path = run_dir / "summary.json"
    output_path = run_dir / "output.tif"

    # Snapshot git state
    snapshot_git_state(run_dir / "code_state.txt")

    # Build (or reuse cached) per-segment ortho target before running the
    # alignment. Falls back to legacy pre-stitched TIFF if DECLASS_USE_PRESTITCHED=1.
    target_path = ensure_target_built(scene_cfg)
    reference_path = scene_cfg["reference_path"]

    # Pre-flight: confirm target + reference actually decode to non-NoData
    # pixels. This catches the ASP image_mosaic "silent unfinalized TIFF"
    # failure mode (zeroed TileOffsets → everything decodes as NoData) and
    # similar broken caches before we burn 20-40 min on a doomed alignment.
    from preprocess.stitch import verify_tiff_decodes_nonempty
    if not verify_tiff_decodes_nonempty(target_path, label="run_test TARGET"):
        print(f"\nERROR: target {target_path} is not readable. Delete it "
              f"(and any cached siblings) and re-run.", file=sys.stderr)
        sys.exit(2)
    if not verify_tiff_decodes_nonempty(reference_path, label="run_test REFERENCE"):
        print(f"\nERROR: reference {reference_path} is not readable.", file=sys.stderr)
        sys.exit(2)

    cmd = [
        sys.executable, str(PROJECT_ROOT / "auto-align.py"),
        target_path,
        "-r", reference_path,
        "--anchors", ANCHORS,
        "-y", "--best",
        "--diagnostics-dir", str(run_dir) + "/",
        "--qa-json", str(qa_path),
        "-o", str(output_path),
    ]

    print(f"=== run_test.py: Starting v{version} ===")
    print(f"Output dir: {run_dir}")
    print(f"Command: {' '.join(cmd[:5])}...")
    print(f"Timeout: {timeout}s")
    print()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    start = time.time()
    log_lines = []
    exit_code = -1

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(PROJECT_ROOT),
            bufsize=1,
        )

        # Read stdout line by line for live streaming (stderr merged in)
        stderr_data = b""
        try:
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    decoded = line.decode("utf-8", errors="replace")
                    sys.stdout.write(decoded)
                    sys.stdout.flush()
                    log_lines.append(decoded)

                # Check timeout
                elapsed = time.time() - start
                if elapsed > timeout:
                    print(f"\n=== TIMEOUT after {elapsed:.0f}s, sending SIGTERM ===")
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        print("=== SIGKILL ===")
                        proc.kill()
                        proc.wait()
                    break

            exit_code = proc.returncode

        except KeyboardInterrupt:
            print("\n=== KeyboardInterrupt, sending SIGTERM ===")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            exit_code = proc.returncode if proc.returncode is not None else -2

    except Exception as e:
        print(f"\n=== Failed to start pipeline: {e} ===")
        log_lines.append(f"\nFailed to start: {e}\n")
        exit_code = -1
        stderr_data = str(e).encode()

    wall_clock_s = time.time() - start

    # Write logs
    log_text = "".join(log_lines)
    log_path.write_text(log_text)

    stderr_text = stderr_data.decode("utf-8", errors="replace") if isinstance(stderr_data, bytes) else ""
    if stderr_text.strip():
        stderr_path.write_text(stderr_text)

    print(f"\n=== Pipeline finished: exit_code={exit_code}, wall_clock={wall_clock_s:.1f}s ===\n")

    # Build and write summary
    if exit_code != 0:
        # Include stderr snippets in errors
        pass

    summary = build_summary(version, run_dir, log_text, exit_code, wall_clock_s, qa_path)

    if exit_code != 0 and stderr_text.strip():
        # Add last 500 chars of stderr as error context
        summary["errors"].append(stderr_text.strip()[-500:])

    # Ground-truth evaluation (if output exists and GT is configured)
    if exit_code == 0 and output_path.exists():
        gt_metrics = _run_ground_truth_eval(output_path)
        if gt_metrics:
            summary["ground_truth"] = gt_metrics

    summary_path.write_text(json.dumps(summary, indent=2))

    # Print summary
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    return summary


def cleanup_old_runs(keep_version, scene_name: str = DEFAULT_SCENE):
    """Remove large files from old runs, keeping only the most recent and the best-scoring.

    Preserves summary.json, qa.json, run.log, code_state.txt, and stderr.log
    in all runs (lightweight). Removes output.tif and other large files from
    runs that are neither the most recent nor the best-scoring.
    """
    diag_dir = _diagnostics_root_for_scene(scene_name)
    run_dirs = {}
    for d in sorted(diag_dir.glob("run_v*")):
        if d.is_dir():
            m = re.search(r"run_v(\d+)$", str(d))
            if m:
                run_dirs[int(m.group(1))] = d

    if len(run_dirs) <= 2:
        return

    # Find best-scoring run (lowest grid score = best)
    best_version = None
    best_score = float("inf")
    for ver, d in run_dirs.items():
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
            grid = summary.get("grid")
            if grid and grid.get("score") is not None:
                score = grid["score"]
                if score < best_score:
                    best_score = score
                    best_version = ver
        except Exception:
            continue

    most_recent = keep_version

    keep = {most_recent}
    if best_version is not None:
        keep.add(best_version)

    # Lightweight files to always preserve
    preserve = {"summary.json", "qa.json", "run.log", "code_state.txt", "stderr.log"}

    freed = 0
    for ver, d in run_dirs.items():
        if ver in keep:
            continue
        for item in d.iterdir():
            if item.name in preserve:
                continue
            size = item.stat().st_size if item.is_file() else 0
            if item.is_file():
                size = item.stat().st_size
                item.unlink()
                freed += size
            elif item.is_dir():
                import shutil
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                shutil.rmtree(item)
                freed += size

    if freed > 0:
        kept_str = ", ".join(f"v{v}" for v in sorted(keep))
        print(f"\n=== Cleanup: freed {freed / 1024 / 1024 / 1024:.1f} GB from old runs "
              f"(kept {kept_str}) ===")
        if best_version is not None:
            print(f"    Best: v{best_version} (score={best_score:.1f}), "
                  f"Most recent: v{most_recent}")


def _available_scenes() -> list[str]:
    """Return scene names whose YAML has the fields run_test.py needs.

    The e2e_configs directory also hosts older configs used by
    ``run_e2e_test.py`` that predate the scene-parameterised runner and
    omit ``entity_id`` / ``reference_name``. Filter those out so the
    argparse ``choices`` list only shows configs this runner can load.
    """
    import yaml

    scenes = []
    for p in sorted(E2E_CONFIG_DIR.glob("*.yaml")):
        try:
            with open(p) as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception:
            continue
        if cfg.get("entity_id") and (cfg.get("reference_name") or cfg.get("reference")):
            scenes.append(p.stem)
    return scenes


def main():
    parser = argparse.ArgumentParser(description="Run declass alignment test (scene-parameterised)")
    parser.add_argument("--scene", default=DEFAULT_SCENE,
                        choices=_available_scenes(),
                        help=(f"Which scene config to run (reads "
                              f"scripts/test/e2e_configs/<scene>.yaml). "
                              f"Default: {DEFAULT_SCENE}."))
    parser.add_argument("--version", "-v", type=int, default=None,
                        help="Version number (default: auto-detect next)")
    parser.add_argument("--timeout", "-t", type=int, default=9000,
                        help="Timeout in seconds (default: 9000 = 150 min)")
    parser.add_argument("--stage", choices=["preprocess", "align", "all"],
                        default="all",
                        help="Which stage to run. 'preprocess' builds the "
                             "per-segment ortho target (download/extract/"
                             "stitch/ortho/crop). 'align' runs auto-align "
                             "on the cached target. 'all' runs both (default).")
    parser.add_argument("--preprocess-matcher", choices=["roma", "nift"],
                        default="roma",
                        help="Matcher backend for preprocessing target generation")
    args = parser.parse_args()

    scene_cfg = _load_scene_config(args.scene)
    print(f"=== Scene: {scene_cfg['name']} (entity {scene_cfg['entity_id']}) ===")
    print(f"    reference: {scene_cfg['reference_path']}")

    version = args.version if args.version is not None else detect_next_version(args.scene)

    if args.stage in ("preprocess", "all"):
        target = ensure_target_built(scene_cfg, args.preprocess_matcher)
        print(f"\n=== Preprocess complete: {target} ===\n")
        if args.stage == "preprocess":
            return

    if args.stage in ("align", "all"):
        run_pipeline(version, args.timeout, scene_cfg)
        cleanup_old_runs(version, args.scene)


if __name__ == "__main__":
    main()
