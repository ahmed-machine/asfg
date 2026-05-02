"""Manifest-driven strip and block processing."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from types import SimpleNamespace

from .geo import clear_overlap_cache
from .models import ModelCache, get_torch_device
from .types import AlignmentJob, BlockManifest, StripManifest


# ---------------------------------------------------------------------------
# Strip-pose extrapolation
# ---------------------------------------------------------------------------
#
# When primary + secondary alignment all fail for a frame whose neighbours
# DID align successfully, we can predict the failed frame's true position
# from the strip-order delta between accepted neighbours. The USGS-corner
# priors are systematically biased (~76 km on Bahrain KH-4B v40 — measured
# DA023 prior lon=49.80 vs aligned truth lon=50.57); but the *delta*
# between adjacent USGS-corner priors tracks the real strip-frame stride,
# so by anchoring at one accepted frame and stepping along the strip, we
# can predict the centroid of any failed frame to within a few hundred
# metres.
#
# Once we have a predicted centroid, we write a small GDAL VRT that
# wraps the original ortho.tif but with its origin shifted to put the
# data at the predicted position. Alignment runs against that VRT
# instead of the misplaced ortho.tif, with a tight reference_window
# around the predicted centroid.

# Match the entity-id strip-suffix patterns we see in catalogs:
#   "DS1104-1057DA023" → strip='DA', idx=23
#   "D3C1213-200346A003" → strip='A', idx=3
#   "DZB00403600089H016001" → strip='H', idx=16001
_ENTITY_STRIP_RE = re.compile(
    r"(?:DS\d+-\d+|D3C\d+-\d+|DZB\d+)([A-Z]+)(\d+)$",
    re.IGNORECASE,
)


def _parse_entity_strip_index(entity_or_path: str) -> tuple[str, int] | None:
    """Extract (strip_suffix, frame_index) from an entity id or filename.

    Returns ``("DA", 23)`` for ``DS1104-1057DA023`` etc., or ``None`` if
    the name doesn't match a known KH catalog pattern. Used to put
    accepted-and-failed frames in strip order for pose extrapolation.
    """
    base = os.path.basename(entity_or_path)
    # Strip common suffixes from input/output filenames. Order matters:
    # longest/composite suffixes go first so we don't strip a prefix of
    # a longer suffix. Repeat the loop until stable so chained suffixes
    # ("_ortho.extrapolated.vrt") collapse fully.
    suffixes = (
        ".vrt", ".tif", ".tiff", ".coarse.tif",
        "_ortho.extrapolated", "_ortho.coarse", "_ortho",
        "_aligned", "_georef", "_stitched", ".extrapolated",
    )
    prev = None
    while base != prev:
        prev = base
        for suffix in suffixes:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
    m = _ENTITY_STRIP_RE.search(base)
    if m is None:
        return None
    return m.group(1).upper(), int(m.group(2))


def _entity_id_from_input(path: str) -> str:
    """Strip artifact suffixes from an input path to recover the entity id."""
    base = os.path.basename(path)
    for suffix in (".tif", ".coarse.tif"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    for suffix in ("_ortho.coarse", "_ortho", "_georef", "_stitched"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


def _aligned_center_wgs84(aligned_path: str) -> tuple[float, float] | None:
    """Return ``(lon, lat)`` of the aligned tif's centroid, or None."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        with rasterio.open(aligned_path) as ds:
            w, s, e, n = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
        return ((w + e) / 2.0, (s + n) / 2.0)
    except Exception:
        return None


def _predict_centroid_from_strip(
    target_idx: int,
    accepted: list[tuple[int, tuple[float, float]]],
) -> tuple[float, float] | None:
    """Predict ``(lon, lat)`` for the frame at ``target_idx`` in strip order.

    *accepted* is a sorted-by-index list of ``(idx, (lon, lat))`` for
    successfully aligned neighbour frames.

    - With ≥2 accepted neighbours: linear extrapolation using the per-frame
      step inferred from the closest two.
    - With 1 accepted neighbour: assume zero-step (just use that centroid).
    - With 0: return None.
    """
    if not accepted:
        return None
    if len(accepted) == 1:
        return accepted[0][1]
    # Pick the two accepted frames closest to target_idx (in strip-order
    # distance) so we extrapolate from local context, not the strip end.
    nearest = sorted(accepted, key=lambda kv: abs(kv[0] - target_idx))[:2]
    nearest.sort(key=lambda kv: kv[0])
    (idx_a, (lon_a, lat_a)), (idx_b, (lon_b, lat_b)) = nearest
    if idx_a == idx_b:
        return (lon_a, lat_a)
    delta_lon = (lon_b - lon_a) / (idx_b - idx_a)
    delta_lat = (lat_b - lat_a) / (idx_b - idx_a)
    # Anchor at whichever neighbour is closest to target
    anchor = min(nearest, key=lambda kv: abs(kv[0] - target_idx))
    a_idx, (a_lon, a_lat) = anchor
    n_steps = target_idx - a_idx
    return (a_lon + delta_lon * n_steps, a_lat + delta_lat * n_steps)


def _shifted_input_vrt(ortho_path: str, target_center_wgs84: tuple[float, float],
                       scratch_dir: str) -> str | None:
    """Write a small GDAL VRT pointing at *ortho_path* but with its origin
    shifted so the data lands at *target_center_wgs84* in WGS84.

    Returns the VRT path on success, ``None`` on failure.
    """
    try:
        import rasterio
        from rasterio.warp import transform as warp_transform
        from osgeo import gdal
    except Exception:
        return None
    try:
        with rasterio.open(ortho_path) as src:
            crs = src.crs
            cb = src.bounds
            cur_x = (cb.left + cb.right) / 2.0
            cur_y = (cb.top + cb.bottom) / 2.0
            xs, ys = warp_transform(
                "EPSG:4326", crs,
                [target_center_wgs84[0]], [target_center_wgs84[1]],
            )
            target_x, target_y = xs[0], ys[0]
        dx = target_x - cur_x
        dy = target_y - cur_y
        if abs(dx) < 1.0 and abs(dy) < 1.0:
            # Nothing to shift — the ortho is already at the predicted spot
            return None
        os.makedirs(scratch_dir, exist_ok=True)
        out_vrt = os.path.join(
            scratch_dir,
            os.path.basename(ortho_path).rsplit(".", 1)[0] + ".extrapolated.vrt",
        )
        # gdal CreateCopy → VRT, then mutate geotransform
        src_ds = gdal.Open(os.path.abspath(ortho_path))
        if src_ds is None:
            return None
        vrt_driver = gdal.GetDriverByName("VRT")
        vrt_ds = vrt_driver.CreateCopy(out_vrt, src_ds)
        if vrt_ds is None:
            return None
        gt = list(vrt_ds.GetGeoTransform())
        gt[0] += dx
        gt[3] += dy
        vrt_ds.SetGeoTransform(gt)
        # Ensure changes are flushed by closing the dataset
        vrt_ds = None
        src_ds = None
        return out_vrt
    except Exception as e:
        print(f"  [Extrapolate] VRT shift failed for {ortho_path}: {e}",
              flush=True)
        return None


def _tight_reference_window(center_wgs84: tuple[float, float],
                            half_lon_deg: float = 0.40,
                            half_lat_deg: float = 0.30) -> str:
    """Build a comma-string reference_window around *center_wgs84*.

    Defaults give a ~80×60 km box at 26°N — wide enough to absorb a
    few-km extrapolation error while still cutting most of a continent-
    scale ESRI reference down to the relevant area.
    """
    lon, lat = center_wgs84
    return (
        f"{lon - half_lon_deg},{lat - half_lat_deg},"
        f"{lon + half_lon_deg},{lat + half_lat_deg}"
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_path_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _parse_jobs(payload, shared):
    """Parse job entries from a manifest payload."""
    raw_jobs = payload.get("jobs") or payload.get("frames") or []
    jobs = []
    for item in raw_jobs:
        jobs.append(
            AlignmentJob(
                input_path=item["input"],
                reference_path=item.get("reference") or shared.get("reference"),
                output_path=item.get("output"),
                anchors_path=item.get("anchors") or shared.get("anchors"),
                metadata_priors=_normalize_path_list(item.get("metadata_priors") or shared.get("metadata_priors")),
                qa_json_path=item.get("qa_json") or shared.get("qa_json"),
                diagnostics_dir=item.get("diagnostics_dir") or shared.get("diagnostics_dir"),
                options={
                    key: value
                    for key, value in item.items()
                    if key not in {"input", "reference", "output", "anchors", "metadata_priors", "qa_json", "diagnostics_dir"}
                },
            )
        )
    return jobs


def _job_to_namespace(job: AlignmentJob, shared_options: dict):
    options = dict(shared_options)
    options.update(job.options)
    arap_weight = options.pop("arap_weight", None)
    return SimpleNamespace(
        input=job.input_path,
        reference=job.reference_path,
        output=job.output_path,
        anchors=job.anchors_path,
        metadata_priors=job.metadata_priors,
        qa_json=job.qa_json_path,
        diagnostics_dir=job.diagnostics_dir,
        coarse_pass=int(options.pop("coarse_pass", 0)),
        yes=bool(options.pop("yes", True)),
        best=bool(options.pop("best", False)),
        device=options.pop("device", "auto"),
        match_res=float(options.pop("match_res", 5.0)),
        tin_tarr_thresh=float(options.pop("tin_tarr_thresh", 1.5)),
        skip_fpp=bool(options.pop("skip_fpp", False)),
        matcher_anchor=options.pop("matcher_anchor", "roma"),
        matcher_dense=options.pop("matcher_dense", "roma"),
        profile=options.pop("profile", None),
        grid_size=int(options.pop("grid_size", 20)),
        grid_iters=int(options.pop("grid_iters", 300)),
        arap_weight=None if arap_weight is None else float(arap_weight),
        global_search=bool(options.pop("global_search", True)),
        global_search_res=float(options.pop("global_search_res", 40.0)),
        global_search_top_k=int(options.pop("global_search_top_k", 3)),
        metadata_priors_dir=options.pop("metadata_priors_dir", None),
        reference_window=options.pop("reference_window", None),
        mask_provider=options.pop("mask_provider", "coastal_obia"),
        allow_abstain=bool(options.pop("allow_abstain", False)),
        tps_fallback=bool(options.pop("tps_fallback", False)),
        force_global=bool(options.pop("force_global", False)),
        secondary_references=_normalize_path_list(options.pop("secondary_references", [])),
        resume_from_checkpoint=options.pop("resume_from_checkpoint", None),
        keep_temp_paths=bool(options.pop("keep_temp_paths", False)),
    )


# ---------------------------------------------------------------------------
# Strip manifests
# ---------------------------------------------------------------------------

def load_strip_manifest(path: str) -> StripManifest:
    """Load a strip manifest JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    shared = dict(payload.get("shared", {}))
    jobs = _parse_jobs(payload, shared)
    return StripManifest(manifest_path=path, jobs=jobs, shared_options=shared)


def _get_output_bounds_wgs(output_path: str):
    """Read an aligned output and return its WGS84 bounds."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        with rasterio.open(output_path) as ds:
            return transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
    except Exception:
        return None


def _expand_window_with_aligned(current_window, aligned_bounds_list, margin_deg=0.18):
    """Expand a reference_window string to include aligned frame bounds.

    This gives subsequent frames in a strip access to the geographic context
    of already-aligned adjacent frames, preventing them from searching only
    the original reference footprint.
    """
    if not aligned_bounds_list:
        return current_window

    # Parse current window
    if isinstance(current_window, str) and current_window:
        parts = [float(p.strip()) for p in current_window.split(",") if p.strip()]
        if len(parts) == 4:
            win = list(parts)
        else:
            return current_window
    elif isinstance(current_window, (tuple, list)) and len(current_window) == 4:
        win = list(current_window)
    else:
        return current_window

    # Expand to include all aligned frame bounds
    for bounds in aligned_bounds_list:
        w, s, e, n = bounds
        win[0] = min(win[0], w - margin_deg)
        win[1] = min(win[1], s - margin_deg)
        win[2] = max(win[2], e + margin_deg)
        win[3] = max(win[3], n + margin_deg)

    return f"{win[0]},{win[1]},{win[2]},{win[3]}"


def _qa_accepts_output(qa_json_path: str | None) -> bool:
    if not qa_json_path or not os.path.exists(qa_json_path):
        return False
    try:
        with open(qa_json_path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    selected = payload.get("selected_candidate")
    reports = payload.get("reports") or []
    if selected:
        for report in reports:
            if report.get("candidate") == selected:
                return bool(report.get("accepted", False))
    return any(bool(report.get("accepted", False)) for report in reports)


def _profile_has_unreliable_usgs(profile_name: str | None) -> bool:
    """Return True when the named profile flags `usgs_corners_reliable: false`.

    Used by the strip-manifest dispatcher to decide whether to try
    pose-extrapolation from accepted neighbours BEFORE wasting time on
    a primary alignment that is unlikely to converge (KH-4A/4B/7 USGS
    catalog corners are 10-30 km off; on Bahrain v40 measurements showed
    ~76 km systematic west bias, which makes RoMa scattering on the
    coarse stage and matching collapse below the RANSAC viability floor).
    """
    if not profile_name:
        return False
    try:
        from .params import load_profile
        params = load_profile(profile_name)
        return not bool(params.camera.usgs_corners_reliable)
    except Exception:
        return False


def _alignment_attempt(args, runner, *, model_cache, reference_override: str | None,
                       label: str):
    """Run one alignment attempt with optional reference override.

    Returns ``(result, output_exists, accepted)`` where ``result`` is whatever
    the runner returned (typically the output path) and ``output_exists`` /
    ``accepted`` reflect on-disk state after the run.
    """
    if reference_override is not None:
        attempt_args = SimpleNamespace(**vars(args))
        attempt_args.reference = reference_override
        ref_label = os.path.basename(reference_override)
    else:
        attempt_args = args
        ref_label = "primary"
    print(f"  [Align:{label}] reference={ref_label}", flush=True)
    try:
        result = runner(attempt_args, model_cache=model_cache)
    except Exception as e:
        input_name = os.path.basename(getattr(attempt_args, 'input', '?'))
        print(f"  [Align:{label}] {input_name}: {type(e).__name__}: {e}",
              flush=True)
        # Remove partial output so a later attempt isn't confused by it
        out = getattr(attempt_args, 'output', None)
        if out and os.path.exists(out):
            try:
                os.remove(out)
            except OSError:
                pass
        return None, False, False
    out = getattr(attempt_args, 'output', None)
    output_exists = bool(out and os.path.exists(out))
    accepted = output_exists and _qa_accepts_output(
        getattr(attempt_args, 'qa_json', None))
    return result, output_exists, accepted


def run_strip_manifest(path: str, runner, *, model_cache=None):
    """Run all jobs in a strip manifest with shared model cache reuse.

    Jobs are processed sequentially.  After each successful alignment, the
    output's geographic bounds are tracked and used to expand the
    reference_window for subsequent jobs — this progressive anchoring helps
    adjacent frames find the correct position even when they have limited
    overlap with the original reference.

    If the primary alignment fails (or its QA refuses to accept the output),
    each registered secondary reference is tried in turn. Secondaries are
    typically prior accepted aligned outputs from the same strip — they share
    the source camera's panchromatic film aesthetic and so survive cross-modal
    matching better than a modern RGB basemap.  The first accepted attempt
    wins; if all attempts fail, the entity is recorded as unaligned.
    """
    manifest = load_strip_manifest(path)
    created_cache = False
    if model_cache is None:
        device = get_torch_device(manifest.shared_options.get("device", "auto"))
        model_cache = ModelCache(device)
        created_cache = True

    outputs = []
    aligned_bounds = []  # WGS84 bounds of successfully aligned frames
    accepted_outputs = []  # accepted aligned outputs usable as strip-neighbour refs
    job_idx_by_entity: dict[str, int] = {}  # for strip-pose extrapolation
    accepted_centers: list[tuple[int, tuple[float, float]]] = []  # (strip_idx, (lon, lat))
    try:
        for job in manifest.jobs:
            clear_overlap_cache()  # Each job has different overlap bounds
            args = _job_to_namespace(job, manifest.shared_options)

            # Idempotency: if the output already exists and its QA accepted
            # the prior alignment, skip the work and register it as accepted.
            existing_out = getattr(args, 'output', None)
            existing_qa = getattr(args, 'qa_json', None)
            if (existing_out and os.path.exists(existing_out)
                    and _qa_accepts_output(existing_qa)):
                print(
                    f"  [Align] Reusing accepted prior output for "
                    f"{os.path.basename(args.input)}",
                    flush=True,
                )
                outputs.append(existing_out)
                bounds = _get_output_bounds_wgs(existing_out)
                if bounds:
                    aligned_bounds.append(bounds)
                    accepted_outputs.append(existing_out)
                parsed = _parse_entity_strip_index(args.input)
                center = _aligned_center_wgs84(existing_out)
                if parsed and center:
                    accepted_centers.append((parsed[1], center))
                continue

            # Expand reference_window to include previously aligned frames
            if aligned_bounds and hasattr(args, 'reference_window'):
                args.reference_window = _expand_window_with_aligned(
                    args.reference_window, aligned_bounds)
            if accepted_outputs:
                existing = list(getattr(args, "secondary_references", []) or [])
                # Closest neighbours first (most recent two)
                args.secondary_references = existing + accepted_outputs[-2:][::-1]

            # 0) Pose-extrapolation pre-attempt — when the profile flags USGS
            # corners as unreliable AND we have accepted neighbour poses, try
            # the extrapolated VRT first. This skips the 30-min doomed primary
            # attempt for the common KH-4 strip-end case (Bahrain DA025/26
            # measured 76 km systematic west bias; primary attempts produce
            # 0-2 dense matches even with cross-modal MatchAnything).
            parsed = _parse_entity_strip_index(args.input)
            profile_name = getattr(args, "profile", None)
            extrap_used = False
            extrap_input = None
            if (parsed and accepted_centers
                    and _profile_has_unreliable_usgs(profile_name)):
                predicted = _predict_centroid_from_strip(parsed[1], accepted_centers)
                if predicted:
                    scratch_dir = os.path.join(
                        os.path.dirname(manifest.manifest_path) or ".",
                        "extrapolated",
                    )
                    extrap_input = _shifted_input_vrt(
                        args.input, predicted, scratch_dir)
                    if extrap_input:
                        print(
                            f"  [Strip extrapolation] Pre-attempt for "
                            f"{os.path.basename(args.input)} @ predicted="
                            f"({predicted[0]:.4f},{predicted[1]:.4f})",
                            flush=True,
                        )
                        original_input = args.input
                        original_window = getattr(args, "reference_window", None)
                        args.input = extrap_input
                        args.reference_window = _tight_reference_window(predicted)
                        result, _, accepted = _alignment_attempt(
                            args, runner, model_cache=model_cache,
                            reference_override=None, label="extrapolated")
                        if accepted:
                            extrap_used = True
                        else:
                            # Restore for the regular primary path below
                            args.input = original_input
                            args.reference_window = original_window

            # 1) Primary attempt (skipped when extrapolation already accepted)
            if not extrap_used:
                result, _, accepted = _alignment_attempt(
                    args, runner, model_cache=model_cache,
                    reference_override=None, label="primary")
            chosen_ref = None if accepted else None  # set on success below

            # 2) Secondary fallback chain — only if primary did not accept
            if not accepted:
                secondaries = list(getattr(args, "secondary_references", []) or [])
                # Filter out non-existent and any input == output sanity case
                secondaries = [s for s in secondaries
                               if s and os.path.exists(s) and s != args.output]
                if secondaries:
                    print(
                        f"  [Align] Primary did not accept; "
                        f"falling back through {len(secondaries)} "
                        f"secondary reference(s)...",
                        flush=True,
                    )
                for sec_ref in secondaries:
                    clear_overlap_cache()  # different ref = different overlap
                    sec_result, _, sec_accepted = _alignment_attempt(
                        args, runner, model_cache=model_cache,
                        reference_override=sec_ref, label="secondary")
                    if sec_accepted:
                        result = sec_result
                        accepted = True
                        chosen_ref = sec_ref
                        print(
                            f"  [Align] Accepted via secondary "
                            f"{os.path.basename(sec_ref)}",
                            flush=True,
                        )
                        break

            outputs.append(result if accepted else None)

            # Track output for next-frame chaining only if accepted
            if accepted:
                output_path = getattr(args, 'output', None)
                if output_path and os.path.exists(output_path):
                    bounds = _get_output_bounds_wgs(output_path)
                    if bounds:
                        aligned_bounds.append(bounds)
                        accepted_outputs.append(output_path)
                    # Also remember strip position for pose extrapolation
                    parsed = _parse_entity_strip_index(args.input)
                    center = _aligned_center_wgs84(output_path)
                    if parsed and center:
                        accepted_centers.append((parsed[1], center))
            else:
                input_name = os.path.basename(getattr(args, 'input', '?'))
                print(f"\n  WARNING: All alignment attempts failed for {input_name}",
                      flush=True)
                output_path = getattr(args, 'output', None)
                if output_path and os.path.exists(output_path):
                    # Preserve the artifact for inspection but rename so the
                    # mosaic glob (``*_aligned.tif``) skips it. Cross-temporal
                    # cases (e.g. KH-4B 1968 vs modern ESRI pan, 55+ yr gap)
                    # commonly produce geometrically-sound alignments
                    # (patch_med ~50-150 m, comparable to accepted neighbours)
                    # that nonetheless fail the QA score threshold because the
                    # grid_contrib saturates on long era gaps. Keeping the tif
                    # under a ``_aligned_rejected.tif`` name lets the user
                    # inspect the warp manually; ``qa.json`` already records
                    # ``accepted: false`` so automated downstream stages stay
                    # filtered.
                    rejected_path = (
                        output_path[: -len("_aligned.tif")] + "_aligned_rejected.tif"
                        if output_path.endswith("_aligned.tif")
                        else output_path + ".rejected"
                    )
                    try:
                        if os.path.exists(rejected_path):
                            os.remove(rejected_path)
                        os.rename(output_path, rejected_path)
                        print(
                            f"  Renamed rejected output: "
                            f"{os.path.basename(output_path)} → "
                            f"{os.path.basename(rejected_path)}",
                            flush=True,
                        )
                    except OSError as e:
                        print(
                            f"  Could not rename rejected output ({e}); "
                            f"removing instead",
                            flush=True,
                        )
                        try:
                            os.remove(output_path)
                        except OSError:
                            pass

        # Strip-pose extrapolation pass: retry failed frames using accepted
        # neighbours' poses as a prior. Useful when USGS-corner georefs are
        # systematically off (KH-4B / KH-7) and matching against the modern
        # primary reference depends on the input being roughly in the right
        # spot. See `_predict_centroid_from_strip` and `_shifted_input_vrt`.
        if accepted_centers and any(o is None for o in outputs):
            n_failed = sum(1 for o in outputs if o is None)
            print(
                f"\n  [Strip extrapolation] {n_failed} frame(s) failed and "
                f"{len(accepted_centers)} accepted; attempting pose-prior retry",
                flush=True,
            )
            accepted_centers.sort(key=lambda kv: kv[0])
            scratch_dir = os.path.join(
                os.path.dirname(manifest.manifest_path) or ".",
                "extrapolated",
            )
            for i, (job, current) in enumerate(zip(manifest.jobs, outputs)):
                if current is not None:
                    continue
                args = _job_to_namespace(job, manifest.shared_options)
                parsed = _parse_entity_strip_index(args.input)
                if parsed is None:
                    print(
                        f"  [Strip extrapolation] Skipping "
                        f"{os.path.basename(args.input)}: cannot parse strip index",
                        flush=True,
                    )
                    continue
                strip_idx = parsed[1]
                predicted = _predict_centroid_from_strip(strip_idx, accepted_centers)
                if predicted is None:
                    continue
                # Build a shifted VRT so the input's CRS positions data at
                # the predicted centroid, then narrow the reference_window.
                shifted = _shifted_input_vrt(args.input, predicted, scratch_dir)
                if shifted is None:
                    print(
                        f"  [Strip extrapolation] {os.path.basename(args.input)}: "
                        f"VRT shift skipped (already at predicted spot or write failed)",
                        flush=True,
                    )
                    continue
                args.input = shifted
                args.reference_window = _tight_reference_window(predicted)
                args.global_search = True
                # Drop secondary references on the retry — extrapolation means
                # the primary reference should overlap correctly now.
                args.secondary_references = []
                clear_overlap_cache()
                print(
                    f"  [Strip extrapolation] retry {os.path.basename(job['input'])} "
                    f"@ predicted=({predicted[0]:.4f},{predicted[1]:.4f}) "
                    f"window={args.reference_window}",
                    flush=True,
                )
                result, _, accepted_now = _alignment_attempt(
                    args, runner, model_cache=model_cache,
                    reference_override=None, label="extrapolated")
                if accepted_now:
                    print(
                        f"  [Strip extrapolation] Accepted {os.path.basename(job['input'])}",
                        flush=True,
                    )
                    outputs[i] = result
                    output_path = getattr(args, 'output', None)
                    if output_path and os.path.exists(output_path):
                        b = _get_output_bounds_wgs(output_path)
                        if b:
                            aligned_bounds.append(b)
                            accepted_outputs.append(output_path)
                        c = _aligned_center_wgs84(output_path)
                        if c:
                            accepted_centers.append((strip_idx, c))
                            accepted_centers.sort(key=lambda kv: kv[0])
                else:
                    print(
                        f"  [Strip extrapolation] Still failed for "
                        f"{os.path.basename(job['input'])}",
                        flush=True,
                    )
    finally:
        if created_cache:
            model_cache.close()
    return outputs


# ---------------------------------------------------------------------------
# Block manifests
# ---------------------------------------------------------------------------

def load_block_manifest(path: str) -> BlockManifest:
    """Load a block manifest JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    shared = dict(payload.get("shared", {}))
    jobs = _parse_jobs(payload, shared)
    return BlockManifest(manifest_path=path, jobs=jobs, shared_options=shared)


def run_block_manifest(path: str, runner, *, model_cache=None):
    """Run all jobs or strip manifests in a block manifest."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    shared = dict(payload.get("shared", {}))
    strip_manifests = payload.get("strips", [])

    created_cache = False
    if model_cache is None:
        device = get_torch_device(shared.get("device", "auto"))
        model_cache = ModelCache(device)
        created_cache = True

    outputs = []
    try:
        for strip in strip_manifests:
            strip_path = strip if isinstance(strip, str) else strip.get("manifest")
            if not strip_path:
                continue
            if not os.path.isabs(strip_path):
                strip_path = os.path.join(os.path.dirname(path), strip_path)
            outputs.extend(run_strip_manifest(strip_path, runner, model_cache=model_cache))

        if payload.get("jobs"):
            manifest = load_block_manifest(path)
            for job in manifest.jobs:
                args = _job_to_namespace(job, manifest.shared_options)
                outputs.append(runner(args, model_cache=model_cache))
    finally:
        if created_cache:
            model_cache.close()
    return outputs
