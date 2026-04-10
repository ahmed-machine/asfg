"""Manifest-driven strip and block processing."""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

from .geo import clear_overlap_cache
from .models import ModelCache, get_torch_device
from .types import AlignmentJob, BlockManifest, StripManifest


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
        device=options.pop("device", "auto"),
        match_res=float(options.pop("match_res", 5.0)),
        tin_tarr_thresh=float(options.pop("tin_tarr_thresh", 1.5)),
        skip_fpp=bool(options.pop("skip_fpp", False)),
        matcher_anchor=options.pop("matcher_anchor", "roma"),
        matcher_dense=options.pop("matcher_dense", "roma"),
        grid_size=int(options.pop("grid_size", 20)),
        grid_iters=int(options.pop("grid_iters", 300)),
        arap_weight=float(options.pop("arap_weight", 1.0)),
        global_search=bool(options.pop("global_search", True)),
        global_search_res=float(options.pop("global_search_res", 40.0)),
        global_search_top_k=int(options.pop("global_search_top_k", 3)),
        metadata_priors_dir=options.pop("metadata_priors_dir", None),
        reference_window=options.pop("reference_window", None),
        mask_provider=options.pop("mask_provider", "coastal_obia"),
        allow_abstain=bool(options.pop("allow_abstain", False)),
        force_global=bool(options.pop("force_global", False)),
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


def run_strip_manifest(path: str, runner, *, model_cache=None):
    """Run all jobs in a strip manifest with shared model cache reuse.

    Jobs are processed sequentially.  After each successful alignment, the
    output's geographic bounds are tracked and used to expand the
    reference_window for subsequent jobs — this progressive anchoring helps
    adjacent frames find the correct position even when they have limited
    overlap with the original reference.
    """
    manifest = load_strip_manifest(path)
    created_cache = False
    if model_cache is None:
        device = get_torch_device(manifest.shared_options.get("device", "auto"))
        model_cache = ModelCache(device)
        created_cache = True

    outputs = []
    aligned_bounds = []  # WGS84 bounds of successfully aligned frames
    try:
        for job in manifest.jobs:
            clear_overlap_cache()  # Each job has different overlap bounds
            args = _job_to_namespace(job, manifest.shared_options)

            # Expand reference_window to include previously aligned frames
            if aligned_bounds and hasattr(args, 'reference_window'):
                args.reference_window = _expand_window_with_aligned(
                    args.reference_window, aligned_bounds)

            try:
                result = runner(args, model_cache=model_cache)
                outputs.append(result)
                # Track aligned output bounds for progressive anchoring.
                # run() returns the output path string; if the frame abstained
                # or failed, the file won't exist.
                output_path = getattr(args, 'output', None)
                if output_path and os.path.exists(output_path):
                    bounds = _get_output_bounds_wgs(output_path)
                    if bounds:
                        aligned_bounds.append(bounds)
            except Exception as e:
                input_name = os.path.basename(getattr(args, 'input', '?'))
                print(f"\n  WARNING: Alignment failed for {input_name}: {e}")
                # Remove partial output to prevent it polluting the mosaic
                output_path = getattr(args, 'output', None)
                if output_path and os.path.exists(output_path):
                    os.remove(output_path)
                    print(f"  Removed partial output: {os.path.basename(output_path)}")
                outputs.append(None)
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
