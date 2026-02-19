#!/usr/bin/env python3
"""
Benchmark harness for auto-align.py optimisation comparison.

Creates a pair of synthetic GeoTIFFs with a known offset and runs the
full pipeline, measuring wall-clock time for each major step.

Usage:
    # Optimised version (current code):
    python benchmark.py

    # To compare against unoptimised, checkout the original and run:
    python benchmark.py --tag baseline
"""

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Synthetic image generation
# ---------------------------------------------------------------------------

def _make_synthetic_image(path, bounds, size, seed, offset_m=(0, 0), crs=None):
    """Write a synthetic GeoTIFF with realistic land/water structure.

    *bounds* is (left, bottom, right, top) in EPSG:3857 meters.
    *offset_m* shifts the image content by (dx, dy) meters to simulate misalignment.
    """
    if crs is None:
        crs = CRS.from_epsg(3857)

    left, bottom, right, top = bounds
    # Apply simulated misalignment by shifting the bounding box
    dx, dy = offset_m
    shift_left = left - dx
    shift_bottom = bottom + dy
    shift_right = right - dx
    shift_top = top + dy

    tf = from_bounds(shift_left, shift_bottom, shift_right, shift_top, size, size)

    rng = np.random.default_rng(seed)

    # Build terrain: low-frequency "land/water" blobs
    lo = rng.integers(20, 60, (size // 16, size // 16), dtype=np.uint8)
    hi = rng.integers(160, 220, (size // 16, size // 16), dtype=np.uint8)
    mask = rng.integers(0, 2, (size // 16, size // 16), dtype=bool)
    base_small = np.where(mask, hi, lo).astype(np.uint8)

    import cv2
    base = cv2.resize(base_small, (size, size), interpolation=cv2.INTER_CUBIC)
    # Add fine-grained texture so feature matchers have something to work with
    texture = rng.integers(0, 30, (size, size), dtype=np.uint8)
    img = np.clip(base.astype(np.int32) + texture, 0, 255).astype(np.uint8)

    with rasterio.open(path, 'w', driver='GTiff',
                       height=size, width=size, count=1, dtype='uint8',
                       crs=crs, transform=tf,
                       compress='lzw') as dst:
        dst.write(img, 1)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(tag="optimised", size=1024, offset_m=150, tmpdir=None):
    """Run the full pipeline on a synthetic image pair, return timing dict."""
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()

    ref_path = os.path.join(tmpdir, "ref.tif")
    inp_path = os.path.join(tmpdir, "input.tif")
    out_path = os.path.join(tmpdir, "output.tif")

    # EPSG:3857 bounds centred roughly on Bahrain for realism
    bounds = (5490000, 2700000, 5540000, 2750000)

    print(f"\n{'='*60}")
    print(f"Benchmark: {tag}")
    print(f"Image size: {size}x{size} px, simulated offset: {offset_m}m")
    print(f"{'='*60}\n")

    print("Generating synthetic test images...")
    t0 = time.perf_counter()
    _make_synthetic_image(ref_path, bounds, size, seed=1)
    _make_synthetic_image(inp_path, bounds, size, seed=2,
                          offset_m=(offset_m, offset_m // 2))
    print(f"  Generated in {time.perf_counter() - t0:.2f}s")

    # Monkey-patch the _timed function to capture per-step timings
    from align import pipeline as pl

    step_times = {}
    original_timed = None

    def _capturing_timed(name, fn, *a, **kw):
        t0 = time.perf_counter()
        result = fn(*a, **kw)
        elapsed = time.perf_counter() - t0
        step_times[name] = elapsed
        print(f"  [{name}] {elapsed:.1f}s")
        return result

    # Patch run() to use our capturing timed wrapper
    original_run = pl.run

    def _patched_run(args, model_cache=None):
        pipeline_t0 = time.perf_counter()

        state = _capturing_timed("setup", pl.step_setup, args)

        if model_cache is not None:
            state.model_cache = model_cache
        elif state.model_cache is None:
            from align.models import ModelCache
            device = pl.get_torch_device(
                override=getattr(args, 'device', None))
            state.model_cache = ModelCache(device)

        try:
            state = _capturing_timed("global_localization", pl.step_global_localization, state, args)
            state = _capturing_timed("coarse_offset", pl.step_coarse_offset, state)
            state = _capturing_timed("handle_large_offset",
                                     pl.step_handle_large_offset, state, args)
            state = _capturing_timed("scale_rotation",
                                     pl.step_scale_rotation, state, args)
            state = _capturing_timed("feature_matching",
                                     pl.step_feature_matching, state, args)
            state = _capturing_timed("validate_and_filter",
                                     pl.step_validate_and_filter, state)
            state = _capturing_timed("select_warp_and_apply",
                                     pl.step_select_warp_and_apply, state)
            _capturing_timed("post_refinement", pl.step_post_refinement, state)
        finally:
            if model_cache is None and state.model_cache is not None:
                state.model_cache.close()

        total = time.perf_counter() - pipeline_t0
        step_times["TOTAL"] = total
        print(f"\n  [TOTAL] {total:.1f}s")
        return state.output_path

    import argparse
    args = argparse.Namespace(
        input=inp_path,
        reference=ref_path,
        output=out_path,
        match_res=5.0,
        coarse_pass=0,
        best=False,
        anchors=None,
        metadata_priors=None,
        metadata_priors_dir=None,
        skip_post_anchor=False,
        yes=True,
        device="auto",
        global_search=True,
        global_search_res=40.0,
        global_search_top_k=3,
        force_global=False,
        reference_window=None,
        mask_provider="coastal_obia",
        qa_json=None,
        diagnostics_dir=None,
        allow_abstain=False,
    )

    wall_t0 = time.perf_counter()
    try:
        _patched_run(args)
    except SystemExit as e:
        if str(e) != "0":
            print(f"Pipeline exited with code {e}")
    wall_elapsed = time.perf_counter() - wall_t0
    step_times["wall_clock"] = wall_elapsed

    # Capture output quality metrics if the output was produced
    quality = {}
    if os.path.exists(out_path):
        with rasterio.open(out_path) as src:
            quality["output_size"] = f"{src.width}x{src.height}"
            quality["crs"] = src.crs.to_epsg() if src.crs else None
        print(f"\nOutput: {quality}")
    else:
        print("\nNo output produced.")

    return {
        "tag": tag,
        "step_times": step_times,
        "quality": quality,
    }


def print_comparison(results):
    """Print a side-by-side comparison table."""
    if len(results) < 2:
        r = results[0]
        print(f"\n{'='*60}")
        print(f"Timings for: {r['tag']}")
        print(f"{'='*60}")
        for step, t in r["step_times"].items():
            print(f"  {step:<30} {t:>8.2f}s")
        return

    a, b = results[0], results[1]
    all_steps = sorted(set(list(a["step_times"].keys()) + list(b["step_times"].keys())))

    print(f"\n{'='*70}")
    print(f"{'Step':<28} {a['tag']:>14}  {b['tag']:>14}  {'Delta':>10}")
    print(f"{'-'*28} {'-'*14}  {'-'*14}  {'-'*10}")

    for step in all_steps:
        ta = a["step_times"].get(step, float('nan'))
        tb = b["step_times"].get(step, float('nan'))
        if ta and tb and not (ta != ta or tb != tb):  # nan check
            delta = tb - ta
            marker = " <" if delta > 0.5 else (" >" if delta < -0.5 else "")
            print(f"  {step:<26} {ta:>12.2f}s  {tb:>12.2f}s  {delta:>+9.2f}s{marker}")
        else:
            ta_s = f"{ta:.2f}s" if ta == ta else "  N/A"
            tb_s = f"{tb:.2f}s" if tb == tb else "  N/A"
            print(f"  {step:<26} {ta_s:>13}  {tb_s:>13}")

    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the auto-align pipeline")
    parser.add_argument("--tag", default="optimised",
                        help="Label for this run (default: optimised)")
    parser.add_argument("--size", type=int, default=1024,
                        help="Synthetic image size in pixels (default: 1024)")
    parser.add_argument("--offset", type=int, default=150,
                        help="Simulated offset in meters (default: 150)")
    parser.add_argument("--results-file", default="benchmark_results.json",
                        help="JSON file to append results to")
    args = parser.parse_args()

    result = run_benchmark(tag=args.tag, size=args.size, offset_m=args.offset)

    # Load existing results for comparison
    existing = []
    if os.path.exists(args.results_file):
        with open(args.results_file) as f:
            existing = json.load(f)

    # Save this result
    existing.append(result)
    with open(args.results_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {args.results_file}")

    print_comparison(existing)
