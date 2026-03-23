#!/usr/bin/env python3
"""
A/B test: Anchors vs No-Anchors alignment.

Runs shared pipeline steps (setup, global localization, coarse offset,
coarse translation, scale/rotation) once, then forks into two variants:
  A: with anchor GCPs (current behavior)
  B: no anchors (purely automatic RoMa + DINOv3 + SEA-RAFT pipeline)

Handles large offsets (>2km) inline without the recursive run() call,
so shared steps always execute exactly once.

Saves results to diagnostics/run_vN_AB/ with shared-step outputs in
the root and variant-specific outputs in A/ and B/ subdirectories.

Usage:
    python3 scripts/test/run_ab_test.py [--version N] [--timeout 6000]
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import rasterio

from align.pipeline import (
    step_setup,
    step_global_localization,
    step_coarse_offset,
    step_scale_rotation,
    step_feature_matching,
    step_validate_and_filter,
    step_select_warp_and_apply,
    step_post_refinement,
)
from align.errors import AlreadyAlignedError
from align.models import ModelCache
from align.geo import get_torch_device
from align.profiler import PipelineProfiler

from scripts.paths_config import get_target, get_reference

TARGET = get_target("bahrain_1977")
REFERENCE = get_reference("kh9_dzb1212")
ANCHORS = str(PROJECT_ROOT / "data" / "bahrain_anchor_gcps.json")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TeeStream:
    """Tee stdout/stderr to a log file with continuous flushing."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def fileno(self):
        return self.stream.fileno()

    def isatty(self):
        return self.stream.isatty()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_next_version():
    """Scan diagnostics/run_v*/ and return the next version number."""
    diag_dir = PROJECT_ROOT / "diagnostics"
    existing = glob(str(diag_dir / "run_v*"))
    versions = []
    for d in existing:
        m = re.search(r"run_v(\d+)", d)
        if m:
            versions.append(int(m.group(1)))
    return max(versions) + 1 if versions else 1


def build_args():
    """Build an args namespace mimicking auto-align.py CLI."""
    return SimpleNamespace(
        input=TARGET,
        reference=REFERENCE,
        output=None,  # set per variant
        yes=True,
        best=True,
        match_res=5.0,
        anchors=ANCHORS,  # will be overridden per variant
        coarse_pass=0,
        device="auto",
        tin_tarr_thresh=1.5,
        skip_fpp=False,
        matcher_anchor="roma",
        matcher_dense="roma",
        mask_provider="coastal_obia",
        global_search=True,
        global_search_res=40.0,
        global_search_top_k=3,
        force_global=False,
        reference_window=None,
        metadata_priors=None,
        metadata_priors_dir=None,
        qa_json=None,  # set per variant
        diagnostics_dir=None,  # set per variant
        allow_abstain=False,
        tps_fallback=False,
        grid_size=20,
        grid_iters=300,
        arap_weight=1.0,
    )


def _apply_coarse_translation(state, output_path):
    """Apply coarse offset via gdal_translate without recursive refinement."""
    src = rasterio.open(state.current_input)
    left = src.bounds.left - state.coarse_dx
    bottom = src.bounds.bottom + state.coarse_dy
    right = src.bounds.right - state.coarse_dx
    top = src.bounds.top + state.coarse_dy
    src.close()
    cmd = [
        "gdal_translate",
        "-a_ullr", f"{left:.6f}", f"{top:.6f}", f"{right:.6f}", f"{bottom:.6f}",
        "-co", "COMPRESS=LZW", "-co", "PREDICTOR=2", "-co", "TILED=YES",
        state.current_input, output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gdal_translate failed:\n{result.stderr}")
    print(f"  Translated by ({state.coarse_dx:+.0f}m, {state.coarse_dy:+.0f}m) -> {os.path.basename(output_path)}")


def _fork_state(state, anchors_path, output_dir, device):
    """Create a variant state with fresh mutable fields."""
    import shutil
    s = copy.copy(state)
    s.anchors_path = anchors_path
    s.output_path = str(output_dir / "output.tif")
    s.diagnostics_dir = str(output_dir) + "/"
    s.qa_json_path = str(output_dir / "qa.json")
    # Copy shared input file so each variant has its own copy
    # (the original temp file gets deleted by the first variant's warp step)
    if state.current_input and os.path.exists(state.current_input):
        variant_input = str(output_dir / "variant_input.tif")
        shutil.copy2(state.current_input, variant_input)
        s.current_input = variant_input
        if state.precorrection_tmp:
            s.precorrection_tmp = variant_input
    # Reset mutable fields populated during feature matching
    s.matched_pairs = []
    s.gcps = []
    s.boundary_gcps = []
    s.geo_residuals = []
    s.qa_holdout_pairs = []
    s.qa_reports = []
    s.correction_outliers = []
    s.match_weights = None
    s.temp_paths = []
    return s


def run_variant(label, state, args, profiler):
    """Run feature matching -> validate -> warp -> post-refinement for one variant."""
    t0 = time.time()
    error = None
    try:
        with profiler.section("feature_matching"):
            state = step_feature_matching(state, args, profiler=profiler)
        with profiler.section("validate_and_filter"):
            state = step_validate_and_filter(state)
        with profiler.section("select_warp_and_apply"):
            state = step_select_warp_and_apply(state, profiler=profiler)
        with profiler.section("post_refinement"):
            step_post_refinement(state)
    except Exception as e:
        error = str(e)
        import traceback
        traceback.print_exc()

    elapsed = time.time() - t0
    if error:
        print(f"\n  [{label}] Variant FAILED after {elapsed:.1f}s: {error}")
    else:
        print(f"\n  [{label}] Variant completed in {elapsed:.1f}s")

    # Save profile
    if state.diagnostics_dir:
        profile_path = os.path.join(state.diagnostics_dir, "profile.json")
        try:
            with open(profile_path, "w") as f:
                json.dump(profiler.to_dict(), f, indent=2)
        except Exception:
            pass

    return state, elapsed, error


def load_qa_metrics(qa_path):
    """Load key metrics from a qa.json file."""
    if not os.path.exists(qa_path):
        return None
    try:
        data = json.loads(Path(qa_path).read_text())
    except Exception:
        return None

    # Find the selected candidate report
    selected = data.get("selected_candidate", "grid")
    report = None
    for r in data.get("reports", []):
        if r.get("candidate") == selected:
            report = r
            break
    if not report:
        reports = data.get("reports", [])
        report = reports[0] if reports else {}

    im = report.get("image_metrics", {})
    return {
        "score": round(im.get("score", 0), 1),
        "accepted": report.get("accepted", False),
        "west": round(im["west"]) if im.get("west") is not None else None,
        "center": round(im["center"]) if im.get("center") is not None else None,
        "east": round(im["east"]) if im.get("east") is not None else None,
        "north": round(im["north_shift"]) if im.get("north_shift") is not None else None,
        "patch_med": round(im.get("patch_med", 0)),
        "stable_iou": round(im.get("stable_iou", 0), 3),
        "shore_iou": round(im.get("shore_iou", 0), 3),
        "coverage": round(report.get("coverage", 0), 3),
        "cv_mean_m": round(report.get("cv_mean_m", 0), 1) if report.get("cv_mean_m") is not None else None,
        "candidate": selected,
    }


def print_comparison(metrics_a, metrics_b, elapsed_a, elapsed_b, error_a, error_b):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 60)
    print("  A/B TEST: Anchors vs No-Anchors")
    print("=" * 60)

    if error_a and error_b:
        print(f"  Both variants failed!")
        print(f"  A error: {error_a}")
        print(f"  B error: {error_b}")
        return

    header = f"{'':18s} {'With Anchors':>14s} {'No Anchors':>14s} {'Delta':>10s}"
    print(header)
    print("-" * 60)

    def row(label, key, fmt=".0f", invert=False):
        va = metrics_a.get(key) if metrics_a else None
        vb = metrics_b.get(key) if metrics_b else None
        sa = f"{va:{fmt}}" if va is not None else "FAIL"
        sb = f"{vb:{fmt}}" if vb is not None else "FAIL"
        if va is not None and vb is not None:
            delta = vb - va
            sign = "+" if delta > 0 else ""
            sd = f"{sign}{delta:{fmt}}"
        else:
            sd = "---"
        print(f"  {label:16s} {sa:>14s} {sb:>14s} {sd:>10s}")

    row("score", "score")
    row("west", "west")
    row("center", "center")
    row("east", "east")
    row("north", "north")
    row("patch_med", "patch_med")
    row("stable_iou", "stable_iou", ".3f")
    row("shore_iou", "shore_iou", ".3f")
    row("coverage", "coverage", ".3f")

    # Accepted
    aa = metrics_a.get("accepted") if metrics_a else None
    ab = metrics_b.get("accepted") if metrics_b else None
    print(f"  {'accepted':16s} {str(aa):>14s} {str(ab):>14s}")

    # Wall clock
    print(f"  {'wall_clock':16s} {elapsed_a:>13.0f}s {elapsed_b:>13.0f}s {elapsed_b - elapsed_a:>+9.0f}s")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="A/B test: Anchors vs No-Anchors")
    parser.add_argument("--version", "-v", type=int, default=None,
                        help="Version number (default: auto-detect next)")
    parser.add_argument("--timeout", "-t", type=int, default=6000,
                        help="Timeout in seconds (default: 6000)")
    cli_args = parser.parse_args()

    version = cli_args.version if cli_args.version is not None else detect_next_version()

    # Create diagnostics directories
    dir_root = PROJECT_ROOT / "diagnostics" / f"run_v{version}_AB"
    dir_a = dir_root / "A"
    dir_b = dir_root / "B"
    dir_root.mkdir(parents=True, exist_ok=True)
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    # ---- Continuous logging ----
    log_path = dir_root / "ab_test.log"
    log_fh = open(log_path, "w")
    sys.stdout = TeeStream(sys.__stdout__, log_fh)
    sys.stderr = TeeStream(sys.__stderr__, log_fh)

    args = build_args()
    # Shared steps write diagnostics to root; output/qa are placeholders
    args.output = str(dir_root / "output.tif")
    args.diagnostics_dir = str(dir_root) + "/"
    args.qa_json = str(dir_root / "qa.json")

    pipeline_t0 = time.time()

    # ---- Shared steps (run once) ----
    print(f"=== A/B Test v{version}: Running shared steps ===\n")
    shared_profiler = PipelineProfiler()

    device = get_torch_device(override=None)
    model_cache = ModelCache(device)

    try:
        with shared_profiler.section("setup"):
            state = step_setup(args)
        state.model_cache = model_cache

        with shared_profiler.section("global_localization"):
            state = step_global_localization(state, args)
        with shared_profiler.section("coarse_offset"):
            state = step_coarse_offset(state, profiler=shared_profiler)

        # Handle large offset inline (no recursive run() call)
        with shared_profiler.section("handle_large_offset"):
            if state.coarse_total < 10 and abs(state.expected_scale - 1.0) < 0.05:
                print("Already aligned, nothing to A/B test")
                model_cache.close()
                sys.exit(0)
            elif state.coarse_total > 2000:
                print(f"Large offset ({state.coarse_total:.0f}m) — applying coarse translation inline")
                translated_path = str(dir_root / "translated.tif")
                _apply_coarse_translation(state, translated_path)
                state.temp_paths.append(translated_path)
                # Re-setup on translated image
                args.input = translated_path
                args.coarse_pass = 1
                state = step_setup(args)
                state.model_cache = model_cache
                # Re-run coarse on translated image
                state = step_coarse_offset(state, profiler=shared_profiler)
                print(f"  Residual after translation: {state.coarse_total:.0f}m")

        with shared_profiler.section("scale_rotation"):
            state = step_scale_rotation(state, args, profiler=shared_profiler)

    except Exception as e:
        print(f"\nShared steps FAILED: {e}")
        import traceback
        traceback.print_exc()
        model_cache.close()
        log_fh.close()
        sys.exit(1)

    shared_elapsed = time.time() - pipeline_t0
    shared_profiler.print_waterfall()
    print(f"\n  [Shared steps] {shared_elapsed:.1f}s\n")

    # ---- Fork into two variants ----

    # Variant A: with anchors
    print(f"\n{'=' * 60}")
    print(f"  VARIANT A: With Anchors")
    print(f"{'=' * 60}\n")

    state_a = _fork_state(state, ANCHORS, dir_a, device)
    profiler_a = PipelineProfiler()
    state_a, elapsed_a, error_a = run_variant("A", state_a, args, profiler_a)
    profiler_a.print_waterfall()

    # Variant B: no anchors
    print(f"\n{'=' * 60}")
    print(f"  VARIANT B: No Anchors")
    print(f"{'=' * 60}\n")

    state_b = _fork_state(state, None, dir_b, device)
    state_b.model_cache = ModelCache(device)  # fresh cache, A may have closed resources
    profiler_b = PipelineProfiler()
    state_b, elapsed_b, error_b = run_variant("B", state_b, args, profiler_b)
    profiler_b.print_waterfall()

    # Cleanup
    model_cache.close()
    for temp_path in state.temp_paths:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    total_elapsed = time.time() - pipeline_t0

    # ---- Load QA results and compare ----
    metrics_a = load_qa_metrics(str(dir_a / "qa.json"))
    metrics_b = load_qa_metrics(str(dir_b / "qa.json"))

    print_comparison(metrics_a, metrics_b, elapsed_a, elapsed_b, error_a, error_b)

    # Save comparison JSON
    comparison = {
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shared_elapsed_s": round(shared_elapsed, 1),
        "total_elapsed_s": round(total_elapsed, 1),
        "variant_a": {
            "label": "with_anchors",
            "diagnostics_dir": str(dir_a),
            "elapsed_s": round(elapsed_a, 1),
            "error": error_a,
            "metrics": metrics_a,
        },
        "variant_b": {
            "label": "no_anchors",
            "diagnostics_dir": str(dir_b),
            "elapsed_s": round(elapsed_b, 1),
            "error": error_b,
            "metrics": metrics_b,
        },
    }
    comparison_path = dir_root / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nComparison saved to: {comparison_path}")
    print(f"Total wall clock: {total_elapsed:.0f}s")

    log_fh.close()


if __name__ == "__main__":
    main()
