"""Main orchestration: run() + step functions."""

import copy
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import rasterio
import rasterio.transform
from osgeo import gdal
from pyproj import Transformer
from rasterio.crs import CRS
from .constants import (CV_REFIT_THRESHOLD_M, DEFAULT_SEARCH_MARGIN_M,
                        DEFAULT_TEMPLATE_RADIUS_M, LAND_MASK_FRAC_MIN,
                        OUTPUT_CRS_EPSG, RANSAC_REPROJ_THRESHOLD)
from .geo import (
    clear_overlap_cache,
    compute_overlap,
    compute_overlap_or_none,
    dataset_bounds_in_crs,
    get_metric_crs,
    get_native_resolution_m,
    get_torch_device,
    read_overlap_region,
)
from .global_localization import localize_to_reference, translate_input_to_hypothesis
from .image import shift_array, to_u8, is_cloudy_patch, make_land_mask, clahe_normalize
from .metadata_priors import load_metadata_priors
from .models import ModelCache
from .coarse import detect_offset_at_resolution
from .scale import (detect_scale_rotation, detect_local_scales, fit_affine_from_gcps,
                    apply_scale_rotation_precorrection, apply_local_scale_precorrection)
from .anchors import locate_anchors
from .matching import match_with_roma
from .filtering import (matched_pairs_sufficient, refine_matches_phase_correlation,
                        correct_reference_offset, select_best_gcps,
                        iterative_outlier_removal,
                        detect_and_correct_reference_offset,
                        local_consistency_filter)
from .boundary import _generate_boundary_gcps
from .errors import (AlignmentError, AlreadyAlignedError, CoarseOffsetError,
                     InsufficientDataError, UserAbortError, WarpError)
from .io import open_pair, read_overlap_pair
from .flow_refine import apply_flow_refinement_to_file
from .tin_filter import filter_by_tin_tarr, optimize_fpps_accuracy
from .warp import apply_warp
from .diagnostics import generate_debug_image
from .qa import evaluate_alignment_quality_paths
from .qa_runner import build_candidate_report, split_holdout_pairs, write_qa_report
from .types import GlobalHypothesis, LegacyGCP, LegacyMatch, MetadataPrior, QaReport


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclass
class AlignState:
    """Explicit state passed between pipeline steps."""
    input_path: str
    reference_path: str
    output_path: str
    work_crs: Optional[CRS] = None
    overlap: Optional[tuple[float, float, float, float]] = None
    anchors_path: Optional[str] = None
    best: bool = True
    match_res: float = 5.0
    coarse_pass: int = 0
    yes: bool = False

    # Populated during setup
    offset_res_m: float = 0.0
    ref_res_m: float = 0.0
    expected_scale: float = 1.0

    # Updated during pipeline
    current_input: str = ""
    coarse_dx: float = 0.0
    coarse_dy: float = 0.0
    coarse_total: float = 0.0
    coarse_corr: float = 0.0
    precorrection_applied: bool = False
    precorrection_tmp: Optional[str] = None
    needs_scale_rotation: bool = False

    matched_pairs: list[LegacyMatch] = field(default_factory=list)
    gcps: list[LegacyGCP] = field(default_factory=list)
    boundary_gcps: list[LegacyGCP] = field(default_factory=list)
    geo_residuals: list[float] = field(default_factory=list)
    mean_residual: float = float('inf')
    max_residual: float = float('inf')
    gcp_coverage: float = 1.0
    used_neural: bool = False
    use_sift_refinement: bool = False

    was_corrected: bool = False
    correction_outliers: List[str] = field(default_factory=list)

    M_geo: Optional[np.ndarray] = None
    cv_mean: Optional[float] = None
    model_cache: Optional[ModelCache] = None
    tin_tarr_thresh: float = 1.5
    skip_fpp: bool = False
    matcher_anchor: str = "roma"
    matcher_dense: str = "roma"
    grid_size: int = 20
    grid_iters: int = 300
    arap_weight: float = 1.0
    mask_provider: str = "coastal_obia"
    global_search: bool = True
    global_search_res: float = 40.0
    global_search_top_k: int = 3
    force_global: bool = False
    reference_window: Optional[tuple] = None
    metadata_prior_paths: List[str] = field(default_factory=list)
    metadata_priors: List[MetadataPrior] = field(default_factory=list)
    global_hypotheses: List[GlobalHypothesis] = field(default_factory=list)
    chosen_hypothesis: Optional[GlobalHypothesis] = None
    reference_bounds_work: Optional[tuple] = None
    target_bounds_work: Optional[tuple] = None
    qa_holdout_pairs: list[LegacyMatch] = field(default_factory=list)
    qa_reports: List[QaReport] = field(default_factory=list)
    qa_json_path: Optional[str] = None
    diagnostics_dir: Optional[str] = None
    allow_abstain: bool = False
    tps_fallback: bool = False
    abstained: bool = False
    temp_paths: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def _ensure_overviews(path, width, height):
    """Build GDAL overviews if missing and image is large enough."""
    if max(width, height) < 10000:
        return
    # Validate path exists and is a real file (not injected)
    if not os.path.isfile(path):
        print(f"  WARNING: Cannot build overviews — path does not exist: {path}")
        return
    with rasterio.open(path) as src:
        has_overviews = bool(src.overviews(1))
    if has_overviews:
        return
    print(f"  Building GDAL overviews for {os.path.basename(path)}...")
    result = subprocess.run(
        ["gdaladdo", "-r", "average", path, "2", "4", "8", "16"],
        capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: gdaladdo failed: {result.stderr.strip()}")
    else:
        print(f"  Overviews built for {os.path.basename(path)}")


def step_setup(args) -> AlignState:
    """Parse args, inspect datasets, and load priors."""
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}.aligned{ext}"

    src_offset = rasterio.open(args.input)
    src_ref = rasterio.open(args.reference)

    print(f"Target image:    {args.input}")
    print(f"  CRS: {src_offset.crs}, Size: {src_offset.width}x{src_offset.height}")
    print(f"  Bounds: {src_offset.bounds}")
    print(f"Reference image: {args.reference}")
    print(f"  CRS: {src_ref.crs}, Size: {src_ref.width}x{src_ref.height}")
    print(f"  Bounds: {src_ref.bounds}")
    print()

    priors = load_metadata_priors(
        args.input,
        explicit_paths=getattr(args, "metadata_priors", None),
        priors_dir=getattr(args, "metadata_priors_dir", None),
    )
    metadata_prior_paths = [str(path) for path in (getattr(args, "metadata_priors", None) or [])]
    work_crs = get_metric_crs(src_offset, src_ref, priors=priors)

    # Build GDAL overviews for faster reads at lower resolutions
    off_w, off_h = src_offset.width, src_offset.height
    ref_w, ref_h = src_ref.width, src_ref.height
    src_offset.close()
    src_ref.close()
    
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_ensure_overviews, args.input, off_w, off_h)
        f2 = pool.submit(_ensure_overviews, args.reference, ref_w, ref_h)
        f1.result()
        f2.result()

    # Re-open after overviews are built
    src_offset = rasterio.open(args.input)
    src_ref = rasterio.open(args.reference)

    offset_res_m = get_native_resolution_m(src_offset)
    ref_res_m = get_native_resolution_m(src_ref)
    expected_scale = offset_res_m / ref_res_m
    print(f"Native resolution: target={offset_res_m:.2f} m/px, reference={ref_res_m:.2f} m/px")
    print(f"Expected scale ratio: {expected_scale:.3f}x")
    print()

    target_bounds_work = dataset_bounds_in_crs(src_offset, work_crs)
    reference_bounds_work = dataset_bounds_in_crs(src_ref, work_crs)
    overlap = compute_overlap_or_none(src_offset, src_ref, work_crs)
    if overlap is not None:
        overlap_w = overlap[2] - overlap[0]
        overlap_h = overlap[3] - overlap[1]
        print(f"Overlap region: {overlap_w:.0f} x {overlap_h:.0f} meters")
    else:
        print("Overlap region: none detected from rough georeferencing")
    print()

    reference_window = getattr(args, "reference_window", None)
    if isinstance(reference_window, str):
        parts = [p.strip() for p in reference_window.split(",") if p.strip()]
        if len(parts) == 4:
            reference_window = tuple(float(p) for p in parts)
        else:
            raise ValueError("--reference-window must be left,bottom,right,top")

    src_offset.close()
    src_ref.close()

    return AlignState(
        input_path=args.input,
        reference_path=args.reference,
        output_path=args.output,
        work_crs=work_crs,
        overlap=overlap,
        anchors_path=getattr(args, 'anchors', None),
        best=bool(getattr(args, 'best', False)),
        match_res=args.match_res,
        coarse_pass=args.coarse_pass,
        yes=args.yes,
        offset_res_m=offset_res_m,
        ref_res_m=ref_res_m,
        expected_scale=expected_scale,
        current_input=args.input,
        tin_tarr_thresh=getattr(args, 'tin_tarr_thresh', 1.5),
        skip_fpp=getattr(args, 'skip_fpp', False),
        matcher_anchor=getattr(args, 'matcher_anchor', 'roma'),
        matcher_dense=getattr(args, 'matcher_dense', 'roma'),
        grid_size=getattr(args, 'grid_size', 20),
        grid_iters=getattr(args, 'grid_iters', 300),
        arap_weight=getattr(args, 'arap_weight', 1.0),
        mask_provider=getattr(args, 'mask_provider', 'coastal_obia'),
        global_search=bool(getattr(args, 'global_search', True)),
        global_search_res=float(getattr(args, 'global_search_res', 40.0)),
        global_search_top_k=int(getattr(args, 'global_search_top_k', 3)),
        force_global=bool(getattr(args, 'force_global', False)),
        reference_window=reference_window,
        metadata_prior_paths=metadata_prior_paths,
        metadata_priors=priors,
        reference_bounds_work=reference_bounds_work,
        target_bounds_work=target_bounds_work,
        qa_json_path=getattr(args, 'qa_json', None),
        diagnostics_dir=getattr(args, 'diagnostics_dir', None),
        allow_abstain=bool(getattr(args, 'allow_abstain', False)),
        tps_fallback=bool(getattr(args, 'tps_fallback', False)),
    )


def _refresh_work_region(state: AlignState) -> AlignState:
    """Recompute work CRS bounds and overlap for the current input."""

    with open_pair(state.current_input, state.reference_path) as (src_offset, src_ref):
        state.work_crs = get_metric_crs(src_offset, src_ref, priors=state.metadata_priors)
        state.target_bounds_work = dataset_bounds_in_crs(src_offset, state.work_crs)
        state.reference_bounds_work = dataset_bounds_in_crs(src_ref, state.work_crs)
        state.overlap = compute_overlap_or_none(src_offset, src_ref, state.work_crs)
    return state


def step_global_localization(state: AlignState, args) -> AlignState:
    """Add a coarse localization stage before overlap-dependent matching."""

    if state.overlap is not None and not state.force_global:
        state.global_hypotheses = [
            GlobalHypothesis(
                hypothesis_id="rough_overlap",
                score=1.0,
                source="rough_georef_overlap",
                left=state.overlap[0],
                bottom=state.overlap[1],
                right=state.overlap[2],
                top=state.overlap[3],
                dx_m=0.0,
                dy_m=0.0,
                work_crs=str(state.work_crs),
            )
        ]
        state.chosen_hypothesis = state.global_hypotheses[0]
        return state

    if not state.global_search:
        if state.overlap is None:
            raise ValueError("Global localization is disabled and no rough overlap exists")
        return state

    print("Step 0: Global localization against reference", flush=True)
    with open_pair(state.current_input, state.reference_path) as (src_offset, src_ref):
        hypotheses = localize_to_reference(
            src_offset,
            src_ref,
            state.work_crs,
            priors=state.metadata_priors,
            coarse_res=state.global_search_res,
            top_k=state.global_search_top_k,
            mask_mode=state.mask_provider,
            search_bounds=state.reference_window,
        )

    if not hypotheses:
        raise ValueError("Global localization did not produce any hypotheses")

    state.global_hypotheses = hypotheses
    state.chosen_hypothesis = max(hypotheses, key=lambda hyp: hyp.score)
    best = state.chosen_hypothesis
    assert best is not None
    print(
        f"  Selected hypothesis {best.hypothesis_id}: score={best.score:.3f}, "
        f"dx={best.dx_m:+.0f}m, dy={best.dy_m:+.0f}m, "
        f"scale={best.scale_hint:.2f}x, rot={best.rotation_hint_deg:+.1f}deg"
    )

    if state.overlap is None or abs(best.dx_m) > 50.0 or abs(best.dy_m) > 50.0 or state.force_global:
        shifted_input = translate_input_to_hypothesis(state.current_input, best, state.work_crs)
        state.current_input = shifted_input
        state.temp_paths.append(shifted_input)
        state = _refresh_work_region(state)
        if state.overlap is None:
            raise ValueError("Global localization translated the image but no overlap was produced")
        overlap_w = state.overlap[2] - state.overlap[0]
        overlap_h = state.overlap[3] - state.overlap[1]
        print(f"  Localized overlap region: {overlap_w:.0f} x {overlap_h:.0f} meters")
    print()
    return state


def _grayscale_ncc_crosscheck(state: AlignState, src_ref, src_offset) -> None:
    """Pass 3: Independent CLAHE-grayscale NCC cross-check at 5m/px.

    The land-mask based estimates can be biased by mask disagreements
    (tide differences, shoal misclassification).  Run a CLAHE-grayscale
    NCC around the current best estimate; if it converges to a nearby
    but different peak, prefer it when the correlation is better.
    Mutates *state.coarse_dx / coarse_dy* in place.
    """
    print("  Pass 3: grayscale NCC cross-check at 5 m/px...")
    try:
        arr_ref, _ = read_overlap_region(src_ref, state.overlap, state.work_crs, 5.0)
        arr_off, _ = read_overlap_region(src_offset, state.overlap, state.work_crs, 5.0)

        valid_ref = arr_ref > 0
        if not np.any(valid_ref):
            return
        rr, cc = np.where(valid_ref)
        rc, cc_c = int(np.mean(rr)), int(np.mean(cc))
        th = min(600, min(arr_ref.shape) // 3)
        best_dx_px = int(round(state.coarse_dx / 5.0))
        best_dy_px = int(round(state.coarse_dy / 5.0))
        margin = int(200 / 5.0)

        # Reference template bounds
        tr0 = max(0, rc - th)
        tr1 = min(arr_ref.shape[0], rc + th)
        tc0 = max(0, cc_c - th)
        tc1 = min(arr_ref.shape[1], cc_c + th)

        # Offset search bounds
        s_c0 = max(0, tc0 + best_dx_px - margin)
        s_c1 = min(arr_off.shape[1], tc0 + (tr1 - tr0) + best_dx_px + margin)
        s_r0 = max(0, tr0 + best_dy_px - margin)
        s_r1 = min(arr_off.shape[0], tr0 + (tr1 - tr0) + best_dy_px + margin)

        # Crop first, then CLAHE (only on small patches, with 100 px context padding)
        pad = 100
        ref_crop = arr_ref[max(0, tr0 - pad):min(arr_ref.shape[0], tr1 + pad),
                           max(0, tc0 - pad):min(arr_ref.shape[1], tc1 + pad)]
        off_crop = arr_off[max(0, s_r0 - pad):min(arr_off.shape[0], s_r1 + pad),
                           max(0, s_c0 - pad):min(arr_off.shape[1], s_c1 + pad)]

        ref_u8 = clahe_normalize(ref_crop)
        off_u8 = clahe_normalize(off_crop)

        # Extract template and search from CLAHE'd crops, adjusting for padding
        rpad_t = tr0 - max(0, tr0 - pad)
        cpad_t = tc0 - max(0, tc0 - pad)
        templ = ref_u8[rpad_t:rpad_t + (tr1 - tr0), cpad_t:cpad_t + (tc1 - tc0)]

        rpad_s = s_r0 - max(0, s_r0 - pad)
        cpad_s = s_c0 - max(0, s_c0 - pad)
        search = off_u8[rpad_s:rpad_s + (s_r1 - s_r0), cpad_s:cpad_s + (s_c1 - s_c0)]

        if (templ.shape[0] < 64 or templ.shape[1] < 64 or
                search.shape[0] <= templ.shape[0] or search.shape[1] <= templ.shape[1]):
            return

        res = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        dx_ncc = (s_c0 + max_loc[0] - tc0) * 5.0
        dy_ncc = (s_r0 + max_loc[1] - tr0) * 5.0
        delta = np.sqrt((dx_ncc - state.coarse_dx) ** 2 + (dy_ncc - state.coarse_dy) ** 2)
        print(f"    grayscale NCC: dx={dx_ncc:+.0f}m, dy={dy_ncc:+.0f}m "
              f"(corr={max_val:.4f}, delta={delta:.0f}m from mask)")
        if max_val > 0.3 and delta < 150:
            state.coarse_dx = float(dx_ncc)
            state.coarse_dy = float(dy_ncc)
            print(f"    Accepted grayscale cross-check")
        elif max_val > 0.3:
            print(f"    Grayscale cross-check diverged too far, keeping mask estimate")
    except Exception as e:
        print(f"    Grayscale cross-check failed: {e}")


def step_coarse_offset(state: AlignState) -> AlignState:
    """Step 1: Detect coarse offset via land mask matching."""
    if state.overlap is None or state.work_crs is None:
        raise ValueError("Coarse offset requires an established work region")
    print("Step 1: Coarse offset detection via land mask matching", flush=True)

    src_offset = rasterio.open(state.current_input)
    src_ref = rasterio.open(state.reference_path)

    print("  Pass 1: coarse scan at 15 m/px...")
    dx_c, dy_c, corr_c = detect_offset_at_resolution(
        src_offset, src_ref, state.overlap, state.work_crs, 15.0,
        template_radius_m=DEFAULT_TEMPLATE_RADIUS_M, mask_mode=state.mask_provider)
    if dx_c is None or dy_c is None or corr_c is None:
        if state.chosen_hypothesis is not None:
            # Global search already positioned the image — coarse refinement
            # is non-critical. Continue with zero residual offset.
            print("WARNING: Could not detect offset at 15m resolution — "
                  "continuing with global search position")
            src_offset.close()
            src_ref.close()
            state.coarse_dx = 0.0
            state.coarse_dy = 0.0
            state.coarse_total = 0.0
            return state
        src_offset.close()
        src_ref.close()
        raise CoarseOffsetError("Could not detect offset at 15m resolution")
    
    # We now know dx_c and dy_c are floats
    dx_c_f, dy_c_f = float(dx_c), float(dy_c)
    print(f"    dx={dx_c_f:+8.0f}m, dy={dy_c_f:+8.0f}m (corr={corr_c:.4f})")

    print("  Pass 2: refine at 5 m/px (search +/-300m around coarse)...")
    dx_r, dy_r, corr_r = detect_offset_at_resolution(
        src_offset, src_ref, state.overlap, state.work_crs, 5.0,
        template_radius_m=DEFAULT_TEMPLATE_RADIUS_M, coarse_offset=(dx_c_f, dy_c_f), search_margin_m=DEFAULT_SEARCH_MARGIN_M,
        mask_mode=state.mask_provider)
    
    if dx_r is not None and dy_r is not None and corr_r is not None and corr_r > 0.3:
        dx_r_f, dy_r_f = float(dx_r), float(dy_r)
        print(f"    dx={dx_r_f:+8.0f}m, dy={dy_r_f:+8.0f}m (corr={corr_r:.4f})")
        refinement_dist = np.sqrt((dx_r_f - dx_c_f) ** 2 + (dy_r_f - dy_c_f) ** 2)
        if refinement_dist < 250:
            state.coarse_dx, state.coarse_dy = dx_r_f, dy_r_f
            state.coarse_corr = float(corr_r)
            print(f"    Using refined estimate (delta={refinement_dist:.0f}m from coarse)")
        else:
            state.coarse_dx, state.coarse_dy = dx_c_f, dy_c_f
            state.coarse_corr = float(corr_c)
            print(f"    Refinement too far from coarse ({refinement_dist:.0f}m), using coarse")
    else:
        if dx_r is None or corr_r is None:
            print("    Refinement failed, using coarse estimate")
        else:
            print(f"    Refinement failed (corr={corr_r:.4f}), using coarse estimate")
        state.coarse_dx, state.coarse_dy = dx_c_f, dy_c_f
        state.coarse_corr = float(corr_c)

    _grayscale_ncc_crosscheck(state, src_ref, src_offset)

    src_offset.close()
    src_ref.close()

    state.coarse_total = np.sqrt(state.coarse_dx ** 2 + state.coarse_dy ** 2)
    print(f"  Coarse estimate: dx={state.coarse_dx:+.0f}m, dy={state.coarse_dy:+.0f}m "
          f"(total: {state.coarse_total:.0f}m)")

    # Sanity check: coarse offset should not exceed half the overlap extent
    if state.overlap is not None:
        overlap_w = state.overlap[2] - state.overlap[0]
        overlap_h = state.overlap[3] - state.overlap[1]
        max_reasonable = 0.5 * min(overlap_w, overlap_h)
        if state.coarse_total > max_reasonable:
            print(f"  WARNING: Coarse offset {state.coarse_total:.0f}m exceeds "
                  f"half the overlap extent ({max_reasonable:.0f}m) "
                  f"— estimate may be unreliable")

    print()

    return state


def step_handle_large_offset(state: AlignState, args) -> AlignState:
    """Handle already-aligned or very large offsets (>2km)."""
    if state.coarse_total < 10:
        if abs(state.expected_scale - 1.0) < 0.05:
            raise AlreadyAlignedError(
                "Images appear already well-aligned (offset < 10m, scale ~1.0)")
        else:
            print(f"  Translation offset < 10m but expected scale ratio "
                  f"{state.expected_scale:.3f}x -- continuing with scale correction")

    max_coarse_passes = 1
    if state.coarse_total > 2000:
        print(f"Large offset detected -- applying coarse translation first."
              f" (pass {state.coarse_pass + 1}/{max_coarse_passes})")

        if state.coarse_pass >= max_coarse_passes:
            print(f"  Coarse pass limit reached ({max_coarse_passes}) -- "
                  f"continuing to feature matching with {state.coarse_total:.0f}m residual")
        else:
            src = rasterio.open(state.current_input)
            left = src.bounds.left - state.coarse_dx
            bottom = src.bounds.bottom + state.coarse_dy
            right = src.bounds.right - state.coarse_dx
            top = src.bounds.top + state.coarse_dy
            src.close()

            if not state.yes:
                response = input(
                    f"Apply translation ({state.coarse_dx:+.0f}m, {state.coarse_dy:+.0f}m)? [y/N] ")
                if response.lower() not in ("y", "yes"):
                    raise UserAbortError("User declined coarse translation")

            if os.path.exists(state.output_path):
                os.remove(state.output_path)

            cmd = [
                "gdal_translate",
                "-a_ullr", f"{left:.6f}", f"{top:.6f}", f"{right:.6f}", f"{bottom:.6f}",
                "-co", "COMPRESS=LZW", "-co", "PREDICTOR=2", "-co", "TILED=YES",
                state.current_input, state.output_path,
            ]
            print(f"  Translating by ({state.coarse_dx:+.0f}m, {state.coarse_dy:+.0f}m)...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise WarpError(f"gdal_translate failed:\n{result.stderr}")

            print(f"\nTranslated image written to: {state.output_path}")

            print(f"\n{'=' * 60}")
            print("Continuing with refinement pass...")
            print(f"{'=' * 60}\n")
            tmp_fd, tmp_path_refine = tempfile.mkstemp(
                suffix=".tif", dir=os.path.dirname(state.output_path) or ".")
            os.close(tmp_fd)
            try:
                refine_args = copy.copy(args)
                refine_args.input = state.output_path
                refine_args.output = tmp_path_refine
                refine_args.coarse_pass = state.coarse_pass + 1
                run(refine_args, model_cache=state.model_cache)
                if os.path.exists(tmp_path_refine):
                    os.replace(tmp_path_refine, state.output_path)
            except (AlignmentError, SystemExit):
                pass
            except Exception as exc:
                print(f"\n  Refinement pass FAILED: {exc}", flush=True)
                traceback.print_exc()
                if os.path.exists(tmp_path_refine):
                    os.remove(tmp_path_refine)
            raise AlreadyAlignedError(
                "Recursive refinement pass complete")

    return state


def step_scale_rotation(state: AlignState, args) -> AlignState:
    """Step 1.5: Scale and rotation detection and pre-correction."""
    if state.overlap is None or state.work_crs is None:
        raise ValueError("Scale/rotation detection requires an established work region")
    state.needs_scale_rotation = (abs(state.expected_scale - 1.0) > 0.05 or
                                  state.expected_scale > 1.1)
    if not state.needs_scale_rotation:
        return state

    print("Step 1.5: Scale and rotation detection", flush=True)
    print(f"  Expected scale ratio: {state.expected_scale:.3f}x (from native resolutions)")

    # --- Try local scale detection first ---
    local_patches = None
    print("  Attempting patch-based local scale detection (4x3 grid)...")
    try:
        src_offset = rasterio.open(state.current_input)
        src_ref = rasterio.open(state.reference_path)
        local_patches = detect_local_scales(
            src_offset, src_ref, state.overlap, state.work_crs,
            state.coarse_dx, state.coarse_dy,
            model_cache=state.model_cache)
        src_offset.close()
        src_ref.close()
    except Exception as e:
        print(f"  Local scale detection error: {e}")
        local_patches = None

    if local_patches is not None:
        state = _apply_local_precorrection(state, local_patches)
    else:
        print("  Local scale detection returned None")

    # --- Global fallback ---
    if local_patches is None and not state.precorrection_applied:
        state = _apply_global_precorrection(state, args)

    # Invalidate overlap cache after precorrection changes
    if state.precorrection_applied:
        clear_overlap_cache()

    print()
    return state


def _apply_local_precorrection(state: AlignState, local_patches) -> AlignState:
    """Apply local scale/rotation pre-correction from patch results."""
    valid_patches = [p for p in local_patches
                     if p['status'] in ('ok', 'filled-neighbor', 'filled-global')]
    avg_sx = np.mean([p['scale_x'] for p in valid_patches])
    avg_sy = np.mean([p['scale_y'] for p in valid_patches])
    avg_rot = np.mean([p['rotation'] for p in valid_patches])
    avg_scale = (avg_sx + avg_sy) / 2

    ok_patches = [p for p in local_patches if p['status'] == 'ok']
    if len(ok_patches) >= 2:
        sx_spread = max(p['scale_x'] for p in ok_patches) - min(p['scale_x'] for p in ok_patches)
        sy_spread = max(p['scale_y'] for p in ok_patches) - min(p['scale_y'] for p in ok_patches)
        print(f"  Local scale variation: sx_spread={sx_spread:.4f}, sy_spread={sy_spread:.4f}")

    print(f"  Average: scale={avg_scale:.4f} (x={avg_sx:.4f}, y={avg_sy:.4f}), "
          f"rotation={avg_rot:.3f} deg")

    if abs(avg_scale - 1.0) <= 0.02 and abs(avg_rot) <= 0.1:
        print("  Scale/rotation within tolerance, skipping pre-correction")
        return state

    overlap_cx = (state.overlap[0] + state.overlap[2]) / 2
    overlap_cy = (state.overlap[1] + state.overlap[3]) / 2
    print(f"  Applying LOCAL scale/rotation pre-correction (center: "
          f"{overlap_cx:.0f}, {overlap_cy:.0f} in work CRS)...")
    precorrected = apply_local_scale_precorrection(
        state.current_input, local_patches, state.work_crs,
        overlap_center=(overlap_cx, overlap_cy))

    if precorrected is None:
        print("  Local pre-correction failed, will try global fallback")
        return state

    state.precorrection_applied = True
    state.precorrection_tmp = precorrected
    state.current_input = precorrected
    print("  Local pre-correction applied successfully")

    state = _redetect_coarse_after_precorrection(state)
    return state


def _apply_global_precorrection(state: AlignState, args) -> AlignState:
    """Fallback global scale/rotation detection and pre-correction."""
    print("  Falling back to global scale/rotation detection...")
    src_offset = rasterio.open(state.current_input)
    src_ref = rasterio.open(state.reference_path)

    result = detect_scale_rotation(
        src_offset, src_ref, state.overlap, state.work_crs,
        state.coarse_dx, state.coarse_dy, state.expected_scale,
        model_cache=state.model_cache)

    src_offset.close()
    src_ref.close()

    if result.method is None:
        print("  No significant scale/rotation detected")
        return state

    det_avg = (result.scale_x + result.scale_y) / 2
    print(f"  Detected: scale={det_avg:.4f} (x={result.scale_x:.4f}, y={result.scale_y:.4f}), "
          f"rotation={result.rotation:.3f} deg (method: {result.method})")

    if abs(det_avg - 1.0) <= 0.02 and abs(result.rotation) <= 0.1:
        print("  Scale/rotation within tolerance, skipping pre-correction")
        return state

    overlap_cx = (state.overlap[0] + state.overlap[2]) / 2
    overlap_cy = (state.overlap[1] + state.overlap[3]) / 2
    print(f"  Applying scale/rotation pre-correction (center: "
          f"{overlap_cx:.0f}, {overlap_cy:.0f} in work CRS)...")
    precorrected = apply_scale_rotation_precorrection(
        state.current_input, result.scale_x, result.rotation, state.work_crs,
        overlap_center=(overlap_cx, overlap_cy), scale_y=result.scale_y)

    if precorrected is None:
        print("  Pre-correction failed, continuing without it")
        return state

    state.precorrection_applied = True
    state.precorrection_tmp = precorrected
    state.current_input = precorrected
    print("  Pre-correction applied successfully")

    state = _redetect_coarse_after_precorrection(state)

    # After global correction, retry local patch-based scale
    # Skip retry if scale residual is small (< 1.5%)
    if state.precorrection_applied and abs(det_avg - 1.0) >= 0.015:
        print("  Retrying local scale detection on globally-corrected image...")
        try:
            src_offset = rasterio.open(state.current_input)
            src_ref = rasterio.open(state.reference_path)
            local_patches_2 = detect_local_scales(
                src_offset, src_ref, state.overlap, state.work_crs,
                state.coarse_dx, state.coarse_dy,
                model_cache=state.model_cache)
            src_offset.close()
            src_ref.close()
        except Exception as e:
            print(f"    Local scale retry error: {e}")
            local_patches_2 = None

        if local_patches_2 is not None:
            ok_p2 = [p for p in local_patches_2 if p['status'] == 'ok']
            valid_p2 = [p for p in local_patches_2
                        if p['status'] in ('ok', 'filled-neighbor', 'filled-global')]
            if valid_p2:
                avg_sx2 = np.mean([p['scale_x'] for p in valid_p2])
                avg_sy2 = np.mean([p['scale_y'] for p in valid_p2])
                avg_rot2 = np.mean([p['rotation'] for p in valid_p2])
                avg_scale2 = (avg_sx2 + avg_sy2) / 2
                print(f"    Local patches: {len(ok_p2)} valid, "
                      f"avg scale={avg_scale2:.4f}, rotation={avg_rot2:.3f} deg")

                if abs(avg_scale2 - 1.0) > 0.005 or abs(avg_rot2) > 0.05:
                    overlap_cx = (state.overlap[0] + state.overlap[2]) / 2
                    overlap_cy = (state.overlap[1] + state.overlap[3]) / 2
                    print("    Applying local refinement...")
                    precorrected_2 = apply_local_scale_precorrection(
                        state.current_input, local_patches_2, state.work_crs,
                        overlap_center=(overlap_cx, overlap_cy))
                    if precorrected_2 is not None:
                        old_tmp = state.precorrection_tmp
                        state.precorrection_tmp = precorrected_2
                        state.current_input = precorrected_2
                        print("    Local refinement applied successfully")

                        ok_inliers = [p.get('n_inliers', 0) for p in ok_p2]
                        high_confidence = (len(ok_p2) >= 3 and np.mean(ok_inliers) >= 20)

                        if high_confidence:
                            print(f"    High confidence ({len(ok_p2)} patches, "
                                  f"avg {np.mean(ok_inliers):.0f} inliers) "
                                  f"-- skipping land mask validation")
                            if old_tmp and os.path.exists(old_tmp):
                                os.remove(old_tmp)
                            # Recompute overlap
                            src_offset = rasterio.open(state.current_input)
                            src_ref = rasterio.open(state.reference_path)
                            state.overlap = compute_overlap(src_offset, src_ref, state.work_crs)
                            src_offset.close()
                            src_ref.close()
                        else:
                            # Validate with SIFT
                            print("    Validating with SIFT match comparison...")
                            src_offset = rasterio.open(state.current_input)
                            src_ref = rasterio.open(state.reference_path)
                            overlap_after = compute_overlap(src_offset, src_ref, state.work_crs)
                            src_offset.close()
                            src_ref.close()

                            sift_before = _quick_sift_count(
                                old_tmp, state.reference_path,
                                state.overlap, state.work_crs)
                            sift_after = _quick_sift_count(
                                precorrected_2, state.reference_path,
                                overlap_after, state.work_crs)
                            print(f"    SIFT matches: before={sift_before}, after={sift_after}")

                            if sift_after < sift_before * 0.8:
                                print("    WARNING: Local refinement reduced SIFT matches, reverting")
                                if os.path.exists(precorrected_2):
                                    os.remove(precorrected_2)
                                state.precorrection_tmp = old_tmp
                                state.current_input = old_tmp if old_tmp is not None else state.input_path
                            else:
                                if old_tmp and os.path.exists(old_tmp):
                                    os.remove(old_tmp)
                                state.overlap = overlap_after
                    else:
                        print("    Local refinement failed, keeping global correction")
                else:
                    print("    Local residual within tolerance, skipping")
    elif state.precorrection_applied and abs(det_avg - 1.0) < 0.015:
        print(f"  Global correction residual < 1.5% ({abs(det_avg - 1.0)*100:.2f}%), "
              f"skipping local retry")

    return state


def _redetect_coarse_after_precorrection(state: AlignState) -> AlignState:
    """Re-detect coarse offset on pre-corrected image."""
    coarse_before = np.sqrt(state.coarse_dx ** 2 + state.coarse_dy ** 2)
    print("  Re-detecting coarse offset on pre-corrected image...")
    src_offset = rasterio.open(state.current_input)
    src_ref = rasterio.open(state.reference_path)

    try:
        state.overlap = compute_overlap(src_offset, src_ref, state.work_crs)
    except ValueError:
        print("  WARNING: No overlap after pre-correction, reverting to original")
        state.precorrection_applied = False
        state.current_input = state.input_path
        if state.precorrection_tmp and os.path.exists(state.precorrection_tmp):
            os.remove(state.precorrection_tmp)
        state.precorrection_tmp = None
        src_offset.close()
        src_ref.close()
        return state

    dx_pc, dy_pc, corr_pc = detect_offset_at_resolution(
        src_offset, src_ref, state.overlap, state.work_crs, 5.0,
        template_radius_m=DEFAULT_TEMPLATE_RADIUS_M)
    if dx_pc is not None and dy_pc is not None and corr_pc is not None and corr_pc > 0.3:
        dx_pc_f, dy_pc_f = float(dx_pc), float(dy_pc)
        coarse_after = np.sqrt(dx_pc_f ** 2 + dy_pc_f ** 2)
        if coarse_after > coarse_before * 1.5:
            print(f"  WARNING: Pre-correction worsened offset "
                  f"({coarse_before:.0f}m -> {coarse_after:.0f}m), reverting")
            state.precorrection_applied = False
            state.current_input = state.input_path
            if state.precorrection_tmp and os.path.exists(state.precorrection_tmp):
                os.remove(state.precorrection_tmp)
            state.precorrection_tmp = None
        else:
            state.coarse_dx, state.coarse_dy = dx_pc_f, dy_pc_f
            state.coarse_total = np.sqrt(state.coarse_dx ** 2 + state.coarse_dy ** 2)
            print(f"  Post-correction offset: dx={state.coarse_dx:+.0f}m, "
                  f"dy={state.coarse_dy:+.0f}m (total: {state.coarse_total:.0f}m)")
    else:
        if corr_pc is None:
            print("  Could not re-detect offset, keeping previous estimate")
        else:
            print(f"  Could not re-detect offset (corr={corr_pc:.3f}), keeping previous estimate")

    src_offset.close()
    src_ref.close()
    return state


def _quick_sift_count(img_path, ref_path, ovlp, crs, res=5.0):
    """Count SIFT matches as alignment quality proxy."""
    s_o = rasterio.open(img_path)
    s_r = rasterio.open(ref_path)
    a_r, _ = read_overlap_region(s_r, ovlp, crs, res)
    a_o, _ = read_overlap_region(s_o, ovlp, crs, res)
    s_o.close()
    s_r.close()
    sift = cv2.SIFT_create(nfeatures=2000) # type: ignore
    _, d1 = sift.detectAndCompute(to_u8(a_r), None)
    _, d2 = sift.detectAndCompute(to_u8(a_o), None)
    if d1 is None or d2 is None:
        return 0
    bf = cv2.BFMatcher()
    raw = bf.knnMatch(d1, d2, k=2)
    return sum(1 for m, n in raw if m.distance < 0.75 * n.distance)


def step_feature_matching(state: AlignState, args) -> AlignState:
    """Step 2: Feature matching (neural cascade -> NCC fallback)."""
    if state.overlap is None or state.work_crs is None:
        raise ValueError("Feature matching requires an established work region")
    state.use_sift_refinement = state.coarse_total < 200

    src_offset = rasterio.open(state.current_input)
    src_ref = rasterio.open(state.reference_path)
    state.matched_pairs = []
    state.used_neural = False

    if state.best:
        print("  --best mode: running all methods at maximum quality")

    neural_res_floor = 1.0 if state.best else 2.0
    neural_res = max(state.offset_res_m, state.ref_res_m, neural_res_floor)
    fine_res_floor = 0.5 if state.best else 1.0
    fine_res = max(state.offset_res_m, state.ref_res_m, fine_res_floor)

    # Memory safeguard
    overlap_w = state.overlap[2] - state.overlap[0]
    overlap_h = state.overlap[3] - state.overlap[1]
    fine_pixels = (overlap_w / fine_res) * (overlap_h / fine_res)
    fine_mem_gb = fine_pixels * 8 * 2 / (1024 ** 3)
    if fine_mem_gb > 8.0 and not state.best:
        fine_res = max(fine_res, 1.5)
        print(f"  Memory safeguard: fine_res bumped to {fine_res}m/px "
              f"(estimated {fine_mem_gb:.1f}GB)")

    print(f"  Neural resolution: {neural_res:.1f}m/px, Fine resolution: {fine_res:.1f}m/px")

    # Neural feature matching
    print("Step 2: Neural feature matching cascade", flush=True)

    print(f"  Reading reference overlap at {neural_res}m/px...")
    arr_ref_neural, ref_neural_transform = read_overlap_region(
        src_ref, state.overlap, state.work_crs, neural_res)
    print(f"  Reading offset overlap at {neural_res}m/px...")
    arr_off_neural, off_neural_transform = read_overlap_region(
        src_offset, state.overlap, state.work_crs, neural_res)

    h_neural, w_neural = arr_ref_neural.shape
    print(f"  Image size: {w_neural} x {h_neural} px at {neural_res}m/px")

    shift_px_x_neural = int(round(state.coarse_dx / neural_res))
    shift_py_y_neural = int(round(state.coarse_dy / neural_res))
    if shift_px_x_neural == 0 and shift_py_y_neural == 0:
        arr_off_neural_shifted = arr_off_neural
    else:
        arr_off_neural_shifted = shift_array(
            arr_off_neural, -shift_px_x_neural, -shift_py_y_neural)

    # 2a: Anchor GCPs
    if state.anchors_path:
        print(f"  2a: Locating anchor GCPs from {os.path.basename(state.anchors_path)}...", flush=True)
        try:
            anchor_pairs = locate_anchors(
                state.anchors_path, src_ref, src_offset, state.overlap, state.work_crs,
                ref_neural_transform, off_neural_transform,
                arr_ref_neural, arr_off_neural_shifted,
                shift_px_x_neural, shift_py_y_neural, neural_res,
                model_cache=state.model_cache,
                arr_off_unshifted=arr_off_neural,
                matcher_type=state.matcher_anchor,
                diagnostics_dir=state.diagnostics_dir)
            state.matched_pairs.extend(anchor_pairs)
            print(f"    {len(anchor_pairs)} anchor GCPs located")
        except Exception as e:
            print(f"    Anchor GCP loading failed: {e}")

    # 2b: Dense matching (LoFTR or RoMa)
    target = 200 if state.best else 25
    if not matched_pairs_sufficient(state.matched_pairs, target=target):
        print(f"  2b: {state.matcher_dense.upper()} tiled dense matching...", flush=True)
        try:
            matcher_fn = _get_dense_matcher(state.matcher_dense)

            dense_pairs = matcher_fn(
                arr_ref_neural, arr_off_neural_shifted,
                ref_neural_transform, off_neural_transform,
                shift_px_x_neural, shift_py_y_neural,
                neural_res=neural_res,
                model_cache=state.model_cache,
                existing_anchors=state.matched_pairs,
                src_offset=src_offset,
                work_crs=state.work_crs,
                mask_mode=state.mask_provider)
            state.matched_pairs.extend(dense_pairs)
            print(f"    {state.matcher_dense.upper()}: {len(dense_pairs)} matches")
            if dense_pairs:
                state.used_neural = True
        except Exception as e:
            print(f"    {state.matcher_dense.upper()} dense matching failed: {e}")

    # Phase correlation refinement
    arr_ref_fine = arr_off_fine_shifted = ref_fine_transform = off_fine_transform = None
    shift_px_x_fine = shift_py_y_fine = 0
    if state.used_neural and state.matched_pairs:
        print(f"  Refining {len(state.matched_pairs)} neural matches with phase correlation "
              f"at {fine_res}m/px...")
        print(f"    Reading reference overlap at {fine_res}m/px...")
        arr_ref_fine, ref_fine_transform = read_overlap_region(
            src_ref, state.overlap, state.work_crs, fine_res)
        print(f"    Reading offset overlap at {fine_res}m/px...")
        arr_off_fine, off_fine_transform = read_overlap_region(
            src_offset, state.overlap, state.work_crs, fine_res)
        shift_px_x_fine = int(round(state.coarse_dx / fine_res))
        shift_py_y_fine = int(round(state.coarse_dy / fine_res))
        if shift_px_x_fine == 0 and shift_py_y_fine == 0:
            arr_off_fine_shifted = arr_off_fine
        else:
            arr_off_fine_shifted = shift_array(
                arr_off_fine, -shift_px_x_fine, -shift_py_y_fine)
        state.matched_pairs = refine_matches_phase_correlation(
            state.matched_pairs, arr_ref_fine, arr_off_fine_shifted,
            ref_fine_transform, off_fine_transform,
            shift_px_x_fine, shift_py_y_fine, fine_res)
        print(f"    Refined to {len(state.matched_pairs)} matches")

    # Geographic RANSAC (auto-detected only, preserve anchors)
    anchor_pairs = [m for m in state.matched_pairs if m[5].startswith("anchor:")]
    auto_pairs = [m for m in state.matched_pairs if not m[5].startswith("anchor:")]
    if len(auto_pairs) >= 6:
        geo_src = np.array([(m[0], m[1]) for m in auto_pairs],
                           dtype=np.float32).reshape(-1, 1, 2)
        geo_dst = np.array([(m[2], m[3]) for m in auto_pairs],
                           dtype=np.float32).reshape(-1, 1, 2)
        geo_ransac_thresh = max(RANSAC_REPROJ_THRESHOLD * neural_res, 20.0)
        _, geo_inliers = cv2.estimateAffine2D(
            geo_src, geo_dst, method=cv2.RANSAC,
            ransacReprojThreshold=geo_ransac_thresh)
        if geo_inliers is not None:
            geo_mask = geo_inliers.ravel().astype(bool)
            n_before = len(auto_pairs)
            auto_pairs = [m for m, k in zip(auto_pairs, geo_mask) if k]
            print(f"  Geographic RANSAC (auto only): {n_before} -> {len(auto_pairs)} "
                  f"({geo_ransac_thresh:.0f}m threshold)")
        state.matched_pairs = anchor_pairs + auto_pairs
        if anchor_pairs:
            print(f"  Anchors preserved: {len(anchor_pairs)} "
                  f"({', '.join(m[5].replace('anchor:', '') for m in anchor_pairs)})")

    # Quality check
    neural_quality_ok = False
    auto_only = [p for p in state.matched_pairs if not p[5].startswith("anchor:")]
    if len(auto_only) >= 6:
        q_src = np.array([(p[2], p[3]) for p in auto_only])
        q_dst = np.array([(p[0], p[1]) for p in auto_only])
        _, q_residuals = fit_affine_from_gcps(q_src, q_dst)
        q_mean = np.mean(q_residuals)
        print(f"  Neural match quality check: {len(auto_only)} matches, "
              f"mean affine residual={q_mean:.1f}m")
        if q_mean < 50:
            neural_quality_ok = True
        else:
            print(f"  Discarding {len(auto_only)} neural matches, keeping anchors")
            state.matched_pairs = [p for p in state.matched_pairs
                                   if p[5].startswith("anchor:")]
            state.used_neural = False

    if neural_quality_ok and matched_pairs_sufficient(
            state.matched_pairs, target=10 if state.best else 15):
        # Correct reference offset before GCP selection so the anchor-truth
        # filter doesn't reject well-matched features on the island interior.
        state.matched_pairs, state.was_corrected, state.correction_outliers = \
            correct_reference_offset(state.matched_pairs)
        train_pairs, holdout_pairs = split_holdout_pairs(
            state.matched_pairs,
            holdout_fraction=0.15 if state.best else 0.20,
        )
        state.qa_holdout_pairs = holdout_pairs
        print(
            f"  Selecting best GCPs from {len(train_pairs)} training matches"
            + (f" (+ {len(holdout_pairs)} holdout)" if holdout_pairs else "")
            + "..."
        )
        state.matched_pairs, state.gcp_coverage = select_best_gcps(
            train_pairs, state.overlap,
            target_count=60 if state.best else 40,
            correction_outliers=state.correction_outliers,
            arr_ref=arr_ref_fine,
            ref_transform=ref_fine_transform,
            shoreline_quota=0.25)
        print(f"    Selected {len(state.matched_pairs)} GCPs "
              f"(spatial coverage: {state.gcp_coverage:.0%})")

    # Second neural pass at coarser resolution (--best mode, low coverage)
    if (state.best and state.used_neural and neural_quality_ok
            and state.gcp_coverage < 0.30
            and arr_ref_fine is not None):
        _second_neural_pass(
            state, src_ref, src_offset, neural_res, fine_res,
            arr_ref_fine, arr_off_fine_shifted,
            ref_fine_transform, off_fine_transform,
            shift_px_x_fine, shift_py_y_fine)

    # 2d: NCC fallback
    if not matched_pairs_sufficient(state.matched_pairs, target=10 if state.best else 15):
        _ncc_fallback(state, src_ref, src_offset, neural_res)

    print()

    src_offset.close()
    src_ref.close()

    return state


# ---------------------------------------------------------------------------
# Module-level worker data + functions for ProcessPoolExecutor (fork-safe)
# ---------------------------------------------------------------------------
_worker_data = {}


def _ncc_match_worker(args):
    """Worker for NCC grid matching (fine NCC mode)."""
    idx, center_r, center_c = args
    d = _worker_data
    ref_u8 = d['ref_u8']
    offset_shifted = d['offset_shifted']
    template_half = d['template_half']
    search_margin = d['search_margin']
    phase_half = d['phase_half']
    h_img, w_img = d['h_img'], d['w_img']
    ref_match_transform = d['ref_match_transform']
    offset_match_transform = d['offset_match_transform']
    shift_px_x = d['shift_px_x']
    shift_py_y = d['shift_py_y']

    tr0 = center_r - template_half
    tr1 = center_r + template_half
    tc0 = center_c - template_half
    tc1 = center_c + template_half
    template = ref_u8[tr0:tr1, tc0:tc1]
    if np.mean(template > 0) < LAND_MASK_FRAC_MIN or is_cloudy_patch(template):
        return None
    sr0 = tr0 - search_margin
    sr1 = tr1 + search_margin
    sc0 = tc0 - search_margin
    sc1 = tc1 + search_margin
    if sr0 < 0 or sr1 > h_img or sc0 < 0 or sc1 > w_img:
        return None
    search = offset_shifted[sr0:sr1, sc0:sc1]
    if np.mean(search > 0) < LAND_MASK_FRAC_MIN or is_cloudy_patch(search):
        return None
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < 0.25:
        return None
    dx_int = max_loc[0] - search_margin
    dy_int = max_loc[1] - search_margin
    match_r = center_r + dy_int
    match_c = center_c + dx_int
    p_r0 = max(0, match_r - phase_half)
    p_r1 = min(h_img, match_r + phase_half)
    p_c0 = max(0, match_c - phase_half)
    p_c1 = min(w_img, match_c + phase_half)
    ref_patch = ref_u8[p_r0:p_r1, p_c0:p_c1].astype(np.float64)
    off_patch = offset_shifted[p_r0:p_r1, p_c0:p_c1].astype(np.float64)
    sub_dx, sub_dy = 0.0, 0.0
    if ref_patch.shape[0] >= 32 and ref_patch.shape[1] >= 32:
        window = np.outer(np.hanning(ref_patch.shape[0]),
                          np.hanning(ref_patch.shape[1]))
        shift_val, response = cv2.phaseCorrelate(ref_patch * window, off_patch * window)
        if response > 0.05:
            sub_dx, sub_dy = shift_val[0], shift_val[1]
    total_dx = float(dx_int) + sub_dx
    total_dy = float(dy_int) + sub_dy
    ref_gx, ref_gy = rasterio.transform.xy(
        ref_match_transform, float(center_r), float(center_c))
    off_overlap_c = center_c + total_dx + shift_px_x
    off_overlap_r = center_r + total_dy + shift_py_y
    off_gx, off_gy = rasterio.transform.xy(
        offset_match_transform, off_overlap_r, off_overlap_c)
    return (ref_gx, ref_gy, off_gx, off_gy, max_val, f"ncc_{idx}")


def _land_match_worker(args):
    """Worker for land mask grid matching."""
    idx, center_r, center_c = args
    d = _worker_data
    ref_land = d['ref_land']
    offset_shifted = d['offset_shifted']
    template_half = d['template_half']
    search_margin = d['search_margin']
    h_img, w_img = d['h_img'], d['w_img']
    ref_match_transform = d['ref_match_transform']
    offset_match_transform = d['offset_match_transform']
    shift_px_x = d['shift_px_x']
    shift_py_y = d['shift_py_y']

    tr0 = center_r - template_half
    tr1 = center_r + template_half
    tc0 = center_c - template_half
    tc1 = center_c + template_half
    template = ref_land[tr0:tr1, tc0:tc1]
    land_frac = np.mean(template > 0)
    if land_frac < 0.05 or land_frac > 0.95:
        return None
    sr0 = tr0 - search_margin
    sr1 = tr1 + search_margin
    sc0 = tc0 - search_margin
    sc1 = tc1 + search_margin
    if sr0 < 0 or sr1 > h_img or sc0 < 0 or sc1 > w_img:
        return None
    search = offset_shifted[sr0:sr1, sc0:sc1]
    if np.mean(search > 0) < 0.03:
        return None
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < 0.4:
        return None
    pr, pc = max_loc[1], max_loc[0]
    sub_c, sub_r = float(pc), float(pr)
    sharpness = 0.0
    if 0 < pr < result.shape[0] - 1 and 0 < pc < result.shape[1] - 1:
        vc = result[pr, pc]
        vl = result[pr, pc - 1]
        vr_val = result[pr, pc + 1]
        denom_x = vl - 2 * vc + vr_val
        if abs(denom_x) > 1e-10:
            sub_c = pc + 0.5 * (vl - vr_val) / denom_x
        vt = result[pr - 1, pc]
        vb = result[pr + 1, pc]
        denom_y = vt - 2 * vc + vb
        if abs(denom_y) > 1e-10:
            sub_r = pr + 0.5 * (vt - vb) / denom_y
        sharpness = min(abs(denom_x), abs(denom_y))
    dx_px = sub_c - search_margin
    dy_px = sub_r - search_margin
    ref_gx, ref_gy = rasterio.transform.xy(
        ref_match_transform, float(center_r), float(center_c))
    off_c = center_c + dx_px + shift_px_x
    off_r = center_r + dy_px + shift_py_y
    off_gx, off_gy = rasterio.transform.xy(offset_match_transform, off_r, off_c)
    return (ref_gx, ref_gy, off_gx, off_gy, sharpness * max_val, f"tm_{idx}")


def _second_neural_pass(state, src_ref, src_offset, neural_res, fine_res,
                        arr_ref_fine, arr_off_fine_shifted,
                        ref_fine_transform, off_fine_transform,
                        shift_px_x_fine, shift_py_y_fine):
    """Run a second neural matching pass at 3x coarser resolution to fill gaps.

    Used in --best mode when initial coverage is below 30%.  New matches are
    phase-refined, filtered for local consistency, then merged with existing
    matches before re-running GCP selection.
    """
    coarse_neural_res = neural_res * 3
    print(f"\n  Low coverage ({state.gcp_coverage:.0%}) -- second neural pass "
          f"at {coarse_neural_res:.1f}m/px...")
    try:
        arr_ref_c, ref_c_t = read_overlap_region(
            src_ref, state.overlap, state.work_crs, coarse_neural_res)
        arr_off_c, off_c_t = read_overlap_region(
            src_offset, state.overlap, state.work_crs, coarse_neural_res)
        shift_x_c = int(round(state.coarse_dx / coarse_neural_res))
        shift_y_c = int(round(state.coarse_dy / coarse_neural_res))
        if shift_x_c == 0 and shift_y_c == 0:
            arr_off_c_shifted = arr_off_c
        else:
            arr_off_c_shifted = shift_array(arr_off_c, -shift_x_c, -shift_y_c)

        matcher_fn = _get_dense_matcher(state.matcher_dense)

        coarse_pairs = matcher_fn(
            arr_ref_c, arr_off_c_shifted, ref_c_t, off_c_t,
            shift_x_c, shift_y_c, neural_res=coarse_neural_res,
            min_valid_frac=0.15, skip_ransac=True,
            model_cache=state.model_cache,
            existing_anchors=state.matched_pairs,
            mask_mode=state.mask_provider)
        print(f"    {state.matcher_dense.upper()} (coarse): {len(coarse_pairs)} matches")

        if coarse_pairs:
            coarse_pairs = refine_matches_phase_correlation(
                coarse_pairs, arr_ref_fine, arr_off_fine_shifted,
                ref_fine_transform, off_fine_transform,
                shift_px_x_fine, shift_py_y_fine, fine_res)

            n_before = len(coarse_pairs)
            coarse_pairs = local_consistency_filter(
                coarse_pairs, state.matched_pairs)
            print(f"  Local consistency filter: {n_before} -> {len(coarse_pairs)}")
            combined_pairs = state.matched_pairs + coarse_pairs + state.qa_holdout_pairs
            train_pairs, holdout_pairs = split_holdout_pairs(
                combined_pairs,
                holdout_fraction=0.15 if state.best else 0.20,
            )
            state.qa_holdout_pairs = holdout_pairs

            print(
                f"  Re-selecting best GCPs from {len(train_pairs)} training matches"
                + (f" (+ {len(holdout_pairs)} holdout)" if holdout_pairs else "")
                + "..."
            )
            state.matched_pairs, state.gcp_coverage = select_best_gcps(
                train_pairs, state.overlap,
                target_count=60 if state.best else 40,
                correction_outliers=state.correction_outliers,
                arr_ref=arr_ref_fine,
                ref_transform=ref_fine_transform,
                shoreline_quota=0.25)
            print(f"    Selected {len(state.matched_pairs)} GCPs "
                  f"(spatial coverage: {state.gcp_coverage:.0%})")
    except Exception as e:
        print(f"    Second neural pass failed: {e}")


def _ncc_fallback(state: AlignState, src_ref, src_offset, neural_res):
    """NCC / land mask template matching fallback."""
    global _worker_data
    if state.matched_pairs:
        print(f"  Neural matching found {len(state.matched_pairs)} matches "
              f"(insufficient) -- augmenting with NCC fallback")

    if state.use_sift_refinement:
        print("Step 2 (NCC fallback): Fine NCC + phase-correlation refinement (small offset mode)", flush=True)
        ncc_fine_res = 1.0 if state.best else 2.0
        arr_ref_match, ref_match_transform = read_overlap_region(
            src_ref, state.overlap, state.work_crs, ncc_fine_res)
        arr_offset_match, offset_match_transform = read_overlap_region(
            src_offset, state.overlap, state.work_crs, ncc_fine_res)

        h_img, w_img = arr_ref_match.shape
        ref_u8 = clahe_normalize(arr_ref_match)
        offset_u8 = clahe_normalize(arr_offset_match)

        shift_px_x = int(round(state.coarse_dx / ncc_fine_res))
        shift_py_y = int(round(state.coarse_dy / ncc_fine_res))
        if shift_px_x == 0 and shift_py_y == 0:
            offset_shifted = offset_u8
        else:
            offset_shifted = shift_array(offset_u8, -shift_px_x, -shift_py_y)

        template_half = 128
        search_margin = int(max(40, state.coarse_total / ncc_fine_res + 20))
        phase_half = 32
        grid_spacing_px = int(750 / ncc_fine_res)
        margin_px = template_half + search_margin + 10
        grid_rows = list(range(margin_px, h_img - margin_px, grid_spacing_px))
        grid_cols = list(range(margin_px, w_img - margin_px, grid_spacing_px))
        grid_points = [(r, c) for r in grid_rows for c in grid_cols]

        print(f"  Grid: {len(grid_cols)}x{len(grid_rows)} = {len(grid_points)} points")

        _worker_data = {
            'ref_u8': ref_u8, 'offset_shifted': offset_shifted,
            'template_half': template_half, 'search_margin': search_margin,
            'phase_half': phase_half, 'h_img': h_img, 'w_img': w_img,
            'ref_match_transform': ref_match_transform,
            'offset_match_transform': offset_match_transform,
            'shift_px_x': shift_px_x, 'shift_py_y': shift_py_y,
        }
        n_workers = min(len(grid_points), max(1, (os.cpu_count() or 4) - 1))
        ncc_pairs = []
        ctx = mp.get_context('fork')
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            work_items = [(idx, r, c) for idx, (r, c) in enumerate(grid_points)]
            for result in pool.map(_ncc_match_worker, work_items, chunksize=4):
                if result is not None:
                    ncc_pairs.append(result)
        _worker_data = {}
        print(f"  {len(ncc_pairs)} NCC matches found")
        state.matched_pairs.extend(ncc_pairs)

        if len(state.matched_pairs) < 4:
            state.use_sift_refinement = False

    if not state.use_sift_refinement:
        print("Step 2 (NCC fallback): Land mask template matching at grid points", flush=True)
        match_res = state.match_res
        arr_ref_match, ref_match_transform = read_overlap_region(
            src_ref, state.overlap, state.work_crs, match_res)
        arr_offset_match, offset_match_transform = read_overlap_region(
            src_offset, state.overlap, state.work_crs, match_res)

        h_img, w_img = arr_ref_match.shape
        ref_land = make_land_mask(arr_ref_match)
        offset_land = make_land_mask(arr_offset_match)

        shift_px_x = int(round(state.coarse_dx / match_res))
        shift_py_y = int(round(state.coarse_dy / match_res))
        if shift_px_x == 0 and shift_py_y == 0:
            offset_shifted = offset_land
        else:
            offset_shifted = shift_array(offset_land, -shift_px_x, -shift_py_y)

        template_half = 100
        search_margin = 40
        grid_spacing_px = int(1000 / match_res)
        margin_px = template_half + search_margin + 10
        grid_rows = list(range(margin_px, h_img - margin_px, grid_spacing_px))
        grid_cols = list(range(margin_px, w_img - margin_px, grid_spacing_px))
        grid_points = [(r, c) for r in grid_rows for c in grid_cols]

        _worker_data = {
            'ref_land': ref_land, 'offset_shifted': offset_shifted,
            'template_half': template_half, 'search_margin': search_margin,
            'h_img': h_img, 'w_img': w_img,
            'ref_match_transform': ref_match_transform,
            'offset_match_transform': offset_match_transform,
            'shift_px_x': shift_px_x, 'shift_py_y': shift_py_y,
        }
        n_workers = min(len(grid_points), max(1, (os.cpu_count() or 4) - 1))
        ncc_pairs = []
        ctx = mp.get_context('fork')
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            work_items = [(idx, r, c) for idx, (r, c) in enumerate(grid_points)]
            for result in pool.map(_land_match_worker, work_items, chunksize=4):
                if result is not None:
                    ncc_pairs.append(result)
        _worker_data = {}
        print(f"  {len(ncc_pairs)} land mask matches found")
        state.matched_pairs.extend(ncc_pairs)


_DENSE_MATCHERS = {
    "roma": match_with_roma,
}


def _get_dense_matcher(name: str):
    """Return the dense matcher function for the given name."""
    return _DENSE_MATCHERS.get(name, match_with_roma)


def _cross_validate_and_robust_refit(state: AlignState) -> None:
    """Run k-fold cross-validation on the affine fit; RANSAC-refit if CV error is high.

    Sets state.cv_mean.  If the initial affine M_geo has high CV error (>40 m)
    due to outlier GCPs, performs a 200-iteration RANSAC to find a robust affine
    subset and updates state.M_geo and state.cv_mean accordingly.
    GCPs for the grid optimizer are NOT changed — only M_geo for QA scoring.
    """
    if len(state.matched_pairs) < 10:
        return

    native_res = max(state.offset_res_m, state.ref_res_m)
    residual_threshold = max(native_res * 5, 10.0)

    # 5-fold cross-validation
    k_folds = 5
    indices = np.arange(len(state.matched_pairs))
    np.random.seed(42)
    np.random.shuffle(indices)
    cv_errors = []
    fold_size = len(indices) // k_folds
    for fold in range(k_folds):
        start = fold * fold_size
        end = start + fold_size if fold < k_folds - 1 else len(indices)
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        train_pairs = [state.matched_pairs[i] for i in train_idx]
        train_offset = np.array([(p[2], p[3]) for p in train_pairs])
        train_ref = np.array([(p[0], p[1]) for p in train_pairs])
        train_weights = np.array([p[4] for p in train_pairs])
        M_cv, _ = fit_affine_from_gcps(train_offset, train_ref, weights=train_weights)
        for i in test_idx:
            ogx, ogy = state.matched_pairs[i][2], state.matched_pairs[i][3]
            rgx, rgy = state.matched_pairs[i][0], state.matched_pairs[i][1]
            pred_x = M_cv[0, 0] * ogx + M_cv[0, 1] * ogy + M_cv[0, 2]
            pred_y = M_cv[1, 0] * ogx + M_cv[1, 1] * ogy + M_cv[1, 2]
            cv_errors.append(np.sqrt((pred_x - rgx) ** 2 + (pred_y - rgy) ** 2))

    state.cv_mean = float(np.mean(cv_errors))
    print(f"  Cross-validation ({k_folds}-fold, n={len(state.matched_pairs)}): "
          f"fit={state.mean_residual:.1f}m, "
          f"CV={state.cv_mean:.1f}m (threshold={residual_threshold:.0f}m)")

    # RANSAC robust refit when CV error is high
    if state.cv_mean <= CV_REFIT_THRESHOLD_M or len(state.matched_pairs) < 8:
        return

    all_offset = np.array([(p[2], p[3]) for p in state.matched_pairs])
    all_ref = np.array([(p[0], p[1]) for p in state.matched_pairs])
    n_pts = len(state.matched_pairs)
    rng = np.random.RandomState(42)

    best_inliers = 0
    best_mask = np.zeros(n_pts, dtype=bool)
    for _ in range(200):
        sample = rng.choice(n_pts, 3, replace=False)
        M_trial, _ = fit_affine_from_gcps(all_offset[sample], all_ref[sample])
        pred_x = M_trial[0, 0] * all_offset[:, 0] + M_trial[0, 1] * all_offset[:, 1] + M_trial[0, 2]
        pred_y = M_trial[1, 0] * all_offset[:, 0] + M_trial[1, 1] * all_offset[:, 1] + M_trial[1, 2]
        errs = np.sqrt((pred_x - all_ref[:, 0]) ** 2 + (pred_y - all_ref[:, 1]) ** 2)
        inlier_mask = errs < CV_REFIT_THRESHOLD_M
        n_in = int(inlier_mask.sum())
        if n_in > best_inliers:
            best_inliers = n_in
            best_mask = inlier_mask

    if best_inliers < 6:
        return

    inlier_weights = np.array([
        state.matched_pairs[i][4] for i in range(n_pts) if best_mask[i]])
    M_robust, robust_res = fit_affine_from_gcps(
        all_offset[best_mask], all_ref[best_mask], weights=inlier_weights)
    robust_mean = float(np.mean(robust_res))

    if robust_mean < state.cv_mean * 0.5:
        state.M_geo = M_robust
        cv_errors_robust = []
        for i in range(n_pts):
            ogx, ogy = state.matched_pairs[i][2], state.matched_pairs[i][3]
            rgx, rgy = state.matched_pairs[i][0], state.matched_pairs[i][1]
            pred_x = M_robust[0, 0] * ogx + M_robust[0, 1] * ogy + M_robust[0, 2]
            pred_y = M_robust[1, 0] * ogx + M_robust[1, 1] * ogy + M_robust[1, 2]
            cv_errors_robust.append(np.sqrt((pred_x - rgx) ** 2 + (pred_y - rgy) ** 2))
        state.cv_mean = float(np.mean(cv_errors_robust))
        print(f"  Robust M_geo refit: {best_inliers}/{n_pts} inliers, "
              f"mean residual={robust_mean:.1f}m, CV={state.cv_mean:.1f}m")


def _boundary_gcps_to_geo_pairs(state: AlignState) -> list:
    """Convert boundary GCPs to geo-geo pairs via inverse of M_geo.

    boundary_gcps are (px, py, ref_gx, ref_gy).  M_geo maps offset_geo →
    ref_geo, so we invert to get ref_geo → offset_geo for each boundary point.
    Returns list of (ref_gx, ref_gy, off_gx, off_gy) tuples.
    """
    if state.M_geo is None or len(state.boundary_gcps) == 0:
        return []
    M_full = np.vstack([state.M_geo, [0, 0, 1]])
    try:
        M_inv = np.linalg.inv(M_full)[:2]
    except np.linalg.LinAlgError:
        return []
    pairs = []
    for bg in state.boundary_gcps:
        ref_gx, ref_gy = bg[2], bg[3]
        off_gx = M_inv[0, 0] * ref_gx + M_inv[0, 1] * ref_gy + M_inv[0, 2]
        off_gy = M_inv[1, 0] * ref_gx + M_inv[1, 1] * ref_gy + M_inv[1, 2]
        pairs.append((ref_gx, ref_gy, off_gx, off_gy))
    return pairs


def _reject_mad_outliers(state: AlignState) -> None:
    """Remove matched pairs whose offset deviates >N*MAD from the median.

    Anchors are always kept.  Modifies state.matched_pairs in place.
    """
    pairs = state.matched_pairs
    if len(pairs) <= 4:
        return

    offsets_e = np.array([p[2] - p[0] for p in pairs])
    offsets_n = np.array([p[3] - p[1] for p in pairs])

    med_e, med_n = np.median(offsets_e), np.median(offsets_n)
    mad_e = np.median(np.abs(offsets_e - med_e)) * 1.4826
    mad_n = np.median(np.abs(offsets_n - med_n)) * 1.4826

    min_sigma = min(200.0, max(20.0, state.coarse_total * 0.03))
    sigma_e = max(mad_e, min_sigma)
    sigma_n = max(mad_n, min_sigma)
    mad_sigma = 3.5 if state.needs_scale_rotation else 2.5

    keep = []
    for i in range(len(pairs)):
        is_anchor = pairs[i][5].startswith("anchor:")
        if is_anchor or (abs(offsets_e[i] - med_e) < mad_sigma * sigma_e and
                         abs(offsets_n[i] - med_n) < mad_sigma * sigma_n):
            keep.append(i)

    if len(keep) >= 4 and len(keep) < len(pairs):
        n_rejected = len(pairs) - len(keep)
        state.matched_pairs = [pairs[i] for i in keep]
        print(f"  Rejected {n_rejected} outliers, {len(state.matched_pairs)} points remain")


def step_validate_and_filter(state: AlignState) -> AlignState:
    """Step 3-4: Spot check, outlier removal, build GCPs, fit affine."""
    if state.overlap is None or state.work_crs is None:
        raise ValueError("Validation requires an established work region")
    print(f"Step 3: Spot check -- {len(state.matched_pairs)} matched points", flush=True)
    print()
    if len(state.matched_pairs) < 4:
        if state.precorrection_tmp and os.path.exists(state.precorrection_tmp):
            os.remove(state.precorrection_tmp)
        raise InsufficientDataError(
            f"Not enough matched points ({len(state.matched_pairs)}, need at least 4)")

    offsets_e = np.array([p[2] - p[0] for p in state.matched_pairs])
    offsets_n = np.array([p[3] - p[1] for p in state.matched_pairs])
    print(f"  Mean offset:  dE={np.mean(offsets_e):+.0f}m, dN={np.mean(offsets_n):+.0f}m")
    print(f"  Std dev:      dE={np.std(offsets_e):.0f}m, dN={np.std(offsets_n):.0f}m")

    _reject_mad_outliers(state)
    print()

    # Build GCPs
    print("Step 4: Building GCPs for correction", flush=True)

    src_offset = rasterio.open(state.current_input)
    try:
        original_transform = src_offset.transform
        original_crs = src_offset.crs

        # Transform matched pair coords from work_crs to source CRS for rowcol lookup
        # (ogx, ogy are in work_crs but original_transform maps in source CRS)
        needs_crs_transform = str(original_crs) != str(state.work_crs)
        to_src_crs = None
        if needs_crs_transform:
            to_src_crs = Transformer.from_crs(
                state.work_crs.to_epsg(), original_crs.to_epsg(), always_xy=True)

        def _to_pixel(ogx, ogy, _t=to_src_crs):
            """Convert work_crs geo coords to source pixel coords."""
            if needs_crs_transform and _t is not None:
                sx, sy = _t.transform(ogx, ogy)
            else:
                sx, sy = ogx, ogy
            orig_row, orig_col = rasterio.transform.rowcol(original_transform, sx, sy)
            return float(orig_col), float(orig_row)

        def _pairs_to_gcps(pairs):
            """Convert matched_pairs → GCP list [(col, row, ref_gx, ref_gy), ...]."""
            gcps = []
            for pair in pairs:
                rgx, rgy, ogx, ogy = pair[0], pair[1], pair[2], pair[3]
                col, row = _to_pixel(ogx, ogy)
                gcps.append((col, row, rgx, rgy))
            return gcps

        def _recompute_residuals(pairs, M):
            """Compute per-GCP affine residuals (metres) from M_geo."""
            if len(pairs) < 3 or M is None:
                return []
            residuals = []
            for p in pairs:
                ogx, ogy = p[2], p[3]
                rgx, rgy = p[0], p[1]
                pred_x = M[0, 0] * ogx + M[0, 1] * ogy + M[0, 2]
                pred_y = M[1, 0] * ogx + M[1, 1] * ogy + M[1, 2]
                residuals.append(np.sqrt((pred_x - rgx) ** 2 + (pred_y - rgy) ** 2))
            return residuals

        state.gcps = _pairs_to_gcps(state.matched_pairs)

        neural_res = max(state.offset_res_m, state.ref_res_m, 2.0)
        pre_outlier_pairs = list(state.matched_pairs)
        state.matched_pairs, state.M_geo, state.geo_residuals = iterative_outlier_removal(
            state.matched_pairs, neural_res, state.use_sift_refinement, state.used_neural)

        # Detect and correct systematic reference image offset.
        # When the reference has a geographic shift from ground truth, neural matches
        # are self-consistent but anchors (ground truth) get rejected as outliers.
        corrected, M_new, res_new, was_corrected = detect_and_correct_reference_offset(
            pre_outlier_pairs, state.matched_pairs, state.M_geo,
            neural_res, state.use_sift_refinement, state.used_neural)
        if was_corrected:
            state.matched_pairs = corrected
            state.M_geo = M_new
            state.geo_residuals = list(res_new) if res_new is not None else []

        # Rebuild GCPs after outlier removal
        state.gcps = _pairs_to_gcps(state.matched_pairs)

        # Generate boundary GCPs for piecewise affine warp
        boundary_spacing = 2500
        if len(state.gcps) < 40:
            boundary_spacing = 3200
        if len(state.gcps) < 30:
            boundary_spacing = 3800
        state.boundary_gcps = _generate_boundary_gcps(
            state.gcps, state.M_geo,
            src_offset.width, src_offset.height, spacing_px=boundary_spacing)

        max_boundary = int(max(16, 1.25 * max(1, len(state.gcps))))
        if len(state.boundary_gcps) > max_boundary:
            idx = np.linspace(0, len(state.boundary_gcps) - 1, max_boundary, dtype=int)
            state.boundary_gcps = [state.boundary_gcps[i] for i in idx]
            print(f"  Boundary rebalance: capped synthetic GCPs to {len(state.boundary_gcps)} "
                  f"(max ratio 1.25x)")
        else:
            print(f"  Generated {len(state.boundary_gcps)} boundary GCPs for edge anchoring")

        boundary_pairs_geo = _boundary_gcps_to_geo_pairs(state)

        # --- TIN-TARR topological filter (Phase A) ---
        if len(state.matched_pairs) >= 4 and boundary_pairs_geo:
            n_before_tin = len(state.matched_pairs)
            print(f"  TIN-TARR topological filter (threshold={state.tin_tarr_thresh:.2f})...")
            state.matched_pairs = filter_by_tin_tarr(
                state.matched_pairs, boundary_pairs_geo,
                threshold=state.tin_tarr_thresh)
            n_rejected_tin = n_before_tin - len(state.matched_pairs)
            if n_rejected_tin > 0:
                print(f"    Rejected {n_rejected_tin} GCPs with extreme mesh distortion")
                state.gcps = _pairs_to_gcps(state.matched_pairs)
                state.geo_residuals = _recompute_residuals(
                    state.matched_pairs, state.M_geo)
            else:
                print(f"    All {n_before_tin} GCPs passed topological check")

        # --- FPP Accuracy Difference optimization (Phase B) ---
        if not state.skip_fpp and len(state.matched_pairs) >= 6 and boundary_pairs_geo:
            fpp_res = max(state.offset_res_m, state.ref_res_m, 2.0)
            print(f"  FPP accuracy optimization at {fpp_res:.1f}m/px...")
            try:
                src_ref_fpp = rasterio.open(state.reference_path)
                arr_ref_fpp, ref_fpp_transform = read_overlap_region(
                    src_ref_fpp, state.overlap, state.work_crs, fpp_res)
                arr_off_fpp, off_fpp_transform = read_overlap_region(
                    src_offset, state.overlap, state.work_crs, fpp_res)
                src_ref_fpp.close()

                n_before_fpp = len(state.matched_pairs)
                state.matched_pairs, n_fpp_removed = optimize_fpps_accuracy(
                    state.matched_pairs, boundary_pairs_geo,
                    arr_ref_fpp, arr_off_fpp,
                    ref_fpp_transform, off_fpp_transform)
                if n_fpp_removed > 0:
                    print(f"    Removed {n_fpp_removed} GCPs that degraded local image similarity")
                    state.gcps = _pairs_to_gcps(state.matched_pairs)
                    state.geo_residuals = _recompute_residuals(
                        state.matched_pairs, state.M_geo)
                else:
                    print(f"    All {n_before_fpp} GCPs contribute positively to image similarity")
            except Exception as e:
                print(f"    FPP optimization failed: {e}")

        state.mean_residual = float(np.mean(state.geo_residuals))
        state.max_residual = float(np.max(state.geo_residuals))
        print(f"  Final: {len(state.gcps)} GCPs, mean residual: {state.mean_residual:.1f}m, "
              f"max: {state.max_residual:.1f}m")

        # Affine parameters
        assert state.M_geo is not None
        a, b_val = state.M_geo[0, 0], state.M_geo[0, 1]
        c, d = state.M_geo[1, 0], state.M_geo[1, 1]
        scale_x = np.sqrt(a ** 2 + c ** 2)
        scale_y = np.sqrt(b_val ** 2 + d ** 2)
        rotation = np.degrees(np.arctan2(c, a))
        print(f"  Scale: {(scale_x - 1) * 100:+.2f}% / {(scale_y - 1) * 100:+.2f}%")
        print(f"  Rotation: {rotation:.3f} deg")

        # Debug visualisation
        if state.diagnostics_dir:
            os.makedirs(state.diagnostics_dir, exist_ok=True)
            debug_output = os.path.join(
                state.diagnostics_dir,
                os.path.basename(state.output_path).replace('.tif', '_debug.jpg'),
            )
        else:
            debug_output = state.output_path.replace('.tif', '_debug.jpg')
        print(f"  Saving diagnostic visualization to {os.path.basename(debug_output)}...")
        with rasterio.open(state.reference_path) as src_ref:
            generate_debug_image(src_ref, src_offset, state.overlap, state.work_crs,
                                 state.matched_pairs, state.geo_residuals,
                                 state.mean_residual, debug_output)

        # Determinant check
        det = a * d - b_val * c
        if det < 0.01 or det > 100:
            print(f"  WARNING: Degenerate affine (det={det:.4f}), falling back to translation")
            offsets_e = np.array([p[2] - p[0] for p in state.matched_pairs])
            offsets_n = np.array([p[3] - p[1] for p in state.matched_pairs])
            med_de = np.median(offsets_e)
            med_dn = np.median(offsets_n)
            state.gcps = []
            for pair in state.matched_pairs:
                rgx, rgy, ogx, ogy = pair[0], pair[1], pair[2], pair[3]
                corrected_gx = ogx - med_de
                corrected_gy = ogy - med_dn
                col, row = _to_pixel(ogx, ogy)
                state.gcps.append((col, row, corrected_gx, corrected_gy))
            state.mean_residual = np.sqrt(np.mean((offsets_e - med_de) ** 2 + (offsets_n - med_dn) ** 2))
        else:
            _cross_validate_and_robust_refit(state)

        print()
    finally:
        src_offset.close()
    return state


def _tps_warp_gcps(input_path: str, output_path: str, gcps, work_crs, output_crs,
                   output_bounds, output_res: float) -> bool:
    """GDAL TPS warp using GCPs as a safe fallback.  Returns True on success.

    Two-step: warp to work_crs first (so outputBounds match), then reproject to
    output_crs — same strategy as apply_warp.
    """
    left, bottom, right, top = output_bounds
    needs_reproject = (output_crs is not None and str(output_crs) != str(work_crs))

    uid = uuid.uuid4().hex
    tmp_gcp = f"/vsimem/tps_fb_gcp_{uid}.tif"
    tmp_work = output_path + f".tps_work_{uid}.tif" if needs_reproject else output_path

    gdal.UseExceptions()
    try:
        gcp_list = [gdal.GCP(gx, gy, 0.0, float(px), float(py)) for px, py, gx, gy in gcps]
        ds = gdal.Open(input_path)
        if ds is None:
            return False
        tmp_ds = gdal.Translate(tmp_gcp, ds, GCPs=gcp_list,
                                outputSRS=str(work_crs))
        ds = None
        if tmp_ds is None:
            return False
        tmp_ds = None

        if os.path.exists(tmp_work):
            os.remove(tmp_work)

        # Step 1: TPS warp to work_crs (bounds are in work_crs)
        out_ds = gdal.Warp(
            tmp_work, tmp_gcp,
            tps=True,
            dstSRS=str(work_crs),
            outputBounds=(left, bottom, right, top),
            xRes=output_res, yRes=output_res,
            resampleAlg=gdal.GRA_Bilinear,
            multithread=True,
            warpOptions=['NUM_THREADS=ALL_CPUS'],
            creationOptions=['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES',
                             'NUM_THREADS=ALL_CPUS', 'BIGTIFF=YES'],
            warpMemoryLimit=2048 * 1024 * 1024,
        )
        if out_ds is None:
            return False
        out_ds = None

        # Step 2: reproject to output_crs if needed
        if needs_reproject:
            if os.path.exists(output_path):
                os.remove(output_path)
            rp_ds = gdal.Warp(
                output_path, tmp_work,
                dstSRS=str(output_crs),
                resampleAlg=gdal.GRA_Bilinear,
                multithread=True,
                warpOptions=['NUM_THREADS=ALL_CPUS'],
                creationOptions=['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES',
                                 'NUM_THREADS=ALL_CPUS', 'BIGTIFF=YES'],
                warpMemoryLimit=2048 * 1024 * 1024,
            )
            if rp_ds is None:
                return False
            rp_ds = None

        return True
    except Exception as e:
        print(f"  TPS fallback error: {e}", flush=True)
        return False
    finally:
        try:
            gdal.Unlink(tmp_gcp)
        except Exception:
            pass
        if needs_reproject and os.path.exists(tmp_work):
            try:
                os.remove(tmp_work)
            except Exception:
                pass


def _qa_label(qa) -> str:
    if qa is None:
        return "N/A"
    return (f"west={qa['west']:.0f}m center={qa['center']:.0f}m "
            f"east={qa['east']:.0f}m north={qa['north_shift']:+.0f}m "
            f"patch={qa['patch_med']:.0f}m stable_iou={qa.get('stable_iou', 0.0):.2f} "
            f"score={qa['score']:.0f}")


def step_select_warp_and_apply(state: AlignState) -> AlignState:
    """Step 5: Select warp mode and apply correction."""
    if state.overlap is None or state.work_crs is None:
        raise ValueError("Warp selection requires an established work region")
    if not state.yes:
        response = input("Apply this correction? [y/N] ")
        if response.lower() not in ("y", "yes"):
            if state.precorrection_tmp and os.path.exists(state.precorrection_tmp):
                os.remove(state.precorrection_tmp)
            raise UserAbortError("User declined warp application")

    # Output in Web Mercator for web overlay
    output_crs = CRS.from_epsg(OUTPUT_CRS_EPSG)
    qa_eval_res = max(2.0, min(6.0, max(state.offset_res_m, state.ref_res_m)))

    all_gcps = list(state.gcps) + list(state.boundary_gcps)
    print(f"Step 5: Grid Optimization Warp ({len(state.gcps)} GCPs "
          f"+ {len(state.boundary_gcps)} boundary = {len(all_gcps)} total)", flush=True)
    # Free neural models before grid optimizer + flow refinement to reclaim GPU memory
    if state.model_cache is not None:
        state.model_cache.close()
    apply_warp(state.current_input, state.output_path, state.reference_path, all_gcps,
               state.work_crs,
               output_bounds=state.overlap,
               output_res=state.offset_res_m,
               output_crs=output_crs,
               grid_size=state.grid_size,
               grid_iters=state.grid_iters,
               arap_weight=state.arap_weight,
               n_real_gcps_in=len(state.gcps))

    # ------------------------------------------------------------------
    # QA + TPS fallback gate
    # Compute QA on the grid result, then run a GDAL TPS warp of the
    # same GCPs.  Keep whichever scores better.  This guarantees output
    # is never worse than plain TPS.
    # ------------------------------------------------------------------
    qa_grid = None
    try:
        qa_grid = evaluate_alignment_quality_paths(
            state.output_path,
            state.reference_path,
            state.overlap,
            state.work_crs,
            eval_res=qa_eval_res,
            mask_mode=state.mask_provider,
        )
        if qa_grid is not None:
            print(f"  Grid warp QA: {_qa_label(qa_grid)}", flush=True)
    except Exception as e:
        print(f"  Grid QA failed: {e}", flush=True)

    # TPS fallback — only run when --tps-fallback is set
    tps_tmp = state.output_path.replace('.tif', '_tps_fallback.tif')
    tps_ok = False
    qa_tps = None
    if state.tps_fallback:
        try:
            print("  Running TPS fallback warp for comparison...", flush=True)
            tps_ok = _tps_warp_gcps(
                state.current_input, tps_tmp,
                all_gcps, state.work_crs, output_crs,
                state.overlap, state.offset_res_m,
            )
            if tps_ok:
                try:
                    flow_applied = apply_flow_refinement_to_file(
                        tps_tmp, state.reference_path,
                        state.work_crs, state.overlap,
                        state.offset_res_m,
                    )
                    if flow_applied:
                        print("  [FlowRefine-TPS] Applied flow refinement to TPS result", flush=True)
                except Exception as e:
                    print(f"  [FlowRefine-TPS] Skipped ({e})", flush=True)
                qa_tps = evaluate_alignment_quality_paths(
                    tps_tmp,
                    state.reference_path,
                    state.overlap,
                    state.work_crs,
                    eval_res=qa_eval_res,
                    mask_mode=state.mask_provider,
                )
                print(f"  TPS fallback QA:  {_qa_label(qa_tps)}", flush=True)
        except Exception as e:
            print(f"  TPS fallback failed: {e}", flush=True)

    state.qa_reports = []
    report_grid = build_candidate_report(
        "grid",
        state.output_path,
        state.reference_path,
        state.overlap,
        state.work_crs,
        holdout_pairs=state.qa_holdout_pairs,
        M_geo=state.M_geo,
        coverage=state.gcp_coverage,
        cv_mean_m=state.cv_mean,
        hypothesis_id=state.chosen_hypothesis.hypothesis_id if state.chosen_hypothesis else "",
        eval_res=qa_eval_res,
    )
    state.qa_reports.append(report_grid)
    print(
        f"  Grid independent QA: total={report_grid.total_score:.0f}, "
        f"confidence={report_grid.confidence:.2f}, accepted={report_grid.accepted}"
    )

    report_tps = None
    if state.tps_fallback and tps_ok and os.path.exists(tps_tmp):
        report_tps = build_candidate_report(
            "tps",
            tps_tmp,
            state.reference_path,
            state.overlap,
            state.work_crs,
            holdout_pairs=state.qa_holdout_pairs,
            M_geo=state.M_geo,
            coverage=state.gcp_coverage,
            cv_mean_m=state.cv_mean,
            hypothesis_id=state.chosen_hypothesis.hypothesis_id if state.chosen_hypothesis else "",
            eval_res=qa_eval_res,
        )
        state.qa_reports.append(report_tps)
        print(
            f"  TPS independent QA:  total={report_tps.total_score:.0f}, "
            f"confidence={report_tps.confidence:.2f}, accepted={report_tps.accepted}"
        )

    # Decide which result to keep
    selected_candidate = "grid"
    if state.tps_fallback:
        grid_score = report_grid.total_score
        tps_score = report_tps.total_score if report_tps is not None else float('inf')
        if tps_ok and tps_score < grid_score * 0.97:
            print(f"  TPS fallback is better (score {tps_score:.0f} vs grid {grid_score:.0f}) "
                  f"— using TPS result.", flush=True)
            os.replace(tps_tmp, state.output_path)
            selected_candidate = "tps"
        else:
            if grid_score < float('inf'):
                print(f"  Grid optimizer wins (score {grid_score:.0f} vs TPS {tps_score:.0f}).",
                      flush=True)
            if tps_ok and os.path.exists(tps_tmp):
                os.remove(tps_tmp)

    selected_report = report_grid if selected_candidate == "grid" else report_tps
    if selected_report is not None:
        print(
            f"  Selected candidate confidence={selected_report.confidence:.2f} "
            f"accepted={selected_report.accepted}"
        )

    qa_json_path = state.qa_json_path
    if not qa_json_path and (state.allow_abstain or state.diagnostics_dir):
        qa_json_path = state.output_path.replace('.tif', '_qa.json')
    if qa_json_path:
        write_qa_report(
            qa_json_path,
            state.qa_reports,
            selected_candidate=selected_candidate,
            metadata={
                "input_path": state.input_path,
                "current_input": state.current_input,
                "reference_path": state.reference_path,
                "gcp_count": len(state.gcps),
                "holdout_count": len(state.qa_holdout_pairs),
                "global_hypotheses": [hyp.to_dict() for hyp in state.global_hypotheses],
            },
        )
        print(f"  QA report written to: {qa_json_path}")

    if selected_report is not None and state.allow_abstain and (
        not selected_report.accepted or selected_report.confidence < 0.35
    ):
        state.abstained = True
        if state.precorrection_tmp and os.path.exists(state.precorrection_tmp):
            os.remove(state.precorrection_tmp)
        if os.path.exists(state.output_path):
            os.remove(state.output_path)
        print(
            "\nAlignment abstained: independent QA marked this result as low confidence."
        )
        return state

    if state.precorrection_tmp and os.path.exists(state.precorrection_tmp):
        os.remove(state.precorrection_tmp)

    print(f"\nAligned image written to: {state.output_path}")

    return state


def step_post_refinement(state: AlignState) -> None:
    """Step 6: Suggest manual verification."""
    if state.abstained:
        print("\nNo output retained because the run abstained on QA grounds.")
        return
    print("\nTo verify, run again on the output:")
    print(f"  python auto-align.py {state.output_path} -r {state.reference_path}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run(args, model_cache=None) -> str:
    """Main pipeline orchestrator.  Returns output path."""
    pipeline_t0 = time.time()
    state = None

    def _timed(name, fn, *a, **kw):
        t0 = time.time()
        result = fn(*a, **kw)
        print(f"  [{name}] {time.time() - t0:.1f}s", flush=True)
        return result

    state = _timed("setup", step_setup, args)

    # Attach or create model cache
    if model_cache is not None:
        state.model_cache = model_cache
    elif state.model_cache is None:
        device_override = getattr(args, 'device', 'auto')
        device = get_torch_device(override=device_override if device_override != 'auto' else None)
        state.model_cache = ModelCache(device)

    try:
        state = _timed("global_localization", step_global_localization, state, args)
        state = _timed("coarse_offset", step_coarse_offset, state)
        state = _timed("handle_large_offset", step_handle_large_offset, state, args)
        state = _timed("scale_rotation", step_scale_rotation, state, args)
        state = _timed("feature_matching", step_feature_matching, state, args)
        state = _timed("validate_and_filter", step_validate_and_filter, state)
        state = _timed("select_warp_and_apply", step_select_warp_and_apply, state)
        _timed("post_refinement", step_post_refinement, state)
    except AlreadyAlignedError as e:
        print(f"\n  {e}", flush=True)
    except UserAbortError as e:
        print(f"\n  Aborted: {e}", flush=True)
    except AlignmentError as e:
        print(f"\nERROR: {e}", flush=True)
        raise
    finally:
        # Only close if we created the cache (not passed in)
        if model_cache is None and state is not None and state.model_cache is not None:
            state.model_cache.close()
        if state is not None:
            for temp_path in state.temp_paths:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

    print(f"\n  [TOTAL] {time.time() - pipeline_t0:.1f}s", flush=True)
    return state.output_path if state is not None else getattr(args, "output", "")
