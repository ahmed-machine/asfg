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
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import rasterio
import rasterio.transform
from osgeo import gdal
from pyproj import Transformer
from rasterio.crs import CRS
from . import constants as _C
from .geo import (
    WEB_MERCATOR,
    WGS84,
    clear_overlap_cache,
    compute_overlap,
    compute_overlap_or_none,
    dataset_bounds_in_crs,
    fit_affine_from_gcps,
    generate_boundary_gcps,
    get_metric_crs,
    get_native_resolution_m,
    get_torch_device,
    open_pair,
    read_overlap_pair,
    read_overlap_region,
    work_shift_to_dataset_shift,
)
from .global_localization import localize_to_reference, translate_input_to_hypothesis
from .image import shift_array, to_u8, is_cloudy_patch, make_land_mask, clahe_normalize
from .metadata_priors import load_metadata_priors
from .models import ModelCache
from .coarse import detect_offset_at_resolution
from .scale import (detect_scale_rotation, detect_local_scales,
                    apply_scale_rotation_precorrection, apply_local_scale_precorrection)
from .anchors import locate_anchors
from .matching import match_with_roma
from .filtering import (matched_pairs_sufficient, refine_matches_phase_correlation,
                        correct_reference_offset, select_best_gcps,
                        iterative_outlier_removal,
                        detect_and_correct_reference_offset,
                        local_consistency_filter)
from .errors import (AlignmentError, AlreadyAlignedError, CoarseOffsetError,
                     InsufficientDataError, UserAbortError, WarpError)
from .flow_refine import apply_flow_refinement_to_file
from .tin_filter import filter_by_tin_tarr, optimize_fpps_accuracy
from .profiler import PipelineProfiler, _NullProfiler
from .warp import apply_warp
from .checkpoint import save_checkpoint
from .qa import evaluate_alignment_quality_paths, generate_debug_image
from .qa_runner import build_candidate_report, split_holdout_pairs, write_qa_report
from .types import BBox, GCP, GlobalHypothesis, MaskProvider, MatchPair, MetadataPrior, QaReport


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
    overlap: Optional[BBox] = None
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

    # Scale/rotation detection quality (populated by step_scale_rotation)
    scale_valid_patches: int = 0
    scale_total_patches: int = 0
    scale_x_spread: float = 0.0
    scale_y_spread: float = 0.0

    matched_pairs: list[MatchPair] = field(default_factory=list)
    gcps: list[GCP] = field(default_factory=list)
    match_weights: Optional[np.ndarray] = None  # per-GCP precision weights from RoMa
    ransac_survivor_count: int = 0  # auto matches surviving geographic RANSAC
    match_quality_residual: float = float('inf')  # affine residual from quality check
    boundary_gcps: list[GCP] = field(default_factory=list)
    geo_residuals: list[float] = field(default_factory=list)
    mean_residual: float = float('inf')
    max_residual: float = float('inf')
    gcp_coverage: float = 1.0
    used_neural: bool = False
    use_sift_refinement: bool = False

    was_corrected: bool = False
    needs_ortho: bool = False             # deferred orthorectification pending
    rough_georef_path: Optional[str] = None  # path to rough georef for ortho
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
    mask_provider: MaskProvider = MaskProvider.COASTAL_OBIA
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
    qa_holdout_pairs: list[MatchPair] = field(default_factory=list)
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

def _assign_rough_georeference(input_path: str, prior_bounds: tuple,
                               width: int, height: int) -> str:
    """Create a temporary copy of *input_path* with a rough geotransform.

    Uses the metadata prior bounding box (EPSG:4326) to assign a simple
    affine transform so downstream code can reproject the raw scan.
    Returns the path to the temporary georeferenced file.
    """
    west, south, east, north = prior_bounds
    transform = rasterio.transform.from_bounds(west, south, east, north, width, height)
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(crs=WGS84, transform=transform)

        base, ext = os.path.splitext(input_path)
        tmp_path = f"{base}.rough_georef{ext}"
        with rasterio.open(tmp_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                for _, window in src.block_windows(band_idx):
                    data = src.read(band_idx, window=window)
                    dst.write(data, band_idx, window=window)
    return tmp_path


def _orthorectify_against_reference(rough_georef_path: str, reference_path: str,
                                     work_crs, overlap: tuple,
                                     coarse_dx: float = 0.0,
                                     coarse_dy: float = 0.0,
                                     model_cache=None) -> str | None:
    """Automatic orthorectification using reference-based feature matching.

    When a raw scan has only a rough affine georef from metadata, this function
    uses RoMa neural matching against the reference image to find dense
    correspondences, then TPS-warps the scan to correct panoramic distortion,
    rotation, scale, and translation errors simultaneously.

    Parameters
    ----------
    coarse_dx, coarse_dy : float
        Coarse translation offset in metres (east, south) from prior
        alignment steps.  Applied as pixel shifts to the offset array
        before neural matching so that corresponding terrain overlaps.
    model_cache : ModelCache, optional
        Shared model cache.  If None a temporary one is created.

    Returns the path to the orthorectified file, or None if insufficient
    matches were found (caller should keep the rough georef).
    """
    import cv2
    from .matching import match_with_roma
    from .models import ModelCache, get_torch_device

    ortho_res = 5.0  # matching resolution for orthorectification (m/px)
    min_matches = 30

    # Convert coarse metric offset to pixel shifts at ortho_res
    shift_px_x = int(round(coarse_dx / ortho_res))
    shift_py_y = int(round(coarse_dy / ortho_res))

    print(f"  [Orthorectify] Matching against reference at {ortho_res}m/px "
          f"(coarse shift: dx={coarse_dx:+.0f}m dy={coarse_dy:+.0f}m → "
          f"px shift {shift_px_x:+d},{shift_py_y:+d})...",
          flush=True)

    src_off = rasterio.open(rough_georef_path)
    src_ref = rasterio.open(reference_path)

    try:
        arr_off, off_transform = read_overlap_region(
            src_off, overlap, work_crs, ortho_res)
        arr_ref, ref_transform = read_overlap_region(
            src_ref, overlap, work_crs, ortho_res)
    except Exception as e:
        print(f"  [Orthorectify] Failed to read overlap: {e}", flush=True)
        src_off.close()
        src_ref.close()
        return None

    # Apply coarse shift to offset array so it overlaps with reference
    if shift_px_x != 0 or shift_py_y != 0:
        arr_off = shift_array(arr_off, -shift_px_x, -shift_py_y)

    valid_off = float(np.mean(arr_off > 0))
    valid_ref = float(np.mean(arr_ref > 0))
    if valid_off < 0.05 or valid_ref < 0.05:
        print(f"  [Orthorectify] Insufficient overlap "
              f"(off={valid_off:.0%}, ref={valid_ref:.0%})", flush=True)
        src_off.close()
        src_ref.close()
        return None

    # Use shared model cache or create a temporary one
    cache = model_cache if model_cache is not None else ModelCache(get_torch_device())

    # Run RoMa matching — pass shifted arrays and coarse pixel shifts
    try:
        matches = match_with_roma(
            arr_ref, arr_off, ref_transform, off_transform,
            shift_px_x=shift_px_x, shift_py_y=shift_py_y,
            neural_res=ortho_res,
            model_cache=cache,
            src_offset=src_off, work_crs=work_crs,
        )
    except Exception as e:
        print(f"  [Orthorectify] RoMa matching failed: {e}", flush=True)
        src_off.close()
        src_ref.close()
        return None

    src_off.close()
    src_ref.close()

    print(f"  [Orthorectify] RoMa found {len(matches)} matches", flush=True)

    if len(matches) < min_matches:
        print(f"  [Orthorectify] Only {len(matches)} matches (need {min_matches}), "
              f"keeping rough georef", flush=True)
        return None

    # RANSAC filter
    src_pts = np.array([(m.off_x, m.off_y) for m in matches],
                       dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array([(m.ref_x, m.ref_y) for m in matches],
                       dtype=np.float32).reshape(-1, 1, 2)
    _, inliers = cv2.estimateAffine2D(
        src_pts, dst_pts, method=cv2.RANSAC,
        ransacReprojThreshold=30.0, confidence=0.95)
    if inliers is None:
        print(f"  [Orthorectify] RANSAC failed", flush=True)
        return None
    inlier_mask = inliers.ravel().astype(bool)
    inlier_matches = [m for m, k in zip(matches, inlier_mask) if k]
    print(f"  [Orthorectify] {len(inlier_matches)}/{len(matches)} RANSAC inliers",
          flush=True)

    if len(inlier_matches) < min_matches:
        print(f"  [Orthorectify] Only {len(inlier_matches)} inliers, keeping rough georef",
              flush=True)
        return None

    # Convert to GDAL GCPs: offset pixel coords → reference geographic coords
    # The offset coords (m.off_x/y) are in work_crs (UTM meters).
    # We need pixel coords in the rough georef, and geographic coords for the GCP target.
    from rasterio.warp import transform as _transform
    ref_xs = [m.ref_x for m in inlier_matches]
    ref_ys = [m.ref_y for m in inlier_matches]
    # Transform reference coords from work_crs to EPSG:4326 for GCP targets
    ref_lons, ref_lats = _transform(work_crs, WGS84, ref_xs, ref_ys)

    with rasterio.open(rough_georef_path) as src:
        rough_transform = src.transform

    gcp_list = []
    for i, m in enumerate(inlier_matches):
        # Convert offset work_crs coords to pixel coords in the rough georef
        # First transform from work_crs to the rough georef CRS (EPSG:4326)
        off_lons, off_lats = _transform(work_crs, WGS84,
                                        [m.off_x], [m.off_y])
        px_col, px_row = ~rough_transform * (off_lons[0], off_lats[0])
        gcp_list.append(gdal.GCP(ref_lons[i], ref_lats[i], 0,
                                 float(px_col), float(px_row)))

    # TPS warp — follow the pattern from apply_scale_rotation_precorrection():
    # let GDAL auto-determine bounds from the GCPs (no forced outputBounds).
    # Set explicit resolution from the reference to avoid inheriting the
    # rough georef's wrong pixel size.
    base, ext = os.path.splitext(rough_georef_path)
    tmp_gcp_path = base + ".ortho_gcps" + ext
    ortho_path = base.replace(".rough_georef", ".orthorectified") + ext

    ds_in = gdal.Open(rough_georef_path)
    ds_gcp = gdal.Translate(tmp_gcp_path, ds_in, GCPs=gcp_list,
                            outputSRS="EPSG:4326")
    ds_gcp = None
    ds_in = None

    # Use reference pixel size so the ortho output has consistent scale
    with rasterio.open(reference_path) as ref:
        ref_res = abs(ref.transform.a)  # degrees per pixel

    warp_kwargs = {
        "dstSRS": "EPSG:4326",
        "format": "GTiff",
        "tps": True,
        "xRes": ref_res,
        "yRes": ref_res,
        "resampleAlg": "bilinear",
        "multithread": True,
        "creationOptions": ["COMPRESS=LZW", "TILED=YES", "PREDICTOR=2",
                            "BIGTIFF=YES", "NUM_THREADS=ALL_CPUS"],
        "warpMemoryLimit": 2048 * 1024 * 1024,
    }

    ds_out = gdal.Warp(ortho_path, tmp_gcp_path, **warp_kwargs)
    ds_out = None

    if os.path.exists(tmp_gcp_path):
        os.remove(tmp_gcp_path)

    if not os.path.exists(ortho_path):
        print(f"  [Orthorectify] GDAL warp failed", flush=True)
        return None

    with rasterio.open(ortho_path) as src:
        print(f"  [Orthorectify] Success: {len(inlier_matches)} GCPs, "
              f"size {src.width}x{src.height}, "
              f"res {abs(src.transform.a)*111000:.1f}m/px, "
              f"bounds {src.bounds.left:.3f}-{src.bounds.right:.3f}E",
              flush=True)

    return ortho_path


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

    # When target has no CRS, assign a rough geotransform from metadata priors.
    # Orthorectification is deferred to step_orthorectify (after coarse offset)
    # so that coarse translation shifts are available for RoMa matching.
    rough_georef_tmp = None
    ortho_applied = False
    needs_ortho = False
    if src_offset.crs is None and priors:
        from .metadata_priors import merge_prior_bounds
        prior_bounds = merge_prior_bounds(priors)
        if prior_bounds is not None:
            rough_georef_tmp = _assign_rough_georeference(
                args.input, prior_bounds, src_offset.width, src_offset.height)
            src_offset.close()
            args.input = rough_georef_tmp
            src_offset = rasterio.open(args.input)
            print(f"  Assigned rough georeference from metadata priors")
            print(f"  CRS: {src_offset.crs}, Bounds: {src_offset.bounds}")
            needs_ortho = True

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

    offset_res_m = get_native_resolution_m(src_offset, priors=priors)
    ref_res_m = get_native_resolution_m(src_ref)
    expected_scale = offset_res_m / ref_res_m
    if ortho_applied:
        # Orthorectification already corrected scale/rotation/distortion.
        # Force expected_scale=1.0 to skip scale/rotation detection.
        expected_scale = 1.0
        print(f"Native resolution: target={offset_res_m:.2f} m/px, reference={ref_res_m:.2f} m/px")
        print(f"Expected scale ratio: {expected_scale:.3f}x (ortho-corrected)")
    else:
        print(f"Native resolution: target={offset_res_m:.2f} m/px, reference={ref_res_m:.2f} m/px")
        print(f"Expected scale ratio: {expected_scale:.3f}x")
    print()

    target_bounds_work = dataset_bounds_in_crs(src_offset, work_crs)
    reference_bounds_work = dataset_bounds_in_crs(src_ref, work_crs)
    _overlap_raw = compute_overlap_or_none(src_offset, src_ref, work_crs)
    overlap = BBox(*_overlap_raw) if _overlap_raw is not None else None
    if overlap is not None:
        overlap_w = overlap.right - overlap.left
        overlap_h = overlap.top - overlap.bottom
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

    state = AlignState(
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
        mask_provider=MaskProvider(getattr(args, 'mask_provider', 'coastal_obia')),
        global_search=bool(getattr(args, 'global_search', True)),
        global_search_res=float(getattr(args, 'global_search_res', 40.0)),
        global_search_top_k=int(getattr(args, 'global_search_top_k', 3)),
        force_global=bool(getattr(args, 'force_global', False)),
        reference_window=reference_window,
        metadata_prior_paths=metadata_prior_paths,
        metadata_priors=priors,
        reference_bounds_work=reference_bounds_work,
        target_bounds_work=target_bounds_work,
        was_corrected=ortho_applied,
        needs_ortho=needs_ortho,
        rough_georef_path=rough_georef_tmp,
        qa_json_path=getattr(args, 'qa_json', None),
        diagnostics_dir=getattr(args, 'diagnostics_dir', None),
        allow_abstain=bool(getattr(args, 'allow_abstain', False)),
        tps_fallback=bool(getattr(args, 'tps_fallback', False)),
    )
    if rough_georef_tmp is not None:
        state.temp_paths.append(rough_georef_tmp)
    return state


def _refresh_work_region(state: AlignState) -> AlignState:
    """Recompute work CRS bounds and overlap for the current input."""

    with open_pair(state.current_input, state.reference_path) as (src_offset, src_ref):
        state.work_crs = get_metric_crs(src_offset, src_ref, priors=state.metadata_priors)
        state.target_bounds_work = dataset_bounds_in_crs(src_offset, state.work_crs)
        state.reference_bounds_work = dataset_bounds_in_crs(src_ref, state.work_crs)
        _raw = compute_overlap_or_none(src_offset, src_ref, state.work_crs)
        state.overlap = BBox(*_raw) if _raw is not None else None
    return state


def step_global_localization(state: AlignState, args) -> AlignState:
    """Add a coarse localization stage before overlap-dependent matching."""

    if state.overlap is not None and not state.force_global:
        state.global_hypotheses = [
            GlobalHypothesis(
                hypothesis_id="rough_overlap",
                score=1.0,
                source="rough_georef_overlap",
                left=state.overlap.left,
                bottom=state.overlap.bottom,
                right=state.overlap.right,
                top=state.overlap.top,
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
        overlap_w = state.overlap.right - state.overlap.left
        overlap_h = state.overlap.top - state.overlap.bottom
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
        arr_ref, _ = read_overlap_region(src_ref, state.overlap.as_tuple(), state.work_crs, 5.0)
        arr_off, _ = read_overlap_region(src_offset, state.overlap.as_tuple(), state.work_crs, 5.0)

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


def step_coarse_offset(state: AlignState, profiler=None) -> AlignState:
    """Step 1: Detect coarse offset via land mask matching."""
    _p = profiler or _NullProfiler()
    if state.overlap is None or state.work_crs is None:
        raise ValueError("Coarse offset requires an established work region")
    print("Step 1: Coarse offset detection via land mask matching", flush=True)

    src_offset = rasterio.open(state.current_input)
    src_ref = rasterio.open(state.reference_path)

    from .params import get_params as _get_params  # noqa: used in multiple functions
    _coarse_p = _get_params().coarse
    print(f"  Pass 1: coarse scan at {_C.COARSE_RES} m/px...", flush=True)
    with _p.section("pass1"):
        dx_c, dy_c, corr_c = detect_offset_at_resolution(
            src_offset, src_ref, state.overlap.as_tuple(), state.work_crs, _C.COARSE_RES,
            template_radius_m=_C.DEFAULT_TEMPLATE_RADIUS_M, mask_mode=state.mask_provider,
            diagnostics_dir=state.diagnostics_dir, min_ncc=_coarse_p.min_ncc)
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
            if state.diagnostics_dir:
                save_checkpoint(state, "post_coarse",
                                os.path.join(state.diagnostics_dir, "checkpoints"))
            return state
        src_offset.close()
        src_ref.close()
        raise CoarseOffsetError("Could not detect offset at 15m resolution")
    
    # We now know dx_c and dy_c are floats
    dx_c_f, dy_c_f = float(dx_c), float(dy_c)
    print(f"    dx={dx_c_f:+8.0f}m, dy={dy_c_f:+8.0f}m (corr={corr_c:.4f})")

    # Pass 2: refine with adaptive search margin expansion.
    # If the initial margin is insufficient (e.g. large metadata error), we
    # progressively double it up to 4× the default, giving the matcher a
    # chance to recover before giving up.
    search_margin = _C.DEFAULT_SEARCH_MARGIN_M
    max_search_margin = _C.DEFAULT_SEARCH_MARGIN_M * 4
    dx_r, dy_r, corr_r = None, None, None
    with _p.section("pass2"):
        while search_margin <= max_search_margin:
            print(f"  Pass 2: refine at {_C.REFINE_RES} m/px "
                  f"(search +/-{search_margin:.0f}m around coarse)...")
            dx_r, dy_r, corr_r = detect_offset_at_resolution(
                src_offset, src_ref, state.overlap.as_tuple(), state.work_crs, _C.REFINE_RES,
                template_radius_m=_C.DEFAULT_TEMPLATE_RADIUS_M,
                coarse_offset=(dx_c_f, dy_c_f),
                search_margin_m=search_margin,
                mask_mode=state.mask_provider,
                diagnostics_dir=state.diagnostics_dir,
                min_ncc=_coarse_p.min_ncc)
            if (dx_r is not None and dy_r is not None and
                    corr_r is not None and corr_r > _coarse_p.min_ncc):
                break
            # Expand margin and retry
            next_margin = search_margin * 2
            if next_margin <= max_search_margin:
                print(f"    Refinement failed at +/-{search_margin:.0f}m, "
                      f"expanding to +/-{next_margin:.0f}m...")
            search_margin = next_margin

    if dx_r is not None and dy_r is not None and corr_r is not None and corr_r > _coarse_p.min_ncc:
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
        overlap_w = state.overlap.right - state.overlap.left
        overlap_h = state.overlap.top - state.overlap.bottom
        max_reasonable = 0.5 * min(overlap_w, overlap_h)
        if state.coarse_total > max_reasonable:
            print(f"  WARNING: Coarse offset {state.coarse_total:.0f}m exceeds "
                  f"half the overlap extent ({max_reasonable:.0f}m) "
                  f"— estimate may be unreliable")

    print()

    if state.diagnostics_dir:
        save_checkpoint(state, "post_coarse",
                        os.path.join(state.diagnostics_dir, "checkpoints"))

    return state


def step_orthorectify(state: AlignState) -> AlignState:
    """Step 1.25: Deferred orthorectification using coarse offset.

    Runs only when the input was a raw scan that received a rough georeference
    from metadata priors (needs_ortho=True).  Now that coarse_dx/coarse_dy are
    known, the neural matcher can receive properly-shifted patches and find
    dense correspondences for TPS warping.
    """
    if not state.needs_ortho:
        return state

    if state.overlap is None:
        print("  [Orthorectify] Skipped — no overlap region available", flush=True)
        return state

    print("Step 1.25: Deferred orthorectification (post coarse offset)", flush=True)

    ortho_path = _orthorectify_against_reference(
        state.current_input,
        state.reference_path,
        state.work_crs,
        state.overlap.as_tuple(),
        coarse_dx=state.coarse_dx,
        coarse_dy=state.coarse_dy,
        model_cache=state.model_cache,
    )

    if ortho_path is not None:
        state.current_input = ortho_path
        state.temp_paths.append(ortho_path)
        state.was_corrected = True
        state.needs_ortho = False

        # Invalidate overlap cache and refresh after orthorectification
        clear_overlap_cache()
        state = _refresh_work_region(state)

        # Orthorectification corrected scale/rotation — skip re-detection
        state.expected_scale = 1.0

        # Re-detect coarse offset on the orthorectified image (residual should
        # be much smaller now)
        print("  Re-detecting coarse offset on orthorectified image...", flush=True)
        state = _redetect_coarse_after_precorrection(state)

        print(f"  [Orthorectify] Pipeline continuing with orthorectified image",
              flush=True)
    else:
        print("  [Orthorectify] Could not orthorectify — continuing with rough georef",
              flush=True)

    print()
    return state


def step_handle_large_offset(state: AlignState, args) -> AlignState:
    """Handle already-aligned or very large offsets (>2km)."""
    if state.coarse_total < 10:
        if abs(state.expected_scale - 1.0) < 0.05 and not state.was_corrected:
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
            # Convert metric offset to source CRS units.
            # coarse_dx > 0 means offset image is east of reference →
            # shift bounds west (subtract); coarse_dy > 0 means south →
            # shift bounds north (add). work_shift_to_dataset_shift maps
            # (dx_m, dy_m) such that positive dx shifts west, positive dy
            # shifts north.
            ds_dx, ds_dy = work_shift_to_dataset_shift(
                src, state.work_crs, state.coarse_dx, state.coarse_dy)
            left = src.bounds.left + ds_dx
            bottom = src.bounds.bottom + ds_dy
            right = src.bounds.right + ds_dx
            top = src.bounds.top + ds_dy
            unit = "°" if src.crs.is_geographic else "m"
            print(f"  Bounds shift: dx={ds_dx:.6f}{unit}, dy={ds_dy:.6f}{unit} "
                  f"(new center: {(left+right)/2:.4f}, {(bottom+top)/2:.4f})")
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


def step_scale_rotation(state: AlignState, args, profiler=None) -> AlignState:
    """Step 1.5: Scale and rotation detection and pre-correction."""
    from .params import get_params as _get_params
    _p = profiler or _NullProfiler()
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
    _sr_p = _get_params().scale_rotation
    print(f"  Attempting patch-based local scale detection ({_sr_p.grid_cols}x{_sr_p.grid_rows} grid)...")
    with _p.section("local_scales"):
        try:
            src_offset = rasterio.open(state.current_input)
            src_ref = rasterio.open(state.reference_path)
            local_patches = detect_local_scales(
                src_offset, src_ref, state.overlap.as_tuple(), state.work_crs,
                state.coarse_dx, state.coarse_dy,
                grid_cols=_sr_p.grid_cols, grid_rows=_sr_p.grid_rows,
                model_cache=state.model_cache)
            src_offset.close()
            src_ref.close()
        except Exception as e:
            print(f"  Local scale detection error: {e}")
            local_patches = None

    if local_patches is not None:
        with _p.section("apply_precorrection"):
            state = _apply_local_precorrection(state, local_patches)
    else:
        print("  Local scale detection returned None")

    # --- Global fallback ---
    if local_patches is None and not state.precorrection_applied:
        with _p.section("global_scale_rotation"):
            state = _apply_global_precorrection(state, args)

    # Last resort: if no detection method worked but expected_scale is far from 1.0,
    # apply expected_scale directly.  Better than proceeding with wrong scale — all
    # downstream matching would fail at >20% scale mismatch.
    if not state.precorrection_applied and abs(state.expected_scale - 1.0) > 0.15:
        print(f"  Scale/rotation detection failed — applying expected_scale "
              f"{state.expected_scale:.3f} as fallback", flush=True)
        from .scale import apply_scale_rotation_precorrection
        with _p.section("apply_expected_scale"):
            overlap_center = None
            if state.overlap:
                overlap_center = ((state.overlap.left + state.overlap.right) / 2,
                                  (state.overlap.bottom + state.overlap.top) / 2)
            result = apply_scale_rotation_precorrection(
                state.current_input, state.expected_scale, 0.0,
                state.work_crs, overlap_center=overlap_center)
            if result is not None:
                state.precorrection_applied = True
                state.precorrection_tmp = result
                state.current_input = result
                print(f"  Applied expected_scale={state.expected_scale:.3f} precorrection")

    # Invalidate overlap cache after precorrection changes
    if state.precorrection_applied:
        clear_overlap_cache()

    print()
    return state


def _weighted_median(values, weights):
    """Weighted median: value where cumulative weight reaches 50%."""
    arr = np.asarray(values)
    w = np.asarray(weights, dtype=float)
    idx = np.argsort(arr)
    cum = np.cumsum(w[idx])
    return float(arr[idx][cum >= cum[-1] / 2.0][0])


def _apply_local_precorrection(state: AlignState, local_patches) -> AlignState:
    """Apply local scale/rotation pre-correction from patch results."""
    valid_patches = [p for p in local_patches
                     if p['status'] in ('ok', 'filled-neighbor', 'filled-global')]

    _sx = np.array([p['scale_x'] for p in valid_patches])
    _sy = np.array([p['scale_y'] for p in valid_patches])
    _rot = np.array([p['rotation'] for p in valid_patches])
    _w = np.array([max(p.get('n_inliers', 1), 1) for p in valid_patches], dtype=float)
    avg_sx = _weighted_median(_sx, _w)
    avg_sy = _weighted_median(_sy, _w)
    avg_rot = _weighted_median(_rot, _w)
    avg_scale = (avg_sx + avg_sy) / 2

    ok_patches = [p for p in local_patches if p['status'] == 'ok']
    sx_spread = sy_spread = 0.0
    if len(ok_patches) >= 2:
        sx_spread = max(p['scale_x'] for p in ok_patches) - min(p['scale_x'] for p in ok_patches)
        sy_spread = max(p['scale_y'] for p in ok_patches) - min(p['scale_y'] for p in ok_patches)
        print(f"  Local scale variation: sx_spread={sx_spread:.4f}, sy_spread={sy_spread:.4f}")

    # Store scale detection quality metrics for proxy scoring
    state.scale_total_patches = len(local_patches)
    state.scale_valid_patches = len(valid_patches)
    state.scale_x_spread = sx_spread
    state.scale_y_spread = sy_spread

    print(f"  Average: scale={avg_scale:.4f} (x={avg_sx:.4f}, y={avg_sy:.4f}), "
          f"rotation={avg_rot:.3f} deg")

    if abs(avg_scale - 1.0) <= 0.02 and abs(avg_rot) <= 0.1:
        print("  Scale/rotation within tolerance, skipping pre-correction")
        return state

    overlap_cx = (state.overlap.left + state.overlap.right) / 2
    overlap_cy = (state.overlap.bottom + state.overlap.top) / 2
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
    try:
        result = detect_scale_rotation(
            src_offset, src_ref, state.overlap.as_tuple(), state.work_crs,
            state.coarse_dx, state.coarse_dy, state.expected_scale,
            model_cache=state.model_cache,
            diagnostics_dir=state.diagnostics_dir)
    finally:
        src_offset.close()
        src_ref.close()

    if result.method is None:
        print("  No significant scale/rotation detected")
        # Store metrics: global fallback found nothing
        state.scale_total_patches = 1
        state.scale_valid_patches = 0
        return state

    # Store metrics for proxy scoring (global fallback = 1 "patch")
    state.scale_total_patches = 1
    state.scale_valid_patches = 1
    state.scale_x_spread = 0.0
    state.scale_y_spread = 0.0

    det_avg = (result.scale_x + result.scale_y) / 2
    print(f"  Detected: scale={det_avg:.4f} (x={result.scale_x:.4f}, y={result.scale_y:.4f}), "
          f"rotation={result.rotation:.3f} deg (method: {result.method})")

    if abs(det_avg - 1.0) <= 0.02 and abs(result.rotation) <= 0.1:
        print("  Scale/rotation within tolerance, skipping pre-correction")
        return state

    overlap_cx = (state.overlap.left + state.overlap.right) / 2
    overlap_cy = (state.overlap.bottom + state.overlap.top) / 2
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
                src_offset, src_ref, state.overlap.as_tuple(), state.work_crs,
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
                _sx2 = np.array([p['scale_x'] for p in valid_p2])
                _sy2 = np.array([p['scale_y'] for p in valid_p2])
                _rot2 = np.array([p['rotation'] for p in valid_p2])
                _w2 = np.array([max(p.get('n_inliers', 1), 1) for p in valid_p2], dtype=float)
                avg_sx2 = _weighted_median(_sx2, _w2)
                avg_sy2 = _weighted_median(_sy2, _w2)
                avg_rot2 = _weighted_median(_rot2, _w2)
                avg_scale2 = (avg_sx2 + avg_sy2) / 2
                print(f"    Local patches: {len(ok_p2)} valid, "
                      f"avg scale={avg_scale2:.4f}, rotation={avg_rot2:.3f} deg")

                if abs(avg_scale2 - 1.0) > 0.005 or abs(avg_rot2) > 0.05:
                    overlap_cx = (state.overlap.left + state.overlap.right) / 2
                    overlap_cy = (state.overlap.bottom + state.overlap.top) / 2
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
                            state.overlap = BBox(*compute_overlap(src_offset, src_ref, state.work_crs))
                            src_offset.close()
                            src_ref.close()
                        else:
                            # Validate with SIFT
                            print("    Validating with SIFT match comparison...")
                            src_offset = rasterio.open(state.current_input)
                            src_ref = rasterio.open(state.reference_path)
                            overlap_after = BBox(*compute_overlap(src_offset, src_ref, state.work_crs))
                            src_offset.close()
                            src_ref.close()

                            sift_before = _quick_sift_count(
                                old_tmp, state.reference_path,
                                state.overlap.as_tuple(), state.work_crs)
                            sift_after = _quick_sift_count(
                                precorrected_2, state.reference_path,
                                overlap_after.as_tuple(), state.work_crs)
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
    from .params import get_params as _get_params
    coarse_before = np.sqrt(state.coarse_dx ** 2 + state.coarse_dy ** 2)
    print("  Re-detecting coarse offset on pre-corrected image...")
    src_offset = rasterio.open(state.current_input)
    src_ref = rasterio.open(state.reference_path)

    try:
        state.overlap = BBox(*compute_overlap(src_offset, src_ref, state.work_crs))
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
        src_offset, src_ref, state.overlap.as_tuple(), state.work_crs, 5.0,
        template_radius_m=_C.DEFAULT_TEMPLATE_RADIUS_M,
        min_ncc=_get_params().coarse.min_ncc)
    if dx_pc is not None and dy_pc is not None and corr_pc is not None and corr_pc > _get_params().coarse.min_ncc:
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

    if state.diagnostics_dir:
        checkpoint_dir = os.path.join(state.diagnostics_dir, "checkpoints")
        save_checkpoint(state, "post_scale_rotation", checkpoint_dir)

    return state


def _quick_sift_count(img_path, ref_path, ovlp, crs, res=5.0):
    """Count SIFT matches as alignment quality proxy."""
    with rasterio.open(img_path) as s_o, rasterio.open(ref_path) as s_r:
        a_r, _ = read_overlap_region(s_r, ovlp, crs, res)
        a_o, _ = read_overlap_region(s_o, ovlp, crs, res)
    sift = cv2.SIFT_create(nfeatures=2000) # type: ignore
    _, d1 = sift.detectAndCompute(to_u8(a_r), None)
    _, d2 = sift.detectAndCompute(to_u8(a_o), None)
    if d1 is None or d2 is None:
        return 0
    bf = cv2.BFMatcher()
    raw = bf.knnMatch(d1, d2, k=2)
    return sum(1 for m, n in raw if m.distance < 0.75 * n.distance)


def _save_match_debug_image(arr_ref, arr_off, ref_transform, off_transform,
                            matched_pairs, diagnostics_dir):
    """Save a debug image showing feature matches after RANSAC filtering."""
    try:
        ref_rgb = cv2.cvtColor(to_u8(arr_ref), cv2.COLOR_GRAY2BGR)
        off_rgb = cv2.cvtColor(to_u8(arr_off), cv2.COLOR_GRAY2BGR)

        # Downsample if too large (cap at 1024px on long side)
        max_dim = 1024
        h, w = ref_rgb.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            ref_rgb = cv2.resize(ref_rgb, (new_w, new_h))
            off_rgb = cv2.resize(off_rgb, (new_w, new_h))
        else:
            scale = 1.0

        n_anchor = 0
        n_auto = 0
        for pair in matched_pairs:
            rgx, rgy, ogx, ogy = pair.ref_x, pair.ref_y, pair.off_x, pair.off_y
            name = pair.name

            ref_row, ref_col = rasterio.transform.rowcol(ref_transform, rgx, rgy)
            off_row, off_col = rasterio.transform.rowcol(off_transform, ogx, ogy)
            ref_col = int(ref_col * scale)
            ref_row = int(ref_row * scale)
            off_col = int(off_col * scale)
            off_row = int(off_row * scale)

            if not (0 <= ref_row < ref_rgb.shape[0] and
                    0 <= ref_col < ref_rgb.shape[1] and
                    0 <= off_row < off_rgb.shape[0] and
                    0 <= off_col < off_rgb.shape[1]):
                continue

            if name.startswith("anchor:"):
                color = (0, 255, 255)  # yellow
                n_anchor += 1
                cv2.drawMarker(ref_rgb, (ref_col, ref_row), color,
                               cv2.MARKER_CROSS, 10, 2)
                cv2.drawMarker(off_rgb, (off_col, off_row), color,
                               cv2.MARKER_CROSS, 10, 2)
            else:
                color = (0, 220, 0)  # green
                n_auto += 1
                cv2.circle(ref_rgb, (ref_col, ref_row), 3, color, -1)
                cv2.circle(ref_rgb, (ref_col, ref_row), 5, (255, 255, 255), 1)
                cv2.circle(off_rgb, (off_col, off_row), 3, color, -1)
                cv2.circle(off_rgb, (off_col, off_row), 5, (255, 255, 255), 1)

        # Side-by-side
        h_ref, w_ref = ref_rgb.shape[:2]
        h_off, w_off = off_rgb.shape[:2]
        h_max = max(h_ref, h_off)
        if h_ref < h_max:
            ref_rgb = cv2.copyMakeBorder(
                ref_rgb, 0, h_max - h_ref, 0, 0, cv2.BORDER_CONSTANT)
        if h_off < h_max:
            off_rgb = cv2.copyMakeBorder(
                off_rgb, 0, h_max - h_off, 0, 0, cv2.BORDER_CONSTANT)
        combined = np.hstack([ref_rgb, off_rgb])

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Reference", (10, 25), font, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Offset (shifted)", (w_ref + 10, 25),
                    font, 0.7, (255, 255, 255), 2)
        cv2.putText(combined,
                    f"Post-RANSAC: {n_anchor} anchors + {n_auto} auto = {n_anchor + n_auto}",
                    (10, combined.shape[0] - 10), font, 0.6, (255, 255, 255), 2)

        out_path = os.path.join(diagnostics_dir, "feature_matches.jpg")
        cv2.imwrite(out_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  Feature match diagnostic saved: {out_path}")
    except Exception as e:
        print(f"  WARNING: Could not save feature match diagnostic: {e}")


def step_feature_matching(state: AlignState, args, profiler=None) -> AlignState:
    """Step 2: Feature matching (neural cascade -> NCC fallback)."""
    from .params import get_params as _get_params
    _p = profiler or _NullProfiler()
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

    # Memory safeguard — matching creates ~28 bytes per pixel across both images
    # (2× float32 overlap cache, 2× uint8 CLAHE, masks, Sobel gradients, weight maps).
    # Cap total array footprint at ~24 GB to leave headroom for models + OS.
    overlap_w = state.overlap.right - state.overlap.left
    overlap_h = state.overlap.top - state.overlap.bottom
    BYTES_PER_PIXEL = 28  # empirical: all arrays combined, both images
    MEM_CAP_BYTES = 24 * 1024 ** 3

    def _safe_res(res, label):
        npx = (overlap_w / res) * (overlap_h / res)
        mem = npx * BYTES_PER_PIXEL
        if mem > MEM_CAP_BYTES:
            min_res = (overlap_w * overlap_h * BYTES_PER_PIXEL / MEM_CAP_BYTES) ** 0.5
            min_res = max(min_res, 1.0)
            # Round up to nearest 0.5
            min_res = round(min_res * 2 + 0.49) / 2
            print(f"  Memory safeguard: {label} bumped {res:.1f} -> {min_res:.1f}m/px "
                  f"({mem / 1024**3:.1f}GB est. for {overlap_w:.0f}x{overlap_h:.0f}m overlap)")
            return min_res
        return res

    neural_res = _safe_res(neural_res, "neural_res")
    fine_res = _safe_res(fine_res, "fine_res")

    print(f"  Neural resolution: {neural_res:.1f}m/px, Fine resolution: {fine_res:.1f}m/px")

    # Neural feature matching
    print("Step 2: Neural feature matching cascade", flush=True)

    print(f"  Reading reference overlap at {neural_res}m/px...")
    arr_ref_neural, ref_neural_transform = read_overlap_region(
        src_ref, state.overlap.as_tuple(), state.work_crs, neural_res)
    print(f"  Reading offset overlap at {neural_res}m/px...")
    arr_off_neural, off_neural_transform = read_overlap_region(
        src_offset, state.overlap.as_tuple(), state.work_crs, neural_res)

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
        with _p.section("anchors"):
            try:
                anchor_pairs, anchor_meta = locate_anchors(
                    state.anchors_path, src_ref, src_offset, state.overlap.as_tuple(), state.work_crs,
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
        with _p.section("roma_dense"):
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
        with _p.section("phase_correlation"):
            print(f"    Reading reference overlap at {fine_res}m/px...")
            arr_ref_fine, ref_fine_transform = read_overlap_region(
                src_ref, state.overlap.as_tuple(), state.work_crs, fine_res)
            print(f"    Reading offset overlap at {fine_res}m/px...")
            arr_off_fine, off_fine_transform = read_overlap_region(
                src_offset, state.overlap.as_tuple(), state.work_crs, fine_res)
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
    with _p.section("ransac"):
        anchor_pairs = [m for m in state.matched_pairs if m.is_anchor]
        auto_pairs = [m for m in state.matched_pairs if not m.is_anchor]
        if len(auto_pairs) >= 6:
            geo_src = np.array([m.ref_coords() for m in auto_pairs],
                               dtype=np.float32).reshape(-1, 1, 2)
            geo_dst = np.array([m.off_coords() for m in auto_pairs],
                               dtype=np.float32).reshape(-1, 1, 2)
            _match_p = _get_params().matching
            geo_ransac_thresh = max(_match_p.ransac_reproj_threshold * neural_res, 20.0)
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
            state.ransac_survivor_count = len(auto_pairs)
            if anchor_pairs:
                print(f"  Anchors preserved: {len(anchor_pairs)} "
                      f"({', '.join(m.name.replace('anchor:', '') for m in anchor_pairs)})")

    # Confidence floor: reject low-quality auto matches (water-based, low certainty)
    if len(auto_pairs) >= 10:
        quality_floor = 0.25
        n_before_qf = len(auto_pairs)
        auto_pairs = [m for m in auto_pairs if m.confidence >= quality_floor]
        n_qf_rejected = n_before_qf - len(auto_pairs)
        if n_qf_rejected > 0:
            print(f"  Confidence floor: {n_before_qf} -> {len(auto_pairs)} "
                  f"({n_qf_rejected} below quality {quality_floor})")
        state.matched_pairs = anchor_pairs + auto_pairs

    # Local displacement consistency: reject auto matches whose offset
    # disagrees with their spatial neighbors (catches isolated outliers)
    if len(auto_pairs) >= 10:
        n_before_lc = len(auto_pairs)
        auto_pairs = local_consistency_filter(
            auto_pairs, anchor_pairs + auto_pairs,
            threshold_m=50.0, k_neighbors=5, search_radius=5000.0)
        n_lc_rejected = n_before_lc - len(auto_pairs)
        if n_lc_rejected > 0:
            print(f"  Local consistency filter: {n_before_lc} -> {len(auto_pairs)} "
                  f"({n_lc_rejected} spatially inconsistent)")
        state.matched_pairs = anchor_pairs + auto_pairs

    # Save feature match debug image (post-RANSAC, pre-GCP-selection)
    if state.diagnostics_dir and state.matched_pairs:
        _save_match_debug_image(
            arr_ref_neural, arr_off_neural_shifted,
            ref_neural_transform, off_neural_transform,
            state.matched_pairs, state.diagnostics_dir)

    # Quality check
    neural_quality_ok = False
    auto_only = [p for p in state.matched_pairs if not p.is_anchor]
    if len(auto_only) >= 6:
        q_src = np.array([p.off_coords() for p in auto_only])
        q_dst = np.array([p.ref_coords() for p in auto_only])
        _, q_residuals = fit_affine_from_gcps(q_src, q_dst)
        q_mean = np.mean(q_residuals)
        state.match_quality_residual = float(q_mean)
        print(f"  Neural match quality check: {len(auto_only)} matches, "
              f"mean affine residual={q_mean:.1f}m")
        if q_mean < 50:
            neural_quality_ok = True
        else:
            print(f"  Discarding {len(auto_only)} neural matches, keeping anchors")
            state.matched_pairs = [p for p in state.matched_pairs
                                   if p.is_anchor]
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
        with _p.section("gcp_selection"):
            state.matched_pairs, state.gcp_coverage = select_best_gcps(
                train_pairs, state.overlap.as_tuple(),
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
        with _p.section("second_neural_pass"):
            _second_neural_pass(
                state, src_ref, src_offset, neural_res, fine_res,
                arr_ref_fine, arr_off_fine_shifted,
                ref_fine_transform, off_fine_transform,
                shift_px_x_fine, shift_py_y_fine)

    # 2d: NCC fallback
    if not matched_pairs_sufficient(state.matched_pairs, target=10 if state.best else 15):
        with _p.section("ncc_fallback"):
            _ncc_fallback(state, src_ref, src_offset, neural_res)

    print()

    src_offset.close()
    src_ref.close()

    if state.diagnostics_dir:
        save_checkpoint(state, "post_match",
                        os.path.join(state.diagnostics_dir, "checkpoints"))

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
    if np.mean(template > 0) < _C.LAND_MASK_FRAC_MIN or is_cloudy_patch(template):
        return None
    sr0 = tr0 - search_margin
    sr1 = tr1 + search_margin
    sc0 = tc0 - search_margin
    sc1 = tc1 + search_margin
    if sr0 < 0 or sr1 > h_img or sc0 < 0 or sc1 > w_img:
        return None
    search = offset_shifted[sr0:sr1, sc0:sc1]
    if np.mean(search > 0) < _C.LAND_MASK_FRAC_MIN or is_cloudy_patch(search):
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
            src_ref, state.overlap.as_tuple(), state.work_crs, coarse_neural_res)
        arr_off_c, off_c_t = read_overlap_region(
            src_offset, state.overlap.as_tuple(), state.work_crs, coarse_neural_res)
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
                train_pairs, state.overlap.as_tuple(),
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
            src_ref, state.overlap.as_tuple(), state.work_crs, ncc_fine_res)
        arr_offset_match, offset_match_transform = read_overlap_region(
            src_offset, state.overlap.as_tuple(), state.work_crs, ncc_fine_res)

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
        try:
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                work_items = [(idx, r, c) for idx, (r, c) in enumerate(grid_points)]
                for result in pool.map(_ncc_match_worker, work_items, chunksize=4):
                    if result is not None:
                        ncc_pairs.append(result)
        except (BrokenProcessPool, Exception) as e:
            print(f"  WARNING: NCC fallback pool failed ({e}), skipping")
            ncc_pairs = []
        _worker_data = {}
        print(f"  {len(ncc_pairs)} NCC matches found")
        state.matched_pairs.extend(ncc_pairs)

        if len(state.matched_pairs) < 4:
            state.use_sift_refinement = False

    if not state.use_sift_refinement:
        print("Step 2 (NCC fallback): Land mask template matching at grid points", flush=True)
        match_res = state.match_res
        arr_ref_match, ref_match_transform = read_overlap_region(
            src_ref, state.overlap.as_tuple(), state.work_crs, match_res)
        arr_offset_match, offset_match_transform = read_overlap_region(
            src_offset, state.overlap.as_tuple(), state.work_crs, match_res)

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
        train_offset = np.array([p.off_coords() for p in train_pairs])
        train_ref = np.array([p.ref_coords() for p in train_pairs])
        train_weights = np.array([p.confidence for p in train_pairs])
        M_cv, _ = fit_affine_from_gcps(train_offset, train_ref, weights=train_weights)
        for i in test_idx:
            mp_i = state.matched_pairs[i]
            ogx, ogy = mp_i.off_x, mp_i.off_y
            rgx, rgy = mp_i.ref_x, mp_i.ref_y
            pred_x = M_cv[0, 0] * ogx + M_cv[0, 1] * ogy + M_cv[0, 2]
            pred_y = M_cv[1, 0] * ogx + M_cv[1, 1] * ogy + M_cv[1, 2]
            cv_errors.append(np.sqrt((pred_x - rgx) ** 2 + (pred_y - rgy) ** 2))

    state.cv_mean = float(np.mean(cv_errors))
    print(f"  Cross-validation ({k_folds}-fold, n={len(state.matched_pairs)}): "
          f"fit={state.mean_residual:.1f}m, "
          f"CV={state.cv_mean:.1f}m (threshold={residual_threshold:.0f}m)")

    # RANSAC robust refit when CV error is high
    if state.cv_mean <= _C.CV_REFIT_THRESHOLD_M or len(state.matched_pairs) < 8:
        return

    all_offset = np.array([p.off_coords() for p in state.matched_pairs])
    all_ref = np.array([p.ref_coords() for p in state.matched_pairs])
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
        inlier_mask = errs < _C.CV_REFIT_THRESHOLD_M
        n_in = int(inlier_mask.sum())
        if n_in > best_inliers:
            best_inliers = n_in
            best_mask = inlier_mask

    if best_inliers < 6:
        return

    inlier_weights = np.array([
        state.matched_pairs[i].confidence for i in range(n_pts) if best_mask[i]])
    M_robust, robust_res = fit_affine_from_gcps(
        all_offset[best_mask], all_ref[best_mask], weights=inlier_weights)
    robust_mean = float(np.mean(robust_res))

    if robust_mean < state.cv_mean * 0.5:
        state.M_geo = M_robust
        cv_errors_robust = []
        for i in range(n_pts):
            mp_i = state.matched_pairs[i]
            ogx, ogy = mp_i.off_x, mp_i.off_y
            rgx, rgy = mp_i.ref_x, mp_i.ref_y
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
    Returns list of MatchPair objects.
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
        ref_gx, ref_gy = bg.gx, bg.gy
        off_gx = M_inv[0, 0] * ref_gx + M_inv[0, 1] * ref_gy + M_inv[0, 2]
        off_gy = M_inv[1, 0] * ref_gx + M_inv[1, 1] * ref_gy + M_inv[1, 2]
        pairs.append(MatchPair(ref_x=ref_gx, ref_y=ref_gy,
                               off_x=off_gx, off_y=off_gy,
                               confidence=1.0, name=f"boundary:{bg.name}"))
    return pairs


def _reject_mad_outliers(state: AlignState) -> None:
    """Remove matched pairs whose offset deviates >N*MAD from the median.

    Anchors are always kept.  Modifies state.matched_pairs in place.
    """
    pairs = state.matched_pairs
    if len(pairs) <= 4:
        return

    offsets_e = np.array([p.off_x - p.ref_x for p in pairs])
    offsets_n = np.array([p.off_y - p.ref_y for p in pairs])

    med_e, med_n = np.median(offsets_e), np.median(offsets_n)
    mad_e = np.median(np.abs(offsets_e - med_e)) * 1.4826
    mad_n = np.median(np.abs(offsets_n - med_n)) * 1.4826

    min_sigma = min(200.0, max(20.0, state.coarse_total * 0.03))
    sigma_e = max(mad_e, min_sigma)
    sigma_n = max(mad_n, min_sigma)
    mad_sigma = (_C.MAD_SIGMA_SCALED if state.needs_scale_rotation
                 else _C.MAD_SIGMA)

    keep = []
    for i in range(len(pairs)):
        is_anchor = pairs[i].is_anchor
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

    offsets_e = np.array([p.off_x - p.ref_x for p in state.matched_pairs])
    offsets_n = np.array([p.off_y - p.ref_y for p in state.matched_pairs])
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
            """Convert matched_pairs → GCP list, with per-GCP precision weights."""
            gcps = []
            precisions = []
            for pair in pairs:
                col, row = _to_pixel(pair.off_x, pair.off_y)
                gcps.append(GCP(col=col, row=row, gx=pair.ref_x, gy=pair.ref_y,
                                source="match", name=pair.name))
                precisions.append(pair.precision)
            # Normalize: divide by median, clamp to [0.2, 3.0]
            prec_arr = np.array(precisions, dtype=np.float32)
            med = np.median(prec_arr) if len(prec_arr) > 0 else 1.0
            if med > 0:
                prec_arr = prec_arr / med
            prec_arr = np.clip(prec_arr, 0.2, 3.0)
            return gcps, prec_arr

        def _recompute_residuals(pairs, M):
            """Compute per-GCP affine residuals (metres) from M_geo."""
            if len(pairs) < 3 or M is None:
                return []
            residuals = []
            for p in pairs:
                ogx, ogy = p.off_x, p.off_y
                rgx, rgy = p.ref_x, p.ref_y
                pred_x = M[0, 0] * ogx + M[0, 1] * ogy + M[0, 2]
                pred_y = M[1, 0] * ogx + M[1, 1] * ogy + M[1, 2]
                residuals.append(np.sqrt((pred_x - rgx) ** 2 + (pred_y - rgy) ** 2))
            return residuals

        state.gcps, state.match_weights = _pairs_to_gcps(state.matched_pairs)

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
        state.gcps, state.match_weights = _pairs_to_gcps(state.matched_pairs)

        # Generate boundary GCPs for piecewise affine warp
        boundary_spacing = 2500
        if len(state.gcps) < 40:
            boundary_spacing = 3200
        if len(state.gcps) < 30:
            boundary_spacing = 3800
        raw_boundary = generate_boundary_gcps(
            state.gcps, state.M_geo,
            src_offset.width, src_offset.height, spacing_px=boundary_spacing)
        state.boundary_gcps = [
            GCP(col=bg[0], row=bg[1], gx=bg[2], gy=bg[3],
                synthetic=True, source="boundary")
            for bg in raw_boundary
        ]

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
        _tin_thresh = getattr(_C, 'TIN_TARR_THRESH', state.tin_tarr_thresh)
        if len(state.matched_pairs) >= 4 and boundary_pairs_geo:
            n_before_tin = len(state.matched_pairs)
            print(f"  TIN-TARR topological filter (threshold={_tin_thresh:.2f})...")
            state.matched_pairs = filter_by_tin_tarr(
                state.matched_pairs, boundary_pairs_geo,
                threshold=_tin_thresh)
            n_rejected_tin = n_before_tin - len(state.matched_pairs)
            if n_rejected_tin > 0:
                print(f"    Rejected {n_rejected_tin} GCPs with extreme mesh distortion")
                state.gcps, state.match_weights = _pairs_to_gcps(state.matched_pairs)
                state.geo_residuals = _recompute_residuals(
                    state.matched_pairs, state.M_geo)
            else:
                print(f"    All {n_before_tin} GCPs passed topological check")

        # --- FPP Accuracy Difference optimization (Phase B) ---
        _skip_fpp = state.skip_fpp
        if not _skip_fpp and len(state.matched_pairs) >= 6 and boundary_pairs_geo:
            fpp_res = max(state.offset_res_m, state.ref_res_m, 2.0)
            print(f"  FPP accuracy optimization at {fpp_res:.1f}m/px...")
            try:
                src_ref_fpp = rasterio.open(state.reference_path)
                arr_ref_fpp, ref_fpp_transform = read_overlap_region(
                    src_ref_fpp, state.overlap.as_tuple(), state.work_crs, fpp_res)
                arr_off_fpp, off_fpp_transform = read_overlap_region(
                    src_offset, state.overlap.as_tuple(), state.work_crs, fpp_res)
                src_ref_fpp.close()

                n_before_fpp = len(state.matched_pairs)
                state.matched_pairs, n_fpp_removed = optimize_fpps_accuracy(
                    state.matched_pairs, boundary_pairs_geo,
                    arr_ref_fpp, arr_off_fpp,
                    ref_fpp_transform, off_fpp_transform)
                if n_fpp_removed > 0:
                    print(f"    Removed {n_fpp_removed} GCPs that degraded local image similarity")
                    state.gcps, state.match_weights = _pairs_to_gcps(state.matched_pairs)
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
            generate_debug_image(src_ref, src_offset, state.overlap.as_tuple(), state.work_crs,
                                 state.matched_pairs, state.geo_residuals,
                                 state.mean_residual, debug_output)

        # Determinant check
        det = a * d - b_val * c
        if det < 0.01 or det > 100:
            print(f"  WARNING: Degenerate affine (det={det:.4f}), falling back to translation")
            offsets_e = np.array([p.off_x - p.ref_x for p in state.matched_pairs])
            offsets_n = np.array([p.off_y - p.ref_y for p in state.matched_pairs])
            med_de = np.median(offsets_e)
            med_dn = np.median(offsets_n)
            state.gcps = []
            for pair in state.matched_pairs:
                corrected_gx = pair.off_x - med_de
                corrected_gy = pair.off_y - med_dn
                col, row = _to_pixel(pair.off_x, pair.off_y)
                state.gcps.append(GCP(col=col, row=row, gx=corrected_gx, gy=corrected_gy,
                                      source="translation_fallback", name=pair.name))
            state.mean_residual = np.sqrt(np.mean((offsets_e - med_de) ** 2 + (offsets_n - med_dn) ** 2))
        else:
            _cross_validate_and_robust_refit(state)

        print()
    finally:
        src_offset.close()

    if state.diagnostics_dir:
        save_checkpoint(state, "post_validate",
                        os.path.join(state.diagnostics_dir, "checkpoints"))

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
        gcp_list = [gdal.GCP(g.gx, g.gy, 0.0, float(g.col), float(g.row)) for g in gcps]
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
    def _v(key):
        v = qa.get(key)
        return f"{key}={v:.0f}m" if v is not None else f"{key}=n/a"
    grid = qa.get("grid", {})
    grid_str = f"grid={grid.get('valid_count', '?')}/{grid.get('total_count', '?')}"
    return (f"{_v('west')} {_v('center')} {_v('east')} "
            f"patch={qa.get('patch_med', 0):.0f}m {grid_str} "
            f"stable_iou={qa.get('stable_iou', 0.0):.2f} "
            f"score={qa['score']:.0f}")


def step_select_warp_and_apply(state: AlignState, profiler=None) -> AlignState:
    """Step 5: Select warp mode and apply correction."""
    _p = profiler or _NullProfiler()
    if state.overlap is None or state.work_crs is None:
        raise ValueError("Warp selection requires an established work region")
    if not state.yes:
        response = input("Apply this correction? [y/N] ")
        if response.lower() not in ("y", "yes"):
            if state.precorrection_tmp and os.path.exists(state.precorrection_tmp):
                os.remove(state.precorrection_tmp)
            raise UserAbortError("User declined warp application")

    # Output in Web Mercator for web overlay
    output_crs = WEB_MERCATOR
    qa_eval_res = max(2.0, min(6.0, max(state.offset_res_m, state.ref_res_m)))

    all_gcps = list(state.gcps) + list(state.boundary_gcps)
    print(f"Step 5: Grid Optimization Warp ({len(state.gcps)} GCPs "
          f"+ {len(state.boundary_gcps)} boundary = {len(all_gcps)} total)", flush=True)
    # Close model_cache before apply_warp to free GPU memory
    if state.model_cache is not None:
        state.model_cache.close()
        state.model_cache = None
    with _p.section("apply_warp"):
        apply_warp(state.current_input, state.output_path, state.reference_path, all_gcps,
                   state.work_crs,
                   output_bounds=state.overlap.as_tuple(),
                   output_res=state.offset_res_m,
                   output_crs=output_crs,
                   grid_size=state.grid_size,
                   grid_iters=state.grid_iters,
                   arap_weight=state.arap_weight,
                   n_real_gcps_in=len(state.gcps),
                   match_weights=state.match_weights,
                   diagnostics_dir=state.diagnostics_dir,
                   profiler=_p,
                   model_cache=state.model_cache)

    # ------------------------------------------------------------------
    # QA + TPS fallback gate
    # Compute QA on the grid result, then run a GDAL TPS warp of the
    # same GCPs.  Keep whichever scores better.  This guarantees output
    # is never worse than plain TPS.
    # ------------------------------------------------------------------
    qa_grid = None
    with _p.section("qa_eval"):
        try:
            qa_grid = evaluate_alignment_quality_paths(
                state.output_path,
                state.reference_path,
                state.overlap.as_tuple(),
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
        with _p.section("tps_fallback"):
            try:
                print("  Running TPS fallback warp for comparison...", flush=True)
                tps_ok = _tps_warp_gcps(
                    state.current_input, tps_tmp,
                    all_gcps, state.work_crs, output_crs,
                    state.overlap.as_tuple(), state.offset_res_m,
                )
                if tps_ok:
                    try:
                        flow_applied = apply_flow_refinement_to_file(
                            tps_tmp, state.reference_path,
                            state.work_crs, state.overlap.as_tuple(),
                            state.offset_res_m,
                        )
                        if flow_applied:
                            print("  [FlowRefine-TPS] Applied flow refinement to TPS result", flush=True)
                    except Exception as e:
                        print(f"  [FlowRefine-TPS] Skipped ({e})", flush=True)
                    qa_tps = evaluate_alignment_quality_paths(
                        tps_tmp,
                        state.reference_path,
                        state.overlap.as_tuple(),
                        state.work_crs,
                        eval_res=qa_eval_res,
                        mask_mode=state.mask_provider,
                    )
                    print(f"  TPS fallback QA:  {_qa_label(qa_tps)}", flush=True)
            except Exception as e:
                print(f"  TPS fallback failed: {e}", flush=True)

    state.qa_reports = []
    with _p.section("qa_report"):
        report_grid = build_candidate_report(
            "grid",
            state.output_path,
            state.reference_path,
            state.overlap.as_tuple(),
            state.work_crs,
            holdout_pairs=state.qa_holdout_pairs,
            M_geo=state.M_geo,
            coverage=state.gcp_coverage,
            cv_mean_m=state.cv_mean,
            hypothesis_id=state.chosen_hypothesis.hypothesis_id if state.chosen_hypothesis else "",
            eval_res=qa_eval_res,
            image_metrics=qa_grid,
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
                state.overlap.as_tuple(),
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

def run(args, model_cache=None, profile=True) -> str:
    """Main pipeline orchestrator.  Returns output path."""
    pipeline_t0 = time.time()
    state = None
    profiler = PipelineProfiler() if profile else _NullProfiler()

    with profiler.section("setup"):
        state = step_setup(args)

    # Attach or create model cache
    if model_cache is not None:
        state.model_cache = model_cache
    elif state.model_cache is None:
        device_override = getattr(args, 'device', 'auto')
        device = get_torch_device(override=device_override if device_override != 'auto' else None)
        state.model_cache = ModelCache(device)

    try:
        with profiler.section("global_localization"):
            state = step_global_localization(state, args)
        if state.diagnostics_dir:
            save_checkpoint(state, "post_setup",
                            os.path.join(state.diagnostics_dir, "checkpoints"))
        with profiler.section("coarse_offset"):
            state = step_coarse_offset(state, profiler=profiler)
        with profiler.section("orthorectify"):
            state = step_orthorectify(state)
        with profiler.section("handle_large_offset"):
            state = step_handle_large_offset(state, args)
        with profiler.section("scale_rotation"):
            state = step_scale_rotation(state, args, profiler=profiler)
        with profiler.section("feature_matching"):
            state = step_feature_matching(state, args, profiler=profiler)
        with profiler.section("validate_and_filter"):
            state = step_validate_and_filter(state)
        with profiler.section("select_warp_and_apply"):
            state = step_select_warp_and_apply(state, profiler=profiler)
        with profiler.section("post_refinement"):
            step_post_refinement(state)
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

    # Print waterfall and save profile
    if profile:
        profiler.print_waterfall()
        if state is not None and state.diagnostics_dir:
            import json as _json
            # Only save profile JSON for the final nested run (coarse_pass > 0 if large offset)
            # Otherwise we overwrite the comprehensive profile with the outer stub
            is_root_pass = (getattr(args, 'coarse_pass', 0) == 0)
            if not is_root_pass or state.coarse_total <= 2000:
                profile_path = os.path.join(state.diagnostics_dir, "profile.json")
                try:
                    with open(profile_path, "w") as f:
                        _json.dump(profiler.to_dict(), f, indent=2)
                except Exception:
                    pass

    print(f"\n  [TOTAL] {time.time() - pipeline_t0:.1f}s", flush=True)
    return state.output_path if state is not None else getattr(args, "output", "")
