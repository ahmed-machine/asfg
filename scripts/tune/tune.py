#!/usr/bin/env python3
"""Optuna-based hyperparameter tuning for the alignment pipeline.

Runs per-phase studies, loading checkpointed state to skip expensive
earlier phases.  Supports fast proxy modes that reduce iteration count
and skip downstream phases for rapid feedback.

Usage:
    python3 scripts/tune/tune.py --phase grid_optim --profile kh9 --n-trials 20 --fast-proxy
    python3 scripts/tune/tune.py --phase matching --profile kh9 --n-trials 10 --build-checkpoint
    python3 scripts/tune/tune.py --phase flow --n-trials 30 --fast-proxy
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# ---------------------------------------------------------------------------
# Shared model cache — created once, reused across all trials
# ---------------------------------------------------------------------------

_shared_model_cache = None


def _get_model_cache():
    """Lazily create a shared ModelCache for phases that need neural models."""
    global _shared_model_cache
    if _shared_model_cache is None:
        from align.models import ModelCache
        from align.geo import get_torch_device
        device = get_torch_device()
        print(f"  [tune] Creating shared ModelCache on {device}")
        _shared_model_cache = ModelCache(device=device)
    return _shared_model_cache


def _ensure_model_cache(state):
    """Inject shared model cache into state if missing."""
    if state.model_cache is None:
        state.model_cache = _get_model_cache()
    return state


# ---------------------------------------------------------------------------
# Test-case config — loaded from tune_cases.yaml or CLI overrides
# ---------------------------------------------------------------------------

_TUNE_CASES_PATH = PROJECT_ROOT / "data" / "tune_cases.yaml"


def _load_tune_cases() -> dict:
    """Load tune_cases.yaml and return the cases dict."""
    import yaml
    with open(_TUNE_CASES_PATH) as f:
        return yaml.safe_load(f)


def _resolve_case(args) -> tuple[str, str, str]:
    """Return (target, reference, anchors) from CLI args or case config.

    Priority: explicit --target/--reference/--anchors > --case > first case in config.
    """
    if args.target and args.reference:
        anchors = args.anchors or str(PROJECT_ROOT / "data" / "bahrain_anchor_gcps.json")
        return args.target, args.reference, anchors

    config = _load_tune_cases()
    cases = config.get("cases", {})

    case_id = args.case
    if case_id is None:
        # Default to first case
        case_id = next(iter(cases))
        print(f"No --case specified, using default: {case_id}")

    if case_id not in cases:
        print(f"ERROR: Unknown case '{case_id}'. Available: {list(cases.keys())}")
        sys.exit(1)

    case = cases[case_id]
    target = os.path.expanduser(case["target"])
    reference = os.path.expanduser(case["reference"])
    anchors_path = case.get("anchors", "data/bahrain_anchor_gcps.json")
    # Resolve relative anchors path
    if not os.path.isabs(anchors_path):
        anchors_path = str(PROJECT_ROOT / anchors_path)

    # Set profile from case config if not explicitly given
    if args.profile == "_base" and "profile" in case:
        args.profile = case["profile"]
        print(f"Using profile from case config: {args.profile}")

    return target, reference, anchors_path


def build_checkpoint(args, target: str = None, reference: str = None,
                     anchors: str = None):
    """Run the pipeline up to the requested phase, saving checkpoints.

    If *target*/*reference*/*anchors* are not given, resolves from args.
    """
    from align.params import set_profile
    set_profile(args.profile)

    if target is None:
        target, reference, anchors = _resolve_case(args)

    case_label = getattr(args, "case", None) or args.profile
    diag_dir = os.path.join(str(PROJECT_ROOT), "diagnostics", f"tune_{case_label}")
    os.makedirs(diag_dir, exist_ok=True)
    checkpoint_dir = os.path.join(diag_dir, "checkpoints")

    # Check if checkpoint already exists
    target_phase = _phase_checkpoint_id(args.phase)
    if os.path.exists(os.path.join(checkpoint_dir, f"{target_phase}.json")):
        print(f"Checkpoint {target_phase} already exists at {checkpoint_dir}")
        if not args.force_rebuild:
            print("Use --force-rebuild to regenerate. Skipping.")
            return checkpoint_dir
        print("Force rebuilding...")

    # Run pipeline via subprocess (same as run_test.py)
    import subprocess
    output_path = os.path.join(diag_dir, "tune_baseline.tif")
    cmd = [
        sys.executable, str(PROJECT_ROOT / "auto-align.py"),
        target, "--reference", reference,
        "--anchors", anchors,
        "--output", output_path,
        "--yes", "--best",
        "--profile", args.profile,
        "--diagnostics-dir", diag_dir,
    ]
    print(f"Building checkpoint by running full pipeline...")
    print(f"  cmd: {' '.join(cmd[:6])}...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"Pipeline failed after {elapsed:.0f}s")
        print(result.stderr[-2000:] if result.stderr else "(no stderr)")
        sys.exit(1)

    print(f"Pipeline completed in {elapsed:.0f}s")

    # Verify checkpoint exists
    if not os.path.exists(os.path.join(checkpoint_dir, f"{target_phase}.json")):
        print(f"ERROR: Expected checkpoint {target_phase} not found at {checkpoint_dir}")
        print("Available checkpoints:")
        for f in sorted(Path(checkpoint_dir).glob("*.json")):
            print(f"  {f.name}")
        sys.exit(1)

    print(f"Checkpoint {target_phase} ready.")
    return checkpoint_dir


def _phase_checkpoint_id(phase: str) -> str:
    """Map a tuning phase to the checkpoint it needs as input."""
    return {
        "coarse": "post_setup",
        "scale_rotation": "post_coarse",
        "matching": "post_scale_rotation",
        "validation": "post_match",
        "grid_optim": "post_validate",
        "flow": "post_validate",
        "normalization": "post_validate",
    }[phase]


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------

def objective_coarse(trial, checkpoint_dir: str, args):
    """Tune coarse offset detection hyperparameters."""
    from align.checkpoint import load_checkpoint
    from align.params import override

    state = load_checkpoint("post_setup", checkpoint_dir)

    template_radius = trial.suggest_float("template_radius_m", 3000, 10000, step=500)
    search_margin = trial.suggest_float("search_margin_m", 150, 600, step=50)
    coarse_res = trial.suggest_float("coarse_res", 10.0, 25.0, step=1.0)
    refine_res = trial.suggest_float("refine_res", 3.0, 8.0, step=0.5)

    overrides = dict(
        coarse__template_radius_m=template_radius,
        coarse__search_margin_m=search_margin,
        coarse__coarse_res=coarse_res,
        coarse__refine_res=refine_res,
    )

    with override(**overrides):
        score = _run_coarse_and_score(state, args, trial.number)

    return score


def objective_scale_rotation(trial, checkpoint_dir: str, args):
    """Tune scale/rotation detection hyperparameters."""
    from align.checkpoint import load_checkpoint
    from align.params import override

    state = load_checkpoint("post_coarse", checkpoint_dir)
    _ensure_model_cache(state)

    grid_cols = trial.suggest_int("grid_cols", 2, 5)
    grid_rows = trial.suggest_int("grid_rows", 2, 5)
    detection_res = trial.suggest_float("detection_res", 3.0, 8.0, step=0.5)
    min_valid_frac = trial.suggest_float("min_valid_frac", 0.15, 0.50)

    overrides = dict(
        scale_rotation__grid_cols=grid_cols,
        scale_rotation__grid_rows=grid_rows,
        scale_rotation__detection_res=detection_res,
        scale_rotation__min_valid_frac=min_valid_frac,
    )

    with override(**overrides):
        score = _run_scale_rotation_and_score(state, args, trial.number)

    return score


def objective_grid_optim(trial, checkpoint_dir: str, args):
    """Tune grid optimisation hyperparameters."""
    from align.checkpoint import load_checkpoint
    from align.params import override

    state = load_checkpoint("post_validate", checkpoint_dir)

    # Strip anchor GCPs when --no-anchors is set
    if getattr(args, 'no_anchors', False) and hasattr(state, 'matched_pairs'):
        state.matched_pairs = [m for m in state.matched_pairs if not m.is_anchor]

    feat_only = getattr(args, 'feat_only', False)
    pixel_only = getattr(args, 'pixel_only', False)
    lock_grid = feat_only or pixel_only

    from align.params import get_params as _gp
    _bp = _gp().grid_optim

    # --- Grid optim weights (fixed when --feat-only or --pixel-only) ---
    if lock_grid:
        w_data = _bp.w_data
        w_chamfer = _bp.w_chamfer
        w_arap = _bp.w_arap
        w_laplacian = _bp.w_laplacian
        w_disp = _bp.w_disp
        lr = _bp.lr
        early_stop = _bp.early_stop_threshold
        l1_data = _bp.level_w_data_scale[1] if len(_bp.level_w_data_scale) > 1 else 1.5
        l2_data = _bp.level_w_data_scale[2] if len(_bp.level_w_data_scale) > 2 else 2.0
        l1_disp = _bp.level_w_disp_scale[1] if len(_bp.level_w_disp_scale) > 1 else 0.667
        l2_disp = _bp.level_w_disp_scale[2] if len(_bp.level_w_disp_scale) > 2 else 0.5
        l1_reg = _bp.level_reg_scale[1] if len(_bp.level_reg_scale) > 1 else 1.15
        l2_reg = _bp.level_reg_scale[2] if len(_bp.level_reg_scale) > 2 else 1.30
        l1_chamfer = _bp.level_chamfer_scale[1] if len(_bp.level_chamfer_scale) > 1 else 0.667
        l2_chamfer = _bp.level_chamfer_scale[2] if len(_bp.level_chamfer_scale) > 2 else 0.5
    else:
        w_data = trial.suggest_float("w_data", 0.5, 3.0)
        w_chamfer = trial.suggest_float("w_chamfer", 0.05, 0.8)
        w_arap = trial.suggest_float("w_arap", 0.1, 2.0)
        w_laplacian = trial.suggest_float("w_laplacian", 0.05, 1.0)
        w_disp = trial.suggest_float("w_disp", 0.01, 0.15)
        lr = trial.suggest_float("lr", 0.0005, 0.01, log=True)
        early_stop = trial.suggest_float("early_stop_threshold", 0.0001, 0.001, log=True)
        # Per-level scales (3 levels)
        l1_data = trial.suggest_float("level1_data_scale", 1.0, 3.0)
        l2_data = trial.suggest_float("level2_data_scale", 1.0, 4.0)
        l1_disp = trial.suggest_float("level1_disp_scale", 0.3, 1.0)
        l2_disp = trial.suggest_float("level2_disp_scale", 0.2, 0.8)
        l1_reg = trial.suggest_float("level1_reg_scale", 1.0, 1.5)
        l2_reg = trial.suggest_float("level2_reg_scale", 1.0, 2.0)
        l1_chamfer = trial.suggest_float("level1_chamfer_scale", 0.3, 1.0)
        l2_chamfer = trial.suggest_float("level2_chamfer_scale", 0.2, 0.8)

    # --- DINOv3 feature loss params (skipped when --pixel-only) ---
    if pixel_only:
        w_feat = 0.0
        feat_extract_res = _bp.feat_extract_res
        feat_coverage_gate = _bp.feat_coverage_gate
        feat_scale_factor = _bp.feat_scale_factor
        feat_mid_ratio = _bp.feat_mid_ratio
        feat_lncc_window = _bp.feat_lncc_window
        l1_feat = _bp.level_w_feat_scale[1] if len(_bp.level_w_feat_scale) > 1 else 1.0
        l2_feat = _bp.level_w_feat_scale[2] if len(_bp.level_w_feat_scale) > 2 else 1.0
    else:
        w_feat = trial.suggest_float("w_feat", 0.001, 0.05, log=True)
        feat_extract_res = trial.suggest_float("feat_extract_res", 4.0, 12.0)
        feat_coverage_gate = trial.suggest_categorical("feat_coverage_gate", [True, False])
        feat_scale_factor = trial.suggest_float("feat_scale_factor", 0.3, 2.0)
        feat_mid_ratio = trial.suggest_float("feat_mid_ratio", 0.0, 0.5)
        feat_lncc_window = trial.suggest_int("feat_lncc_window", 5, 11, step=2)
        l1_feat = trial.suggest_float("level1_feat_scale", 0.5, 2.0)
        l2_feat = trial.suggest_float("level2_feat_scale", 0.5, 3.0)

    # --- Pixel NCC params (swept when --pixel-only or full sweep) ---
    if pixel_only or not feat_only:
        w_pixel = trial.suggest_float("w_pixel", 1.0, 100.0, log=True)
        pixel_ncc_res = trial.suggest_float("pixel_ncc_res", 2.0, 8.0)
        pixel_ncc_window = trial.suggest_int("pixel_ncc_window", 5, 15, step=2)
        pixel_ncc_cadence = trial.suggest_int("pixel_ncc_cadence", 1, 10)
        l1_pixel = trial.suggest_float("level1_pixel_scale", 0.0, 1.0)
        l2_pixel = trial.suggest_float("level2_pixel_scale", 0.5, 2.0)
    else:
        w_pixel = 0.0
        pixel_ncc_res = _bp.pixel_ncc_res
        pixel_ncc_window = _bp.pixel_ncc_window
        pixel_ncc_cadence = _bp.pixel_ncc_cadence
        l1_pixel = _bp.level_w_pixel_scale[1] if len(_bp.level_w_pixel_scale) > 1 else 0.5
        l2_pixel = _bp.level_w_pixel_scale[2] if len(_bp.level_w_pixel_scale) > 2 else 1.0

    overrides = dict(
        grid_optim__w_data=w_data,
        grid_optim__w_chamfer=w_chamfer,
        grid_optim__w_feat=w_feat,
        grid_optim__w_arap=w_arap,
        grid_optim__w_laplacian=w_laplacian,
        grid_optim__w_disp=w_disp,
        grid_optim__lr=lr,
        grid_optim__early_stop_threshold=early_stop,
        grid_optim__feat_extract_res=feat_extract_res,
        grid_optim__feat_coverage_gate=feat_coverage_gate,
        grid_optim__feat_scale_factor=feat_scale_factor,
        grid_optim__feat_mid_ratio=feat_mid_ratio,
        grid_optim__feat_lncc_window=feat_lncc_window,
        grid_optim__level_w_data_scale=[1.0, l1_data, l2_data],
        grid_optim__level_w_disp_scale=[1.0, l1_disp, l2_disp],
        grid_optim__level_reg_scale=[1.0, l1_reg, l2_reg],
        grid_optim__level_chamfer_scale=[1.0, l1_chamfer, l2_chamfer],
        grid_optim__level_w_feat_scale=[0.0, l1_feat, l2_feat],
        grid_optim__w_pixel=w_pixel,
        grid_optim__pixel_ncc_res=pixel_ncc_res,
        grid_optim__pixel_ncc_window=pixel_ncc_window,
        grid_optim__pixel_ncc_cadence=pixel_ncc_cadence,
        grid_optim__level_w_pixel_scale=[0.0, l1_pixel, l2_pixel],
    )

    if args.fast_proxy:
        overrides["grid_optim__pyramid_levels"] = [[8, 50], [24, 50]]

    with override(**overrides):
        score = _run_warp_and_score(state, args, trial.number)

    return score


def objective_matching(trial, checkpoint_dir: str, args):
    """Tune feature matching hyperparameters."""
    from align.checkpoint import load_checkpoint
    from align.params import override

    state = load_checkpoint("post_scale_rotation", checkpoint_dir)
    _ensure_model_cache(state)

    roma_size_choice = trial.suggest_categorical("roma_size", [560, 576, 592, 608, 616, 624, 640, 672, 784, 800])
    roma_num_corresp = trial.suggest_int("roma_num_corresp", 200, 1000, step=50)
    ransac_thresh = trial.suggest_float("ransac_reproj_threshold", 2.0, 10.0)
    land_mask_frac = trial.suggest_float("land_mask_frac_min", 0.10, 0.50)
    tile_joint = trial.suggest_float("tile_joint_land_min", 0.02, 0.15)
    quota_per_cell = trial.suggest_int("match_quota_per_cell", 10, 60, step=5)
    estimation_method = trial.suggest_categorical("estimation_method", ["ransac", "lmeds", "magsac"])

    overrides = dict(
        matching__roma_size=roma_size_choice,
        matching__roma_num_corresp=roma_num_corresp,
        matching__ransac_reproj_threshold=ransac_thresh,
        matching__land_mask_frac_min=land_mask_frac,
        matching__tile_joint_land_min=tile_joint,
        matching__match_quota_per_cell=quota_per_cell,
        matching__estimation_method=estimation_method,
    )

    with override(**overrides):
        score = _run_matching_and_score(state, args, trial.number)

    return score


def objective_validation(trial, checkpoint_dir: str, args):
    """Tune validation hyperparameters.

    Expanded param space: includes TIN-TARR threshold, FPP toggle, and MAD
    sigma — these directly affect GCP selection and have measurable impact
    on cv_mean and coverage, unlike the original 2 params which were flat.
    """
    from align.checkpoint import load_checkpoint
    from align.params import override

    state = load_checkpoint("post_match", checkpoint_dir)

    anchor_thresh = trial.suggest_int("anchor_inlier_threshold", 3, 15)
    cv_refit = trial.suggest_float("cv_refit_threshold_m", 20.0, 80.0)
    tin_tarr = trial.suggest_float("tin_tarr_thresh", 1.0, 3.0)
    skip_fpp = trial.suggest_categorical("skip_fpp", [True, False])
    mad_sigma = trial.suggest_float("mad_sigma", 1.5, 4.0)

    overrides = dict(
        validation__anchor_inlier_threshold=anchor_thresh,
        validation__cv_refit_threshold_m=cv_refit,
        validation__tin_tarr_thresh=tin_tarr,
        validation__skip_fpp=skip_fpp,
        validation__mad_sigma=mad_sigma,
        validation__mad_sigma_scaled=mad_sigma + 1.0,
    )

    with override(**overrides):
        score = _run_validation_and_score(state, args, trial.number)

    return score


def objective_flow(trial, checkpoint_dir: str, args):
    """Tune flow refinement hyperparameters.

    Expanded from threshold-only params (fb_consistency, max_bias — flat for
    well-aligned imagery) to include architecture params that directly affect
    the flow field: DIS iterations, SEA-RAFT tile size, correction clamps,
    and median kernel size.
    """
    from align.checkpoint import load_checkpoint
    from align.params import override

    state = load_checkpoint("post_validate", checkpoint_dir)

    fb_consistency = trial.suggest_float("fb_consistency_px", 1.0, 8.0)
    max_bias = trial.suggest_float("max_flow_bias_m", 5.0, 30.0)
    dis_iters = trial.suggest_int("dis_variational_iters", 1, 10)
    sea_raft_tile = trial.suggest_categorical("sea_raft_tile_size", [512, 768, 1024])
    max_corr_fine = trial.suggest_float("max_correction_fine_m", 15.0, 60.0)
    max_corr_combined = trial.suggest_float("max_correction_combined_m", 50.0, 150.0)
    median_k = trial.suggest_categorical("median_kernel", [3, 5, 7])

    overrides = dict(
        flow__fb_consistency_px=fb_consistency,
        flow__max_flow_bias_m=max_bias,
        flow__dis_variational_iters=dis_iters,
        flow__sea_raft_tile_size=sea_raft_tile,
        flow__max_correction_fine_m=max_corr_fine,
        flow__max_correction_combined_m=max_corr_combined,
        flow__median_kernel=median_k,
    )

    grid_cache = os.path.join(checkpoint_dir, "grid_optim_cache.npz")
    with override(**overrides):
        if os.path.exists(grid_cache):
            score = _run_flow_and_score(state, args, trial.number, grid_cache)
        else:
            score = _run_warp_and_score(state, args, trial.number)

    return score


def objective_normalization(trial, checkpoint_dir: str, args):
    """Tune normalization hyperparameters."""
    from align.checkpoint import load_checkpoint
    from align.params import override

    state = load_checkpoint("post_validate", checkpoint_dir)

    clahe_clip = trial.suggest_float("clahe_clip_limit", 1.0, 5.0)
    wallis = trial.suggest_categorical("wallis_matching", [True, False])
    joint_pct = trial.suggest_categorical("flow_joint_percentile", [True, False])
    pct_lo = trial.suggest_int("flow_percentile_lo", 0, 5)
    pct_hi = trial.suggest_int("flow_percentile_hi", 95, 100)

    overrides = dict(
        normalization__clahe_clip_limit=clahe_clip,
        normalization__wallis_matching=wallis,
        normalization__flow_joint_percentile=joint_pct,
        normalization__flow_percentile_lo=pct_lo,
        normalization__flow_percentile_hi=pct_hi,
    )

    grid_cache = os.path.join(checkpoint_dir, "grid_optim_cache.npz")
    with override(**overrides):
        if os.path.exists(grid_cache):
            score = _run_flow_and_score(state, args, trial.number, grid_cache)
        else:
            score = _run_warp_and_score(state, args, trial.number)

    return score


# ---------------------------------------------------------------------------
# Phase runners (execute pipeline steps and return QA score)
# ---------------------------------------------------------------------------

def _make_trial_state(state, args, trial_num: int):
    """Create a deep-copied trial state with trial-specific output dirs."""
    import copy
    # Exclude model_cache from deepcopy — neural models can't be copied
    # and attempting it hangs on MPS device
    mc = state.model_cache
    state.model_cache = None
    trial_state = copy.deepcopy(state)
    state.model_cache = mc  # restore original
    trial_state.model_cache = mc  # share reference
    trial_state.yes = True
    # Use case_label (case ID or profile) to give each case its own
    # trial subdir like tune_{case_id}/ so parallel per-case runs don't collide.
    case_label = getattr(args, 'case', None) or args.profile or '_base'
    trial_dir = os.path.join(str(PROJECT_ROOT), "diagnostics",
                             f"tune_{case_label}", f"trial_{trial_num}")
    os.makedirs(trial_dir, exist_ok=True)
    trial_state.diagnostics_dir = trial_dir
    trial_state.output_path = os.path.join(trial_dir, "aligned.tif")
    return trial_state, trial_dir


def _run_coarse_and_score(state, args, trial_num: int) -> float:
    """Run only step_coarse_offset; proxy score from correlation.

    When NCC detection fails entirely (corr=0, coarse_corr=0), we re-run
    coarse detection at each resolution to capture the best NCC value
    achieved even below the 0.3 threshold. This provides a continuous
    gradient signal instead of a flat 100.0 for all failures.
    """
    from align.pipeline import step_coarse_offset
    from align.coarse import detect_offset_at_resolution
    from align.profiler import _NullProfiler
    from align import constants as _C

    trial_state, trial_dir = _make_trial_state(state, args, trial_num)

    try:
        trial_state = step_coarse_offset(trial_state, profiler=_NullProfiler())
        corr = getattr(trial_state, 'coarse_corr', 0.0)

        if corr > 0.01:
            # Normal case: detection succeeded
            score = (1.0 - corr) * 100
        else:
            # Detection failed — get the raw best NCC value for gradient signal.
            # Re-run template matching just to extract max_val (lightweight).
            import rasterio
            from align.geo import read_overlap_region
            from align.image import make_land_mask

            best_ncc = 0.0
            try:
                with rasterio.open(trial_state.current_input) as src_off, \
                     rasterio.open(trial_state.reference_path) as src_ref:
                    for res in [_C.COARSE_RES, _C.REFINE_RES]:
                        arr_off, _ = read_overlap_region(
                            src_off, trial_state.overlap, trial_state.work_crs, res)
                        arr_ref, _ = read_overlap_region(
                            src_ref, trial_state.overlap, trial_state.work_crs, res)
                        import cv2
                        land_off = make_land_mask(arr_off)
                        land_ref = make_land_mask(arr_ref)
                        # Quick NCC on land masks
                        if (land_ref.shape[0] > 20 and land_ref.shape[1] > 20 and
                                land_off.shape[0] > land_ref.shape[0] and
                                land_off.shape[1] > land_ref.shape[1]):
                            # Use center crop as template
                            h, w = land_ref.shape
                            th = min(h // 2, int(_C.DEFAULT_TEMPLATE_RADIUS_M / res))
                            tw = min(w // 2, int(_C.DEFAULT_TEMPLATE_RADIUS_M / res))
                            cy, cx = h // 2, w // 2
                            tpl = land_ref[max(0, cy-th):cy+th, max(0, cx-tw):cx+tw]
                            if tpl.shape[0] >= 10 and tpl.shape[1] >= 10:
                                result = cv2.matchTemplate(
                                    land_off.astype(np.float32),
                                    tpl.astype(np.float32),
                                    cv2.TM_CCOEFF_NORMED)
                                _, max_val, _, _ = cv2.minMaxLoc(result)
                                best_ncc = max(best_ncc, float(max_val))
            except Exception:
                pass

            # Continuous score: failed detection uses NCC as gradient
            # best_ncc ranges 0..0.3 (below threshold). Map to score range 70..100
            # so even failed trials can be ranked by how close they got.
            score = 100.0 - max(best_ncc, 0.01) * 100
    except Exception as e:
        print(f"  Trial {trial_num} FAILED: {e}")
        traceback.print_exc()
        score = 999.0
    finally:
        _cleanup_trial(trial_dir, keep_summary=True)
        gc.collect()

    return _clamp_score(score)


def _run_scale_rotation_and_score(state, args, trial_num: int) -> float:
    """Run only detect_local_scales; proxy score from patch quality.

    Skips the expensive precorrection warp + coarse re-detection — we only
    need the patch detection metrics for proxy scoring.
    """
    import torch
    import rasterio
    from align.scale import detect_local_scales
    from align import constants as _C

    trial_state, trial_dir = _make_trial_state(state, args, trial_num)

    try:
        needs_scale = (abs(trial_state.expected_scale - 1.0) > 0.05 or
                       trial_state.expected_scale > 1.1)
        if not needs_scale:
            return 50.0  # scale/rotation not needed

        src_offset = rasterio.open(trial_state.current_input)
        src_ref = rasterio.open(trial_state.reference_path)
        try:
            local_patches = detect_local_scales(
                src_offset, src_ref, trial_state.overlap, trial_state.work_crs,
                trial_state.coarse_dx, trial_state.coarse_dy,
                grid_cols=_C.SCALE_GRID_COLS, grid_rows=_C.SCALE_GRID_ROWS,
                model_cache=trial_state.model_cache)
        finally:
            src_offset.close()
            src_ref.close()

        if local_patches is None:
            score = 500.0  # detection failed entirely
        else:
            valid = [p for p in local_patches
                     if p['status'] in ('ok', 'filled-neighbor', 'filled-global')]
            ok = [p for p in local_patches if p['status'] == 'ok']
            total = max(len(local_patches), 1)
            valid_frac = len(valid) / total
            sx_spread = sy_spread = 0.0
            if len(ok) >= 2:
                sx_spread = max(p['scale_x'] for p in ok) - min(p['scale_x'] for p in ok)
                sy_spread = max(p['scale_y'] for p in ok) - min(p['scale_y'] for p in ok)
            score = (sx_spread + sy_spread) * 100 + (1.0 - valid_frac) * 100
    except Exception as e:
        print(f"  Trial {trial_num} FAILED: {e}")
        traceback.print_exc()
        score = 999.0
    finally:
        _cleanup_trial(trial_dir, keep_summary=True)
        gc.collect()
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return _clamp_score(score)


def _run_warp_and_score(state, args, trial_num: int) -> float:
    """Run grid optim + optional flow + QA.  Returns full QA score."""
    from align.pipeline import step_select_warp_and_apply
    from align.profiler import _NullProfiler

    trial_state, trial_dir = _make_trial_state(state, args, trial_num)

    try:
        trial_state = step_select_warp_and_apply(trial_state, profiler=_NullProfiler())
        score = _clamp_score(_extract_score(trial_state, trial_dir))
    except Exception as e:
        print(f"  Trial {trial_num} FAILED: {e}")
        traceback.print_exc()
        score = 999.0
    finally:
        _cleanup_trial(trial_dir, keep_summary=True)
        gc.collect()

    return score


def _run_flow_and_score(state, args, trial_num: int, grid_cache_path: str) -> float:
    """Run flow+remap+QA from cached grid result (skips grid optim)."""
    from align.warp import load_grid_optim_result, apply_warp_flow_and_remap
    from align.qa import evaluate_alignment_quality_paths
    from align.qa import build_candidate_report, write_qa_report
    from align.profiler import _NullProfiler

    trial_state, trial_dir = _make_trial_state(state, args, trial_num)

    try:
        print(f"  [GridCache] Loading cached grid result...", flush=True)
        grid_result = load_grid_optim_result(
            grid_cache_path, input_path_override=trial_state.current_input)

        apply_warp_flow_and_remap(
            grid_result, trial_state.reference_path,
            trial_state.output_path, profiler=_NullProfiler())

        # Run QA (same as step_select_warp_and_apply)
        qa_eval_res = max(2.0, min(6.0, max(state.offset_res_m, state.ref_res_m)))
        qa_metrics = evaluate_alignment_quality_paths(
            trial_state.output_path,
            trial_state.reference_path,
            trial_state.overlap,
            trial_state.work_crs,
            eval_res=qa_eval_res,
            mask_mode=trial_state.mask_provider,
        )
        if qa_metrics is not None:
            from align.pipeline import _qa_label
            print(f"  Grid warp QA: {_qa_label(qa_metrics)}", flush=True)

        report = build_candidate_report(
            "grid",
            trial_state.output_path,
            trial_state.reference_path,
            trial_state.overlap,
            trial_state.work_crs,
            holdout_pairs=trial_state.qa_holdout_pairs,
            M_geo=trial_state.M_geo,
            coverage=trial_state.gcp_coverage,
            cv_mean_m=trial_state.cv_mean,
            hypothesis_id=trial_state.chosen_hypothesis.hypothesis_id if trial_state.chosen_hypothesis else "",
            eval_res=qa_eval_res,
            image_metrics=qa_metrics,
        )
        print(f"  Grid independent QA: total={report.total_score:.0f}, "
              f"confidence={report.confidence:.2f}, accepted={report.accepted}")

        # Write QA report
        qa_path = os.path.join(trial_dir, "aligned_qa.json")
        write_qa_report(qa_path, [report], selected_candidate="grid")
        print(f"  QA report written to: {qa_path}")

        trial_state.qa_reports = [report]
        score = _clamp_score(_extract_score(trial_state, trial_dir))
    except Exception as e:
        print(f"  Trial {trial_num} FAILED: {e}")
        traceback.print_exc()
        score = 999.0
    finally:
        _cleanup_trial(trial_dir, keep_summary=True)
        gc.collect()

    return score


def _run_matching_and_score(state, args, trial_num: int) -> float:
    """Run only step_feature_matching; proxy score from match count + residual."""
    from align.pipeline import step_feature_matching
    from align.profiler import _NullProfiler

    trial_state, trial_dir = _make_trial_state(state, args, trial_num)

    try:
        mock_args = argparse.Namespace(device="auto")
        trial_state = step_feature_matching(trial_state, mock_args, profiler=_NullProfiler())
        n_matches = len(trial_state.matched_pairs)
        if n_matches < 4:
            score = 999.0
        else:
            # More matches with lower residual = better
            # match_quality_residual is set during the quality check in step_feature_matching
            residual = trial_state.match_quality_residual
            if not np.isfinite(residual):
                residual = 50.0  # default when quality check didn't run
            score = 200.0 / max(n_matches, 1) + residual
    except Exception as e:
        print(f"  Trial {trial_num} FAILED: {e}")
        traceback.print_exc()
        score = 999.0
    finally:
        _cleanup_trial(trial_dir, keep_summary=True)
        gc.collect()
        import torch
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return _clamp_score(score)


def _run_validation_and_score(state, args, trial_num: int) -> float:
    """Run only step_validate_and_filter; proxy score from CV + residual + coverage.

    The expanded proxy captures TIN-TARR and FPP effects through mean_residual
    (changes as GCPs are filtered) and n_gcps (changes with mad_sigma and
    tin_tarr_thresh). Also penalizes max_residual to discourage keeping
    high-error outliers.
    """
    from align.pipeline import step_validate_and_filter

    trial_state, trial_dir = _make_trial_state(state, args, trial_num)

    try:
        trial_state = step_validate_and_filter(trial_state)
        cv = trial_state.cv_mean if trial_state.cv_mean is not None else 200.0
        coverage = trial_state.gcp_coverage or 0.0
        n_gcps = len(trial_state.gcps)
        mean_res = trial_state.mean_residual if np.isfinite(trial_state.mean_residual) else 100.0
        max_res = trial_state.max_residual if np.isfinite(trial_state.max_residual) else 200.0
        # Lower CV + lower residual + higher coverage + reasonable GCP count = better
        score = (cv
                 + mean_res * 0.5
                 + max_res * 0.1
                 + (1.0 - coverage) * 50
                 + max(0, 15 - n_gcps) * 5)
    except Exception as e:
        print(f"  Trial {trial_num} FAILED: {e}")
        traceback.print_exc()
        score = 999.0
    finally:
        _cleanup_trial(trial_dir, keep_summary=True)
        gc.collect()

    return _clamp_score(score)


def _extract_score(state, trial_dir: str) -> float:
    """Extract QA score from trial state or summary.json."""
    # Check if qa_reports has score
    if state.qa_reports:
        for rpt in state.qa_reports:
            if hasattr(rpt, "score") and rpt.score is not None:
                return rpt.score
            if isinstance(rpt, dict):
                # Direct score key
                if rpt.get("score") is not None:
                    return rpt["score"]
                # total_score from QA report
                if rpt.get("total_score") is not None:
                    return rpt["total_score"]
                # Nested image_metrics.score
                im = rpt.get("image_metrics", {})
                if isinstance(im, dict) and im.get("score") is not None:
                    return im["score"]

    # Fallback: check qa_json_path
    if state.qa_json_path and os.path.exists(state.qa_json_path):
        with open(state.qa_json_path) as f:
            qa = json.load(f)
        if "score" in qa:
            return qa["score"]

    # Fallback: check aligned_qa.json in trial_dir
    qa_path = os.path.join(trial_dir, "aligned_qa.json")
    if os.path.exists(qa_path):
        with open(qa_path) as f:
            qa = json.load(f)
        reports = qa.get("reports", [])
        for rpt in reports:
            if rpt.get("total_score") is not None:
                return rpt["total_score"]

    # Last resort: look for summary.json in trial_dir
    summary_path = os.path.join(trial_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        if "score" in summary:
            return summary["score"]

    print(f"  WARNING: Could not extract score from trial, returning 999")
    return 999.0


def _clamp_score(score: float) -> float:
    """Clamp infinite or NaN scores to 999.0 for Optuna."""
    if score is None or not np.isfinite(score):
        return 999.0
    return float(score)


def _cleanup_trial(trial_dir: str, keep_summary: bool = True):
    """Remove large files from trial dir, keeping only summary data."""
    if not os.path.isdir(trial_dir):
        return
    for fname in os.listdir(trial_dir):
        fpath = os.path.join(trial_dir, fname)
        if not os.path.isfile(fpath):
            continue
        # Keep JSON summaries and small files
        if keep_summary and fname.endswith(".json"):
            continue
        # Remove large outputs (TIF, NPZ, JPG)
        ext = os.path.splitext(fname)[1].lower()
        if ext in (".tif", ".tiff", ".npz", ".jpg", ".png"):
            try:
                os.remove(fpath)
            except OSError:
                pass
    # Remove checkpoint subdirs
    ckpt_dir = os.path.join(trial_dir, "checkpoints")
    if os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)


def _cleanup_old_trials(study_dir: str, keep_best_n: int = 3):
    """Keep only the N best trial dirs (by score), remove others."""
    trial_dirs = sorted(Path(study_dir).glob("trial_*"))
    if len(trial_dirs) <= keep_best_n:
        return

    scores = {}
    for td in trial_dirs:
        # Try to read score from any JSON in the dir
        for jf in td.glob("*.json"):
            try:
                with open(jf) as f:
                    data = json.load(f)
                if "score" in data:
                    scores[td] = data["score"]
                    break
            except Exception:
                continue
        if td not in scores:
            scores[td] = 999.0

    sorted_dirs = sorted(scores.items(), key=lambda x: x[1])
    keep = {d for d, _ in sorted_dirs[:keep_best_n]}

    for td in trial_dirs:
        if td not in keep:
            shutil.rmtree(td, ignore_errors=True)


# ---------------------------------------------------------------------------
# Plateau detection
# ---------------------------------------------------------------------------

def check_plateau(study, patience: int = 5, min_improvement_pct: float = 1.0) -> bool:
    """True if last `patience` trials improved < min_improvement_pct."""
    try:
        from optuna.trial import TrialState
    except ImportError:
        return False

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if len(completed) < patience + 1:
        return False

    best_before = min(t.value for t in completed[:-patience])
    recent_best = min(t.value for t in completed[-patience:])
    improvement = (best_before - recent_best) / max(best_before, 1e-6) * 100
    return improvement < min_improvement_pct


def _write_plateau_report(study, phase: str, study_dir: str):
    """Write a structured summary when plateau is detected."""
    try:
        from optuna.trial import TrialState
    except ImportError:
        return

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        return

    best_trial = min(completed, key=lambda t: t.value)
    report = {
        "phase": phase,
        "total_trials": len(completed),
        "best_score": best_trial.value,
        "best_params": best_trial.params,
        "best_trial_number": best_trial.number,
        "plateau_detected": True,
        "suggestion": (
            f"Score plateaued at {best_trial.value:.1f} after {len(completed)} trials. "
            f"Consider: (1) literature search for SOTA in {phase}, "
            f"(2) widening search space, (3) tuning a different phase."
        ),
    }

    report_path = os.path.join(study_dir, f"tune_plateau_{phase}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Plateau report written to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OBJECTIVE_MAP = {
    "coarse": objective_coarse,
    "scale_rotation": objective_scale_rotation,
    "grid_optim": objective_grid_optim,
    "matching": objective_matching,
    "validation": objective_validation,
    "flow": objective_flow,
    "normalization": objective_normalization,
}


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--phase", required=True, choices=list(OBJECTIVE_MAP.keys()),
                        help="Which phase to tune")
    parser.add_argument("--profile", default="_base",
                        help="Camera profile name (default: _base)")
    parser.add_argument("--case", default=None,
                        help="Case ID from data/tune_cases.yaml (default: first case)")
    parser.add_argument("--target", default=None,
                        help="Override target image path")
    parser.add_argument("--reference", default=None,
                        help="Override reference image path")
    parser.add_argument("--anchors", default=None,
                        help="Override anchors JSON path")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Total timeout in seconds")
    parser.add_argument("--fast-proxy", action="store_true",
                        help="Use fast proxy mode (reduced iters, skip flow)")
    parser.add_argument("--build-checkpoint", action="store_true",
                        help="Run pipeline first to build required checkpoint")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Force rebuild checkpoint even if it exists")
    parser.add_argument("--plateau-patience", type=int, default=5,
                        help="Trials without improvement before plateau alert")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-anchors", action="store_true",
                        help="Remove anchor GCPs from checkpoint (tune w_feat as replacement)")
    parser.add_argument("--feat-only", action="store_true",
                        help="Fix grid params at profile values, only sweep feat params")
    parser.add_argument("--pixel-only", action="store_true",
                        help="Fix grid params at profile values, only sweep pixel NCC params")
    parser.add_argument("--build-grid-cache", action="store_true",
                        help="Build grid optim cache for fast flow/normalization tuning")
    args = parser.parse_args()

    # Resolve case config (may update args.profile as side effect)
    target, reference, anchors = _resolve_case(args)

    try:
        import optuna
    except ImportError:
        print("ERROR: optuna not installed. Run: pip install optuna")
        sys.exit(1)

    # Set profile
    from align.params import set_profile
    set_profile(args.profile)

    case_label = args.case or args.profile
    study_dir = os.path.join(str(PROJECT_ROOT), "diagnostics", f"tune_{case_label}")
    os.makedirs(study_dir, exist_ok=True)
    checkpoint_dir = os.path.join(study_dir, "checkpoints")

    # Build checkpoint if needed
    required_checkpoint = _phase_checkpoint_id(args.phase)
    checkpoint_json = os.path.join(checkpoint_dir, f"{required_checkpoint}.json")

    if args.build_checkpoint or not os.path.exists(checkpoint_json):
        if not os.path.exists(checkpoint_json):
            print(f"Checkpoint {required_checkpoint} not found. Building...")
        checkpoint_dir = build_checkpoint(args, target, reference, anchors)
    else:
        print(f"Using existing checkpoint: {checkpoint_json}")

    # Verify checkpoint loadable
    from align.checkpoint import load_checkpoint
    try:
        test_state = load_checkpoint(required_checkpoint, checkpoint_dir)
        del test_state
    except Exception as e:
        print(f"ERROR: Cannot load checkpoint {required_checkpoint}: {e}")
        print("Try rebuilding with --build-checkpoint --force-rebuild")
        sys.exit(1)

    # Build grid optim cache for flow/normalization tuning
    if args.build_grid_cache and args.phase in ("flow", "normalization"):
        grid_cache_path = os.path.join(checkpoint_dir, "grid_optim_cache.npz")
        print(f"\nBuilding grid optim cache...")
        from align.checkpoint import load_checkpoint as _lc
        from align.warp import apply_warp_grid_only, save_grid_optim_result
        from align.pipeline import OUTPUT_CRS_EPSG
        from rasterio.crs import CRS

        cache_state = _lc(required_checkpoint, checkpoint_dir)
        _ensure_model_cache(cache_state)
        output_crs = CRS.from_epsg(OUTPUT_CRS_EPSG)
        all_gcps = list(cache_state.gcps) + list(cache_state.boundary_gcps)

        # Close model_cache before grid optim to free GPU memory
        if cache_state.model_cache is not None:
            cache_state.model_cache.close()
            cache_state.model_cache = None

        grid_result = apply_warp_grid_only(
            cache_state.current_input,
            cache_state.reference_path,
            all_gcps,
            cache_state.work_crs,
            output_bounds=cache_state.overlap,
            output_res=cache_state.offset_res_m,
            output_crs=output_crs,
        )
        save_grid_optim_result(grid_result, grid_cache_path)
        del grid_result, cache_state
        gc.collect()
        print(f"Grid optim cache ready: {grid_cache_path}\n")

        if args.n_trials == 0:
            print("--n-trials=0, exiting after cache build.")
            sys.exit(0)

    # Create Optuna study
    db_path = os.path.join(study_dir, f"tune_{case_label}.db")
    storage = f"sqlite:///{db_path}"
    study_name = f"tune_{args.phase}_{case_label}"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=args.seed, n_startup_trials=5, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
        storage=storage,
        load_if_exists=True,
    )

    objective_fn = OBJECTIVE_MAP[args.phase]
    plateau_reported = False

    print(f"\nStarting {study_name}: {args.n_trials} trials, "
          f"timeout {args.timeout}s, fast_proxy={args.fast_proxy}")
    print(f"  Storage: {db_path}")
    print(f"  Checkpoint: {checkpoint_dir}/{required_checkpoint}.json")
    print()

    t0 = time.time()

    def _objective(trial):
        nonlocal plateau_reported
        trial_t0 = time.time()
        print(f"--- Trial {trial.number} ---")

        score = objective_fn(trial, checkpoint_dir, args)

        elapsed = time.time() - trial_t0
        print(f"  Trial {trial.number}: score={score:.1f} ({elapsed:.0f}s)")

        # Check plateau
        if not plateau_reported and check_plateau(
                study, patience=args.plateau_patience):
            print(f"\n  *** PLATEAU DETECTED after {len(study.trials)} trials ***")
            _write_plateau_report(study, args.phase, study_dir)
            plateau_reported = True

        # Periodic cleanup
        if trial.number % 5 == 4:
            _cleanup_old_trials(study_dir, keep_best_n=3)

        return score

    try:
        study.optimize(
            _objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # Final summary
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Study complete: {study_name}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"  Completed trials: {len(study.trials)}")

    try:
        best = study.best_trial
        print(f"  Best score: {best.value:.1f} (trial {best.number})")
        print(f"  Best params:")
        for k, v in sorted(best.params.items()):
            print(f"    {k}: {v}")

        # Save best params
        best_path = os.path.join(study_dir, f"best_{args.phase}.json")
        with open(best_path, "w") as f:
            json.dump({
                "phase": args.phase,
                "profile": args.profile,
                "score": best.value,
                "trial_number": best.number,
                "params": best.params,
                "total_trials": len(study.trials),
                "total_time_s": total_time,
                "fast_proxy": args.fast_proxy,
            }, f, indent=2)
        print(f"\n  Best params saved to {best_path}")
    except ValueError:
        print("  No completed trials.")

    # Final cleanup
    _cleanup_old_trials(study_dir, keep_best_n=3)


if __name__ == "__main__":
    main()
