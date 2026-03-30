#!/usr/bin/env python3
"""Rebuild checkpoint with best tuned params, then tune remaining phases.

Reads best params from all completed Optuna studies, applies them to the
profile, runs a full pipeline pass to create a new checkpoint, then tunes
flow and normalization from that better-quality checkpoint.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_best_params(db_path: str) -> dict:
    """Extract best params from all completed Optuna studies."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = f"sqlite:///{db_path}"
    studies = optuna.study.get_all_study_names(storage)
    best = {}
    for name in sorted(studies):
        study = optuna.load_study(study_name=name, storage=storage)
        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        if not completed:
            continue
        phase = name.replace("tune_", "").replace("_kh9", "")
        best[phase] = study.best_params
        print(f"  {phase}: score={study.best_value:.1f} ({len(completed)} trials)")
    return best


def params_to_profile_overrides(best: dict) -> dict:
    """Convert Optuna best params to profile override kwargs."""
    overrides = {}

    if "coarse" in best:
        p = best["coarse"]
        overrides["coarse__template_radius_m"] = p.get("template_radius_m")
        overrides["coarse__coarse_res"] = p.get("coarse_res")
        overrides["coarse__refine_res"] = p.get("refine_res")
        overrides["coarse__search_margin_m"] = p.get("search_margin_m")

    if "scale_rotation" in best:
        p = best["scale_rotation"]
        overrides["scale_rotation__detection_res"] = p.get("detection_res")
        overrides["scale_rotation__grid_rows"] = p.get("grid_rows")
        overrides["scale_rotation__grid_cols"] = p.get("grid_cols")
        overrides["scale_rotation__min_valid_frac"] = p.get("min_valid_frac")

    if "matching" in best:
        p = best["matching"]
        overrides["matching__roma_size"] = p.get("roma_size")
        overrides["matching__roma_num_corresp"] = p.get("roma_num_corresp")
        overrides["matching__ransac_reproj_threshold"] = p.get("ransac_reproj_threshold")
        overrides["matching__land_mask_frac_min"] = p.get("land_mask_frac_min")
        overrides["matching__tile_joint_land_min"] = p.get("tile_joint_land_min")
        overrides["matching__match_quota_per_cell"] = p.get("match_quota_per_cell")

    if "validation" in best:
        p = best["validation"]
        overrides["validation__anchor_inlier_threshold"] = p.get("anchor_inlier_threshold")
        overrides["validation__cv_refit_threshold_m"] = p.get("cv_refit_threshold_m")
        overrides["validation__tin_tarr_thresh"] = p.get("tin_tarr_thresh")
        overrides["validation__skip_fpp"] = p.get("skip_fpp")
        overrides["validation__mad_sigma"] = p.get("mad_sigma")

    if "grid_optim" in best:
        p = best["grid_optim"]
        for k, v in p.items():
            overrides[f"grid_optim__{k}"] = v

    # Remove None values
    return {k: v for k, v in overrides.items() if v is not None}


def write_temp_profile(overrides: dict, base_profile: str = "kh9") -> str:
    """Write overrides to a JSON file that auto-align.py can load."""
    out_path = str(PROJECT_ROOT / "diagnostics" / "tune_kh9" / "best_overrides.json")
    with open(out_path, "w") as f:
        json.dump({"base_profile": base_profile, "overrides": overrides}, f, indent=2)
    print(f"  Wrote {len(overrides)} overrides to {out_path}")
    return out_path


def rebuild_checkpoint(profile: str, overrides: dict):
    """Run full pipeline with best params to create updated checkpoint."""
    from align.params import set_profile, get_params, override
    import yaml

    set_profile(profile)

    # Load case config
    cases_path = PROJECT_ROOT / "data" / "tune_cases.yaml"
    with open(cases_path) as f:
        config = yaml.safe_load(f)
    case = config["cases"]["kh9_1982"]
    target = os.path.expanduser(case["target"])
    reference = os.path.expanduser(case["reference"])
    anchors = case.get("anchors", "data/bahrain_anchor_gcps.json")
    if not os.path.isabs(anchors):
        anchors = str(PROJECT_ROOT / anchors)

    diag_dir = str(PROJECT_ROOT / "diagnostics" / "tune_kh9")
    checkpoint_dir = os.path.join(diag_dir, "checkpoints")
    output_path = os.path.join(diag_dir, "tune_baseline.tif")

    # Write overrides to a file the pipeline can read
    overrides_path = write_temp_profile(overrides, profile)

    # Build override args for auto-align.py CLI
    # Since auto-align doesn't support arbitrary overrides via CLI,
    # we'll run the pipeline directly with overrides applied
    print(f"\n  Running full pipeline with best params...")
    print(f"  Target: {os.path.basename(target)}")
    print(f"  Reference: {os.path.basename(reference)}")

    t0 = time.time()

    # Apply overrides and run pipeline
    with override(**overrides):
        from align.pipeline import run_alignment
        run_alignment(
            target_path=target,
            reference_path=reference,
            output_path=output_path,
            anchor_path=anchors,
            diagnostics_dir=diag_dir,
            profile=profile,
            yes=True,
            best_candidate=True,
        )

    elapsed = time.time() - t0
    print(f"\n  Pipeline completed in {elapsed:.0f}s")

    # Verify checkpoints
    for cp in ["post_setup", "post_coarse", "post_scale_rotation",
               "post_match", "post_validate"]:
        cp_path = os.path.join(checkpoint_dir, f"{cp}.json")
        if os.path.exists(cp_path):
            print(f"  ✓ {cp}")
        else:
            print(f"  ✗ {cp} MISSING")

    return checkpoint_dir


def tune_phase(phase: str, profile: str, n_trials: int):
    """Run Optuna tuning for a single phase."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "tune" / "tune.py"),
        "--phase", phase,
        "--profile", profile,
        "--n-trials", str(n_trials),
        "--timeout", "14400",
        "--fast-proxy",
    ]
    print(f"\n{'='*60}")
    print(f"  Tuning {phase}: {n_trials} trials")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, timeout=18000)
    if result.returncode != 0:
        print(f"  WARNING: {phase} exited with code {result.returncode}")


def main():
    db_path = str(PROJECT_ROOT / "diagnostics" / "tune_kh9" / "tune_kh9.db")
    profile = "kh9"

    print("=" * 60)
    print("  Step 1: Collect best params from all studies")
    print("=" * 60)
    best = get_best_params(db_path)
    overrides = params_to_profile_overrides(best)
    print(f"\n  Total overrides: {len(overrides)}")

    print(f"\n{'='*60}")
    print("  Step 2: Rebuild checkpoint with best params")
    print("=" * 60)
    rebuild_checkpoint(profile, overrides)

    print(f"\n{'='*60}")
    print("  Step 3: Tune flow and normalization")
    print("=" * 60)

    # Delete old flow/normalization studies so they start fresh
    # with the new checkpoint
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{db_path}"
    for study_name in ["tune_flow_kh9", "tune_normalization_kh9"]:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f"  Deleted old study: {study_name}")
        except KeyError:
            pass

    tune_phase("flow", profile, 20)
    tune_phase("normalization", profile, 20)

    print(f"\n{'='*60}")
    print("  All done!")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
