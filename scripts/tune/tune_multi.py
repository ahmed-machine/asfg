#!/usr/bin/env python3
"""Multi-image Bayesian hyperparameter tuning orchestrator.

Runs Optuna studies across multiple test images with SEQUENTIAL phase
optimisation: each phase is fully tuned before moving to the next, and
checkpoints are rebuilt between phases using the best-so-far parameters.

A best profile YAML (data/profiles/best_{profile}.yaml) is automatically
maintained and updated after each phase completes.

Usage:
    python3 scripts/tune/tune_multi.py --phases matching,grid_optim --n-trials 20 --fast-proxy
    python3 scripts/tune/tune_multi.py --phases matching --n-trials 15
    python3 scripts/tune/tune_multi.py --validate
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

# Phase execution order — earlier phases must be tuned before later ones
PHASE_ORDER = ["coarse", "scale_rotation", "matching", "validation",
               "grid_optim", "flow", "normalization"]

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_TUNE_CASES_PATH = PROJECT_ROOT / "data" / "tune_cases.yaml"


def load_config(path: str | None = None) -> dict:
    with open(path or _TUNE_CASES_PATH) as f:
        return yaml.safe_load(f)


def get_cases(config: dict) -> dict:
    return config.get("cases", {})


def get_param_groups(config: dict) -> dict:
    return config.get("param_groups", {})


def group_cases_by_profile(cases: dict) -> dict[str, list[str]]:
    """Group case IDs by profile name."""
    groups: dict[str, list[str]] = defaultdict(list)
    for case_id, case in cases.items():
        groups[case["profile"]].append(case_id)
    return dict(groups)


# ---------------------------------------------------------------------------
# Best-profile management
# ---------------------------------------------------------------------------

def _best_profile_path(profile: str) -> Path:
    return PROJECT_ROOT / "data" / "profiles" / f"best_{profile}.yaml"


def _init_best_profile(profile: str) -> Path:
    """Create best_{profile}.yaml from the base profile if it doesn't exist."""
    bp = _best_profile_path(profile)
    if bp.exists():
        return bp

    # Copy the source profile as starting point
    src = PROJECT_ROOT / "data" / "profiles" / f"{profile}.yaml"
    if not src.exists():
        src = PROJECT_ROOT / "data" / "profiles" / "_base.yaml"

    src_data = yaml.safe_load(src.read_text()) or {}
    # Ensure it inherits from the real profile (not _base)
    src_data.setdefault("inherits", profile)
    src_data.setdefault("meta", {})
    src_data["meta"]["name"] = f"best_{profile}"
    src_data["meta"]["description"] = f"Auto-tuned best params for {profile}"

    with open(bp, "w") as f:
        yaml.dump(src_data, f, default_flow_style=False, sort_keys=False)

    print(f"  Initialised best profile: {bp}")
    return bp


def _update_best_profile(profile: str, phase: str, optuna_params: dict):
    """Merge best params from a completed phase into best_{profile}.yaml."""
    bp = _init_best_profile(profile)
    data = yaml.safe_load(bp.read_text()) or {}

    updates = _map_params_to_profile(optuna_params, phase)
    for key, value in updates.items():
        parts = key.split("__")
        if len(parts) == 2:
            section, attr = parts
            if section not in data:
                data[section] = {}
            data[section][attr] = value

    with open(bp, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"  Updated best profile: {bp}")
    for k, v in sorted(updates.items()):
        print(f"    {k} = {v}")


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def _resolve_case_paths(case: dict) -> tuple[str, str, str]:
    """Return (target, reference, anchors) with absolute paths."""
    anchors = case.get("anchors", "data/bahrain_anchor_gcps.json")
    if not os.path.isabs(anchors):
        anchors = str(PROJECT_ROOT / anchors)
    target = os.path.expanduser(case["target"])
    if not os.path.isabs(target):
        target = str(PROJECT_ROOT / target)
    reference = os.path.expanduser(case["reference"])
    if not os.path.isabs(reference):
        reference = str(PROJECT_ROOT / reference)
    return target, reference, anchors


def build_checkpoint(case_id: str, case: dict, phase: str,
                     profile_override: str | None = None,
                     force: bool = False) -> str | None:
    """Build checkpoint for a case by running the full pipeline.

    Uses *profile_override* (e.g. best_{profile}) if given, otherwise
    uses the case's own profile.  Returns checkpoint_dir or None on failure.
    """
    from scripts.tune.tune import _phase_checkpoint_id

    required = _phase_checkpoint_id(phase)
    case_dir = PROJECT_ROOT / "diagnostics" / f"tune_{case_id}"
    checkpoint_dir = case_dir / "checkpoints"
    checkpoint_json = checkpoint_dir / f"{required}.json"

    if checkpoint_json.exists() and not force:
        print(f"  [{case_id}] Checkpoint {required} exists")
        return str(checkpoint_dir)

    print(f"  [{case_id}] Building checkpoint {required}...")

    target, reference, anchors = _resolve_case_paths(case)
    profile = profile_override or case["profile"]

    output_path = str(case_dir / "tune_baseline.tif")
    os.makedirs(str(case_dir), exist_ok=True)

    cmd = [
        sys.executable, str(PROJECT_ROOT / "auto-align.py"),
        target, "--reference", reference,
        "--anchors", anchors,
        "--output", output_path,
        "--yes", "--best",
        "--profile", profile,
        "--diagnostics-dir", str(case_dir),
    ]

    for prior_path in case.get("metadata_priors", []):
        cmd.extend(["--metadata-priors", os.path.expanduser(prior_path)])

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        elapsed = time.time() - t0
    except subprocess.TimeoutExpired:
        print(f"  [{case_id}] Pipeline timed out after 7200s")
        return None

    if result.returncode != 0:
        print(f"  [{case_id}] Pipeline FAILED after {time.time() - t0:.0f}s")
        fail_log = case_dir / "checkpoint_failure.log"
        with open(fail_log, "w") as f:
            f.write(f"STDOUT:\n{result.stdout[-3000:]}\n\nSTDERR:\n{result.stderr[-3000:]}")
        print(f"  [{case_id}] Failure log: {fail_log}")
        return None

    if not checkpoint_json.exists():
        print(f"  [{case_id}] ERROR: Expected checkpoint {required} not created")
        return None

    print(f"  [{case_id}] Checkpoint ready ({elapsed:.0f}s)")
    return str(checkpoint_dir)


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------

def aggregate_scores(scores: dict[str, float], cases: dict,
                     case_ids: list[str] | None = None) -> float:
    """Weighted mean score across cases. Returns 999.0 if no valid scores."""
    if case_ids is None:
        case_ids = list(scores.keys())

    total_weight = 0.0
    weighted_sum = 0.0

    for cid in case_ids:
        if cid not in scores or scores[cid] >= 999.0:
            continue
        w = cases[cid].get("weight", 1.0)
        weighted_sum += w * scores[cid]
        total_weight += w

    if total_weight == 0:
        return 999.0
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Sequential per-profile tuning
# ---------------------------------------------------------------------------

def run_sequential(config: dict, phases: list[str], n_trials: int,
                   fast_proxy: bool, seed: int, force_rebuild: bool,
                   plateau_patience: int):
    """Tune phases in pipeline order, rebuilding checkpoints between phases.

    For each profile:
      1. Build/refresh checkpoints using best profile so far
      2. Tune phase N → update best_{profile}.yaml
      3. Rebuild checkpoints with updated best profile
      4. Tune phase N+1 → update best_{profile}.yaml
      ...
    """
    cases = get_cases(config)
    profile_groups = group_cases_by_profile(cases)

    # Sort phases into pipeline order
    phase_set = set(phases)
    ordered_phases = [p for p in PHASE_ORDER if p in phase_set]
    if set(ordered_phases) != phase_set:
        unknown = phase_set - set(PHASE_ORDER)
        print(f"WARNING: Unknown phases ignored: {unknown}")

    print(f"\n{'='*60}")
    print(f"SEQUENTIAL TUNING")
    print(f"  Profiles: {list(profile_groups.keys())}")
    print(f"  Phases (in order): {ordered_phases}")
    print(f"  Trials per study: {n_trials}")
    print(f"{'='*60}")

    for profile, case_ids in profile_groups.items():
        print(f"\n{'='*60}")
        print(f"Profile: {profile} (cases: {case_ids})")
        print(f"{'='*60}")

        # Initialise best profile YAML
        best_profile = f"best_{profile}"
        _init_best_profile(profile)

        for phase_idx, phase in enumerate(ordered_phases):
            print(f"\n--- Phase {phase_idx + 1}/{len(ordered_phases)}: "
                  f"{phase} (profile={profile}) ---")

            # Step 1: Build/rebuild checkpoints using best profile
            # Force rebuild if this isn't the first phase (so we pick up
            # best params from previous phase)
            need_rebuild = force_rebuild or (phase_idx > 0)
            checkpoint_dirs = {}
            for cid in case_ids:
                ckpt = build_checkpoint(
                    cid, cases[cid], phase,
                    profile_override=best_profile,
                    force=need_rebuild,
                )
                if ckpt is None:
                    print(f"  Skipping case {cid} — checkpoint build failed")
                    continue
                checkpoint_dirs[cid] = ckpt

            if not checkpoint_dirs:
                print(f"  No valid checkpoints for {profile}/{phase}, skipping")
                continue

            # Step 2: Run Optuna study for this phase
            best_params = _run_profile_study(
                profile=profile,
                phase=phase,
                case_ids=list(checkpoint_dirs.keys()),
                cases=cases,
                checkpoint_dirs=checkpoint_dirs,
                n_trials=n_trials,
                fast_proxy=fast_proxy,
                seed=seed,
                plateau_patience=plateau_patience,
            )

            # Step 3: Update best profile with winning params
            if best_params is not None:
                _update_best_profile(profile, phase, best_params)
            else:
                print(f"  No improvement found for {phase}, keeping prior best")


def _run_profile_study(profile: str, phase: str, case_ids: list[str],
                       cases: dict, checkpoint_dirs: dict[str, str],
                       n_trials: int, fast_proxy: bool, seed: int,
                       plateau_patience: int) -> dict | None:
    """Run an Optuna study for one profile + phase.

    Returns the best trial's params dict, or None if no trials completed.
    """
    try:
        import optuna
    except ImportError:
        print("ERROR: optuna not installed. Run: pip install optuna")
        sys.exit(1)

    from scripts.tune.tune import (OBJECTIVE_MAP, check_plateau,
                               _write_plateau_report, _cleanup_old_trials)

    study_dir = str(PROJECT_ROOT / "diagnostics" / f"tune_{profile}")
    os.makedirs(study_dir, exist_ok=True)

    db_path = os.path.join(study_dir, f"tune_{profile}_multi.db")
    storage = f"sqlite:///{db_path}"
    study_name = f"tune_{phase}_{profile}_multi"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=5, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
        storage=storage,
        load_if_exists=True,
    )

    print(f"\n  Study: {study_name} ({n_trials} trials)")
    print(f"  Cases: {case_ids}")

    objective_fn = OBJECTIVE_MAP[phase]
    plateau_reported = False
    t0 = time.time()

    def _multi_objective(trial):
        nonlocal plateau_reported

        mock_args = argparse.Namespace(
            profile=profile, fast_proxy=fast_proxy,
            case=None, target=None, reference=None, anchors=None,
        )

        scores = {}
        for cid in case_ids:
            try:
                score = objective_fn(trial, checkpoint_dirs[cid], mock_args)
                scores[cid] = score
                print(f"    {cid}: {score:.1f}", flush=True)
            except Exception as e:
                print(f"    {cid}: FAILED ({e})", flush=True)
                scores[cid] = 999.0
            # Free memory between cases
            gc.collect()
            try:
                import torch
                if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except ImportError:
                pass

        agg = aggregate_scores(scores, cases, case_ids)
        print(f"    aggregate: {agg:.1f}")

        for cid, s in scores.items():
            trial.set_user_attr(f"score_{cid}", s)

        if not plateau_reported and check_plateau(
                study, patience=plateau_patience):
            print(f"\n  *** PLATEAU DETECTED ***")
            _write_plateau_report(study, phase, study_dir)
            plateau_reported = True

        if trial.number % 5 == 4:
            _cleanup_old_trials(study_dir, keep_best_n=3)

        return agg

    try:
        study.optimize(_multi_objective, n_trials=n_trials, gc_after_trial=True)
    except KeyboardInterrupt:
        print("\nInterrupted.")

    total_time = time.time() - t0
    print(f"\n  Study complete: {study_name} ({total_time:.0f}s)")

    best_params = None
    try:
        best = study.best_trial
        best_params = best.params
        print(f"  Best aggregate score: {best.value:.1f} (trial {best.number})")
        for k, v in sorted(best.params.items()):
            print(f"    {k}: {v}")

        best_path = os.path.join(study_dir, f"best_{phase}.json")
        with open(best_path, "w") as f:
            json.dump({
                "phase": phase,
                "profile": profile,
                "aggregate_score": best.value,
                "trial_number": best.number,
                "params": best.params,
                "per_case_scores": {
                    k: v for k, v in best.user_attrs.items()
                    if k.startswith("score_")
                },
                "total_trials": len(study.trials),
                "total_time_s": total_time,
                "fast_proxy": fast_proxy,
            }, f, indent=2)
        print(f"  Saved: {best_path}")
    except ValueError:
        print("  No completed trials.")

    _cleanup_old_trials(study_dir, keep_best_n=3)
    return best_params


# ---------------------------------------------------------------------------
# Param mapping
# ---------------------------------------------------------------------------

def _map_params_to_profile(optuna_params: dict, phase: str) -> dict:
    """Map Optuna trial param names to double-underscore profile keys."""
    phase_section = {
        "coarse": "coarse",
        "scale_rotation": "scale_rotation",
        "grid_optim": "grid_optim",
        "matching": "matching",
        "validation": "validation",
        "flow": "flow",
        "normalization": "normalization",
    }

    section = phase_section.get(phase, "")
    result = {}

    level_scales = {
        "level1_data_scale": ("grid_optim__level_w_data_scale", 1),
        "level2_data_scale": ("grid_optim__level_w_data_scale", 2),
        "level1_disp_scale": ("grid_optim__level_w_disp_scale", 1),
        "level2_disp_scale": ("grid_optim__level_w_disp_scale", 2),
        "level1_reg_scale": ("grid_optim__level_reg_scale", 1),
        "level2_reg_scale": ("grid_optim__level_reg_scale", 2),
        "level1_chamfer_scale": ("grid_optim__level_chamfer_scale", 1),
        "level2_chamfer_scale": ("grid_optim__level_chamfer_scale", 2),
    }

    scale_lists: dict[str, list[float]] = {}

    for param_name, value in optuna_params.items():
        if param_name in level_scales:
            key, idx = level_scales[param_name]
            if key not in scale_lists:
                scale_lists[key] = [1.0, 1.0, 1.0]
            scale_lists[key][idx] = value
        else:
            if section:
                result[f"{section}__{param_name}"] = value
            else:
                result[param_name] = value

    result.update(scale_lists)
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_all(config: dict):
    """Run full pipeline on all cases with best profiles, report scores."""
    cases = get_cases(config)

    print(f"\n{'='*60}")
    print(f"VALIDATION — full pipeline on all cases")
    print(f"{'='*60}")

    results = {}
    for cid, case in cases.items():
        profile = case["profile"]
        # Use best profile if it exists, otherwise original
        best_path = _best_profile_path(profile)
        use_profile = f"best_{profile}" if best_path.exists() else profile
        print(f"\n--- {cid} ({use_profile}) ---")

        target, reference, anchors = _resolve_case_paths(case)

        val_dir = str(PROJECT_ROOT / "diagnostics" / f"validate_{cid}")
        os.makedirs(val_dir, exist_ok=True)
        output_path = os.path.join(val_dir, "aligned.tif")

        cmd = [
            sys.executable, str(PROJECT_ROOT / "auto-align.py"),
            target, "--reference", reference,
            "--anchors", anchors,
            "--output", output_path,
            "--yes", "--best",
            "--profile", use_profile,
            "--diagnostics-dir", val_dir,
        ]

        for prior_path in case.get("metadata_priors", []):
            cmd.extend(["--metadata-priors", prior_path])

        t0 = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            elapsed = time.time() - t0
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after 7200s")
            results[cid] = {"score": 999.0, "error": "timeout"}
            continue

        if result.returncode != 0:
            print(f"  FAILED after {time.time() - t0:.0f}s")
            results[cid] = {"score": 999.0, "error": "pipeline_failed"}
            continue

        summary_path = os.path.join(val_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            score = summary.get("score", 999.0)
            accepted = summary.get("accepted", False)
            print(f"  score={score:.1f}, accepted={accepted} ({elapsed:.0f}s)")
            results[cid] = {
                "score": score,
                "accepted": accepted,
                "elapsed_s": elapsed,
            }
        else:
            print(f"  No summary.json found")
            results[cid] = {"score": 999.0, "error": "no_summary"}

    scores = {cid: r["score"] for cid, r in results.items()}
    agg = aggregate_scores(scores, cases)
    print(f"\n{'='*60}")
    print(f"Aggregate score: {agg:.1f}")
    for cid, r in results.items():
        print(f"  {cid}: score={r['score']:.1f}")
    print(f"{'='*60}")

    report_path = str(PROJECT_ROOT / "diagnostics" / "validation_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "aggregate_score": agg,
            "cases": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"\nReport: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-image sequential hyperparameter tuning")
    parser.add_argument("--phases", default="matching,grid_optim",
                        help="Comma-separated phases to tune (run in pipeline order)")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Trials per study (default: 20)")
    parser.add_argument("--fast-proxy", action="store_true",
                        help="Use fast proxy mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Force rebuild all checkpoints")
    parser.add_argument("--plateau-patience", type=int, default=5,
                        help="Trials without improvement before plateau")
    parser.add_argument("--config", default=None,
                        help="Path to tune cases YAML (default: data/tune_cases.yaml)")
    parser.add_argument("--validate", action="store_true",
                        help="Run full validation on all cases")
    args = parser.parse_args()

    config = load_config(args.config)
    phases = [p.strip() for p in args.phases.split(",")]

    from scripts.tune.tune import OBJECTIVE_MAP
    for p in phases:
        if p not in OBJECTIVE_MAP:
            print(f"ERROR: Unknown phase '{p}'. Available: {list(OBJECTIVE_MAP.keys())}")
            sys.exit(1)

    if args.validate:
        validate_all(config)
        return

    t0 = time.time()

    run_sequential(
        config, phases, args.n_trials, args.fast_proxy,
        args.seed, args.force_rebuild, args.plateau_patience)

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Total tuning time: {total:.0f}s ({total/60:.1f}m)")
    print(f"{'='*60}")
    print(f"\nBest profiles saved to data/profiles/best_*.yaml")
    print(f"Validate: python3 scripts/tune/tune_multi.py --validate")


if __name__ == "__main__":
    main()
