#!/usr/bin/env python3
"""
Single-scene phase iteration harness.

The full e2e pipeline is 40-80 min per scene; iterating on a tail phase
(grid optim, flow refinement, QA) only costs the time of that phase
plus warp + QA — typically 2-10 min — when a checkpoint is loaded.

Workflow:
    # 1. Pick a manifest from any successful run (e2e_v38 here)
    poetry run python scripts/test/iterate_phase.py \
        --manifest diagnostics/e2e_v38/pipeline_output/manifests/alignment_manifest.json \
        --scene DS1104-1057DA023 \
        --build

    # 2. Iterate — costs ~5 min instead of 40+
    poetry run python scripts/test/iterate_phase.py \
        --manifest diagnostics/e2e_v38/pipeline_output/manifests/alignment_manifest.json \
        --scene DS1104-1057DA023 \
        --from-phase post_validate

The harness:
  - Snapshots the baseline qa.json on first build
  - On each iteration: reruns auto-align with --resume-from-checkpoint
  - Prints baseline vs latest QA score delta
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PHASE_ORDER = [
    "post_setup",
    "post_coarse",
    "post_scale_rotation",
    "post_match",
    "post_validate",
]


def load_job(manifest_path: Path, scene: str) -> tuple[dict, dict]:
    """Return (shared, job-dict) for the named scene in the manifest."""
    with open(manifest_path) as f:
        m = json.load(f)
    shared = m.get("shared", {})
    for job in m.get("jobs", []):
        diag = job.get("diagnostics_dir") or ""
        if scene in diag or scene in os.path.basename(job.get("input", "")):
            return shared, job
    raise SystemExit(
        f"No job for {scene!r} in {manifest_path}.\n"
        f"Available: " + ", ".join(
            os.path.basename(j.get('diagnostics_dir', '?'))
            for j in m.get('jobs', [])
        )
    )


def find_latest_checkpoint(checkpoint_dir: Path) -> str | None:
    """Return the highest checkpoint name present, or None."""
    for name in reversed(PHASE_ORDER):
        if (checkpoint_dir / f"{name}.json").exists():
            return name
    return None


def read_qa(qa_path: Path) -> dict:
    """Extract selected-candidate metrics from qa.json."""
    if not qa_path.exists():
        return {}
    payload = json.loads(qa_path.read_text())
    selected = payload.get("selected_candidate")
    for r in payload.get("reports", []):
        if r.get("candidate") == selected:
            img = r.get("image_metrics", {}) or {}
            return {
                "candidate": selected,
                "score": r.get("total_score"),
                "accepted": r.get("accepted"),
                "patch_med": img.get("patch_med"),
                "grid_score": img.get("grid_score"),
                "stable_boundary_m": img.get("stable_boundary_m"),
                "shore_boundary_m": img.get("shore_boundary_m"),
                "cv_mean_m": r.get("cv_mean_m"),
                "holdout_median_m": (
                    r.get("holdout_metrics", {}) or {}).get("post_warp", {}).get("median_m"),
            }
    return {}


def fmt(v, places=2):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    try:
        return f"{float(v):.{places}f}"
    except (TypeError, ValueError):
        return str(v)


def print_qa_compare(baseline: dict, latest: dict) -> None:
    fields = ["score", "accepted", "candidate", "patch_med", "grid_score",
              "stable_boundary_m", "shore_boundary_m", "cv_mean_m",
              "holdout_median_m"]
    name_w = max(len(f) for f in fields)
    print(f"\n{'metric':<{name_w}}  {'baseline':>14}  {'latest':>14}  {'delta':>10}")
    print("-" * (name_w + 2 + 14 + 2 + 14 + 2 + 10))
    for f in fields:
        b = baseline.get(f)
        l = latest.get(f)
        delta = "—"
        try:
            if isinstance(b, (int, float)) and isinstance(l, (int, float)):
                d = float(l) - float(b)
                delta = f"{d:+.2f}"
        except (TypeError, ValueError):
            pass
        print(f"{f:<{name_w}}  {fmt(b):>14}  {fmt(l):>14}  {delta:>10}")


def build_cmd(job: dict, shared: dict, *, resume_from: str | None,
              keep_temp: bool, qa_out: Path | None) -> list[str]:
    cmd = [
        sys.executable, str(PROJECT_ROOT / "auto-align.py"),
        job["input"],
        "--reference", job.get("reference") or shared["reference"],
        "--output", job["output"],
        "--diagnostics-dir", job["diagnostics_dir"],
        "--profile", job.get("profile") or shared.get("profile") or "_base",
        "--yes",
    ]
    if qa_out is not None:
        cmd += ["--qa-json", str(qa_out)]
    elif job.get("qa_json"):
        cmd += ["--qa-json", job["qa_json"]]
    rw = job.get("reference_window") or shared.get("reference_window")
    if rw:
        cmd += ["--reference-window", rw]
    priors = job.get("metadata_priors") or shared.get("metadata_priors") or []
    for p in (priors if isinstance(priors, list) else [priors]):
        cmd += ["--metadata-priors", p]
    if shared.get("global_search") is False or job.get("global_search") is False:
        cmd += ["--no-global-search"]
    else:
        cmd += ["--global-search"]
    if shared.get("allow_abstain") or job.get("allow_abstain"):
        cmd += ["--allow-abstain"]
    if resume_from:
        cmd += ["--resume-from-checkpoint", resume_from]
    if keep_temp:
        cmd += ["--keep-temp-paths"]
    extra = job.get("_extra_args") or []
    cmd += list(extra)
    return cmd


def main():
    ap = argparse.ArgumentParser(
        description="Iterate one phase of the alignment pipeline using a checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--manifest", required=True, type=Path,
                    help="alignment_manifest.json from a prior e2e run")
    ap.add_argument("--scene", required=True,
                    help="entity_id substring to select one job")
    ap.add_argument("--build", action="store_true",
                    help="Run the full pipeline once and snapshot a baseline. "
                         "Required on first use; later overwrites baseline.")
    ap.add_argument("--from-phase", default=None,
                    choices=PHASE_ORDER,
                    help="Resume from this checkpoint. Defaults to the highest "
                         "checkpoint present.")
    ap.add_argument("--label", default=None,
                    help="Optional label for the iteration (e.g. 'roma_size_640'). "
                         "Tags qa output and the printed run header.")
    ap.add_argument("--no-snapshot", action="store_true",
                    help="Don't write a separate qa_<label>.json — just overwrite "
                         "the canonical qa.json (and compare to baseline).")
    ap.add_argument("--extra-arg", action="append", default=[],
                    help="Pass-through to auto-align.py (repeatable). "
                         "Example: --extra-arg --tps-fallback")
    args = ap.parse_args()

    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        sys.exit(f"manifest not found: {manifest_path}")

    shared, job = load_job(manifest_path, args.scene)
    job = {**job, "_extra_args": args.extra_arg}
    diag_dir = Path(job["diagnostics_dir"])
    ckpt_dir = diag_dir / "checkpoints"
    qa_canon = Path(job.get("qa_json") or (diag_dir / "qa.json"))
    qa_baseline = diag_dir / "qa_baseline.json"

    diag_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    label = args.label or time.strftime("%Y%m%d_%H%M%S")
    qa_iter = qa_canon if args.no_snapshot else (diag_dir / f"qa_{label}.json")

    # Determine mode
    have_ckpt = find_latest_checkpoint(ckpt_dir) is not None
    if args.build or not have_ckpt:
        if not have_ckpt:
            print(f"[iterate_phase] No checkpoints in {ckpt_dir}; building.")
        else:
            print(f"[iterate_phase] --build set; rebuilding from scratch.")
        # Wipe stale checkpoints to avoid accidental reuse
        for f in ckpt_dir.glob("*"):
            f.unlink()
        cmd = build_cmd(job, shared, resume_from=None, keep_temp=True,
                        qa_out=qa_canon)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
        elapsed = time.time() - t0
        if rc != 0:
            sys.exit(f"build run failed (rc={rc}, elapsed={elapsed:.0f}s)")
        if qa_canon.exists():
            shutil.copy2(qa_canon, qa_baseline)
            print(f"[iterate_phase] Baseline QA snapshot → {qa_baseline}")
        else:
            print(f"[iterate_phase] WARNING: no qa.json produced at {qa_canon}")
        baseline = read_qa(qa_baseline)
        latest = read_qa(qa_canon)
        print_qa_compare(baseline, latest)
        print(f"\n[iterate_phase] Build elapsed: {elapsed/60:.1f} min")
        return

    # Iteration mode
    resume_from = args.from_phase or find_latest_checkpoint(ckpt_dir)
    if resume_from is None:
        sys.exit("No checkpoints — pass --build first.")
    if not (ckpt_dir / f"{resume_from}.json").exists():
        present = sorted(p.stem for p in ckpt_dir.glob("post_*.json"))
        sys.exit(
            f"checkpoint {resume_from} not present in {ckpt_dir}. "
            f"Have: {present}")

    if not qa_baseline.exists() and qa_canon.exists():
        shutil.copy2(qa_canon, qa_baseline)
        print(f"[iterate_phase] Adopted existing qa.json as baseline → {qa_baseline}")

    print(f"[iterate_phase] scene={args.scene} resume_from={resume_from} "
          f"label={label}")
    cmd = build_cmd(job, shared, resume_from=resume_from, keep_temp=True,
                    qa_out=qa_iter)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
    elapsed = time.time() - t0
    if rc != 0:
        sys.exit(f"iteration run failed (rc={rc}, elapsed={elapsed:.0f}s)")

    baseline = read_qa(qa_baseline)
    latest = read_qa(qa_iter)
    print_qa_compare(baseline, latest)
    print(f"\n[iterate_phase] Iteration elapsed: {elapsed/60:.1f} min "
          f"(resume_from={resume_from}, label={label})")
    if not args.no_snapshot:
        print(f"[iterate_phase] QA snapshot: {qa_iter}")


if __name__ == "__main__":
    main()
