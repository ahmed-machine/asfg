#!/usr/bin/env python3
"""
Cross-version comparison of alignment pipeline results.

Reads diagnostics/run_v*/qa.json (and summary.json when available) to produce
a comparison table and delta analysis.

Usage:
    python3 scripts/test/compare.py [--format text|json] [--baseline N]
"""

import argparse
import json
import re
import sys
from glob import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DIAG_DIR = PROJECT_ROOT / "diagnostics"


def load_version(version_dir):
    """Load metrics for a single version from qa.json and optionally summary.json."""
    v_path = Path(version_dir)
    m = re.search(r"run_v(\d+)$", str(v_path))
    if not m:
        return None
    version = int(m.group(1))

    qa_path = v_path / "qa.json"
    summary_path = v_path / "summary.json"

    def extract_metrics(report):
        if not report:
            return None
        im = report.get("image_metrics", {})
        m = {
            "score": round(im.get("score", 0), 1),
            "total": round(report.get("total_score", 0), 1),
            "accepted": report.get("accepted", False),
            "west": round(im["west"]) if im.get("west") is not None else None,
            "center": round(im["center"]) if im.get("center") is not None else None,
            "east": round(im["east"]) if im.get("east") is not None else None,
            "north": round(im["north_shift"]) if im.get("north_shift") is not None else None,
            "patch_med": round(im.get("patch_med", 0)),
            "patch_p90": round(im.get("patch_p90", 0), 1) if im.get("patch_p90") is not None else None,
            "patch_count": im.get("patch_count"),
            "stable_iou": round(im.get("stable_iou", 0), 3),
            "shore_iou": round(im.get("shore_iou", 0), 3),
            "cv_mean_m": round(report.get("cv_mean_m", 0), 1) if report.get("cv_mean_m") is not None else None,
            "coverage": round(report.get("coverage", 0), 3) if report.get("coverage") is not None else None,
            "grid_score": round(im["grid_score"], 1) if im.get("grid_score") is not None else None,
            "reasons": report.get("reasons", []),
        }
        if im.get("grid"):
            m["grid_valid"] = im["grid"].get("valid_count")
            m["grid_total"] = im["grid"].get("total_count")
        if im.get("score_breakdown"):
            m["score_breakdown"] = im["score_breakdown"]
        return m

    grid_metrics = None
    tps_metrics = None
    selected = "grid"
    gcp_count = None

    if qa_path.exists():
        try:
            qa = json.loads(qa_path.read_text())
            grid = {}
            tps = {}
            for report in qa.get("reports", []):
                if report.get("candidate") == "grid":
                    grid = report
                elif report.get("candidate") == "tps":
                    tps = report
            selected = qa.get("selected_candidate", "grid")
            grid_metrics = extract_metrics(grid)
            tps_metrics = extract_metrics(tps)
            gcp_count = qa.get("metadata", {}).get("gcp_count")
        except Exception:
            pass

    # Enrich/fallback with summary.json
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            # Use summary grid/tps if qa.json didn't provide them
            if grid_metrics is None and summary.get("grid"):
                grid_metrics = summary["grid"]
            if tps_metrics is None and summary.get("tps"):
                tps_metrics = summary["tps"]
            if not gcp_count:
                gcp_count = summary.get("gcp_count")
            if summary.get("selected"):
                selected = summary["selected"]
        except Exception:
            pass

    # Need at least one set of metrics
    if grid_metrics is None and tps_metrics is None:
        return None

    row = {
        "version": version,
        "grid": grid_metrics,
        "tps": tps_metrics,
        "selected": selected,
        "gcp_count": gcp_count,
    }

    # Enrich with summary.json details
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            row["git_commit"] = summary.get("git_commit")
            row["wall_clock_s"] = summary.get("wall_clock_s")
            row["anchors"] = summary.get("anchors")
            row["auto_gcps"] = summary.get("auto_gcps")
            row["flow"] = summary.get("flow")
            row["grid_optimizer"] = summary.get("grid_optimizer")
            row["reclamation"] = summary.get("reclamation")
            row["icp"] = summary.get("icp")
            row["step_timings"] = summary.get("step_timings")
            row["exit_code"] = summary.get("exit_code")
        except Exception:
            pass

    return row


def load_all_versions():
    """Load all versions sorted by version number."""
    dirs = glob(str(DIAG_DIR / "run_v*"))
    rows = []
    for d in dirs:
        row = load_version(d)
        if row:
            rows.append(row)
    rows.sort(key=lambda r: r["version"])
    return rows


def compute_delta(current, baseline, field, lower_is_better=True):
    """Compute delta and improvement status."""
    if current is None or baseline is None:
        return None
    delta = current - baseline
    if lower_is_better:
        improved = delta < 0
    else:
        improved = delta > 0
    return {
        "baseline": baseline,
        "current": current,
        "delta": round(delta, 3),
        "improved": improved,
    }


def find_best(rows, candidate_key, field, lower_is_better=True):
    """Find the best version for a given field."""
    best_val = None
    best_ver = None
    for row in rows:
        metrics = row.get(candidate_key)
        if not metrics:
            continue
        val = metrics.get(field)
        if val is None:
            continue
        if best_val is None or (lower_is_better and val < best_val) or (not lower_is_better and val > best_val):
            best_val = val
            best_ver = row["version"]
    return best_ver, best_val


def format_text(rows, baseline_version=None):
    """Format comparison as text table."""
    if not rows:
        print("No versions found.")
        return

    latest = rows[-1]
    if baseline_version is not None:
        baseline = next((r for r in rows if r["version"] == baseline_version), None)
    else:
        baseline = rows[-2] if len(rows) >= 2 else None

    # Table header
    print("=== Version Comparison (grid candidate) ===")
    header = f" {'Ver':>4}  {'GCPs':>4}  {'W':>4}  {'C':>4}  {'E':>5}  {'N':>5}  {'Patch':>5}  {'StIoU':>6}  {'ShIoU':>6}  {'Score':>5}  {'CV_m':>5}  {'Total':>5}  {'Acc':>3}"
    print(header)

    for row in rows:
        g = row.get("grid")
        if not g:
            continue
        v = row["version"]
        prefix = "*" if v == latest["version"] else " "
        gcps = row.get("gcp_count", "?")
        cv = g.get("cv_mean_m")
        cv_str = f"{cv:5.0f}" if cv is not None else "    ?"
        acc = "yes" if g.get("accepted") else " no"
        siou = g.get('stable_iou')
        siou_str = f"{siou:>6.3f}" if siou is not None else "     ?"
        shiou = g.get('shore_iou')
        shiou_str = f"{shiou:>6.3f}" if shiou is not None else "     ?"
        w_str = f"{g['west']:>3}m" if g.get('west') is not None else " n/a"
        c_str = f"{g['center']:>3}m" if g.get('center') is not None else " n/a"
        e_str = f"{g['east']:>4}m" if g.get('east') is not None else "  n/a"
        n_str = f"{g['north']:>+4}m" if g.get('north') is not None else "  n/a"
        print(f"{prefix}v{v:>3}  {gcps:>4}  {w_str}  {c_str}  {e_str}  {n_str}  {g['patch_med']:>4}m  {siou_str}  {shiou_str}  {g['score']:>5.0f}  {cv_str}  {g['total']:>5.0f}  {acc:>3}")

    # Delta vs baseline
    if baseline and latest["version"] != baseline["version"]:
        print(f"\n=== Delta vs v{baseline['version']} (previous) ===")
        lg = latest.get("grid", {})
        bg = baseline.get("grid", {})

        deltas = [
            ("score", "score", True),
            ("total", "total", True),
            ("cv_mean_m", "cv_mean_m", True),
            ("west", "west", True),
            ("center", "center", True),
            ("east", "east", True),
            ("north", "north", None),  # absolute value matters
            ("patch_med", "patch_med", True),
            ("patch_p90", "patch_p90", True),
            ("stable_iou", "stable_iou", False),
            ("shore_iou", "shore_iou", False),
        ]

        for label, key, lower_better in deltas:
            cv = lg.get(key)
            bv = bg.get(key)
            if cv is None or bv is None:
                continue
            delta = cv - bv
            if lower_better is not None:
                improved = (delta < 0) if lower_better else (delta > 0)
                tag = "IMPROVED" if improved else "regressed" if abs(delta) > 0.001 else ""
            else:
                tag = ""
            if isinstance(cv, float) and abs(cv) < 10:
                print(f"  {label:>12}: {bv:.3f} -> {cv:.3f}  ({delta:+.3f}{', ' + tag if tag else ''})")
            else:
                print(f"  {label:>12}: {bv:.0f} -> {cv:.0f}  ({delta:+.0f}{', ' + tag if tag else ''})")

        # GCP count
        lgc = latest.get("gcp_count")
        bgc = baseline.get("gcp_count")
        if lgc is not None and bgc is not None:
            print(f"  {'gcp_count':>12}: {bgc} -> {lgc}  ({lgc - bgc:+d})")

        # Accepted change
        la = lg.get("accepted")
        ba = bg.get("accepted")
        if la is not None and ba is not None and la != ba:
            milestone = "MILESTONE" if la else "REGRESSION"
            print(f"  {'accepted':>12}: {'yes' if ba else 'no'} -> {'yes' if la else 'no'}  ({milestone})")

        # Wall clock
        lwc = latest.get("wall_clock_s")
        bwc = baseline.get("wall_clock_s")
        if lwc is not None and bwc is not None:
            delta = lwc - bwc
            tag = "faster" if delta < 0 else "slower" if delta > 0 else ""
            print(f"  {'wall_clock':>12}: {bwc:.0f}s -> {lwc:.0f}s  ({delta:+.0f}s{', ' + tag if tag else ''})")

        # Flow reliable_pct
        lf = latest.get("flow", {})
        bf = baseline.get("flow", {})
        lrp = lf.get("reliable_pct") if lf else None
        brp = bf.get("reliable_pct") if bf else None
        if lrp is not None and brp is not None:
            delta = lrp - brp
            tag = "IMPROVED" if delta > 0 else "regressed" if delta < 0 else ""
            print(f"  {'flow_rel%':>12}: {brp}% -> {lrp}%  ({delta:+d}%{', ' + tag if tag else ''})")

    # Score breakdown (latest)
    lg = latest.get("grid", {})
    sb = lg.get("score_breakdown")
    if sb:
        print(f"\n=== Score Breakdown (v{latest['version']}) ===")
        total_score = lg.get("score", 0)
        # Support both old keys (west_contrib etc.) and new keys (grid_contrib etc.)
        breakdown_keys = ["grid_contrib", "patch_contrib", "stable_iou_penalty", "shore_iou_penalty",
                          "west_contrib", "center_contrib", "east_contrib", "north_contrib"]
        for key in breakdown_keys:
            val = sb.get(key)
            if val is None:
                continue
            pct = (val / total_score * 100) if total_score > 0 else 0
            label = key.replace("_contrib", "").replace("_penalty", " penalty")
            print(f"  {label:>20}: {val:5.1f} pts  ({pct:4.1f}%)")

    # Process Health (latest)
    print(f"\n=== Process Health (v{latest['version']}) ===")
    # Anchors
    anch = latest.get("anchors", {})
    if anch:
        rej_str = f", {anch['rejected']} rejected" if anch.get('rejected') else ""
        print(f"  Anchors: {anch.get('located', '?')}/{anch.get('total', '?')} located{rej_str}")
    # Flow
    fl = latest.get("flow", {})
    if fl:
        parts = []
        if fl.get("reliable_pct") is not None:
            parts.append(f"{fl['reliable_pct']}% reliable")
        if fl.get("applied") is not None:
            parts.append("applied" if fl["applied"] else "SKIPPED")
        if fl.get("mean_correction_m") is not None:
            parts.append(f"mean {fl['mean_correction_m']:.1f}m")
        if parts:
            print(f"  Flow: {', '.join(parts)}")
    # Grid optimizer / fold
    go = latest.get("grid_optimizer", {})
    if go:
        if go.get("fold_fallback"):
            print(f"  Fold: FALLBACK to pure affine (fold_frac={go.get('fold_frac', '?')})")
        elif go.get("fold_frac") is not None:
            print(f"  Fold: {go['fold_frac']*100:.2f}% (ok)")
        else:
            print(f"  Fold: clean")
    # Reclamation
    recl = latest.get("reclamation", {})
    if recl:
        print(f"  Reclamation: {recl.get('cleaned_pct', '?')}% "
              f"(raw {recl.get('raw_pct', '?')}%, {recl.get('n_large', '?')} large blobs)")
    # ICP
    icp = latest.get("icp", {})
    if icp:
        if icp.get("applied"):
            print(f"  ICP: applied (dx={icp.get('dx_m', '?')}m, dy={icp.get('dy_m', '?')}m)")
        elif icp.get("applied") is False:
            print(f"  ICP: skipped ({icp.get('reason', 'unknown')})")

    # Best-ever
    print(f"\n=== Best-Ever Comparison ===")
    best_fields = [
        ("score", "score", True),
        ("total", "total", True),
        ("stable_iou", "stable_iou", False),
        ("shore_iou", "shore_iou", False),
        ("patch_med", "patch_med", True),
        ("patch_p90", "patch_p90", True),
        ("east", "east", True),
    ]
    lg = latest.get("grid", {})
    for label, key, lower_better in best_fields:
        best_ver, best_val = find_best(rows, "grid", key, lower_better)
        current = lg.get(key)
        if best_ver is None or current is None:
            continue
        is_new_best = (
            (lower_better and current <= best_val) or
            (not lower_better and current >= best_val)
        )
        if isinstance(best_val, float) and abs(best_val) < 10:
            best_str = f"v{best_ver} ({best_val:.3f})"
            cur_str = f"{current:.3f}"
        else:
            best_str = f"v{best_ver} ({best_val:.0f})"
            cur_str = f"{current:.0f}"
        tag = " (NEW BEST)" if is_new_best and latest["version"] != best_ver else ""
        if is_new_best and latest["version"] == best_ver:
            tag = " (NEW BEST)"
        print(f"  Best {label:>10}: {best_str} -- current: {cur_str}{tag}")


def format_json(rows, baseline_version=None):
    """Format comparison as JSON."""
    if not rows:
        print(json.dumps({"versions": [], "latest": None}))
        return

    latest = rows[-1]
    if baseline_version is not None:
        baseline = next((r for r in rows if r["version"] == baseline_version), None)
    else:
        baseline = rows[-2] if len(rows) >= 2 else None

    # Compute deltas
    deltas = {}
    if baseline and latest["version"] != baseline["version"]:
        lg = latest.get("grid", {})
        bg = baseline.get("grid", {})
        for key, lower_better in [("score", True), ("total", True), ("cv_mean_m", True),
                                   ("west", True), ("center", True), ("east", True),
                                   ("patch_med", True), ("patch_p90", True),
                                   ("stable_iou", False), ("shore_iou", False)]:
            d = compute_delta(lg.get(key), bg.get(key), key, lower_better)
            if d:
                deltas[key] = d

        # Non-grid deltas
        lwc = latest.get("wall_clock_s")
        bwc = baseline.get("wall_clock_s")
        if lwc is not None and bwc is not None:
            deltas["wall_clock_s"] = compute_delta(lwc, bwc, "wall_clock_s", True)

        lf = latest.get("flow", {}) or {}
        bf = baseline.get("flow", {}) or {}
        if lf.get("reliable_pct") is not None and bf.get("reliable_pct") is not None:
            deltas["flow_reliable_pct"] = compute_delta(lf["reliable_pct"], bf["reliable_pct"],
                                                         "flow_reliable_pct", False)

    # Best-ever
    best_ever = {}
    lg = latest.get("grid", {})
    for key, lower_better in [("score", True), ("total", True), ("stable_iou", False),
                               ("shore_iou", False), ("patch_med", True), ("patch_p90", True),
                               ("east", True)]:
        best_ver, best_val = find_best(rows, "grid", key, lower_better)
        current = lg.get(key)
        if best_ver is not None and current is not None:
            is_new_best = (
                (lower_better and current <= best_val) or
                (not lower_better and current >= best_val)
            )
            best_ever[key] = {
                "best_version": best_ver,
                "best_value": best_val,
                "current": current,
                "is_new_best": is_new_best,
            }

    # Process health
    process_health = {}
    if latest.get("anchors"):
        process_health["anchors"] = latest["anchors"]
    if latest.get("flow"):
        process_health["flow"] = latest["flow"]
    if latest.get("grid_optimizer"):
        process_health["grid_optimizer"] = latest["grid_optimizer"]
    if latest.get("reclamation"):
        process_health["reclamation"] = latest["reclamation"]
    if latest.get("icp"):
        process_health["icp"] = latest["icp"]
    if latest.get("wall_clock_s") is not None:
        process_health["wall_clock_s"] = latest["wall_clock_s"]

    # Score breakdown
    score_breakdown = lg.get("score_breakdown") if lg else None

    output = {
        "versions": rows,
        "latest": latest,
        "baseline_version": baseline["version"] if baseline else None,
        "deltas": deltas,
        "best_ever": best_ever,
        "process_health": process_health,
        "score_breakdown": score_breakdown,
    }

    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Compare alignment pipeline versions")
    parser.add_argument("--format", "-f", choices=["text", "json"], default="text",
                        help="Output format (default: text)")
    parser.add_argument("--baseline", "-b", type=int, default=None,
                        help="Baseline version for delta comparison (default: previous)")
    args = parser.parse_args()

    rows = load_all_versions()

    if args.format == "json":
        format_json(rows, args.baseline)
    else:
        format_text(rows, args.baseline)


if __name__ == "__main__":
    main()
