"""Diagnose per-segment ortho seam quality.

Reads the {scene_id}_seg{i:02d}_ortho.tif outputs from
``preprocess.camera_model.opticalbar_per_segment_precorrect`` (and the
sidecar corners JSONs) and reports:

- Per-adjacent-pair ZNCC in the geographic overlap
- Per-adjacent-pair pixel shift estimated by phase correlation (how far
  the same feature ends up apart between the two segments — this is the
  quantity that determines visible "doubling" in the blended mosaic)
- Per-segment base vs applied corner deltas (Pass 2 shift magnitudes)

Used to confirm Phase 0's root cause hypothesis (cam_gen disagreement) and
as an acceptance gate for Phase 1 fixes: after Phase 1 lands, per-adjacent
ZNCC should be > 0.4 and phase-correlation pixel shift should be < 1 px.

Usage:
    python3 scripts/debug/debug_per_segment.py <segments_dir> [scene_id]

If scene_id is omitted the script discovers it from the first matching
file. Writes a JSON report next to the segments.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Any


def _find_scene_id(segments_dir: str) -> str | None:
    pattern = os.path.join(segments_dir, "*_seg00_ortho.tif")
    matches = glob.glob(pattern)
    if not matches:
        return None
    base = os.path.basename(matches[0])
    return base.replace("_seg00_ortho.tif", "")


def _list_segments(segments_dir: str, scene_id: str) -> list[str]:
    pattern = os.path.join(segments_dir, f"{scene_id}_seg*_ortho.tif")
    paths = sorted(glob.glob(pattern))
    # Exclude Phase-3 warped, stage-1, and seam-temp variants so the
    # canonical per-segment seam report is measured on pre-warp orthos.
    return [p for p in paths if not any(
        p.endswith(suffix)
        for suffix in ("_ortho_warped.tif", "_ortho_stage1.tif",
                       "_ortho_seam.tif", "_ortho_iter0.tif",
                       "_ortho_iter1.tif", "_ortho_iter2.tif")
    )]


def _list_warped_segments(segments_dir: str, scene_id: str) -> list[str]:
    """Phase-3 warped variants, if present."""
    pattern = os.path.join(segments_dir, f"{scene_id}_seg*_ortho_warped.tif")
    return sorted(glob.glob(pattern))


def _sidecar_delta(ortho_path: str) -> dict[str, Any] | None:
    sidecar = os.path.splitext(ortho_path)[0] + "_corners.json"
    if not os.path.isfile(sidecar):
        return None
    try:
        with open(sidecar) as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or "base" not in raw or "applied" not in raw:
        return None
    base = raw["base"]
    applied = raw["applied"]
    try:
        deltas_m: dict[str, float] = {}
        for key in ("NW", "NE", "SE", "SW"):
            b_lat, b_lon = base[key]
            a_lat, a_lon = applied[key]
            lat_rad = math.radians((b_lat + a_lat) / 2)
            dlat_m = (a_lat - b_lat) * 111320.0
            dlon_m = (a_lon - b_lon) * 111320.0 * math.cos(lat_rad)
            deltas_m[key] = math.hypot(dlat_m, dlon_m)
        return {
            "base": base,
            "applied": applied,
            "max_delta_m": max(deltas_m.values()),
            "mean_delta_m": sum(deltas_m.values()) / 4,
            "per_corner_m": deltas_m,
        }
    except (KeyError, TypeError, ValueError):
        return None


def _measure_seam(path_a: str, path_b: str) -> dict[str, Any]:
    """ZNCC + phase correlation on the geographic overlap between two orthos."""
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds

    result: dict[str, Any] = {
        "a": os.path.basename(path_a),
        "b": os.path.basename(path_b),
    }

    with rasterio.open(path_a) as sa, rasterio.open(path_b) as sb:
        ol = max(sa.bounds.left, sb.bounds.left)
        ob = max(sa.bounds.bottom, sb.bounds.bottom)
        or_ = min(sa.bounds.right, sb.bounds.right)
        ot = min(sa.bounds.top, sb.bounds.top)

        if ol >= or_ or ob >= ot:
            result["status"] = "no_overlap"
            result["gap_w_m"] = ol - or_
            result["gap_h_m"] = ob - ot
            return result

        result["overlap_bounds"] = [ol, ob, or_, ot]
        result["overlap_w_m"] = or_ - ol
        result["overlap_h_m"] = ot - ob

        wa = from_bounds(ol, ob, or_, ot, sa.transform)
        wb = from_bounds(ol, ob, or_, ot, sb.transform)

        pa = sa.read(1, window=wa).astype(np.float64)
        pb = sb.read(1, window=wb).astype(np.float64)

        nodata_a = sa.nodata if sa.nodata is not None else -32768
        nodata_b = sb.nodata if sb.nodata is not None else -32768

    h = min(pa.shape[0], pb.shape[0])
    w = min(pa.shape[1], pb.shape[1])
    if h < 16 or w < 16:
        result["status"] = "overlap_too_small"
        result["h"] = h
        result["w"] = w
        return result
    pa, pb = pa[:h, :w], pb[:h, :w]

    valid = (pa != nodata_a) & (pb != nodata_b)
    result["valid_px"] = int(valid.sum())
    if valid.sum() < 100:
        result["status"] = "no_valid_overlap"
        return result

    a_v = pa[valid] - pa[valid].mean()
    b_v = pb[valid] - pb[valid].mean()
    denom = math.sqrt(float((a_v ** 2).sum()) * float((b_v ** 2).sum()))
    ncc = float((a_v * b_v).sum() / denom) if denom > 0 else 0.0
    result["zncc"] = ncc

    try:
        import cv2
        pa_u8 = np.zeros_like(pa, dtype=np.uint8)
        pb_u8 = np.zeros_like(pb, dtype=np.uint8)
        lo = min(pa[valid].min(), pb[valid].min())
        hi = max(pa[valid].max(), pb[valid].max())
        if hi - lo > 1:
            scale = 255.0 / (hi - lo)
            pa_u8[valid] = np.clip((pa[valid] - lo) * scale, 0, 255).astype(np.uint8)
            pb_u8[valid] = np.clip((pb[valid] - lo) * scale, 0, 255).astype(np.uint8)

        pa_f = pa_u8.astype(np.float32)
        pb_f = pb_u8.astype(np.float32)
        window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        (dx, dy), phase_response = cv2.phaseCorrelate(pa_f, pb_f, window)
        result["phase_corr"] = {
            "dx_px": float(dx),
            "dy_px": float(dy),
            "shift_px": math.hypot(dx, dy),
            "response": float(phase_response),
        }
    except Exception as e:
        result["phase_corr_error"] = str(e)

    result["status"] = "ok"
    return result


def run(segments_dir: str, scene_id: str | None) -> dict[str, Any]:
    if scene_id is None:
        scene_id = _find_scene_id(segments_dir)
    if scene_id is None:
        return {"error": f"no _seg00_ortho.tif found in {segments_dir}"}

    segs = _list_segments(segments_dir, scene_id)
    if not segs:
        return {"error": f"no segment orthos for scene_id={scene_id!r}"}

    report: dict[str, Any] = {
        "scene_id": scene_id,
        "segments_dir": segments_dir,
        "n_segments": len(segs),
        "segment_paths": [os.path.basename(p) for p in segs],
        "per_segment_sidecar": [],
        "seams": [],
        "summary": {},
    }

    for p in segs:
        sc = _sidecar_delta(p)
        report["per_segment_sidecar"].append({
            "seg": os.path.basename(p),
            "sidecar": sc,
        })

    for i in range(len(segs) - 1):
        seam = _measure_seam(segs[i], segs[i + 1])
        seam["index"] = f"{i}-{i+1}"
        report["seams"].append(seam)

    # Phase 3: if warped variants exist, measure their seams too so we can
    # compare pre-warp vs post-warp phase-shift and ZNCC.
    warped_segs = _list_warped_segments(segments_dir, scene_id)
    if len(warped_segs) >= 2:
        report["warped_segments"] = [os.path.basename(p) for p in warped_segs]
        report["warped_seams"] = []
        for i in range(len(warped_segs) - 1):
            seam = _measure_seam(warped_segs[i], warped_segs[i + 1])
            seam["index"] = f"{i}-{i+1}"
            report["warped_seams"].append(seam)
        warped_oks = [
            s for s in report["warped_seams"] if s.get("status") == "ok"
        ]
        if warped_oks:
            wz = [s["zncc"] for s in warped_oks]
            ws = [s["phase_corr"]["shift_px"] for s in warped_oks
                  if "phase_corr" in s]
            report["summary"]["warped_zncc_min"] = min(wz)
            report["summary"]["warped_zncc_mean"] = sum(wz) / len(wz)
            if ws:
                report["summary"]["warped_phase_shift_px_max"] = max(ws)
                report["summary"]["warped_phase_shift_px_mean"] = sum(ws) / len(ws)

    oks = [s for s in report["seams"] if s.get("status") == "ok"]
    if oks:
        zncc_values = [s["zncc"] for s in oks]
        report["summary"]["zncc_min"] = min(zncc_values)
        report["summary"]["zncc_mean"] = sum(zncc_values) / len(zncc_values)
        report["summary"]["zncc_max"] = max(zncc_values)
        shifts = [
            s["phase_corr"]["shift_px"]
            for s in oks
            if "phase_corr" in s
        ]
        if shifts:
            report["summary"]["phase_shift_px_max"] = max(shifts)
            report["summary"]["phase_shift_px_mean"] = sum(shifts) / len(shifts)
        report["summary"]["passing_seams_zncc_gt_0_4"] = sum(1 for z in zncc_values if z > 0.4)
        report["summary"]["total_seams"] = len(report["seams"])
    else:
        report["summary"]["note"] = "no seams could be measured (all gaps / too small)"

    deltas = [
        e["sidecar"]["max_delta_m"]
        for e in report["per_segment_sidecar"]
        if e.get("sidecar") and "max_delta_m" in e["sidecar"]
    ]
    if deltas:
        report["summary"]["pass2_max_corner_shift_m"] = max(deltas)
        report["summary"]["pass2_mean_corner_shift_m"] = sum(deltas) / len(deltas)

    return report


def _print_report(report: dict[str, Any]) -> None:
    if "error" in report:
        print(f"ERROR: {report['error']}")
        return

    print(f"\n=== per-segment diagnostic: {report['scene_id']} ===")
    print(f"segments_dir: {report['segments_dir']}")
    print(f"n_segments:   {report['n_segments']}")

    summary = report.get("summary", {})
    if "zncc_min" in summary:
        print(f"\nZNCC: min={summary['zncc_min']:.3f} "
              f"mean={summary['zncc_mean']:.3f} max={summary['zncc_max']:.3f}")
        print(f"      {summary['passing_seams_zncc_gt_0_4']}/"
              f"{summary['total_seams']} seams pass ZNCC > 0.4")
    if "phase_shift_px_max" in summary:
        print(f"Phase shift: mean={summary['phase_shift_px_mean']:.2f} px  "
              f"max={summary['phase_shift_px_max']:.2f} px")
    if "pass2_max_corner_shift_m" in summary:
        print(f"Pass 2 corner shifts: mean={summary['pass2_mean_corner_shift_m']:.0f}m  "
              f"max={summary['pass2_max_corner_shift_m']:.0f}m")
    if "warped_zncc_min" in summary:
        print(
            f"\nPhase-3 warped seam report:"
        )
        print(
            f"  ZNCC: min={summary['warped_zncc_min']:.3f} "
            f"mean={summary['warped_zncc_mean']:.3f}"
        )
        if "warped_phase_shift_px_max" in summary:
            print(
                f"  Phase shift: mean={summary['warped_phase_shift_px_mean']:.2f} px  "
                f"max={summary['warped_phase_shift_px_max']:.2f} px"
            )
        pre = summary.get("phase_shift_px_max")
        post = summary.get("warped_phase_shift_px_max")
        if pre is not None and post is not None:
            delta = pre - post
            print(f"  Pre → post max shift: {pre:.2f}px → {post:.2f}px (Δ={delta:+.2f}px)")
    if "note" in summary:
        print(f"note: {summary['note']}")

    print("\nPer-seam details:")
    for s in report["seams"]:
        idx = s["index"]
        status = s.get("status", "unknown")
        if status == "ok":
            zncc = s["zncc"]
            shift = s.get("phase_corr", {}).get("shift_px", float("nan"))
            valid = s["valid_px"]
            print(f"  seam {idx}: zncc={zncc:+.3f}  shift={shift:5.2f}px  valid={valid:,}px")
        else:
            print(f"  seam {idx}: {status}")

    print("\nPer-segment Pass 2 corner deltas (base vs applied):")
    for e in report["per_segment_sidecar"]:
        sc = e.get("sidecar")
        if sc is None:
            print(f"  {e['seg']}: no sidecar")
        else:
            print(f"  {e['seg']}: max={sc['max_delta_m']:.0f}m  mean={sc['mean_delta_m']:.0f}m")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("segments_dir", help="Directory containing {scene_id}_seg*_ortho.tif files")
    ap.add_argument("scene_id", nargs="?", help="Explicit scene ID (auto-discovered if omitted)")
    ap.add_argument("--json", help="Path to write JSON report (default: <segments_dir>/per_segment_diag.json)")
    args = ap.parse_args()

    if not os.path.isdir(args.segments_dir):
        print(f"ERROR: {args.segments_dir!r} is not a directory", file=sys.stderr)
        return 1

    report = run(args.segments_dir, args.scene_id)
    _print_report(report)

    json_path = args.json or os.path.join(args.segments_dir, "per_segment_diag.json")
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReport written to {json_path}")
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    sys.exit(main())
