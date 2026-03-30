"""Independent QA orchestration and report persistence."""

from __future__ import annotations

import json
from typing import Iterable

import numpy as np

from .qa import evaluate_alignment_quality_paths
from .types import MatchPair, QaReport, match_pairs_from_legacy


def split_holdout_pairs(matched_pairs: Iterable[MatchPair | tuple], holdout_fraction: float = 0.2,
                        min_holdout: int = 4, seed: int = 42):
    """Split match pairs into training and holdout sets."""

    pairs = match_pairs_from_legacy(matched_pairs)
    auto_idx = [idx for idx, pair in enumerate(pairs) if not pair.is_anchor]
    if len(auto_idx) < max(min_holdout + 6, 12):
        return pairs, []

    holdout_n = max(min_holdout, int(round(len(auto_idx) * holdout_fraction)))
    holdout_n = min(holdout_n, max(0, len(auto_idx) - 6))
    if holdout_n <= 0:
        return pairs, []

    rng = np.random.default_rng(seed)
    shuffled = np.array(auto_idx, dtype=np.int32)
    rng.shuffle(shuffled)
    holdout_idx = set(int(idx) for idx in shuffled[:holdout_n])
    train = [pair for idx, pair in enumerate(pairs) if idx not in holdout_idx]
    holdout = [pair for idx, pair in enumerate(pairs) if idx in holdout_idx]
    return train, holdout


def compute_holdout_affine_metrics(M_geo, holdout_pairs: Iterable[MatchPair | tuple]):
    """Return independent residual stats for held-out matches."""

    if M_geo is None:
        return {}

    holdout = match_pairs_from_legacy(holdout_pairs)
    if not holdout:
        return {}

    errors = []
    for pair in holdout:
        pred_x = M_geo[0, 0] * pair.off_x + M_geo[0, 1] * pair.off_y + M_geo[0, 2]
        pred_y = M_geo[1, 0] * pair.off_x + M_geo[1, 1] * pair.off_y + M_geo[1, 2]
        errors.append(float(np.hypot(pred_x - pair.ref_x, pred_y - pair.ref_y)))

    err_arr = np.array(errors, dtype=np.float32)
    return {
        "count": int(len(errors)),
        "mean_m": float(np.mean(err_arr)),
        "median_m": float(np.median(err_arr)),
        "p90_m": float(np.percentile(err_arr, 90)),
        "max_m": float(np.max(err_arr)),
    }


def compute_holdout_warp_metrics(output_path: str, reference_path: str,
                                 holdout_pairs: Iterable[MatchPair | tuple],
                                 work_crs, overlap, eval_res: float = 4.0):
    """Validate holdout pairs against the final warped output (not just affine).

    For each holdout pair, read the local region around the reference point
    in both the reference and the warped output, then use phase correlation
    to measure the actual residual displacement in the final product.
    """
    import rasterio
    from .geo import read_overlap_region
    from .qa import _batch_phase_correlate
    import torch
    from .models import get_torch_device

    holdout = match_pairs_from_legacy(holdout_pairs)
    if not holdout or len(holdout) < 4:
        return {}

    try:
        src_ref = rasterio.open(reference_path)
        src_out = rasterio.open(output_path)
        arr_ref, ref_transform = read_overlap_region(src_ref, overlap, work_crs, eval_res)
        arr_out, out_transform = read_overlap_region(src_out, overlap, work_crs, eval_res)
        src_ref.close()
        src_out.close()
    except Exception:
        return {}

    valid = (arr_ref > 0) & (arr_out > 0)
    h, w = arr_ref.shape
    patch = 64
    half = patch // 2
    device = get_torch_device()

    ref_patches = []
    out_patches = []
    win = np.outer(np.hanning(patch), np.hanning(patch)).astype(np.float32)

    for pair in holdout:
        # Convert reference coordinates to pixel position
        col, row = ~ref_transform * (pair.ref_x, pair.ref_y)
        row, col = int(round(row)), int(round(col))
        r0, r1 = row - half, row + half
        c0, c1 = col - half, col + half
        if r0 < 0 or r1 > h or c0 < 0 or c1 > w:
            continue
        if np.mean(valid[r0:r1, c0:c1]) < 0.5:
            continue
        ref_patches.append(arr_ref[r0:r1, c0:c1].astype(np.float32) * win)
        out_patches.append(arr_out[r0:r1, c0:c1].astype(np.float32) * win)

    if len(ref_patches) < 4:
        return {}

    ref_t = torch.from_numpy(np.stack(ref_patches)).to(device)
    out_t = torch.from_numpy(np.stack(out_patches)).to(device)

    with torch.no_grad():
        shifts, responses, vmask = _batch_phase_correlate(ref_t, out_t, 0.02)

    mags = np.hypot(shifts[vmask, 0], shifts[vmask, 1]) * eval_res

    if len(mags) < 2:
        return {}

    return {
        "count": int(len(mags)),
        "mean_m": float(np.mean(mags)),
        "median_m": float(np.median(mags)),
        "p90_m": float(np.percentile(mags, 90)),
        "max_m": float(np.max(mags)),
        "method": "post_warp_phase_correlation",
    }


def _confidence_from_metrics(image_metrics, holdout_metrics, coverage, cv_mean_m):
    components = []

    image_score = float(image_metrics.get("score", 300.0)) if image_metrics else 300.0
    components.append(np.clip(1.0 - (image_score / 200.0), 0.0, 1.0))

    if holdout_metrics:
        components.append(np.clip(1.0 - (float(holdout_metrics.get("mean_m", 200.0)) / 120.0), 0.0, 1.0))
        components.append(np.clip(1.0 - (float(holdout_metrics.get("p90_m", 240.0)) / 180.0), 0.0, 1.0))

    if cv_mean_m is not None:
        components.append(np.clip(1.0 - (float(cv_mean_m) / 120.0), 0.0, 1.0))

    components.append(np.clip(float(coverage) / 0.35, 0.0, 1.0))
    return float(np.mean(components)) if components else 0.0


def _total_score(image_metrics, holdout_metrics, coverage, cv_mean_m):
    score = float(image_metrics.get("score", 500.0)) if image_metrics else 500.0
    if holdout_metrics:
        score += 0.60 * float(holdout_metrics.get("mean_m", 0.0))
        score += 0.25 * float(holdout_metrics.get("p90_m", 0.0))
    if cv_mean_m is not None:
        score += 0.35 * float(cv_mean_m)
    if coverage < 0.12:
        score += (0.12 - coverage) * 400.0
    return float(score)


def _compute_quality_grade(image_score, holdout_median, cv_mean, qa_params=None):
    """Compute quality grade A/B/C/D from metrics and thresholds."""
    from .params import get_params
    qp = qa_params or get_params().qa

    score = image_score or 999.0
    holdout = holdout_median or 999.0
    cv = cv_mean or 999.0

    if score <= qp.grade_a_score and holdout <= qp.grade_a_holdout and cv <= qp.grade_a_holdout:
        return "A"
    if score <= qp.grade_b_score and holdout <= qp.grade_b_holdout and cv <= qp.grade_b_holdout:
        return "B"
    if score <= qp.accept_image_score_max and holdout <= qp.accept_holdout_median_max and cv <= qp.accept_cv_mean_max:
        return "C"
    return "D"


def build_candidate_report(candidate_name: str, output_path: str, reference_path: str,
                           overlap, work_crs, *, holdout_pairs=None, M_geo=None,
                           coverage: float = 0.0, cv_mean_m: float | None = None,
                           hypothesis_id: str = "", eval_res: float = 4.0,
                           image_metrics=None, qa_params=None) -> QaReport:
    """Compute an independent QA report for a candidate output.

    Args:
        qa_params: Optional QaParams from profile. If None, uses active profile.
    """
    from .params import get_params
    qp = qa_params or get_params().qa

    if image_metrics is None:
        image_metrics = evaluate_alignment_quality_paths(
            output_path,
            reference_path,
            overlap,
            work_crs,
            eval_res=eval_res,
        ) or {}
    holdout_metrics = compute_holdout_affine_metrics(M_geo, holdout_pairs or [])
    # Post-warp holdout: measure displacement in the final warped output
    holdout_warp_metrics = compute_holdout_warp_metrics(
        output_path, reference_path, holdout_pairs or [],
        work_crs, overlap, eval_res=eval_res,
    )
    total = _total_score(image_metrics, holdout_metrics, coverage, cv_mean_m)
    confidence = _confidence_from_metrics(image_metrics, holdout_metrics, coverage, cv_mean_m)

    image_score = float(image_metrics.get("score", 0.0)) if image_metrics else 0.0
    holdout_median = float(holdout_metrics.get("median_m", 0.0)) if holdout_metrics else 0.0

    # Acceptance with per-profile thresholds
    reasons = []
    accepted = True
    if image_metrics and image_score > qp.accept_image_score_max:
        accepted = False
        reasons.append("image_alignment_score_high")
    if holdout_metrics and holdout_median > qp.accept_holdout_median_max:
        accepted = False
        reasons.append("holdout_residual_high")
    if cv_mean_m is not None and float(cv_mean_m) > qp.accept_cv_mean_max:
        accepted = False
        reasons.append("cross_validation_high")
    if coverage < qp.accept_coverage_min:
        accepted = False
        reasons.append("gcp_coverage_low")

    grade = _compute_quality_grade(image_score, holdout_median, cv_mean_m, qp)

    # Attach post-warp holdout to holdout_metrics for persistence
    if holdout_warp_metrics:
        holdout_metrics["post_warp"] = holdout_warp_metrics

    return QaReport(
        candidate=candidate_name,
        output_path=output_path,
        total_score=total,
        confidence=confidence,
        accepted=accepted,
        image_metrics=image_metrics,
        holdout_metrics=holdout_metrics,
        coverage=float(coverage),
        cv_mean_m=None if cv_mean_m is None else float(cv_mean_m),
        hypothesis_id=hypothesis_id,
        reasons=reasons,
        quality_grade=grade,
    )


def write_qa_report(path: str, reports: list[QaReport], *, selected_candidate: str | None = None,
                    metadata: dict | None = None):
    """Persist QA reports to JSON."""

    payload = {
        "selected_candidate": selected_candidate,
        "metadata": metadata or {},
        "reports": [report.to_dict() for report in reports],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
