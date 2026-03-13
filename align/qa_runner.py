"""Independent QA orchestration and report persistence."""

from __future__ import annotations

import json
from typing import Iterable

import numpy as np

from .qa import evaluate_alignment_quality_paths
from .types import MatchPair, QaReport, match_pairs_from_legacy, match_pairs_to_legacy


def split_holdout_pairs(matched_pairs: Iterable[MatchPair | tuple], holdout_fraction: float = 0.2,
                        min_holdout: int = 4, seed: int = 42):
    """Split match pairs into training and holdout sets."""

    pairs = match_pairs_from_legacy(matched_pairs)
    auto_idx = [idx for idx, pair in enumerate(pairs) if not pair.is_anchor]
    if len(auto_idx) < max(min_holdout + 6, 12):
        return match_pairs_to_legacy(pairs), []

    holdout_n = max(min_holdout, int(round(len(auto_idx) * holdout_fraction)))
    holdout_n = min(holdout_n, max(0, len(auto_idx) - 6))
    if holdout_n <= 0:
        return match_pairs_to_legacy(pairs), []

    rng = np.random.default_rng(seed)
    shuffled = np.array(auto_idx, dtype=np.int32)
    rng.shuffle(shuffled)
    holdout_idx = set(int(idx) for idx in shuffled[:holdout_n])
    train = [pair for idx, pair in enumerate(pairs) if idx not in holdout_idx]
    holdout = [pair for idx, pair in enumerate(pairs) if idx in holdout_idx]
    return match_pairs_to_legacy(train), match_pairs_to_legacy(holdout)


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


def build_candidate_report(candidate_name: str, output_path: str, reference_path: str,
                           overlap, work_crs, *, holdout_pairs=None, M_geo=None,
                           coverage: float = 0.0, cv_mean_m: float | None = None,
                           hypothesis_id: str = "", eval_res: float = 4.0) -> QaReport:
    """Compute an independent QA report for a candidate output."""

    image_metrics = evaluate_alignment_quality_paths(
        output_path,
        reference_path,
        overlap,
        work_crs,
        eval_res=eval_res,
    ) or {}
    holdout_metrics = compute_holdout_affine_metrics(M_geo, holdout_pairs or [])
    total = _total_score(image_metrics, holdout_metrics, coverage, cv_mean_m)
    confidence = _confidence_from_metrics(image_metrics, holdout_metrics, coverage, cv_mean_m)

    reasons = []
    accepted = True
    if image_metrics and float(image_metrics.get("score", 0.0)) > 140.0:
        accepted = False
        reasons.append("image_alignment_score_high")
    if holdout_metrics and float(holdout_metrics.get("median_m", 0.0)) > 90.0:
        accepted = False
        reasons.append("holdout_residual_high")
    if cv_mean_m is not None and float(cv_mean_m) > 90.0:
        accepted = False
        reasons.append("cross_validation_high")
    if coverage < 0.10:
        accepted = False
        reasons.append("gcp_coverage_low")

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
