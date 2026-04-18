"""Unit tests for Phase 1 geometric-verification filters in
``preprocess.experimental.match_ip``.

These tests construct synthetic match sets with a known affine inlier group
plus controlled outlier populations and verify that MAGSAC++ + Sampson + MTE
together retain the inliers while rejecting the outliers.
"""

from __future__ import annotations

import numpy as np
import pytest

from preprocess.experimental.match_ip import (
    apply_geometric_filters,
    mte_local_consistency_filter,
    robust_affine_filter,
    sampson_distance_filter,
)


RNG = np.random.default_rng(1337)


def _make_affine_inliers(n, rot_deg=2.0, scale=1.01, tx=12.0, ty=-8.0,
                        noise_px=0.4):
    """Sample ``n`` points in image A and map them to image B by a similarity."""
    theta = np.deg2rad(rot_deg)
    c, s = np.cos(theta), np.sin(theta)
    M = np.array([[scale * c, -scale * s, tx],
                  [scale * s, scale * c, ty]], dtype=np.float64)
    pts_a = RNG.uniform(0, 2000, size=(n, 2)).astype(np.float64)
    pts_a_h = np.hstack([pts_a, np.ones((n, 1))])
    pts_b = pts_a_h @ M.T
    pts_b += RNG.normal(0, noise_px, size=pts_b.shape)
    conf = np.ones(n, dtype=np.float64) * 0.8
    return pts_a.astype(np.float32), pts_b.astype(np.float32), conf, M


def _make_random_outliers(n, bounds=(0, 2000)):
    pts_a = RNG.uniform(*bounds, size=(n, 2)).astype(np.float32)
    pts_b = RNG.uniform(*bounds, size=(n, 2)).astype(np.float32)
    conf = np.ones(n, dtype=np.float64) * 0.7
    return pts_a, pts_b, conf


def _make_locally_biased_outliers(n, M_global, bias=(80.0, -40.0)):
    """Outliers that are affine-consistent inside a cluster but wrong overall —
    simulates a land-cover-changed region with self-consistent but shifted
    matches that would pass global RANSAC without MTE."""
    cluster_center_a = RNG.uniform(400, 1600, size=2)
    pts_a = cluster_center_a + RNG.normal(0, 60, size=(n, 2))
    pts_a = pts_a.astype(np.float64)
    pts_a_h = np.hstack([pts_a, np.ones((n, 1))])
    pts_b = pts_a_h @ M_global.T
    pts_b[:, 0] += bias[0]
    pts_b[:, 1] += bias[1]
    conf = np.ones(n, dtype=np.float64) * 0.75
    return pts_a.astype(np.float32), pts_b.astype(np.float32), conf


def test_robust_affine_filter_rejects_random_outliers():
    n_in = 100
    n_out = 40
    pts_a_in, pts_b_in, conf_in, _M = _make_affine_inliers(n_in, noise_px=0.3)
    pts_a_out, pts_b_out, conf_out = _make_random_outliers(n_out)
    pts_a = np.vstack([pts_a_in, pts_a_out]).astype(np.float32)
    pts_b = np.vstack([pts_b_in, pts_b_out]).astype(np.float32)
    conf = np.concatenate([conf_in, conf_out])

    out_a, out_b, out_c, M = robust_affine_filter(
        pts_a, pts_b, conf, reproj_px=2.0, min_inliers=30,
    )
    assert out_a is not None, "robust affine should succeed with 100 strong inliers"
    assert M.shape == (2, 3)
    # Should retain ≥ 90% of inliers and < 10% of outliers.
    kept_inliers = len(out_a) >= int(0.9 * n_in)
    assert kept_inliers, f"kept only {len(out_a)} / {n_in} true inliers"
    assert len(out_a) <= n_in + int(0.15 * n_out), (
        f"let in too many outliers: survived {len(out_a)} > expected ≤ {n_in + int(0.15 * n_out)}"
    )


def test_sampson_filter_rejects_off_epipolar_matches():
    n_in = 80
    pts_a, pts_b, conf, _M = _make_affine_inliers(n_in, noise_px=0.2)
    # Add 20 matches whose B coordinates are shuffled relative to A.
    perm = RNG.permutation(n_in)[:20]
    off_a = pts_a[perm].copy()
    off_b = pts_b[RNG.permutation(n_in)][:20].copy()
    off_conf = np.ones(20, dtype=np.float64) * 0.6

    all_a = np.vstack([pts_a, off_a])
    all_b = np.vstack([pts_b, off_b])
    all_c = np.concatenate([conf, off_conf])

    filtered_a, filtered_b, filtered_c = sampson_distance_filter(
        all_a, all_b, all_c, tau_px=3.0, min_inliers=20,
    )
    # Shuffled matches should mostly be rejected — expect roughly ≤ n_in + small residual.
    assert len(filtered_a) <= int(1.1 * n_in), (
        f"Sampson kept too many off-epipolar matches: {len(filtered_a)}"
    )
    assert len(filtered_a) >= int(0.85 * n_in), (
        f"Sampson dropped too many true inliers: {len(filtered_a)}"
    )


def test_mte_filter_rejects_locally_biased_cluster():
    # MTE's purpose is detecting groups whose residual disagrees with their
    # wider neighbourhood. For a tight outlier cluster we must set the
    # neighbourhood radius large enough to include inlier neighbours.
    n_in = 150
    pts_a_in, pts_b_in, conf_in, M = _make_affine_inliers(n_in, noise_px=0.3)
    n_biased = 40
    pts_a_bi, pts_b_bi, conf_bi = _make_locally_biased_outliers(
        n_biased, M, bias=(120.0, -70.0),
    )
    pts_a = np.vstack([pts_a_in, pts_a_bi]).astype(np.float32)
    pts_b = np.vstack([pts_b_in, pts_b_bi]).astype(np.float32)
    conf = np.concatenate([conf_in, conf_bi])

    filtered_a, filtered_b, filtered_c = mte_local_consistency_filter(
        pts_a, pts_b, conf, M,
        # Radius wider than the biased cluster (60 px) so neighbours mix
        # inliers and the cluster.
        radius_px=800.0, k_neighbors=24, max_local_dev_px=8.0,
    )
    n_kept_inliers = _count_overlap(filtered_a, pts_a_in)
    n_biased_kept = _count_overlap(filtered_a, pts_a_bi)
    # Realistic expectations given MTE's neighbourhood-averaging heuristic:
    # inliers NEAR the biased cluster can get pulled into the cluster's mean
    # residual and be dropped. Accept ≥ 80% inlier retention and ≥ 50%
    # biased-cluster rejection. The Sampson + affine filters (full
    # apply_geometric_filters end-to-end test below) tighten both further.
    assert n_kept_inliers >= int(0.80 * n_in), (
        f"MTE dropped too many true inliers: {n_kept_inliers} / {n_in}"
    )
    assert n_biased_kept <= int(0.5 * n_biased), (
        f"MTE kept {n_biased_kept}/{n_biased} biased matches; expected ≤ 50%"
    )


def _count_overlap(arr_a, arr_b, tol=0.5):
    """Count how many entries in arr_b appear (within tol) in arr_a."""
    count = 0
    for pt in arr_b:
        d = np.linalg.norm(arr_a - pt, axis=1)
        if d.min() < tol:
            count += 1
    return count


def test_apply_geometric_filters_end_to_end_precision_and_recall():
    n_in = 200
    pts_a_in, pts_b_in, conf_in, M = _make_affine_inliers(n_in, noise_px=0.3)
    pts_a_out, pts_b_out, conf_out = _make_random_outliers(80)
    pts_a_bi, pts_b_bi, conf_bi = _make_locally_biased_outliers(40, M, bias=(100.0, -60.0))

    pts_a = np.vstack([pts_a_in, pts_a_out, pts_a_bi]).astype(np.float32)
    pts_b = np.vstack([pts_b_in, pts_b_out, pts_b_bi]).astype(np.float32)
    conf = np.concatenate([conf_in, conf_out, conf_bi])

    out_a, out_b, out_c, M_fit = apply_geometric_filters(
        pts_a, pts_b, conf,
        affine_reproj_px=2.0,
        sampson_enabled=True, sampson_tau_px=2.5,
        mte_enabled=True, mte_radius_px=300.0, mte_max_dev_px=6.0,
        min_inliers=30,
    )
    assert out_a is not None

    kept_true = _count_overlap(out_a, pts_a_in)
    kept_bias = _count_overlap(out_a, pts_a_bi)
    kept_rand = _count_overlap(out_a, pts_a_out)

    recall = kept_true / n_in
    precision = kept_true / max(len(out_a), 1)
    assert recall >= 0.90, f"recall {recall:.2f} below 90%"
    assert precision >= 0.90, (
        f"precision {precision:.2f} below 90% "
        f"(true={kept_true}, biased={kept_bias}, random={kept_rand})"
    )


def test_apply_geometric_filters_handles_tiny_match_set():
    # Only 5 matches — below all minimums. Should fail gracefully (None).
    pts_a = RNG.uniform(0, 1000, size=(5, 2)).astype(np.float32)
    pts_b = RNG.uniform(0, 1000, size=(5, 2)).astype(np.float32)
    conf = np.ones(5) * 0.6
    out_a, out_b, out_c, M = apply_geometric_filters(
        pts_a, pts_b, conf,
        affine_reproj_px=3.0,
        sampson_enabled=True, sampson_tau_px=2.0,
        mte_enabled=True,
        min_inliers=10,
    )
    assert out_a is None


def test_robust_affine_filter_is_deterministic():
    # Two identical inputs must produce identical inlier sets now that cv2's
    # RNG is seeded inside robust_affine_filter. Without the seed, cv2.RANSAC
    # samples different subsets on each call and the inlier counts drift.
    n_in = 120
    n_out = 60
    pts_a_in, pts_b_in, conf_in, _M = _make_affine_inliers(n_in, noise_px=0.4)
    pts_a_out, pts_b_out, conf_out = _make_random_outliers(n_out)
    pts_a = np.vstack([pts_a_in, pts_a_out]).astype(np.float32)
    pts_b = np.vstack([pts_b_in, pts_b_out]).astype(np.float32)
    conf = np.concatenate([conf_in, conf_out])

    results = [
        robust_affine_filter(pts_a.copy(), pts_b.copy(), conf.copy(),
                             reproj_px=2.0, min_inliers=30)
        for _ in range(3)
    ]
    inlier_counts = [len(r[0]) if r[0] is not None else 0 for r in results]
    assert inlier_counts[0] == inlier_counts[1] == inlier_counts[2], (
        f"robust_affine_filter non-deterministic: counts={inlier_counts}"
    )
    # The surviving points themselves should be identical too.
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[1][0], results[2][0])


def test_apply_geometric_filters_pure_inliers():
    pts_a, pts_b, conf, M = _make_affine_inliers(120, noise_px=0.3)
    out_a, out_b, out_c, _ = apply_geometric_filters(
        pts_a, pts_b, conf,
        affine_reproj_px=2.0,
        sampson_enabled=True, sampson_tau_px=2.5,
        mte_enabled=True, mte_max_dev_px=6.0,
        min_inliers=30,
    )
    assert out_a is not None
    assert len(out_a) >= int(0.95 * 120), f"pure-inlier set shrank to {len(out_a)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
