"""Affine transform utilities: fitting, residuals, and RANSAC."""

import cv2
import numpy as np

from .constants import RANSAC_REPROJ_THRESHOLD


def fit_affine_from_gcps(src_points, dst_points, weights=None):
    """Fit a 6-parameter affine transformation from matched point pairs.

    Returns the 2x3 affine matrix M and the per-point residuals in metres.
    """
    n = len(src_points)
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    for i in range(n):
        sx, sy = src_points[i]
        dx, dy = dst_points[i]
        A[2 * i] = [sx, sy, 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, sx, sy, 1]
        b[2 * i] = dx
        b[2 * i + 1] = dy

    if weights is not None:
        W = np.zeros(2 * n)
        for i in range(n):
            W[2 * i] = weights[i]
            W[2 * i + 1] = weights[i]
        W_sqrt = np.sqrt(W)
        A = A * W_sqrt[:, np.newaxis]
        b = b * W_sqrt

    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, bv, tx, c, d, ty = result
    M = np.array([[a, bv, tx], [c, d, ty]])

    residuals = compute_affine_residuals(M, src_points, dst_points)

    return M, residuals


def compute_affine_residuals(M, src_points, dst_points):
    """Compute per-point residuals (metres) for an affine transform.

    M is a 2x3 affine matrix. src_points and dst_points are (N, 2) arrays.
    """
    src_arr = np.asarray(src_points)
    dst_arr = np.asarray(dst_points)
    pred_x = M[0, 0] * src_arr[:, 0] + M[0, 1] * src_arr[:, 1] + M[0, 2]
    pred_y = M[1, 0] * src_arr[:, 0] + M[1, 1] * src_arr[:, 1] + M[1, 2]
    return list(np.sqrt((pred_x - dst_arr[:, 0]) ** 2 + (pred_y - dst_arr[:, 1]) ** 2))


def ransac_affine(src_pts, dst_pts, threshold=None):
    """RANSAC affine estimation wrapping cv2.estimateAffine2D.

    Returns (M, inlier_mask) where M is the 2x3 affine matrix and
    inlier_mask is a boolean array. Returns (None, None) on failure.
    """
    if threshold is None:
        threshold = RANSAC_REPROJ_THRESHOLD
    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=threshold)
    if M is None or inliers is None:
        return None, None
    return M, inliers.ravel().astype(bool)
