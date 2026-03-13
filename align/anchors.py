"""Anchor GCP matching: framework with 5 pluggable matchers.

The common boilerplate (bounds check, patch extraction, quality validation,
sub-pixel phase refinement, geo conversion) lives in :func:`locate_anchor`.
Each matcher implements only its unique matching logic.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rasterio
import rasterio.transform
from rasterio.crs import CRS
from rasterio.warp import transform as transform_coords

import torch
import kornia.feature as KF

from .geo import get_torch_device
from .image import to_u8, make_land_mask, clahe_normalize, sobel_gradient
from .scale import fit_affine_from_gcps


# ---------------------------------------------------------------------------
# Anchor-specific mask policy
# ---------------------------------------------------------------------------

def _anchor_mask_mode(anchor: dict) -> str:
    """Choose the semantic mask mode for a given anchor definition.

    Road anchors near shorelines are sensitive to aggressive coastal
    demotion (``coastal_obia`` can strip the causeway/coastal road land),
    so they use the simpler ``heuristic`` mask.  Island anchors benefit
    from the OBIA mask which better separates shoals from true land.
    """
    feature_type = anchor.get("feature_type", "")
    if feature_type in ("island_center", "reef_feature"):
        return "coastal_obia"
    # Roads, buildings, intersections, and anything else -- use heuristic
    # so that shore-adjacent land is not aggressively demoted.
    return "heuristic"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AnchorMatchResult:
    """Displacement + quality returned by a matcher."""
    dx_px: float
    dy_px: float
    quality: float
    method: str


# ---------------------------------------------------------------------------
# Shared helpers (formerly duplicated in every _anchor_*_match function)
# ---------------------------------------------------------------------------

def _anchor_phase_refine(ref_row, ref_col, dx_int, dy_int,
                         arr_ref, arr_off_shifted, h_img, w_img, phase_half):
    """Phase correlation sub-pixel refinement."""
    match_r = ref_row + dy_int
    match_c = ref_col + dx_int
    p_r0 = max(0, match_r - phase_half)
    p_r1 = min(h_img, match_r + phase_half)
    p_c0 = max(0, match_c - phase_half)
    p_c1 = min(w_img, match_c + phase_half)

    ref_patch_raw = arr_ref[p_r0:p_r1, p_c0:p_c1]
    off_patch_raw = arr_off_shifted[p_r0:p_r1, p_c0:p_c1]

    if ref_patch_raw.shape[0] >= 32 and ref_patch_raw.shape[1] >= 32:
        ref_patch = ref_patch_raw.astype(np.float64)
        off_patch = off_patch_raw.astype(np.float64)
        hann_r = np.hanning(ref_patch.shape[0])
        hann_c = np.hanning(ref_patch.shape[1])
        window = np.outer(hann_r, hann_c)
        shift, response = cv2.phaseCorrelate(ref_patch * window, off_patch * window)
        if response > 0.05:
            return shift[0], shift[1]

    return 0.0, 0.0


def _anchor_to_geo(ref_row, ref_col, total_dx, total_dy,
                   ref_transform, offset_transform, shift_px_x, shift_py_y):
    """Convert anchor pixel match to geographic coordinate tuple."""
    ref_gx, ref_gy = rasterio.transform.xy(ref_transform, float(ref_row), float(ref_col))
    off_overlap_c = ref_col + total_dx + shift_px_x
    off_overlap_r = ref_row + total_dy + shift_py_y
    off_gx, off_gy = rasterio.transform.xy(offset_transform, off_overlap_r, off_overlap_c)
    return ref_gx, ref_gy, off_gx, off_gy


def _extract_patches(ref_row, ref_col, patch_half, arr_ref, arr_off_shifted,
                     h_img, w_img, off_row_adj=0, off_col_adj=0):
    """Extract matching patches from ref and offset.

    Returns (ref_patch, off_patch) or (None, None) if out of bounds or
    insufficient valid data.
    """
    pr0 = ref_row - patch_half
    pr1 = ref_row + patch_half
    pc0 = ref_col - patch_half
    pc1 = ref_col + patch_half

    if pr0 < 0 or pr1 > h_img or pc0 < 0 or pc1 > w_img:
        return None, None

    opr0 = max(0, pr0 + off_row_adj)
    opr1 = min(h_img, pr1 + off_row_adj)
    opc0 = max(0, pc0 + off_col_adj)
    opc1 = min(w_img, pc1 + off_col_adj)

    ref_patch = arr_ref[pr0:pr1, pc0:pc1]
    off_patch = arr_off_shifted[opr0:opr1, opc0:opc1]

    if off_patch.shape != ref_patch.shape:
        padded = np.zeros_like(ref_patch)
        ph = min(off_patch.shape[0], padded.shape[0])
        pw = min(off_patch.shape[1], padded.shape[1])
        padded[:ph, :pw] = off_patch[:ph, :pw]
        off_patch = padded

    if np.mean(ref_patch > 0) < 0.3 or np.mean(off_patch > 0) < 0.3:
        return None, None

    return ref_patch, off_patch


def _island_presearch(name, ref_row, ref_col, arr_ref, arr_off_shifted,
                       h_img, w_img, patch_half, fine_res, search_expansion=2.0,
                       mask_mode="heuristic"):
    """Find actual island position in offset using land-mask NCC."""
    pr0 = max(0, ref_row - patch_half)
    pr1 = min(h_img, ref_row + patch_half)
    pc0 = max(0, ref_col - patch_half)
    pc1 = min(w_img, ref_col + patch_half)
    ref_patch = arr_ref[pr0:pr1, pc0:pc1]
    ref_land = (make_land_mask(ref_patch, mode=mask_mode) > 0).astype(np.float32)

    if np.mean(ref_land) < 0.01:
        return 0, 0, 0.0

    search_half = int(patch_half * search_expansion)
    sr0 = max(0, ref_row - search_half)
    sr1 = min(h_img, ref_row + search_half)
    sc0 = max(0, ref_col - search_half)
    sc1 = min(w_img, ref_col + search_half)
    off_search = arr_off_shifted[sr0:sr1, sc0:sc1]
    off_land = (make_land_mask(off_search, mode=mask_mode) > 0).astype(np.float32)

    if np.mean(off_land) < 0.005:
        return 0, 0, 0.0

    ref_land_u8 = (ref_land * 255).astype(np.uint8)
    off_land_u8 = (off_land * 255).astype(np.uint8)

    if (off_land_u8.shape[0] <= ref_land_u8.shape[0] or
            off_land_u8.shape[1] <= ref_land_u8.shape[1]):
        return 0, 0, 0.0

    result = cv2.matchTemplate(off_land_u8, ref_land_u8, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < 0.25:
        return 0, 0, 0.0

    matched_row = sr0 + max_loc[1] + (pr1 - pr0) // 2
    matched_col = sc0 + max_loc[0] + (pc1 - pc0) // 2
    off_row_adj = matched_row - ref_row
    off_col_adj = matched_col - ref_col

    shift_m = np.sqrt((off_row_adj * fine_res) ** 2 + (off_col_adj * fine_res) ** 2)
    if shift_m > 5:
        print(f"    Island pre-search: {name} offset by "
              f"{off_row_adj * fine_res:.0f}m N, {off_col_adj * fine_res:.0f}m E "
              f"(NCC={max_val:.2f})")
    return off_row_adj, off_col_adj, max_val


def _filter_anchor_displacement_outliers(results, min_keep=4):
    """Remove gross anchor displacement outliers with robust statistics.

    Operates on final anchor matches to avoid poisoning downstream affine fits.
    """
    if len(results) < 6:
        return results

    de = np.array([r[2] - r[0] for r in results], dtype=np.float64)
    dn = np.array([r[3] - r[1] for r in results], dtype=np.float64)
    med_de = float(np.median(de))
    med_dn = float(np.median(dn))
    dist = np.sqrt((de - med_de) ** 2 + (dn - med_dn) ** 2)
    mad = float(np.median(np.abs(dist - np.median(dist))) * 1.4826)
    threshold = max(180.0, 3.5 * mad)

    keep_mask = dist <= threshold
    if int(np.sum(keep_mask)) < min_keep:
        return results

    n_rejected = len(results) - int(np.sum(keep_mask))
    if n_rejected <= 0:
        return results

    rejected_names = [results[i][5].replace("anchor:", "")
                      for i in range(len(results)) if not keep_mask[i]]
    print(f"    Anchor QA: rejected {n_rejected} displacement outlier(s) "
          f"(threshold={threshold:.0f}m): {', '.join(rejected_names)}")
    return [r for r, keep in zip(results, keep_mask) if keep]


# ---------------------------------------------------------------------------
# ABC matcher base class + concrete matchers
# ---------------------------------------------------------------------------

class AnchorMatcher(ABC):
    """Base class for anchor matching strategies."""
    name: str

    @abstractmethod
    def match(self, ref_patch, off_patch, patch_half, name,
              arr_ref, arr_off_shifted, ref_row, ref_col,
              h_img, w_img, fine_res,
              off_row_adj, off_col_adj,
              ref_transform, offset_transform,
              shift_px_x, shift_py_y,
              loftr_model=None, loftr_device=None,
              phase_half=32, mask_mode="heuristic",
              roma_model=None) -> Optional[AnchorMatchResult]:
        ...


class LoFTRMatcher(AnchorMatcher):
    """LoFTR dense matching with local affine prediction."""

    def __init__(self, edge_preprocess=False):
        self.name = f"LoFTR{'(edge)' if edge_preprocess else ''}"
        self.edge_preprocess = edge_preprocess

    def match(self, ref_patch, off_patch, patch_half, name,
              arr_ref, arr_off_shifted, ref_row, ref_col,
              h_img, w_img, fine_res,
              off_row_adj, off_col_adj,
              ref_transform, offset_transform,
              shift_px_x, shift_py_y,
              loftr_model=None, loftr_device=None,
              phase_half=32, mask_mode="heuristic",
              roma_model=None) -> Optional[AnchorMatchResult]:
        if loftr_model is None:
            return None

        device = loftr_device or 'cpu'
        ref_u8 = clahe_normalize(ref_patch)
        off_u8 = clahe_normalize(off_patch)

        if self.edge_preprocess:
            ref_u8 = clahe_normalize(to_u8(sobel_gradient(ref_u8)))
            off_u8 = clahe_normalize(to_u8(sobel_gradient(off_u8)))

        ref_t = torch.from_numpy(ref_u8).float()[None, None] / 255.0
        off_t = torch.from_numpy(off_u8).float()[None, None] / 255.0
        ref_t = ref_t.to(device)
        off_t = off_t.to(device)

        try:
            with torch.no_grad():
                correspondences = loftr_model({"image0": ref_t, "image1": off_t})
            kpts0 = correspondences['keypoints0'].cpu().numpy()
            kpts1 = correspondences['keypoints1'].cpu().numpy()
            conf = correspondences['confidence'].cpu().numpy()
        except Exception as e:
            print(f"    LoFTR error for {name}: {e}")
            return None

        mask = conf > 0.5
        if not mask.any():
            return None

        kpts0 = kpts0[mask]
        kpts1 = kpts1[mask]
        conf = conf[mask]

        # Filter water pixels using the anchor-specific mask mode
        ref_land_mask = make_land_mask(ref_patch, mode=mask_mode)
        off_land_mask = make_land_mask(off_patch, mode=mask_mode)
        ref_land_frac = np.mean(ref_land_mask > 0)

        if ref_land_frac < 0.7:
            kpt0_rows = np.clip(kpts0[:, 1].astype(int), 0, ref_land_mask.shape[0] - 1)
            kpt0_cols = np.clip(kpts0[:, 0].astype(int), 0, ref_land_mask.shape[1] - 1)
            kpt1_rows = np.clip(kpts1[:, 1].astype(int), 0, off_land_mask.shape[0] - 1)
            kpt1_cols = np.clip(kpts1[:, 0].astype(int), 0, off_land_mask.shape[1] - 1)
            on_land = ((ref_land_mask[kpt0_rows, kpt0_cols] > 0) &
                       (off_land_mask[kpt1_rows, kpt1_cols] > 0))
            n_before = len(kpts0)
            kpts0, kpts1, conf = kpts0[on_land], kpts1[on_land], conf[on_land]
            if len(kpts0) < n_before:
                print(f"    Land-pixel filter: {n_before} -> {len(kpts0)} matches "
                      f"for {name} (land={ref_land_frac:.0%})")

        if len(kpts0) < 4:
            return None

        center_x = float(patch_half)
        center_y = float(patch_half)

        src_pts = kpts0.astype(np.float32).reshape(-1, 1, 2)
        dst_pts = kpts1.astype(np.float32).reshape(-1, 1, 2)
        M_local, inliers = cv2.estimateAffine2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

        if M_local is None or inliers is None:
            return None

        n_inliers = int(inliers.sum())
        if n_inliers < 4:
            return None
        if n_inliers < 6:
            inlier_ratio = n_inliers / len(kpts0)
            if inlier_ratio < 0.6:
                return None

        # Quality gate
        a_qg, b_qg = M_local[0, 0], M_local[0, 1]
        c_qg, d_qg = M_local[1, 0], M_local[1, 1]
        qg_scale = (np.sqrt(a_qg ** 2 + c_qg ** 2) + np.sqrt(b_qg ** 2 + d_qg ** 2)) / 2
        qg_rot = abs(np.degrees(np.arctan2(c_qg, a_qg)))

        use_translation = qg_rot > 5.0 or qg_scale < 0.6 or qg_scale > 1.4

        if use_translation:
            inlier_mask = inliers.ravel().astype(bool)
            if np.sum(inlier_mask) < 3:
                return None
            displacements = kpts1[inlier_mask] - kpts0[inlier_mask]
            median_dx = float(np.median(displacements[:, 0]))
            median_dy = float(np.median(displacements[:, 1]))
            residuals = np.sqrt(np.sum((displacements - [median_dx, median_dy]) ** 2, axis=1))
            if float(np.mean(residuals)) > 15:
                return None
            total_dx = median_dx + off_col_adj
            total_dy = median_dy + off_row_adj
            print(f"    Located: {name} (TRANSLATION fallback: {np.sum(inlier_mask)} inliers)")
        else:
            center_pt = np.array([[[center_x, center_y]]], dtype=np.float32)
            pred_pt = cv2.transform(center_pt, M_local)
            total_dx = float(pred_pt[0, 0, 0]) - center_x + off_col_adj
            total_dy = float(pred_pt[0, 0, 1]) - center_y + off_row_adj

            inlier_mask = inliers.ravel().astype(bool)
            inlier_src = kpts0[inlier_mask]
            inlier_dst = kpts1[inlier_mask]
            pred_all = cv2.transform(
                inlier_src.astype(np.float32).reshape(-1, 1, 2), M_local
            ).reshape(-1, 2)
            residuals = np.sqrt(np.sum((pred_all - inlier_dst) ** 2, axis=1))
            mean_res_m = float(np.mean(residuals)) * fine_res
            local_scale = (np.sqrt(a_qg ** 2 + c_qg ** 2) + np.sqrt(b_qg ** 2 + d_qg ** 2)) / 2
            local_rot = np.degrees(np.arctan2(c_qg, a_qg))
            print(f"    Located: {name} (local affine: {n_inliers}/{len(kpts0)} inliers, "
                  f"res={mean_res_m:.1f}m, scale={local_scale:.3f}, rot={local_rot:.2f}deg)")

        return AnchorMatchResult(total_dx, total_dy, 1.0, self.name)


class GradientNCCMatcher(AnchorMatcher):
    """Gradient-enhanced NCC template matching."""
    name = "GradientNCC"

    def match(self, ref_patch, off_patch, patch_half, name,
              arr_ref, arr_off_shifted, ref_row, ref_col,
              h_img, w_img, fine_res,
              off_row_adj, off_col_adj,
              ref_transform, offset_transform,
              shift_px_x, shift_py_y,
              loftr_model=None, loftr_device=None,
              phase_half=32, mask_mode="heuristic",
              roma_model=None, template_half=None, search_margin=None) -> Optional[AnchorMatchResult]:
        if template_half is None:
            template_half = patch_half // 2
        if search_margin is None:
            search_margin = patch_half

        tr0 = ref_row - template_half
        tr1 = ref_row + template_half
        tc0 = ref_col - template_half
        tc1 = ref_col + template_half
        if tr0 < 0 or tr1 > h_img or tc0 < 0 or tc1 > w_img:
            return None

        template = arr_ref[tr0:tr1, tc0:tc1]
        if np.mean(template > 0) < 0.3:
            return None

        off_tr0 = tr0 + off_row_adj
        off_tr1 = tr1 + off_row_adj
        off_tc0 = tc0 + off_col_adj
        off_tc1 = tc1 + off_col_adj
        sr0 = max(0, off_tr0 - search_margin)
        sr1 = min(h_img, off_tr1 + search_margin)
        sc0 = max(0, off_tc0 - search_margin)
        sc1 = min(w_img, off_tc1 + search_margin)

        template_u8 = to_u8(template)
        if sr1 - sr0 <= template_u8.shape[0] or sc1 - sc0 <= template_u8.shape[1]:
            return None

        off_search = arr_off_shifted[sr0:sr1, sc0:sc1]
        if np.mean(off_search > 0) < 0.3:
            return None

        off_search_u8 = to_u8(off_search)
        template_grad = to_u8(sobel_gradient(template_u8))
        off_grad = to_u8(sobel_gradient(off_search_u8))

        result = cv2.matchTemplate(off_grad, template_grad, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < 0.4:
            return None

        dx_int = (sc0 + max_loc[0]) - tc0
        dy_int = (sr0 + max_loc[1]) - tr0

        sub_dx, sub_dy = _anchor_phase_refine(
            ref_row, ref_col, dx_int, dy_int,
            arr_ref, arr_off_shifted, h_img, w_img, phase_half)

        total_dx = float(dx_int) + sub_dx
        total_dy = float(dy_int) + sub_dy
        print(f"    Located: {name} (gradient NCC={max_val:.3f})")
        return AnchorMatchResult(total_dx, total_dy, 1.0, self.name)


class StandardNCCMatcher(AnchorMatcher):
    """Standard CLAHE-enhanced NCC template matching."""
    name = "StandardNCC"

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def match(self, ref_patch, off_patch, patch_half, name,
              arr_ref, arr_off_shifted, ref_row, ref_col,
              h_img, w_img, fine_res,
              off_row_adj, off_col_adj,
              ref_transform, offset_transform,
              shift_px_x, shift_py_y,
              loftr_model=None, loftr_device=None,
              phase_half=32, mask_mode="heuristic",
              roma_model=None, template_half=None, search_margin=None) -> Optional[AnchorMatchResult]:
        if template_half is None:
            template_half = patch_half // 2
        if search_margin is None:
            search_margin = patch_half

        tr0 = ref_row - template_half
        tr1 = ref_row + template_half
        tc0 = ref_col - template_half
        tc1 = ref_col + template_half
        if tr0 < 0 or tr1 > h_img or tc0 < 0 or tc1 > w_img:
            return None

        template = arr_ref[tr0:tr1, tc0:tc1]
        if np.mean(template > 0) < 0.3:
            return None

        template_u8 = clahe_normalize(template)

        off_tr0 = tr0 + off_row_adj
        off_tr1 = tr1 + off_row_adj
        off_tc0 = tc0 + off_col_adj
        off_tc1 = tc1 + off_col_adj
        sr0 = max(0, off_tr0 - search_margin)
        sr1 = min(h_img, off_tr1 + search_margin)
        sc0 = max(0, off_tc0 - search_margin)
        sc1 = min(w_img, off_tc1 + search_margin)

        if sr1 - sr0 <= template_u8.shape[0] or sc1 - sc0 <= template_u8.shape[1]:
            return None

        off_search = arr_off_shifted[sr0:sr1, sc0:sc1]
        if np.mean(off_search > 0) < 0.3:
            return None

        off_search_u8 = clahe_normalize(off_search)

        result = cv2.matchTemplate(off_search_u8, template_u8, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < self.threshold:
            return None

        dx_int = (sc0 + max_loc[0]) - tc0
        dy_int = (sr0 + max_loc[1]) - tr0

        sub_dx, sub_dy = _anchor_phase_refine(
            ref_row, ref_col, dx_int, dy_int,
            arr_ref, arr_off_shifted, h_img, w_img, phase_half)

        total_dx = float(dx_int) + sub_dx
        total_dy = float(dy_int) + sub_dy
        print(f"    Located: {name} (NCC={max_val:.3f})")
        return AnchorMatchResult(total_dx, total_dy, 1.0, self.name)


class RoMaMatcher(AnchorMatcher):
    """Robust Dense Feature Matching (RoMa) with foundation model guidance."""
    name = "RoMa"

    def match(self, ref_patch, off_patch, patch_half, name,
              arr_ref, arr_off_shifted, ref_row, ref_col,
              h_img, w_img, fine_res,
              off_row_adj, off_col_adj,
              ref_transform, offset_transform,
              shift_px_x, shift_py_y,
              loftr_model=None, loftr_device=None,
              phase_half=32, mask_mode="heuristic",
              roma_model=None) -> Optional[AnchorMatchResult]:
        if roma_model is None:
            return None

        device = loftr_device or 'cpu'

        # RoMa expects 3-channel RGB [0, 1] normalized tensors
        def _to_roma_tensor(patch):
            p_rgb = cv2.cvtColor(to_u8(patch), cv2.COLOR_GRAY2RGB)

            cur_h, cur_w = patch.shape[:2]
            min_dim = 448
            if cur_h < min_dim or cur_w < min_dim:
                scale = max(min_dim / cur_h, min_dim / cur_w)
                target_h = int(round((cur_h * scale) / 14) * 14)
                target_w = int(round((cur_w * scale) / 14) * 14)
                p_rgb = cv2.resize(p_rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            else:
                target_h = (cur_h // 14) * 14
                target_w = (cur_w // 14) * 14
                p_rgb = cv2.resize(p_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)

            p_t = torch.from_numpy(p_rgb).permute(2, 0, 1).float() / 255.0
            return p_t[None].to(device)

        ref_t = _to_roma_tensor(ref_patch)
        off_t = _to_roma_tensor(off_patch)

        def _deterministic_topk_sample(warp, certainty, num=400, grid_cells=8):
            """Deterministic top-k sampling with spatial distribution.

            Divides the certainty map into grid_cells x grid_cells regions and
            picks the top-k highest-certainty matches per cell, ensuring spatial
            coverage instead of clustering in one high-confidence area.
            """
            cert_2d = certainty[0]  # [H, W]
            warp_2d = warp[0]       # [H, W, 4]
            H, W = cert_2d.shape

            per_cell = max(1, num // (grid_cells * grid_cells))
            cell_h = max(1, H // grid_cells)
            cell_w = max(1, W // grid_cells)

            all_matches = []
            all_certs = []

            for gi in range(grid_cells):
                for gj in range(grid_cells):
                    r0 = gi * cell_h
                    r1 = min(r0 + cell_h, H) if gi < grid_cells - 1 else H
                    c0 = gj * cell_w
                    c1 = min(c0 + cell_w, W) if gj < grid_cells - 1 else W

                    cell_cert = cert_2d[r0:r1, c0:c1]
                    cell_warp = warp_2d[r0:r1, c0:c1]

                    cell_flat_cert = cell_cert.reshape(-1)
                    cell_flat_warp = cell_warp.reshape(-1, 4)

                    k = min(per_cell, cell_flat_cert.numel())
                    if k == 0:
                        continue

                    topk_vals, topk_idx = cell_flat_cert.topk(k)
                    all_matches.append(cell_flat_warp[topk_idx])
                    all_certs.append(topk_vals)

            if not all_matches:
                return None, None

            matches = torch.cat(all_matches, dim=0).unsqueeze(0)  # [1, K, 4]
            certs = torch.cat(all_certs, dim=0).unsqueeze(0)       # [1, K]
            return matches, certs

        # Single deterministic run (no stochastic sampling)
        # Use hashlib instead of hash() which is randomized per-process (PEP 456)
        import hashlib
        seed = int(hashlib.sha256(name.encode()).hexdigest(), 16) % 10000
        torch.manual_seed(seed)
        np.random.seed(seed)
        cv2.setRNGSeed(seed)

        try:
            with torch.no_grad():
                roma_model.apply_setting("satast")  # high quality for anchor matching (B=1)
                preds = roma_model.match(ref_t, off_t)
                warp_AB = preds["warp_AB"]  # (1, H, W, 2)
                B, H, W, _ = warp_AB.shape
                from .romav2.geometry import get_normalized_grid
                grid = get_normalized_grid(B, H, W, overload_device=warp_AB.device)
                warp = torch.cat([grid, warp_AB], dim=-1)  # (1, H, W, 4)
                certainty = preds["confidence_AB"][..., 0]  # (1, H, W) logits
                matches, certs = _deterministic_topk_sample(warp, certainty, num=400)
                if matches is None:
                    return None
                matches = matches.cpu().numpy()
                certs = certs.cpu().numpy()
        except Exception as e:
            print(f"    RoMa error for {name}: {e}")
            return None

        mask = certs.squeeze(0) > 0.2
        if not mask.any():
            return None
        matches = matches[0][mask]

        psize = patch_half * 2
        mkpts0 = (matches[:, :2] + 1) / 2 * psize
        mkpts1 = (matches[:, 2:] + 1) / 2 * psize

        src_pts = mkpts0.astype(np.float32).reshape(-1, 1, 2)
        dst_pts = mkpts1.astype(np.float32).reshape(-1, 1, 2)
        M, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

        if M is None or inliers is None:
            return None

        scale_x = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
        scale_y = np.sqrt(M[1, 0]**2 + M[1, 1]**2)
        rot_deg = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
        if abs(rot_deg) > 5.0 or abs(scale_x - 1.0) > 0.20 or abs(scale_y - 1.0) > 0.20:
            return None

        n_inliers = int(inliers.sum())
        if n_inliers < 6:
            return None

        # Calculate center displacement
        center_pt = np.array([[[float(patch_half), float(patch_half)]]], dtype=np.float32)
        pred_pt = cv2.transform(center_pt, M)
        total_dx = float(pred_pt[0, 0, 0]) - patch_half + off_col_adj
        total_dy = float(pred_pt[0, 0, 1]) - patch_half + off_row_adj

        # Quality stats
        a, b = M[0, 0], M[0, 1]
        c, d = M[1, 0], M[1, 1]
        local_rot = np.degrees(np.arctan2(c, a))
        print(f"    Located: {name} (RoMa foundation: {n_inliers} inliers, deterministic top-k, rot={local_rot:.2f}deg)")

        return AnchorMatchResult(total_dx, total_dy, 1.0, self.name)


class MutualInfoMatcher(AnchorMatcher):
    """Mutual information template matching."""
    name = "MI"

    def match(self, ref_patch, off_patch, patch_half, name,
              arr_ref, arr_off_shifted, ref_row, ref_col,
              h_img, w_img, fine_res,
              off_row_adj, off_col_adj,
              ref_transform, offset_transform,
              shift_px_x, shift_py_y,
              loftr_model=None, loftr_device=None,
              phase_half=32, mask_mode="heuristic",
              roma_model=None, template_half=None, search_margin=None) -> Optional[AnchorMatchResult]:
        from skimage.metrics import normalized_mutual_information

        if template_half is None:
            template_half = patch_half // 2
        if search_margin is None:
            search_margin = patch_half

        tr0 = max(0, ref_row - template_half)
        tr1 = min(h_img, ref_row + template_half)
        tc0 = max(0, ref_col - template_half)
        tc1 = min(w_img, ref_col + template_half)
        template = arr_ref[tr0:tr1, tc0:tc1]

        off_center_r = ref_row + off_row_adj
        off_center_c = ref_col + off_col_adj
        sr0 = max(0, off_center_r - template_half - search_margin)
        sr1 = min(h_img, off_center_r + template_half + search_margin)
        sc0 = max(0, off_center_c - template_half - search_margin)
        sc1 = min(w_img, off_center_c + template_half + search_margin)
        search = arr_off_shifted[sr0:sr1, sc0:sc1]

        if template.size == 0 or search.size == 0:
            return None

        th, tw = template.shape
        sh, sw = search.shape
        if th >= sh or tw >= sw:
            return None

        if np.mean(template > 0) < 0.3 or np.mean(search > 0) < 0.3:
            return None

        template_u8 = to_u8(template)
        search_u8 = to_u8(search)

        best_mi = -np.inf
        best_dy, best_dx = 0, 0

        for dy in range(0, sh - th + 1, 4):
            for dx in range(0, sw - tw + 1, 4):
                candidate = search_u8[dy:dy + th, dx:dx + tw]
                mi = normalized_mutual_information(template_u8, candidate, bins=32)
                if mi > best_mi:
                    best_mi = mi
                    best_dy, best_dx = dy, dx

        fine_best_mi = best_mi
        fine_dy, fine_dx = best_dy, best_dx
        for dy in range(max(0, best_dy - 4), min(sh - th + 1, best_dy + 5)):
            for dx in range(max(0, best_dx - 4), min(sw - tw + 1, best_dx + 5)):
                candidate = search_u8[dy:dy + th, dx:dx + tw]
                mi = normalized_mutual_information(template_u8, candidate, bins=32)
                if mi > fine_best_mi:
                    fine_best_mi = mi
                    fine_dy, fine_dx = dy, dx

        match_row = sr0 + fine_dy + th // 2
        match_col = sc0 + fine_dx + tw // 2
        total_dy = match_row - ref_row
        total_dx = match_col - ref_col

        if fine_best_mi < 1.15:
            return None

        print(f"    Located: {name} (MI match: NMI={fine_best_mi:.3f})")
        return AnchorMatchResult(total_dx, total_dy, 1.0, self.name)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _write_anchor_diagnostics(name, ref_p, off_p, mask_mode, diag_dir, fine_res):
    """Write diagnostic images for a failed anchor showing both mask modes."""
    import os
    diag_dir = os.path.join(diag_dir, "anchor_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    safe_name = name.replace(" ", "_").replace("/", "_").replace(",", "")

    ref_u8 = to_u8(ref_p)
    off_u8 = to_u8(off_p) if off_p is not None and off_p.size > 0 else None

    for mode in ["heuristic", "coastal_obia"]:
        ref_land = (make_land_mask(ref_p, mode=mode) > 0).astype(np.uint8) * 255
        # Overlay: green = land, blue = water
        ref_rgb = cv2.cvtColor(ref_u8, cv2.COLOR_GRAY2BGR)
        ref_overlay = ref_rgb.copy()
        ref_overlay[ref_land > 0] = (0, 200, 0)
        ref_overlay[ref_land == 0] = (180, 90, 30)
        ref_out = cv2.addWeighted(ref_rgb, 0.45, ref_overlay, 0.55, 0)
        ref_land_frac = float(np.mean(ref_land > 0))

        if off_u8 is not None:
            off_land = (make_land_mask(off_p, mode=mode) > 0).astype(np.uint8) * 255
            off_rgb = cv2.cvtColor(off_u8, cv2.COLOR_GRAY2BGR)
            off_overlay = off_rgb.copy()
            off_overlay[off_land > 0] = (0, 200, 0)
            off_overlay[off_land == 0] = (180, 90, 30)
            off_out = cv2.addWeighted(off_rgb, 0.45, off_overlay, 0.55, 0)
            off_land_frac = float(np.mean(off_land > 0))
            # Side by side
            canvas = np.hstack([ref_out, off_out])
        else:
            off_land_frac = 0.0
            canvas = ref_out

        tag = "ACTIVE" if mode == mask_mode else "alt"
        path = os.path.join(diag_dir, f"{safe_name}_{mode}_{tag}.jpg")
        cv2.imwrite(path, canvas)
        print(f"      diag: {path} (ref_land={ref_land_frac:.1%}, off_land={off_land_frac:.1%})")


def locate_anchors(anchors_path, src_ref, src_offset, overlap, work_crs,
                   ref_transform, offset_transform, arr_ref, arr_off_shifted,
                   shift_px_x, shift_py_y, fine_res, model_cache=None,
                   arr_off_unshifted=None, matcher_type="roma",
                   diagnostics_dir=None):
    """Locate anchor GCPs in both reference and offset images.

    *matcher_type*: "roma", "lightglue", or "loftr".  Controls priority.
    """
    anchor_file = open(anchors_path)
    anchor_data = json.load(anchor_file)
    anchor_file.close()

    # Auto-detect format
    if "type" in anchor_data and anchor_data["type"] == "FeatureCollection":
        features = anchor_data.get("features", [])
        gcps_list = []
        for feat in features:
            props = feat.get("properties", {})
            geom = feat.get("geometry")
            if geom is None or not isinstance(geom, dict):
                continue
            if geom.get("type") == "Point" and geom.get("coordinates"):
                lon, lat = geom["coordinates"][0], geom["coordinates"][1]
                gcps_list.append({
                    "name": props.get("name", "Unknown"),
                    "lon": lon, "lat": lat,
                    "confidence": props.get("confidence", "medium"),
                    "patch_size_m": props.get("patch_size_m", None),
                    "feature_type": props.get("feature_type", ""),
                })
        print(f"    Loaded {len(gcps_list)} anchors from GeoJSON FeatureCollection", flush=True)
    else:
        gcps_list = anchor_data.get("gcps", [])
        print(f"    Loaded {len(gcps_list)} anchors from custom JSON format", flush=True)

    results = []
    failed_anchors = []
    h_img, w_img = arr_ref.shape

    patch_half = int(512 / fine_res)
    template_half = int(256 / fine_res)
    search_margin = int(500 / fine_res)
    phase_half = 32

    # Initialise models from cache
    loftr_model = None
    roma_model = None
    loftr_device = None
    if model_cache is not None:
        loftr_device = model_cache.device
        if matcher_type == "loftr":
            try:
                loftr_model = model_cache.loftr
            except Exception as e:
                print(f"    WARNING: LoFTR from cache failed ({e})")
        elif matcher_type == "roma":
            try:
                roma_model = model_cache.roma
            except Exception as e:
                print(f"    WARNING: RoMa from cache failed ({e}), using LoFTR")
                try:
                    loftr_model = model_cache.loftr
                except Exception: pass
    
    # Fallback to local init if cache failed/missing
    if matcher_type == "loftr" and loftr_model is None:
        try:
            loftr_device = get_torch_device()
            loftr_model = KF.LoFTR(pretrained="outdoor").eval().to(loftr_device)
        except Exception: pass

    # Build matcher cascade based on preference
    matchers_neural = []
    if matcher_type == "roma" and roma_model is not None:
        matchers_neural.append(RoMaMatcher())
    elif (matcher_type == "loftr" or roma_model is None) and loftr_model is not None:
        matchers_neural.append(LoFTRMatcher(edge_preprocess=False))
        matchers_neural.append(LoFTRMatcher(edge_preprocess=True))

    # We no longer use NCC/Phase matchers as fallbacks because they are prone
    # to hallucinating matches on flat terrain (e.g. ocean, missing data borders)
    # which poisons the consensus. If neural models fail, the anchor is skipped.
    matchers_ncc = []

    for anchor in gcps_list:
        name = anchor["name"]
        lon = anchor["lon"]
        lat = anchor["lat"]
        confidence = anchor.get("confidence", "medium")

        if confidence == "low":
            continue

        # Per-anchor mask policy: roads use heuristic to avoid
        # aggressive coastal demotion near causeways/shorelines;
        # islands use coastal_obia which better separates shoals.
        anchor_mm = _anchor_mask_mode(anchor)

        custom_patch = anchor.get("patch_size_m", None)
        if custom_patch:
            anchor_patch_half = int(custom_patch / 2 / fine_res)
            print(f"    Using custom patch size: {custom_patch}m for {name}")
        else:
            anchor_patch_half = patch_half

        work_xy = transform_coords(CRS.from_epsg(4326), work_crs, [lon], [lat])
        anchor_x, anchor_y = work_xy[0][0], work_xy[1][0]

        if not (overlap[0] <= anchor_x <= overlap[2] and
                overlap[1] <= anchor_y <= overlap[3]):
            print(f"    Skipped: {name} (outside overlap)")
            continue

        ref_row, ref_col = rasterio.transform.rowcol(ref_transform, anchor_x, anchor_y)
        ref_row, ref_col = int(ref_row), int(ref_col)

        # Verify the offset image actually has data at this location.
        # After shifting, the anchor may fall in a zero-filled region
        # (outside actual image coverage).
        _check_r = max(0, min(ref_row, h_img - 1))
        _check_c = max(0, min(ref_col, w_img - 1))
        _check_half = min(32, anchor_patch_half)
        _cr0 = max(0, _check_r - _check_half)
        _cr1 = min(h_img, _check_r + _check_half)
        _cc0 = max(0, _check_c - _check_half)
        _cc1 = min(w_img, _check_c + _check_half)
        _off_coverage = float(np.mean(arr_off_shifted[_cr0:_cr1, _cc0:_cc1] > 0))
        if _off_coverage < 0.05:
            print(f"    Skipped: {name} (offset image has no data at anchor location, "
                  f"coverage={_off_coverage:.0%})")
            failed_anchors.append({
                "name": name, "ref_row": ref_row, "ref_col": ref_col,
                "ref_gx": anchor_x, "ref_gy": anchor_y,
                "eff_patch_half": anchor_patch_half,
                "off_row_adj": 0, "off_col_adj": 0,
                "no_offset_coverage": True,
                "mask_mode": anchor_mm,
            })
            if diagnostics_dir:
                diag_ph = anchor_patch_half
                dr0 = max(0, ref_row - diag_ph)
                dr1 = min(h_img, ref_row + diag_ph)
                dc0 = max(0, ref_col - diag_ph)
                dc1 = min(w_img, ref_col + diag_ph)
                _write_anchor_diagnostics(
                    name, arr_ref[dr0:dr1, dc0:dc1],
                    arr_off_shifted[dr0:dr1, dc0:dc1],
                    anchor_mm, diagnostics_dir, fine_res)
            continue

        # Auto-shrink patches when land content is low
        eff_patch_half = anchor_patch_half
        shrink_sizes = [anchor_patch_half, anchor_patch_half // 2, anchor_patch_half // 4]
        
        # Neural matchers (RoMa/LoFTR) are more robust to low land content
        # than NCC, so we are more lenient.
        # For islands, we are extremely lenient because they are small features in big sea.
        feature_type = anchor.get("feature_type", "")
        if feature_type in ("island_center", "reef_feature"):
            min_land = 0.005
        else:
            min_land = 0.05 if (roma_model or loftr_model) else 0.15
        
        for try_ph in shrink_sizes:
            if try_ph < 16:
                continue
            pr0 = max(0, ref_row - try_ph)
            pr1 = min(h_img, ref_row + try_ph)
            pc0 = max(0, ref_col - try_ph)
            pc1 = min(w_img, ref_col + try_ph)
            ref_p = arr_ref[pr0:pr1, pc0:pc1]
            off_p = arr_off_shifted[pr0:pr1, pc0:pc1]
            if (ref_p.size > 0 and off_p.size > 0 and
                    np.mean(ref_p > 0) >= 0.05 and np.mean(off_p > 0) >= 0.05):
                ref_land_frac = np.mean(make_land_mask(ref_p, mode=anchor_mm) > 0)
                off_land_frac = np.mean(make_land_mask(off_p, mode=anchor_mm) > 0)
                if ref_land_frac >= min_land and off_land_frac >= min_land:
                    eff_patch_half = try_ph
                    break
        else:
            print(f"    Skipped: {name} (insufficient land content even at smallest patch)")
            failed_anchors.append({
                "name": name, "ref_row": ref_row, "ref_col": ref_col,
                "ref_gx": anchor_x, "ref_gy": anchor_y,
                "eff_patch_half": anchor_patch_half,
                "off_row_adj": 0, "off_col_adj": 0,
                "land_check_failed": True,
                "mask_mode": anchor_mm,
            })
            if diagnostics_dir:
                # Write diagnostic for land-content failure
                diag_ph = anchor_patch_half
                dr0 = max(0, ref_row - diag_ph)
                dr1 = min(h_img, ref_row + diag_ph)
                dc0 = max(0, ref_col - diag_ph)
                dc1 = min(w_img, ref_col + diag_ph)
                _write_anchor_diagnostics(
                    name, arr_ref[dr0:dr1, dc0:dc1],
                    arr_off_shifted[dr0:dr1, dc0:dc1],
                    anchor_mm, diagnostics_dir, fine_res)
            continue

        if eff_patch_half != anchor_patch_half:
            eff_size_m = eff_patch_half * 2 * fine_res
            print(f"    Auto-shrunk patch for {name}: "
                  f"{anchor_patch_half * 2 * fine_res:.0f}m -> {eff_size_m:.0f}m")

        # Pre-search: coarse NCC to correct residual offset before neural matching.
        # Applied to ALL anchor types (not just islands) with wider search.
        feature_type = anchor.get("feature_type", "")
        off_row_adj, off_col_adj = 0, 0
        presearch_ncc = 0.0
        off_row_adj, off_col_adj, presearch_ncc = _island_presearch(
            name, ref_row, ref_col, arr_ref, arr_off_shifted,
            h_img, w_img, eff_patch_half, fine_res,
            search_expansion=3.0,
            mask_mode=anchor_mm)
        if presearch_ncc > 0.3:
            print(f"    Pre-search: {name} offset by {off_row_adj*fine_res:.0f}m N, "
                  f"{off_col_adj*fine_res:.0f}m E (NCC={presearch_ncc:.2f})")

        # Multi-scale patch cascade
        patch_sizes = [eff_patch_half]
        if eff_patch_half > 64:
            patch_sizes.append(eff_patch_half // 2)
        if eff_patch_half > 128:
            patch_sizes.append(eff_patch_half // 4)

        best_match = None
        for try_ph in patch_sizes:
            eff_th = min(template_half, try_ph // 2)
            eff_sm = min(search_margin, try_ph)

            ref_patch, off_patch = _extract_patches(
                ref_row, ref_col, try_ph, arr_ref, arr_off_shifted,
                h_img, w_img, off_row_adj, off_col_adj)

            # Try neural matchers first
            if ref_patch is not None:
                for matcher in matchers_neural:
                    result = matcher.match(
                        ref_patch, off_patch, try_ph, name,
                        arr_ref, arr_off_shifted, ref_row, ref_col,
                        h_img, w_img, fine_res,
                        off_row_adj, off_col_adj,
                        ref_transform, offset_transform,
                        shift_px_x, shift_py_y,
                        loftr_model=loftr_model, loftr_device=loftr_device,
                        phase_half=phase_half, mask_mode=anchor_mm,
                        roma_model=roma_model)
                    if result is not None:
                        best_match = result
                        break

            if best_match is not None:
                break

            # NCC-based matchers (use the full arrays, not extracted patches)
            for matcher in matchers_ncc:
                kwargs = {}
                if isinstance(matcher, (GradientNCCMatcher, StandardNCCMatcher, MutualInfoMatcher)):
                    kwargs = {'template_half': eff_th, 'search_margin': eff_sm}

                result = matcher.match(
                    ref_patch, off_patch, try_ph, name,
                    arr_ref, arr_off_shifted, ref_row, ref_col,
                    h_img, w_img, fine_res,
                    off_row_adj, off_col_adj,
                    ref_transform, offset_transform,
                    shift_px_x, shift_py_y,
                    loftr_model=loftr_model, loftr_device=loftr_device,
                    phase_half=phase_half, mask_mode=anchor_mm,
                    **kwargs)
                if result is not None:
                    best_match = result
                    break

            if best_match is not None:
                break

        # Presearch fallback: for island/reef types the land mask shape IS the
        # feature.  For corners (e.g. fort), the mask NCC is less precise but
        # still useful when the offset is modest (< 300m) and NCC is strong.
        presearch_offset_m = np.sqrt(off_row_adj**2 + off_col_adj**2) * fine_res
        if best_match is None and presearch_ncc >= 0.44 and feature_type in ("island_center", "reef_feature"):
            print(f"    Located: {name} (presearch mask fallback: NCC={presearch_ncc:.2f})")
            best_match = AnchorMatchResult(float(off_col_adj), float(off_row_adj), presearch_ncc * 0.8, "PresearchMask")
        elif best_match is None and presearch_ncc >= 0.50 and feature_type == "corner" and presearch_offset_m < 300.0:
            print(f"    Located: {name} (presearch corner fallback: NCC={presearch_ncc:.2f}, offset={presearch_offset_m:.0f}m)")
            best_match = AnchorMatchResult(float(off_col_adj), float(off_row_adj), presearch_ncc * 0.7, "PresearchCorner")
        elif best_match is None and presearch_ncc >= 0.35:
            print(f"    Skipping presearch fallback for {name} (type={feature_type!r}, not island/reef/corner)")

        if best_match is not None:
            ref_gx, ref_gy, off_gx, off_gy = _anchor_to_geo(
                ref_row, ref_col, best_match.dx_px, best_match.dy_px,
                ref_transform, offset_transform, shift_px_x, shift_py_y)
            results.append((ref_gx, ref_gy, off_gx, off_gy,
                           best_match.quality, f"anchor:{name}"))
            disp_e = off_gx - ref_gx
            disp_n = off_gy - ref_gy
            disp_tot = np.sqrt(disp_e ** 2 + disp_n ** 2)
            print(f"      displacement: dE={disp_e:+.1f}m, dN={disp_n:+.1f}m, "
                  f"total={disp_tot:.1f}m")
            continue

        failed_anchors.append({
            "name": name, "ref_row": ref_row, "ref_col": ref_col,
            "ref_gx": anchor_x, "ref_gy": anchor_y,
            "eff_patch_half": eff_patch_half,
            "off_row_adj": off_row_adj, "off_col_adj": off_col_adj,
            "mask_mode": anchor_mm,
        })
        print(f"    Skipped: {name} (no method matched at any patch size)")
        if diagnostics_dir:
            diag_ph = eff_patch_half
            dr0 = max(0, ref_row - diag_ph)
            dr1 = min(h_img, ref_row + diag_ph)
            dc0 = max(0, ref_col - diag_ph)
            dc1 = min(w_img, ref_col + diag_ph)
            _write_anchor_diagnostics(
                name, arr_ref[dr0:dr1, dc0:dc1],
                arr_off_shifted[dr0:dr1, dc0:dc1],
                anchor_mm, diagnostics_dir, fine_res)

    # Pass 1.5: Re-try failed anchors with affine-predicted offset adjustments
    # The main failure mode is anchors on islands where the global coarse shift
    # is wrong enough that the offset patch lands in the sea, failing the land
    # content check.  With 3+ successful anchors we can fit a forward affine,
    # invert it, and predict where the offset patch should actually be.
    #
    # The predicted offset position often falls OUTSIDE the overlap array
    # (the coarse georeferencing error pushes it beyond the reference extent).
    # We read the offset patch directly from the source file at the predicted
    # geo position, bypassing the overlap-limited arrays entirely.
    if failed_anchors and len(results) >= 6:
        from .geo import read_overlap_region

        # Fit forward affine: offset_pos -> ref_pos
        succ_off = np.array([(r[2], r[3]) for r in results])
        succ_ref = np.array([(r[0], r[1]) for r in results])
        M_fwd, fit_residuals = fit_affine_from_gcps(succ_off, succ_ref)
        fit_med_res = float(np.median(fit_residuals)) if len(fit_residuals) else np.inf

        # Pass 1.5 is only safe when the existing anchors define a reasonably
        # coherent affine. With too few anchors or a noisy fit, the predicted
        # offset patch can land on unrelated terrain and RoMa will happily
        # "recover" a bogus landmark.
        if fit_med_res > 120.0:
            print(f"\n    Pass 1.5 skipped: anchor affine too inconsistent "
                  f"(median residual {fit_med_res:.1f}m)", flush=True)
            results = _filter_anchor_displacement_outliers(results)
            return results

        # Invert to get ref_pos -> offset_pos
        M_3x3 = np.vstack([M_fwd, [0, 0, 1]])
        M_inv = np.linalg.inv(M_3x3)[:2]

        print(f"\n    Pass 1.5: Re-trying {len(failed_anchors)} failed anchors "
              f"with affine-predicted offsets ({len(results)} anchors in fit, "
              f"median residual {fit_med_res:.1f}m)", flush=True)

        still_failed = []
        for fa in failed_anchors:
            fname = fa["name"]
            frow, fcol = fa["ref_row"], fa["ref_col"]
            fph = fa["eff_patch_half"]
            fa_ref_gx, fa_ref_gy = fa["ref_gx"], fa["ref_gy"]
            fa_mm = fa.get("mask_mode", "heuristic")

            # Predict offset position from known ref position (work_crs)
            pred_off_gx = (M_inv[0, 0] * fa_ref_gx
                           + M_inv[0, 1] * fa_ref_gy + M_inv[0, 2])
            pred_off_gy = (M_inv[1, 0] * fa_ref_gx
                           + M_inv[1, 1] * fa_ref_gy + M_inv[1, 2])

            shift_m = np.sqrt((pred_off_gx - fa_ref_gx) ** 2
                              + (pred_off_gy - fa_ref_gy) ** 2)
            print(f"      {fname}: predicted offset at "
                  f"({pred_off_gx:.0f}, {pred_off_gy:.0f}) "
                  f"({shift_m:.0f}m from ref)")

            # Read offset patch directly from source file at the
            # predicted position.  This works even when the position is
            # outside the overlap array (common with large coarse shifts).
            eff_patch_half = fph
            land_ok = False
            off_patch_direct = None
            ref_patch_direct = None
            shrink_sizes = [fph, fph // 2, fph // 4]
            for try_ph in shrink_sizes:
                if try_ph < 16:
                    continue
                extent_m = try_ph * fine_res
                try_size_m = try_ph * 2 * fine_res

                # Ref patch from the pre-read array
                pr0 = max(0, frow - try_ph)
                pr1 = min(h_img, frow + try_ph)
                pc0 = max(0, fcol - try_ph)
                pc1 = min(w_img, fcol + try_ph)
                ref_p = arr_ref[pr0:pr1, pc0:pc1]
                if ref_p.size == 0 or np.mean(ref_p > 0) < 0.1:
                    continue

                # Offset patch: read from file at predicted geo position
                off_extent = (pred_off_gx - extent_m, pred_off_gy - extent_m,
                              pred_off_gx + extent_m, pred_off_gy + extent_m)
                try:
                    off_p, _ = read_overlap_region(
                        src_offset, off_extent, work_crs, fine_res)
                except Exception:
                    continue

                if off_p.size == 0 or np.mean(off_p > 0) < 0.1:
                    print(f"        {fname} @ {try_size_m:.0f}m: "
                          f"off_valid={np.mean(off_p > 0):.1%}")
                    continue

                ref_land_frac = np.mean(make_land_mask(ref_p, mode=fa_mm) > 0)
                off_land_frac = np.mean(make_land_mask(off_p, mode=fa_mm) > 0)
                print(f"        {fname} @ {try_size_m:.0f}m: "
                      f"ref_land={ref_land_frac:.1%}, "
                      f"off_land={off_land_frac:.1%}")

                if ref_land_frac >= 0.15 and off_land_frac >= 0.15:
                    eff_patch_half = try_ph
                    # Make patches same shape for matcher
                    min_h = min(ref_p.shape[0], off_p.shape[0])
                    min_w = min(ref_p.shape[1], off_p.shape[1])
                    ref_patch_direct = ref_p[:min_h, :min_w]
                    off_patch_direct = off_p[:min_h, :min_w]
                    land_ok = True
                    break

            if not land_ok:
                print(f"      {fname}: still insufficient land content")
                still_failed.append(fa)
                continue

            # Run neural matcher on the directly-read patches.
            # Displacement 0 means the patches are already centred on the
            # correct positions, so off_row_adj = off_col_adj = 0 for
            # the matcher.  For geo conversion we compute the result
            # directly from the predicted offset position + matcher delta.
            best_match = None
            if ref_patch_direct is not None and off_patch_direct is not None:
                try_ph = min(ref_patch_direct.shape) // 2
                for matcher in matchers_neural:
                    result = matcher.match(
                        ref_patch_direct, off_patch_direct,
                        try_ph, fname,
                        arr_ref, arr_off_shifted, frow, fcol,
                        h_img, w_img, fine_res,
                        0, 0,  # patches already aligned
                        ref_transform, offset_transform,
                        shift_px_x, shift_py_y,
                        loftr_model=loftr_model,
                        loftr_device=loftr_device,
                        phase_half=phase_half, mask_mode=fa_mm,
                        roma_model=roma_model)
                    if result is not None:
                        best_match = result
                        break

            if best_match is not None:
                # Convert to geo: the matcher returns (dx, dy) displacement
                # between the patch centres.  Since the ref patch is centred
                # on (frow, fcol) and the offset patch on the predicted
                # position, we compute the final offset geo position
                # directly.
                ref_gx, ref_gy = rasterio.transform.xy(
                    ref_transform, float(frow), float(fcol))
                # Offset position = predicted centre + matcher sub-pixel
                off_gx = pred_off_gx + best_match.dx_px * fine_res
                off_gy = pred_off_gy - best_match.dy_px * fine_res
                results.append((ref_gx, ref_gy, off_gx, off_gy,
                               best_match.quality, f"anchor:{fname}"))
                disp_e = off_gx - ref_gx
                disp_n = off_gy - ref_gy
                disp_tot = np.sqrt(disp_e ** 2 + disp_n ** 2)
                print(f"      {fname}: RECOVERED via Pass 1.5 "
                      f"(dx_px={best_match.dx_px:.1f}, dy_px={best_match.dy_px:.1f}, "
                      f"disp: dE={disp_e:+.1f}m, dN={disp_n:+.1f}m, "
                      f"total={disp_tot:.1f}m)")
            else:
                still_failed.append(fa)

        failed_anchors = still_failed

    # Pass 2: Retry failed anchors at different patch sizes.
    # Lazy-load LoFTR if only RoMa was used, so failed RoMa anchors still
    # get a chance with a different matcher.
    if len(failed_anchors) > 0 and loftr_model is None and len(results) >= 2:
        try:
            loftr_device = loftr_device or get_torch_device()
            loftr_model = KF.LoFTR(pretrained="outdoor").eval().to(loftr_device)
            print("  [Pass 2] Lazy-loaded LoFTR for failed anchor retry", flush=True)
        except Exception:
            pass

    if len(failed_anchors) > 0 and loftr_model is not None and len(results) >= 2:
        print(f"\n    Pass 2: Re-evaluating {len(failed_anchors)} failed anchors "
              f"with relaxed quality gates ({len(results)} anchors succeeded in Pass 1)", flush=True)
        for fa in failed_anchors:
            fname = fa["name"]
            frow, fcol = fa["ref_row"], fa["ref_col"]
            fph = fa["eff_patch_half"]
            f_ora = fa.get("off_row_adj", 0)
            f_oca = fa.get("off_col_adj", 0)
            fa_mm = fa.get("mask_mode", "heuristic")

            retry_sizes = []
            if fph > 64:
                retry_sizes.append(fph // 3)
            if fph > 96:
                retry_sizes.append(fph // 5)

            for rph in retry_sizes:
                if rph < 32:
                    continue
                retry_size_m = rph * 2 * fine_res
                print(f"    Pass 2 retry: {fname} at {retry_size_m:.0f}m")

                ref_patch, off_patch = _extract_patches(
                    frow, fcol, rph, arr_ref, arr_off_shifted,
                    h_img, w_img, f_ora, f_oca)
                if ref_patch is None:
                    continue

                for matcher in matchers_neural:
                    result = matcher.match(
                        ref_patch, off_patch, rph, fname,
                        arr_ref, arr_off_shifted, frow, fcol,
                        h_img, w_img, fine_res,
                        f_ora, f_oca,
                        ref_transform, offset_transform,
                        shift_px_x, shift_py_y,
                        loftr_model=loftr_model, loftr_device=loftr_device,
                        phase_half=phase_half, mask_mode=fa_mm,
                        roma_model=roma_model)
                    if result is not None:
                        ref_gx, ref_gy, off_gx, off_gy = _anchor_to_geo(
                            frow, fcol, result.dx_px, result.dy_px,
                            ref_transform, offset_transform,
                            shift_px_x, shift_py_y)
                        results.append((ref_gx, ref_gy, off_gx, off_gy,
                                       result.quality, f"anchor:{fname}"))
                        break
                else:
                    continue
                break

    results = _filter_anchor_displacement_outliers(results)
    return results
