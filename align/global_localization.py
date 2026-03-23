"""Coarse global localization against the full reference image."""

from __future__ import annotations

import math
import os
import tempfile

import cv2
import numpy as np
import rasterio
import rasterio.transform
from affine import Affine
from rasterio.warp import Resampling, reproject, transform_bounds

from .geo import dataset_bounds_in_crs, transform_point, work_shift_to_dataset_shift
from .image import clahe_normalize, to_u8
from .semantic_masking import class_weight_map, stable_feature_mask
from .types import GlobalHypothesis, MetadataPrior


def _adaptive_res(bounds, requested_res, max_pixels=4096):
    width_m = max(1.0, bounds[2] - bounds[0])
    height_m = max(1.0, bounds[3] - bounds[1])
    return max(float(requested_res), width_m / max_pixels, height_m / max_pixels)


def _read_raw_resized(src, bounds, target_res):
    """Read a non-georeferenced image and resize to match expected bounds/res.

    For raw scans with no CRS, reads band 1 and resizes to the pixel
    dimensions implied by the prior bounds and target resolution.
    """
    left, bottom, right, top = bounds
    width = max(1, int(round((right - left) / target_res)))
    height = max(1, int(round((top - bottom) / target_res)))
    raw = src.read(1).astype(np.float32)
    if raw.shape[0] != height or raw.shape[1] != width:
        raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_AREA)
    return raw


def _read_bounds(src, bounds, target_crs, target_res):
    left, bottom, right, top = bounds
    width = max(1, int(round((right - left) / target_res)))
    height = max(1, int(round((top - bottom) / target_res)))
    dst_transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)
    dst_array = np.zeros((height, width), dtype=np.float32)
    reproject(
        source=rasterio.band(src, 1),
        destination=dst_array,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear,
    )
    return dst_array, dst_transform


def _prior_bounds_in_work_crs(prior: MetadataPrior, work_crs):
    bounds = prior.bounds()
    if bounds is None:
        return None
    west, south, east, north = bounds
    return transform_bounds(prior.crs, work_crs, west, south, east, north)


def _expand_bounds(bounds, margin_m, clamp_bounds=None):
    left, bottom, right, top = bounds
    expanded = (left - margin_m, bottom - margin_m, right + margin_m, top + margin_m)
    if clamp_bounds is None:
        return expanded
    return (
        max(clamp_bounds[0], expanded[0]),
        max(clamp_bounds[1], expanded[1]),
        min(clamp_bounds[2], expanded[2]),
        min(clamp_bounds[3], expanded[3]),
    )


def _prepare_search_bounds(ref_bounds, priors, work_crs, explicit_bounds=None):
    if explicit_bounds is not None:
        return explicit_bounds
    candidate = None
    for prior in priors or []:
        bounds = _prior_bounds_in_work_crs(prior, work_crs)
        if bounds is None:
            continue
        margin = max((bounds[2] - bounds[0]) * 0.5, (bounds[3] - bounds[1]) * 0.5, 5000.0)
        bounds = _expand_bounds(bounds, margin, clamp_bounds=ref_bounds)
        if candidate is None:
            candidate = bounds
        else:
            candidate = (
                max(ref_bounds[0], min(candidate[0], bounds[0])),
                max(ref_bounds[1], min(candidate[1], bounds[1])),
                min(ref_bounds[2], max(candidate[2], bounds[2])),
                min(ref_bounds[3], max(candidate[3], bounds[3])),
            )
    return candidate or ref_bounds


def _transform_template(arr, scale, angle_deg):
    src = to_u8(arr)
    if abs(scale - 1.0) > 1e-3:
        new_w = max(32, int(round(src.shape[1] * scale)))
        new_h = max(32, int(round(src.shape[0] * scale)))
        src = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
    if abs(angle_deg) > 1e-3:
        h, w = src.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        cos_v = abs(M[0, 0])
        sin_v = abs(M[0, 1])
        new_w = int((h * sin_v) + (w * cos_v))
        new_h = int((h * cos_v) + (w * sin_v))
        M[0, 2] += (new_w / 2.0) - center[0]
        M[1, 2] += (new_h / 2.0) - center[1]
        src = cv2.warpAffine(src, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return src


def localize_to_reference(src_offset, src_ref, work_crs, priors=None, coarse_res=40.0,
                          top_k=3, mask_mode="coastal_obia", search_bounds=None):
    """Search the full reference image for the target footprint."""

    priors = priors or []
    ref_bounds = dataset_bounds_in_crs(src_ref, work_crs)
    search_bounds = _prepare_search_bounds(ref_bounds, priors, work_crs, explicit_bounds=search_bounds)
    ref_res = _adaptive_res(search_bounds, coarse_res, max_pixels=4096)
    ref_arr, _ = _read_bounds(src_ref, search_bounds, work_crs, ref_res)

    target_bounds = dataset_bounds_in_crs(src_offset, work_crs)
    if target_bounds is None:
        # Image has no CRS — fall back to metadata prior bounds
        for prior in priors:
            target_bounds = _prior_bounds_in_work_crs(prior, work_crs)
            if target_bounds is not None:
                break
    if target_bounds is None:
        return []
    target_res = max(ref_res, _adaptive_res(target_bounds, coarse_res, max_pixels=1024))
    if src_offset.crs is not None:
        target_arr, _ = _read_bounds(src_offset, target_bounds, work_crs, target_res)
    else:
        # Raw scan — read pixels directly and resize to expected dimensions.
        # The prior bounds are approximate (~10 mile accuracy per USGS docs),
        # which is fine for coarse localization template matching.
        target_arr = _read_raw_resized(src_offset, target_bounds, target_res)

    if min(target_arr.shape[:2]) < 32 or min(ref_arr.shape[:2]) < 32:
        return []

    target_weight = _transform_template(class_weight_map(target_arr, mode=mask_mode), 1.0, 0.0)
    target_stable = _transform_template(stable_feature_mask(target_arr, mode=mask_mode), 1.0, 0.0)
    ref_weight = class_weight_map(ref_arr, mode=mask_mode).astype(np.float32)
    ref_stable = stable_feature_mask(ref_arr, mode=mask_mode).astype(np.float32)
    ref_grad = clahe_normalize(ref_arr).astype(np.float32)
    target_grad_base = clahe_normalize(target_arr).astype(np.float32)

    angles = [-8.0, -4.0, 0.0, 4.0, 8.0]
    scales = [0.9, 1.0, 1.1]
    candidates = []

    for scale in scales:
        for angle in angles:
            templ_weight = _transform_template(target_weight, scale, angle).astype(np.float32)
            templ_stable = _transform_template(target_stable, scale, angle).astype(np.float32)
            templ_grad = _transform_template(target_grad_base, scale, angle).astype(np.float32)
            if templ_weight.shape[0] >= ref_weight.shape[0] or templ_weight.shape[1] >= ref_weight.shape[1]:
                continue
            if np.mean(templ_weight > 0) < 0.05 and np.mean(templ_stable > 0) < 0.03:
                continue

            result_weight = cv2.matchTemplate(ref_weight, templ_weight, cv2.TM_CCOEFF_NORMED)
            result_stable = cv2.matchTemplate(ref_stable, templ_stable, cv2.TM_CCOEFF_NORMED)
            result_grad = cv2.matchTemplate(ref_grad, templ_grad, cv2.TM_CCOEFF_NORMED)
            combined = (0.45 * result_weight) + (0.35 * result_stable) + (0.20 * result_grad)
            _, max_val, _, max_loc = cv2.minMaxLoc(combined)
            candidates.append((float(max_val), max_loc, scale, angle, templ_weight.shape[1], templ_weight.shape[0]))

    candidates.sort(key=lambda item: item[0], reverse=True)
    hypotheses = []
    target_center_x = (target_bounds[0] + target_bounds[2]) / 2.0
    target_center_y = (target_bounds[1] + target_bounds[3]) / 2.0
    ref_full_bounds = dataset_bounds_in_crs(src_ref, work_crs)
    for idx, (score, (c_px, r_px), scale, angle, templ_w, templ_h) in enumerate(candidates[:top_k]):
        left = search_bounds[0] + (c_px * ref_res)
        top = search_bounds[3] - (r_px * ref_res)
        right = left + (templ_w * ref_res)
        bottom = top - (templ_h * ref_res)
        margin = max((right - left) * 0.20, (top - bottom) * 0.20, 2000.0)
        left, bottom, right, top = _expand_bounds((left, bottom, right, top), margin, clamp_bounds=ref_full_bounds)
        center_x = (left + right) / 2.0
        center_y = (bottom + top) / 2.0
        hypotheses.append(
            GlobalHypothesis(
                hypothesis_id=f"global_{idx}",
                score=score,
                source="global_template_search",
                left=left,
                bottom=bottom,
                right=right,
                top=top,
                dx_m=target_center_x - center_x,
                dy_m=center_y - target_center_y,
                scale_hint=float(scale),
                rotation_hint_deg=float(angle),
                work_crs=str(work_crs),
                diagnostics={
                    "search_res": ref_res,
                    "target_res": target_res,
                    "raw_score": score,
                },
            )
        )
    return hypotheses


def translate_input_to_hypothesis(input_path, hypothesis: GlobalHypothesis, work_crs, *, suffix=".global_shift.tif"):
    """Create a temporary copy of *input_path* with a translated georeference."""

    src = rasterio.open(input_path)
    dx_src, dy_src = work_shift_to_dataset_shift(src, work_crs, hypothesis.dx_m, hypothesis.dy_m)
    profile = src.profile.copy()
    transform = src.transform
    profile["transform"] = Affine(
        transform.a,
        transform.b,
        transform.c + dx_src,
        transform.d,
        transform.e,
        transform.f + dy_src,
    )
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=os.path.dirname(input_path) or ".")
    os.close(tmp_fd)

    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.update_tags(**src.tags())
        for band_idx in range(1, src.count + 1):
            for _, window in src.block_windows(band_idx):
                data = src.read(band_idx, window=window)
                dst.write(data, band_idx, window=window)

    src.close()
    return tmp_path
