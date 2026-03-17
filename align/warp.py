"""Grid-based DINOv3 optimization warping."""

import gc
import os
import traceback

import cv2
import numpy as np
import rasterio
import rasterio.transform
import torch
import torch.nn.functional as F
from affine import Affine
from osgeo import gdal
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter

from .constants import COAST_MAX_DIM, DINO_MAX_DIM, DEFAULT_PYRAMID_LEVELS, FB_CONSISTENCY_PX
from .flow_refine import (
    _estimate_flow,
    _forward_backward_mask,
    clamp_flow_magnitude,
)
from .grid_optim import optimize_grid_hierarchical
from .image import make_land_mask, to_u8, to_u8_percentile, chunked_remap


def _reproject_to_grid(src_ds, grid_transform, grid_w, grid_h, work_crs):
    """Reproject a rasterio dataset to a target grid, returning float32 array."""
    arr = np.zeros((grid_h, grid_w), dtype=np.float32)
    src_data = src_ds.read(1).astype(np.float32)
    reproject(source=src_data, destination=arr,
              src_transform=src_ds.transform, src_crs=src_ds.crs,
              dst_transform=grid_transform, dst_crs=work_crs,
              resampling=Resampling.bilinear)
    return arr


def _extract_coastline(land_mask_u8: np.ndarray, scale: float) -> np.ndarray:
    """Extract coastline points from a land mask, returning (N, 2) float32 array."""
    k = np.ones((3, 3), np.uint8)
    shore = cv2.morphologyEx(land_mask_u8, cv2.MORPH_GRADIENT, k) > 0
    y, x = np.where(shore)
    pts = np.column_stack((x, y)).astype(np.float32)
    if len(pts) > 10000:
        idx = np.linspace(0, len(pts) - 1, 10000).astype(np.int32)
        pts = pts[idx]
    if scale < 1.0:
        pts /= scale
    return pts


def _compute_reclamation_mask(
    target_u8_coast: np.ndarray,
    src_u8_coast: np.ndarray,
    coast_scale_t: float,
    coast_scale_s: float,
    output_res: float,
) -> np.ndarray | None:
    """Compute XOR-based reclamation mask between target and source land masks.

    Returns a bool mask at coastline resolution, or None on failure.
    """
    try:
        tgt_land = (make_land_mask(target_u8_coast) > 0).astype(np.uint8)
        if coast_scale_s != coast_scale_t:
            src_u8_coast_resized = cv2.resize(
                src_u8_coast, (target_u8_coast.shape[1], target_u8_coast.shape[0]),
                interpolation=cv2.INTER_AREA)
        else:
            src_u8_coast_resized = src_u8_coast
        src_land = (make_land_mask(src_u8_coast_resized) > 0).astype(np.uint8)
        xor_raw = (tgt_land ^ src_land).astype(np.uint8)

        # Morphological open to remove small scattered noise (alignment error)
        open_px = max(3, int(100.0 / (output_res / coast_scale_t)))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        xor_cleaned = cv2.morphologyEx(xor_raw, cv2.MORPH_OPEN, k_open)

        # Keep only large connected components (> min_area_px²)
        min_area_m2 = 300 * 300
        px_per_m = coast_scale_t / output_res
        min_area_px = max(50, int(min_area_m2 * px_per_m * px_per_m))
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(xor_cleaned, connectivity=8)
        xor_mask = np.zeros_like(xor_cleaned)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_area_px:
                xor_mask[labels == lbl] = 1

        dilate_px = max(1, int(50.0 / (output_res / coast_scale_t)))
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask = cv2.dilate(xor_mask, k_dilate, iterations=1).astype(bool)
        n_changed = int(mask.sum())
        pct_changed = 100.0 * n_changed / max(1, mask.size)
        raw_pct = 100.0 * int(xor_raw.sum()) / max(1, xor_raw.size)
        n_large = sum(1 for l in range(1, n_labels) if stats[l, cv2.CC_STAT_AREA] >= min_area_px)
        print(f"  [Reclamation] Raw XOR: {raw_pct:.1f}%, after cleaning: {pct_changed:.1f}% "
              f"({n_labels - 1} blobs, {n_large} large)", flush=True)
        return mask
    except Exception as e:
        print(f"  [Reclamation] WARNING: mask computation failed ({e}), proceeding without", flush=True)
        return None


def _reproject_to_output_crs(write_path: str, output_path: str, output_crs) -> None:
    """Reproject a work-CRS GeoTIFF to the final output CRS."""
    if os.path.exists(output_path):
        os.remove(output_path)
    print(f"  Reprojecting to {output_crs}...", flush=True)
    gdal.UseExceptions()
    try:
        reproject_ds = gdal.Warp(output_path, write_path,
                                 dstSRS=str(output_crs),
                                 resampleAlg=gdal.GRA_Bilinear,
                                 multithread=True,
                                 warpOptions=['NUM_THREADS=ALL_CPUS'],
                                 creationOptions=['COMPRESS=LZW', 'PREDICTOR=2',
                                                  'TILED=YES', 'NUM_THREADS=ALL_CPUS',
                                                  'BIGTIFF=YES'],
                                 warpMemoryLimit=2048 * 1024 * 1024)
        if reproject_ds is None:
            raise RuntimeError("Reprojection to output CRS failed")
        reproject_ds = None
    finally:
        if os.path.exists(write_path):
            os.remove(write_path)


def _load_target_features(reference_path, out_transform, out_w, out_h, work_crs):
    """Load reference image and extract coastline + DINOv3 features."""
    print("  Reading and extracting features for target (reference) image...", flush=True)
    with rasterio.open(reference_path) as src_ref:
        target_arr = _reproject_to_grid(src_ref, out_transform, out_w, out_h, work_crs)

    coast_scale_t = min(1.0, COAST_MAX_DIM / max(out_h, out_w))
    target_u8 = to_u8(target_arr)
    if coast_scale_t < 1.0:
        target_u8_coast = cv2.resize(target_u8, (0, 0), fx=coast_scale_t, fy=coast_scale_t)
    else:
        target_u8_coast = target_u8
    target_coast = _extract_coastline(
        (make_land_mask(target_u8_coast) > 0).astype(np.uint8), coast_scale_t)

    dino_scale_t = min(1.0, DINO_MAX_DIM / max(out_h, out_w))
    if dino_scale_t < 1.0:
        target_u8_dino = cv2.resize(target_u8, (0, 0), fx=dino_scale_t, fy=dino_scale_t)
    else:
        target_u8_dino = target_u8
    target_img_rgb = cv2.cvtColor(target_u8_dino, cv2.COLOR_GRAY2RGB)

    del target_arr, target_u8
    gc.collect()

    return target_coast, target_img_rgb, target_u8_coast, coast_scale_t


def _load_source_features(input_path):
    """Load source image and extract coastline + DINOv3 features.

    Returns (src_data, src_w, src_h, source_coast, src_img_rgb,
             src_u8_coast, coast_scale_s).
    """
    with rasterio.open(input_path) as src_ds:
        src_w, src_h = src_ds.width, src_ds.height
        print("  Reading source image...", flush=True)
        src_data = src_ds.read(1).astype(np.float32)

    coast_scale_s = min(1.0, COAST_MAX_DIM / max(src_h, src_w))
    src_u8 = to_u8(src_data)
    if coast_scale_s < 1.0:
        src_u8_coast = cv2.resize(src_u8, (0, 0), fx=coast_scale_s, fy=coast_scale_s)
    else:
        src_u8_coast = src_u8
    source_coast = _extract_coastline(
        (make_land_mask(src_u8_coast) > 0).astype(np.uint8), coast_scale_s)

    dino_scale_s = min(1.0, DINO_MAX_DIM / max(src_h, src_w))
    if dino_scale_s < 1.0:
        src_u8_dino = cv2.resize(src_u8, (0, 0), fx=dino_scale_s, fy=dino_scale_s)
    else:
        src_u8_dino = src_u8
    src_img_rgb = cv2.cvtColor(src_u8_dino, cv2.COLOR_GRAY2RGB)

    del src_u8
    gc.collect()

    return src_data, src_w, src_h, source_coast, src_img_rgb, src_u8_coast, coast_scale_s


def _prepare_gcps(gcps, out_transform, out_w, out_h, src_w, src_h,
                  n_real_gcps_in=None):
    """Project GCPs to target grid and add virtual boundary GCPs.

    Returns (source_pts, target_pts, n_real_gcps) as numpy arrays.
    """
    target_pts = []
    source_pts = []
    is_real = []
    inv_out_transform = ~out_transform  # type: ignore
    n_input_real = n_real_gcps_in if n_real_gcps_in is not None else len(gcps)
    for idx, (px, py, gx, gy) in enumerate(gcps):
        t_col, t_row = inv_out_transform * (gx, gy)
        if 0 <= t_col < out_w and 0 <= t_row < out_h:
            source_pts.append([px, py])
            target_pts.append([t_col, t_row])
            is_real.append(idx < n_input_real)

    source_pts = np.array(source_pts, dtype=np.float32)
    target_pts = np.array(target_pts, dtype=np.float32)
    n_real_gcps = sum(is_real)

    # Add virtual boundary GCPs along target grid edges where real GCPs
    # are sparse — anchors extrapolation to the affine baseline.
    if n_real_gcps >= 3:
        n = n_real_gcps
        A_aff = np.zeros((2 * n, 6), dtype=np.float32)
        b_aff = np.zeros(2 * n, dtype=np.float32)
        for i in range(n):
            tx, ty = target_pts[i]
            A_aff[2*i]   = [tx, ty, 1, 0, 0, 0]
            A_aff[2*i+1] = [0, 0, 0, tx, ty, 1]
            b_aff[2*i]   = source_pts[i, 0]
            b_aff[2*i+1] = source_pts[i, 1]
        aff_params, _, _, _ = np.linalg.lstsq(A_aff, b_aff, rcond=None)

        spacing_px = max(100, min(out_w, out_h) // 15)
        min_dist_px = spacing_px * 2.0
        edge_pts = []
        for x in np.arange(0, out_w, spacing_px):
            edge_pts.append((float(x), 0.0))
            edge_pts.append((float(x), float(out_h - 1)))
        for y in np.arange(spacing_px, out_h - spacing_px, spacing_px):
            edge_pts.append((0.0, float(y)))
            edge_pts.append((float(out_w - 1), float(y)))
        for c in [(0, 0), (out_w-1, 0), (0, out_h-1), (out_w-1, out_h-1)]:
            edge_pts.append((float(c[0]), float(c[1])))

        virtual_target = []
        virtual_source = []
        for tx, ty in edge_pts:
            dists = np.sqrt((target_pts[:, 0] - tx)**2 + (target_pts[:, 1] - ty)**2)
            if np.min(dists) > min_dist_px:
                sx = aff_params[0]*tx + aff_params[1]*ty + aff_params[2]
                sy = aff_params[3]*tx + aff_params[4]*ty + aff_params[5]
                if -src_w*0.1 < sx < src_w*1.1 and -src_h*0.1 < sy < src_h*1.1:
                    virtual_target.append([tx, ty])
                    virtual_source.append([sx, sy])

        if virtual_target:
            n_virtual = len(virtual_target)
            target_pts = np.vstack([target_pts, np.array(virtual_target, dtype=np.float32)])
            source_pts = np.vstack([source_pts, np.array(virtual_source, dtype=np.float32)])
            print(f"  Virtual boundary GCPs: {n_virtual} added (0.15x weight)", flush=True)

    return source_pts, target_pts, n_real_gcps


def _compute_flow_corrections(warper, reference_path, src_data, src_w, src_h,
                               output_bounds, output_res, out_w, out_h, work_crs):
    """Compute dense optical flow post-refinement (coarse-to-fine).

    Returns (flow_x_full, flow_y_full) or (None, None) on failure.
    """
    left, bottom, right, top = output_bounds
    fr_work_res = max(4.0, output_res)
    fr_scale = output_res / fr_work_res
    fr_w = max(64, int(out_w * fr_scale))
    fr_h = max(64, int(out_h * fr_scale))

    coarse_res = max(8.0, output_res)
    do_two_pass = coarse_res > fr_work_res * 1.3
    coarse_w = max(64, int(out_w * output_res / coarse_res)) if do_two_pass else fr_w
    coarse_h = max(64, int(out_h * output_res / coarse_res)) if do_two_pass else fr_h

    # Read reference at reduced resolution
    fr_transform = rasterio.transform.from_bounds(left, bottom, right, top, fr_w, fr_h)
    with rasterio.open(reference_path) as src_ref_fr:
        ref_arr_fr = _reproject_to_grid(src_ref_fr, fr_transform, fr_w, fr_h, work_crs)

    # Generate warped preview in-memory
    print("  [FlowRefine] Generating warped preview in-memory...", flush=True)
    with torch.no_grad():
        _fr_disp = warper.displacements.detach()
        _yr = np.linspace(-1.0, 1.0, fr_h)
        _xr = np.linspace(-1.0, 1.0, fr_w)
        _yg_fr, _xg_fr = np.meshgrid(_yr, _xr, indexing='ij')
        _gt_fr = torch.from_numpy(
            np.stack([_xg_fr, _yg_fr], axis=-1)
        ).float().unsqueeze(0).clamp(-1.0, 1.0).to(_fr_disp.device)
        _sd_fr = F.grid_sample(
            _fr_disp, _gt_fr, mode='bicubic',
            padding_mode='zeros', align_corners=True
        ).squeeze(0).cpu().numpy()

    _smx_fr = ((_xg_fr + _sd_fr[0] + 1.0) / 2.0 * (src_w - 1)).astype(np.float32)
    _smy_fr = ((_yg_fr + _sd_fr[1] + 1.0) / 2.0 * (src_h - 1)).astype(np.float32)
    warped_arr_fr = chunked_remap(src_data, _smx_fr, _smy_fr)
    del _sd_fr, _smx_fr, _smy_fr, _gt_fr, _xg_fr, _yg_fr

    ref_u8_fr = to_u8_percentile(ref_arr_fr)

    # ---- Coarse pass at ~8m/px ----
    flow_coarse_fine = np.zeros((fr_h, fr_w, 2), dtype=np.float32)
    if do_two_pass:
        print(f"  [FlowRefine] Coarse pass at {coarse_res:.0f}m/px "
              f"({coarse_w}x{coarse_h})...", flush=True)
        ref_arr_c = cv2.resize(ref_arr_fr, (coarse_w, coarse_h),
                               interpolation=cv2.INTER_AREA)
        warped_arr_c = cv2.resize(warped_arr_fr, (coarse_w, coarse_h),
                                  interpolation=cv2.INTER_AREA)
        ref_u8_c = to_u8_percentile(ref_arr_c)
        warped_u8_c = to_u8_percentile(warped_arr_c)

        flow_c = _estimate_flow(ref_u8_c, warped_u8_c)
        fb_mask_c = _forward_backward_mask(ref_u8_c, warped_u8_c, flow_c, FB_CONSISTENCY_PX)

        flow_c[:, :, 0] = cv2.medianBlur(flow_c[:, :, 0], 5)
        flow_c[:, :, 1] = cv2.medianBlur(flow_c[:, :, 1], 5)
        flow_c[~fb_mask_c] = 0
        valid_mask_c = (ref_arr_c > 0) & (warped_arr_c > 0)
        flow_c[~valid_mask_c] = 0

        flow_mag_c = clamp_flow_magnitude(flow_c, 75.0, coarse_res)

        reliable_c_pct = float(fb_mask_c.sum()) / max(1, fb_mask_c.size) * 100
        mean_corr_c = (float(np.mean(flow_mag_c[fb_mask_c])) * coarse_res
                       if fb_mask_c.any() else 0.0)
        print(f"  [FlowRefine] Coarse: {reliable_c_pct:.0f}% reliable, "
              f"mean {mean_corr_c:.1f}m, "
              f"max {float(flow_mag_c.max()) * coarse_res:.1f}m", flush=True)

        if reliable_c_pct < 5.0:
            print(f"  [FlowRefine] Coarse pass only {reliable_c_pct:.0f}% reliable "
                  f"— skipping flow entirely", flush=True)
            raise RuntimeError("Insufficient reliable coarse flow pixels")
        elif reliable_c_pct < 10.0:
            print(f"  [FlowRefine] Marginal coarse reliability ({reliable_c_pct:.0f}%) "
                  f"— applying smoothed flow", flush=True)
            fb_float = fb_mask_c.astype(np.float64)
            flow_c[:, :, 0] = gaussian_filter(
                flow_c[:, :, 0] * fb_float, sigma=15
            ) / np.maximum(gaussian_filter(fb_float, sigma=15), 1e-6)
            flow_c[:, :, 1] = gaussian_filter(
                flow_c[:, :, 1] * fb_float, sigma=15
            ) / np.maximum(gaussian_filter(fb_float, sigma=15), 1e-6)

        px_ratio = coarse_res / fr_work_res
        flow_coarse_fine[:, :, 0] = cv2.resize(
            flow_c[:, :, 0], (fr_w, fr_h),
            interpolation=cv2.INTER_LINEAR) * px_ratio
        flow_coarse_fine[:, :, 1] = cv2.resize(
            flow_c[:, :, 1], (fr_w, fr_h),
            interpolation=cv2.INTER_LINEAR) * px_ratio

        # Apply coarse correction to warped preview for fine pass
        ys_grid = np.arange(fr_h, dtype=np.float32)
        xs_grid = np.arange(fr_w, dtype=np.float32)
        xg_remap, yg_remap = np.meshgrid(xs_grid, ys_grid)
        map_x = (xg_remap + flow_coarse_fine[:, :, 0]).astype(np.float32)
        map_y = (yg_remap + flow_coarse_fine[:, :, 1]).astype(np.float32)
        warped_arr_fr = cv2.remap(warped_arr_fr, map_x, map_y,
                                   cv2.INTER_LINEAR, borderValue=0)

        del ref_arr_c, warped_arr_c, ref_u8_c, warped_u8_c
        del flow_c, fb_mask_c, valid_mask_c
        del map_x, map_y, xg_remap, yg_remap

    # ---- Fine pass at ~4m/px ----
    print(f"  [FlowRefine] Fine pass at {fr_work_res:.0f}m/px "
          f"({fr_w}x{fr_h})...", flush=True)
    warped_u8_fr = to_u8_percentile(warped_arr_fr)

    flow_fwd = _estimate_flow(ref_u8_fr, warped_u8_fr)
    fb_mask = _forward_backward_mask(ref_u8_fr, warped_u8_fr, flow_fwd, FB_CONSISTENCY_PX)

    flow_fwd[:, :, 0] = cv2.medianBlur(flow_fwd[:, :, 0], 5)
    flow_fwd[:, :, 1] = cv2.medianBlur(flow_fwd[:, :, 1], 5)

    if fb_mask.any():
        reliable_dx = flow_fwd[fb_mask, 0]
        reliable_dy = flow_fwd[fb_mask, 1]
        median_dx = float(np.median(reliable_dx))
        median_dy = float(np.median(reliable_dy))
        bias_m_dx = abs(median_dx) * fr_work_res
        bias_m_dy = abs(median_dy) * fr_work_res
        if bias_m_dx > 5.0 or bias_m_dy > 5.0:
            print(f"  [FlowRefine] Fine bias: "
                  f"dx={median_dx * fr_work_res:+.1f}m, "
                  f"dy={median_dy * fr_work_res:+.1f}m"
                  f" (NOT subtracted)", flush=True)

    flow_fwd[~fb_mask] = 0
    valid_mask_fr = (ref_arr_fr > 0) & (warped_arr_fr > 0)
    flow_fwd[~valid_mask_fr] = 0

    fine_clamp_m = 30.0 if do_two_pass else 75.0
    flow_mag = clamp_flow_magnitude(flow_fwd, fine_clamp_m, fr_work_res)

    reliable_pct = float(fb_mask.sum()) / max(1, fb_mask.size) * 100
    mean_corr = float(np.mean(flow_mag[fb_mask])) * fr_work_res if fb_mask.any() else 0.0
    print(f"  [FlowRefine] Fine: {reliable_pct:.0f}% reliable, "
          f"mean correction {mean_corr:.1f}m, "
          f"max {float(flow_mag.max()) * fr_work_res:.1f}m", flush=True)

    if reliable_pct < 10.0 and not do_two_pass:
        print(f"  [FlowRefine] Only {reliable_pct:.0f}% reliable — skipping",
              flush=True)
        raise RuntimeError("Insufficient reliable flow pixels")

    if reliable_pct < 10.0 and do_two_pass:
        print(f"  [FlowRefine] Fine pass only {reliable_pct:.0f}% reliable "
              f"— using coarse flow only", flush=True)
        flow_fwd[:] = 0

    del ref_arr_fr, warped_arr_fr, ref_u8_fr, warped_u8_fr, fb_mask, valid_mask_fr

    # Combine coarse + fine flow
    flow_total_x = flow_coarse_fine[:, :, 0] + flow_fwd[:, :, 0]
    flow_total_y = flow_coarse_fine[:, :, 1] + flow_fwd[:, :, 1]
    del flow_coarse_fine, flow_fwd

    # Final clamp on combined flow
    combined_clamp_m = 100.0 if do_two_pass else 75.0
    max_px_total = combined_clamp_m / fr_work_res
    flow_mag_total = np.sqrt(flow_total_x**2 + flow_total_y**2)
    too_large_t = flow_mag_total > max_px_total
    if np.any(too_large_t):
        clip_t = np.where(too_large_t, max_px_total / (flow_mag_total + 1e-8), 1.0)
        flow_total_x *= clip_t
        flow_total_y *= clip_t

    total_mean_m = (float(np.mean(flow_mag_total[flow_mag_total > 0])) * fr_work_res
                    if np.any(flow_mag_total > 0) else 0.0)
    if do_two_pass:
        print(f"  [FlowRefine] Combined: mean {total_mean_m:.1f}m, "
              f"max {float(flow_mag_total.max()) * fr_work_res:.1f}m", flush=True)

    # Upsample to full output resolution
    flow_x_full = cv2.resize(flow_total_x, (out_w, out_h),
                              interpolation=cv2.INTER_LINEAR) / fr_scale
    flow_y_full = cv2.resize(flow_total_y, (out_w, out_h),
                              interpolation=cv2.INTER_LINEAR) / fr_scale
    del flow_total_x, flow_total_y
    gc.collect()
    print("  [FlowRefine] Flow computed, will merge into single remap pass", flush=True)
    print("  [FlowRefine] Post-refinement complete", flush=True)
    return flow_x_full, flow_y_full


def _remap_and_write(write_path, warper, src_data, src_w, src_h, out_w, out_h,
                     work_crs, out_transform, flow_x_full=None, flow_y_full=None):
    """Single-pass remap: grid displacement (+ flow if available)."""
    has_flow = flow_x_full is not None and flow_y_full is not None
    label = "grid+flow" if has_flow else "grid-only"
    print(f"  Applying {label} remap...", flush=True)

    estimated_size = out_w * out_h * 4
    bigtiff = 'YES' if estimated_size > 3_000_000_000 else 'NO'

    with rasterio.open(write_path, 'w', driver='GTiff',
                       width=out_w, height=out_h, count=1,
                       dtype='float32', crs=work_crs,
                       transform=out_transform,
                       compress='lzw', predictor=2, tiled=True,
                       nodata=0, BIGTIFF=bigtiff) as dst:

        strip_height = min(2048, out_h)
        n_strips = (out_h + strip_height - 1) // strip_height

        with torch.no_grad():
            displacements = warper.displacements.detach()

        for strip_idx in range(n_strips):
            row_start = strip_idx * strip_height
            row_end = min(row_start + strip_height, out_h)
            strip_h = row_end - row_start

            y_coords = np.linspace(row_start, row_end - 1, strip_h)
            x_coords = np.linspace(0, out_w - 1, out_w)
            y_norm = (y_coords / (out_h - 1)) * 2.0 - 1.0
            x_norm = (x_coords / (out_w - 1)) * 2.0 - 1.0
            y_grid, x_grid = np.meshgrid(y_norm, x_norm, indexing='ij')

            grid_tensor = torch.from_numpy(
                np.stack([x_grid, y_grid], axis=-1)
            ).float().unsqueeze(0).to(displacements.device).clamp(-1.0, 1.0)

            with torch.no_grad():
                strip_disp = F.grid_sample(
                    displacements, grid_tensor, mode='bicubic',
                    padding_mode='zeros', align_corners=True
                ).squeeze(0).cpu().numpy()

            strip_map_x = ((x_grid + strip_disp[0] + 1.0) / 2.0 * (src_w - 1)).astype(np.float32)
            strip_map_y = ((y_grid + strip_disp[1] + 1.0) / 2.0 * (src_h - 1)).astype(np.float32)

            if has_flow:
                strip_map_x += flow_x_full[row_start:row_end]
                strip_map_y += flow_y_full[row_start:row_end]

            strip_data = chunked_remap(src_data, strip_map_x, strip_map_y)

            nodata_mask = ((strip_map_x < 0) | (strip_map_x >= src_w - 1) |
                           (strip_map_y < 0) | (strip_map_y >= src_h - 1))
            strip_data[nodata_mask] = 0

            dst.write(strip_data, indexes=1, window=((row_start, row_end), (0, out_w)))

            if (strip_idx + 1) % 5 == 0 or strip_idx == n_strips - 1:
                print(f"    Warp strip {strip_idx + 1}/{n_strips} complete", flush=True)

    print(f"  {label.capitalize()} remap complete", flush=True)


def apply_warp(
    input_path: str,
    output_path: str,
    reference_path: str,
    gcps: list[tuple[float, float, float, float]],
    work_crs,
    output_bounds: tuple[float, float, float, float],
    output_res: float,
    output_crs=None,
    grid_size: int = 20,
    grid_iters: int = 300,
    arap_weight: float = 1.0,
    n_real_gcps_in: int | None = None,
) -> str:
    """Run DINOv3 grid optimization and warp.

    output_bounds: (left, bottom, right, top) in work_crs to constrain extent.
    output_res: pixel size in work_crs units (metres for projected CRS).
    output_crs: final output CRS (default: same as work_crs). If different,
                warps to work_crs first, then reprojects to output_crs.
    """
    needs_reproject = (output_crs is not None and str(output_crs) != str(work_crs))
    write_path = output_path + ".grid_tmp.tif" if needs_reproject else output_path

    if os.path.exists(write_path):
        os.remove(write_path)

    left, bottom, right, top = output_bounds
    out_w = int(round((right - left) / output_res))
    out_h = int(round((top - bottom) / output_res))
    out_transform: Affine = rasterio.transform.from_bounds(left, bottom, right, top, out_w, out_h) # type: ignore
    print(f"  Target grid: {out_w} x {out_h} px at {output_res:.2f} m/px", flush=True)

    # 1. Load target and source features
    target_coast, target_img_rgb, target_u8_coast, coast_scale_t = \
        _load_target_features(reference_path, out_transform, out_w, out_h, work_crs)

    src_data, src_w, src_h, source_coast, src_img_rgb, src_u8_coast, coast_scale_s = \
        _load_source_features(input_path)

    # 2. Prepare GCPs (real + virtual boundary)
    source_pts, target_pts, n_real_gcps = _prepare_gcps(
        gcps, out_transform, out_w, out_h, src_w, src_h, n_real_gcps_in)

    # 3. Compute reclamation mask
    reclamation_mask = _compute_reclamation_mask(
        target_u8_coast, src_u8_coast, coast_scale_t, coast_scale_s, output_res)

    # 4. Run hierarchical grid optimizer
    warper = optimize_grid_hierarchical(
        source_img=src_img_rgb,
        target_img=target_img_rgb,
        source_pts=source_pts,
        target_pts=target_pts,
        source_coast=source_coast,
        target_coast=target_coast,
        src_shape=(src_h, src_w),
        tgt_shape=(out_h, out_w),
        output_res_m=output_res,
        levels=DEFAULT_PYRAMID_LEVELS,
        w_arap=arap_weight,
        w_feat=2.0,
        reclamation_mask_tgt=reclamation_mask,
        n_real_gcps=n_real_gcps,
    )

    # 5. Dense optical flow post-refinement
    flow_x_full = None
    flow_y_full = None
    try:
        print("  [FlowRefine] Starting dense optical flow post-refinement (coarse-to-fine)...", flush=True)
        flow_x_full, flow_y_full = _compute_flow_corrections(
            warper, reference_path, src_data, src_w, src_h,
            output_bounds, output_res, out_w, out_h, work_crs)
    except Exception as e:
        traceback.print_exc()
        print(f"  [FlowRefine] WARNING: Flow refinement failed ({e}), "
              f"keeping grid-only result", flush=True)
        flow_x_full = None
        flow_y_full = None

    # 6. Single-pass remap to disk
    _remap_and_write(write_path, warper, src_data, src_w, src_h, out_w, out_h,
                     work_crs, out_transform, flow_x_full, flow_y_full)

    if flow_x_full is not None:
        del flow_x_full, flow_y_full
    gc.collect()

    # 7. Reproject to output CRS if needed
    if needs_reproject:
        _reproject_to_output_crs(write_path, output_path, output_crs)

    return output_path
