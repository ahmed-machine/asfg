"""Grid-based DINOv2 optimization warping."""

import os
import cv2
import numpy as np
import rasterio
import rasterio.transform
from osgeo import gdal
from rasterio.warp import reproject, Resampling

from .grid_optim import optimize_grid, optimize_grid_hierarchical
from .flow_refine import dense_flow_refinement
from .image import make_land_mask, to_u8


_REMAP_MAX = 30000

def _chunked_remap(src_data, map_x, map_y):
    """cv2.remap with automatic chunking for images exceeding SHRT_MAX.

    Splits the destination into column chunks and crops the source per-chunk
    so both src and dst stay under the 32767 pixel limit in each dimension.
    """
    src_h, src_w = src_data.shape
    dst_h, dst_w = map_x.shape

    # Check if we can use cv2.remap directly
    if src_h <= _REMAP_MAX and src_w <= _REMAP_MAX and dst_h <= _REMAP_MAX and dst_w <= _REMAP_MAX:
        return cv2.remap(src_data, map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Split destination into column chunks
    result = np.zeros((dst_h, dst_w), dtype=np.float32)
    n_col_chunks = (dst_w + _REMAP_MAX - 1) // _REMAP_MAX

    from scipy.ndimage import map_coordinates

    for ci in range(n_col_chunks):
        c0 = ci * _REMAP_MAX
        c1 = min(c0 + _REMAP_MAX, dst_w)
        chunk_map_x = map_x[:, c0:c1]
        chunk_map_y = map_y[:, c0:c1]

        # Find source bounding box for this chunk
        valid = (chunk_map_x >= 0) & (chunk_map_y >= 0)
        if not np.any(valid):
            continue

        src_x_min = max(0, int(np.floor(chunk_map_x[valid].min())) - 2)
        src_x_max = min(src_w, int(np.ceil(chunk_map_x[valid].max())) + 3)
        src_y_min = max(0, int(np.floor(chunk_map_y[valid].min())) - 2)
        src_y_max = min(src_h, int(np.ceil(chunk_map_y[valid].max())) + 3)

        crop_w = src_x_max - src_x_min
        crop_h = src_y_max - src_y_min

        if crop_w <= 0 or crop_h <= 0:
            continue

        # Crop source and adjust map coordinates
        src_crop = src_data[src_y_min:src_y_max, src_x_min:src_x_max]
        adj_map_x = chunk_map_x - src_x_min
        adj_map_y = chunk_map_y - src_y_min

        if crop_h <= _REMAP_MAX and crop_w <= _REMAP_MAX:
            chunk_result = cv2.remap(src_crop, adj_map_x, adj_map_y,
                                     interpolation=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            # Source crop still too large - use scipy fallback
            coords = np.array([adj_map_y.astype(np.float64),
                               adj_map_x.astype(np.float64)])
            chunk_result = map_coordinates(src_crop, coords,
                                           order=1, mode='constant', cval=0.0,
                                           prefilter=False).astype(np.float32)

        result[:, c0:c1] = chunk_result

    return result


def _reproject_to_grid(src_ds, grid_transform, grid_w, grid_h, work_crs):
    """Reproject a rasterio dataset to a target grid, returning float32 array."""
    arr = np.zeros((grid_h, grid_w), dtype=np.float32)
    src_data = src_ds.read(1).astype(np.float32)
    reproject(source=src_data, destination=arr,
              src_transform=src_ds.transform, src_crs=src_ds.crs,
              dst_transform=grid_transform, dst_crs=work_crs,
              resampling=Resampling.bilinear)
    return arr


def _icp_coastline_refine(write_path, reference_path, out_transform,
                          out_w, out_h, work_crs, output_res,
                          max_iters=20, max_correction_m=30.0):
    """ICP-based coastline alignment refinement (Improvement 4).

    Extracts coastline points from the warped output and reference,
    filters out tidally-unstable segments, runs point-to-point ICP,
    and applies a thin correction that fades with distance from coast.
    """
    # Read both images directly at a reduced resolution for coastline extraction
    icp_res = max(8.0, output_res)
    scale = output_res / icp_res
    icp_w = max(64, int(out_w * scale))
    icp_h = max(64, int(out_h * scale))

    left_b, bottom_b, right_b, top_b = rasterio.transform.array_bounds(
        out_h, out_w, out_transform)
    icp_transform = rasterio.transform.from_bounds(
        left_b, bottom_b, right_b, top_b, icp_w, icp_h)

    src_ref = rasterio.open(reference_path)
    ref_arr_icp = _reproject_to_grid(src_ref, icp_transform, icp_w, icp_h, work_crs)
    src_ref.close()

    # Read warped output at reduced resolution using GDAL overviews
    with rasterio.open(write_path, 'r') as ds:
        out_full = ds.read(1, out_shape=(icp_h, icp_w),
                           resampling=Resampling.bilinear).astype(np.float32)

    ref_small = to_u8(ref_arr_icp)
    out_small = to_u8(out_full)
    del ref_arr_icp, out_full

    # Extract coastlines using morphological gradient
    k = np.ones((3, 3), np.uint8)

    ref_land = (make_land_mask(ref_small) > 0).astype(np.uint8)
    out_land = (make_land_mask(out_small) > 0).astype(np.uint8)

    # Morphological stability filter: keep coastline points that persist
    # under ±1px erosion/dilation (filters out tidal flats)
    ref_eroded = cv2.erode(ref_land, k, iterations=1)
    ref_dilated = cv2.dilate(ref_land, k, iterations=1)
    ref_stable = (ref_eroded == ref_dilated)

    out_eroded = cv2.erode(out_land, k, iterations=1)
    out_dilated = cv2.dilate(out_land, k, iterations=1)
    out_stable = (out_eroded == out_dilated)

    ref_coast = cv2.morphologyEx(ref_land, cv2.MORPH_GRADIENT, k) > 0
    out_coast = cv2.morphologyEx(out_land, cv2.MORPH_GRADIENT, k) > 0

    # Filter to stable coastline segments only
    ref_coast = ref_coast & ref_stable
    out_coast = out_coast & out_stable

    ref_pts = np.column_stack(np.where(ref_coast)).astype(np.float32)  # (N, 2) as (row, col)
    out_pts = np.column_stack(np.where(out_coast)).astype(np.float32)

    if len(ref_pts) < 50 or len(out_pts) < 50:
        print("  [ICP] Too few stable coastline points, skipping", flush=True)
        return

    # Subsample for speed
    if len(ref_pts) > 5000:
        idx = np.linspace(0, len(ref_pts) - 1, 5000).astype(int)
        ref_pts = ref_pts[idx]
    if len(out_pts) > 5000:
        idx = np.linspace(0, len(out_pts) - 1, 5000).astype(int)
        out_pts = out_pts[idx]

    # Simple point-to-point ICP: find translation + rotation that aligns out_pts to ref_pts
    from scipy.spatial import cKDTree

    best_T = np.eye(3, dtype=np.float64)
    moving = out_pts.copy()

    for icp_iter in range(max_iters):
        tree = cKDTree(ref_pts)
        dists, idxs = tree.query(moving, k=1)

        # Robust: reject outliers > 100m / icp_res pixels
        max_dist_px = 100.0 / icp_res
        inlier = dists < max_dist_px
        if np.sum(inlier) < 20:
            break

        src = moving[inlier]
        tgt = ref_pts[idxs[inlier]]

        # Compute rigid transform (translation + rotation)
        src_mean = src.mean(axis=0)
        tgt_mean = tgt.mean(axis=0)
        src_c = src - src_mean
        tgt_c = tgt - tgt_mean

        H = src_c.T @ tgt_c
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = tgt_mean - R @ src_mean

        # Apply to moving points
        moving = (R @ moving.T).T + t

        # Accumulate transform
        T_step = np.eye(3)
        T_step[:2, :2] = R
        T_step[:2, 2] = t
        best_T = T_step @ best_T

        # Check convergence
        delta = np.sqrt(t[0]**2 + t[1]**2)
        if delta < 0.1:  # sub-pixel convergence
            break

    # Extract final translation in pixels (at icp_res)
    tx_px = best_T[0, 2]  # col shift
    ty_px = best_T[1, 2]  # row shift
    tx_m = tx_px * icp_res
    ty_m = ty_px * icp_res

    if abs(tx_m) > max_correction_m or abs(ty_m) > max_correction_m:
        print(f"  [ICP] Correction too large ({tx_m:.1f}m, {ty_m:.1f}m), skipping", flush=True)
        return

    if abs(tx_m) < 0.5 and abs(ty_m) < 0.5:
        print(f"  [ICP] Negligible correction ({tx_m:.2f}m, {ty_m:.2f}m), skipping", flush=True)
        return

    print(f"  [ICP] Coastline correction: dx={tx_m:.1f}m, dy={ty_m:.1f}m", flush=True)

    # Create a distance-from-coast weight map: corrections fade to zero inland
    # Use the reference coastline mask for distance computation
    coast_dist = cv2.distanceTransform((~ref_coast).astype(np.uint8), cv2.DIST_L2, 3)
    fade_dist_px = 2000.0 / icp_res  # fade over 2km
    coast_weight = np.clip(1.0 - coast_dist / fade_dist_px, 0.0, 1.0).astype(np.float32)

    # Apply weighted shift strip by strip to avoid OOM
    shift_x_px = tx_m / output_res
    shift_y_px = ty_m / output_res

    icp_strip_h = min(2048, out_h)
    icp_n_strips = (out_h + icp_strip_h - 1) // icp_strip_h

    with rasterio.open(write_path, 'r+') as ds:
        for si in range(icp_n_strips):
            rs = si * icp_strip_h
            re = min(rs + icp_strip_h, out_h)
            sh = re - rs

            strip_data = ds.read(1, window=((rs, re), (0, out_w))).astype(np.float32)

            # Compute coast weight for this strip (from ICP-res coast_weight)
            strip_cw = cv2.resize(
                coast_weight,
                (out_w, sh),
                interpolation=cv2.INTER_LINEAR,
            )
            # Actually we need the correct rows from coast_weight. Let's interpolate properly.
            # Map strip rows to icp_h rows
            strip_y_frac_start = rs / max(1, out_h - 1)
            strip_y_frac_end = (re - 1) / max(1, out_h - 1)
            cw_r0 = int(strip_y_frac_start * (icp_h - 1))
            cw_r1 = min(icp_h, int(strip_y_frac_end * (icp_h - 1)) + 2)
            cw_slice = coast_weight[cw_r0:cw_r1]
            strip_cw = cv2.resize(cw_slice, (out_w, sh), interpolation=cv2.INTER_LINEAR)

            y_coords = np.arange(rs, re, dtype=np.float32).reshape(-1, 1)
            y_coords = np.broadcast_to(y_coords, (sh, out_w)).copy()
            x_coords = np.arange(0, out_w, dtype=np.float32).reshape(1, -1)
            x_coords = np.broadcast_to(x_coords, (sh, out_w)).copy()

            corr_mx = (x_coords - shift_x_px * strip_cw).astype(np.float32)
            corr_my = (y_coords - shift_y_px * strip_cw).astype(np.float32)
            # Adjust to strip-local coordinates for remap
            corr_my -= rs

            corrected = cv2.remap(strip_data, corr_mx, corr_my,
                                  interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            ds.write(corrected, indexes=1, window=((rs, re), (0, out_w)))

    print(f"  [ICP] Coastline refinement applied", flush=True)


def apply_warp(input_path, output_path, reference_path, gcps, work_crs,
               output_bounds, output_res, output_crs=None,
               grid_size=20, grid_iters=300, arap_weight=1.0):
    """Run DINOv2 grid optimization and warp.

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
    from affine import Affine
    out_transform: Affine = rasterio.transform.from_bounds(left, bottom, right, top, out_w, out_h) # type: ignore

    print(f"  Target grid: {out_w} x {out_h} px at {output_res:.2f} m/px", flush=True)

    # 1. Load reference image on target grid
    src_ref = rasterio.open(reference_path)
    print("  Reading and extracting features for target (reference) image...", flush=True)
    target_arr = _reproject_to_grid(src_ref, out_transform, out_w, out_h, work_crs)
    src_ref.close()

    def extract_coastline(land_mask_u8, scale):
        k = np.ones((3, 3), np.uint8)
        shore = cv2.morphologyEx(land_mask_u8, cv2.MORPH_GRADIENT, k) > 0
        y, x = np.where(shore)
        pts = np.column_stack((x, y)).astype(np.float32)
        # Deterministically sub-sample to keep optimisation fast
        if len(pts) > 10000:
            idx = np.linspace(0, len(pts) - 1, 10000).astype(np.int32)
            pts = pts[idx]
        # Scale back to full-resolution coordinates
        if scale < 1.0:
            pts /= scale
        return pts

    # Coastline extraction at higher resolution preserves narrow coastal
    # features (causeways, peninsulas) that disappear at 2048px.
    coast_max_dim = 4096
    coast_scale_t = min(1.0, coast_max_dim / max(out_h, out_w))
    target_u8 = to_u8(target_arr)
    if coast_scale_t < 1.0:
        target_u8_coast = cv2.resize(target_u8, (0, 0), fx=coast_scale_t, fy=coast_scale_t)
    else:
        target_u8_coast = target_u8
    target_coast = extract_coastline(
        (make_land_mask(target_u8_coast) > 0).astype(np.uint8), coast_scale_t
    )

    # High-res downsample for DINOv2 (~50 m feature cells at 16384 px)
    dino_max_dim = 16384
    dino_scale_t = min(1.0, dino_max_dim / max(out_h, out_w))
    if dino_scale_t < 1.0:
        target_u8_dino = cv2.resize(target_u8, (0, 0), fx=dino_scale_t, fy=dino_scale_t)
    else:
        target_u8_dino = target_u8
    target_img_rgb = cv2.cvtColor(target_u8_dino, cv2.COLOR_GRAY2RGB)

    # Free memory
    del target_arr
    del target_u8
    import gc
    gc.collect()

    # 2. Load source image (pre-aligned by affine/scale pass)
    src_ds = rasterio.open(input_path)
    src_w, src_h = src_ds.width, src_ds.height
    print("  Reading source image...", flush=True)
    src_data = src_ds.read(1).astype(np.float32)

    coast_scale_s = min(1.0, coast_max_dim / max(src_h, src_w))
    src_u8 = to_u8(src_data)
    if coast_scale_s < 1.0:
        src_u8_coast = cv2.resize(src_u8, (0, 0), fx=coast_scale_s, fy=coast_scale_s)
    else:
        src_u8_coast = src_u8
    source_coast = extract_coastline(
        (make_land_mask(src_u8_coast) > 0).astype(np.uint8), coast_scale_s
    )

    dino_scale_s = min(1.0, dino_max_dim / max(src_h, src_w))
    if dino_scale_s < 1.0:
        src_u8_dino = cv2.resize(src_u8, (0, 0), fx=dino_scale_s, fy=dino_scale_s)
    else:
        src_u8_dino = src_u8
    src_img_rgb = cv2.cvtColor(src_u8_dino, cv2.COLOR_GRAY2RGB)

    del src_u8
    gc.collect()

    # 4. Process GCPs
    target_pts = []
    source_pts = []
    inv_out_transform = ~out_transform  # type: ignore
    for px, py, gx, gy in gcps:
        t_col, t_row = inv_out_transform * (gx, gy)
        # Only keep points strictly within the target grid
        if 0 <= t_col < out_w and 0 <= t_row < out_h:
            source_pts.append([px, py])
            target_pts.append([t_col, t_row])

    source_pts = np.array(source_pts, dtype=np.float32)
    target_pts = np.array(target_pts, dtype=np.float32)

    # 4b. Compute reclamation mask (XOR of land masks, dilated)
    # Only large coherent XOR blobs are real reclamation; scattered small
    # differences are alignment noise and should NOT be masked.
    reclamation_mask = None
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
        min_area_m2 = 500 * 500  # 500m × 500m minimum reclamation area
        px_per_m = coast_scale_t / output_res
        min_area_px = max(50, int(min_area_m2 * px_per_m * px_per_m))
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(xor_cleaned, connectivity=8)
        xor_mask = np.zeros_like(xor_cleaned)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_area_px:
                xor_mask[labels == lbl] = 1

        # Dilate by ~100m at coastline resolution (reduced from 200m to
        # preserve more coast points; the soft-clamp chamfer handles the rest)
        dilate_px = max(1, int(100.0 / (output_res / coast_scale_t)))
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        reclamation_mask = cv2.dilate(xor_mask, k_dilate, iterations=1).astype(bool)
        n_changed = int(reclamation_mask.sum())
        pct_changed = 100.0 * n_changed / max(1, reclamation_mask.size)
        raw_pct = 100.0 * int(xor_raw.sum()) / max(1, xor_raw.size)
        print(f"  [Reclamation] Raw XOR: {raw_pct:.1f}%, after cleaning: {pct_changed:.1f}% "
              f"({n_labels - 1} blobs, {sum(1 for l in range(1, n_labels) if stats[l, cv2.CC_STAT_AREA] >= min_area_px)} large)",
              flush=True)
        del tgt_land, src_land, xor_raw, xor_cleaned, xor_mask
    except Exception as e:
        print(f"  [Reclamation] WARNING: mask computation failed ({e}), proceeding without", flush=True)
        reclamation_mask = None

    # 5. Run hierarchical grid optimizer (coarse-to-fine pyramid)
    # levels: (grid_size, iters) — coarse captures global warp,
    # fine captures local distortion (~250m node spacing at 64×64)
    pyramid_levels = [(8, 200), (24, 200), (64, 200)]
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
        levels=pyramid_levels,
        w_arap=arap_weight,
        reclamation_mask_tgt=reclamation_mask,
    )

    print("  Applying chunked remap...", flush=True)
    
    # 6. Apply remap and write to output
    estimated_size = out_w * out_h * 4
    bigtiff = 'YES' if estimated_size > 3_000_000_000 else 'NO'

    import torch
    import torch.nn.functional as F

    with rasterio.open(write_path, 'w', driver='GTiff',
                       width=out_w, height=out_h, count=1,
                       dtype='float32', crs=work_crs,
                       transform=out_transform,
                       compress='lzw', predictor=2, tiled=True,
                       nodata=0, BIGTIFF=bigtiff) as dst:
        
        strip_height = min(2048, out_h)
        n_strips = (out_h + strip_height - 1) // strip_height
        
        # Displacements are on target domain [-1, 1]
        with torch.no_grad():
            displacements = warper.displacements.detach() # (1, 2, grid_h, grid_w)

        for strip_idx in range(n_strips):
            row_start = strip_idx * strip_height
            row_end = min(row_start + strip_height, out_h)
            strip_h = row_end - row_start
            
            # Create target pixel coordinates for the strip
            y_coords = np.linspace(row_start, row_end - 1, strip_h)
            x_coords = np.linspace(0, out_w - 1, out_w)
            
            # Normalize to [-1, 1]
            y_norm = (y_coords / (out_h - 1)) * 2.0 - 1.0
            x_norm = (x_coords / (out_w - 1)) * 2.0 - 1.0
            
            y_grid, x_grid = np.meshgrid(y_norm, x_norm, indexing='ij')
            
            grid_tensor = torch.from_numpy(np.stack([x_grid, y_grid], axis=-1)).float().unsqueeze(0).to(warper.displacements.device)
            grid_tensor = grid_tensor.clamp(-1.0, 1.0)

            with torch.no_grad():
                strip_disp = F.grid_sample(displacements, grid_tensor, mode='bilinear', padding_mode='zeros', align_corners=True)
                strip_disp = strip_disp.squeeze(0).cpu().numpy() # (2, strip_h, out_w)
            
            src_x_norm = x_grid + strip_disp[0]
            src_y_norm = y_grid + strip_disp[1]
            
            strip_map_x = ((src_x_norm + 1.0) / 2.0) * (src_w - 1)
            strip_map_y = ((src_y_norm + 1.0) / 2.0) * (src_h - 1)
            
            strip_map_x = strip_map_x.astype(np.float32)
            strip_map_y = strip_map_y.astype(np.float32)
            
            strip_data = _chunked_remap(src_data, strip_map_x, strip_map_y)
            
            nodata_mask = ((strip_map_x < 0) | (strip_map_x >= src_w - 1) |
                           (strip_map_y < 0) | (strip_map_y >= src_h - 1))
            strip_data[nodata_mask] = 0
            
            dst.write(strip_data, indexes=1, window=((row_start, row_end), (0, out_w)))
            
            if (strip_idx + 1) % 5 == 0 or strip_idx == n_strips - 1:
                print(f"    Warp strip {strip_idx + 1}/{n_strips} complete", flush=True)

    src_ds.close()
    print("  Grid deformation warp complete", flush=True)

    # 6b. Dense optical flow post-refinement (Improvement 2)
    # Work at a reduced resolution (~4m/px) to keep memory reasonable,
    # then upsample the flow correction to full res for strip-based re-remap.
    try:
        print("  [FlowRefine] Starting dense optical flow post-refinement...", flush=True)
        # Work resolution: ~4m/px (or output_res if coarser)
        fr_work_res = max(4.0, output_res)
        fr_scale = output_res / fr_work_res
        fr_w = max(64, int(out_w * fr_scale))
        fr_h = max(64, int(out_h * fr_scale))

        # Read reference at reduced resolution
        from affine import Affine as _Affine
        fr_transform = rasterio.transform.from_bounds(
            left, bottom, right, top, fr_w, fr_h)
        src_ref_fr = rasterio.open(reference_path)
        ref_arr_fr = _reproject_to_grid(src_ref_fr, fr_transform, fr_w, fr_h, work_crs)
        src_ref_fr.close()

        # Read warped output at reduced resolution
        with rasterio.open(write_path, 'r') as warped_ds:
            warped_full = warped_ds.read(1).astype(np.float32)
        warped_arr_fr = cv2.resize(warped_full, (fr_w, fr_h), interpolation=cv2.INTER_AREA)
        del warped_full

        # Compute flow at reduced resolution
        from .flow_refine import _estimate_flow_dis, _forward_backward_mask

        def _to_u8(arr):
            valid = arr[arr > 0]
            if len(valid) == 0:
                return np.zeros_like(arr, dtype=np.uint8)
            lo, hi = np.percentile(valid, [1, 99])
            if hi <= lo:
                return np.zeros_like(arr, dtype=np.uint8)
            return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

        ref_u8_fr = _to_u8(ref_arr_fr)
        warped_u8_fr = _to_u8(warped_arr_fr)

        flow_fwd = _estimate_flow_dis(ref_u8_fr, warped_u8_fr)
        fb_mask = _forward_backward_mask(ref_u8_fr, warped_u8_fr, flow_fwd, 3.0)

        # Median filter
        flow_fwd[:, :, 0] = cv2.medianBlur(flow_fwd[:, :, 0], 5)
        flow_fwd[:, :, 1] = cv2.medianBlur(flow_fwd[:, :, 1], 5)

        # Subtract systematic flow bias before zeroing unreliable pixels.
        # Cap at 15m/axis: larger biases are usually legitimate warp corrections,
        # not DIS artifacts.
        MAX_BIAS_M = 15.0
        if fb_mask.any():
            reliable_dx = flow_fwd[fb_mask, 0]
            reliable_dy = flow_fwd[fb_mask, 1]
            median_dx = float(np.median(reliable_dx))
            median_dy = float(np.median(reliable_dy))
            bias_m_dx = abs(median_dx) * fr_work_res
            bias_m_dy = abs(median_dy) * fr_work_res
            if bias_m_dx > 5.0 or bias_m_dy > 5.0:
                max_bias_px = MAX_BIAS_M / fr_work_res
                capped_dx = float(np.clip(median_dx, -max_bias_px, max_bias_px))
                capped_dy = float(np.clip(median_dy, -max_bias_px, max_bias_px))
                flow_fwd[:, :, 0] -= capped_dx
                flow_fwd[:, :, 1] -= capped_dy
                print(f"  [FlowRefine] Subtracted median bias: "
                      f"dx={median_dx * fr_work_res:+.1f}m, dy={median_dy * fr_work_res:+.1f}m"
                      f" (capped to dx={capped_dx * fr_work_res:+.1f}m, dy={capped_dy * fr_work_res:+.1f}m)",
                      flush=True)

        flow_fwd[~fb_mask] = 0

        # Zero where no data
        valid_mask_fr = (ref_arr_fr > 0) & (warped_arr_fr > 0)
        flow_fwd[~valid_mask_fr] = 0

        # Clamp corrections
        max_px = 50.0 / fr_work_res
        flow_mag = np.sqrt(flow_fwd[:, :, 0]**2 + flow_fwd[:, :, 1]**2)
        too_large = flow_mag > max_px
        if np.any(too_large):
            clip_factor = np.where(too_large, max_px / (flow_mag + 1e-8), 1.0)
            flow_fwd[:, :, 0] *= clip_factor
            flow_fwd[:, :, 1] *= clip_factor
            flow_mag = np.clip(flow_mag, 0, max_px)

        reliable_pct = float(fb_mask.sum()) / max(1, fb_mask.size) * 100
        mean_corr = float(np.mean(flow_mag[fb_mask])) * fr_work_res if fb_mask.any() else 0.0
        print(f"  [FlowRefine] {reliable_pct:.0f}% reliable, "
              f"mean correction {mean_corr:.1f}m, "
              f"max {float(flow_mag.max()) * fr_work_res:.1f}m", flush=True)

        # Skip if too few reliable pixels — flow will do more harm than good.
        # Also skip if reliability is marginal AND mean correction is large,
        # which indicates the flow is making big dubious corrections.
        if reliable_pct < 10.0:
            print(f"  [FlowRefine] Only {reliable_pct:.0f}% reliable (mean corr {mean_corr:.1f}m) — skipping "
                  f"(threshold 10%)", flush=True)
            del ref_arr_fr, warped_arr_fr, ref_u8_fr, warped_u8_fr
            del fb_mask, valid_mask_fr, flow_fwd
            gc.collect()
            raise RuntimeError("Insufficient reliable flow pixels")

        del ref_arr_fr, warped_arr_fr, ref_u8_fr, warped_u8_fr, fb_mask, valid_mask_fr

        # Upsample flow to full output resolution
        flow_x_full = cv2.resize(flow_fwd[:, :, 0], (out_w, out_h),
                                  interpolation=cv2.INTER_LINEAR) / fr_scale
        flow_y_full = cv2.resize(flow_fwd[:, :, 1], (out_w, out_h),
                                  interpolation=cv2.INTER_LINEAR) / fr_scale
        del flow_fwd
        gc.collect()

        # Re-read source for re-remap
        src_ds_fr = rasterio.open(input_path)
        src_data_fr = src_ds_fr.read(1).astype(np.float32)
        src_w_fr, src_h_fr = src_ds_fr.width, src_ds_fr.height

        # Re-remap strip by strip: recompute grid remap + add flow correction
        print("  [FlowRefine] Re-applying remap with flow corrections...", flush=True)
        with torch.no_grad():
            fr_disp = warper.displacements.detach()

        with rasterio.open(write_path, 'r+') as dst_fr:
            fr_strip_h = min(2048, out_h)
            fr_n_strips = (out_h + fr_strip_h - 1) // fr_strip_h
            for si in range(fr_n_strips):
                rs = si * fr_strip_h
                re = min(rs + fr_strip_h, out_h)
                sh = re - rs

                # Recompute grid remap for this strip
                yc = np.linspace(rs, re - 1, sh)
                xc = np.linspace(0, out_w - 1, out_w)
                yn = (yc / (out_h - 1)) * 2.0 - 1.0
                xn = (xc / (out_w - 1)) * 2.0 - 1.0
                yg, xg = np.meshgrid(yn, xn, indexing='ij')
                gt = torch.from_numpy(np.stack([xg, yg], axis=-1)).float().unsqueeze(0).clamp(-1.0, 1.0).to(fr_disp.device)
                with torch.no_grad():
                    sd = F.grid_sample(fr_disp, gt, mode='bilinear',
                                       padding_mode='zeros', align_corners=True)
                    sd = sd.squeeze(0).cpu().numpy()

                smx = ((xg + sd[0] + 1.0) / 2.0 * (src_w_fr - 1)).astype(np.float32)
                smy = ((yg + sd[1] + 1.0) / 2.0 * (src_h_fr - 1)).astype(np.float32)

                # Add flow correction (in source pixel units)
                smx += flow_x_full[rs:re]
                smy += flow_y_full[rs:re]

                strip_data = _chunked_remap(src_data_fr, smx, smy)
                nodata = ((smx < 0) | (smx >= src_w_fr - 1) |
                          (smy < 0) | (smy >= src_h_fr - 1))
                strip_data[nodata] = 0
                dst_fr.write(strip_data, indexes=1, window=((rs, re), (0, out_w)))

                if (si + 1) % 5 == 0 or si == fr_n_strips - 1:
                    print(f"    FlowRefine strip {si + 1}/{fr_n_strips} complete", flush=True)

        src_ds_fr.close()
        del flow_x_full, flow_y_full, src_data_fr
        gc.collect()
        print("  [FlowRefine] Post-refinement complete", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [FlowRefine] WARNING: Flow refinement failed ({e}), "
              f"keeping grid-only result", flush=True)

    # 6c. ICP coastline refinement (Improvement 4)
    try:
        print("  [ICP] Starting coastline ICP refinement...", flush=True)
        _icp_coastline_refine(write_path, reference_path, out_transform,
                              out_w, out_h, work_crs, output_res)
    except Exception as e:
        print(f"  [ICP] WARNING: Coastline ICP failed ({e}), keeping result", flush=True)

    # 7. Reproject to output CRS if needed
    if needs_reproject:
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

    return output_path
