"""Grid-based DINOv2 optimization warping."""

import os
import cv2
import numpy as np
import rasterio
import rasterio.transform
from osgeo import gdal
from rasterio.warp import reproject, Resampling

from .grid_optim import optimize_grid_hierarchical
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

    ref_coast = cv2.morphologyEx(ref_land, cv2.MORPH_GRADIENT, k) > 0
    out_coast = cv2.morphologyEx(out_land, cv2.MORPH_GRADIENT, k) > 0

    # Mutual agreement filter: keep coastline points where both images
    # agree on nearby land (filters noise without the old erode==dilate bug)
    ref_land_d = cv2.dilate(ref_land, k, iterations=2)
    out_land_d = cv2.dilate(out_land, k, iterations=2)
    ref_coast = ref_coast & (out_land_d > 0)
    out_coast = out_coast & (ref_land_d > 0)

    ref_pts = np.column_stack(np.where(ref_coast)).astype(np.float32)  # (N, 2) as (row, col)
    out_pts = np.column_stack(np.where(out_coast)).astype(np.float32)

    print(f"  [ICP] Coast points: ref={len(ref_pts)}, out={len(out_pts)}", flush=True)

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
    fade_dist_px = 500.0 / icp_res  # fade over 500m (coastline-local only)
    coast_weight = np.clip(1.0 - coast_dist / fade_dist_px, 0.0, 1.0).astype(np.float32)

    # Apply weighted shift tile by tile to avoid OOM and cv2.remap SHRT_MAX limit
    shift_x_px = tx_m / output_res
    shift_y_px = ty_m / output_res
    pad = int(max(abs(shift_x_px), abs(shift_y_px))) + 2  # margin for remap

    MAX_TILE = 16384  # well under SHRT_MAX (32767)
    tile_h = min(MAX_TILE, out_h)
    tile_w = min(MAX_TILE, out_w)
    n_strips_y = (out_h + tile_h - 1) // tile_h
    n_strips_x = (out_w + tile_w - 1) // tile_w

    with rasterio.open(write_path, 'r+') as ds:
        for si in range(n_strips_y):
            rs = si * tile_h
            re = min(rs + tile_h, out_h)
            sh = re - rs
            for sj in range(n_strips_x):
                cs = sj * tile_w
                ce = min(cs + tile_w, out_w)
                sw = ce - cs

                # Read with padding for remap boundary pixels
                rs_p = max(0, rs - pad)
                re_p = min(out_h, re + pad)
                cs_p = max(0, cs - pad)
                ce_p = min(out_w, ce + pad)
                tile_data = ds.read(1, window=((rs_p, re_p), (cs_p, ce_p))).astype(np.float32)

                # Offsets of the actual tile within the padded read
                oy = rs - rs_p
                ox = cs - cs_p

                # Interpolate coast weight for this tile
                y_frac_s = rs / max(1, out_h - 1)
                y_frac_e = (re - 1) / max(1, out_h - 1)
                x_frac_s = cs / max(1, out_w - 1)
                x_frac_e = (ce - 1) / max(1, out_w - 1)
                cw_r0 = int(y_frac_s * (icp_h - 1))
                cw_r1 = min(icp_h, int(y_frac_e * (icp_h - 1)) + 2)
                cw_c0 = int(x_frac_s * (icp_w - 1))
                cw_c1 = min(icp_w, int(x_frac_e * (icp_w - 1)) + 2)
                cw_slice = coast_weight[cw_r0:cw_r1, cw_c0:cw_c1]
                tile_cw = cv2.resize(cw_slice, (sw, sh), interpolation=cv2.INTER_LINEAR)

                # Build remap coords in padded-tile space
                y_coords = np.arange(sh, dtype=np.float32).reshape(-1, 1) + oy
                y_coords = np.broadcast_to(y_coords, (sh, sw)).copy()
                x_coords = np.arange(sw, dtype=np.float32).reshape(1, -1) + ox
                x_coords = np.broadcast_to(x_coords, (sh, sw)).copy()

                corr_mx = (x_coords - shift_x_px * tile_cw).astype(np.float32)
                corr_my = (y_coords - shift_y_px * tile_cw).astype(np.float32)

                corrected = cv2.remap(tile_data, corr_mx, corr_my,
                                      interpolation=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                ds.write(corrected, indexes=1, window=((rs, re), (cs, ce)))

    print(f"  [ICP] Coastline refinement applied", flush=True)


def apply_warp(input_path, output_path, reference_path, gcps, work_crs,
               output_bounds, output_res, output_crs=None,
               grid_size=20, grid_iters=300, arap_weight=1.0,
               n_real_gcps_in=None):
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

    # 4. Process GCPs — track which are real vs pipeline-boundary
    target_pts = []
    source_pts = []
    is_real = []
    inv_out_transform = ~out_transform  # type: ignore
    n_input_real = n_real_gcps_in if n_real_gcps_in is not None else len(gcps)
    for idx, (px, py, gx, gy) in enumerate(gcps):
        t_col, t_row = inv_out_transform * (gx, gy)
        # Only keep points strictly within the target grid
        if 0 <= t_col < out_w and 0 <= t_row < out_h:
            source_pts.append([px, py])
            target_pts.append([t_col, t_row])
            is_real.append(idx < n_input_real)

    source_pts = np.array(source_pts, dtype=np.float32)
    target_pts = np.array(target_pts, dtype=np.float32)
    n_real_gcps = sum(is_real)

    # 4a. Add virtual boundary GCPs along target grid edges where real GCPs
    # are sparse.  These anchor extrapolation to the affine baseline, preventing
    # wild divergence in data-sparse regions (e.g., east edge).
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
        min_dist_px = spacing_px * 2.0  # only add where truly no GCP coverage
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
        min_area_m2 = 300 * 300  # 300m × 300m minimum reclamation area
        px_per_m = coast_scale_t / output_res
        min_area_px = max(50, int(min_area_m2 * px_per_m * px_per_m))
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(xor_cleaned, connectivity=8)
        xor_mask = np.zeros_like(xor_cleaned)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_area_px:
                xor_mask[labels == lbl] = 1

        # Dilate by ~50m at coastline resolution (reduced from 100m to
        # preserve more coast points for chamfer alignment)
        dilate_px = max(1, int(50.0 / (output_res / coast_scale_t)))
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
    # Note: 4th level at 96×96 caused flow refinement write failures (v21)
    pyramid_levels = [(8, 300), (24, 300), (64, 500)]
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
        n_real_gcps=n_real_gcps,
    )

    import torch
    import torch.nn.functional as F

    # ------------------------------------------------------------------
    # 6b. Dense optical flow post-refinement (single-pass optimisation)
    #
    # Instead of writing grid-only remap to disk, reading it back for
    # flow estimation, and re-remapping, we:
    #   1. Generate a small warped preview in-memory via the grid
    #      displacement (no disk I/O).
    #   2. Compute optical flow at reduced resolution (~16 m/px).
    #   3. Upsample flow and combine with grid displacement in a
    #      single remap pass that writes the final output.
    # This eliminates the first full-res remap + disk write, halving
    # the I/O and saving several minutes on large images.
    # ------------------------------------------------------------------
    flow_x_full = None
    flow_y_full = None
    try:
        print("  [FlowRefine] Starting dense optical flow post-refinement (coarse-to-fine)...", flush=True)
        # Fine resolution: ~4m/px (preserves sub-pixel corrections)
        fr_work_res = max(4.0, output_res)
        fr_scale = output_res / fr_work_res
        fr_w = max(64, int(out_w * fr_scale))
        fr_h = max(64, int(out_h * fr_scale))

        # Coarse resolution: ~8m/px (captures large-scale corrections robustly)
        coarse_res = max(8.0, output_res)
        do_two_pass = coarse_res > fr_work_res * 1.3  # only if meaningfully different
        coarse_w = max(64, int(out_w * output_res / coarse_res)) if do_two_pass else fr_w
        coarse_h = max(64, int(out_h * output_res / coarse_res)) if do_two_pass else fr_h

        # Read reference at reduced resolution
        fr_transform = rasterio.transform.from_bounds(
            left, bottom, right, top, fr_w, fr_h)
        src_ref_fr = rasterio.open(reference_path)
        ref_arr_fr = _reproject_to_grid(src_ref_fr, fr_transform, fr_w, fr_h, work_crs)
        src_ref_fr.close()

        # Generate warped preview in-memory at flow resolution by applying
        # the grid displacement to the source image directly (avoids a
        # full-res remap → disk → read-back cycle).
        print("  [FlowRefine] Generating warped preview in-memory...", flush=True)
        with torch.no_grad():
            _fr_disp = warper.displacements.detach()
            # Sample displacement at flow resolution
            _yr = np.linspace(-1.0, 1.0, fr_h)
            _xr = np.linspace(-1.0, 1.0, fr_w)
            _yg_fr, _xg_fr = np.meshgrid(_yr, _xr, indexing='ij')
            _gt_fr = torch.from_numpy(
                np.stack([_xg_fr, _yg_fr], axis=-1)
            ).float().unsqueeze(0).clamp(-1.0, 1.0).to(_fr_disp.device)
            _sd_fr = F.grid_sample(
                _fr_disp, _gt_fr, mode='bicubic',
                padding_mode='zeros', align_corners=True
            ).squeeze(0).cpu().numpy()  # (2, fr_h, fr_w)

        _smx_fr = ((_xg_fr + _sd_fr[0] + 1.0) / 2.0 * (src_w - 1)).astype(np.float32)
        _smy_fr = ((_yg_fr + _sd_fr[1] + 1.0) / 2.0 * (src_h - 1)).astype(np.float32)
        warped_arr_fr = _chunked_remap(src_data, _smx_fr, _smy_fr)
        del _sd_fr, _smx_fr, _smy_fr, _gt_fr, _xg_fr, _yg_fr

        # Two-pass coarse-to-fine flow refinement:
        # Pass 1 (coarse, ~8m/px): captures large-scale corrections robustly
        # Pass 2 (fine, ~4m/px): captures sub-pixel residuals on corrected image
        from .flow_refine import _estimate_flow, _forward_backward_mask

        def _to_u8(arr):
            valid = arr[arr > 0]
            if len(valid) == 0:
                return np.zeros_like(arr, dtype=np.uint8)
            lo, hi = np.percentile(valid, [1, 99])
            if hi <= lo:
                return np.zeros_like(arr, dtype=np.uint8)
            return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

        ref_u8_fr = _to_u8(ref_arr_fr)

        # ---- Coarse pass at ~8m/px ----
        flow_coarse_fine = np.zeros((fr_h, fr_w, 2), dtype=np.float32)
        if do_two_pass:
            print(f"  [FlowRefine] Coarse pass at {coarse_res:.0f}m/px "
                  f"({coarse_w}x{coarse_h})...", flush=True)
            ref_arr_c = cv2.resize(ref_arr_fr, (coarse_w, coarse_h),
                                   interpolation=cv2.INTER_AREA)
            warped_arr_c = cv2.resize(warped_arr_fr, (coarse_w, coarse_h),
                                      interpolation=cv2.INTER_AREA)
            ref_u8_c = _to_u8(ref_arr_c)
            warped_u8_c = _to_u8(warped_arr_c)

            flow_c = _estimate_flow(ref_u8_c, warped_u8_c)
            fb_mask_c = _forward_backward_mask(ref_u8_c, warped_u8_c, flow_c, 3.0)

            flow_c[:, :, 0] = cv2.medianBlur(flow_c[:, :, 0], 5)
            flow_c[:, :, 1] = cv2.medianBlur(flow_c[:, :, 1], 5)
            flow_c[~fb_mask_c] = 0
            valid_mask_c = (ref_arr_c > 0) & (warped_arr_c > 0)
            flow_c[~valid_mask_c] = 0

            # Clamp coarse corrections at 75m
            max_px_c = 75.0 / coarse_res
            flow_mag_c = np.sqrt(flow_c[:, :, 0]**2 + flow_c[:, :, 1]**2)
            too_large_c = flow_mag_c > max_px_c
            if np.any(too_large_c):
                clip_c = np.where(too_large_c, max_px_c / (flow_mag_c + 1e-8), 1.0)
                flow_c[:, :, 0] *= clip_c
                flow_c[:, :, 1] *= clip_c

            reliable_c_pct = float(fb_mask_c.sum()) / max(1, fb_mask_c.size) * 100
            mean_corr_c = (float(np.mean(flow_mag_c[fb_mask_c])) * coarse_res
                           if fb_mask_c.any() else 0.0)
            print(f"  [FlowRefine] Coarse: {reliable_c_pct:.0f}% reliable, "
                  f"mean {mean_corr_c:.1f}m, "
                  f"max {float(flow_mag_c.max()) * coarse_res:.1f}m", flush=True)

            if reliable_c_pct < 10.0:
                print(f"  [FlowRefine] Coarse pass only {reliable_c_pct:.0f}% reliable "
                      f"— skipping flow entirely", flush=True)
                del ref_arr_c, warped_arr_c, ref_u8_c, warped_u8_c
                del flow_c, fb_mask_c, valid_mask_c
                del ref_arr_fr, warped_arr_fr, ref_u8_fr
                gc.collect()
                raise RuntimeError("Insufficient reliable coarse flow pixels")

            # Upsample coarse flow to fine resolution (convert pixel units)
            px_ratio = coarse_res / fr_work_res  # e.g. 8/4 = 2.0
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
        warped_u8_fr = _to_u8(warped_arr_fr)

        flow_fwd = _estimate_flow(ref_u8_fr, warped_u8_fr)
        fb_mask = _forward_backward_mask(ref_u8_fr, warped_u8_fr, flow_fwd, 3.0)

        # Median filter to remove salt-and-pepper noise
        flow_fwd[:, :, 0] = cv2.medianBlur(flow_fwd[:, :, 0], 5)
        flow_fwd[:, :, 1] = cv2.medianBlur(flow_fwd[:, :, 1], 5)

        # Log systematic flow bias
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

        # Clamp fine corrections (tighter cap when coarse already handled large offsets)
        fine_clamp_m = 30.0 if do_two_pass else 75.0
        max_px = fine_clamp_m / fr_work_res
        flow_mag = np.sqrt(flow_fwd[:, :, 0]**2 + flow_fwd[:, :, 1]**2)
        too_large = flow_mag > max_px
        if np.any(too_large):
            clip_factor = np.where(too_large, max_px / (flow_mag + 1e-8), 1.0)
            flow_fwd[:, :, 0] *= clip_factor
            flow_fwd[:, :, 1] *= clip_factor
            flow_mag = np.clip(flow_mag, 0, max_px)

        reliable_pct = float(fb_mask.sum()) / max(1, fb_mask.size) * 100
        mean_corr = float(np.mean(flow_mag[fb_mask])) * fr_work_res if fb_mask.any() else 0.0
        print(f"  [FlowRefine] Fine: {reliable_pct:.0f}% reliable, "
              f"mean correction {mean_corr:.1f}m, "
              f"max {float(flow_mag.max()) * fr_work_res:.1f}m", flush=True)

        # If fine pass has low reliability in single-pass mode, skip entirely.
        # In two-pass mode, we still have the coarse flow.
        if reliable_pct < 10.0 and not do_two_pass:
            print(f"  [FlowRefine] Only {reliable_pct:.0f}% reliable — skipping",
                  flush=True)
            del ref_arr_fr, warped_arr_fr, ref_u8_fr, warped_u8_fr
            del fb_mask, valid_mask_fr, flow_fwd
            gc.collect()
            raise RuntimeError("Insufficient reliable flow pixels")

        if reliable_pct < 10.0 and do_two_pass:
            print(f"  [FlowRefine] Fine pass only {reliable_pct:.0f}% reliable "
                  f"— using coarse flow only", flush=True)
            flow_fwd[:] = 0  # zero out unreliable fine contribution

        del ref_arr_fr, warped_arr_fr, ref_u8_fr, warped_u8_fr, fb_mask, valid_mask_fr

        # Combine coarse + fine flow (both in fine-pixel units)
        flow_total_x = flow_coarse_fine[:, :, 0] + flow_fwd[:, :, 0]
        flow_total_y = flow_coarse_fine[:, :, 1] + flow_fwd[:, :, 1]
        del flow_coarse_fine, flow_fwd

        # Final clamp on combined flow — must accommodate coarse (75m) + fine (30m)
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
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [FlowRefine] WARNING: Flow refinement failed ({e}), "
              f"keeping grid-only result", flush=True)
        flow_x_full = None
        flow_y_full = None

    # ------------------------------------------------------------------
    # 6c. Single-pass remap: grid displacement (+ flow if available)
    # This replaces the old two-pass approach (grid-only write, then
    # flow re-remap) with one combined write to disk.
    # ------------------------------------------------------------------
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
            displacements = warper.displacements.detach()  # (1, 2, grid_h, grid_w)

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
                ).squeeze(0).cpu().numpy()  # (2, strip_h, out_w)

            strip_map_x = ((x_grid + strip_disp[0] + 1.0) / 2.0 * (src_w - 1)).astype(np.float32)
            strip_map_y = ((y_grid + strip_disp[1] + 1.0) / 2.0 * (src_h - 1)).astype(np.float32)

            # Add flow correction when available (flow is in source pixel units)
            if has_flow:
                strip_map_x += flow_x_full[row_start:row_end]
                strip_map_y += flow_y_full[row_start:row_end]

            strip_data = _chunked_remap(src_data, strip_map_x, strip_map_y)

            nodata_mask = ((strip_map_x < 0) | (strip_map_x >= src_w - 1) |
                           (strip_map_y < 0) | (strip_map_y >= src_h - 1))
            strip_data[nodata_mask] = 0

            dst.write(strip_data, indexes=1, window=((row_start, row_end), (0, out_w)))

            if (strip_idx + 1) % 5 == 0 or strip_idx == n_strips - 1:
                print(f"    Warp strip {strip_idx + 1}/{n_strips} complete", flush=True)

    src_ds.close()
    if has_flow:
        del flow_x_full, flow_y_full
    gc.collect()
    print(f"  {label.capitalize()} remap complete", flush=True)

    # 6d. ICP coastline refinement (Improvement 4)
    # ICP coastline refinement disabled — cross-temporal coastline changes
    # (reclamation) produce spurious shifts that damage alignment (v37: -17m,
    # v45: -27.9m).  The fixed ICP code is preserved above for future use
    # when reclamation masking is integrated into the ICP pipeline.
    print("  [ICP] Skipped (disabled — cross-temporal coastline changes)", flush=True)

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
