"""Automatic anchor GCP generation via coarse RoMa matching.

Generates spatially-distributed anchor GCPs by matching a georeferenced
frame against the reference image at coarse resolution.  The anchors are
written in the same JSON format as hand-curated anchor files so the
alignment pipeline's existing anchor infrastructure can consume them
directly.
"""

import json
import os

import cv2
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds


def _read_overlap_at_res(src, overlap_bounds_wgs, target_res_m=10.0):
    """Read the overlap region of a dataset at target metric resolution."""
    src_bounds_wgs = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    # Intersect
    west = max(overlap_bounds_wgs[0], src_bounds_wgs[0])
    south = max(overlap_bounds_wgs[1], src_bounds_wgs[1])
    east = min(overlap_bounds_wgs[2], src_bounds_wgs[2])
    north = min(overlap_bounds_wgs[3], src_bounds_wgs[3])
    if west >= east or south >= north:
        return None, None, None

    # Work in EPSG:3857 for metric resolution
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    mx_w, my_s = t.transform(west, south)
    mx_e, my_n = t.transform(east, north)

    width = max(1, int(round((mx_e - mx_w) / target_res_m)))
    height = max(1, int(round((my_n - my_s) / target_res_m)))
    if width < 64 or height < 64:
        return None, None, None

    from rasterio.transform import from_bounds
    dst_transform = from_bounds(mx_w, my_s, mx_e, my_n, width, height)
    dst_array = np.zeros((height, width), dtype=np.float32)
    reproject(
        source=rasterio.band(src, 1),
        destination=dst_array,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:3857",
        resampling=Resampling.bilinear,
    )
    return dst_array, dst_transform, (mx_w, my_s, mx_e, my_n)


def _clahe_u8(arr):
    """CLAHE normalize a float array to uint8."""
    valid = arr[arr > 0]
    if valid.size < 100:
        return np.zeros_like(arr, dtype=np.uint8)
    lo, hi = np.percentile(valid, [1, 99])
    if hi <= lo:
        hi = lo + 1
    stretched = np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(stretched)


def generate_auto_anchors(
    target_path: str,
    reference_path: str,
    frame_bbox_wgs: tuple,
    output_path: str,
    coarse_res_m: float = 10.0,
    grid_cells: int = 4,
    min_matches_per_cell: int = 1,
    min_total_anchors: int = 4,
    ransac_thresh_px: float = 8.0,
    model_cache=None,
    device_override: str | None = None,
) -> str | None:
    """Generate automatic anchor GCPs by RoMa matching at coarse resolution.

    Reads the target frame and reference in their overlap region at
    ~10 m/px, runs RoMa matching, filters by RANSAC, and selects
    spatially-distributed points on a grid.

    Args:
        target_path: Georeferenced target frame.
        reference_path: Georeferenced reference image.
        frame_bbox_wgs: (west, south, east, north) of the frame in EPSG:4326.
        output_path: Where to write the anchor JSON.
        coarse_res_m: Resolution for matching (metres/pixel).
        grid_cells: Grid subdivision for spatial distribution (grid_cells × grid_cells).
        min_matches_per_cell: Minimum matches to keep per grid cell.
        min_total_anchors: Skip if fewer anchors than this survive.
        ransac_thresh_px: RANSAC inlier threshold in pixels.

    Returns:
        Path to generated anchor JSON, or None on failure.
    """
    import torch

    # Compute overlap in WGS84
    with rasterio.open(reference_path) as ref_ds:
        ref_bounds_wgs = transform_bounds(ref_ds.crs, "EPSG:4326", *ref_ds.bounds)

    west = max(frame_bbox_wgs[0], ref_bounds_wgs[0])
    south = max(frame_bbox_wgs[1], ref_bounds_wgs[1])
    east = min(frame_bbox_wgs[2], ref_bounds_wgs[2])
    north = min(frame_bbox_wgs[3], ref_bounds_wgs[3])
    if west >= east or south >= north:
        return None

    overlap_wgs = (west, south, east, north)
    # Expand slightly for matching context
    margin = 0.02  # ~2km
    expanded = (west - margin, south - margin, east + margin, north + margin)

    # Read both images at coarse resolution in their overlap
    with rasterio.open(target_path) as tgt_ds:
        arr_tgt, tf_tgt, bounds_tgt = _read_overlap_at_res(tgt_ds, expanded, coarse_res_m)
    if arr_tgt is None:
        return None

    with rasterio.open(reference_path) as ref_ds:
        arr_ref, tf_ref, bounds_ref = _read_overlap_at_res(ref_ds, expanded, coarse_res_m)
    if arr_ref is None:
        return None

    # Ensure same dimensions (crop to smaller)
    h = min(arr_tgt.shape[0], arr_ref.shape[0])
    w = min(arr_tgt.shape[1], arr_ref.shape[1])
    arr_tgt = arr_tgt[:h, :w]
    arr_ref = arr_ref[:h, :w]

    if h < 64 or w < 64:
        return None

    # Check content overlap — use a low threshold because panoramic strips
    # are mostly ocean/desert with only a small portion containing land
    tgt_valid = np.count_nonzero(arr_tgt > 10) / max(arr_tgt.size, 1)
    ref_valid = np.count_nonzero(arr_ref > 10) / max(arr_ref.size, 1)
    if tgt_valid < 0.02 or ref_valid < 0.02:
        print(f"  [auto_anchors] Insufficient content (tgt={tgt_valid:.0%} ref={ref_valid:.0%})")
        return None

    # CLAHE normalize
    tgt_u8 = _clahe_u8(arr_tgt)
    ref_u8 = _clahe_u8(arr_ref)

    # Prepare RoMa tensors
    from align.models import ModelCache, clear_torch_cache, get_torch_device

    cache = model_cache
    created_cache = False
    if cache is None:
        cache = ModelCache(get_torch_device(device_override))
        created_cache = True
    device = cache.device

    def _to_roma_tensor(img_u8):
        rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
        cur_h, cur_w = rgb.shape[:2]
        # RoMa needs dimensions divisible by 14, minimum 448
        min_dim = 448
        if cur_h < min_dim or cur_w < min_dim:
            scale = max(min_dim / cur_h, min_dim / cur_w)
            target_h = int(round((cur_h * scale) / 14) * 14)
            target_w = int(round((cur_w * scale) / 14) * 14)
            rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        else:
            target_h = (cur_h // 14) * 14
            target_w = (cur_w // 14) * 14
            rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return t[None].to(device), (target_h, target_w), (cur_h, cur_w)

    ref_t, ref_roma_shape, ref_orig_shape = _to_roma_tensor(ref_u8)
    tgt_t, tgt_roma_shape, tgt_orig_shape = _to_roma_tensor(tgt_u8)

    # Run RoMa
    roma = cache.roma

    try:
        with torch.no_grad():
            roma.apply_setting("satast")
            preds = roma.match(ref_t, tgt_t)
            warp_AB = preds["warp_AB"]
            B, H, W, _ = warp_AB.shape
            from align.romav2.geometry import get_normalized_grid
            grid = get_normalized_grid(B, H, W, overload_device=warp_AB.device)
            warp = torch.cat([grid, warp_AB], dim=-1)
            certainty = preds["confidence_AB"][..., 0]

            # Sample top matches
            cert_2d = certainty[0]
            warp_2d = warp[0]
            flat_cert = cert_2d.reshape(-1)
            flat_warp = warp_2d.reshape(-1, 4)
            k = min(2000, flat_cert.numel())
            topk_vals, topk_idx = flat_cert.topk(k)
            matches = flat_warp[topk_idx].cpu().numpy()
            certs = topk_vals.cpu().numpy()
    except Exception as e:
        print(f"  [auto_anchors] RoMa error: {e}")
        if created_cache:
            cache.close()
        else:
            clear_torch_cache(device)
        return None
    finally:
        del ref_t, tgt_t
        clear_torch_cache(device)

    if created_cache:
        cache.close()

    if len(matches) < min_total_anchors:
        return None

    # Convert normalized coords [-1, 1] to pixel coords
    ref_h_roma, ref_w_roma = ref_roma_shape
    tgt_h_roma, tgt_w_roma = tgt_roma_shape
    ref_px = matches[:, :2].copy()
    tgt_px = matches[:, 2:].copy()
    ref_px[:, 0] = (ref_px[:, 0] + 1) / 2 * ref_w_roma
    ref_px[:, 1] = (ref_px[:, 1] + 1) / 2 * ref_h_roma
    tgt_px[:, 0] = (tgt_px[:, 0] + 1) / 2 * tgt_w_roma
    tgt_px[:, 1] = (tgt_px[:, 1] + 1) / 2 * tgt_h_roma

    # Scale back to original pixel coords
    ref_px[:, 0] *= ref_orig_shape[1] / ref_w_roma
    ref_px[:, 1] *= ref_orig_shape[0] / ref_h_roma
    tgt_px[:, 0] *= tgt_orig_shape[1] / tgt_w_roma
    tgt_px[:, 1] *= tgt_orig_shape[0] / tgt_h_roma

    # RANSAC filter
    if len(ref_px) < 4:
        return None
    M, inliers = cv2.estimateAffinePartial2D(
        ref_px.astype(np.float32),
        tgt_px.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh_px,
    )
    if inliers is None:
        return None
    inlier_mask = inliers.ravel().astype(bool)
    if inlier_mask.sum() < min_total_anchors:
        print(f"  [auto_anchors] Only {inlier_mask.sum()} robust inliers (need {min_total_anchors})")
        return None

    ref_px = ref_px[inlier_mask]
    tgt_px = tgt_px[inlier_mask]
    certs = certs[inlier_mask]

    # Convert reference pixel coords to WGS84 lon/lat.
    # ref_px is in the overlap array which was reprojected to EPSG:3857.
    from pyproj import Transformer
    t_inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    ref_mx = bounds_ref[0] + ref_px[:, 0] * coarse_res_m
    ref_my = bounds_ref[3] - ref_px[:, 1] * coarse_res_m  # y decreases downward
    ref_lons, ref_lats = t_inv.transform(ref_mx, ref_my)

    # Grid-based spatial selection: pick best match per cell
    lon_min, lon_max = ref_lons.min(), ref_lons.max()
    lat_min, lat_max = ref_lats.min(), ref_lats.max()
    if lon_max <= lon_min or lat_max <= lat_min:
        return None

    gcps = []
    cell_w = (lon_max - lon_min) / grid_cells
    cell_h = (lat_max - lat_min) / grid_cells

    for gi in range(grid_cells):
        for gj in range(grid_cells):
            cell_lon_lo = lon_min + gj * cell_w
            cell_lon_hi = cell_lon_lo + cell_w
            cell_lat_lo = lat_min + gi * cell_h
            cell_lat_hi = cell_lat_lo + cell_h

            in_cell = (
                (ref_lons >= cell_lon_lo) & (ref_lons < cell_lon_hi) &
                (ref_lats >= cell_lat_lo) & (ref_lats < cell_lat_hi)
            )
            if in_cell.sum() < min_matches_per_cell:
                continue

            # Pick highest-confidence match in this cell
            cell_indices = np.where(in_cell)[0]
            best_idx = cell_indices[np.argmax(certs[cell_indices])]
            gcps.append({
                "name": f"auto_r{gi}c{gj}",
                "lon": float(ref_lons[best_idx]),
                "lat": float(ref_lats[best_idx]),
                "feature_type": "auto_match",
                "confidence": "medium" if certs[best_idx] > 0.5 else "low",
                "patch_size_m": 300,
            })

    if len(gcps) < min_total_anchors:
        print(f"  [auto_anchors] Only {len(gcps)} spatially-distributed anchors "
              f"(need {min_total_anchors})")
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    anchor_data = {
        "description": f"Auto-generated anchors ({len(gcps)} points, "
                       f"{grid_cells}x{grid_cells} grid)",
        "gcps": gcps,
    }
    with open(output_path, "w") as f:
        json.dump(anchor_data, f, indent=2)

    print(f"  [auto_anchors] Generated {len(gcps)} anchors → {os.path.basename(output_path)}")
    return output_path
