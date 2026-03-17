"""Scale and rotation detection: 3 methods with a registry cascade.

Method A (ELoFTR) runs first. Methods B+C (NCC grids) run sequentially as fallback.
"""

import multiprocessing as mp
import os
import subprocess
import tempfile
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import rasterio
import rasterio.transform
from rasterio.warp import transform as transform_coords, transform_bounds
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

import torch

from .constants import RANSAC_REPROJ_THRESHOLD
from .geo import get_torch_device, read_overlap_region
from .image import shift_array, make_land_mask, clahe_normalize, sobel_gradient

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

ScaleResult = namedtuple("ScaleResult", ["scale_x", "scale_y", "rotation", "method"])


# ---------------------------------------------------------------------------
# Individual detection functions
# ---------------------------------------------------------------------------


def _solve_scale_affine(kpts0, kpts1):
    """Estimate affine transform from keypoints and extract scale/rotation."""
    if len(kpts0) < 8:
        return None

    src_pts = kpts0.reshape(-1, 1, 2)
    dst_pts = kpts1.reshape(-1, 1, 2)

    # RANSAC
    try:
        M, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC,
                                           ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD)
    except Exception:
        return None

    if M is None or inliers is None:
        return None

    n_inliers = int(inliers.sum())
    if n_inliers < 8:
        return None

    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    scale_x = np.sqrt(a ** 2 + c ** 2)
    scale_y = np.sqrt(b ** 2 + d ** 2)
    rotation_deg = np.degrees(np.arctan2(c, a))

    return scale_x, scale_y, rotation_deg, n_inliers


def _get_matcher_crops(ref_img, off_img):
    """Generate patches for ELoFTR inference (single center or 3x3 grid).

    Returns list of dicts: {'ref': ref_crop, 'off': off_crop, 'r0': r0, 'c0': c0}
    """
    h, w = ref_img.shape
    patch_size = min(800, h, w)
    
    margin = patch_size // 2
    if h <= patch_size or w <= patch_size:
        centers = [(h // 2, w // 2)]
    else:
        rows = np.linspace(margin, h - margin, 3, dtype=int)
        cols = np.linspace(margin, w - margin, 3, dtype=int)
        centers = [(r, c) for r in rows for c in cols]
        
    crops = []
    for cy, cx in centers:
        r0 = max(0, cy - patch_size // 2)
        r1 = min(h, cy + patch_size // 2)
        c0 = max(0, cx - patch_size // 2)
        c1 = min(w, cx + patch_size // 2)
        
        ref_patch = ref_img[r0:r1, c0:c1]
        off_patch = off_img[r0:r1, c0:c1]
        
        if np.mean(ref_patch > 0) < 0.3 or np.mean(off_patch > 0) < 0.3:
            continue
            
        crops.append({
            'ref': ref_patch,
            'off': off_patch,
            'r0': r0,
            'c0': c0
        })
    return crops

def _run_eloftr_batch(crops, model, device, batch_size=4):
    """Run ELoFTR on a list of crops using batching.

    Returns list of (kpts0, kpts1, conf) corresponding to crops.
    Returns empty arrays for failed/skipped items in the batch.
    """
    if not crops:
        return []

    from typing import List, Tuple, Any
    from numpy.typing import NDArray
    results: List[Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]] = [
        (np.array([]), np.array([]), np.array([])) for _ in range(len(crops))
    ]
    
    # Process in chunks of batch_size
    for i in range(0, len(crops), batch_size):
        batch_slice = crops[i:i+batch_size]
        indices_slice = range(i, i+len(batch_slice))
        
        # Group by shape to handle different crop sizes
        groups = {}
        for idx, item in zip(indices_slice, batch_slice):
            sh = item['ref'].shape
            if sh not in groups:
                groups[sh] = []
            groups[sh].append((idx, item))
            
        for shape, group_items in groups.items():
            # Build tensors
            ref_tensors = []
            off_tensors = []
            original_indices = []
            
            for idx, item in group_items:
                rt = torch.from_numpy(item['ref']).float()[None, None] / 255.0
                ot = torch.from_numpy(item['off']).float()[None, None] / 255.0
                ref_tensors.append(rt)
                off_tensors.append(ot)
                original_indices.append(idx)
                
            batch_ref = torch.cat(ref_tensors, dim=0).to(device)
            batch_off = torch.cat(off_tensors, dim=0).to(device)
            
            try:
                with torch.no_grad():
                    out = model({"image0": batch_ref, "image1": batch_off})

                kpts0 = out['mkpts0_f'].cpu().numpy()
                kpts1 = out['mkpts1_f'].cpu().numpy()
                conf = out['mconf'].cpu().numpy()
                batch_ids = out['m_bids'].cpu().numpy()

                for local_idx, global_idx in enumerate(original_indices):
                    # Extract matches for this item
                    mask = (batch_ids == local_idx)
                    if mask.any():
                        results[global_idx] = (kpts0[mask], kpts1[mask], conf[mask])
                    else:
                        # No matches found for this item
                        results[global_idx] = (np.array([]), np.array([]), np.array([]))

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower() and "MPS" not in str(e):
                    raise
                del batch_ref, batch_off
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                print(f"      ELoFTR batch OOM ({len(group_items)} items), retrying one-by-one")
                for local_idx, (global_idx, item) in enumerate(group_items):
                    try:
                        rt = torch.from_numpy(item['ref']).float()[None, None] / 255.0
                        ot = torch.from_numpy(item['off']).float()[None, None] / 255.0
                        single_ref = rt.to(device)
                        single_off = ot.to(device)
                        with torch.no_grad():
                            out = model({"image0": single_ref, "image1": single_off})
                        kp0 = out['mkpts0_f'].cpu().numpy()
                        kp1 = out['mkpts1_f'].cpu().numpy()
                        cf = out['mconf'].cpu().numpy()
                        results[global_idx] = (kp0, kp1, cf)
                        del single_ref, single_off
                    except Exception as e2:
                        print(f"      ELoFTR single-item retry also failed: {e2}")
            except Exception as e:
                print(f"      ELoFTR batch inference error: {e}")
                
    return results

def detect_eloftr_scale(ref_img, offset_img, model=None):
    """ELoFTR dense matching + RANSAC affine (tiled over 3x3 grid).

    Returns (scale_x, scale_y, rotation_deg, n_inliers) or None.
    """
    device = get_torch_device()
    if model is None:
        print("      ELoFTR model not provided")
        return None

    ref_u8 = clahe_normalize(ref_img)
    off_u8 = clahe_normalize(offset_img)

    h, w = ref_u8.shape

    crops = _get_matcher_crops(ref_u8, off_u8)
    if not crops:
         return None

    # Run batched inference
    batch_results = _run_eloftr_batch(crops, model, device, batch_size=4)

    # Aggregate keypoints from batch results
    all_kpts0 = []
    all_kpts1 = []

    for i, res in enumerate(batch_results):
        if res is None:
            continue
        kp0, kp1, conf = res
        if kp0 is None or len(kp0) == 0:
            continue
        mask = conf > 0.5
        if mask.sum() < 3:
            continue
        crop = crops[i]
        kp0_global = kp0[mask] + np.array([crop['c0'], crop['r0']])
        kp1_global = kp1[mask] + np.array([crop['c0'], crop['r0']])
        all_kpts0.append(kp0_global)
        all_kpts1.append(kp1_global)

    if not all_kpts0:
        print("      ELoFTR scale: no patches yielded matches")
        return None

    kpts0 = np.vstack(all_kpts0).astype(np.float32)
    kpts1 = np.vstack(all_kpts1).astype(np.float32)

    result = _solve_scale_affine(kpts0, kpts1)
    if result is None:
        print("      ELoFTR scale: affine estimation failed")
        return None

    scale_x, scale_y, rotation_deg, n_inliers = result
    print(f"      ELoFTR scale: {n_inliers} inliers from {len(kpts0)} matches, "
          f"scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")

    return scale_x, scale_y, rotation_deg, n_inliers


# Module-level shared data for ProcessPoolExecutor workers
_scale_worker_data = {}


def _try_scale_rotation(args):
    """Worker: try a single (scale, rotation) combination."""
    scale_val, rot_val = args
    d = _scale_worker_data
    template = d['template']
    offset_land = d['offset_land']

    new_h = max(10, int(template.shape[0] * scale_val))
    new_w = max(10, int(template.shape[1] * scale_val))
    resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if abs(rot_val) > 0.01:
        center = (new_w // 2, new_h // 2)
        M_rot = cv2.getRotationMatrix2D(center, rot_val, 1.0)
        resized = cv2.warpAffine(resized, M_rot, (new_w, new_h),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if (resized.shape[0] >= offset_land.shape[0] or
            resized.shape[1] >= offset_land.shape[1]):
        return scale_val, rot_val, -1

    if resized.shape[0] < 10 or resized.shape[1] < 10:
        return scale_val, rot_val, -1

    result = cv2.matchTemplate(offset_land, resized, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return scale_val, rot_val, max_val


def detect_multiscale_ncc(ref_land, offset_land, expected_scale):
    """Brute-force multi-scale NCC on land masks or gradient images.

    Returns (scale, rotation_deg, correlation) or None.
    """
    h, w = ref_land.shape
    th, tw = h // 3, w // 3
    cy, cx = h // 2, w // 2
    template = ref_land[cy - th // 2:cy + th // 2, cx - tw // 2:cx + tw // 2]

    if template.shape[0] < 20 or template.shape[1] < 20:
        return None

    scale_min, scale_max = 0.7, 1.35
    n_scales = 25
    scales = np.linspace(scale_min, scale_max, n_scales)
    rotations = np.linspace(-6, 6, 13)

    best_corr = -1
    best_scale = 1.0
    best_rot = 0.0

    global _scale_worker_data
    _scale_worker_data = {'template': template, 'offset_land': offset_land}

    combos = [(s, r) for s in scales for r in rotations]
    n_workers = min(len(combos), max(1, (os.cpu_count() or 4) - 1))

    ctx = mp.get_context('fork')
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        for s, r, corr in pool.map(_try_scale_rotation, combos, chunksize=8):
            if corr > best_corr:
                best_corr = corr
                best_scale = s
                best_rot = r

    _scale_worker_data = {}

    if best_corr < 0.3:
        return None

    return best_scale, best_rot, best_corr


# ---------------------------------------------------------------------------
# Registry / cascade
# ---------------------------------------------------------------------------

def _gate_neural(result):
    if result is None:
        return False
    sx, sy, rot, n_inliers = result
    avg = (sx + sy) / 2
    return n_inliers >= 8 and 0.75 < avg < 1.30


def _gate_ncc_land(result):
    if result is None:
        return False
    scale, rot, corr = result
    return corr >= 0.55 and 0.75 < scale < 1.30


def _gate_ncc_gradient(result):
    if result is None:
        return False
    scale, rot, corr = result
    return corr >= 0.4 and 0.75 < scale < 1.30


def _to_scale_result_4(result, method):
    """Convert (sx, sy, rot, n_inliers) -> ScaleResult."""
    return ScaleResult(result[0], result[1], result[2], method)


def _to_scale_result_ncc(result, method):
    """Convert (scale, rot, corr) -> ScaleResult (isotropic)."""
    return ScaleResult(result[0], result[0], result[1], method)


def detect_scale_rotation(src_offset, src_ref, overlap, work_crs,
                          coarse_dx, coarse_dy, expected_scale, model_cache=None):
    """Orchestrate scale/rotation detection: ELoFTR first, NCC grids as fallback.

    Returns ScaleResult (scale_x, scale_y, rotation, method_name).
    """
    detection_res = 5.0

    arr_ref, _ = read_overlap_region(src_ref, overlap, work_crs, detection_res)
    arr_off, _ = read_overlap_region(src_offset, overlap, work_crs, detection_res)

    # Shift offset image by coarse translation for better overlap
    shift_px_x = int(round(coarse_dx / detection_res))
    shift_py_y = int(round(coarse_dy / detection_res))
    h, w = arr_off.shape
    M_shift = np.array([[1.0, 0.0, float(-shift_px_x)], [0.0, 1.0, float(-shift_py_y)]], dtype=np.float32)
    arr_off_shifted = cv2.warpAffine(arr_off, M_shift, (w, h),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # --- Method A: ELoFTR ---
    print("    Method: ELoFTR...")
    try:
        eloftr_model = model_cache.eloftr if model_cache else None
        result = detect_eloftr_scale(arr_ref, arr_off_shifted, model=eloftr_model)
        if _gate_neural(result):
            sr = _to_scale_result_4(result, "eloftr")
            avg = (sr.scale_x + sr.scale_y) / 2
            print(f"      Accepted (scale={avg:.4f}, rotation={sr.rotation:.3f} deg)")
            return sr
        elif result is not None:
            print("      Gate rejected, trying NCC fallback")
        else:
            print("      Failed to detect, trying NCC fallback")
    except Exception as e:
        print(f"      Error: {e}, trying NCC fallback")

    # --- Methods B+C: parallel NCC grids ---
    ref_land = make_land_mask(arr_ref)
    off_land = make_land_mask(arr_off_shifted)

    ref_u8 = clahe_normalize(arr_ref)
    off_u8 = clahe_normalize(arr_off_shifted)
    ref_grad = sobel_gradient(ref_u8).astype(np.float32)
    off_grad = sobel_gradient(off_u8).astype(np.float32)

    parallel_specs = [
        ("NCC-land", ref_land, off_land, _gate_ncc_land, _to_scale_result_ncc, "multiscale-ncc"),
        ("NCC-gradient", ref_grad, off_grad, _gate_ncc_gradient, _to_scale_result_ncc, "gradient-ncc"),
    ]

    print("    Methods B+C: NCC-land and NCC-gradient (sequential)...")
    for label, ref_data, off_data, gate, convert, method_name in parallel_specs:
        try:
            result = detect_multiscale_ncc(ref_data, off_data, expected_scale)
            if gate(result):
                sr = convert(result, method_name)
                avg = (sr.scale_x + sr.scale_y) / 2
                print(f"      {label}: Accepted (scale={avg:.4f}, rotation={sr.rotation:.3f} deg)")
                return sr
            elif result is not None:
                scale_val, rot_val, corr_val = result
                print(f"      {label}: Gate rejected (scale={scale_val:.4f}, corr={corr_val:.3f})")
            else:
                print(f"      {label}: Failed to detect")
        except Exception as e:
            print(f"      {label}: Error: {e}")

    print("    No significant scale/rotation detected")
    return ScaleResult(1.0, 1.0, 0.0, None)


# ---------------------------------------------------------------------------
# Local (patch-based) scale detection
# ---------------------------------------------------------------------------

def detect_local_scales(src_offset, src_ref, overlap, work_crs,
                        coarse_dx, coarse_dy, grid_cols=3, grid_rows=3,
                        model_cache=None):
    """Detect scale and rotation per-patch across the overlap region using ELoFTR.

    Returns list of dicts with keys: cx, cy, scale_x, scale_y, rotation,
    n_inliers, row, col, status.  Returns None if fewer than 3 patches succeed.
    """
    detection_res = 5.0

    arr_ref, ref_transform = read_overlap_region(
        src_ref, overlap, work_crs, detection_res)
    arr_off, off_transform = read_overlap_region(
        src_offset, overlap, work_crs, detection_res)

    arr_off_shifted = shift_array(arr_off, -int(round(coarse_dx / detection_res)),
                                  -int(round(coarse_dy / detection_res)))

    h, w = arr_ref.shape
    print(f"    Overlap array: {w} x {h} px at {detection_res} m/px")
    print(f"    Grid: {grid_cols} x {grid_rows} = {grid_cols * grid_rows} patches")

    device = get_torch_device()
    if model_cache is None:
        print(f"    ELoFTR model cache not provided")
        return None
    model = model_cache.eloftr

    patch_w = min(int(w / grid_cols * 1.2), w)
    patch_h = min(int(h / grid_rows * 1.2), h)
    stride_x = (w - patch_w) / max(grid_cols - 1, 1) if grid_cols > 1 else 0
    stride_y = (h - patch_h) / max(grid_rows - 1, 1) if grid_rows > 1 else 0

    ol_left, ol_bottom, ol_right, ol_top = overlap

    patch_results = []
    inference_tasks = [] # (patch_idx, crops, x0, y0)

    for gr in range(grid_rows):
        for gc in range(grid_cols):
            y0 = int(round(gr * stride_y))
            x0 = int(round(gc * stride_x))
            y1 = min(y0 + patch_h, h)
            x1 = min(x0 + patch_w, w)

            ref_patch = arr_ref[y0:y1, x0:x1]
            off_patch = arr_off_shifted[y0:y1, x0:x1]

            valid_frac_ref = np.count_nonzero(ref_patch > 0) / ref_patch.size
            valid_frac_off = np.count_nonzero(off_patch > 0) / off_patch.size
            min_valid = min(valid_frac_ref, valid_frac_off)

            patch_cx = ol_left + ((x0 + x1) / 2) * detection_res
            patch_cy = ol_top - ((y0 + y1) / 2) * detection_res

            label = f"    Patch [{gr},{gc}] ({x1 - x0}x{y1 - y0}px)"

            if min_valid < 0.30:
                print(f"{label}: skipped (valid={min_valid:.0%})")
                patch_results.append({
                    'row': gr, 'col': gc, 'cx': patch_cx, 'cy': patch_cy,
                    'scale_x': None, 'scale_y': None,
                    'rotation': None, 'n_inliers': 0, 'status': 'skipped'
                })
                continue

            # Prepare crops for this patch
            crops = _get_matcher_crops(ref_patch, off_patch)
            if not crops:
                print(f"{label}: no valid crops")
                patch_results.append({
                    'row': gr, 'col': gc, 'cx': patch_cx, 'cy': patch_cy,
                    'scale_x': None, 'scale_y': None,
                    'rotation': None, 'n_inliers': 0, 'status': 'failed'
                })
                continue

            # Add to pending tasks
            patch_idx = len(patch_results)
            patch_results.append({
                'row': gr, 'col': gc, 'cx': patch_cx, 'cy': patch_cy,
                'scale_x': None, 'scale_y': None,
                'rotation': None, 'n_inliers': 0, 'status': 'pending'
            })
            inference_tasks.append((patch_idx, crops, x0, y0, label))

    # Run batch inference if tasks exist
    if inference_tasks:
        all_crops = []
        task_ranges = [] # (patch_idx, start, end, x0, y0, label)
        
        for p_idx, p_crops, x0, y0, label in inference_tasks:
            start = len(all_crops)
            all_crops.extend(p_crops)
            end = len(all_crops)
            task_ranges.append((p_idx, start, end, x0, y0, label))
            
        print(f"    Running batched ELoFTR on {len(all_crops)} total crops/sub-patches...")
        batch_results = _run_eloftr_batch(all_crops, model, device, batch_size=4)
        
        # Process results
        for p_idx, start, end, x0, y0, label in task_ranges:
            # Aggregate matches for this patch
            all_kpts0 = []
            all_kpts1 = []
            
            for i in range(start, end):
                res = batch_results[i]
                if res is None: continue
                kp0, kp1, conf = res
                if kp0 is None or len(kp0) == 0: continue
                
                mask = conf > 0.5
                if mask.sum() < 3:
                    continue
                
                # Crops are relative to patch (which is at x0,y0 in overlap)
                # Reconstruct patch-local coords for RANSAC
                
                c0_c, r0_c = all_crops[i]['c0'], all_crops[i]['r0']
                kp0_local = kp0[mask] + np.array([c0_c, r0_c])
                kp1_local = kp1[mask] + np.array([c0_c, r0_c])
                
                all_kpts0.append(kp0_local)
                all_kpts1.append(kp1_local)
                
            status = 'failed'
            sx, sy, rot, n_inliers = None, None, None, 0
            
            if all_kpts0:
                kpts0 = np.vstack(all_kpts0).astype(np.float32)
                kpts1 = np.vstack(all_kpts1).astype(np.float32)
                
                res_affine = _solve_scale_affine(kpts0, kpts1)
                if res_affine:
                    sx, sy, rot, n_inliers = res_affine
                    avg_s = (sx + sy) / 2
                    
                    if not (0.75 < avg_s < 1.30) or n_inliers < 8:
                        reason = "scale out of range" if not (0.75 < avg_s < 1.30) else "too few inliers"
                        print(f"{label}: rejected ({reason}, s={avg_s:.4f}, n={n_inliers})")
                        status = 'rejected'
                    else:
                        print(f"{label}: sx={sx:.4f} sy={sy:.4f} rot={rot:.3f} deg n={n_inliers}")
                        status = 'ok'
                else:
                    print(f"{label}: affine failed")
            else:
                print(f"{label}: ELoFTR failed (no matches)")
                
            # Update result
            p = patch_results[p_idx]
            p['scale_x'] = sx
            p['scale_y'] = sy
            p['rotation'] = rot
            p['n_inliers'] = n_inliers
            p['status'] = status

    valid = [p for p in patch_results if p['status'] == 'ok']
    print(f"    Valid patches: {len(valid)}/{len(patch_results)}")

    if len(valid) < 3:
        print("    Too few valid patches (<3), falling back to global detection")
        return None

    # MAD-based outlier rejection
    valid_sx = np.array([p['scale_x'] for p in valid])
    valid_sy = np.array([p['scale_y'] for p in valid])
    valid_rot = np.array([p['rotation'] for p in valid])

    def _mad_filter(values, threshold=3.0):
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad < 1e-6:
            return np.ones(len(values), dtype=bool)
        modified_z = 0.6745 * np.abs(values - median) / mad
        return modified_z < threshold

    mask_sx = _mad_filter(valid_sx)
    mask_sy = _mad_filter(valid_sy)
    mask_rot = _mad_filter(valid_rot)
    inlier_mask = mask_sx & mask_sy & mask_rot

    n_outliers = int((~inlier_mask).sum())
    if n_outliers > 0:
        print(f"    MAD outlier rejection: removed {n_outliers} patch(es)")
        for i, p in enumerate(valid):
            if not inlier_mask[i]:
                p['status'] = 'outlier'
                p['scale_x'] = None
                p['scale_y'] = None
                p['rotation'] = None

    valid = [p for p in patch_results if p['status'] == 'ok']
    if len(valid) < 3:
        print("    Too few patches after outlier rejection (<3), falling back to global")
        return None

    # Fill failed/skipped/rejected with median of neighbours or global median
    global_med_sx = np.median([p['scale_x'] for p in valid])
    global_med_sy = np.median([p['scale_y'] for p in valid])
    global_med_rot = np.median([p['rotation'] for p in valid])

    for p in patch_results:
        if p['scale_x'] is not None:
            continue
        neighbors = [q for q in valid
                     if abs(q['row'] - p['row']) <= 1 and abs(q['col'] - p['col']) <= 1]
        if neighbors:
            p['scale_x'] = np.median([n['scale_x'] for n in neighbors])
            p['scale_y'] = np.median([n['scale_y'] for n in neighbors])
            p['rotation'] = np.median([n['rotation'] for n in neighbors])
            p['status'] = 'filled-neighbor'
        else:
            p['scale_x'] = global_med_sx
            p['scale_y'] = global_med_sy
            p['rotation'] = global_med_rot
            p['status'] = 'filled-global'

    # Diagnostic table
    print(f"\n    {'Patch':>10} {'Status':>15} {'Scale_X':>8} {'Scale_Y':>8} "
          f"{'Rotation':>9} {'Inliers':>8}")
    print(f"    {'-' * 10} {'-' * 15} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 8}")
    for p in patch_results:
        print(f"    [{p['row']},{p['col']}]"
              f"  {p['status']:>15}"
              f"  {p['scale_x']:8.4f}"
              f"  {p['scale_y']:8.4f}"
              f"  {p['rotation']:+9.3f} deg"
              f"  {p['n_inliers']:>8}")

    all_sx = [p['scale_x'] for p in patch_results]
    all_sy = [p['scale_y'] for p in patch_results]
    print(f"\n    Scale_X range: {min(all_sx):.4f} - {max(all_sx):.4f} "
          f"(spread: {max(all_sx) - min(all_sx):.4f})")
    print(f"    Scale_Y range: {min(all_sy):.4f} - {max(all_sy):.4f} "
          f"(spread: {max(all_sy) - min(all_sy):.4f})")

    return patch_results


# ---------------------------------------------------------------------------
# Affine fitting — canonical implementation lives in align.affine;
# re-exported here for backwards compatibility with existing imports.
# ---------------------------------------------------------------------------

from .affine import fit_affine_from_gcps, compute_affine_residuals, ransac_affine  # noqa: F401,E402


# ---------------------------------------------------------------------------
# Pre-correction application (global and local)
# ---------------------------------------------------------------------------

def apply_scale_rotation_precorrection(input_path, scale, rotation_deg, work_crs,
                                       overlap_center=None, scale_y=None):
    """Apply scale and rotation correction to a GeoTIFF using synthetic GCPs.

    Returns path to the corrected temporary file, or None on failure.
    """
    src = rasterio.open(input_path)
    width = src.width
    height = src.height
    src_transform = src.transform
    src_crs = src.crs
    src_bounds = src.bounds
    src.close()

    if overlap_center is not None:
        cx, cy = overlap_center
    else:
        work_bounds = transform_bounds(src_crs, work_crs, *src_bounds)
        cx = (work_bounds[0] + work_bounds[2]) / 2
        cy = (work_bounds[1] + work_bounds[3]) / 2

    cos_r = np.cos(np.radians(-rotation_deg))
    sin_r = np.sin(np.radians(-rotation_deg))
    inv_scale_x = 1.0 / scale
    inv_scale_y = 1.0 / (scale_y if scale_y is not None else scale)

    if scale_y is not None and abs(scale - scale_y) > 0.01:
        print(f"  Anisotropic correction: scale_x={scale:.4f}, scale_y={scale_y:.4f}")

    n_grid = 7
    cols = np.linspace(0, width - 1, n_grid)
    rows = np.linspace(0, height - 1, n_grid)

    gcps = []
    for col in cols:
        for row in rows:
            src_gx, src_gy = rasterio.transform.xy(src_transform, int(row), int(col))
            if src_crs != work_crs:
                work_xy = transform_coords(src_crs, work_crs, [src_gx], [src_gy])
                work_gx, work_gy = work_xy[0][0], work_xy[1][0]
            else:
                work_gx, work_gy = src_gx, src_gy

            dx = work_gx - cx
            dy = work_gy - cy
            rotated_dx = cos_r * dx - sin_r * dy
            rotated_dy = sin_r * dx + cos_r * dy
            new_dx = inv_scale_x * rotated_dx
            new_dy = inv_scale_y * rotated_dy
            corrected_work_gx = cx + new_dx
            corrected_work_gy = cy + new_dy

            if src_crs != work_crs:
                corr_xy = transform_coords(work_crs, src_crs,
                                           [corrected_work_gx], [corrected_work_gy])
                corr_gx, corr_gy = corr_xy[0][0], corr_xy[1][0]
            else:
                corr_gx, corr_gy = corrected_work_gx, corrected_work_gy

            gcps.append((float(col), float(row), corr_gx, corr_gy))

    tmp_fd2, tmp_warped = tempfile.mkstemp(suffix=".tif")
    os.close(tmp_fd2)

    import uuid
    from osgeo import gdal

    tmp_id = uuid.uuid4().hex
    tmp_gcp = f"/vsimem/tmp_gcp_{tmp_id}.tif"

    gdal.UseExceptions()
    try:
        gcp_list = [gdal.GCP(gx, gy, 0, px, py) for px, py, gx, gy in gcps]

        ds = gdal.Open(input_path)
        if ds is None:
            raise RuntimeError(f"Could not open {input_path}")

        tmp_ds = gdal.Translate(tmp_gcp, ds, GCPs=gcp_list)
        if tmp_ds is None:
            raise RuntimeError("gdal.Translate failed")

        warp_kwargs = {
            'multithread': True,
            'warpOptions': ['NUM_THREADS=ALL_CPUS'],
            'resampleAlg': gdal.GRA_Bilinear,
            'dstSRS': str(src_crs),
            'creationOptions': ['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES',
                                'NUM_THREADS=ALL_CPUS', 'BIGTIFF=YES'],
            'warpMemoryLimit': 2048 * 1024 * 1024,
            'tps': True
        }

        out_ds = gdal.Warp(tmp_warped, tmp_ds, **warp_kwargs)
        if out_ds is None:
            raise RuntimeError("gdal.Warp failed")

        out_ds = None
        tmp_ds = None
        ds = None

        return tmp_warped
    except Exception as e:
        print(f"  ERROR: Pre-correction failed: {e}")
        if os.path.exists(tmp_warped):
            os.remove(tmp_warped)
        return None
    finally:
        gdal.Unlink(tmp_gcp)


def apply_local_scale_precorrection(input_path, patch_results, work_crs,
                                    overlap_center=None):
    """Apply spatially-varying scale/rotation correction using per-patch results.

    Returns path to the corrected temporary file, or None on failure.
    """
    src = rasterio.open(input_path)
    width = src.width
    height = src.height
    src_transform = src.transform
    src_crs = src.crs
    src.close()

    centers = np.array([[p['cx'], p['cy']] for p in patch_results])
    sx_vals = np.array([p['scale_x'] for p in patch_results])
    sy_vals = np.array([p['scale_y'] for p in patch_results])
    rot_vals = np.array([p['rotation'] for p in patch_results])

    interp_sx = LinearNDInterpolator(centers, sx_vals)
    interp_sy = LinearNDInterpolator(centers, sy_vals)
    interp_rot = LinearNDInterpolator(centers, rot_vals)
    nearest_sx = NearestNDInterpolator(centers, sx_vals)
    nearest_sy = NearestNDInterpolator(centers, sy_vals)
    nearest_rot = NearestNDInterpolator(centers, rot_vals)

    def _interp_with_fallback(x, y):
        pt = np.array([[x, y]])
        sx = interp_sx(pt)[0]
        sy = interp_sy(pt)[0]
        rot = interp_rot(pt)[0]
        if np.isnan(sx):
            sx = nearest_sx(pt)[0]
        if np.isnan(sy):
            sy = nearest_sy(pt)[0]
        if np.isnan(rot):
            rot = nearest_rot(pt)[0]
        return float(sx), float(sy), float(rot)

    if overlap_center is not None:
        cx, cy = overlap_center
    else:
        cx = float(np.mean(centers[:, 0]))
        cy = float(np.mean(centers[:, 1]))

    n_grid = 11
    cols = np.linspace(0, width - 1, n_grid)
    rows = np.linspace(0, height - 1, n_grid)

    gcps = []
    for col in cols:
        for row in rows:
            src_gx, src_gy = rasterio.transform.xy(src_transform, int(row), int(col))
            if src_crs != work_crs:
                work_xy = transform_coords(src_crs, work_crs, [src_gx], [src_gy])
                work_gx, work_gy = work_xy[0][0], work_xy[1][0]
            else:
                work_gx, work_gy = src_gx, src_gy

            local_sx, local_sy, local_rot = _interp_with_fallback(work_gx, work_gy)

            cos_r = np.cos(np.radians(-local_rot))
            sin_r = np.sin(np.radians(-local_rot))
            inv_sx = 1.0 / local_sx
            inv_sy = 1.0 / local_sy

            dx = work_gx - cx
            dy = work_gy - cy
            rotated_dx = cos_r * dx - sin_r * dy
            rotated_dy = sin_r * dx + cos_r * dy
            new_dx = inv_sx * rotated_dx
            new_dy = inv_sy * rotated_dy
            corrected_work_gx = cx + new_dx
            corrected_work_gy = cy + new_dy

            if src_crs != work_crs:
                corr_xy = transform_coords(work_crs, src_crs,
                                           [corrected_work_gx], [corrected_work_gy])
                corr_gx, corr_gy = corr_xy[0][0], corr_xy[1][0]
            else:
                corr_gx, corr_gy = corrected_work_gx, corrected_work_gy

            gcps.append((float(col), float(row), corr_gx, corr_gy))

    print(f"  Generated {len(gcps)} GCPs ({n_grid}x{n_grid} grid)")

    tmp_fd2, tmp_warped = tempfile.mkstemp(suffix=".tif")
    os.close(tmp_fd2)

    import uuid
    from osgeo import gdal

    tmp_id = uuid.uuid4().hex
    tmp_gcp = f"/vsimem/tmp_gcp_local_{tmp_id}.tif"

    gdal.UseExceptions()
    try:
        gcp_list = [gdal.GCP(gx, gy, 0, px, py) for px, py, gx, gy in gcps]

        ds = gdal.Open(input_path)
        if ds is None:
            raise RuntimeError(f"Could not open {input_path}")

        tmp_ds = gdal.Translate(tmp_gcp, ds, GCPs=gcp_list)
        if tmp_ds is None:
            raise RuntimeError("gdal.Translate failed")

        warp_kwargs = {
            'multithread': True,
            'warpOptions': ['NUM_THREADS=ALL_CPUS'],
            'resampleAlg': gdal.GRA_Bilinear,
            'dstSRS': str(src_crs),
            'creationOptions': ['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES',
                                'NUM_THREADS=ALL_CPUS', 'BIGTIFF=YES'],
            'warpMemoryLimit': 2048 * 1024 * 1024,
            'tps': True
        }

        out_ds = gdal.Warp(tmp_warped, tmp_ds, **warp_kwargs)
        if out_ds is None:
            raise RuntimeError("gdal.Warp failed")

        out_ds = None
        tmp_ds = None
        ds = None

        return tmp_warped
    except Exception as e:
        print(f"  ERROR: Local pre-correction failed: {e}")
        if os.path.exists(tmp_warped):
            os.remove(tmp_warped)
        return None
    finally:
        gdal.Unlink(tmp_gcp)
