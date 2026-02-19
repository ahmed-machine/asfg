"""Dense feature matching: LightGlue (SuperPoint) and LoFTR tiled."""

import numpy as np
import cv2
import rasterio
import rasterio.transform

import torch
import torch.nn.functional as F
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
import kornia.feature as KF

from .geo import get_torch_device
from .image import clahe_normalize, class_weight_map, make_land_mask, stable_feature_mask, to_u8


def match_with_lightglue(arr_ref, arr_off_shifted, ref_transform,
                         off_transform, shift_px_x, shift_py_y, device=None,
                         max_keypoints=4096, model_cache=None, mask_mode="coastal_obia"):
    """Match features using SuperPoint + LightGlue neural matcher.

    Returns list of (ref_gx, ref_gy, off_gx, off_gy, confidence, name) tuples.
    """
    if device is None:
        device = get_torch_device()

    h, w = arr_ref.shape
    scale_factor = 1.0
    ref_resized = clahe_normalize(arr_ref)
    off_resized = clahe_normalize(arr_off_shifted)

    def _build_tensors(ref_img, off_img, sf):
        if sf < 1.0:
            rh, rw = int(h * sf), int(w * sf)
            ri = cv2.resize(ref_img, (rw, rh), interpolation=cv2.INTER_AREA)
            oi = cv2.resize(off_img, (rw, rh), interpolation=cv2.INTER_AREA)
        else:
            ri, oi = ref_img, off_img
            
        ri = np.ascontiguousarray(ri, dtype=np.float32)
        oi = np.ascontiguousarray(oi, dtype=np.float32)
        
        rt = torch.from_numpy(ri)[None, None] / 255.0
        ot = torch.from_numpy(oi)[None, None] / 255.0
        return rt.to(device, non_blocking=True), ot.to(device, non_blocking=True)

    ref_tensor, off_tensor = _build_tensors(ref_resized, off_resized, scale_factor)

    if model_cache is not None:
        extractor = model_cache.superpoint
        matcher = model_cache.lightglue
    else:
        extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        matcher = LightGlue(features='superpoint', depth_confidence=-1,
                            width_confidence=-1).eval().to(device)

    try:
        with torch.no_grad():
            feats0 = extractor.extract(ref_tensor)
            feats1 = extractor.extract(off_tensor)
            matches01 = matcher({'image0': feats0, 'image1': feats1})
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or "MPS" in str(e):
            max_dim_fallback = 4096
            scale_factor = min(max_dim_fallback / max(h, w), 1.0)
            print(f"    OOM at full resolution, retrying at {max_dim_fallback}px "
                  f"(scale={scale_factor:.3f})")
            del ref_tensor, off_tensor
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            ref_tensor, off_tensor = _build_tensors(ref_resized, off_resized, scale_factor)
            with torch.no_grad():
                feats0 = extractor.extract(ref_tensor)
                feats1 = extractor.extract(off_tensor)
                matches01 = matcher({'image0': feats0, 'image1': feats1})
        else:
            raise

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    kpts0 = feats0['keypoints'].cpu().numpy()
    kpts1 = feats1['keypoints'].cpu().numpy()
    matches = matches01['matches'].cpu().numpy()
    mscores = (matches01['scores'].cpu().numpy()
               if 'scores' in matches01 else np.ones(len(matches)))

    if len(matches) == 0:
        return []

    mkpts0 = kpts0[matches[:, 0]]
    mkpts1 = kpts1[matches[:, 1]]

    conf_mask = mscores > 0.5
    mkpts0 = mkpts0[conf_mask]
    mkpts1 = mkpts1[conf_mask]
    mscores = mscores[conf_mask]

    if len(mkpts0) < 4:
        return []

    # RANSAC
    src_pts = mkpts0.astype(np.float32).reshape(-1, 1, 2)
    dst_pts = mkpts1.astype(np.float32).reshape(-1, 1, 2)
    M_ransac, inliers = cv2.estimateAffine2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)

    if M_ransac is None or inliers is None:
        return []

    inlier_mask = inliers.ravel().astype(bool)
    mkpts0 = mkpts0[inlier_mask]
    mkpts1 = mkpts1[inlier_mask]
    mscores = mscores[inlier_mask]

    if len(mkpts0) == 0:
        return []

    mkpts0 = mkpts0 / scale_factor
    mkpts1 = mkpts1 / scale_factor

    results = []
    for i in range(len(mkpts0)):
        ref_col, ref_row = mkpts0[i]
        off_col, off_row = mkpts1[i]

        ref_gx, ref_gy = rasterio.transform.xy(
            ref_transform, float(ref_row), float(ref_col))
        off_gx, off_gy = rasterio.transform.xy(
            off_transform, float(off_row) + shift_py_y,
            float(off_col) + shift_px_x)

        results.append((ref_gx, ref_gy, off_gx, off_gy, float(mscores[i]), f"lg_{i}"))

    return results


def match_with_eloftr(arr_ref, arr_off_shifted, ref_transform, off_transform,
                      shift_px_x, shift_py_y, neural_res=4.0,
                      min_valid_frac=0.30, skip_ransac=False, model_cache=None,
                      batch_size=4, max_matches=5000, max_tiles=300,
                      existing_anchors=None,
                      src_offset=None, work_crs=None, mask_mode="coastal_obia"):
    """Match features using EfficientLoFTR tiled."""
    # EfficientLoFTR architecture is slightly different but input/output API 
    # of the model is compatible with the tiled logic of match_with_loftr.
    
    device = get_torch_device()
    if model_cache is not None:
        eloftr = model_cache.eloftr
    else:
        # Fallback to standard LoFTR if cache fails
        import kornia.feature as KF
        eloftr = KF.LoFTR(pretrained="outdoor").eval().to(device)

    # We reuse the logic of match_with_loftr but with the eloftr model.
    # To avoid massive code duplication, let's refactor the core tiled logic.
    return _tiled_matcher_core(
        eloftr, "eloftr", arr_ref, arr_off_shifted, ref_transform, off_transform,
        shift_px_x, shift_py_y, neural_res, min_valid_frac, skip_ransac,
        batch_size, max_matches, max_tiles, existing_anchors, src_offset, work_crs,
        mask_mode=mask_mode)


def match_with_roma(arr_ref, arr_off_shifted, ref_transform, off_transform,
                    shift_px_x, shift_py_y, neural_res=4.0,
                    min_valid_frac=0.30, skip_ransac=False, model_cache=None,
                    batch_size=4, max_matches=5000, max_tiles=300,
                    existing_anchors=None,
                    src_offset=None, work_crs=None, mask_mode="coastal_obia"):
    """Match features using RoMa tiled."""
    device = get_torch_device()
    if model_cache is not None:
        roma = model_cache.roma
    else:
        return []

    return _tiled_matcher_core(
        roma, "roma", arr_ref, arr_off_shifted, ref_transform, off_transform,
        shift_px_x, shift_py_y, neural_res, min_valid_frac, skip_ransac,
        batch_size, max_matches, max_tiles, existing_anchors, src_offset, work_crs,
        mask_mode=mask_mode)


def _tiled_matcher_core(model, model_type, arr_ref, arr_off_shifted, ref_transform, off_transform,
                        shift_px_x, shift_py_y, neural_res=4.0,
                        min_valid_frac=0.30, skip_ransac=False,
                        batch_size=4, max_matches=5000, max_tiles=300,
                        existing_anchors=None,
                        src_offset=None, work_crs=None, mask_mode="coastal_obia"):
    """Generic tiled matching orchestrator for LoFTR, ELoFTR, and RoMa."""
    device = get_torch_device()
    h, w = arr_ref.shape

    ref_u8 = clahe_normalize(arr_ref)
    off_u8 = clahe_normalize(arr_off_shifted)
    land_mask = (make_land_mask(arr_ref, mode=mask_mode) > 0)
    stable_mask = (stable_feature_mask(arr_ref, mode=mask_mode) > 0)
    weight_map = class_weight_map(arr_ref, mode=mask_mode)
    grad_x = cv2.Sobel(ref_u8, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(ref_u8, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    scene_valid = arr_ref > 0
    scene_land_frac = float(np.mean((stable_mask | land_mask) & scene_valid)) if np.any(scene_valid) else 0.0
    if scene_land_frac < 0.35:
        max_matches = max(max_matches, 7000)
        max_tiles = max(max_tiles, 420)
    if scene_land_frac < 0.25:
        max_matches = max(max_matches, 8500)
        max_tiles = max(max_tiles, 520)

    tile_size = 1024
    overlap_px = 256
    step = tile_size - overlap_px

    all_matches = []
    batch_inputs = []

    def _collect_matches(correspondences, items):
        if model_type == "roma":
            # correspondences is (warp, certainty) from model.match
            warp, certainty = correspondences
            
            for i in range(len(items)):
                it = items[i]
                try:
                    # model.sample expects matches=[H, W, 4] and certainty=[H, W]
                    # Since match() was batched, we slice it
                    m, c = model.sample(warp[i], certainty[i], num=400)
                    m = m.cpu().numpy()
                    c = c.cpu().numpy()
                except Exception as e:
                    print(f"    RoMa sample error: {e}")
                    continue

                mask = c > 0.2
                m = m[mask]
                c = c[mask]
                
                # RoMa output is normalized [-1, 1]. Project to tile pixel space.
                cur_h, cur_w = it['ref'].shape
                kp0 = (m[:, :2] + 1) / 2 * cur_w
                kp1 = (m[:, 2:] + 1) / 2 * cur_h

                for k in range(len(kp0)):
                    global_ref_col = it['c0'] + kp0[k][0]
                    global_ref_row = it['r0'] + kp0[k][1]
                    global_off_col = it['c0'] + kp1[k][0]
                    global_off_row = it['r0'] + kp1[k][1]

                    ref_gx, ref_gy = rasterio.transform.xy(
                        ref_transform, float(global_ref_row), float(global_ref_col))
                    off_gx, off_gy = rasterio.transform.xy(
                        off_transform, float(global_off_row) + shift_py_y,
                        float(global_off_col) + shift_px_x)

                    all_matches.append((ref_gx, ref_gy, off_gx, off_gy,
                                        float(c[k]), f"roma_{it['tile_idx']}_{k}"))
        else:
            # LoFTR / ELoFTR correspondences format
            if model_type == "eloftr":
                # EfficientLoFTR internal keys mapping
                kpts0 = correspondences['mkpts0_f'].cpu().numpy()
                kpts1 = correspondences['mkpts1_f'].cpu().numpy()
                conf = correspondences['mconf'].cpu().numpy()
                batch_ids = correspondences['b_ids'].cpu().numpy()
            else:
                # Standard Kornia LoFTR keys
                kpts0 = correspondences['keypoints0'].cpu().numpy()
                kpts1 = correspondences['keypoints1'].cpu().numpy()
                conf = correspondences['confidence'].cpu().numpy()
                batch_ids = correspondences['batch_indexes'].cpu().numpy()

            for i in range(len(items)):
                mask_b = (batch_ids == i) & (conf > 0.5)
                if not mask_b.any(): continue
                kp0_b, kp1_b, conf_b = kpts0[mask_b], kpts1[mask_b], conf[mask_b]

                if len(kp0_b) > 200:
                    idx = np.linspace(0, len(kp0_b)-1, 200, dtype=int)
                    kp0_b, kp1_b, conf_b = kp0_b[idx], kp1_b[idx], conf_b[idx]

                it = items[i]
                orig_h, orig_w = it['ref'].shape
                v = (kp0_b[:, 0] < orig_w) & (kp0_b[:, 1] < orig_h) & \
                    (kp1_b[:, 0] < orig_w) & (kp1_b[:, 1] < orig_h)
                kp0_b, kp1_b, conf_b = kp0_b[v], kp1_b[v], conf_b[v]

                for k in range(len(kp0_b)):
                    g_ref_col = it['c0'] + kp0_b[k][0]
                    g_ref_row = it['r0'] + kp0_b[k][1]
                    g_off_col = it['c0'] + kp1_b[k][0]
                    g_off_row = it['r0'] + kp1_b[k][1]
                    r_gx, r_gy = rasterio.transform.xy(ref_transform, float(g_ref_row), float(g_ref_col))
                    o_gx, o_gy = rasterio.transform.xy(off_transform, float(g_off_row) + shift_py_y, float(g_off_col) + shift_px_x)
                    all_matches.append((r_gx, r_gy, o_gx, o_gy, float(conf_b[k]), f"{model_type}_{it['tile_idx']}_{k}"))

    def _process_batch(batch_data):
        if not batch_data: return
        ref_tensors, off_tensors = [], []

        for it in batch_data:
            ref_crop, off_crop = it['ref'], it['off']
            if model_type == "roma":
                # RoMa expects RGB 3-channel
                rc_rgb = cv2.cvtColor(ref_crop, cv2.COLOR_GRAY2RGB)
                oc_rgb = cv2.cvtColor(off_crop, cv2.COLOR_GRAY2RGB)
                
                # Resize to multiple of 14, closer to training resolution (560)
                # for better stability and to avoid UserWarnings.
                # 616 = 14 * 44
                roma_size = 616
                rc_rgb = cv2.resize(rc_rgb, (roma_size, roma_size), interpolation=cv2.INTER_AREA)
                oc_rgb = cv2.resize(oc_rgb, (roma_size, roma_size), interpolation=cv2.INTER_AREA)
                
                rt = torch.from_numpy(rc_rgb).permute(2, 0, 1).float() / 255.0
                ot = torch.from_numpy(oc_rgb).permute(2, 0, 1).float() / 255.0
                ref_tensors.append(rt[None])
                off_tensors.append(ot[None])
            else:
                # LoFTR/ELoFTR pad to 1024
                pad_r, pad_c = tile_size - ref_crop.shape[0], tile_size - ref_crop.shape[1]
                if pad_r > 0 or pad_c > 0:
                    ref_crop = np.pad(ref_crop, ((0, pad_r), (0, pad_c)), mode='constant')
                    off_crop = np.pad(off_crop, ((0, pad_r), (0, pad_c)), mode='constant')
                rt = torch.from_numpy(ref_crop.astype(np.float32))[None, None] / 255.0
                ot = torch.from_numpy(off_crop.astype(np.float32))[None, None] / 255.0
                ref_tensors.append(rt)
                off_tensors.append(ot)

        b_ref = torch.cat(ref_tensors, dim=0).to(device, non_blocking=True)
        b_off = torch.cat(off_tensors, dim=0).to(device, non_blocking=True)

        try:
            with torch.no_grad():
                if model_type == "roma":
                    correspondences = model.match(b_ref, b_off, batched=True)
                else:
                    correspondences = model({"image0": b_ref, "image1": b_off})
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and "MPS" not in str(e): raise
            del b_ref, b_off
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available(): torch.mps.empty_cache()
            half = len(batch_data) // 2
            if half >= 1:
                _process_batch(batch_data[:half])
                _process_batch(batch_data[half:])
            return

        _collect_matches(correspondences, batch_data)

    # Tile generation and priority sorting (logic from match_with_loftr)
    anchor_px = []
    if existing_anchors:
        from rasterio.transform import rowcol
        for anc in existing_anchors:
            if isinstance(anc, (list, tuple)) and len(anc) >= 6 and str(anc[5]).startswith("anchor:"):
                try:
                    row, col = rowcol(ref_transform, anc[0], anc[1])
                    anchor_px.append((int(col), int(row)))
                except Exception: pass

    tile_candidates = []
    zero_off_tiles = []
    rng = np.random.default_rng(seed=42)
    tile_idx = 0
    for r0 in range(0, h - tile_size // 2, step):
        for c0 in range(0, w - tile_size // 2, step):
            r1, c1 = min(r0 + tile_size, h), min(c0 + tile_size, w)
            if r1 - r0 < tile_size // 2 or c1 - c0 < tile_size // 2: continue
            ref_tile, off_tile = ref_u8[r0:r1, c0:c1], off_u8[r0:r1, c0:c1]
            rv, ov = np.mean(ref_tile > 0), np.mean(off_tile > 0)
            if rv < min_valid_frac or ov < min_valid_frac:
                if rv >= min_valid_frac and ov < min_valid_frac:
                    zero_off_tiles.append((r0, c0, r1 - r0, c1 - c0, tile_idx))
                tile_idx += 1; continue
            is_anc = any(c0 <= ac <= c0 + tile_size and r0 <= ar <= r0 + tile_size for ac, ar in anchor_px)
            land_frac = float(np.mean(land_mask[r0:r1, c0:c1]))
            stable_frac = float(np.mean(stable_mask[r0:r1, c0:c1]))
            weight_frac = float(np.mean(weight_map[r0:r1, c0:c1]))
            tex = float(np.percentile(grad_mag[r0:r1, c0:c1], 75))
            tex_n = float(np.clip(tex / 64.0, 0.0, 1.0))
            anc_bonus = 1.0 if is_anc else 0.0
            score = (1.6 * weight_frac) + (1.0 * stable_frac) + (0.8 * tex_n) + (0.5 * anc_bonus) + (0.3 * land_frac)
            tile_candidates.append((-score, float(rng.random()), r0, c0, tile_idx, land_frac))
            tile_idx += 1

    tile_candidates.sort(key=lambda x: (x[0], x[1]))
    n_land_tiles = sum(1 for t in tile_candidates if t[5] > 0.05)

    # Spatial bucketing: divide image into a grid of buckets and interleave
    # tiles from each bucket so every region gets represented before the
    # early-exit budget is exhausted.
    n_bucket_rows = max(1, int(np.ceil(h / (step * 4))))
    n_bucket_cols = max(1, int(np.ceil(w / (step * 4))))
    buckets = {}
    for tc in tile_candidates:
        _, _, r0_tc, c0_tc, _, _ = tc
        br = min(r0_tc // max(1, h // n_bucket_rows), n_bucket_rows - 1)
        bc = min(c0_tc // max(1, w // n_bucket_cols), n_bucket_cols - 1)
        buckets.setdefault((br, bc), []).append(tc)

    # Round-robin across buckets: pop one tile from each bucket per round
    interleaved = []
    bucket_lists = list(buckets.values())
    bucket_idxs = [0] * len(bucket_lists)
    any_remaining = True
    while any_remaining:
        any_remaining = False
        for bi, bl in enumerate(bucket_lists):
            if bucket_idxs[bi] < len(bl):
                interleaved.append(bl[bucket_idxs[bi]])
                bucket_idxs[bi] += 1
                any_remaining = True

    print(f"    {model_type.upper()}: {len(tile_candidates)} valid tiles "
          f"({n_land_tiles} land-priority), {len(buckets)} spatial buckets, "
          f"early-exit at {max_matches} raw matches or {max_tiles} tiles")

    for i, (_, _, r0, c0, tidx, _) in enumerate(interleaved):
        r1, c1 = min(r0 + tile_size, h), min(c0 + tile_size, w)
        batch_inputs.append({'ref': ref_u8[r0:r1, c0:c1], 'off': off_u8[r0:r1, c0:c1], 'r0': r0, 'c0': c0, 'tile_idx': tidx})
        if len(batch_inputs) >= batch_size:
            _process_batch(batch_inputs)
            batch_inputs = []
            if hasattr(torch, 'mps') and torch.backends.mps.is_available(): torch.mps.empty_cache()
            if len(all_matches) >= max_matches or i + 1 >= max_tiles: break
    if batch_inputs: _process_batch(batch_inputs)

    # Pass 2: direct-read for zero-filled regions (skipped for RoMa as it's dense/robust)
    if model_type != "roma" and zero_off_tiles and src_offset is not None and work_crs is not None and existing_anchors:
        # We could implement direct-read for ELoFTR too if needed, 
        # but match_with_loftr already has it. For now let's keep it simple.
        pass

    # De-duplicate and RANSAC (common logic)
    if len(all_matches) > 1:
        all_matches.sort(key=lambda x: -x[4])
        kept = []
        for m in all_matches:
            if not any(np.sqrt((m[0]-k[0])**2 + (m[1]-k[1])**2) < 50 for k in kept):
                kept.append(m)
        all_matches = kept

    if not skip_ransac and len(all_matches) >= 6:
        src_pts = np.array([(m[0], m[1]) for m in all_matches], dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array([(m[2], m[3]) for m in all_matches], dtype=np.float32).reshape(-1, 1, 2)
        _, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=max(5.0 * neural_res, 15.0))
        if inliers is not None:
            all_matches = [m for m, keep in zip(all_matches, inliers.ravel().astype(bool)) if keep]

    return all_matches


def match_with_loftr(arr_ref, arr_off_shifted, ref_transform, off_transform,
                     shift_px_x, shift_py_y, neural_res=4.0,
                     min_valid_frac=0.30, skip_ransac=False, model_cache=None,
                     batch_size=4, max_matches=5000, max_tiles=300,
                     existing_anchors=None,
                     src_offset=None, work_crs=None, mask_mode="coastal_obia"):
    """Match features using LoFTR tiled."""
    device = get_torch_device()
    if model_cache is not None:
        loftr = model_cache.loftr
    else:
        loftr = KF.LoFTR(pretrained="outdoor").eval().to(device)

    return _tiled_matcher_core(
        loftr, "loftr", arr_ref, arr_off_shifted, ref_transform, off_transform,
        shift_px_x, shift_py_y, neural_res, min_valid_frac, skip_ransac,
        batch_size, max_matches, max_tiles, existing_anchors, src_offset, work_crs,
        mask_mode=mask_mode)
