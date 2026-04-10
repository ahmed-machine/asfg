"""Dense feature matching: RoMa tiled."""

import os
import numpy as np
import cv2
import rasterio
import rasterio.transform

import torch

from . import constants as _C
from .models import clear_torch_cache, get_torch_device
from .image import clahe_normalize, wallis_match, build_semantic_masks
from .types import MatchPair


def _weight_map_from_bundle(bundle):
    """Compute class weight map from a pre-built MaskBundle (avoids redundant build_semantic_masks call)."""
    weights = np.zeros_like(bundle.land, dtype=np.float32)
    weights += 1.25 * bundle.stable
    weights += 0.75 * bundle.shoreline
    weights += 0.35 * bundle.dark_farmland
    weights -= 0.80 * bundle.shallow_water
    return np.clip(weights, 0.0, 1.5)


def match_with_roma(arr_ref, arr_off_shifted, ref_transform, off_transform,
                    shift_px_x, shift_py_y, neural_res=4.0,
                    min_valid_frac=_C.LAND_MASK_FRAC_MIN, skip_ransac=False, model_cache=None,
                    batch_size=8, max_matches=5000, max_tiles=300,
                    existing_anchors=None,
                    src_offset=None, work_crs=None, mask_mode="coastal_obia"):
    """Match features using RoMa tiled."""
    if model_cache is not None:
        roma = model_cache.roma
        device = model_cache.device
    else:
        return []

    return _tiled_matcher_core(
        roma, "roma", arr_ref, arr_off_shifted, ref_transform, off_transform,
        shift_px_x, shift_py_y, neural_res, min_valid_frac, skip_ransac,
        batch_size, max_matches, max_tiles, existing_anchors, src_offset, work_crs,
        mask_mode=mask_mode, device=device)


def _tiled_matcher_core(model, model_type, arr_ref, arr_off_shifted, ref_transform, off_transform,
                        shift_px_x, shift_py_y, neural_res=4.0,
                        min_valid_frac=_C.LAND_MASK_FRAC_MIN, skip_ransac=False,
                        batch_size=8, max_matches=5000, max_tiles=300,
                        existing_anchors=None,
                        src_offset=None, work_crs=None, mask_mode="coastal_obia",
                        device=None):
    """Tiled matching orchestrator for RoMa."""
    device = device or getattr(model, "device", None) or get_torch_device()
    h, w = arr_ref.shape

    from .params import get_params
    norm_p = get_params().normalization
    if norm_p.wallis_matching:
        arr_off_shifted = wallis_match(arr_off_shifted, arr_ref)
    ref_u8 = clahe_normalize(arr_ref)
    off_u8 = clahe_normalize(arr_off_shifted)
    # Heuristic-only land mask for tile filtering + stability reweight:
    # decoupled from demotion rules D/E which shrink the mask and kill
    # tile coverage (220 → 45 tiles with coastal_obia).
    ref_heuristic_bundle = build_semantic_masks(arr_ref, mode="heuristic")
    land_mask = (ref_heuristic_bundle.land > 0)
    off_land_mask = (build_semantic_masks(arr_off_shifted, mode="heuristic").land > 0)
    # Full semantic bundle still used for stable_mask / weight_map (tile scoring).
    # When mask_mode is already "heuristic", reuse the bundle we just built.
    if mask_mode == "heuristic":
        ref_bundle = ref_heuristic_bundle
    else:
        ref_bundle = build_semantic_masks(arr_ref, mode=mask_mode)
    del ref_heuristic_bundle
    stable_mask = (ref_bundle.stable > 0)
    weight_map = _weight_map_from_bundle(ref_bundle)
    del ref_bundle
    grad_x = cv2.Sobel(ref_u8, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(ref_u8, cv2.CV_32F, 0, 1, ksize=3)
    np.multiply(grad_x, grad_x, out=grad_x)
    np.multiply(grad_y, grad_y, out=grad_y)
    grad_x += grad_y
    del grad_y
    np.sqrt(grad_x, out=grad_x)
    grad_mag = grad_x

    ref_valid = (arr_ref > 0)
    off_valid = (arr_off_shifted > 0)

    scene_valid = arr_ref > 0
    scene_land_frac = float(np.mean((stable_mask | land_mask) & scene_valid)) if np.any(scene_valid) else 0.0
    # Base bump: more tiles from 50% overlap need higher match budget
    max_matches = max(max_matches, 8000)
    max_tiles = max(max_tiles, 420)
    if scene_land_frac < 0.35:
        max_matches = max(max_matches, 10000)
        max_tiles = max(max_tiles, 520)
    if scene_land_frac < 0.25:
        max_matches = max(max_matches, 12000)
        max_tiles = max(max_tiles, 650)

    tile_size = _C.ROMA_TILE_SIZE
    overlap_px = _C.ROMA_TILE_OVERLAP
    step = tile_size - overlap_px

    all_matches = []
    batch_inputs = []
    _tile_fail_count = [0]  # mutable container for nested function access
    _tile_total_count = [0]

    def _collect_matches(correspondences, items):
        # correspondences is a dict from v2's match()
        preds = correspondences

        for i in range(len(items)):
            it = items[i]
            _tile_total_count[0] += 1
            try:
                # Build per-element preds dict for v2's sample()
                preds_i = {k: v[i:i+1] if v is not None else None for k, v in preds.items()}
                m, c, prec_fwd, _ = model.sample(preds_i, num_corresp=_C.ROMA_NUM_CORRESP)
                m = m.cpu().numpy()
                c = c.cpu().numpy()
                # Per-match precision: trace of 2x2 precision matrix
                if prec_fwd is not None:
                    prec_trace = (prec_fwd[:, 0, 0] + prec_fwd[:, 1, 1]).cpu().numpy()
                else:
                    prec_trace = np.ones(len(m), dtype=np.float32)
            except Exception as e:
                _tile_fail_count[0] += 1
                print(f"    RoMa sample error: {e}")
                continue

            # c is post-sigmoid overlap; satast clamps >0.05 to 1.0
            
            # --- SSD EXTRACTION HOOK ---
            # If the environment variable SSD_EXTRACT_DIR is set, dump the raw un-thresholded 
            # neural correspondences to disk for Simple Self-Distillation training.
            ssd_dir = os.environ.get("SSD_EXTRACT_DIR", None)
            if ssd_dir:
                import torch
                os.makedirs(ssd_dir, exist_ok=True)
                ext_id = os.environ.get("SSD_EXTRACT_ID", "default")
                tile_idx = it.get('tile_idx', 0)
                pt_path = os.path.join(ssd_dir, f"roma_raw_{ext_id}_{tile_idx}.pt")
                torch.save({
                    'm': m, 
                    'c': c, 
                    'prec_trace': prec_trace,
                    'tile_meta': {
                        'r0': it['r0'], 'c0': it['c0'], 
                        'h': it['ref'].shape[0], 'w': it['ref'].shape[1]
                    }
                }, pt_path)
            # ---------------------------

            mask = c > 0.55
            m = m[mask]
            c = c[mask]
            prec_trace = prec_trace[mask]

            # RoMa output is normalized [-1, 1]. Project to tile pixel space.
            cur_h, cur_w = it['ref'].shape
            kp0 = (m[:, :2] + 1) / 2 * cur_w
            kp1 = (m[:, 2:] + 1) / 2 * cur_h

            for k in range(len(kp0)):
                global_ref_col = it['c0'] + kp0[k][0]
                global_ref_row = it['r0'] + kp0[k][1]
                global_off_col = it['c0'] + kp1[k][0]
                global_off_row = it['r0'] + kp1[k][1]

                # Fix 1: reject matches where either endpoint is in zero-data
                ref_r = int(round(global_ref_row))
                ref_c = int(round(global_ref_col))
                off_r = int(round(global_off_row))
                off_c = int(round(global_off_col))
                if not (0 <= ref_r < h and 0 <= ref_c < w and ref_valid[ref_r, ref_c]):
                    continue
                if not (0 <= off_r < h and 0 <= off_c < w and off_valid[off_r, off_c]):
                    continue

                ref_gx, ref_gy = rasterio.transform.xy(
                    ref_transform, float(global_ref_row), float(global_ref_col))
                off_gx, off_gy = rasterio.transform.xy(
                    off_transform, float(global_off_row) + shift_py_y,
                    float(global_off_col) + shift_px_x)

                all_matches.append(MatchPair(
                    ref_x=ref_gx, ref_y=ref_gy,
                    off_x=off_gx, off_y=off_gy,
                    confidence=float(c[k]),
                    name=f"roma_{it['tile_idx']}_{k}",
                    precision=float(prec_trace[k])))

    def _run_batch(batch_data):
        """Run model inference on a batch, returning correspondences or None on OOM."""
        ref_tensors, off_tensors = [], []
        for it in batch_data:
            ref_crop, off_crop = it['ref'], it['off']
            rc_rgb = cv2.cvtColor(ref_crop, cv2.COLOR_GRAY2RGB)
            oc_rgb = cv2.cvtColor(off_crop, cv2.COLOR_GRAY2RGB)
            roma_size = _C.ROMA_SIZE
            rc_rgb = cv2.resize(rc_rgb, (roma_size, roma_size), interpolation=cv2.INTER_AREA)
            oc_rgb = cv2.resize(oc_rgb, (roma_size, roma_size), interpolation=cv2.INTER_AREA)
            rt = torch.from_numpy(rc_rgb).permute(2, 0, 1).float() / 255.0
            ot = torch.from_numpy(oc_rgb).permute(2, 0, 1).float() / 255.0
            ref_tensors.append(rt[None])
            off_tensors.append(ot[None])

        b_ref = torch.cat(ref_tensors, dim=0).to(device, non_blocking=True)
        b_off = torch.cat(off_tensors, dim=0).to(device, non_blocking=True)

        try:
            with torch.no_grad():
                model.apply_setting("satast")
                correspondences = model.match(b_ref, b_off)
                return correspondences
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and "MPS" not in str(e): raise
            del b_ref, b_off
            clear_torch_cache(device)
            return None
        finally:
            ref_tensors.clear()
            off_tensors.clear()

    def _process_batch(batch_data):
        """Process a batch with iterative halving on OOM (avoids stack overflow)."""
        if not batch_data: return
        queue = [batch_data]
        while queue:
            chunk = queue.pop(0)
            correspondences = _run_batch(chunk)
            if correspondences is not None:
                _collect_matches(correspondences, chunk)
                del correspondences
                clear_torch_cache(device)
            else:
                half = len(chunk) // 2
                if half >= 1:
                    queue.append(chunk[:half])
                    queue.append(chunk[half:])

    # Tile generation and priority sorting
    anchor_px = []
    if existing_anchors:
        from rasterio.transform import rowcol
        for anc in existing_anchors:
            if isinstance(anc, MatchPair) and anc.is_anchor:
                try:
                    row, col = rowcol(ref_transform, anc.ref_x, anc.ref_y)
                    anchor_px.append((int(col), int(row)))
                except Exception: pass
            elif isinstance(anc, (list, tuple)) and len(anc) >= 6 and str(anc[5]).startswith("anchor:"):
                try:
                    row, col = rowcol(ref_transform, anc[0], anc[1])
                    anchor_px.append((int(col), int(row)))
                except Exception: pass

    tile_candidates = []
    rng = np.random.default_rng(seed=42)
    tile_idx = 0

    def _gen_tiles(row_start, col_start):
        """Generate tile candidates from a grid starting at (row_start, col_start).

        Land filtering is done per 4x4 sub-cell (consistent with the
        downstream quota allocation): a tile is accepted if ANY of its
        sub-cells has joint-land coverage >= SUB_LAND_MIN. This prevents
        the old tile-level TILE_JOINT_LAND_MIN gate from dropping entire
        tiles that contain a small island or a narrow land spit.
        """
        nonlocal tile_idx
        SUB = 4
        SUB_LAND_MIN = 0.15
        for r0 in range(row_start, h - tile_size // 2, step):
            for c0 in range(col_start, w - tile_size // 2, step):
                r1, c1 = min(r0 + tile_size, h), min(c0 + tile_size, w)
                if r1 - r0 < tile_size // 2 or c1 - c0 < tile_size // 2: continue
                ref_tile, off_tile = ref_u8[r0:r1, c0:c1], off_u8[r0:r1, c0:c1]
                rv, ov = np.mean(ref_tile > 0), np.mean(off_tile > 0)
                if rv < min_valid_frac or ov < min_valid_frac:
                    tile_idx += 1; continue
                is_anc = any(c0 <= ac <= c0 + tile_size and r0 <= ar <= r0 + tile_size for ac, ar in anchor_px)

                # Per-sub-cell land evaluation (4x4 grid over the tile).
                ref_tile_land = land_mask[r0:r1, c0:c1]
                off_tile_land = off_land_mask[r0:r1, c0:c1]
                th, tw = ref_tile_land.shape
                sub_h, sub_w = max(1, th // SUB), max(1, tw // SUB)
                any_land = False
                max_sub_joint = 0.0
                for si in range(SUB):
                    sr0 = si * sub_h
                    sr1 = (si + 1) * sub_h if si < SUB - 1 else th
                    for sj in range(SUB):
                        sc0 = sj * sub_w
                        sc1 = (sj + 1) * sub_w if sj < SUB - 1 else tw
                        rl = float(np.mean(ref_tile_land[sr0:sr1, sc0:sc1]))
                        ol = float(np.mean(off_tile_land[sr0:sr1, sc0:sc1]))
                        j = min(rl, ol)
                        if j > max_sub_joint:
                            max_sub_joint = j
                        if j >= SUB_LAND_MIN:
                            any_land = True
                if not any_land:
                    tile_idx += 1; continue

                stable_frac = float(np.mean(stable_mask[r0:r1, c0:c1]))
                weight_frac = float(np.mean(weight_map[r0:r1, c0:c1]))
                tex = float(np.percentile(grad_mag[r0:r1, c0:c1], 75))
                tex_n = float(np.clip(tex / 64.0, 0.0, 1.0))
                anc_bonus = 1.0 if is_anc else 0.0
                # Score with max-sub-joint so tiles with strong land sub-cells
                # rank well even when overall tile land fraction is low.
                score = (1.6 * weight_frac) + (1.0 * stable_frac) + (0.8 * tex_n) + (0.5 * anc_bonus) + (0.3 * max_sub_joint)
                tile_candidates.append((-score, float(rng.random()), r0, c0, tile_idx, max_sub_joint))
                tile_idx += 1

    # Primary grid
    _gen_tiles(0, 0)
    # Offset grid: shifted by half a step so primary-grid edges become offset-grid centers.
    # RoMa hard-zeros confidence near tile edges (romav2.py:431), so the offset grid
    # produces full-confidence matches exactly where the primary grid has gaps.
    offset = step // 2
    _gen_tiles(offset, offset)

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
            if len(all_matches) >= max_matches or i + 1 >= max_tiles: break
    if batch_inputs: _process_batch(batch_inputs)

    # Warn if a significant fraction of tiles failed during sampling
    if _tile_total_count[0] > 0 and _tile_fail_count[0] > 0:
        fail_pct = 100.0 * _tile_fail_count[0] / _tile_total_count[0]
        if fail_pct > 10.0:
            print(f"    WARNING: {_tile_fail_count[0]}/{_tile_total_count[0]} tiles "
                  f"({fail_pct:.0f}%) failed during RoMa sampling")

    # De-duplicate (spatial hash grid) with coverage rescue
    if len(all_matches) > 1:
        all_matches.sort(key=lambda m: -m.confidence)
        occupied = set()
        kept = []
        discarded = []
        for m in all_matches:
            bx, by = int(m.ref_x / 50), int(m.ref_y / 50)
            if not any((bx + di, by + dj) in occupied for di in (-1, 0, 1) for dj in (-1, 0, 1)):
                kept.append(m)
                occupied.add((bx, by))
            else:
                discarded.append(m)

        # Coverage rescue: for empty 2km cells, rescue the best discarded match
        cell_m = _C.MATCH_QUOTA_GRID_M
        kept_cells = set()
        for m in kept:
            kept_cells.add((int(m.ref_x / cell_m), int(m.ref_y / cell_m)))
        rescued = 0
        rescue_cells = set()
        for m in discarded:
            cell = (int(m.ref_x / cell_m), int(m.ref_y / cell_m))
            if cell not in kept_cells and cell not in rescue_cells:
                kept.append(m)
                rescue_cells.add(cell)
                rescued += 1
        if rescued > 0:
            print(f"    De-dup coverage rescue: {rescued} matches in empty cells")
        all_matches = kept

    # Sub-cell spatial distribution quota: inside each 2km cell, group matches
    # by 500m sub-cell (4x4 grid) and keep only the top-2 highest-confidence
    # matches per sub-cell. This trades raw match count for spatial uniformity
    # and quality, so low-confidence regions still get sampled while the global
    # total stays small enough that downstream GCP selection gets better picks.
    if len(all_matches) > 100:
        cell_m = _C.MATCH_QUOTA_GRID_M
        sub_n = 4  # 4x4 sub-cells per 2km cell (500m each)
        sub_m = cell_m / sub_n
        per_subcell_cap = 2  # top-N per sub-cell; fewer, higher-quality matches

        cell_buckets: dict[tuple[int, int], list] = {}
        for m in all_matches:
            cx, cy = int(m.ref_x / cell_m), int(m.ref_y / cell_m)
            cell_buckets.setdefault((cx, cy), []).append(m)

        quota_kept = []
        n_capped = 0
        for cell_matches in cell_buckets.values():
            # Group by sub-cell, then keep top-N by confidence inside each.
            sub_groups: dict[tuple[int, int], list] = {}
            for m in cell_matches:
                sx = int(m.ref_x / sub_m) % sub_n
                sy = int(m.ref_y / sub_m) % sub_n
                sub_groups.setdefault((sx, sy), []).append(m)

            selected = []
            for group in sub_groups.values():
                group.sort(key=lambda mm: -mm.confidence)
                selected.extend(group[:per_subcell_cap])

            if len(cell_matches) > len(selected):
                n_capped += 1
            quota_kept.extend(selected)

        if n_capped > 0:
            print(f"    Sub-cell quota: {len(all_matches)} -> {len(quota_kept)} "
                  f"({n_capped} cells capped, {sub_n}x{sub_n} sub-cells × top-{per_subcell_cap})")
        all_matches = quota_kept

    if not skip_ransac and len(all_matches) >= 6:
        src_pts = np.array([m.ref_coords() for m in all_matches], dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array([m.off_coords() for m in all_matches], dtype=np.float32).reshape(-1, 1, 2)
        _thresh = max(_C.RANSAC_REPROJ_THRESHOLD * neural_res, 15.0)
        from .geo import ransac_affine
        _est_method = get_params().matching.estimation_method
        _, inliers = ransac_affine(src_pts, dst_pts, threshold=_thresh, method=_est_method)
        if inliers is not None:
            all_matches = [m for m, keep in zip(all_matches, inliers) if keep]

    # Stability reweighting (post-RANSAC): down-weight survivors where
    # one or both endpoints fall on water/unstable ground.
    if len(all_matches) > 0:
        reweighted = []
        n_one_land = 0
        n_neither_land = 0
        for m in all_matches:
            ref_row, ref_col = rasterio.transform.rowcol(ref_transform, m.ref_x, m.ref_y)
            off_row, off_col = rasterio.transform.rowcol(off_transform, m.off_x, m.off_y)
            off_row_local = int(off_row) - int(shift_py_y)
            off_col_local = int(off_col) - int(shift_px_x)

            ref_ok = (0 <= int(ref_row) < h and 0 <= int(ref_col) < w
                      and land_mask[int(ref_row), int(ref_col)])
            off_ok = (0 <= off_row_local < h and 0 <= off_col_local < w
                      and off_land_mask[off_row_local, off_col_local])

            if ref_ok and off_ok:
                q = m.confidence
            elif ref_ok or off_ok:
                q = m.confidence * 0.6
                n_one_land += 1
            else:
                q = m.confidence * 0.25
                n_neither_land += 1

            reweighted.append(m.with_confidence(q))

        if n_one_land > 0 or n_neither_land > 0:
            n_pen = n_one_land + n_neither_land
            print(f"    Stability reweight: penalized {n_pen}/{len(all_matches)} "
                  f"(one-land={n_one_land}, neither-land={n_neither_land})")
        all_matches = reweighted

    return all_matches
