"""Generate ASP-compatible .match files from RoMa v2 dense correspondences.

Replaces ASP's CPU-based ipfind + ipmatch with GPU-accelerated neural
matching for inter-frame tie points used by bundle_adjust.  The output
binary .match files follow ASP's IPRecord format and can be consumed
directly via ``bundle_adjust --match-files-prefix``.
"""

import os
import struct

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# ASP binary .match file writer
# ---------------------------------------------------------------------------

def write_asp_match_file(pts_a: np.ndarray, pts_b: np.ndarray,
                         output_path: str) -> None:
    """Write an ASP-format binary .match file.

    The format stores two parallel arrays of IPRecord structs — one per
    image — where index correspondence implies a match (pts_a[i] ↔ pts_b[i]).

    Parameters
    ----------
    pts_a : ndarray of shape (N, 2)
        (col, row) pixel coordinates in image A.
    pts_b : ndarray of shape (N, 2)
        (col, row) pixel coordinates in image B.
    output_path : str
        Path to write the .match file.
    """
    n = len(pts_a)
    assert len(pts_b) == n, "Point arrays must have equal length"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "wb") as f:
        # Header: count of IP records for each image (uint64 LE)
        f.write(struct.pack("<QQ", n, n))

        # IPRecord structs for image A
        for i in range(n):
            _write_ip_record(f, float(pts_a[i, 0]), float(pts_a[i, 1]))

        # IPRecord structs for image B (same index = matched pair)
        for i in range(n):
            _write_ip_record(f, float(pts_b[i, 0]), float(pts_b[i, 1]))


def _write_ip_record(f, x: float, y: float) -> None:
    """Write a single ASP IPRecord struct (little-endian)."""
    f.write(struct.pack("<ff", x, y))            # x, y (float32)
    f.write(struct.pack("<ii", int(round(x)),
                        int(round(y))))           # xi, yi (int32)
    f.write(struct.pack("<fff", 0.0, 1.0, 1.0))  # orientation, scale, interest
    f.write(struct.pack("<B", 0))                 # polarity (uint8)
    f.write(struct.pack("<II", 0, 0))             # octave, scale_lvl (uint32)
    f.write(struct.pack("<Q", 0))                 # desc_length=0 (uint64)


# ---------------------------------------------------------------------------
# Overlap detection from USGS corners
# ---------------------------------------------------------------------------

def _geo_bbox(corners: dict) -> tuple:
    """(west, south, east, north) from a corners dict with NW/NE/SE/SW keys."""
    keys = [k for k in corners if k.upper() in ("NW", "NE", "SE", "SW")]
    lats = [corners[k][0] for k in keys]
    lons = [corners[k][1] for k in keys]
    return (min(lons), min(lats), max(lons), max(lats))


def _image_dims(path: str) -> tuple[int, int]:
    """Return (width, height) of an image."""
    import rasterio
    with rasterio.open(path) as src:
        return src.width, src.height


def _corners_to_pixel_transform(corners: dict, width: int,
                                 height: int) -> np.ndarray:
    """Build a perspective transform from WGS84 (lon, lat) → pixel (col, row).

    Maps the four USGS corners to image corners.
    """
    # Normalise keys to uppercase
    c = {k.upper(): v for k, v in corners.items()}
    src = np.array([
        [c["NW"][1], c["NW"][0]],
        [c["NE"][1], c["NE"][0]],
        [c["SE"][1], c["SE"][0]],
        [c["SW"][1], c["SW"][0]],
    ], dtype=np.float32)
    dst = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def _compute_overlap_crops(corners_a, corners_b, dims_a, dims_b,
                           padding_frac=0.2):
    """Compute pixel crop bounds for the overlap region in each frame.

    Returns ((col0_a, row0_a, col1_a, row1_a),
             (col0_b, row0_b, col1_b, row1_b))
    or (None, None) if no significant overlap.
    """
    bbox_a = _geo_bbox(corners_a)
    bbox_b = _geo_bbox(corners_b)

    # Geographic intersection
    west = max(bbox_a[0], bbox_b[0])
    south = max(bbox_a[1], bbox_b[1])
    east = min(bbox_a[2], bbox_b[2])
    north = min(bbox_a[3], bbox_b[3])
    if west >= east or south >= north:
        return None, None

    # Pad to account for corner inaccuracy
    pad_lon = (east - west) * padding_frac
    pad_lat = (north - south) * padding_frac
    west -= pad_lon
    south -= pad_lat
    east += pad_lon
    north += pad_lat

    overlap_geo = np.array([
        [west, south], [east, south], [east, north], [west, north]
    ], dtype=np.float32).reshape(1, -1, 2)

    w_a, h_a = dims_a
    w_b, h_b = dims_b
    M_a = _corners_to_pixel_transform(corners_a, w_a, h_a)
    M_b = _corners_to_pixel_transform(corners_b, w_b, h_b)

    px_a = cv2.perspectiveTransform(overlap_geo, M_a)[0]
    px_b = cv2.perspectiveTransform(overlap_geo, M_b)[0]

    crop_a = _clamp_crop(px_a, w_a, h_a)
    crop_b = _clamp_crop(px_b, w_b, h_b)

    if crop_a is None or crop_b is None:
        return None, None

    # Reject if either crop is tiny (< 256 px in any dimension)
    if (crop_a[2] - crop_a[0] < 256 or crop_a[3] - crop_a[1] < 256 or
            crop_b[2] - crop_b[0] < 256 or crop_b[3] - crop_b[1] < 256):
        return None, None

    return crop_a, crop_b


def _clamp_crop(pts: np.ndarray, width: int,
                height: int) -> tuple | None:
    """Axis-aligned bounding box from projected points, clamped to image."""
    col0 = max(0, int(np.floor(pts[:, 0].min())))
    row0 = max(0, int(np.floor(pts[:, 1].min())))
    col1 = min(width, int(np.ceil(pts[:, 0].max())))
    row1 = min(height, int(np.ceil(pts[:, 1].max())))
    if col1 <= col0 or row1 <= row0:
        return None
    return (col0, row0, col1, row1)


# ---------------------------------------------------------------------------
# Image reading and normalisation
# ---------------------------------------------------------------------------

def _read_crop(path: str, crop: tuple) -> np.ndarray:
    """Read a pixel crop from a single-band image.

    crop is (col0, row0, col1, row1).
    Returns a float32 array.
    """
    import rasterio
    col0, row0, col1, row1 = crop
    with rasterio.open(path) as src:
        window = rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)
        arr = src.read(1, window=window)
    return arr.astype(np.float32)


def _clahe_u8(arr: np.ndarray) -> np.ndarray:
    """CLAHE-normalise a float/uint16 array to uint8."""
    valid = arr[arr > 0]
    if valid.size < 100:
        return np.zeros_like(arr, dtype=np.uint8)
    lo, hi = np.percentile(valid, [1, 99])
    if hi <= lo:
        hi = lo + 1
    stretched = np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(stretched)


# ---------------------------------------------------------------------------
# Tiled RoMa matching on overlap crops
# ---------------------------------------------------------------------------

_TILE_SIZE = 1024
_TILE_OVERLAP = 512
_ROMA_SIZE = 640
_NUM_CORRESP = 600
_CONFIDENCE_THRESH = 0.55
_MIN_VALID_FRAC = 0.30


def _run_roma_tiled(crop_a: np.ndarray, crop_b: np.ndarray, model,
                    device: str, batch_size: int = 8,
                    max_matches: int = 5000,
                    max_tiles: int = 300) -> tuple:
    """Run tiled RoMa matching on two overlap crops.

    Returns (pts_a, pts_b, confidences) in crop-local pixel coordinates,
    or (None, None, None) if too few matches.
    """
    import torch

    a_u8 = _clahe_u8(crop_a)
    b_u8 = _clahe_u8(crop_b)

    # Resize to common dimensions (minimum of each axis)
    h = min(a_u8.shape[0], b_u8.shape[0])
    w = min(a_u8.shape[1], b_u8.shape[1])
    if a_u8.shape != (h, w):
        a_u8 = cv2.resize(a_u8, (w, h), interpolation=cv2.INTER_AREA)
    if b_u8.shape != (h, w):
        b_u8 = cv2.resize(b_u8, (w, h), interpolation=cv2.INTER_AREA)

    a_valid = a_u8 > 0
    b_valid = b_u8 > 0

    step = _TILE_SIZE - _TILE_OVERLAP
    all_pts_a, all_pts_b, all_conf = [], [], []

    # -- match collection callback --
    def _collect(correspondences, items):
        for i, it in enumerate(items):
            try:
                preds_i = {k: v[i:i+1] if v is not None else None
                           for k, v in correspondences.items()}
                m, c, _, _ = model.sample(preds_i, num_corresp=_NUM_CORRESP)
                m = m.cpu().numpy()
                c = c.cpu().numpy()
            except Exception as e:
                print(f"    [MatchIP] sample error: {e}")
                continue

            mask = c > _CONFIDENCE_THRESH
            m, c = m[mask], c[mask]
            if len(m) == 0:
                continue

            cur_h, cur_w = it["h"], it["w"]
            # RoMa normalised [-1,1] → tile pixel coords
            kp_a = (m[:, :2] + 1) / 2 * np.array([cur_w, cur_h])
            kp_b = (m[:, 2:] + 1) / 2 * np.array([cur_w, cur_h])

            for k in range(len(kp_a)):
                ga_col = it["c0"] + kp_a[k, 0]
                ga_row = it["r0"] + kp_a[k, 1]
                gb_col = it["c0"] + kp_b[k, 0]
                gb_row = it["r0"] + kp_b[k, 1]

                # Bounds and validity check
                ra, ca = int(round(ga_row)), int(round(ga_col))
                rb, cb = int(round(gb_row)), int(round(gb_col))
                if not (0 <= ra < h and 0 <= ca < w and a_valid[ra, ca]):
                    continue
                if not (0 <= rb < h and 0 <= cb < w and b_valid[rb, cb]):
                    continue

                all_pts_a.append([ga_col, ga_row])
                all_pts_b.append([gb_col, gb_row])
                all_conf.append(float(c[k]))

    # -- batch inference with OOM halving --
    def _run_batch(batch_data):
        ref_t, off_t = [], []
        for it in batch_data:
            rc = cv2.cvtColor(it["ref"], cv2.COLOR_GRAY2RGB)
            oc = cv2.cvtColor(it["off"], cv2.COLOR_GRAY2RGB)
            rc = cv2.resize(rc, (_ROMA_SIZE, _ROMA_SIZE),
                            interpolation=cv2.INTER_AREA)
            oc = cv2.resize(oc, (_ROMA_SIZE, _ROMA_SIZE),
                            interpolation=cv2.INTER_AREA)
            ref_t.append(torch.from_numpy(rc).permute(2, 0, 1).float()[None] / 255.0)
            off_t.append(torch.from_numpy(oc).permute(2, 0, 1).float()[None] / 255.0)

        b_ref = torch.cat(ref_t).to(device, non_blocking=True)
        b_off = torch.cat(off_t).to(device, non_blocking=True)
        try:
            with torch.no_grad():
                model.apply_setting("satast")
                return model.match(b_ref, b_off)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and "MPS" not in str(e):
                raise
            del b_ref, b_off
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return None

    def _process_batch(batch_data):
        if not batch_data:
            return
        queue = [batch_data]
        while queue:
            chunk = queue.pop(0)
            corr = _run_batch(chunk)
            if corr is not None:
                _collect(corr, chunk)
            else:
                half = len(chunk) // 2
                if half >= 1:
                    queue.append(chunk[:half])
                    queue.append(chunk[half:])

    # -- tile generation --
    batch_inputs = []
    tiles_processed = 0
    for r0 in range(0, h - _TILE_SIZE // 2, step):
        for c0 in range(0, w - _TILE_SIZE // 2, step):
            r1 = min(r0 + _TILE_SIZE, h)
            c1 = min(c0 + _TILE_SIZE, w)
            if r1 - r0 < _TILE_SIZE // 2 or c1 - c0 < _TILE_SIZE // 2:
                continue

            a_tile = a_u8[r0:r1, c0:c1]
            b_tile = b_u8[r0:r1, c0:c1]
            if np.mean(a_tile > 0) < _MIN_VALID_FRAC:
                continue
            if np.mean(b_tile > 0) < _MIN_VALID_FRAC:
                continue

            batch_inputs.append({
                "ref": a_tile, "off": b_tile,
                "r0": r0, "c0": c0,
                "h": r1 - r0, "w": c1 - c0,
            })
            if len(batch_inputs) >= batch_size:
                _process_batch(batch_inputs)
                batch_inputs = []
                tiles_processed += batch_size
                if len(all_pts_a) >= max_matches or tiles_processed >= max_tiles:
                    break
        if len(all_pts_a) >= max_matches or tiles_processed >= max_tiles:
            break
    if batch_inputs:
        _process_batch(batch_inputs)

    if len(all_pts_a) < 6:
        return None, None, None

    return (np.array(all_pts_a, dtype=np.float32),
            np.array(all_pts_b, dtype=np.float32),
            np.array(all_conf, dtype=np.float32))


# ---------------------------------------------------------------------------
# Per-pair match file generation
# ---------------------------------------------------------------------------

def generate_pair_matches(frame_a: str, frame_b: str,
                          corners_a: dict, corners_b: dict,
                          output_path: str,
                          max_matches: int = 3000,
                          model_cache=None) -> str | None:
    """Generate an ASP .match file for one pair of overlapping frames.

    Parameters
    ----------
    frame_a, frame_b : str
        Paths to raw/stitched frame images.
    corners_a, corners_b : dict
        Per-frame USGS corners ``{'NW': (lat, lon), ...}``.
    output_path : str
        Where to write the .match file.
    max_matches : int
        Maximum matches to retain after RANSAC and de-duplication.
    model_cache : ModelCache, optional
        Shared model cache.  Created internally if None.

    Returns
    -------
    str or None
        Path to written .match file, or None on failure.
    """
    dims_a = _image_dims(frame_a)
    dims_b = _image_dims(frame_b)

    crop_a, crop_b = _compute_overlap_crops(corners_a, corners_b,
                                            dims_a, dims_b)
    if crop_a is None:
        print(f"  [MatchIP] No overlap between {os.path.basename(frame_a)} "
              f"and {os.path.basename(frame_b)}")
        return None

    print(f"  [MatchIP] Overlap crops: A={_crop_str(crop_a)} B={_crop_str(crop_b)}")

    arr_a = _read_crop(frame_a, crop_a)
    arr_b = _read_crop(frame_b, crop_b)

    # Lazy-load RoMa if no cache provided
    own_cache = False
    if model_cache is None:
        from align.models import ModelCache, get_torch_device
        device = get_torch_device()
        model_cache = ModelCache(device)
        own_cache = True

    device = model_cache.device
    model = model_cache.roma

    pts_a, pts_b, conf = _run_roma_tiled(arr_a, arr_b, model, device,
                                         max_matches=max_matches)

    if pts_a is None:
        print(f"  [MatchIP] Too few matches for pair")
        if own_cache:
            model_cache.close()
        return None

    # RANSAC affine filter
    src = pts_a.reshape(-1, 1, 2)
    dst = pts_b.reshape(-1, 1, 2)
    _, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=8.0)

    if inliers is None or inliers.sum() < 10:
        print(f"  [MatchIP] RANSAC failed — {0 if inliers is None else int(inliers.sum())} inliers")
        if own_cache:
            model_cache.close()
        return None

    mask = inliers.ravel().astype(bool)
    pts_a, pts_b, conf = pts_a[mask], pts_b[mask], conf[mask]

    # Spatial de-duplication: keep best per 50 px cell
    pts_a, pts_b, conf = _dedup_spatial(pts_a, pts_b, conf, cell_px=50)

    # Cap at max_matches (keep highest confidence)
    if len(pts_a) > max_matches:
        order = np.argsort(-conf)[:max_matches]
        pts_a, pts_b, conf = pts_a[order], pts_b[order], conf[order]

    # Transform from crop-local coords to full raw image coords.
    # The two crops may have different sizes if the overlap maps to different
    # pixel extents in each frame.  The RoMa matching was done on crops
    # resized to a common size, so we need to scale back to each crop's
    # original pixel extent.
    crop_a_w = crop_a[2] - crop_a[0]
    crop_a_h = crop_a[3] - crop_a[1]
    crop_b_w = crop_b[2] - crop_b[0]
    crop_b_h = crop_b[3] - crop_b[1]

    # The common size used in _run_roma_tiled
    common_w = min(arr_a.shape[1], arr_b.shape[1])
    common_h = min(arr_a.shape[0], arr_b.shape[0])

    # Scale pts_a from common coords to crop_a original coords, then offset
    full_a = pts_a.copy()
    full_a[:, 0] = pts_a[:, 0] * (crop_a_w / common_w) + crop_a[0]
    full_a[:, 1] = pts_a[:, 1] * (crop_a_h / common_h) + crop_a[1]

    # Scale pts_b from common coords to crop_b original coords, then offset
    full_b = pts_b.copy()
    full_b[:, 0] = pts_b[:, 0] * (crop_b_w / common_w) + crop_b[0]
    full_b[:, 1] = pts_b[:, 1] * (crop_b_h / common_h) + crop_b[1]

    write_asp_match_file(full_a, full_b, output_path)
    print(f"  [MatchIP] Wrote {len(full_a)} matches → {os.path.basename(output_path)}")

    if own_cache:
        model_cache.close()

    return output_path


def _crop_str(crop: tuple) -> str:
    """Human-readable crop string."""
    return f"[{crop[0]}:{crop[2]}, {crop[1]}:{crop[3]}]"


def _dedup_spatial(pts_a, pts_b, conf, cell_px=50):
    """Keep highest-confidence match per spatial cell (based on pts_a)."""
    cells = {}
    for i in range(len(pts_a)):
        key = (int(pts_a[i, 0] // cell_px), int(pts_a[i, 1] // cell_px))
        if key not in cells or conf[i] > conf[cells[key]]:
            cells[key] = i
    idx = sorted(cells.values())
    return pts_a[idx], pts_b[idx], conf[idx]


# ---------------------------------------------------------------------------
# Strip-level match file generation
# ---------------------------------------------------------------------------

def asp_match_filename(prefix: str, frame_a: str, frame_b: str) -> str:
    """ASP match file naming convention: <prefix>-<stemA>__<stemB>.match"""
    stem_a = os.path.splitext(os.path.basename(frame_a))[0]
    stem_b = os.path.splitext(os.path.basename(frame_b))[0]
    return f"{prefix}-{stem_a}__{stem_b}.match"


def generate_strip_matches(frames: list[str],
                           corners_list: list[dict],
                           output_dir: str,
                           match_prefix: str = "roma",
                           max_matches_per_pair: int = 3000) -> str | None:
    """Generate ASP .match files for all adjacent frame pairs in a strip.

    Parameters
    ----------
    frames : list of str
        Paths to raw/stitched frame images in strip order.
    corners_list : list of dict
        Per-frame USGS corners in strip order.
    output_dir : str
        Directory to write .match files.
    match_prefix : str
        Prefix for match file naming.
    max_matches_per_pair : int
        Maximum matches to retain per frame pair.

    Returns
    -------
    str or None
        The full match prefix path for ``--match-files-prefix``,
        or None if no match files were produced.
    """
    if len(frames) < 2:
        print("  [MatchIP] Need at least 2 frames for inter-frame matching")
        return None

    os.makedirs(output_dir, exist_ok=True)
    prefix_path = os.path.join(output_dir, match_prefix)

    # Share a single ModelCache across all pairs
    from align.models import ModelCache, get_torch_device
    device = get_torch_device()
    cache = ModelCache(device)

    produced = 0
    for i in range(len(frames) - 1):
        fa, fb = frames[i], frames[i + 1]
        ca, cb = corners_list[i], corners_list[i + 1]

        match_name = asp_match_filename(match_prefix, fa, fb)
        match_path = os.path.join(output_dir, match_name)

        if os.path.exists(match_path):
            print(f"  [MatchIP] Skipping existing: {match_name}")
            produced += 1
            continue

        print(f"  [MatchIP] Matching pair {i+1}/{len(frames)-1}: "
              f"{os.path.basename(fa)} ↔ {os.path.basename(fb)}")

        result = generate_pair_matches(
            fa, fb, ca, cb, match_path,
            max_matches=max_matches_per_pair,
            model_cache=cache,
        )
        if result is not None:
            produced += 1

    cache.close()

    if produced == 0:
        print("  [MatchIP] No match files produced")
        return None

    print(f"  [MatchIP] Generated {produced}/{len(frames)-1} match files "
          f"→ prefix={prefix_path}")
    return prefix_path
