"""Frame stitching for multi-frame satellite imagery strips.

Generalized from process_d3c.py. Handles:
- Unordered frame layout discovery via all-pairs SIFT matching
- 180-degree rotation detection and correction
- VRT-based stitching (memory-safe for large strips)
- Graph-based ordering (no assumption about input frame sequence)
"""

import json
import os
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import run_gdal_cmd as _run_cmd


def flip_frame_180(src_path: str, dst_path: str):
    """Rotate a frame 180 degrees (flip both axes) and save as compressed TIFF."""
    from osgeo import gdal
    import numpy as np
    gdal.UseExceptions()

    ds = gdal.Open(src_path)
    w, h, n_bands = ds.RasterXSize, ds.RasterYSize, ds.RasterCount
    dt = ds.GetRasterBand(1).DataType
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(dst_path, w, h, n_bands, dt,
                           ["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES"])
    for b in range(1, n_bands + 1):
        arr = ds.GetRasterBand(b).ReadAsArray()
        out_ds.GetRasterBand(b).WriteArray(arr[::-1, ::-1])
    out_ds.FlushCache()
    out_ds = None
    ds = None


def _ncc_overlap(frame_a_path: str, frame_b_path: str, rotated: bool = False,
                  scale: float = 0.05) -> int:
    """Fallback overlap detection using NCC template matching.

    When SIFT fails (e.g. featureless desert), extract a narrow vertical
    strip from the right edge of A and slide it across the left portion
    of B to find where it matches.

    Returns overlap in pixels at full resolution, or 0 on failure.
    """
    from osgeo import gdal
    import numpy as np
    import cv2

    gdal.UseExceptions()

    ds_a = gdal.Open(frame_a_path)
    ds_b = gdal.Open(frame_b_path)
    w_a, h_a = ds_a.RasterXSize, ds_a.RasterYSize
    w_b, h_b = ds_b.RasterXSize, ds_b.RasterYSize

    out_h = int(min(h_a, h_b) * scale)

    # Template: a narrow strip from the rightmost 5% of A
    tmpl_frac = 0.05
    tmpl_w_full = int(w_a * tmpl_frac)
    tmpl_w = max(20, int(tmpl_w_full * scale))

    tmpl = ds_a.GetRasterBand(1).ReadAsArray(
        xoff=w_a - tmpl_w_full, yoff=0, win_xsize=tmpl_w_full, win_ysize=h_a,
        buf_xsize=tmpl_w, buf_ysize=out_h,
    )

    # Search area: left 50% of B
    search_frac = 0.50
    search_w_full = int(w_b * search_frac)
    search_w = max(tmpl_w + 10, int(search_w_full * scale))

    if rotated:
        # Read right edge of B, then flip
        search_img = ds_b.GetRasterBand(1).ReadAsArray(
            xoff=w_b - search_w_full, yoff=0, win_xsize=search_w_full, win_ysize=h_b,
            buf_xsize=search_w, buf_ysize=out_h,
        )
        if search_img is not None:
            search_img = search_img[::-1, ::-1]
    else:
        search_img = ds_b.GetRasterBand(1).ReadAsArray(
            xoff=0, yoff=0, win_xsize=search_w_full, win_ysize=h_b,
            buf_xsize=search_w, buf_ysize=out_h,
        )

    ds_a = ds_b = None
    if tmpl is None or search_img is None:
        return 0

    tmpl = tmpl.astype(np.uint8)
    search_img = search_img.astype(np.uint8)

    # Apply CLAHE to both for better matching
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    tmpl_eq = clahe.apply(tmpl)
    search_eq = clahe.apply(search_img)

    # NCC template matching
    if tmpl_eq.shape[0] > search_eq.shape[0] or tmpl_eq.shape[1] > search_eq.shape[1]:
        return 0

    res = cv2.matchTemplate(search_eq, tmpl_eq, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val < 0.3:
        return 0

    # max_loc[0] is where the template starts in the search area (reduced coords)
    # This means the right edge of A starts at position max_loc[0] in B's left half
    # The overlap = (position of template in B + template width) at full resolution
    # But we need: overlap = tmpl position in B coords at full scale
    # match_x_full = position of template start in B at full resolution
    match_x_full = round(max_loc[0] / scale * (search_w_full / (search_w / scale)))

    # Actually simpler: match position in full-res B coords
    match_x_full = round(max_loc[0] * search_w_full / search_w)

    # Overlap = how far into B frame A extends
    # The template is from A's right edge (last 5%), and it was found at match_x_full in B
    # So the overlap is: tmpl_w_full + match_x_full (since template starts at A_rightedge - tmpl_w)
    # Actually: overlap = (w_a_rightedge maps to B position match_x_full + tmpl_w_full)
    overlap_px = match_x_full + tmpl_w_full

    if overlap_px < 0 or overlap_px > w_a * 0.5:
        return 0

    print(f"    NCC fallback: overlap={overlap_px}px ({100 * overlap_px / w_a:.1f}%), "
          f"corr={max_val:.3f}")
    return overlap_px


def compute_overlap(frame_a_path: str, frame_b_path: str, scale: float = 0.10) -> tuple:
    """Compute horizontal overlap between two adjacent frames using SIFT.

    Falls back to NCC template matching if SIFT doesn't find enough matches.

    Returns (overlap_px, frame_b_is_rotated, match_count).
    """
    from osgeo import gdal
    import numpy as np
    import cv2

    gdal.UseExceptions()

    ds_a = gdal.Open(frame_a_path)
    ds_b = gdal.Open(frame_b_path)
    w_a, h_a = ds_a.RasterXSize, ds_a.RasterYSize
    w_b, h_b = ds_b.RasterXSize, ds_b.RasterYSize

    strip_frac = 0.30
    strip_w_a = int(w_a * strip_frac)
    strip_w_b = int(w_b * strip_frac)

    out_w_a = int(strip_w_a * scale)
    out_h_a = int(h_a * scale)
    out_w_b = int(strip_w_b * scale)
    out_h_b = int(h_b * scale)

    band_a = ds_a.GetRasterBand(1)
    band_b = ds_b.GetRasterBand(1)

    # Right edge of A
    img_a = band_a.ReadAsArray(
        xoff=w_a - strip_w_a, yoff=0, win_xsize=strip_w_a, win_ysize=h_a,
        buf_xsize=out_w_a, buf_ysize=out_h_a,
    )
    # Left edge of B (normal orientation)
    img_b = band_b.ReadAsArray(
        xoff=0, yoff=0, win_xsize=strip_w_b, win_ysize=h_b,
        buf_xsize=out_w_b, buf_ysize=out_h_b,
    )

    if img_a is None or img_b is None:
        ds_a = ds_b = None
        return (0, False, 0)

    img_a = img_a.astype(np.uint8)
    img_b = img_b.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_a_eq = clahe.apply(img_a)
    img_b_eq = clahe.apply(img_b)

    sift = cv2.SIFT_create(nfeatures=5000)
    kp_a, desc_a = sift.detectAndCompute(img_a_eq, None)
    kp_b, desc_b = sift.detectAndCompute(img_b_eq, None)

    if desc_a is None or desc_b is None or len(kp_a) < 10 or len(kp_b) < 10:
        ds_a = ds_b = None
        return (0, False, 0)

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
        dict(checks=50),
    )
    raw_matches = flann.knnMatch(desc_b, desc_a, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                good.append(m)

    if len(good) >= 10:
        ds_a = ds_b = None
        pts_b = np.float32([kp_b[m.queryIdx].pt for m in good])
        pts_a = np.float32([kp_a[m.trainIdx].pt for m in good])

        dx_values = []
        for pa, pb in zip(pts_a, pts_b):
            overlap_px = strip_w_a - pa[0] / scale + pb[0] / scale
            dx_values.append(overlap_px)

        median_overlap = round(np.median(dx_values))
        if median_overlap < 0 or median_overlap > w_a * 0.5:
            return (0, False, 0)

        print(f"    Matches: {len(good)}, median overlap: {median_overlap}px "
              f"({100 * median_overlap / w_a:.1f}%)")
        return (median_overlap, False, len(good))

    # Try with B rotated 180 degrees
    print(f"    Only {len(good)} matches in normal orientation, trying 180 rotation...")

    strip_w_b_right = int(w_b * strip_frac)
    out_w_b_right = int(strip_w_b_right * scale)
    out_h_b_right = int(h_b * scale)

    img_b_right = band_b.ReadAsArray(
        xoff=w_b - strip_w_b_right, yoff=0,
        win_xsize=strip_w_b_right, win_ysize=h_b,
        buf_xsize=out_w_b_right, buf_ysize=out_h_b_right,
    )
    ds_a = ds_b = None

    if img_b_right is None:
        return (0, False, 0)

    img_b_rot = img_b_right[::-1, ::-1].astype(np.uint8)
    img_b_rot_eq = clahe.apply(img_b_rot)

    kp_b_rot, desc_b_rot = sift.detectAndCompute(img_b_rot_eq, None)
    if desc_b_rot is None or len(kp_b_rot) < 10:
        return (0, False, 0)

    raw_matches_rot = flann.knnMatch(desc_b_rot, desc_a, k=2)
    good_rot = []
    for pair in raw_matches_rot:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                good_rot.append(m)

    if len(good_rot) < 10:
        # SIFT failed in both orientations — try NCC fallback
        print(f"    Only {len(good_rot)} rotated matches. Trying NCC fallback...")
        # Try normal orientation first
        ncc_overlap = _ncc_overlap(frame_a_path, frame_b_path, rotated=False)
        if ncc_overlap > 0:
            return (ncc_overlap, False, 0)
        # Try rotated
        ncc_overlap = _ncc_overlap(frame_a_path, frame_b_path, rotated=True)
        if ncc_overlap > 0:
            return (ncc_overlap, True, 0)
        # Try reverse direction (B->A) — frames might be in wrong order
        ncc_overlap = _ncc_overlap(frame_b_path, frame_a_path, rotated=False)
        if ncc_overlap > 0:
            print(f"    NOTE: Reverse direction B->A worked — frames may be misordered")
            return (ncc_overlap, False, 0)
        return (0, False, 0)

    pts_b_rot = np.float32([kp_b_rot[m.queryIdx].pt for m in good_rot])
    pts_a_rot = np.float32([kp_a[m.trainIdx].pt for m in good_rot])

    dx_values = []
    for pa, pb in zip(pts_a_rot, pts_b_rot):
        overlap_px = strip_w_a - pa[0] / scale + pb[0] / scale
        dx_values.append(overlap_px)

    median_overlap = round(np.median(dx_values))
    if median_overlap < 0 or median_overlap > w_a * 0.5:
        return (0, False, 0)

    print(f"    ROTATED matches: {len(good_rot)}, median overlap: {median_overlap}px "
          f"({100 * median_overlap / w_a:.1f}%)")
    return (median_overlap, True, len(good_rot))


def _blend_overlaps(output_path, frame_info, x_offsets, overlaps):
    """Post-process overlap zones with exposure matching and linear feathered blending.

    For each overlap zone:
    1. Compute gain+offset correction between adjacent frames (exposure matching)
    2. Apply correction to frame A's overlap region
    3. Linear alpha blend between corrected frame A and frame B
    """
    from osgeo import gdal
    import numpy as np

    ds_out = gdal.Open(output_path, gdal.GA_Update)
    n_bands = ds_out.RasterCount
    h = ds_out.RasterYSize
    chunk_h = 1024

    for i in range(len(overlaps)):
        overlap_w = overlaps[i]
        if overlap_w <= 0:
            continue

        zone_start = x_offsets[i + 1]  # where frame B starts in output

        # Open frame A source to read its right edge
        ds_a = gdal.Open(frame_info[i]["path"])
        a_xoff = frame_info[i]["w"] - overlap_w
        a_h = ds_a.RasterYSize

        # --- Exposure matching: compute gain+offset from overlap statistics ---
        # Sample a vertical stripe from the overlap zone of both frames
        sample_rows = min(a_h, h)
        sample_scale = max(1, sample_rows // 512)  # downsample for speed
        sample_h = sample_rows // sample_scale

        gains = []
        offsets_corr = []
        for b in range(1, n_bands + 1):
            samp_a = ds_a.GetRasterBand(b).ReadAsArray(
                xoff=a_xoff, yoff=0, win_xsize=overlap_w, win_ysize=sample_rows,
                buf_xsize=overlap_w // max(1, overlap_w // 64), buf_ysize=sample_h,
            ).astype(np.float32)
            samp_b = ds_out.GetRasterBand(b).ReadAsArray(
                xoff=zone_start, yoff=0, win_xsize=overlap_w, win_ysize=sample_rows,
                buf_xsize=overlap_w // max(1, overlap_w // 64), buf_ysize=sample_h,
            ).astype(np.float32)

            # Only use non-black pixels
            valid = (samp_a > 5) & (samp_b > 5)
            if valid.sum() < 100:
                gains.append(1.0)
                offsets_corr.append(0.0)
                continue

            mean_a = samp_a[valid].mean()
            mean_b = samp_b[valid].mean()
            std_a = samp_a[valid].std()
            std_b = samp_b[valid].std()

            if std_a < 1 or std_b < 1:
                gains.append(1.0)
                offsets_corr.append(0.0)
                continue

            # Match frame A to frame B (B is already in the output)
            gain = std_b / std_a
            gain = max(0.7, min(1.4, gain))  # clamp
            offset = mean_b - gain * mean_a
            offset = max(-30, min(30, offset))
            gains.append(gain)
            offsets_corr.append(offset)

        exposure_needed = any(abs(g - 1.0) > 0.01 or abs(o) > 0.5
                              for g, o in zip(gains, offsets_corr))
        if exposure_needed:
            print(f"    Exposure correction overlap {i}: "
                  f"gain={gains[0]:.3f}, offset={offsets_corr[0]:.1f}")

        # --- Feathered blending with exposure correction ---
        alpha = np.linspace(1.0, 0.0, overlap_w, dtype=np.float32)

        for y in range(0, h, chunk_h):
            rows = min(chunk_h, h - y)
            a_rows = min(rows, max(0, a_h - y))
            if a_rows <= 0:
                break
            for b in range(1, n_bands + 1):
                strip_a = ds_a.GetRasterBand(b).ReadAsArray(
                    xoff=a_xoff, yoff=y, win_xsize=overlap_w, win_ysize=a_rows
                ).astype(np.float32)
                if a_rows < rows:
                    padded = np.zeros((rows, overlap_w), dtype=np.float32)
                    padded[:a_rows, :] = strip_a
                    strip_a = padded

                # Apply exposure correction to frame A
                if exposure_needed:
                    valid_px = strip_a > 0
                    strip_a[valid_px] = np.clip(
                        strip_a[valid_px] * gains[b - 1] + offsets_corr[b - 1], 0, 255)

                strip_b = ds_out.GetRasterBand(b).ReadAsArray(
                    xoff=zone_start, yoff=y, win_xsize=overlap_w, win_ysize=rows
                ).astype(np.float32)
                blended = strip_a * alpha[np.newaxis, :] + strip_b * (1 - alpha[np.newaxis, :])
                ds_out.GetRasterBand(b).WriteArray(
                    blended.astype(np.uint8), xoff=zone_start, yoff=y
                )
        ds_a = None
        print(f"    Blended overlap {i}: {overlap_w}px wide")

    ds_out.FlushCache()
    ds_out = None


def crop_black_borders(frame_path: str, output_dir: str) -> tuple:
    """Detect and crop black scanning borders from a raw satellite frame.

    Raw declassified film frames have black borders from scanning that vary
    by camera system. These borders hurt SIFT matching in overlap detection.

    Returns (cropped_path, crop_box) where crop_box is (x, y, w, h) in
    full-res pixels, or (original_path, None) if no significant borders found.
    """
    from osgeo import gdal
    import numpy as np
    import cv2
    from scipy.ndimage import binary_fill_holes

    gdal.UseExceptions()

    ds = gdal.Open(frame_path)
    if ds is None:
        return (frame_path, None)

    full_w, full_h = ds.RasterXSize, ds.RasterYSize

    # Read band 1 at ~2% scale
    scale = 0.02
    small_w = max(100, int(full_w * scale))
    small_h = max(100, int(full_h * scale))

    img = ds.GetRasterBand(1).ReadAsArray(
        xoff=0, yoff=0, win_xsize=full_w, win_ysize=full_h,
        buf_xsize=small_w, buf_ysize=small_h,
    )
    ds = None

    if img is None:
        return (frame_path, None)

    # Binary threshold: pixel > 10 is valid content
    mask = (img > 10).astype(np.uint8)

    # Morphological close to fill dark interior features (water, shadows)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Morphological open to remove noise specks in border
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Fill holes — keeps only border-connected dark regions as "border"
    mask = binary_fill_holes(mask).astype(np.uint8)

    # Find bounding box of valid region
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return (frame_path, None)

    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Convert to full-res coordinates
    x_min = int(col_min * full_w / small_w)
    x_max = int((col_max + 1) * full_w / small_w)
    y_min = int(row_min * full_h / small_h)
    y_max = int((row_max + 1) * full_h / small_h)

    # Check if borders are negligible (< 0.5% on all sides)
    left_pct = x_min / full_w
    right_pct = (full_w - x_max) / full_w
    top_pct = y_min / full_h
    bottom_pct = (full_h - y_max) / full_h

    if left_pct < 0.005 and right_pct < 0.005 and top_pct < 0.005 and bottom_pct < 0.005:
        return (frame_path, None)

    # Add safety margin (~50px full-res) to avoid clipping real content
    margin = 50
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(full_w, x_max + margin)
    y_max = min(full_h, y_max + margin)

    crop_w = x_max - x_min
    crop_h = y_max - y_min

    # Crop via gdal_translate
    base = os.path.splitext(os.path.basename(frame_path))[0]
    cropped_path = os.path.join(output_dir, f"{base}_cropped.tif")

    _run_cmd([
        "gdal_translate",
        "-srcwin", str(x_min), str(y_min), str(crop_w), str(crop_h),
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-co", "TILED=YES",
        frame_path,
        cropped_path,
    ])

    borders = []
    if left_pct >= 0.005:
        borders.append(f"left={100*left_pct:.1f}%")
    if right_pct >= 0.005:
        borders.append(f"right={100*right_pct:.1f}%")
    if top_pct >= 0.005:
        borders.append(f"top={100*top_pct:.1f}%")
    if bottom_pct >= 0.005:
        borders.append(f"bottom={100*bottom_pct:.1f}%")
    print(f"    Cropped borders from {os.path.basename(frame_path)}: {', '.join(borders)}")

    return (cropped_path, (x_min, y_min, crop_w, crop_h))


def _crop_all_borders(frames: list, output_dir: str) -> tuple:
    """Crop black borders from all frames.

    Returns (cropped_frames, temp_files) where cropped_frames is a list of
    paths (cropped or original) and temp_files is a list of temp files to
    clean up later.
    """
    cropped_frames = []
    temp_files = []

    for frame in frames:
        cropped_path, crop_box = crop_black_borders(frame, output_dir)
        cropped_frames.append(cropped_path)
        if crop_box is not None:
            temp_files.append(cropped_path)

    return cropped_frames, temp_files


def discover_frame_order(frames, scale=0.10):
    """Discover the correct left-to-right ordering of frames.

    Uses all-pairs SIFT matching to build a directed connectivity graph,
    then chains from the leftmost frame to produce a linear strip ordering.
    Handles arbitrarily-ordered inputs and detects 180-degree rotations.

    Args:
        frames: List of frame file paths in arbitrary order.
        scale: Downscale factor for overlap matching (default 0.10).

    Returns:
        (ordered_frames, overlaps, rotated_flags) where:
        - ordered_frames: frames reordered into correct strip sequence
        - overlaps: list of overlap pixels between consecutive ordered frames
        - rotated_flags: list of bools per frame (True = needs 180 rotation)
    """
    n = len(frames)
    if n <= 1:
        return list(frames), [], [False] * n

    if n == 2:
        ov_ab, rot_ab, n_ab = compute_overlap(frames[0], frames[1], scale)
        ov_ba, rot_ba, n_ba = compute_overlap(frames[1], frames[0], scale)

        # Prefer direction with significantly more matches (20% margin)
        if n_ba > n_ab * 1.2 and n_ba >= 10:
            swap = True
        elif n_ab >= n_ba * 1.2 and n_ab >= 10:
            swap = False
        else:
            # Similar match counts — fall back to overlap
            swap = ov_ba > ov_ab and ov_ba > 0

        if swap:
            print(f"    Swapping frame order (B→A: {n_ba} matches/{ov_ba}px > A→B: {n_ab} matches/{ov_ab}px)")
            return [frames[1], frames[0]], [ov_ba], [False, rot_ba]
        elif ov_ab > 0 or ov_ba > 0:
            return list(frames), [ov_ab if ov_ab > 0 else 0], [False, rot_ab if ov_ab > 0 else False]
        else:
            return list(frames), [0], [False, False]

    print(f"  Discovering frame order ({n} frames, {n*(n-1)} directed pairs)...")

    # Compute all directed pair matches in parallel.
    # compute_overlap(A, B) tests "is A to the LEFT of B?"
    with ProcessPoolExecutor() as executor:
        futures = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                fut = executor.submit(compute_overlap, frames[i], frames[j], scale)
                futures[fut] = (i, j)

        matches = {}
        for fut in as_completed(futures):
            i, j = futures[fut]
            try:
                overlap, is_rotated, _ = fut.result()
                if overlap > 0:
                    matches[(i, j)] = (overlap, is_rotated)
            except Exception as e:
                print(f"    Match ({i},{j}) failed: {e}")

    if not matches:
        print("  WARNING: No pairwise matches found, using input order")
        sequential_overlaps = []
        sequential_rotated = [False]
        for i in range(n - 1):
            ov, rot, _ = compute_overlap(frames[i], frames[i + 1], scale)
            sequential_overlaps.append(ov)
            sequential_rotated.append(rot)
        return list(frames), sequential_overlaps, sequential_rotated

    # Build best-right-neighbor map.
    # right_of[i] = (j, overlap, is_j_rotated)  means j is immediately right of i.
    right_of = {}
    for (i, j), (overlap, is_rotated) in matches.items():
        if i not in right_of or overlap > right_of[i][1]:
            right_of[i] = (j, overlap, is_rotated)

    # Resolve conflicts: if two frames claim the same right-neighbor,
    # keep the stronger match and remove the weaker.
    claimed_right = defaultdict(list)
    for i, (j, ov, rot) in right_of.items():
        claimed_right[j].append((i, ov, rot))
    for j, claimants in claimed_right.items():
        if len(claimants) > 1:
            claimants.sort(key=lambda x: -x[1])
            for loser_i, _, _ in claimants[1:]:
                del right_of[loser_i]

    # Leftmost frame = never appears as anyone's right neighbor.
    appears_as_right = {j for _, (j, _, _) in right_of.items()}
    leftmost = [i for i in range(n) if i not in appears_as_right]
    if not leftmost:
        leftmost = [max(right_of, key=lambda k: right_of[k][1])]

    # Chain from leftmost to build the strip ordering.
    start = leftmost[0]
    ordered = [start]
    overlaps = []
    rot_flags = [False]
    visited = {start}
    current = start

    while current in right_of and len(ordered) < n:
        j, ov, is_rot = right_of[current]
        if j in visited:
            break
        ordered.append(j)
        overlaps.append(ov)
        rot_flags.append(is_rot)
        visited.add(j)
        current = j

    # Append any disconnected frames with zero overlap.
    for i in range(n):
        if i not in visited:
            ordered.append(i)
            overlaps.append(0)
            rot_flags.append(False)
            print(f"    WARNING: Frame {os.path.basename(frames[i])} disconnected from strip")

    result_frames = [frames[i] for i in ordered]
    print(f"    Order: {' -> '.join(os.path.basename(f) for f in result_frames)}")
    for idx, ov in enumerate(overlaps):
        rot_label = " (B rotated 180)" if rot_flags[idx + 1] else ""
        print(f"    Overlap {idx}: {ov}px{rot_label}")

    return result_frames, overlaps, rot_flags


def stitch_frames(frames: list, output_path: str, output_dir: str,
                  preserve_order: bool = False) -> str:
    """Stitch a list of frame paths into a single panoramic strip.

    By default, frame ordering is discovered automatically via all-pairs
    SIFT matching (graph-based), so frames do NOT need to be pre-sorted.

    When preserve_order=True, frames are stitched in the exact order given.
    Use this for sub-frames split from a single image (e.g. via split_at_seams),
    where the input order is known correct and should not be rearranged.

    Args:
        frames: List of frame file paths.
        output_path: Where to write the stitched output.
        output_dir: Working directory for temp files.
        preserve_order: If True, skip graph-based reordering and use input order.

    Returns:
        Path to the stitched output.
    """
    if os.path.exists(output_path):
        print(f"  [skip] Stitched output already exists: {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if len(frames) == 0:
        raise RuntimeError("No frames to stitch")

    if len(frames) == 1:
        shutil.copy2(frames[0], output_path)
        print(f"  Single frame, copied to {output_path}")
        return output_path

    # Pre-process: crop black borders for better overlap detection
    cropped_frames, crop_temps = _crop_all_borders(frames, output_dir)

    if preserve_order:
        # Sequential overlap matching — frames are already in correct order.
        # Only compute overlap between consecutive pairs.
        print(f"  Sequential stitching ({len(cropped_frames)} frames, order preserved)...")
        overlaps = []
        rotated_flags = [False] * len(cropped_frames)
        ordered_frames = list(cropped_frames)

        if len(cropped_frames) == 2:
            # For 2-frame case, test both directions to detect order reversal
            # (e.g. KH-7 where alphabetical _a/_b doesn't match geographic order)
            ov_ab, rot_ab, n_ab = compute_overlap(ordered_frames[0], ordered_frames[1])
            ov_ba, rot_ba, n_ba = compute_overlap(ordered_frames[1], ordered_frames[0])

            # Prefer direction with significantly more matches (20% margin)
            if n_ba > n_ab * 1.2 and n_ba >= 10:
                swap = True
            elif n_ab >= n_ba * 1.2 and n_ab >= 10:
                swap = False
            else:
                # Similar match counts — fall back to overlap
                swap = ov_ba > ov_ab and ov_ba > 0

            if swap:
                print(f"    Swapping frame order (B→A: {n_ba} matches/{ov_ba}px > A→B: {n_ab} matches/{ov_ab}px)")
                ordered_frames[0], ordered_frames[1] = ordered_frames[1], ordered_frames[0]
                overlaps.append(ov_ba)
                rotated_flags[1] = rot_ba
            else:
                overlaps.append(ov_ab if ov_ab > 0 else 0)
                rotated_flags[1] = rot_ab if ov_ab > 0 else False
        else:
            # For >2 frames, keep parallel sequential matching
            with ProcessPoolExecutor() as executor:
                futures = {}
                for i in range(len(ordered_frames) - 1):
                    fut = executor.submit(compute_overlap, ordered_frames[i], ordered_frames[i + 1])
                    futures[fut] = i
                results = [None] * (len(ordered_frames) - 1)
                for fut in as_completed(futures):
                    idx = futures[fut]
                    results[idx] = fut.result()
            for i, r in enumerate(results):
                overlaps.append(r[0] if r else 0)
                if r and r[1]:
                    print(f"    Frame {i+1} needs 180° rotation")
                    rotated_flags[i + 1] = True
    else:
        # Graph-based ordering — discover correct layout from all-pairs matching.
        ordered_frames, overlaps, rotated_flags = discover_frame_order(cropped_frames)

    # Get dimensions of each ordered frame
    frame_info = []
    for f in ordered_frames:
        result = _run_cmd(["gdalinfo", "-json", f])
        info = json.loads(result.stdout)
        w, h = info["size"]
        bands = len(info.get("bands", [{"type": "Byte"}]))
        dtype = info.get("bands", [{"type": "Byte"}])[0].get("type", "Byte")
        frame_info.append({"path": f, "w": w, "h": h, "bands": bands, "dtype": dtype})
        print(f"    Frame {os.path.basename(f)}: {w}x{h}")

    # Handle rotated frames: create 180-flipped copies
    rot180_temps = []
    for i, is_rotated in enumerate(rotated_flags):
        if is_rotated:
            orig_path = frame_info[i]["path"]
            flipped_path = orig_path.replace(".tif", "_rot180.tif")
            print(f"  Frame {os.path.basename(orig_path)} is rotated 180 — creating corrected copy")
            flip_frame_180(orig_path, flipped_path)
            frame_info[i]["path"] = flipped_path
            rot180_temps.append(flipped_path)

    # Compute x-offsets accounting for overlap
    x_offsets = [0]
    for i, fi in enumerate(frame_info[:-1]):
        next_x = x_offsets[-1] + fi["w"] - overlaps[i]
        x_offsets.append(next_x)

    total_w = x_offsets[-1] + frame_info[-1]["w"]
    max_h = max(fi["h"] for fi in frame_info)
    n_bands = frame_info[0]["bands"]
    dtype = frame_info[0]["dtype"]

    # Build VRT
    vrt_path = output_path.replace(".tif", ".vrt")
    vrt_lines = [f'<VRTDataset rasterXSize="{total_w}" rasterYSize="{max_h}">']
    for band_idx in range(1, n_bands + 1):
        vrt_lines.append(f'  <VRTRasterBand dataType="{dtype}" band="{band_idx}">')
        for fi, x_off in zip(frame_info, x_offsets):
            vrt_lines.append(f'    <SimpleSource>')
            vrt_lines.append(f'      <SourceFilename relativeToVRT="0">{fi["path"]}</SourceFilename>')
            vrt_lines.append(f'      <SourceBand>{band_idx}</SourceBand>')
            vrt_lines.append(f'      <SrcRect xOff="0" yOff="0" xSize="{fi["w"]}" ySize="{fi["h"]}"/>')
            vrt_lines.append(f'      <DstRect xOff="{x_off}" yOff="0" xSize="{fi["w"]}" ySize="{fi["h"]}"/>')
            vrt_lines.append(f'    </SimpleSource>')
        vrt_lines.append(f'  </VRTRasterBand>')
    vrt_lines.append('</VRTDataset>')

    with open(vrt_path, "w") as f:
        f.write("\n".join(vrt_lines))

    print(f"  VRT canvas: {total_w}x{max_h} ({len(frames)} frames, overlaps: {overlaps})")

    # Render VRT to GeoTIFF
    _run_cmd([
        "gdal_translate",
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=IF_SAFER",
        vrt_path,
        output_path,
    ])

    os.remove(vrt_path)

    # Feathered blending in overlap zones
    if any(o > 0 for o in overlaps):
        print(f"  Applying feathered blending to {sum(1 for o in overlaps if o > 0)} overlap zones...")
        _blend_overlaps(output_path, frame_info, x_offsets, overlaps)

    # Clean up temp files (rotated copies and cropped borders)
    for temp_path in rot180_temps + crop_temps:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print(f"  Stitched {len(frames)} frames -> {output_path}")
    return output_path


def compute_subset_corners(corners: dict, selected_indices: list,
                           n_total_frames: int, reversed_order: bool = False) -> dict:
    """Interpolate corner coordinates for a subset of frames."""
    n = n_total_frames

    if reversed_order:
        left_frac = (n - 1 - selected_indices[-1]) / n
        right_frac = (n - selected_indices[0]) / n
    else:
        left_frac = selected_indices[0] / n
        right_frac = (selected_indices[-1] + 1) / n

    def interp(corner_left, corner_right, frac):
        lat = corner_left[0] + (corner_right[0] - corner_left[0]) * frac
        lon = corner_left[1] + (corner_right[1] - corner_left[1]) * frac
        return (lat, lon)

    geo_nw = interp(corners["NW"], corners["NE"], left_frac)
    geo_ne = interp(corners["NW"], corners["NE"], right_frac)
    geo_sw = interp(corners["SW"], corners["SE"], left_frac)
    geo_se = interp(corners["SW"], corners["SE"], right_frac)

    if reversed_order:
        return {
            "NW": geo_se, "NE": geo_sw,
            "SE": geo_nw, "SW": geo_ne,
        }

    return {
        "NW": geo_nw, "NE": geo_ne,
        "SW": geo_sw, "SE": geo_se,
    }


def detect_subframe_seams(image_path: str, scale: float = 0.02) -> list:
    """Detect internal sub-frame seam positions in a single USGS TIF.

    USGS pre-stitched TIFs (KH-4, KH-7) may contain multiple sub-segments
    with horizontal offsets between them. This function finds the seam columns.

    Samples horizontal scan lines at multiple heights and looks for columns
    where a brightness discontinuity (or black gap) appears consistently.

    Args:
        image_path: Path to the input TIF.
        scale: Downscale factor for reading (default 0.02 = 2%).

    Returns:
        Sorted list of seam x-positions in full-resolution pixel coordinates.
        Empty list if no seams detected.
    """
    from osgeo import gdal
    import numpy as np

    gdal.UseExceptions()
    ds = gdal.Open(image_path)
    if ds is None:
        return []

    w, h = ds.RasterXSize, ds.RasterYSize
    band = ds.GetRasterBand(1)

    # Determine the scanning direction — detect seams along the long axis
    # For landscape images, seams are vertical (scan horizontally)
    # For portrait images, seams are horizontal (scan vertically)
    is_portrait = h > w

    def _scan_seams_along_axis(length, cross_length, scan_positions):
        """Scan for seams along one axis. Returns seam positions in full resolution."""
        out_len = max(100, int(length * scale))
        gradient_votes = np.zeros(out_len, dtype=np.int32)

        for pos in scan_positions:
            strip_thick = max(1, int(cross_length * 0.02))
            offset = max(0, pos - strip_thick // 2)
            strip_thick = min(strip_thick, cross_length - offset)

            if is_portrait:
                strip = band.ReadAsArray(
                    xoff=offset, yoff=0, win_xsize=strip_thick, win_ysize=h,
                    buf_xsize=1, buf_ysize=out_len)
            else:
                strip = band.ReadAsArray(
                    xoff=0, yoff=offset, win_xsize=w, win_ysize=strip_thick,
                    buf_xsize=out_len, buf_ysize=1)
            if strip is None:
                continue

            line = strip.astype(np.float64).flatten()
            grad = np.abs(np.diff(line))
            is_zero = (line < 5).astype(np.float64)
            threshold = max(np.percentile(grad, 95), 30)
            combined = (grad > threshold) | (np.abs(np.diff(is_zero)) > 0.5)
            gradient_votes[:len(combined)] += combined.astype(np.int32)

        # Find positions where >=3 of 5 scan lines agree
        margin = max(5, out_len // 20)
        candidates = np.where(gradient_votes[margin:-margin] >= 3)[0] + margin
        if len(candidates) == 0:
            return []

        # Cluster nearby candidates (within 2% of image length)
        cluster_dist = max(2, int(out_len * 0.02))
        seam_positions_scaled = []
        cluster = [candidates[0]]
        for i in range(1, len(candidates)):
            if candidates[i] - candidates[i - 1] <= cluster_dist:
                cluster.append(candidates[i])
            else:
                seam_positions_scaled.append(int(np.median(cluster)))
                cluster = [candidates[i]]
        seam_positions_scaled.append(int(np.median(cluster)))

        return [int(pos * length / out_len) for pos in seam_positions_scaled]

    if is_portrait:
        scan_positions = [int(w * frac) for frac in [0.10, 0.30, 0.50, 0.70, 0.90]]
        seam_positions = _scan_seams_along_axis(h, w, scan_positions)
    else:
        scan_positions = [int(h * frac) for frac in [0.10, 0.30, 0.50, 0.70, 0.90]]
        seam_positions = _scan_seams_along_axis(w, h, scan_positions)

    ds = None

    if seam_positions:
        dim = "height" if is_portrait else "width"
        full_dim = h if is_portrait else w
        pcts = [f"{100 * s / full_dim:.1f}%" for s in seam_positions]
        print(f"  Detected {len(seam_positions)} sub-frame seam(s) at {', '.join(pcts)} of {dim}")

    return seam_positions


def split_at_seams(image_path: str, seam_positions: list, output_dir: str,
                   is_portrait: bool = False) -> list:
    """Split an image at detected seam positions into sub-frame TIFs.

    Args:
        image_path: Path to the input TIF.
        seam_positions: List of seam positions (in pixels along the split axis).
        output_dir: Directory to write sub-frame TIFs.
        is_portrait: If True, split horizontally (seams are at y positions).
                     If False, split vertically (seams are at x positions).

    Returns:
        Sorted list of sub-frame file paths.
    """
    from osgeo import gdal
    import json as _json

    gdal.UseExceptions()

    result = subprocess.run(
        ["gdalinfo", "-json", image_path],
        capture_output=True, text=True,
    )
    info = _json.loads(result.stdout)
    width, height = info["size"]

    # Add overlap padding around seams (5% of sub-frame size)
    sorted_seams = sorted(seam_positions)

    if is_portrait:
        boundaries = [0] + sorted_seams + [height]
    else:
        boundaries = [0] + sorted_seams + [width]

    # Compute sub-frame regions with small overlap for stitching
    sub_frames = []
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        # Add padding
        full_dim = height if is_portrait else width
        pad = max(50, int((end - start) * 0.05))
        padded_start = max(0, start - (pad if i > 0 else 0))
        padded_end = min(full_dim, end + (pad if i < len(boundaries) - 2 else 0))

        sub_name = f"{base_name}_sub{i:02d}.tif"
        sub_path = os.path.join(output_dir, sub_name)

        if is_portrait:
            srcwin = f"0 {padded_start} {width} {padded_end - padded_start}"
        else:
            srcwin = f"{padded_start} 0 {padded_end - padded_start} {height}"

        cmd = [
            "gdal_translate",
            "-srcwin", *srcwin.split(),
            "-co", "COMPRESS=LZW",
            "-co", "PREDICTOR=2",
            "-co", "TILED=YES",
            image_path,
            sub_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        sub_frames.append(sub_path)

    print(f"  Split into {len(sub_frames)} sub-frames")
    return sorted(sub_frames)
