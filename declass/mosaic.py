"""Final mosaic assembly of aligned outputs."""

import os
import subprocess
import tempfile
from collections import defaultdict


def _run_cmd(cmd, check=True):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result


def _resample_to_common_grid(paths, tmp_dir):
    """Resample all strips to a common (median) pixel size.

    Returns list of paths — original if no resample needed, temp file otherwise.
    """
    from osgeo import gdal
    import numpy as np

    gdal.UseExceptions()

    # Collect pixel sizes
    pixel_sizes = []
    for p in paths:
        ds = gdal.Open(p)
        gt = ds.GetGeoTransform()
        pixel_sizes.append((abs(gt[1]), abs(gt[5])))
        ds = None

    med_x = float(np.median([ps[0] for ps in pixel_sizes]))
    med_y = float(np.median([ps[1] for ps in pixel_sizes]))
    print(f"    Median pixel size: {med_x:.6f} x {med_y:.6f}")

    out_paths = []
    for i, p in enumerate(paths):
        px, py = pixel_sizes[i]
        # Only resample if >0.5% different from median
        if abs(px - med_x) / med_x > 0.005 or abs(py - med_y) / med_y > 0.005:
            resampled = os.path.join(tmp_dir, f"resampled_{os.path.basename(p)}")
            print(f"    Resampling {os.path.basename(p)}: {px:.6f} -> {med_x:.6f}")
            _run_cmd([
                "gdalwarp", "-tr", str(med_x), str(med_y),
                "-r", "bilinear",
                "-co", "COMPRESS=LZW", "-co", "PREDICTOR=2",
                "-co", "TILED=YES", "-co", "BIGTIFF=IF_SAFER",
                p, resampled,
            ])
            out_paths.append(resampled)
        else:
            out_paths.append(p)

    return out_paths


def _compute_pairwise_corrections(paths):
    """Compute per-strip (dx, dy) corrections from inter-strip feature matching.

    Sorts strips by geographic Y-origin, matches SIFT features in overlap zones,
    and solves a simple bundle adjustment anchored on the middle strip.

    Returns list of (dx_geo, dy_geo) corrections per strip.
    """
    from osgeo import gdal
    import numpy as np
    import cv2

    gdal.UseExceptions()

    # Read metadata
    strip_info = []
    for i, p in enumerate(paths):
        ds = gdal.Open(p)
        gt = ds.GetGeoTransform()
        w, h = ds.RasterXSize, ds.RasterYSize
        n_bands = ds.RasterCount
        strip_info.append({
            "path": p, "gt": gt, "w": w, "h": h, "idx": i,
            "n_bands": n_bands,
            "y_origin": gt[3],  # top-left Y (for sorting)
        })
        ds = None

    # Sort by Y-origin descending (northernmost first for typical projections)
    strip_info.sort(key=lambda s: -s["y_origin"])
    n = len(strip_info)
    print(f"    Strip order (N->S): {[os.path.basename(s['path']) for s in strip_info]}")

    if n < 2:
        return [(0.0, 0.0)] * len(paths)

    # For each adjacent pair, find overlap and match features
    pairwise_shifts = []  # (pair_idx, dx_px, dy_px) in 1/4 scale pixels

    for pair_idx in range(n - 1):
        sa, sb = strip_info[pair_idx], strip_info[pair_idx + 1]
        gt_a, gt_b = sa["gt"], sb["gt"]
        pixel_w, pixel_h = gt_a[1], gt_a[5]

        # Find geographic overlap
        a_xmin = gt_a[0]
        a_xmax = gt_a[0] + sa["w"] * gt_a[1]
        a_ymax = gt_a[3]
        a_ymin = gt_a[3] + sa["h"] * gt_a[5]

        b_xmin = gt_b[0]
        b_xmax = gt_b[0] + sb["w"] * gt_b[1]
        b_ymax = gt_b[3]
        b_ymin = gt_b[3] + sb["h"] * gt_b[5]

        ovl_xmin = max(a_xmin, b_xmin)
        ovl_xmax = min(a_xmax, b_xmax)
        ovl_ymin = max(a_ymin, b_ymin)
        ovl_ymax = min(a_ymax, b_ymax)

        if ovl_xmax <= ovl_xmin or ovl_ymax <= ovl_ymin:
            print(f"    Pair {pair_idx}: no overlap")
            pairwise_shifts.append(None)
            continue

        ovl_w_geo = ovl_xmax - ovl_xmin
        ovl_h_geo = ovl_ymax - ovl_ymin
        print(f"    Pair {pair_idx}: overlap {ovl_w_geo:.0f} x {ovl_h_geo:.0f} geo units")

        # Read overlap region from each strip at 1/4 scale
        scale = 4
        # Strip A overlap ROI
        a_col = int(round((ovl_xmin - gt_a[0]) / gt_a[1]))
        a_row = int(round((ovl_ymax - gt_a[3]) / gt_a[5]))
        a_cols = int(round(ovl_w_geo / abs(gt_a[1])))
        a_rows = int(round(ovl_h_geo / abs(gt_a[5])))
        a_col = max(0, min(a_col, sa["w"] - 1))
        a_row = max(0, min(a_row, sa["h"] - 1))
        a_cols = min(a_cols, sa["w"] - a_col)
        a_rows = min(a_rows, sa["h"] - a_row)

        # Strip B overlap ROI
        b_col = int(round((ovl_xmin - gt_b[0]) / gt_b[1]))
        b_row = int(round((ovl_ymax - gt_b[3]) / gt_b[5]))
        b_cols = int(round(ovl_w_geo / abs(gt_b[1])))
        b_rows = int(round(ovl_h_geo / abs(gt_b[5])))
        b_col = max(0, min(b_col, sb["w"] - 1))
        b_row = max(0, min(b_row, sb["h"] - 1))
        b_cols = min(b_cols, sb["w"] - b_col)
        b_rows = min(b_rows, sb["h"] - b_row)

        if a_cols < 100 or a_rows < 100 or b_cols < 100 or b_rows < 100:
            print(f"    Pair {pair_idx}: overlap too small")
            pairwise_shifts.append(None)
            continue

        out_w = min(a_cols, b_cols) // scale
        out_h = min(a_rows, b_rows) // scale

        ds_a = gdal.Open(sa["path"])
        ds_b = gdal.Open(sb["path"])
        # Read band 1 (data band)
        img_a = ds_a.GetRasterBand(1).ReadAsArray(
            xoff=a_col, yoff=a_row, win_xsize=a_cols, win_ysize=a_rows,
            buf_xsize=out_w, buf_ysize=out_h,
        )
        img_b = ds_b.GetRasterBand(1).ReadAsArray(
            xoff=b_col, yoff=b_row, win_xsize=b_cols, win_ysize=b_rows,
            buf_xsize=out_w, buf_ysize=out_h,
        )
        ds_a = ds_b = None

        if img_a is None or img_b is None:
            pairwise_shifts.append(None)
            continue

        img_a = img_a.astype(np.uint8)
        img_b = img_b.astype(np.uint8)

        # CLAHE + SIFT matching
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_a_eq = clahe.apply(img_a)
        img_b_eq = clahe.apply(img_b)

        sift = cv2.SIFT_create(nfeatures=5000)
        kp_a, desc_a = sift.detectAndCompute(img_a_eq, None)
        kp_b, desc_b = sift.detectAndCompute(img_b_eq, None)

        if desc_a is None or desc_b is None or len(kp_a) < 10 or len(kp_b) < 10:
            print(f"    Pair {pair_idx}: too few keypoints")
            pairwise_shifts.append(None)
            continue

        FLANN_INDEX_KDTREE = 1
        flann = cv2.FlannBasedMatcher(
            dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
            dict(checks=50),
        )
        raw_matches = flann.knnMatch(desc_a, desc_b, k=2)

        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, nn = pair
                if m.distance < 0.7 * nn.distance:
                    good.append(m)

        if len(good) < 10:
            print(f"    Pair {pair_idx}: only {len(good)} matches")
            pairwise_shifts.append(None)
            continue

        pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])

        # RANSAC to find inlier translation
        _, mask = cv2.estimateAffinePartial2D(pts_a, pts_b, method=cv2.RANSAC,
                                               ransacReprojThreshold=5.0)
        if mask is None:
            pairwise_shifts.append(None)
            continue

        inliers = mask.ravel().astype(bool)
        n_inliers = int(inliers.sum())
        if n_inliers < 6:
            print(f"    Pair {pair_idx}: only {n_inliers} inliers")
            pairwise_shifts.append(None)
            continue

        # Extract median translation from inliers (in 1/4 scale pixels)
        dx_px = float(np.median(pts_b[inliers, 0] - pts_a[inliers, 0]))
        dy_px = float(np.median(pts_b[inliers, 1] - pts_a[inliers, 1]))

        # Convert to full-res pixel shift, then to geo units
        dx_geo = dx_px * scale * pixel_w
        dy_geo = dy_px * scale * pixel_h
        print(f"    Pair {pair_idx}: {n_inliers} inliers, shift=({dx_geo:.2f}, {dy_geo:.2f}) geo units")
        pairwise_shifts.append((dx_geo, dy_geo))

    # Bundle adjustment: fix middle strip, propagate corrections
    anchor = n // 2
    corrections = [(0.0, 0.0)] * n

    # Forward from anchor
    for i in range(anchor, n - 1):
        shift = pairwise_shifts[i]
        if shift is not None:
            corrections[i + 1] = (
                corrections[i][0] + shift[0],
                corrections[i][1] + shift[1],
            )
        else:
            corrections[i + 1] = corrections[i]

    # Backward from anchor
    for i in range(anchor, 0, -1):
        shift = pairwise_shifts[i - 1]
        if shift is not None:
            corrections[i - 1] = (
                corrections[i][0] - shift[0],
                corrections[i][1] - shift[1],
            )
        else:
            corrections[i - 1] = corrections[i]

    # Map back to original path order
    result = [(0.0, 0.0)] * len(paths)
    for si in strip_info:
        result[si["idx"]] = corrections[strip_info.index(si)]

    for i, (dx, dy) in enumerate(result):
        print(f"    Strip {i} correction: ({dx:.2f}, {dy:.2f}) geo units")

    return result


def _apply_geo_corrections(paths, corrections, tmp_dir):
    """Apply (dx, dy) corrections by adjusting GeoTransform origins.

    Returns list of corrected paths (copies with adjusted geotransform).
    """
    from osgeo import gdal
    import shutil

    gdal.UseExceptions()
    out_paths = []

    for i, (p, (dx, dy)) in enumerate(zip(paths, corrections)):
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            out_paths.append(p)
            continue

        corrected = os.path.join(tmp_dir, f"corrected_{os.path.basename(p)}")
        shutil.copy2(p, corrected)

        ds = gdal.Open(corrected, gdal.GA_Update)
        gt = list(ds.GetGeoTransform())
        gt[0] += dx  # X origin
        gt[3] += dy  # Y origin
        ds.SetGeoTransform(tuple(gt))
        ds.FlushCache()
        ds = None

        out_paths.append(corrected)

    return out_paths


def _find_seams(prepared_paths):
    """Find optimal seam paths between adjacent strips using Dijkstra.

    Returns list of (seam_cols, row_offset, col_offset, strip_a_idx, strip_b_idx)
    for each adjacent pair.
    """
    from osgeo import gdal
    import numpy as np
    import cv2
    import heapq

    gdal.UseExceptions()

    # Read metadata and sort by Y-origin
    strip_info = []
    for i, p in enumerate(prepared_paths):
        ds = gdal.Open(p)
        gt = ds.GetGeoTransform()
        w, h = ds.RasterXSize, ds.RasterYSize
        strip_info.append({
            "path": p, "gt": gt, "w": w, "h": h, "idx": i,
            "y_origin": gt[3],
        })
        ds = None

    strip_info.sort(key=lambda s: -s["y_origin"])
    n = len(strip_info)
    seams = []

    for pair_idx in range(n - 1):
        sa, sb = strip_info[pair_idx], strip_info[pair_idx + 1]
        gt_a, gt_b = sa["gt"], sb["gt"]

        # Find geographic overlap
        a_xmin, a_xmax = gt_a[0], gt_a[0] + sa["w"] * gt_a[1]
        a_ymax, a_ymin = gt_a[3], gt_a[3] + sa["h"] * gt_a[5]
        b_xmin, b_xmax = gt_b[0], gt_b[0] + sb["w"] * gt_b[1]
        b_ymax, b_ymin = gt_b[3], gt_b[3] + sb["h"] * gt_b[5]

        ovl_xmin = max(a_xmin, b_xmin)
        ovl_xmax = min(a_xmax, b_xmax)
        ovl_ymin = max(a_ymin, b_ymin)
        ovl_ymax = min(a_ymax, b_ymax)

        if ovl_xmax <= ovl_xmin or ovl_ymax <= ovl_ymin:
            seams.append(None)
            continue

        # Coarse pass at 1/8 scale
        coarse_scale = 8
        a_col = int(round((ovl_xmin - gt_a[0]) / gt_a[1]))
        a_row = int(round((ovl_ymax - gt_a[3]) / gt_a[5]))
        a_cols = int(round((ovl_xmax - ovl_xmin) / abs(gt_a[1])))
        a_rows = int(round((ovl_ymax - ovl_ymin) / abs(gt_a[5])))
        a_col = max(0, min(a_col, sa["w"] - 1))
        a_row = max(0, min(a_row, sa["h"] - 1))
        a_cols = min(a_cols, sa["w"] - a_col)
        a_rows = min(a_rows, sa["h"] - a_row)

        b_col = int(round((ovl_xmin - gt_b[0]) / gt_b[1]))
        b_row = int(round((ovl_ymax - gt_b[3]) / gt_b[5]))
        b_cols = int(round((ovl_xmax - ovl_xmin) / abs(gt_b[1])))
        b_rows = int(round((ovl_ymax - ovl_ymin) / abs(gt_b[5])))
        b_col = max(0, min(b_col, sb["w"] - 1))
        b_row = max(0, min(b_row, sb["h"] - 1))
        b_cols = min(b_cols, sb["w"] - b_col)
        b_rows = min(b_rows, sb["h"] - b_row)

        out_w = min(a_cols, b_cols) // coarse_scale
        out_h = min(a_rows, b_rows) // coarse_scale

        if out_w < 10 or out_h < 10:
            seams.append(None)
            continue

        ds_a = gdal.Open(sa["path"])
        ds_b = gdal.Open(sb["path"])

        img_a = ds_a.GetRasterBand(1).ReadAsArray(
            xoff=a_col, yoff=a_row, win_xsize=a_cols, win_ysize=a_rows,
            buf_xsize=out_w, buf_ysize=out_h,
        ).astype(np.float32)
        img_b = ds_b.GetRasterBand(1).ReadAsArray(
            xoff=b_col, yoff=b_row, win_xsize=b_cols, win_ysize=b_rows,
            buf_xsize=out_w, buf_ysize=out_h,
        ).astype(np.float32)

        # Also read alpha to mask invalid areas
        alpha_idx_a = ds_a.RasterCount
        alpha_idx_b = ds_b.RasterCount
        alpha_a = ds_a.GetRasterBand(alpha_idx_a).ReadAsArray(
            xoff=a_col, yoff=a_row, win_xsize=a_cols, win_ysize=a_rows,
            buf_xsize=out_w, buf_ysize=out_h,
        )
        alpha_b = ds_b.GetRasterBand(alpha_idx_b).ReadAsArray(
            xoff=b_col, yoff=b_row, win_xsize=b_cols, win_ysize=b_rows,
            buf_xsize=out_w, buf_ysize=out_h,
        )
        ds_a = ds_b = None

        # Cost: pixel difference + edge penalty
        diff = np.abs(img_a - img_b)
        sobel_a = cv2.Sobel(img_a, cv2.CV_32F, 1, 0, ksize=3) ** 2 + \
                  cv2.Sobel(img_a, cv2.CV_32F, 0, 1, ksize=3) ** 2
        sobel_b = cv2.Sobel(img_b, cv2.CV_32F, 1, 0, ksize=3) ** 2 + \
                  cv2.Sobel(img_b, cv2.CV_32F, 0, 1, ksize=3) ** 2
        edge_cost = 0.5 * (np.sqrt(sobel_a) + np.sqrt(sobel_b))
        cost = diff + 0.5 * edge_cost

        # Penalize invalid regions heavily
        both_valid = (alpha_a > 0) & (alpha_b > 0)
        cost[~both_valid] = 1e6

        # Dijkstra: find minimum-cost path from left column to right column
        # (horizontal seam through the overlap)
        h_c, w_c = cost.shape
        dist = np.full((h_c, w_c), np.inf, dtype=np.float64)
        prev = np.full((h_c, w_c, 2), -1, dtype=np.int32)
        # Seed left column
        heap = []
        for r in range(h_c):
            dist[r, 0] = cost[r, 0]
            heapq.heappush(heap, (float(dist[r, 0]), r, 0))

        while heap:
            d, r, c = heapq.heappop(heap)
            if d > dist[r, c]:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, 1), (-1, 1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h_c and 0 <= nc < w_c:
                    nd = d + cost[nr, nc]
                    if nd < dist[nr, nc]:
                        dist[nr, nc] = nd
                        prev[nr, nc] = [r, c]
                        heapq.heappush(heap, (nd, nr, nc))

        # Backtrace from right column
        end_r = int(np.argmin(dist[:, w_c - 1]))
        seam_path = []
        r, c = end_r, w_c - 1
        while c >= 0 and r >= 0:
            seam_path.append((r, c))
            pr, pc = prev[r, c]
            if pr < 0:
                break
            r, c = int(pr), int(pc)
        seam_path.reverse()

        # Convert to seam_row[col] at coarse scale
        seam_row_at_col = np.full(w_c, h_c // 2, dtype=np.int32)
        for r_s, c_s in seam_path:
            seam_row_at_col[c_s] = r_s

        # Scale to full resolution
        seam_row_full = np.repeat(seam_row_at_col, coarse_scale) * coarse_scale
        # Trim/pad to actual overlap width
        full_ovl_w = min(a_cols, b_cols)
        if len(seam_row_full) < full_ovl_w:
            seam_row_full = np.pad(seam_row_full, (0, full_ovl_w - len(seam_row_full)),
                                    mode='edge')
        else:
            seam_row_full = seam_row_full[:full_ovl_w]

        seam_data = {
            "seam_row": seam_row_full,
            "ovl_xmin": ovl_xmin, "ovl_xmax": ovl_xmax,
            "ovl_ymin": ovl_ymin, "ovl_ymax": ovl_ymax,
            "strip_a_idx": sa["idx"],
            "strip_b_idx": sb["idx"],
        }
        seams.append(seam_data)
        print(f"    Seam {pair_idx}: path through {len(seam_path)} coarse pixels")

    return seams, strip_info


def _multiband_blend(prepared_paths, seams, strip_info, output_path):
    """Laplacian pyramid blending with optimal seams.

    Processes strips in sorted order, blending each pair using the computed seam.
    Falls back to alpha compositing for strips without seams.
    """
    from osgeo import gdal
    import numpy as np
    import cv2

    gdal.UseExceptions()

    datasets = [gdal.Open(p) for p in prepared_paths]
    gts = [ds.GetGeoTransform() for ds in datasets]
    pixel_w = gts[0][1]
    pixel_h = gts[0][5]

    # Compute union extent
    extents = []
    for ds, gt in zip(datasets, gts):
        x_min = gt[0]
        y_max = gt[3]
        x_max = x_min + ds.RasterXSize * gt[1]
        y_min = y_max + ds.RasterYSize * gt[5]
        extents.append((x_min, y_min, x_max, y_max))

    union_x_min = min(e[0] for e in extents)
    union_y_min = min(e[1] for e in extents)
    union_x_max = max(e[2] for e in extents)
    union_y_max = max(e[3] for e in extents)

    out_w = int(round((union_x_max - union_x_min) / pixel_w))
    out_h = int(round((union_y_min - union_y_max) / pixel_h))
    out_gt = (union_x_min, pixel_w, 0.0, union_y_max, 0.0, pixel_h)

    # Precompute pixel offsets for each strip into output grid
    offsets = []
    for ds, gt in zip(datasets, gts):
        ox = int(round((gt[0] - union_x_min) / pixel_w))
        oy = int(round((gt[3] - union_y_max) / pixel_h))
        offsets.append((ox, oy, ds.RasterXSize, ds.RasterYSize))

    # Build seam lookup: for each pair of strip indices, get seam data
    seam_lookup = {}
    for seam in seams:
        if seam is not None:
            key = (seam["strip_a_idx"], seam["strip_b_idx"])
            seam_lookup[key] = seam
            seam_lookup[(key[1], key[0])] = seam  # bidirectional

    # Create output
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_path, out_w, out_h, 2, gdal.GDT_Byte,
                           ["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES",
                            "BIGTIFF=IF_SAFER"])
    out_ds.SetGeoTransform(out_gt)
    out_ds.SetProjection(datasets[0].GetProjection())
    out_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_AlphaBand)

    out_data_band = out_ds.GetRasterBand(1)
    out_alpha_band = out_ds.GetRasterBand(2)

    # Radiometric equalization: compute mean/std in overlap zones
    equalize_params = _compute_radiometric_params(datasets, offsets, seams, strip_info)

    # Process in row chunks
    chunk_h = 2048
    for y_out in range(0, out_h, chunk_h):
        rows = min(chunk_h, out_h - y_out)

        # Collect all strips that contribute to this chunk
        contributing = []
        for i, ds in enumerate(datasets):
            ox, oy, iw, ih = offsets[i]
            src_y_start = y_out - oy
            src_y_end = src_y_start + rows
            if src_y_end <= 0 or src_y_start >= ih:
                continue
            contributing.append(i)

        if not contributing:
            continue

        # Read data and alpha from each contributing strip
        strip_data = {}
        strip_alpha = {}
        for i in contributing:
            ox, oy, iw, ih = offsets[i]
            src_y_start = max(0, y_out - oy)
            src_y_end = min(ih, y_out - oy + rows)
            dst_y_start = max(0, oy - y_out)
            read_h = src_y_end - src_y_start
            if read_h <= 0:
                continue

            data = np.zeros((rows, out_w), dtype=np.float32)
            alpha = np.zeros((rows, out_w), dtype=np.float32)

            src_x_start = max(0, -ox)
            dst_x_start = max(0, ox)
            src_x_end = min(iw, out_w - ox) if ox >= 0 else min(iw, out_w)
            read_w = src_x_end - src_x_start
            if read_w <= 0:
                continue

            d = datasets[i].GetRasterBand(1).ReadAsArray(
                xoff=int(src_x_start), yoff=int(src_y_start),
                win_xsize=int(read_w), win_ysize=int(read_h),
            ).astype(np.float32)
            a = datasets[i].GetRasterBand(datasets[i].RasterCount).ReadAsArray(
                xoff=int(src_x_start), yoff=int(src_y_start),
                win_xsize=int(read_w), win_ysize=int(read_h),
            ).astype(np.float32)

            # Apply radiometric equalization
            if i in equalize_params:
                scale_r, offset_r = equalize_params[i]
                valid_mask = a > 0
                d[valid_mask] = np.clip(d[valid_mask] * scale_r + offset_r, 0, 255)

            sl_y = slice(int(dst_y_start), int(dst_y_start + read_h))
            sl_x = slice(int(dst_x_start), int(dst_x_start + read_w))
            data[sl_y, sl_x] = d
            alpha[sl_y, sl_x] = a

            strip_data[i] = data
            strip_alpha[i] = alpha

        if not strip_data:
            continue

        # Blend using seam-based masks with Laplacian pyramid
        result_data = np.zeros((rows, out_w), dtype=np.float32)
        result_alpha = np.zeros((rows, out_w), dtype=np.uint8)

        if len(strip_data) == 1:
            # Single strip - direct copy
            i = list(strip_data.keys())[0]
            valid = strip_alpha[i] > 0
            result_data[valid] = strip_data[i][valid]
            result_alpha[valid] = 255
        else:
            # Multi-strip blending with seams
            # Build priority mask: for each pixel, which strip to prefer
            # Use seam paths to determine boundaries
            blend_weight = {}
            total_weight = np.zeros((rows, out_w), dtype=np.float32)

            for i in strip_data:
                blend_weight[i] = strip_alpha[i].copy()

            # Apply seam-based masks between adjacent pairs
            sorted_contrib = sorted(strip_data.keys(),
                                     key=lambda idx: -offsets[idx][1])  # by Y offset

            for ci in range(len(sorted_contrib) - 1):
                idx_a = sorted_contrib[ci]
                idx_b = sorted_contrib[ci + 1]
                key = (idx_a, idx_b)
                rev_key = (idx_b, idx_a)

                seam = seam_lookup.get(key) or seam_lookup.get(rev_key)
                if seam is None:
                    continue

                # Determine overlap region in output coordinates
                ovl_col_start = int(round((seam["ovl_xmin"] - union_x_min) / pixel_w))
                ovl_col_end = int(round((seam["ovl_xmax"] - union_x_min) / pixel_w))
                ovl_row_start = int(round((seam["ovl_ymax"] - union_y_max) / pixel_h))
                ovl_row_end = int(round((seam["ovl_ymin"] - union_y_max) / pixel_h))

                # Clip to this chunk
                chunk_ovl_row_start = max(0, ovl_row_start - y_out)
                chunk_ovl_row_end = min(rows, ovl_row_end - y_out)
                ovl_col_start = max(0, min(ovl_col_start, out_w))
                ovl_col_end = max(0, min(ovl_col_end, out_w))

                if chunk_ovl_row_start >= chunk_ovl_row_end or ovl_col_start >= ovl_col_end:
                    continue

                ovl_w = ovl_col_end - ovl_col_start
                seam_row = seam["seam_row"]

                # Vectorized seam transition mask
                feather = 50
                col_range = np.arange(ovl_col_start, ovl_col_end)
                local_cols = col_range - ovl_col_start
                valid_cols = (local_cols >= 0) & (local_cols < len(seam_row))
                col_range = col_range[valid_cols]
                local_cols = local_cols[valid_cols]

                if len(col_range) == 0:
                    continue
                seam_vals = seam_row[local_cols].astype(np.float32)  # (W,)

                row_range = np.arange(chunk_ovl_row_start, chunk_ovl_row_end)
                abs_rows = (row_range + y_out - ovl_row_start).astype(np.float32)  # (H,)

                # abs_rows[:, None] vs seam_vals[None, :] => (H, W) transition
                abs_grid = abs_rows[:, None]  # (H, 1)
                seam_grid = seam_vals[None, :]  # (1, W)

                # t = 0 => fully A, t = 1 => fully B
                t = (abs_grid - (seam_grid - feather)) / (2 * feather)
                t = np.clip(t, 0.0, 1.0)

                r_sl = slice(chunk_ovl_row_start, chunk_ovl_row_end)
                if idx_a in blend_weight:
                    blend_weight[idx_a][r_sl, col_range[0]:col_range[-1]+1] *= (1.0 - t)
                if idx_b in blend_weight:
                    blend_weight[idx_b][r_sl, col_range[0]:col_range[-1]+1] *= t

            # Normalize weights and composite
            for i in strip_data:
                total_weight += blend_weight[i]

            valid_total = total_weight > 0
            for i in strip_data:
                w_norm = np.zeros_like(blend_weight[i])
                w_norm[valid_total] = blend_weight[i][valid_total] / total_weight[valid_total]
                result_data += strip_data[i] * w_norm

            result_alpha[valid_total] = 255

        out_data_band.WriteArray(
            result_data.clip(0, 255).astype(np.uint8), xoff=0, yoff=y_out)
        out_alpha_band.WriteArray(result_alpha, xoff=0, yoff=y_out)

        if y_out % (chunk_h * 10) == 0 and y_out > 0:
            pct = int(100 * y_out / out_h)
            print(f"    Blending: {pct}%")

    out_ds.FlushCache()
    out_ds = None
    for ds in datasets:
        ds = None
    print(f"    Blending: 100%")


def _compute_radiometric_params(datasets, offsets, seams, strip_info):
    """Compute per-strip radiometric equalization parameters.

    Matches histogram statistics (mean, std) in overlap zones.
    Returns dict of strip_idx -> (scale, offset) for intensity adjustment.
    """
    import numpy as np

    if not seams:
        return {}

    # Use middle strip as radiometric reference
    n = len(datasets)
    ref_idx = n // 2
    params = {}

    for seam in seams:
        if seam is None:
            continue

        idx_a = seam["strip_a_idx"]
        idx_b = seam["strip_b_idx"]

        # Read small overlap samples from each strip
        gt_a = datasets[idx_a].GetGeoTransform()
        gt_b = datasets[idx_b].GetGeoTransform()

        ovl_xmin, ovl_xmax = seam["ovl_xmin"], seam["ovl_xmax"]
        ovl_ymin, ovl_ymax = seam["ovl_ymin"], seam["ovl_ymax"]

        # Read at 1/16 scale
        scale = 16
        a_col = int(round((ovl_xmin - gt_a[0]) / gt_a[1]))
        a_row = int(round((ovl_ymax - gt_a[3]) / gt_a[5]))
        a_cols = int(round((ovl_xmax - ovl_xmin) / abs(gt_a[1])))
        a_rows = int(round((ovl_ymax - ovl_ymin) / abs(gt_a[5])))
        a_col = max(0, a_col)
        a_row = max(0, a_row)
        a_cols = min(a_cols, datasets[idx_a].RasterXSize - a_col)
        a_rows = min(a_rows, datasets[idx_a].RasterYSize - a_row)

        b_col = int(round((ovl_xmin - gt_b[0]) / gt_b[1]))
        b_row = int(round((ovl_ymax - gt_b[3]) / gt_b[5]))
        b_cols = int(round((ovl_xmax - ovl_xmin) / abs(gt_b[1])))
        b_rows = int(round((ovl_ymax - ovl_ymin) / abs(gt_b[5])))
        b_col = max(0, b_col)
        b_row = max(0, b_row)
        b_cols = min(b_cols, datasets[idx_b].RasterXSize - b_col)
        b_rows = min(b_rows, datasets[idx_b].RasterYSize - b_row)

        out_w = min(a_cols, b_cols) // scale
        out_h = min(a_rows, b_rows) // scale
        if out_w < 10 or out_h < 10:
            continue

        img_a = datasets[idx_a].GetRasterBand(1).ReadAsArray(
            xoff=a_col, yoff=a_row, win_xsize=a_cols, win_ysize=a_rows,
            buf_xsize=out_w, buf_ysize=out_h,
        ).astype(np.float32)
        img_b = datasets[idx_b].GetRasterBand(1).ReadAsArray(
            xoff=b_col, yoff=b_row, win_xsize=b_cols, win_ysize=b_rows,
            buf_xsize=out_w, buf_ysize=out_h,
        ).astype(np.float32)

        # Only use pixels valid in both
        alpha_a = datasets[idx_a].GetRasterBand(datasets[idx_a].RasterCount).ReadAsArray(
            xoff=a_col, yoff=a_row, win_xsize=a_cols, win_ysize=a_rows,
            buf_xsize=out_w, buf_ysize=out_h,
        )
        alpha_b = datasets[idx_b].GetRasterBand(datasets[idx_b].RasterCount).ReadAsArray(
            xoff=b_col, yoff=b_row, win_xsize=b_cols, win_ysize=b_rows,
            buf_xsize=out_w, buf_ysize=out_h,
        )

        both_valid = (alpha_a > 0) & (alpha_b > 0) & (img_a > 5) & (img_b > 5)
        if both_valid.sum() < 100:
            continue

        mean_a = img_a[both_valid].mean()
        std_a = img_a[both_valid].std()
        mean_b = img_b[both_valid].mean()
        std_b = img_b[both_valid].std()

        if std_a < 1 or std_b < 1:
            continue

        # Match B's statistics to A (A is closer to reference)
        # For strips farther from reference, we chain the correction
        if idx_a not in params:
            # A is reference-like, adjust B to match A
            scale_r = std_a / std_b
            offset_r = mean_a - scale_r * mean_b
            # Clamp to reasonable range
            scale_r = max(0.5, min(2.0, scale_r))
            offset_r = max(-50, min(50, offset_r))
            params[idx_b] = (scale_r, offset_r)
        elif idx_b not in params:
            # A has correction, adjust B to match corrected A
            sa, oa = params[idx_a]
            corrected_mean_a = mean_a * sa + oa
            corrected_std_a = std_a * sa
            scale_r = corrected_std_a / std_b
            offset_r = corrected_mean_a - scale_r * mean_b
            scale_r = max(0.5, min(2.0, scale_r))
            offset_r = max(-50, min(50, offset_r))
            params[idx_b] = (scale_r, offset_r)

    return params


def _prepare_strip(input_path, output_path, threshold=10, margin=500):
    """Detect black film borders and apply distance-based alpha feathering.

    Single-pass function that:
    1. Detects border regions at 1/32 scale using morphology + fill holes
    2. Computes distance transform for feathering ramp
    3. Writes full-res output chunked: alpha=0 in borders, feathered ramp elsewhere
    """
    from osgeo import gdal
    import numpy as np
    import cv2
    from scipy import ndimage

    gdal.UseExceptions()

    ds = gdal.Open(input_path)
    n_bands = ds.RasterCount
    w, h = ds.RasterXSize, ds.RasterYSize

    if n_bands < 2:
        import shutil
        shutil.copy2(input_path, output_path)
        return

    alpha_band_idx = n_bands
    scale = 32
    small_w, small_h = w // scale, h // scale

    # --- Border detection at 1/32 scale ---
    data_small = ds.GetRasterBand(1).ReadAsArray(
        buf_xsize=small_w, buf_ysize=small_h
    )
    valid = (data_small > threshold).astype(np.uint8)

    # Close to fill dark interior features (shadows, water bodies)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel_close)

    # Open to remove noise specks in border region
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, kernel_open)

    # Fill interior holes — keeps only border-connected dark as "border"
    valid = ndimage.binary_fill_holes(valid).astype(np.uint8)

    # Largest connected component filter to remove small isolated valid patches
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(valid, connectivity=8)
    if n_labels > 2:
        # Keep only the largest foreground component
        # Label 0 is background; find largest among labels >= 1
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        valid = (labels == largest).astype(np.uint8)

    # --- Distance-based feathering at 1/32 scale ---
    dist_small = cv2.distanceTransform(valid, cv2.DIST_L2, 5).astype(np.float32)
    # Scale distances to full-res pixel units
    dist_small *= scale
    # Compute alpha weight: linear ramp from 0 at border to 1 at margin distance
    margin_f = float(margin)
    weight_small = np.clip(dist_small / margin_f, 0.0, 1.0).astype(np.float32)

    print(f"      Border pixels (1/{scale}): {int(np.sum(valid == 0))} / {small_w * small_h} "
          f"({100 * np.sum(valid == 0) / (small_w * small_h):.1f}%)")

    # --- Full-res write (chunked) ---
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_path, w, h, n_bands,
                           ds.GetRasterBand(1).DataType,
                           ["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES",
                            "BIGTIFF=IF_SAFER"])
    if ds.GetGeoTransform():
        out_ds.SetGeoTransform(ds.GetGeoTransform())
    if ds.GetProjection():
        out_ds.SetProjection(ds.GetProjection())

    chunk_h = 1024
    for y in range(0, h, chunk_h):
        rows = min(chunk_h, h - y)

        # Determine corresponding rows in the small image
        sy_start = y // scale
        sy_end = min((y + rows + scale - 1) // scale, small_h)

        # Upscale mask and weight slices for this chunk
        mask_slice = valid[sy_start:sy_end, :]
        weight_slice = weight_small[sy_start:sy_end, :]

        mask_full = cv2.resize(mask_slice, (w, rows),
                               interpolation=cv2.INTER_NEAREST)
        weight_full = cv2.resize(weight_slice, (w, rows),
                                 interpolation=cv2.INTER_LINEAR)

        # Copy data bands
        for b in range(1, n_bands):
            data = ds.GetRasterBand(b).ReadAsArray(
                xoff=0, yoff=y, win_xsize=w, win_ysize=rows
            )
            out_ds.GetRasterBand(b).WriteArray(data, xoff=0, yoff=y)

        # Process alpha
        alpha_data = ds.GetRasterBand(alpha_band_idx).ReadAsArray(
            xoff=0, yoff=y, win_xsize=w, win_ysize=rows
        ).astype(np.float32)

        # Zero alpha in border regions
        alpha_data[mask_full == 0] = 0
        # Apply feathering ramp in valid regions
        valid_px = mask_full > 0
        alpha_data[valid_px] = (alpha_data[valid_px] * weight_full[valid_px]).clip(0, 255)

        out_ds.GetRasterBand(alpha_band_idx).WriteArray(
            alpha_data.astype(np.uint8), xoff=0, yoff=y
        )

        if y % (chunk_h * 50) == 0 and y > 0:
            pct = int(100 * y / h)
            print(f"      Strip prep: {pct}%")

    out_ds.FlushCache()
    out_ds = None
    ds = None


def _fill_interior_nodata(output_path):
    """Fill internal nodata holes in the composited mosaic with black (data=0, alpha=255).

    Reads alpha at 1/32 scale, fills holes, then patches full-res in chunks.
    """
    from osgeo import gdal
    import numpy as np
    import cv2
    from scipy import ndimage

    gdal.UseExceptions()

    ds = gdal.Open(output_path, gdal.GA_Update)
    w, h = ds.RasterXSize, ds.RasterYSize
    scale = 32
    small_w, small_h = w // scale, h // scale

    # Read alpha at reduced scale
    alpha_small = ds.GetRasterBand(2).ReadAsArray(
        buf_xsize=small_w, buf_ysize=small_h
    )
    has_data = alpha_small > 0
    filled = ndimage.binary_fill_holes(has_data)
    fill_mask_small = filled & ~has_data

    n_fill = int(np.sum(fill_mask_small))
    if n_fill == 0:
        print("    No interior nodata holes to fill")
        ds = None
        return

    print(f"    Filling {n_fill} interior nodata pixels (1/{scale} scale)")

    # Patch full-res in chunks
    data_band = ds.GetRasterBand(1)
    alpha_band = ds.GetRasterBand(2)
    chunk_h = 1024

    for y in range(0, h, chunk_h):
        rows = min(chunk_h, h - y)
        sy_start = y // scale
        sy_end = min((y + rows + scale - 1) // scale, small_h)

        # Check if any fill pixels in this row range
        fill_slice = fill_mask_small[sy_start:sy_end, :]
        if not np.any(fill_slice):
            continue

        # Upscale fill mask for this chunk
        fill_full = cv2.resize(fill_slice.astype(np.uint8), (w, rows),
                               interpolation=cv2.INTER_NEAREST).astype(bool)

        if not np.any(fill_full):
            continue

        data = data_band.ReadAsArray(xoff=0, yoff=y, win_xsize=w, win_ysize=rows)
        alpha = alpha_band.ReadAsArray(xoff=0, yoff=y, win_xsize=w, win_ysize=rows)

        data[fill_full] = 0
        alpha[fill_full] = 255

        data_band.WriteArray(data, xoff=0, yoff=y)
        alpha_band.WriteArray(alpha, xoff=0, yoff=y)

    ds.FlushCache()
    ds = None


def _alpha_composite(feathered_paths, output_path):
    """Alpha-weighted compositing of feathered strips.

    For each pixel: blended = sum(data_i * alpha_i) / sum(alpha_i).
    Processes in row chunks to limit memory usage.
    """
    from osgeo import gdal
    import numpy as np

    gdal.UseExceptions()

    # Open all inputs and compute output extent
    datasets = [gdal.Open(p) for p in feathered_paths]
    gts = [ds.GetGeoTransform() for ds in datasets]

    # All inputs should share the same pixel size
    pixel_w = gts[0][1]
    pixel_h = gts[0][5]

    # Compute union extent in geo coordinates
    extents = []
    for ds, gt in zip(datasets, gts):
        x_min = gt[0]
        y_max = gt[3]
        x_max = x_min + ds.RasterXSize * gt[1]
        y_min = y_max + ds.RasterYSize * gt[5]  # gt[5] is negative
        extents.append((x_min, y_min, x_max, y_max))

    union_x_min = min(e[0] for e in extents)
    union_y_min = min(e[1] for e in extents)
    union_x_max = max(e[2] for e in extents)
    union_y_max = max(e[3] for e in extents)

    out_w = int(round((union_x_max - union_x_min) / pixel_w))
    out_h = int(round((union_y_min - union_y_max) / pixel_h))  # pixel_h is negative

    out_gt = (union_x_min, pixel_w, 0.0, union_y_max, 0.0, pixel_h)

    # Precompute each input's pixel offset into the output grid
    offsets = []
    for ds, gt in zip(datasets, gts):
        ox = int(round((gt[0] - union_x_min) / pixel_w))
        oy = int(round((gt[3] - union_y_max) / pixel_h))
        offsets.append((ox, oy, ds.RasterXSize, ds.RasterYSize))

    # Create output: 1 data band + 1 alpha band
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_path, out_w, out_h, 2, gdal.GDT_Byte,
                           ["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES",
                            "BIGTIFF=IF_SAFER"])
    out_ds.SetGeoTransform(out_gt)
    out_ds.SetProjection(datasets[0].GetProjection())
    out_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_AlphaBand)

    out_data_band = out_ds.GetRasterBand(1)
    out_alpha_band = out_ds.GetRasterBand(2)

    chunk_h = 1024
    for y_out in range(0, out_h, chunk_h):
        rows = min(chunk_h, out_h - y_out)
        weight_sum = np.zeros((rows, out_w), dtype=np.float32)
        data_sum = np.zeros((rows, out_w), dtype=np.float32)

        for i, ds in enumerate(datasets):
            ox, oy, iw, ih = offsets[i]
            # Compute overlap between this chunk and this input
            src_y_start = y_out - oy
            src_y_end = src_y_start + rows
            dst_y_start = 0
            dst_y_end = rows

            if src_y_start < 0:
                dst_y_start = -src_y_start
                src_y_start = 0
            if src_y_end > ih:
                dst_y_end = rows - (src_y_end - ih)
                src_y_end = ih
            if src_y_start >= src_y_end:
                continue

            src_x_start = -ox if ox < 0 else 0
            dst_x_start = ox if ox >= 0 else 0
            src_x_end = min(iw, out_w - ox) if ox >= 0 else min(iw, out_w)
            dst_x_end = dst_x_start + (src_x_end - src_x_start)

            if src_x_start >= src_x_end:
                continue

            read_w = src_x_end - src_x_start
            read_h = src_y_end - src_y_start

            data_chunk = ds.GetRasterBand(1).ReadAsArray(
                xoff=int(src_x_start), yoff=int(src_y_start),
                win_xsize=int(read_w), win_ysize=int(read_h)
            ).astype(np.float32)

            alpha_chunk = ds.GetRasterBand(2).ReadAsArray(
                xoff=int(src_x_start), yoff=int(src_y_start),
                win_xsize=int(read_w), win_ysize=int(read_h)
            ).astype(np.float32)

            sl_y = slice(int(dst_y_start), int(dst_y_end))
            sl_x = slice(int(dst_x_start), int(dst_x_end))
            data_sum[sl_y, sl_x] += data_chunk * alpha_chunk
            weight_sum[sl_y, sl_x] += alpha_chunk

        # Compute blended result
        valid = weight_sum > 0
        out_data = np.zeros((rows, out_w), dtype=np.uint8)
        out_alpha = np.zeros((rows, out_w), dtype=np.uint8)
        out_data[valid] = (data_sum[valid] / weight_sum[valid]).clip(0, 255).astype(np.uint8)
        out_alpha[valid] = 255

        out_data_band.WriteArray(out_data, xoff=0, yoff=y_out)
        out_alpha_band.WriteArray(out_alpha, xoff=0, yoff=y_out)

        if y_out % (chunk_h * 20) == 0 and y_out > 0:
            pct = int(100 * y_out / out_h)
            print(f"    Compositing: {pct}%")

    out_ds.FlushCache()
    out_ds = None
    for ds in datasets:
        ds = None
    print(f"    Compositing: 100%")


def build_mosaic(aligned_paths: list, output_path: str, feather_margin: int = 500):
    """Build a mosaic from multiple aligned GeoTIFFs.

    Pipeline:
    1. Resample to common pixel size
    2. Pairwise feature matching for inter-strip alignment corrections
    3. Border detection + feathering
    4. Optimal seam finding (Dijkstra shortest path)
    5. Seam-based blending with radiometric equalization
    6. Interior nodata fill
    """
    if os.path.exists(output_path):
        print(f"  [skip] Mosaic already exists: {output_path}")
        return output_path

    if not aligned_paths:
        print(f"  WARNING: No aligned files to mosaic")
        return None

    if len(aligned_paths) == 1:
        import shutil
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(aligned_paths[0], output_path)
        print(f"  Single file mosaic: {output_path}")
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tmp_dir = os.path.join(os.path.dirname(output_path), "_mosaic_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # Step 1: Resample to common grid
        print("  Step 1/6: Resampling to common pixel size...")
        resampled_paths = _resample_to_common_grid(aligned_paths, tmp_dir)

        # Step 2: Pairwise corrections
        print("  Step 2/6: Computing pairwise corrections...")
        corrections = _compute_pairwise_corrections(resampled_paths)
        corrected_paths = _apply_geo_corrections(resampled_paths, corrections, tmp_dir)

        # Step 3: Prepare strips (border detection + feathering)
        print("  Step 3/6: Preparing strips...")
        prepared_paths = []
        for path in corrected_paths:
            prepared = os.path.join(tmp_dir, f"prep_{os.path.basename(path)}")
            print(f"    Preparing strip: {os.path.basename(path)} (threshold=10, margin={feather_margin}px)")
            _prepare_strip(path, prepared, threshold=10, margin=feather_margin)
            prepared_paths.append(prepared)

        # Step 4: Find optimal seams
        print("  Step 4/6: Finding optimal seams...")
        seams, strip_info = _find_seams(prepared_paths)

        # Step 5: Multi-band blend with seams
        print("  Step 5/6: Blending with seam-based compositing...")
        _multiband_blend(prepared_paths, seams, strip_info, output_path)

        # Step 6: Fill interior nodata holes
        print("  Step 6/6: Filling interior holes...")
        _fill_interior_nodata(output_path)

    finally:
        # Clean up temp files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"  Mosaic: {output_path} ({len(aligned_paths)} inputs)")
    return output_path


def group_for_mosaic(scenes: list, aligned_dir: str) -> dict:
    """Group aligned outputs by acquisition date + mission for mosaicking.

    Returns dict mapping mosaic_name -> list of aligned file paths.
    """
    groups = defaultdict(list)

    for scene in scenes:
        date_str = scene.acquisition_date.replace("/", "-")
        mission = scene.mission
        camera = scene.camera_system.name
        mosaic_name = f"{date_str}_{camera}_{mission}"

        # Look for aligned output
        aligned_path = os.path.join(aligned_dir, f"{scene.entity_id}_aligned.tif")
        if os.path.exists(aligned_path):
            groups[mosaic_name].append(aligned_path)

    return dict(groups)


def build_all_mosaics(scenes: list, aligned_dir: str, mosaic_dir: str) -> list:
    """Build all mosaics from aligned outputs, grouped by date/mission.

    Returns list of mosaic output paths.
    """
    groups = group_for_mosaic(scenes, aligned_dir)
    mosaic_paths = []

    for mosaic_name, aligned_paths in sorted(groups.items()):
        output_path = os.path.join(mosaic_dir, f"{mosaic_name}_mosaic.tif")
        result = build_mosaic(aligned_paths, output_path)
        if result:
            mosaic_paths.append(result)

    return mosaic_paths
