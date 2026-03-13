"""Frame stitching for multi-frame satellite imagery strips.

Generalized from process_d3c.py. Handles:
- Overlap detection via SIFT feature matching
- 180-degree rotation detection and correction
- VRT-based stitching (memory-safe for large strips)
- Strip direction auto-detection from corner coordinates
"""

import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


def _run_cmd(cmd, check=True):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result


def detect_strip_direction(scene_corners: dict) -> bool:
    """Detect if strip is reversed (frame_a at east end) from corner coordinates.

    Returns True if the strip runs east-to-west (reversed), False if west-to-east.
    A strip is reversed if the NW corner longitude > NE corner longitude,
    meaning the strip's geographic "west" is at the image's "east" pixel position.
    """
    nw_lon = scene_corners["NW"][1]
    ne_lon = scene_corners["NE"][1]
    # If NW lon > NE lon, the pixel coordinate system is flipped relative to geography
    return nw_lon > ne_lon


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


def compute_overlap(frame_a_path: str, frame_b_path: str, scale: float = 0.10) -> tuple:
    """Compute horizontal overlap between two adjacent frames using SIFT.

    Returns (overlap_px, frame_b_is_rotated).
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
        return (0, False)

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
        return (0, False)

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
            return (0, False)

        print(f"    Matches: {len(good)}, median overlap: {median_overlap}px "
              f"({100 * median_overlap / w_a:.1f}%)")
        return (median_overlap, False)

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
        return (0, False)

    img_b_rot = img_b_right[::-1, ::-1].astype(np.uint8)
    img_b_rot_eq = clahe.apply(img_b_rot)

    kp_b_rot, desc_b_rot = sift.detectAndCompute(img_b_rot_eq, None)
    if desc_b_rot is None or len(kp_b_rot) < 10:
        return (0, False)

    raw_matches_rot = flann.knnMatch(desc_b_rot, desc_a, k=2)
    good_rot = []
    for pair in raw_matches_rot:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                good_rot.append(m)

    if len(good_rot) < 10:
        return (0, False)

    pts_b_rot = np.float32([kp_b_rot[m.queryIdx].pt for m in good_rot])
    pts_a_rot = np.float32([kp_a[m.trainIdx].pt for m in good_rot])

    dx_values = []
    for pa, pb in zip(pts_a_rot, pts_b_rot):
        overlap_px = strip_w_a - pa[0] / scale + pb[0] / scale
        dx_values.append(overlap_px)

    median_overlap = round(np.median(dx_values))
    if median_overlap < 0 or median_overlap > w_a * 0.5:
        return (0, False)

    print(f"    ROTATED matches: {len(good_rot)}, median overlap: {median_overlap}px "
          f"({100 * median_overlap / w_a:.1f}%)")
    return (median_overlap, True)


def _blend_overlaps(output_path, frame_info, x_offsets, overlaps):
    """Post-process overlap zones with linear feathered blending."""
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

        alpha = np.linspace(1.0, 0.0, overlap_w, dtype=np.float32)

        for y in range(0, h, chunk_h):
            rows = min(chunk_h, h - y)
            for b in range(1, n_bands + 1):
                strip_a = ds_a.GetRasterBand(b).ReadAsArray(
                    xoff=a_xoff, yoff=y, win_xsize=overlap_w, win_ysize=rows
                ).astype(np.float32)
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


def stitch_frames(frames: list, output_path: str, output_dir: str) -> str:
    """Stitch a list of frame paths into a single panoramic strip.

    Args:
        frames: List of frame file paths, in strip order.
        output_path: Where to write the stitched output.
        output_dir: Working directory for temp files.

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

    # Get dimensions of each frame
    frame_info = []
    for f in frames:
        result = _run_cmd(["gdalinfo", "-json", f])
        info = json.loads(result.stdout)
        w, h = info["size"]
        bands = len(info.get("bands", [{"type": "Byte"}]))
        dtype = info.get("bands", [{"type": "Byte"}])[0].get("type", "Byte")
        frame_info.append({"path": f, "w": w, "h": h, "bands": bands, "dtype": dtype})
        print(f"    Frame {os.path.basename(f)}: {w}x{h}")

    # Compute overlaps in parallel (with rotation detection)
    print(f"  Computing overlaps between {len(frames)} frames...")
    with ProcessPoolExecutor() as executor:
        futures = {}
        for i in range(len(frames) - 1):
            fut = executor.submit(compute_overlap, frames[i], frames[i + 1])
            futures[fut] = i

        results = [None] * (len(frames) - 1)
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()

    overlaps = [r[0] for r in results]
    rotated_flags = [r[1] for r in results]

    # Handle rotated frames: create 180-flipped copies
    rot180_temps = []
    for i, is_rotated in enumerate(rotated_flags):
        if is_rotated:
            frame_idx = i + 1
            orig_path = frame_info[frame_idx]["path"]
            flipped_path = orig_path.replace(".tif", "_rot180.tif")
            print(f"  Frame {os.path.basename(orig_path)} is rotated 180 — creating corrected copy")
            flip_frame_180(orig_path, flipped_path)
            frame_info[frame_idx]["path"] = flipped_path
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

    # Clean up rotated temp files
    for temp_path in rot180_temps:
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
