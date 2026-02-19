#!/usr/bin/env python3
"""
Process KH-9 (Declass 3) satellite imagery end-to-end:
  1. Download EarthExplorer "full" metadata XML (4 corner coords + date)
  2. Extract .tgz archives
  3. Stitch frames using stitch_frames.py
  4. Parse metadata XML for corner coordinates and acquisition date
  5. Georectify using GCPs (handles rotated panoramic strips)
  6. Crop to Bahrain maritime boundary
  7. Rename to final format: YYYY-MM-DD - Bahrain - ENTITY.tif
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed

from align.metadata_priors import parse_ee_metadata_xml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ENTITIES = [
    "D3C1213-200346A003",
    "D3C1214-200421F003",
    "D3C1216-300950A033",
    "D3C1216-401438A002",
    "D3C1216-401438F001",
    "D3C1217-100109A007",
    "D3C1217-100109A008",
    "D3C1217-100109F007",
    "D3C1219-200801F037",
]

# Entities where frame _a is at the east end (strip runs east→west alphabetically).
# Determined from visual inspection of georectified outputs.
REVERSED_ENTITIES = {
    "D3C1213-200346A003",
    "D3C1216-300950A033",
    "D3C1216-401438A002",
    "D3C1217-100109A007",
    "D3C1217-100109A008",
    "D3C1219-200801F037",
}

EE_DATASET_ID = "5e7c41f3ffaaf662"

DIRS = {
    "metadata": os.path.join(BASE_DIR, "metadata"),
    "extracted": os.path.join(BASE_DIR, "extracted"),
    "stitched": os.path.join(BASE_DIR, "stitched"),
    "georef": os.path.join(BASE_DIR, "georef"),
    "cropped": os.path.join(BASE_DIR, "cropped"),
    "final": os.path.join(BASE_DIR, "final"),
}

BAHRAIN_GEOJSON_PATH = os.path.join(BASE_DIR, "data", "bahrain_boundary.geojson")

# Approximate rectangle covering Bahrain + territorial waters + Hawar Islands
BAHRAIN_BOUNDARY = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [50.15, 25.55],
                [50.90, 25.55],
                [50.90, 26.40],
                [50.15, 26.40],
                [50.15, 25.55],
            ]]
        }
    }]
}

NS = {"ee": "http://earthexplorer.usgs.gov/eemetadata.xsd"}


def ensure_dirs():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)


def write_bahrain_geojson():
    if not os.path.exists(BAHRAIN_GEOJSON_PATH):
        with open(BAHRAIN_GEOJSON_PATH, "w") as f:
            json.dump(BAHRAIN_BOUNDARY, f, indent=2)
        print(f"  Wrote {BAHRAIN_GEOJSON_PATH}")


def run_cmd(cmd, check=True):
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


# ---------- Step 1: Download metadata ----------

def download_metadata(entity, cookie_string):
    output = os.path.join(DIRS["metadata"], f"{entity}_metadata.xml")
    if os.path.exists(output):
        print(f"  [skip] Metadata already exists: {output}")
        return output

    url = (
        f"https://earthexplorer.usgs.gov/scene/metadata/full/"
        f"{EE_DATASET_ID}/{entity}/?responseType=saveXml"
    )
    cmd = [
        "curl", "-sS", "-L",
        "--cookie", cookie_string,
        "-o", output,
        url,
    ]
    run_cmd(cmd)

    # Verify it's XML, not an HTML error page
    with open(output, "r", errors="replace") as f:
        head = f.read(200)
    if "<html" in head.lower() or "<!doctype" in head.lower():
        print(f"  WARNING: {output} looks like HTML, not XML. Auth may have failed.")
        os.remove(output)
        raise RuntimeError(f"Metadata download returned HTML for {entity}. Check cookies.")

    print(f"  Downloaded: {output}")
    return output


# ---------- Step 2: Extract tgz ----------

def extract_tgz(entity):
    tgz_path = os.path.join(BASE_DIR, f"{entity}.tgz")
    out_dir = os.path.join(DIRS["extracted"], entity)

    if os.path.exists(out_dir) and os.listdir(out_dir):
        tifs = sorted(f for f in os.listdir(out_dir) if f.endswith(".tif"))
        if tifs:
            print(f"  [skip] Already extracted {len(tifs)} frames in {out_dir}")
            return out_dir

    if not os.path.exists(tgz_path):
        raise FileNotFoundError(f"Archive not found: {tgz_path}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"  Extracting {tgz_path} ...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=out_dir, filter="data")

    tifs = sorted(f for f in os.listdir(out_dir) if f.endswith(".tif"))
    print(f"  Extracted {len(tifs)} frames: {', '.join(tifs)}")
    return out_dir


# ---------- Step 3: Stitch frames ----------

def flip_frame_180(src_path, dst_path):
    """Rotate a frame 180° (flip both axes) and save as compressed TIFF."""
    from osgeo import gdal
    import numpy as np
    gdal.UseExceptions()
    ds = gdal.Open(src_path)
    w, h, n_bands = ds.RasterXSize, ds.RasterYSize, ds.RasterCount
    dt = ds.GetRasterBand(1).DataType
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(dst_path, w, h, n_bands, dt,
                           ['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES'])
    for b in range(1, n_bands + 1):
        arr = ds.GetRasterBand(b).ReadAsArray()
        out_ds.GetRasterBand(b).WriteArray(arr[::-1, ::-1])
    out_ds.FlushCache()
    out_ds = None
    ds = None


def compute_overlap(frame_a_path, frame_b_path, scale=0.10):
    """Compute horizontal overlap between two adjacent frames using SIFT at low resolution.
    Returns (overlap_px, frame_b_is_rotated) — overlap in full-res pixels, and whether
    frame B appears to be stored 180° rotated relative to frame A."""
    from osgeo import gdal
    import numpy as np
    import cv2

    gdal.UseExceptions()

    # Read right edge of frame_a and left edge of frame_b at low resolution
    ds_a = gdal.Open(frame_a_path)
    ds_b = gdal.Open(frame_b_path)
    w_a, h_a = ds_a.RasterXSize, ds_a.RasterYSize
    w_b, h_b = ds_b.RasterXSize, ds_b.RasterYSize

    # Read the right 30% of frame_a and left 30% of frame_b (overlap region)
    strip_frac = 0.30
    strip_w_a = int(w_a * strip_frac)
    strip_w_b = int(w_b * strip_frac)

    # Read at reduced resolution
    out_w_a = int(strip_w_a * scale)
    out_h_a = int(h_a * scale)
    out_w_b = int(strip_w_b * scale)
    out_h_b = int(h_b * scale)

    band_a = ds_a.GetRasterBand(1)
    band_b = ds_b.GetRasterBand(1)

    # Right edge of A
    img_a = band_a.ReadAsArray(
        xoff=w_a - strip_w_a, yoff=0, win_xsize=strip_w_a, win_ysize=h_a,
        buf_xsize=out_w_a, buf_ysize=out_h_a
    )
    # Left edge of B (normal orientation)
    img_b = band_b.ReadAsArray(
        xoff=0, yoff=0, win_xsize=strip_w_b, win_ysize=h_b,
        buf_xsize=out_w_b, buf_ysize=out_h_b
    )

    if img_a is None or img_b is None:
        print(f"    WARNING: Could not read strips, assuming no overlap")
        ds_a = None
        ds_b = None
        return (0, False)

    img_a = img_a.astype(np.uint8)
    img_b = img_b.astype(np.uint8)

    # CLAHE for contrast normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_a_eq = clahe.apply(img_a)
    img_b_eq = clahe.apply(img_b)

    # SIFT feature detection
    sift = cv2.SIFT_create(nfeatures=5000)
    kp_a, desc_a = sift.detectAndCompute(img_a_eq, None)
    kp_b, desc_b = sift.detectAndCompute(img_b_eq, None)

    if desc_a is None or desc_b is None or len(kp_a) < 10 or len(kp_b) < 10:
        print(f"    WARNING: Not enough features, assuming no overlap")
        ds_a = None
        ds_b = None
        return (0, False)

    # FLANN matching
    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
        dict(checks=50),
    )
    raw_matches = flann.knnMatch(desc_b, desc_a, k=2)

    # Lowe's ratio test
    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                good.append(m)

    if len(good) >= 10:
        # Normal (non-rotated) match succeeded
        ds_a = None
        ds_b = None

        pts_b = np.float32([kp_b[m.queryIdx].pt for m in good])
        pts_a = np.float32([kp_a[m.trainIdx].pt for m in good])

        dx_values = []
        for pa, pb in zip(pts_a, pts_b):
            overlap_px = strip_w_a - pa[0] / scale + pb[0] / scale
            dx_values.append(overlap_px)

        median_overlap = int(np.median(dx_values))

        if median_overlap < 0 or median_overlap > w_a * 0.5:
            print(f"    WARNING: Computed overlap {median_overlap}px seems wrong, assuming no overlap")
            return (0, False)

        print(f"    Matches: {len(good)}, median overlap: {median_overlap}px ({100*median_overlap/w_a:.1f}%)")
        return (median_overlap, False)

    # --- Normal matching failed, try with B rotated 180° ---
    print(f"    Only {len(good)} matches in normal orientation, trying 180° rotation...")

    # If B is stored 180° rotated, its correct left edge (which should overlap A's right)
    # is physically at B's right edge, upside-down. Read right strip and flip 180°.
    strip_w_b_right = int(w_b * strip_frac)
    out_w_b_right = int(strip_w_b_right * scale)
    out_h_b_right = int(h_b * scale)

    img_b_right = band_b.ReadAsArray(
        xoff=w_b - strip_w_b_right, yoff=0,
        win_xsize=strip_w_b_right, win_ysize=h_b,
        buf_xsize=out_w_b_right, buf_ysize=out_h_b_right
    )

    ds_a = None
    ds_b = None

    if img_b_right is None:
        print(f"    WARNING: Could not read rotated strip, assuming no overlap")
        return (0, False)

    # Flip 180° to reconstruct what B's left edge should look like
    img_b_rot = img_b_right[::-1, ::-1].astype(np.uint8)
    img_b_rot_eq = clahe.apply(img_b_rot)

    kp_b_rot, desc_b_rot = sift.detectAndCompute(img_b_rot_eq, None)

    if desc_b_rot is None or len(kp_b_rot) < 10:
        print(f"    WARNING: Not enough features in rotated strip, assuming no overlap")
        return (0, False)

    raw_matches_rot = flann.knnMatch(desc_b_rot, desc_a, k=2)

    good_rot = []
    for pair in raw_matches_rot:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                good_rot.append(m)

    if len(good_rot) < 10:
        print(f"    WARNING: Only {len(good_rot)} matches even with rotation, assuming no overlap")
        return (0, False)

    # Rotation match succeeded!
    pts_b_rot = np.float32([kp_b_rot[m.queryIdx].pt for m in good_rot])
    pts_a_rot = np.float32([kp_a[m.trainIdx].pt for m in good_rot])

    dx_values = []
    for pa, pb in zip(pts_a_rot, pts_b_rot):
        # Same formula: the rotated strip now acts as B's left edge
        overlap_px = strip_w_a - pa[0] / scale + pb[0] / scale
        dx_values.append(overlap_px)

    median_overlap = int(np.median(dx_values))

    if median_overlap < 0 or median_overlap > w_a * 0.5:
        print(f"    WARNING: Rotated overlap {median_overlap}px seems wrong, assuming no overlap")
        return (0, False)

    print(f"    ROTATED matches: {len(good_rot)}, median overlap: {median_overlap}px ({100*median_overlap/w_a:.1f}%)")
    return (median_overlap, True)


def select_bahrain_frames(frames, metadata, reversed_order=False):
    """Select only frames whose estimated geographic extent intersects Bahrain.
    Returns list of selected frame indices.

    If reversed_order=True, frame 0 is at the east end of the strip
    (i.e. alphabetical order runs east→west, not west→east)."""
    corners = metadata["corners"]
    n = len(frames)

    # Use average of top/bottom longitude for interpolation
    west_lon = (corners["NW"][1] + corners["SW"][1]) / 2
    east_lon = (corners["NE"][1] + corners["SE"][1]) / 2

    # Estimate per-frame longitude range assuming uniform spacing
    frame_lon_width = (east_lon - west_lon) / n

    bahrain_west = 50.15
    bahrain_east = 50.90

    selected = []
    for i in range(n):
        if reversed_order:
            # Frame 0 is at east end, frame n-1 is at west end
            frame_west = east_lon - (i + 1) * frame_lon_width
            frame_east = east_lon - i * frame_lon_width
        else:
            # Frame 0 is at west end, frame n-1 is at east end
            frame_west = west_lon + i * frame_lon_width
            frame_east = west_lon + (i + 1) * frame_lon_width
        # Check intersection with Bahrain longitude range
        if frame_east >= bahrain_west and frame_west <= bahrain_east:
            selected.append(i)

    # Add 1-frame buffer on each side for overlap computation accuracy
    if selected:
        min_idx = max(0, selected[0] - 1)
        max_idx = min(n - 1, selected[-1] + 1)
        selected = list(range(min_idx, max_idx + 1))

    if not selected:
        # Fallback: use all frames if no intersection found
        print(f"    WARNING: No frames intersect Bahrain, using all frames")
        selected = list(range(n))
    else:
        direction = "reversed" if reversed_order else "forward"
        print(f"    Selected frames {selected[0]}-{selected[-1]} of {n} "
              f"({len(selected)}/{n} frames cover Bahrain) [{direction} strip]")

    return selected


def compute_subset_corners(metadata, selected_indices, n_total_frames, reversed_order=False):
    """Interpolate corner coordinates for a subset of frames."""
    corners = metadata["corners"]
    n = n_total_frames

    # Fractions along the strip (NW→NE) for the subset's geographic west and east edges
    if reversed_order:
        # Frame 0 = east end, high indices = west end
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
        # Frames are stored 180° from geographic north-up. The stitched image
        # (in original index order) has its pixel (0,0) at geographic SE and
        # pixel (W,H) at geographic NW. Swap the corner labels so georectify()
        # assigns the correct coordinates to each pixel position — gdalwarp
        # will handle the 180° flip during warping.
        return {
            "NW": geo_se,
            "NE": geo_sw,
            "SE": geo_nw,
            "SW": geo_ne,
        }

    return {
        "NW": geo_nw,
        "NE": geo_ne,
        "SW": geo_sw,
        "SE": geo_se,
    }


def stitch_frames(entity, metadata=None):
    """Stitch frames using GDAL VRT with overlap detection (memory-safe).
    Returns (output_path, selected_indices, n_total_frames, is_reversed) when
    metadata is provided for frame selection, or (output_path, None, None, False)
    otherwise."""
    output = os.path.join(DIRS["stitched"], f"{entity}_stitched.tif")
    if os.path.exists(output):
        print(f"  [skip] Stitched output already exists: {output}")
        return output, None, None, False

    extract_dir = os.path.join(DIRS["extracted"], entity)
    all_frames = sorted(
        os.path.join(extract_dir, f)
        for f in os.listdir(extract_dir)
        if f.endswith(".tif")
    )
    n_total_frames = len(all_frames)

    if n_total_frames < 2:
        if n_total_frames == 1:
            import shutil
            shutil.copy2(all_frames[0], output)
            print(f"  Single frame, copied to {output}")
            return output, None, None, False
        raise RuntimeError(f"No frames found in {extract_dir}")

    # Strip direction: frame _a at east end (reversed) or west end (forward)
    is_reversed = entity in REVERSED_ENTITIES
    if is_reversed:
        print(f"  Strip direction: REVERSED (frames run east→west)")
    else:
        print(f"  Strip direction: normal (frames run west→east)")

    # Select only frames covering Bahrain if metadata available
    selected_indices = None
    if metadata:
        selected_indices = select_bahrain_frames(all_frames, metadata, reversed_order=is_reversed)
        frames = [all_frames[i] for i in selected_indices]
    else:
        frames = all_frames

    # Get dimensions of each frame
    frame_info = []
    for f in frames:
        result = run_cmd(["gdalinfo", "-json", f])
        info = json.loads(result.stdout)
        w, h = info["size"]
        bands = len(info.get("bands", [{"type": "Byte"}]))
        dtype = info.get("bands", [{"type": "Byte"}])[0].get("type", "Byte")
        frame_info.append({"path": f, "w": w, "h": h, "bands": bands, "dtype": dtype})
        print(f"  Frame {os.path.basename(f)}: {w}x{h}")

    if len(frames) < 2:
        import shutil
        shutil.copy2(frames[0], output)
        print(f"  Single selected frame, copied to {output}")
        return output, selected_indices, n_total_frames, is_reversed

    # Compute overlaps in parallel (with rotation detection)
    print(f"  Computing overlaps between {len(frames)} frames...")
    with ProcessPoolExecutor() as executor:
        futures = {}
        for i in range(len(frames) - 1):
            pair_name = f"{os.path.basename(frames[i])} ↔ {os.path.basename(frames[i+1])}"
            print(f"    Submitting pair {i+1}: {pair_name}")
            fut = executor.submit(compute_overlap, frames[i], frames[i + 1])
            futures[fut] = i

        # Collect results in order
        results = [None] * (len(frames) - 1)
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()

    overlaps = [r[0] for r in results]
    rotated_flags = [r[1] for r in results]

    # Handle rotated frames: create 180°-flipped copies and update paths
    rot180_temps = []
    for i, is_rotated in enumerate(rotated_flags):
        if is_rotated:
            frame_idx = i + 1  # frame B in pair i
            orig_path = frame_info[frame_idx]["path"]
            flipped_path = orig_path.replace('.tif', '_rot180.tif')
            print(f"  Frame {os.path.basename(orig_path)} is rotated 180° — creating corrected copy")
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

    # Build VRT with overlap-aware offsets
    vrt_path = os.path.join(DIRS["stitched"], f"{entity}_stitch.vrt")
    vrt_lines = [
        f'<VRTDataset rasterXSize="{total_w}" rasterYSize="{max_h}">',
    ]
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

    # Render VRT to GeoTIFF (GDAL handles tiled I/O, no OOM)
    run_cmd([
        "gdal_translate",
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=IF_SAFER",
        vrt_path,
        output,
    ])

    os.remove(vrt_path)

    # Clean up rotated temp files
    for temp_path in rot180_temps:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"  Removed temp: {os.path.basename(temp_path)}")

    print(f"  Stitched {len(frames)} frames → {output}")
    return output, selected_indices, n_total_frames, is_reversed


# ---------- Step 4: Parse EE metadata XML ----------

def parse_ee_metadata(entity):
    xml_path = os.path.join(DIRS["metadata"], f"{entity}_metadata.xml")
    prior = parse_ee_metadata_xml(xml_path)
    acq_date = str(prior.attributes.get("acquisition_date") or "unknown").replace("/", "-")
    corners = prior.corners
    entity_id = str(prior.attributes.get("entity_id") or entity)

    print(f"  Date: {acq_date}")
    print(f"  Corners: NW={corners['NW']}, NE={corners['NE']}, "
          f"SE={corners['SE']}, SW={corners['SW']}")

    return {
        "date": acq_date,
        "corners": corners,
        "entity_id": entity_id,
    }


# ---------- Step 5: Georectify ----------

def georectify(entity, metadata):
    output = os.path.join(DIRS["georef"], f"{entity}_georef.tif")
    if os.path.exists(output):
        print(f"  [skip] Georef output already exists: {output}")
        return output

    stitched = os.path.join(DIRS["stitched"], f"{entity}_stitched.tif")
    corners = metadata["corners"]

    # Get image dimensions
    info_result = run_cmd(["gdalinfo", "-json", stitched])
    info = json.loads(info_result.stdout)
    width = info["size"][0]
    height = info["size"][1]
    print(f"  Image size: {width} x {height}")

    # Assign GCPs with gdal_translate
    temp_gcp = os.path.join(DIRS["georef"], f"{entity}_gcp.tif")
    nw_lat, nw_lon = corners["NW"]
    ne_lat, ne_lon = corners["NE"]
    se_lat, se_lon = corners["SE"]
    sw_lat, sw_lon = corners["SW"]

    cmd_translate = [
        "gdal_translate",
        "-a_srs", "EPSG:4326",
        "-gcp", "0", "0", str(nw_lon), str(nw_lat),
        "-gcp", str(width), "0", str(ne_lon), str(ne_lat),
        "-gcp", str(width), str(height), str(se_lon), str(se_lat),
        "-gcp", "0", str(height), str(sw_lon), str(sw_lat),
        stitched,
        temp_gcp,
    ]
    run_cmd(cmd_translate)

    # Warp to EPSG:3857 using the GCPs
    cmd_warp = [
        "gdalwarp",
        "-s_srs", "EPSG:4326",
        "-t_srs", "EPSG:3857",
        "-order", "1",
        "-r", "lanczos",
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=IF_SAFER",
        "-dstalpha",
        temp_gcp,
        output,
    ]
    try:
        run_cmd(cmd_warp)
    finally:
        # Always clean up the large GCP temp file
        if os.path.exists(temp_gcp):
            os.remove(temp_gcp)
            print(f"  Removed temp: {temp_gcp}")

    print(f"  Georeferenced: {output}")
    return output


# ---------- Step 6: Crop to Bahrain ----------

def crop_to_bahrain(entity):
    output = os.path.join(DIRS["cropped"], f"{entity}_cropped.tif")
    if os.path.exists(output):
        print(f"  [skip] Cropped output already exists: {output}")
        return output

    georef_path = os.path.join(DIRS["georef"], f"{entity}_georef.tif")

    cmd = [
        "gdalwarp",
        "-cutline", BAHRAIN_GEOJSON_PATH,
        "-cutline_srs", "EPSG:4326",
        "-crop_to_cutline",
        "-dstalpha",
        "-r", "lanczos",
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=IF_SAFER",
        georef_path,
        output,
    ]
    run_cmd(cmd)
    print(f"  Cropped: {output}")
    return output


# ---------- Step 6b: Trim transparent edges ----------

def trim_nodata(entity):
    """Trim transparent (alpha=0) edges from the cropped image."""
    from osgeo import gdal
    import numpy as np

    cropped = os.path.join(DIRS["cropped"], f"{entity}_cropped.tif")
    trimmed = os.path.join(DIRS["cropped"], f"{entity}_trimmed.tif")

    if os.path.exists(trimmed):
        print(f"  [skip] Trimmed output already exists: {trimmed}")
        return trimmed

    ds = gdal.Open(cropped)
    if ds is None:
        raise RuntimeError(f"Could not open {cropped}")

    # Find the alpha band (last band)
    n_bands = ds.RasterCount
    alpha_band = ds.GetRasterBand(n_bands)

    # Read alpha channel and find bounding box of non-zero pixels
    alpha = alpha_band.ReadAsArray()
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not rows.any():
        print(f"  WARNING: No non-transparent pixels found in {cropped}")
        ds = None
        return cropped

    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Convert pixel coords to geo coords using geotransform
    gt = ds.GetGeoTransform()
    # gt: (origin_x, pixel_width, 0, origin_y, 0, pixel_height)
    ulx = gt[0] + col_min * gt[1]
    uly = gt[3] + row_min * gt[5]
    lrx = gt[0] + (col_max + 1) * gt[1]
    lry = gt[3] + (row_max + 1) * gt[5]

    orig_w, orig_h = ds.RasterXSize, ds.RasterYSize
    trim_w = col_max - col_min + 1
    trim_h = row_max - row_min + 1
    ds = None

    print(f"  Data extent: rows [{row_min}:{row_max}], cols [{col_min}:{col_max}]")
    print(f"  Trimming: {orig_w}x{orig_h} → {trim_w}x{trim_h}")

    run_cmd([
        "gdal_translate",
        "-projwin", str(ulx), str(uly), str(lrx), str(lry),
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=IF_SAFER",
        cropped,
        trimmed,
    ])

    print(f"  Trimmed: {trimmed}")
    return trimmed


# ---------- Step 7: Rename to final ----------

def rename_final(entity, metadata):
    date = metadata["date"]
    final_name = f"{date} - Bahrain - {entity}.tif"
    output = os.path.join(DIRS["final"], final_name)

    if os.path.exists(output):
        print(f"  [skip] Final output already exists: {output}")
        return output

    trimmed = os.path.join(DIRS["cropped"], f"{entity}_trimmed.tif")
    cropped = os.path.join(DIRS["cropped"], f"{entity}_cropped.tif")
    src = trimmed if os.path.exists(trimmed) else cropped
    import shutil
    shutil.copy2(src, output)
    print(f"  Final: {output}")
    return output


# ---------- Main pipeline ----------

def process_entity(entity, cookie_string=None, skip_download=False):
    print(f"\n{'='*60}")
    print(f"Processing: {entity}")
    print(f"{'='*60}")

    # Step 1: Download metadata
    if not skip_download:
        if not cookie_string:
            raise ValueError("--cookie-string required for metadata download")
        print("\n  --- Step 1: Download metadata ---")
        download_metadata(entity, cookie_string)
    else:
        xml_path = os.path.join(DIRS["metadata"], f"{entity}_metadata.xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Metadata XML not found (and --skip-download set): {xml_path}")
        print(f"\n  --- Step 1: [skipped] Using existing metadata ---")

    # Step 2: Extract
    print("\n  --- Step 2: Extract tgz ---")
    extract_tgz(entity)

    # Step 4 (moved up): Parse metadata for corner coords
    print("\n  --- Step 4: Parse metadata ---")
    metadata = parse_ee_metadata(entity)

    # Step 3: Stitch (with frame selection using metadata)
    print("\n  --- Step 3: Stitch frames ---")
    _, selected_indices, n_total_frames, is_reversed = stitch_frames(entity, metadata)

    # Adjust GCPs if frame selection was used
    if selected_indices is not None:
        metadata["corners"] = compute_subset_corners(
            metadata, selected_indices, n_total_frames, reversed_order=is_reversed
        )
        print(f"  Adjusted GCPs for frame subset {selected_indices[0]}-{selected_indices[-1]}")

    # Step 5: Georectify
    print("\n  --- Step 5: Georectify ---")
    georectify(entity, metadata)

    # Clean up stitched intermediate to save disk
    stitched_path = os.path.join(DIRS["stitched"], f"{entity}_stitched.tif")
    if os.path.exists(stitched_path):
        os.remove(stitched_path)
        print(f"  Removed intermediate: {stitched_path}")

    # Step 6: Crop
    print("\n  --- Step 6: Crop to Bahrain ---")
    crop_to_bahrain(entity)

    # Clean up georef intermediate to save disk
    georef_path = os.path.join(DIRS["georef"], f"{entity}_georef.tif")
    if os.path.exists(georef_path):
        os.remove(georef_path)
        print(f"  Removed intermediate: {georef_path}")

    # Step 6b: Trim transparent edges
    print("\n  --- Step 6b: Trim transparent edges ---")
    trim_nodata(entity)

    # Step 7: Rename
    print("\n  --- Step 7: Rename final ---")
    rename_final(entity, metadata)


def main():
    parser = argparse.ArgumentParser(
        description="Process KH-9 (Declass 3) satellite imagery for Bahrain"
    )
    parser.add_argument(
        "--cookie-string",
        help="Raw cookie string from browser DevTools (for EE metadata download)",
    )
    parser.add_argument(
        "--entities",
        nargs="+",
        help="Process only these entities (default: all 9)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip metadata download (use existing XMLs)",
    )
    parser.add_argument(
        "--retrim",
        action="store_true",
        help="Re-trim and re-copy to final (skip steps 1-6, redo 6b+7)",
    )
    parser.add_argument(
        "--restitch",
        action="store_true",
        help="Re-stitch and re-run steps 3-7 (remove stitched/georef/cropped/final, redo from stitch)",
    )
    args = parser.parse_args()

    entities = args.entities if args.entities else ENTITIES

    # Validate entity names
    for e in entities:
        if e not in ENTITIES:
            print(f"WARNING: {e} not in known entity list, proceeding anyway")

    print(f"Working directory: {BASE_DIR}")
    print(f"Entities to process: {len(entities)}")

    ensure_dirs()
    write_bahrain_geojson()

    success = 0
    failed = []

    if args.restitch:
        # Re-run from stitch onwards (steps 3-7)
        for entity in entities:
            try:
                print(f"\n{'='*60}")
                print(f"Re-stitching: {entity}")
                print(f"{'='*60}")

                # Remove existing outputs so they get regenerated
                stitched = os.path.join(DIRS["stitched"], f"{entity}_stitched.tif")
                georef = os.path.join(DIRS["georef"], f"{entity}_georef.tif")
                cropped = os.path.join(DIRS["cropped"], f"{entity}_cropped.tif")
                trimmed = os.path.join(DIRS["cropped"], f"{entity}_trimmed.tif")
                for p in [stitched, georef, cropped, trimmed]:
                    if os.path.exists(p):
                        os.remove(p)
                        print(f"  Removed: {p}")

                metadata = parse_ee_metadata(entity)
                date = metadata["date"]
                final_path = os.path.join(DIRS["final"], f"{date} - Bahrain - {entity}.tif")
                if os.path.exists(final_path):
                    os.remove(final_path)
                    print(f"  Removed: {final_path}")

                print("\n  --- Step 3: Stitch frames ---")
                _, selected_indices, n_total_frames, is_reversed = stitch_frames(entity, metadata)

                # Adjust GCPs if frame selection was used
                if selected_indices is not None:
                    metadata["corners"] = compute_subset_corners(
                        metadata, selected_indices, n_total_frames, reversed_order=is_reversed
                    )
                    print(f"  Adjusted GCPs for frame subset {selected_indices[0]}-{selected_indices[-1]}")

                print("\n  --- Step 5: Georectify ---")
                georectify(entity, metadata)

                # Clean up stitched intermediate
                if os.path.exists(stitched):
                    os.remove(stitched)
                    print(f"  Removed intermediate: {stitched}")

                print("\n  --- Step 6: Crop to Bahrain ---")
                crop_to_bahrain(entity)

                # Clean up georef intermediate
                georef = os.path.join(DIRS["georef"], f"{entity}_georef.tif")
                if os.path.exists(georef):
                    os.remove(georef)
                    print(f"  Removed intermediate: {georef}")

                print("\n  --- Step 6b: Trim transparent edges ---")
                trim_nodata(entity)

                print("\n  --- Step 7: Rename final ---")
                rename_final(entity, metadata)
                success += 1
            except Exception as e:
                print(f"\n  ERROR processing {entity}: {e}")
                failed.append((entity, str(e)))
    elif args.retrim:
        # Just re-trim existing cropped files and update final
        for entity in entities:
            try:
                print(f"\n{'='*60}")
                print(f"Re-trimming: {entity}")
                print(f"{'='*60}")

                # Remove old trimmed + final so they get regenerated
                trimmed = os.path.join(DIRS["cropped"], f"{entity}_trimmed.tif")
                if os.path.exists(trimmed):
                    os.remove(trimmed)

                # Parse metadata for date (needed for final filename)
                metadata = parse_ee_metadata(entity)

                # Remove old final
                date = metadata["date"]
                final_path = os.path.join(DIRS["final"], f"{date} - Bahrain - {entity}.tif")
                if os.path.exists(final_path):
                    os.remove(final_path)

                print("\n  --- Step 6b: Trim transparent edges ---")
                trim_nodata(entity)

                print("\n  --- Step 7: Rename final ---")
                rename_final(entity, metadata)
                success += 1
            except Exception as e:
                print(f"\n  ERROR processing {entity}: {e}")
                failed.append((entity, str(e)))
    else:
        for entity in entities:
            try:
                process_entity(
                    entity,
                    cookie_string=args.cookie_string,
                    skip_download=args.skip_download,
                )
                success += 1
            except Exception as e:
                print(f"\n  ERROR processing {entity}: {e}")
                failed.append((entity, str(e)))

    print(f"\n{'='*60}")
    print(f"Done. {success}/{len(entities)} succeeded.")
    if failed:
        print("Failed:")
        for entity, err in failed:
            print(f"  {entity}: {err}")


if __name__ == "__main__":
    main()
