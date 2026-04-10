#!/usr/bin/env python3
"""Run bundle_adjust + stereo + DEM on the 003 stereo pair using RoMa matches.

Strategy: mapproject both images through camera models first, match in
geographic space via RoMa (where both images are co-registered), then
unproject back to raw pixel coords via RPC models for bundle_adjust.

All outputs go to output/stereo/ba_roma/.

Usage:
    python3 scripts/run_stereo_roma_ba.py
"""

import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

STEREO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "output", "stereo"))

IMG_A = os.path.join(STEREO_DIR, "A003_stitched.tif")
IMG_B = os.path.join(STEREO_DIR, "F003_stitched.tif")
CAM_A = os.path.join(STEREO_DIR, "A003_stitched.tsai")
CAM_B = os.path.join(STEREO_DIR, "F003_stitched.tsai")
DEM = os.path.join(STEREO_DIR, "dem_wgs84.tif")

OUT_DIR = os.path.join(STEREO_DIR, "ba_roma")
MATCH_DIR = os.path.join(OUT_DIR, "matches")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from preprocess.asp import find_asp_tool


def read_band_window(path, col0, row0, width, height):
    """Read a window from band 1 of a raster."""
    import rasterio
    with rasterio.open(path) as src:
        window = rasterio.windows.Window(col0, row0, width, height)
        return src.read(1, window=window).astype(np.float32)


def clahe_u8(arr):
    valid = arr[arr > 0]
    if valid.size < 100:
        return np.zeros_like(arr, dtype=np.uint8)
    lo, hi = np.percentile(valid, [1, 99])
    if hi <= lo:
        hi = lo + 1
    stretched = np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(stretched)


def to_roma_tensor(img_u8, device):
    import torch
    rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    h, w = rgb.shape[:2]
    th = max(448, (h // 14) * 14)
    tw = max(448, (w // 14) * 14)
    rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t[None].to(device), (th, tw), (h, w)


# ---------------------------------------------------------------------------
# Step 1: Create RPC models from OpticalBar cameras
# ---------------------------------------------------------------------------

def create_rpc_models():
    """Convert OpticalBar .tsai cameras to RPC .xml via cam2rpc."""
    cam2rpc = find_asp_tool("cam2rpc")
    if cam2rpc is None:
        print("ERROR: cam2rpc not found")
        sys.exit(1)

    rpcs = {}
    for label, img, cam in [("A003", IMG_A, CAM_A), ("F003", IMG_B, CAM_B)]:
        rpc_path = os.path.join(OUT_DIR, f"{label}_stitched.xml")
        if os.path.exists(rpc_path):
            print(f"  RPC exists: {rpc_path}")
            rpcs[label] = rpc_path
            continue

        cmd = [
            cam2rpc, img, cam, rpc_path,
            "-t", "opticalbar",
            "--dem-file", DEM,
            "--num-samples", "50",
        ]
        print(f"  Creating RPC for {label}...")
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  cam2rpc failed for {label}: {result.stderr[:300]}")
            return None
        print(f"  {label} RPC created in {time.time()-t0:.1f}s")
        rpcs[label] = rpc_path

    return rpcs


# ---------------------------------------------------------------------------
# Step 2: Mapproject both images
# ---------------------------------------------------------------------------

def mapproject_images():
    """Mapproject both images at ~20m resolution for matching."""
    mapproj = find_asp_tool("mapproject")
    if mapproj is None:
        print("ERROR: mapproject not found")
        sys.exit(1)

    orthos = {}
    for label, img, cam in [("A003", IMG_A, CAM_A), ("F003", IMG_B, CAM_B)]:
        ortho_path = os.path.join(OUT_DIR, f"{label}_ortho_coarse.tif")
        if os.path.exists(ortho_path):
            print(f"  Ortho exists: {ortho_path}")
            orthos[label] = ortho_path
            continue

        cmd = [
            mapproj, DEM, img, cam, ortho_path,
            "-t", "opticalbar",
            "--t_srs", "EPSG:4326",
            "--tr", "0.0002",  # ~20m at equator
        ]
        print(f"  Mapprojecting {label} at ~20m...")
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            print(f"  mapproject failed for {label}: {result.stderr[:300]}")
            return None
        print(f"  {label} mapprojected in {time.time()-t0:.1f}s")
        orthos[label] = ortho_path

    return orthos


# ---------------------------------------------------------------------------
# Step 3: Match mapprojected images with RoMa
# ---------------------------------------------------------------------------

def match_ortho_images(ortho_a, ortho_b, rpc_a_path, rpc_b_path,
                       match_output, max_matches=5000):
    """Match mapprojected images with RoMa, unproject to raw pixel coords."""
    import rasterio
    import torch
    from align.models import ModelCache, get_torch_device
    from preprocess.experimental.match_ip import write_asp_match_file

    # Read ortho metadata (don't load full arrays — read tiles lazily)
    with rasterio.open(ortho_a) as src_a:
        tf_a = src_a.transform
        w_a, h_a = src_a.width, src_a.height

    with rasterio.open(ortho_b) as src_b:
        tf_b = src_b.transform
        w_b, h_b = src_b.width, src_b.height

    print(f"  Ortho A: {w_a}x{h_a}, Ortho B: {w_b}x{h_b}")

    # Load RPC models for inverse projection (geo → raw pixel)
    rpc_a = _parse_rpc_xml(rpc_a_path)
    rpc_b = _parse_rpc_xml(rpc_b_path)
    if rpc_a is None or rpc_b is None:
        print("  ERROR: Failed to parse RPC models")
        return None

    # Tiled RoMa matching on ortho images
    # Force CPU — MPS causes 38GB memory spikes on these images
    device = "cpu"
    cache = ModelCache(device)
    model = cache.roma

    # Find overlap region in geographic space
    with rasterio.open(ortho_a) as sa, rasterio.open(ortho_b) as sb:
        bounds_a = sa.bounds
        bounds_b = sb.bounds
    west = max(bounds_a.left, bounds_b.left)
    south = max(bounds_a.bottom, bounds_b.bottom)
    east = min(bounds_a.right, bounds_b.right)
    north = min(bounds_a.top, bounds_b.top)
    print(f"  Overlap: {west:.3f},{south:.3f} → {east:.3f},{north:.3f}")

    # Crop both to overlap, resize to manageable dimensions for tiled matching
    tile_size = 1024
    tile_overlap = 512
    step = tile_size - tile_overlap

    # Compute overlap windows in each ortho
    def geo_to_pixel(transform, lon, lat):
        col = (lon - transform.c) / transform.a
        row = (lat - transform.f) / transform.e
        return int(round(col)), int(round(row))

    c0_a, r0_a = geo_to_pixel(tf_a, west, north)
    c1_a, r1_a = geo_to_pixel(tf_a, east, south)
    c0_b, r0_b = geo_to_pixel(tf_b, west, north)
    c1_b, r1_b = geo_to_pixel(tf_b, east, south)

    # Clamp
    c0_a, r0_a = max(0, c0_a), max(0, r0_a)
    c1_a, r1_a = min(w_a, c1_a), min(h_a, r1_a)
    c0_b, r0_b = max(0, c0_b), max(0, r0_b)
    c1_b, r1_b = min(w_b, c1_b), min(h_b, r1_b)

    ow_a, oh_a = c1_a - c0_a, r1_a - r0_a
    ow_b, oh_b = c1_b - c0_b, r1_b - r0_b
    # Use common overlap dimensions for tile grid
    oh = min(oh_a, oh_b)
    ow = min(ow_a, ow_b)
    print(f"  Overlap: A=[{c0_a}:{c1_a},{r0_a}:{r1_a}] ({ow_a}x{oh_a}), "
          f"B=[{c0_b}:{c1_b},{r0_b}:{r1_b}] ({ow_b}x{oh_b}), common={ow}x{oh}")

    all_geo_a, all_geo_b, all_conf = [], [], []
    roma_size = 640
    num_corresp = 600

    def _collect(correspondences, items):
        for i, it in enumerate(items):
            try:
                preds_i = {k: v[i:i+1] if v is not None else None
                           for k, v in correspondences.items()}
                m, c, _, _ = model.sample(preds_i, num_corresp=num_corresp)
                m, c = m.cpu().numpy(), c.cpu().numpy()
            except Exception:
                continue
            mask = c > 0.55
            m, c = m[mask], c[mask]
            if len(m) == 0:
                continue
            h, w = it["h"], it["w"]
            kp_a = (m[:, :2] + 1) / 2 * np.array([w, h])
            kp_b = (m[:, 2:] + 1) / 2 * np.array([w, h])
            for k in range(len(kp_a)):
                # Ortho overlap pixel → ortho full pixel → geographic
                oa_col = c0_a + (it["c0"] + kp_a[k, 0]) * ((c1_a - c0_a) / ow)
                oa_row = r0_a + (it["r0"] + kp_a[k, 1]) * ((r1_a - r0_a) / oh)
                ob_col = c0_b + (it["c0"] + kp_b[k, 0]) * ((c1_b - c0_b) / ow)
                ob_row = r0_b + (it["r0"] + kp_b[k, 1]) * ((r1_b - r0_b) / oh)
                lon_a, lat_a = rasterio.transform.xy(tf_a, oa_row, oa_col)
                lon_b, lat_b = rasterio.transform.xy(tf_b, ob_row, ob_col)
                all_geo_a.append([lon_a, lat_a, oa_col, oa_row])
                all_geo_b.append([lon_b, lat_b, ob_col, ob_row])
                all_conf.append(float(c[k]))

    def _run_batch(batch_data):
        ref_t, off_t = [], []
        for it in batch_data:
            rc = cv2.cvtColor(it["ref"], cv2.COLOR_GRAY2RGB)
            oc = cv2.cvtColor(it["off"], cv2.COLOR_GRAY2RGB)
            rc = cv2.resize(rc, (roma_size, roma_size), interpolation=cv2.INTER_AREA)
            oc = cv2.resize(oc, (roma_size, roma_size), interpolation=cv2.INTER_AREA)
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
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return None

    def _process_batch(bd):
        if not bd: return
        queue = [bd]
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

    # Tile and match — read tiles lazily from disk
    batch_inputs = []
    tile_count = 0
    t0 = time.time()
    for r0 in range(0, oh - tile_size // 2, step):
        for c0 in range(0, ow - tile_size // 2, step):
            r1 = min(r0 + tile_size, oh)
            c1 = min(c0 + tile_size, ow)
            if r1 - r0 < tile_size // 2 or c1 - c0 < tile_size // 2:
                continue

            # Read tiles from disk (windowed) instead of slicing arrays
            tw, th = c1 - c0, r1 - r0
            # Map overlap-local coords to full ortho coords
            ac, ar = c0_a + int(c0 * ow_a / ow), r0_a + int(r0 * oh_a / oh)
            bc, br = c0_b + int(c0 * ow_b / ow), r0_b + int(r0 * oh_b / oh)
            atw = min(int(tw * ow_a / ow), w_a - ac)
            ath = min(int(th * oh_a / oh), h_a - ar)
            btw = min(int(tw * ow_b / ow), w_b - bc)
            bth = min(int(th * oh_b / oh), h_b - br)
            if atw < tile_size // 2 or ath < tile_size // 2:
                continue
            if btw < tile_size // 2 or bth < tile_size // 2:
                continue

            at_raw = read_band_window(ortho_a, ac, ar, atw, ath)
            bt_raw = read_band_window(ortho_b, bc, br, btw, bth)

            # Resize to common tile dims
            min_th, min_tw = min(ath, bth), min(atw, btw)
            at = clahe_u8(cv2.resize(at_raw, (min_tw, min_th),
                                     interpolation=cv2.INTER_AREA)
                          if at_raw.shape != (min_th, min_tw) else at_raw)
            bt = clahe_u8(cv2.resize(bt_raw, (min_tw, min_th),
                                     interpolation=cv2.INTER_AREA)
                          if bt_raw.shape != (min_th, min_tw) else bt_raw)

            if np.mean(at > 0) < 0.2 or np.mean(bt > 0) < 0.2:
                continue
            batch_inputs.append({
                "ref": at, "off": bt,
                "r0": r0, "c0": c0,
                "h": min_th, "w": min_tw,
            })
            if len(batch_inputs) >= 4:
                _process_batch(batch_inputs)
                tile_count += len(batch_inputs)
                batch_inputs = []
                if tile_count % 50 == 0:
                    print(f"    Tiles: {tile_count}, matches: {len(all_conf)}, "
                          f"elapsed: {time.time()-t0:.0f}s")
                if len(all_conf) >= max_matches:
                    break
        if len(all_conf) >= max_matches:
            break
    if batch_inputs:
        _process_batch(batch_inputs)
        tile_count += len(batch_inputs)

    print(f"  Tiled matching: {tile_count} tiles, {len(all_conf)} raw matches, "
          f"{time.time()-t0:.1f}s")

    cache.close()

    if len(all_conf) < 10:
        print("  ERROR: Too few matches")
        return None

    geo_a = np.array(all_geo_a, dtype=np.float64)
    geo_b = np.array(all_geo_b, dtype=np.float64)
    conf = np.array(all_conf, dtype=np.float32)

    # RANSAC filter on geographic coordinates
    F, inliers = cv2.findFundamentalMat(
        geo_a[:, :2].astype(np.float32).reshape(-1, 1, 2),
        geo_b[:, :2].astype(np.float32).reshape(-1, 1, 2),
        cv2.FM_RANSAC, ransacReprojThreshold=0.001)
    if inliers is not None and inliers.sum() >= 10:
        mask = inliers.ravel().astype(bool)
        geo_a, geo_b, conf = geo_a[mask], geo_b[mask], conf[mask]
        print(f"  After RANSAC: {len(conf)} inliers")

    # Unproject geographic → raw pixel via RPC models
    print(f"  Unprojecting {len(conf)} matches via RPC...")
    lons_a = geo_a[:, 0].astype(np.float64)
    lats_a = geo_a[:, 1].astype(np.float64)
    lons_b = geo_b[:, 0].astype(np.float64)
    lats_b = geo_b[:, 1].astype(np.float64)

    cols_a, rows_a = _rpc_ground_to_pixel_batch(rpc_a, lons_a, lats_a)
    cols_b, rows_b = _rpc_ground_to_pixel_batch(rpc_b, lons_b, lats_b)

    raw_pts_a = np.column_stack([cols_a, rows_a]).astype(np.float32)
    raw_pts_b = np.column_stack([cols_b, rows_b]).astype(np.float32)

    # Filter: keep only points that land within image bounds
    from functools import reduce
    w_raw_a, h_raw_a = 249295, 25069  # from .tsai image_size
    w_raw_b, h_raw_b = 247017, 24753
    keep = ((raw_pts_a[:, 0] >= 0) & (raw_pts_a[:, 0] < w_raw_a) &
            (raw_pts_a[:, 1] >= 0) & (raw_pts_a[:, 1] < h_raw_a) &
            (raw_pts_b[:, 0] >= 0) & (raw_pts_b[:, 0] < w_raw_b) &
            (raw_pts_b[:, 1] >= 0) & (raw_pts_b[:, 1] < h_raw_b))
    raw_pts_a = raw_pts_a[keep]
    raw_pts_b = raw_pts_b[keep]
    conf = conf[keep]
    print(f"  Unprojected: {len(conf)} matches within image bounds")

    if len(conf) < 10:
        print("  ERROR: Too few matches after unprojection")
        ds_a = ds_b = None
        return None

    # Spatial de-dup (200px cells on raw A coords)
    cells = {}
    for i in range(len(raw_pts_a)):
        key = (int(raw_pts_a[i, 0] // 200), int(raw_pts_a[i, 1] // 200))
        if key not in cells or conf[i] > conf[cells[key]]:
            cells[key] = i
    idx = sorted(cells.values())
    raw_pts_a = raw_pts_a[idx]
    raw_pts_b = raw_pts_b[idx]
    conf = conf[idx]
    print(f"  After de-dup: {len(conf)} matches")

    if len(conf) > max_matches:
        order = np.argsort(-conf)[:max_matches]
        raw_pts_a, raw_pts_b, conf = raw_pts_a[order], raw_pts_b[order], conf[order]

    write_asp_match_file(raw_pts_a, raw_pts_b, match_output)
    print(f"  Wrote {len(raw_pts_a)} matches → {match_output}")

    ds_a = ds_b = None
    return match_output


def _parse_rpc_xml(path):
    """Parse ASP cam2rpc RPB XML into RPC coefficients dict."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    root = tree.getroot()
    img = root.find(".//IMAGE")
    if img is None:
        return None

    def _get(tag):
        el = img.find(tag)
        return float(el.text) if el is not None else 0.0

    def _get_coeffs(tag):
        el = img.find(f".//{tag}/{tag.replace('List', '').replace('COEF', 'COEF')}")
        if el is None:
            # Try alternate nesting
            el = img.find(f".//{tag}")
            if el is None:
                return [0.0] * 20
            inner = el.find(tag.replace("List", ""))
            if inner is not None:
                el = inner
        return [float(x) for x in el.text.strip().split()]

    return {
        "LINE_OFF": _get("LINEOFFSET"),
        "SAMP_OFF": _get("SAMPOFFSET"),
        "LAT_OFF": _get("LATOFFSET"),
        "LON_OFF": _get("LONGOFFSET"),
        "HEIGHT_OFF": _get("HEIGHTOFFSET"),
        "LINE_SCALE": _get("LINESCALE"),
        "SAMP_SCALE": _get("SAMPSCALE"),
        "LAT_SCALE": _get("LATSCALE"),
        "LON_SCALE": _get("LONGSCALE"),
        "HEIGHT_SCALE": _get("HEIGHTSCALE"),
        "LINE_NUM": _get_coeffs("LINENUMCOEFList"),
        "LINE_DEN": _get_coeffs("LINEDENCOEFList"),
        "SAMP_NUM": _get_coeffs("SAMPNUMCOEFList"),
        "SAMP_DEN": _get_coeffs("SAMPDENCOEFList"),
    }


def _rpc_ground_to_pixel(rpc, lon, lat, height=0.0):
    """Evaluate RPC: (lon, lat, height) → (sample, line) i.e. (col, row)."""
    P = (lat - rpc["LAT_OFF"]) / rpc["LAT_SCALE"]
    L = (lon - rpc["LON_OFF"]) / rpc["LON_SCALE"]
    H = (height - rpc["HEIGHT_OFF"]) / rpc["HEIGHT_SCALE"]

    def _poly(c, P, L, H):
        return (c[0] + c[1]*L + c[2]*P + c[3]*H +
                c[4]*L*P + c[5]*L*H + c[6]*P*H + c[7]*L*L +
                c[8]*P*P + c[9]*H*H + c[10]*L*P*H +
                c[11]*L*L*L + c[12]*L*P*P + c[13]*L*H*H +
                c[14]*L*L*P + c[15]*P*P*P + c[16]*P*H*H +
                c[17]*L*L*H + c[18]*P*P*H + c[19]*H*H*H)

    line_n = _poly(rpc["LINE_NUM"], P, L, H)
    line_d = _poly(rpc["LINE_DEN"], P, L, H)
    samp_n = _poly(rpc["SAMP_NUM"], P, L, H)
    samp_d = _poly(rpc["SAMP_DEN"], P, L, H)

    line = line_n / line_d * rpc["LINE_SCALE"] + rpc["LINE_OFF"]
    samp = samp_n / samp_d * rpc["SAMP_SCALE"] + rpc["SAMP_OFF"]
    return samp, line  # (col, row)


def _rpc_ground_to_pixel_batch(rpc, lons, lats, heights=None):
    """Vectorised RPC evaluation for arrays of coordinates."""
    if heights is None:
        heights = np.zeros_like(lons)
    P = (lats - rpc["LAT_OFF"]) / rpc["LAT_SCALE"]
    L = (lons - rpc["LON_OFF"]) / rpc["LON_SCALE"]
    H = (heights - rpc["HEIGHT_OFF"]) / rpc["HEIGHT_SCALE"]

    c = np.array  # shorthand
    def _poly_v(coeffs, P, L, H):
        co = coeffs
        return (co[0] + co[1]*L + co[2]*P + co[3]*H +
                co[4]*L*P + co[5]*L*H + co[6]*P*H + co[7]*L*L +
                co[8]*P*P + co[9]*H*H + co[10]*L*P*H +
                co[11]*L*L*L + co[12]*L*P*P + co[13]*L*H*H +
                co[14]*L*L*P + co[15]*P*P*P + co[16]*P*H*H +
                co[17]*L*L*H + co[18]*P*P*H + co[19]*H*H*H)

    ln = _poly_v(rpc["LINE_NUM"], P, L, H)
    ld = _poly_v(rpc["LINE_DEN"], P, L, H)
    sn = _poly_v(rpc["SAMP_NUM"], P, L, H)
    sd = _poly_v(rpc["SAMP_DEN"], P, L, H)

    lines = ln / ld * rpc["LINE_SCALE"] + rpc["LINE_OFF"]
    samps = sn / sd * rpc["SAMP_SCALE"] + rpc["SAMP_OFF"]
    return samps, lines  # (cols, rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    for p in [IMG_A, IMG_B, CAM_A, CAM_B, DEM]:
        if not os.path.exists(p):
            print(f"ERROR: Missing {p}")
            sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MATCH_DIR, exist_ok=True)

    cam_a_copy = os.path.join(OUT_DIR, "A003_stitched.tsai")
    cam_b_copy = os.path.join(OUT_DIR, "F003_stitched.tsai")
    shutil.copy2(CAM_A, cam_a_copy)
    shutil.copy2(CAM_B, cam_b_copy)

    # === Step 1: Create RPC models ===
    print("\n" + "=" * 60)
    print("Step 1: Create RPC models from OpticalBar cameras")
    print("=" * 60)
    rpcs = create_rpc_models()
    if rpcs is None:
        sys.exit(1)

    # === Step 2: Mapproject at ~20m ===
    print("\n" + "=" * 60)
    print("Step 2: Mapproject both images")
    print("=" * 60)
    orthos = mapproject_images()
    if orthos is None:
        sys.exit(1)

    # === Step 3: Match ortho images + unproject to raw pixels ===
    print("\n" + "=" * 60)
    print("Step 3: RoMa matching in geographic space")
    print("=" * 60)

    match_prefix = os.path.join(MATCH_DIR, "run")
    match_file = f"{match_prefix}-A003_stitched__F003_stitched.match"

    result = match_ortho_images(
        orthos["A003"], orthos["F003"],
        rpcs["A003"], rpcs["F003"],
        match_file)
    if result is None:
        print("ERROR: Matching failed")
        sys.exit(1)

    # === Step 4: Bundle adjust ===
    print("\n" + "=" * 60)
    print("Step 4: Bundle adjust (RoMa matches)")
    print("=" * 60)

    ba_tool = find_asp_tool("bundle_adjust")
    ba_prefix = os.path.join(OUT_DIR, "run")
    cmd = [
        ba_tool, IMG_A, IMG_B, cam_a_copy, cam_b_copy,
        "-t", "opticalbar",
        "--datum", "WGS84",
        "--heights-from-dem", DEM,
        "--heights-from-dem-uncertainty", "50",
        "--inline-adjustments",
        "--camera-weight", "0",
        "--match-files-prefix", match_prefix,
        "--remove-outliers-params", "75.0 3.0 150 200",
        "--num-passes", "1",
        "-o", ba_prefix,
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0

    log_path = os.path.join(OUT_DIR, "bundle_adjust.log")
    with open(log_path, "w") as f:
        f.write(f"COMMAND: {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}\n\nRETURN CODE: {result.returncode}\n"
                f"ELAPSED: {elapsed:.1f}s\n")

    if result.returncode != 0:
        print(f"  Bundle adjust failed (exit {result.returncode}, {elapsed:.1f}s)")
        for line in result.stdout.strip().split("\n")[-15:]:
            print(f"    {line}")
        sys.exit(1)

    print(f"  Bundle adjust completed in {elapsed:.1f}s")

    # === Step 5: Stereo ===
    print("\n" + "=" * 60)
    print("Step 5: Stereo correlation")
    print("=" * 60)

    stereo_tool = find_asp_tool("parallel_stereo") or find_asp_tool("stereo")
    if stereo_tool is None:
        print("WARNING: stereo not found, stopping")
        return

    stereo_prefix = os.path.join(OUT_DIR, "stereo", "run")
    os.makedirs(os.path.join(OUT_DIR, "stereo"), exist_ok=True)

    cmd = [
        stereo_tool, IMG_A, IMG_B, cam_a_copy, cam_b_copy,
        stereo_prefix, "-t", "opticalbar", "--datum", "WGS84",
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    with open(os.path.join(OUT_DIR, "stereo.log"), "w") as f:
        f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n")

    if result.returncode != 0:
        print(f"  Stereo failed ({elapsed:.1f}s)")
        return
    print(f"  Stereo completed in {elapsed:.1f}s")

    # === Step 6: DEM + Ortho ===
    p2d = find_asp_tool("point2dem")
    pc_file = f"{stereo_prefix}-PC.tif"
    if p2d and os.path.exists(pc_file):
        print("\n  Running point2dem...")
        cmd = [p2d, pc_file, "--datum", "WGS84",
               "-o", os.path.join(OUT_DIR, "dem_roma")]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print(f"  point2dem exit: {result.returncode}")

    print(f"\nComplete! Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
