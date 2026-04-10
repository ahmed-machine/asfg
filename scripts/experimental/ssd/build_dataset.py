"""
Generate the Simple Self-Distillation (SSD) synthetic training dataset.
Fetches georectified KH imagery from CORONA Atlas WMS and matched Sentinel-2 L2A basemaps.
Applies random synthetic affine perturbations to the historical imagery to simulate the 
unaligned state that the pipeline encounters in production.
"""

import argparse
import json
import math
import os
import random
import sys
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from preprocess.georef import fetch_sentinel2_reference
from align.image import to_u8_percentile
from dataclasses import dataclass

@dataclass
class Box:
    west: float
    south: float
    east: float
    north: float

WMS_CAPABILITIES_URL = "https://geoserve.cast.uark.edu/geoserver/wms?request=GetCapabilities"
WMS_MAP_URL = "https://geoserve.cast.uark.edu/geoserver/wms"


def get_corona_layers() -> List[Dict]:
    """Parse WMS capabilities to get list of corona layers and their bounding boxes."""
    cache_file = "corona_layers_cache.json"
    if os.path.exists(cache_file):
        print("Loading WMS capabilities from cache...")
        with open(cache_file, "r") as f:
            return json.load(f)

    print("Fetching WMS capabilities from CORONA Atlas...")
    req = urllib.request.Request(WMS_CAPABILITIES_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as response:
        content = response.read()

    root = ET.fromstring(content)
    
    # Handle namespaces
    layers = root.findall('.//Layer')
    if not layers:
        layers = root.findall('.//{http://www.opengis.net/wms}Layer')

    corona_layers = []
    for layer in layers:
        name_elem = layer.find('Name')
        if name_elem is None:
            name_elem = layer.find('{http://www.opengis.net/wms}Name')
            
        if name_elem is not None and name_elem.text and name_elem.text.startswith('corona:'):
            bbox_elem = layer.find('EX_GeographicBoundingBox')
            if bbox_elem is None:
                bbox_elem = layer.find('{http://www.opengis.net/wms}EX_GeographicBoundingBox')
                
            if bbox_elem is not None:
                try:
                    def _get(tag):
                        el = bbox_elem.find(tag)
                        if el is None:
                            el = bbox_elem.find(f"{{http://www.opengis.net/wms}}{tag}")
                        return float(el.text)
                        
                    bbox = {
                        'west': _get('westBoundLongitude'),
                        'east': _get('eastBoundLongitude'),
                        'south': _get('southBoundLatitude'),
                        'north': _get('northBoundLatitude')
                    }
                    corona_layers.append({
                        'name': name_elem.text,
                        'bbox': bbox
                    })
                except Exception as e:
                    pass
                    
    print(f"Found {len(corona_layers)} valid CORONA layers. Caching...")
    with open(cache_file, "w") as f:
        json.dump(corona_layers, f)
    return corona_layers


def fetch_wms_tile(layer_name: str, bbox: Dict, width: int = 1024, height: int = 1024) -> Optional[np.ndarray]:
    """Fetch a specific tile from the CORONA WMS service."""
    # BBOX format: minx,miny,maxx,maxy (lon, lat)
    bbox_str = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"
    
    params = {
        'service': 'WMS',
        'version': '1.1.1',
        'request': 'GetMap',
        'layers': layer_name,
        'styles': '',
        'bbox': bbox_str,
        'width': str(width),
        'height': str(height),
        'srs': 'EPSG:4326',
        'format': 'image/jpeg'
    }
    
    query_string = urllib.parse.urlencode(params)
    url = f"{WMS_MAP_URL}?{query_string}"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            
            # Check if image is mostly empty/white/black (failed or empty tile)
            if img is None:
                return None
                
            mean_val = np.mean(img)
            if mean_val < 5 or mean_val > 250:
                return None
                
            # Check for large blocks of nodata (usually solid white or black)
            h, w = img.shape
            border_mean = (np.mean(img[0,:]) + np.mean(img[-1,:]) + 
                           np.mean(img[:,0]) + np.mean(img[:,-1])) / 4.0
            if border_mean > 250 or border_mean < 5:
                 # Check if the whole image is just borders
                 if np.std(img) < 10:
                     return None
                     
            return img
    except Exception as e:
        print(f"  Error fetching WMS tile: {e}")
        return None


def _random_tps_warp(h, w, n_points=4, strength=0.05):
    """Generate a smooth non-linear warp field using thin-plate-spline-like control points.

    Places n_points x n_points control points on a regular grid, displaces them
    randomly, then interpolates a dense displacement field via cv2.remap.
    Simulates terrain parallax, film deformation, and panoramic distortion residuals.
    """
    # Regular grid of control points in [0, 1]
    gx = np.linspace(0, 1, n_points + 2)[1:-1]  # skip edges to avoid border artifacts
    gy = np.linspace(0, 1, n_points + 2)[1:-1]
    src_pts = np.array([[x, y] for y in gy for x in gx], dtype=np.float32)
    # Random displacements scaled by strength
    dx = np.random.uniform(-strength, strength, len(src_pts)).astype(np.float32)
    dy = np.random.uniform(-strength, strength, len(src_pts)).astype(np.float32)
    dst_pts = src_pts + np.stack([dx, dy], axis=1)

    # Build dense displacement field via interpolation
    from scipy.interpolate import RBFInterpolator
    # RBF from src_pts -> displacement
    rbf_x = RBFInterpolator(src_pts, dx * w, kernel='thin_plate_spline', smoothing=0.1)
    rbf_y = RBFInterpolator(src_pts, dy * h, kernel='thin_plate_spline', smoothing=0.1)

    yy, xx = np.mgrid[0:h, 0:w]
    query = np.stack([xx.ravel() / w, yy.ravel() / h], axis=1).astype(np.float32)
    map_x = (xx + rbf_x(query).reshape(h, w)).astype(np.float32)
    map_y = (yy + rbf_y(query).reshape(h, w)).astype(np.float32)
    return map_x, map_y


def generate_synthetic_perturbation(shape: Tuple[int, int], mode: str = "auto"):
    """
    Generate a synthetic perturbation simulating unaligned imagery.

    Modes:
        "affine" — rotation + scale + translation only (fast, simple)
        "complex" — affine + non-linear TPS warp + optional radial distortion
        "auto" — 50% affine, 50% complex

    Returns a 2x3 affine matrix (mode="affine") or a (map_x, map_y) remap pair.
    """
    h, w = shape
    center = (w / 2, h / 2)

    if mode == "auto":
        mode = "complex" if random.random() < 0.5 else "affine"

    # Base affine parameters
    angle = random.uniform(-15.0, 15.0)
    scale = random.uniform(0.85, 1.15)
    tx = random.uniform(-w * 0.15, w * 0.15)
    ty = random.uniform(-h * 0.15, h * 0.15)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    if mode == "affine":
        return M

    # Complex mode: apply affine first, then add non-linear distortion
    # Build the affine remap
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.stack([xx.ravel(), yy.ravel(), np.ones(h * w)], axis=0)  # (3, N)
    warped = M @ coords  # (2, N)
    map_x = warped[0].reshape(h, w).astype(np.float32)
    map_y = warped[1].reshape(h, w).astype(np.float32)

    # Add TPS non-linear warp (simulates film deformation, terrain parallax)
    tps_strength = random.uniform(0.01, 0.06)
    n_pts = random.choice([3, 4, 5])
    tps_map_x, tps_map_y = _random_tps_warp(h, w, n_points=n_pts, strength=tps_strength)
    # Compose: sample the affine-warped coords at TPS-displaced locations
    map_x = cv2.remap(map_x, tps_map_x, tps_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    map_y = cv2.remap(map_y, tps_map_x, tps_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Optional radial (barrel/pincushion) distortion (30% chance)
    if random.random() < 0.3:
        k1 = random.uniform(-0.15, 0.15)
        cx, cy = w / 2, h / 2
        r2 = ((map_x - cx) / w) ** 2 + ((map_y - cy) / h) ** 2
        radial = 1 + k1 * r2
        map_x = cx + (map_x - cx) * radial
        map_y = cy + (map_y - cy) * radial

    return (map_x.astype(np.float32), map_y.astype(np.float32))


def process_layer(layer: Dict, output_dir: Path, idx: int, out_shape=(1024, 1024)) -> bool:
    """Process a single layer to extract a paired training sample."""
    name = layer['name']
    l_bbox = layer['bbox']
    
    # We want a random crop from within the layer's bounding box
    # to avoid empty edges. Let's take a crop that is 10% the size of the box
    w_deg = l_bbox['east'] - l_bbox['west']
    h_deg = l_bbox['north'] - l_bbox['south']
    
    # Only process reasonably sized layers to avoid extreme distortions
    if w_deg <= 0 or h_deg <= 0 or w_deg > 5.0 or h_deg > 5.0:
        return False
        
    crop_w = w_deg * 0.2
    crop_h = h_deg * 0.2
    
    # Random center within the safe interior
    safe_west = l_bbox['west'] + crop_w
    safe_east = l_bbox['east'] - crop_w
    safe_south = l_bbox['south'] + crop_h
    safe_north = l_bbox['north'] - crop_h
    
    if safe_west >= safe_east or safe_south >= safe_north:
        return False
        
    cx = random.uniform(safe_west, safe_east)
    cy = random.uniform(safe_south, safe_north)
    
    # 0.05 degrees is roughly 5km
    span = 0.05
    tile_bbox = {
        'west': cx - span,
        'east': cx + span,
        'south': cy - span,
        'north': cy + span
    }
    
    # 1. Fetch Historical (KH) Image
    img_kh = fetch_wms_tile(name, tile_bbox, width=out_shape[0], height=out_shape[1])
    if img_kh is None:
        return False
        
    # 2. Fetch Modern Reference (Sentinel-2)
    # The georef module expects a tuple bbox (west, south, east, north)
    bbox_tuple = (tile_bbox['west'], tile_bbox['south'], tile_bbox['east'], tile_bbox['north'])
    try:
        # fetch_sentinel2_reference returns a path to a downloaded VRT/TIF
        # We need to render it to an array.
        ref_path = fetch_sentinel2_reference(
            bbox=bbox_tuple, 
            output_path=str(output_dir / f"temp_s2_{idx}.tif")
        )
        if not ref_path or not os.path.exists(ref_path):
            return False
            
        # Read the reference, convert to grayscale, resize to match exactly
        ref_bgr = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED)
        if ref_bgr is None:
            return False
            
        if len(ref_bgr.shape) == 3:
            ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = ref_bgr
            
        # Normalize the 16-bit satellite imagery to 8-bit using 1-99 percentile
        if ref_gray.dtype != np.uint8:
            ref_gray = to_u8_percentile(ref_gray)
            
        img_s2 = cv2.resize(ref_gray, out_shape)
        
        # Cleanup temporary S2 file
        os.remove(ref_path)
        
    except Exception as e:
        print(f"  Error fetching Sentinel-2 for {name}: {e}")
        return False
        
    # 3. Apply Synthetic Perturbation to the KH image
    perturbation = generate_synthetic_perturbation(out_shape)
    if isinstance(perturbation, tuple):
        # Complex mode: (map_x, map_y) remap pair
        map_x, map_y = perturbation
        img_kh_perturbed = cv2.remap(
            img_kh, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        perturbation_meta = {"mode": "complex"}
    else:
        # Affine mode: 2x3 matrix
        M_affine = perturbation
        img_kh_perturbed = cv2.warpAffine(
            img_kh,
            M_affine,
            out_shape,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        perturbation_meta = {"mode": "affine", "affine_matrix": M_affine.tolist()}

    # 4. Save the pair and metadata
    prefix = f"pair_{idx:05d}"
    cv2.imwrite(str(output_dir / f"{prefix}_ref.png"), img_s2)
    cv2.imwrite(str(output_dir / f"{prefix}_src_perturbed.png"), img_kh_perturbed)
    cv2.imwrite(str(output_dir / f"{prefix}_src_clean.png"), img_kh)

    meta = {
        'layer': name,
        'bbox': tile_bbox,
        'perturbation': perturbation_meta,
    }
    with open(output_dir / f"{prefix}_meta.json", 'w') as f:
        json.dump(meta, f)
        
    print(f"[{idx}] Successfully generated training pair from {name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build SSD training dataset from CORONA WMS.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--count", "-c", type=int, default=1000, help="Number of pairs to generate")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of concurrent workers")
    args = parser.parse_args()
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    layers = get_corona_layers()
    if not layers:
        print("Failed to fetch layers. Exiting.")
        sys.exit(1)
        
    random.shuffle(layers)
    
    # Find highest existing pair index to resume
    import glob
    import re
    existing = glob.glob(str(out_dir / "pair_*_ref.png"))
    success_count = 0
    current_idx = 0
    if existing:
        indices = [int(re.search(r'pair_(\d+)_', f).group(1)) for f in existing if re.search(r'pair_(\d+)_', f)]
        if indices:
            success_count = len(indices)
            current_idx = max(indices) + 1
            print(f"Found {success_count} existing pairs. Resuming from index {current_idx}...")
            
    # We want to stop when we reach args.count TOTAL pairs
    target_count = args.count
    if success_count >= target_count:
        print(f"Already have {success_count} pairs (target is {target_count}). Exiting.")
        return

    futures = []
    
    print(f"Starting generation of {target_count - success_count} new synthetic pairs using {args.workers} workers...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # We use an infinite loop with random.choice so we can sample multiple 
        # random crops per layer, allowing datasets far larger than the 2,493 layer count.
        while success_count < target_count:
            # Only queue up to 2x workers to avoid API rate limits and memory bloat
            if len(futures) < args.workers * 2:
                random_layer = random.choice(layers)
                # Pass current_idx as the index for the next pair
                future = executor.submit(process_layer, random_layer, out_dir, current_idx)
                current_idx += 1
                futures.append(future)
            else:
                # Wait for at least one to finish
                for f in as_completed(futures):
                    if f.result():
                        success_count += 1
                    futures.remove(f)
                    break
                    
        # Cancel/wait for any remaining
        for f in futures:
            f.cancel()
                
    print(f"Done. Successfully generated {success_count} training pairs in {out_dir}")

if __name__ == "__main__":
    main()
