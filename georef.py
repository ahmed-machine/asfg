#!/usr/bin/env python3
"""
Rough-georeference DS1 satellite imagery using USGS XML bounding boxes.

For each image:
  1. Parses associated XML file(s) to extract bounding box coordinates
  2. For stitched images (multiple XMLs), computes the union bounding box
  3. Runs gdal_translate to assign WGS84 coordinates via -a_ullr
  4. Runs gdalwarp to reproject to Web Mercator (EPSG:3857) for tile serving

Note: USGS documents ~10-mile positional error for CORONA/LANYARD corner points.
This gives a rough georeference suitable for placing images on a web map.
"""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET

from align.metadata_priors import parse_bbox_xml

# Image -> XML file mapping
IMAGES = {
    "1965-07-21_LANYARD_DS1022-1024DA007-DA008.tif": [
        "DS1022-1024DA007.xml",
        "DS1022-1024DA008.xml",
    ],
    "1968-08-11_LANYARD_DS1104-1057DA023-DA025.tif": [
        "DS1104-1057DA023.xml",
        "DS1104-1057DA024.xml",
        "DS1104-1057DA025.xml",
    ],
    "1968-08-11_LANYARD_DS1104-1057DF016-DF019.tif": [
        "DS1104-1057DF016.xml",
        "DS1104-1057DF017.xml",
        "DS1104-1057DF018.xml",
        "DS1104-1057DF019.xml",
    ],
    "1969-09-27_LANYARD_DS1052-1073DA157.tif": [
        "DS1052-1073DA157.xml",
    ],
    "1976-08-26_KH9_DZB1212-500236L002001.tif": [
        "DZB1212-500236L002001.xml",
    ],
}


def parse_bbox(xml_path):
    """Extract bounding box (west, north, east, south) from a USGS metadata XML."""
    prior = parse_bbox_xml(xml_path)
    if not prior.has_bounds:
        raise ValueError(f"No bounding box found in {xml_path}")
    west_val = prior.west
    north_val = prior.north
    east_val = prior.east
    south_val = prior.south
    assert west_val is not None and north_val is not None
    assert east_val is not None and south_val is not None
    west = float(west_val)
    north = float(north_val)
    east = float(east_val)
    south = float(south_val)
    return west, north, east, south


def union_bbox(bboxes):
    """Compute the union bounding box from a list of (west, north, east, south) tuples."""
    west = min(b[0] for b in bboxes)
    north = max(b[1] for b in bboxes)
    east = max(b[2] for b in bboxes)
    south = min(b[3] for b in bboxes)
    return west, north, east, south


def run_cmd(cmd):
    """Run a shell command, printing it and checking for errors."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout


def georef_image(base_dir, image_name, xml_files):
    """Georeference a single image using its XML bounding box metadata."""
    print(f"\n{'='*60}")
    print(f"Processing: {image_name}")
    print(f"  XML files: {xml_files}")

    # Parse bounding boxes from all XML files
    bboxes = []
    for xml_file in xml_files:
        xml_path = os.path.join(base_dir, xml_file)
        bbox = parse_bbox(xml_path)
        print(f"  {xml_file}: W={bbox[0]}, N={bbox[1]}, E={bbox[2]}, S={bbox[3]}")
        bboxes.append(bbox)

    # Compute union bounding box
    west, north, east, south = union_bbox(bboxes)
    print(f"  Union bbox: W={west}, N={north}, E={east}, S={south}")

    input_path = os.path.join(base_dir, image_name)
    stem = image_name.rsplit(".tif", 1)[0]
    temp_path = os.path.join(base_dir, f"{stem}.temp_georef.tif")
    output_path = os.path.join(base_dir, f"{stem}.warped.tif")

    if not os.path.exists(input_path):
        print(f"  WARNING: Input file not found: {input_path}")
        return False

    # Step 1: Assign WGS84 coordinates with gdal_translate
    print("  Step 1: Assigning WGS84 coordinates...")
    run_cmd([
        "gdal_translate",
        "-a_srs", "EPSG:4326",
        "-a_ullr", str(west), str(north), str(east), str(south),
        input_path,
        temp_path,
    ])

    # Step 2: Reproject to Web Mercator
    print("  Step 2: Reprojecting to EPSG:3857 (Web Mercator)...")
    run_cmd([
        "gdalwarp",
        "-s_srs", "EPSG:4326",
        "-t_srs", "EPSG:3857",
        "-r", "lanczos",
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-co", "TILED=YES",
        temp_path,
        output_path,
    ])

    # Clean up temp file
    os.remove(temp_path)
    print(f"  Removed temp file: {temp_path}")

    # Verify with gdalinfo
    print("  Step 3: Verifying output...")
    info = run_cmd(["gdalinfo", output_path])
    # Print just the key lines
    for line in info.splitlines():
        if any(kw in line for kw in ["Size is", "Coordinate System", "Origin", "Pixel Size", "Upper Left", "Lower Right"]):
            print(f"    {line.strip()}")

    print(f"  Output: {output_path}")
    return True


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Working directory: {base_dir}")

    success = 0
    failed = 0

    for image_name, xml_files in IMAGES.items():
        try:
            if georef_image(base_dir, image_name, xml_files):
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Done. {success} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()
