#!/usr/bin/env python3
"""
Rough-georeference satellite imagery using USGS XML bounding boxes.

Thin CLI entry point — delegates to declass.georef for the actual GDAL work.

For each image:
  1. Parses associated XML file(s) to extract bounding box coordinates
  2. For stitched images (multiple XMLs), computes the union bounding box
  3. Georefs to WGS84 then reprojects to EPSG:3857 via declass.georef

Usage:
    python georef.py                          # process all images in IMAGES dict
    python georef.py --input img.tif --xml a.xml b.xml
"""

import argparse
import os
import sys

from align.metadata_priors import parse_bbox_xml
from declass.georef import georef_with_bbox

# Default image → XML file mapping (legacy Bahrain dataset)
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
    return float(prior.west), float(prior.north), float(prior.east), float(prior.south)


def union_bbox(bboxes):
    """Compute the union bounding box from a list of (west, north, east, south) tuples."""
    west = min(b[0] for b in bboxes)
    north = max(b[1] for b in bboxes)
    east = max(b[2] for b in bboxes)
    south = min(b[3] for b in bboxes)
    return west, north, east, south


def georef_image(base_dir, image_name, xml_files):
    """Georeference a single image using its XML bounding box metadata."""
    print(f"\n{'='*60}")
    print(f"Processing: {image_name}")

    bboxes = []
    for xml_file in xml_files:
        xml_path = os.path.join(base_dir, xml_file)
        bbox = parse_bbox(xml_path)
        print(f"  {xml_file}: W={bbox[0]}, N={bbox[1]}, E={bbox[2]}, S={bbox[3]}")
        bboxes.append(bbox)

    west, north, east, south = union_bbox(bboxes)
    print(f"  Union bbox: W={west}, N={north}, E={east}, S={south}")

    input_path = os.path.join(base_dir, image_name)
    stem = image_name.rsplit(".tif", 1)[0]
    output_path = os.path.join(base_dir, f"{stem}.warped.tif")

    if not os.path.exists(input_path):
        print(f"  WARNING: Input file not found: {input_path}")
        return False

    georef_with_bbox(input_path, output_path, west, north, east, south)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Rough-georeference satellite imagery using USGS XML bounding boxes")
    parser.add_argument("--input", help="Single input TIFF to georeference")
    parser.add_argument("--xml", nargs="+", help="XML metadata file(s) for --input")
    parser.add_argument("--base-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Base directory for image/XML lookup (default: script dir)")
    args = parser.parse_args()

    if args.input:
        if not args.xml:
            print("ERROR: --xml required when using --input")
            sys.exit(1)
        bboxes = [parse_bbox(x) for x in args.xml]
        west, north, east, south = union_bbox(bboxes)
        stem = args.input.rsplit(".tif", 1)[0]
        output_path = f"{stem}.warped.tif"
        georef_with_bbox(args.input, output_path, west, north, east, south)
        return

    base_dir = args.base_dir
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
