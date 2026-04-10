"""
Run the frozen RoMa and SEA-RAFT models over the synthetic SSD dataset
to extract the raw distributions (Pseudo-labels).
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio.transform
import torch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from align.models import ModelCache, get_torch_device
from align.matching import match_with_roma
from align.flow_refine import _estimate_flow, _forward_backward_mask

def process_pair(ref_path: str, src_path: str, models: ModelCache):
    print(f"Processing {os.path.basename(ref_path)}...")
    
    # Read images
    ref_bgr = cv2.imread(ref_path)
    src_bgr = cv2.imread(src_path)
    if ref_bgr is None or src_bgr is None:
        return
        
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    
    # We don't have real geographic transforms for the pseudo-images,
    # so we use identity transforms (pixel coordinates).
    identity_transform = rasterio.transform.Affine.identity()
    
    # 1. Extract RoMa pseudo-labels
    # The matching script expects valid masks, which we can approximate
    # by ensuring non-zero pixels.
    # We call match_with_roma. The hook in matching.py will dump the tensors to disk.
    try:
        match_with_roma(
            arr_ref=ref_gray,
            arr_off_shifted=src_gray,
            ref_transform=identity_transform,
            off_transform=identity_transform,
            shift_px_x=0.0,
            shift_py_y=0.0,
            model_cache=models,
            skip_ransac=True, # We only want the raw neural extraction
            mask_mode="heuristic" # Faster, no OBIA required
        )
    except Exception as e:
        print(f"  RoMa extraction failed: {e}")

    # 2. Extract SEA-RAFT pseudo-labels
    # We need to run _estimate_flow to get the forward flow, then
    # _forward_backward_mask triggers the SEA-RAFT hook.
    try:
        # Get baseline dense flow using DIS
        flow_fwd = _estimate_flow(ref_gray, src_gray)
        # This triggers the SEA-RAFT hook
        _forward_backward_mask(ref_gray, src_gray, flow_fwd, threshold_px=3.0)
    except Exception as e:
        print(f"  SEA-RAFT extraction failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract raw distributions for SSD")
    parser.add_argument("--data-dir", "-d", type=str, required=True, help="Directory with generated pairs")
    parser.add_argument("--out-dir", "-o", type=str, required=True, help="Directory to save raw tensors")
    args = parser.parse_args()
    
    os.environ["SSD_EXTRACT_DIR"] = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    
    device = get_torch_device()
    print(f"Using device: {device}")
    
    # Load frozen models
    print("Loading models...")
    models = ModelCache(device=device)
    # Trigger load
    _ = models.roma
    
    ref_files = sorted(glob.glob(os.path.join(args.data_dir, "*_ref.png")))

    for ref_path in ref_files:
        # Teacher sees the CLEAN source to produce high-quality pseudo-labels.
        # The student (ssd_finetune.py) trains on the PERTURBED source,
        # learning to match teacher quality on harder inputs.
        src_path = ref_path.replace("_ref.png", "_src_clean.png")
        if not os.path.exists(src_path):
            # Fallback for datasets without clean source
            src_path = ref_path.replace("_ref.png", "_src_perturbed.png")
        if not os.path.exists(src_path):
            continue
            
        # Get pair ID, e.g. "pair_00000" from "pair_00000_ref.png"
        pair_id = os.path.basename(ref_path).replace("_ref.png", "")
        
        # Check if already processed
        expected_searaft = os.path.join(args.out_dir, f"searaft_raw_{pair_id}.pt")
        if os.path.exists(expected_searaft):
            print(f"Skipping {pair_id}, already extracted.")
            continue
            
        os.environ["SSD_EXTRACT_ID"] = pair_id
        process_pair(ref_path, src_path, models)
        
    print(f"Done. Extracted pseudo-labels to {args.out_dir}")

if __name__ == "__main__":
    main()
