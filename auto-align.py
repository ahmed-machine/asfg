#!/usr/bin/env python3
"""
Auto-align a misaligned GeoTIFF against a correctly-aligned reference image.

Multi-pass pipeline (runs automatically in a single command):
  1. Coarse offset detection via land/water mask template matching at 15m/px
     and 5m/px refinement.
  1.5. Scale and rotation detection via Fourier-Mellin, SIFT+RANSAC, or
       multi-scale NCC matching. Pre-corrects before fine matching.
  2. For large offsets (>2km): pure translation, then auto-continues.
  3. For medium offsets (200m-2km): land mask NCC template matching on a
     grid -> TPS warp, then auto-continues.
  4. For small offsets (<200m): grayscale NCC template matching at 2m/px
     with phase-correlation sub-pixel refinement -> affine warp.

Handles cross-resolution images with up to 6x scale differences.
Achieves sub-meter affine residual for cross-sensor satellite imagery.
Works for any image pair without manual landmark definitions.

Usage:
    python auto-align.py public/maps/1977-D3C1213.warped.tif \\
        --reference public/maps/1976-KH9-DZB1212.warped.tif -y
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Auto-align a GeoTIFF against a reference with global localization and QA"
    )
    parser.add_argument("input", nargs="?", help="Path to the misaligned GeoTIFF")
    parser.add_argument(
        "--reference", "-r", required=False,
        help="Path to a correctly-aligned reference GeoTIFF"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output path (default: <input>.aligned.tif)"
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--best", action="store_true",
        help="Run at maximum quality, even when slower"
    )
    parser.add_argument(
        "--match-res", type=float, default=5.0,
        help="Resolution in meters/pixel for SIFT matching (default: 5)"
    )
    parser.add_argument(
        "--anchors", type=str, default=None,
        help="Path to anchor GCPs JSON file with known landmarks"
    )
    parser.add_argument(
        "--metadata-priors", nargs="*", default=None,
        help="Optional metadata prior files (JSON or XML)"
    )
    parser.add_argument(
        "--metadata-priors-dir", default=None,
        help="Directory to search for sidecar metadata priors"
    )
    parser.add_argument(
        "--coarse-pass", type=int, default=0,
        help=argparse.SUPPRESS  # internal: tracks recursion depth
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Torch device override (default: auto-detect)"
    )
    parser.add_argument(
        "--tin-tarr-thresh", type=float, default=1.5,
        help="TIN-TARR area ratio threshold for topological filtering "
             "(default: 1.5, higher = more tolerant of terrain change)"
    )
    parser.add_argument(
        "--skip-fpp", action="store_true",
        help="Skip FPP image-based accuracy optimization (Phase B)"
    )
    parser.add_argument(
        "--matcher-anchor", type=str, default="roma",
        choices=["roma"],
        help="Model to use for anchor GCP matching (default: roma)"
    )
    parser.add_argument(
        "--matcher-dense", type=str, default="roma",
        choices=["roma"],
        help="Model to use for dense tiled matching (default: roma)"
    )
    parser.add_argument(
        "--mask-provider", type=str, default="coastal_obia",
        choices=["heuristic", "coastal_obia"],
        help="Semantic mask provider to use during localization and QA"
    )
    parser.add_argument(
        "--global-search", dest="global_search", action="store_true", default=True,
        help="Enable full-reference global localization before local refinement"
    )
    parser.add_argument(
        "--no-global-search", dest="global_search", action="store_false",
        help="Disable global localization and require rough overlap"
    )
    parser.add_argument(
        "--global-search-res", type=float, default=40.0,
        help="Resolution in meters/pixel for global localization search (default: 40)"
    )
    parser.add_argument(
        "--global-search-top-k", type=int, default=3,
        help="Number of global localization hypotheses to keep (default: 3)"
    )
    parser.add_argument(
        "--force-global", action="store_true",
        help="Force a global localization pass even when rough overlap exists"
    )
    parser.add_argument(
        "--reference-window", default=None,
        help="Optional search window in work CRS as left,bottom,right,top"
    )
    parser.add_argument(
        "--qa-json", default=None,
        help="Write independent QA report JSON to this path"
    )
    parser.add_argument(
        "--diagnostics-dir", default=None,
        help="Optional directory for diagnostic outputs"
    )
    parser.add_argument(
        "--allow-abstain", action="store_true",
        help="Allow the pipeline to withhold low-confidence outputs"
    )
    parser.add_argument(
        "--strip-manifest", default=None,
        help="Run a strip manifest JSON instead of a single pair"
    )
    parser.add_argument(
        "--block-manifest", default=None,
        help="Run a block manifest JSON instead of a single pair"
    )
    parser.add_argument(
        "--grid-size", type=int, default=20,
        help="Grid size (NxN) for ARAP optimization (default: 20)"
    )
    parser.add_argument(
        "--grid-iters", type=int, default=300,
        help="Optimization iterations for grid warp (default: 300)"
    )
    parser.add_argument(
        "--arap-weight", type=float, default=1.0,
        help="ARAP regularization weight (default: 1.0)"
    )
    parser.add_argument(
        "--tps-fallback", action="store_true", default=False,
        help="Run TPS fallback warp for comparison (default: off)"
    )
    parser.add_argument(
        "--profile", type=str, default=None,
        help="Camera profile name (e.g. kh9, kh4). Auto-detected from filename if omitted."
    )
    args = parser.parse_args()

    # Activate profile BEFORE importing pipeline modules so that
    # module-level `from .constants import X` bindings get the right values.
    from align.params import set_profile
    if args.profile:
        set_profile(args.profile)
    elif args.input:
        from align.params import detect_camera
        detected = detect_camera(args.input)
        if detected:
            print(f"Auto-detected camera profile: {detected}")
            set_profile(detected)

    # Lazy imports — must come AFTER set_profile() call above
    from align.manifest import run_block_manifest, run_strip_manifest
    from align.pipeline import run

    if args.block_manifest:
        run_block_manifest(args.block_manifest, run)
        return
    if args.strip_manifest:
        run_strip_manifest(args.strip_manifest, run)
        return

    if not args.input or not args.reference:
        parser.error("input and --reference are required unless --strip-manifest or --block-manifest is used")

    run(args)


if __name__ == "__main__":
    main()
