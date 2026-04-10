"""On-disk layout of the declass-process pipeline.

Every output the pipeline writes lives under one of two roots:

- ``cache_dir``: shared preprocessing cache (downloads, extracted, stitched,
  georef, ortho, ba). Reused across runs; safe to share between tests.
- ``output_dir``: per-run outputs (aligned, mosaic, manifests, diagnostics,
  cropped, match_files, reference).

The helpers below centralize the path construction so callers don't need to
know the directory layout, and adding a new stage only touches one file.
"""

from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# Preprocessing cache (under cache_dir)
# ---------------------------------------------------------------------------

def georef_path(cache_dir: str, entity_id: str) -> str:
    return os.path.join(cache_dir, "georef", f"{entity_id}_georef.tif")


def georef_metadata_path(cache_dir: str, entity_id: str) -> str:
    return os.path.join(cache_dir, "georef", f"{entity_id}_metadata.json")


def stitched_path(cache_dir: str, entity_id: str) -> str:
    return os.path.join(cache_dir, "stitched", f"{entity_id}_stitched.tif")


def ortho_path(cache_dir: str, entity_id: str, bundle_adjusted: bool = False) -> str:
    suffix = "_ortho_ba.tif" if bundle_adjusted else "_ortho.tif"
    return os.path.join(cache_dir, "ortho", f"{entity_id}{suffix}")


def ortho_segments_dir(cache_dir: str, entity_id: str) -> str:
    return os.path.join(cache_dir, "ortho", f"{entity_id}_segments")


def extracted_dir(cache_dir: str, entity_id: str) -> str:
    return os.path.join(cache_dir, "extracted", entity_id)


def bundle_adjust_dir(cache_dir: str, strip_key: str) -> str:
    return os.path.join(cache_dir, "ba", strip_key)


def ba_camera_path(stitched_path_value: str) -> str:
    """ASP writes the camera model .tsai file alongside the input stitched image."""
    base, _ = os.path.splitext(stitched_path_value)
    return f"{base}.tsai"


# ---------------------------------------------------------------------------
# Per-run outputs (under output_dir)
# ---------------------------------------------------------------------------

def aligned_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "aligned")


def aligned_path(output_dir: str, entity_id: str) -> str:
    return os.path.join(output_dir, "aligned", f"{entity_id}_aligned.tif")


def diagnostics_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "diagnostics")


def scene_diagnostics_dir(output_dir: str, entity_id: str) -> str:
    return os.path.join(output_dir, "diagnostics", entity_id)


def manifests_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "manifests")


def alignment_manifest_path(output_dir: str) -> str:
    return os.path.join(output_dir, "manifests", "alignment_manifest.json")


def scene_prior_path(output_dir: str, entity_id: str) -> str:
    return os.path.join(output_dir, "manifests", f"{entity_id}_prior.json")


def scene_anchors_path(output_dir: str, entity_id: str) -> str:
    return os.path.join(output_dir, "manifests", f"{entity_id}_auto_anchors.json")


def cropped_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "cropped")


def cropped_path(output_dir: str, entity_id: str) -> str:
    return os.path.join(output_dir, "cropped", f"{entity_id}_cropped.tif")


def match_files_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "match_files")


# ---------------------------------------------------------------------------
# Directory ensurance
# ---------------------------------------------------------------------------

CACHE_SUBDIRS = ("downloads", "extracted", "stitched", "georef", "ortho", "ba")
OUTPUT_SUBDIRS = ("aligned", "mosaic", "manifests", "diagnostics", "match_files")


def ensure_pipeline_dirs(output_dir: str, cache_dir: str | None = None) -> None:
    """Create every directory the pipeline writes to.

    Preprocessing dirs land under ``cache_dir`` so they can be shared across
    runs; per-run outputs land under ``output_dir``.
    """
    cache_root = cache_dir or output_dir
    for subdir in CACHE_SUBDIRS:
        os.makedirs(os.path.join(cache_root, subdir), exist_ok=True)
    for subdir in OUTPUT_SUBDIRS:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
