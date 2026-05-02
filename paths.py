"""On-disk layout of the declass-process pipeline.

Every output the pipeline writes lives under one of two roots:

- ``cache_dir``: shared preprocessing cache (downloads, extracted, stitched,
  georef, ortho). Reused across runs; safe to share between tests.
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


def ortho_path(cache_dir: str, entity_id: str) -> str:
    return os.path.join(cache_dir, "ortho", f"{entity_id}_ortho.tif")


def ortho_coarse_path(cache_dir: str, entity_id: str) -> str:
    """Reference-specific coarse-align sidecar next to the canonical ortho.

    The canonical ``_ortho.tif`` is reference-neutral and immutable once
    written; this sidecar holds the coarse-align geotransform shift for a
    particular reference basemap. Validity is gated by the companion
    provenance JSON so that switching ``--reference`` invalidates the sidecar.
    """
    return os.path.join(cache_dir, "ortho", f"{entity_id}_ortho.coarse.tif")


def ortho_coarse_provenance_path(cache_dir: str, entity_id: str) -> str:
    return os.path.join(cache_dir, "ortho", f"{entity_id}_ortho.coarse.json")


def reference_scratch_cleaned_path(reference_path: str) -> str:
    """Path of the scratch-cleaned sidecar next to a reference TIF.

    Declassified film references (KH-9 DZB1212 etc.) often carry bright
    diagonal scratches at the strip edges. The cleaned sidecar holds
    the inpainted version with the same georeferencing; matchers prefer
    it when its companion provenance JSON matches the current reference.
    """
    base, ext = os.path.splitext(reference_path)
    return f"{base}.scratch_cleaned{ext or '.tif'}"


def reference_scratch_cleaned_provenance_path(reference_path: str) -> str:
    base, _ = os.path.splitext(reference_path)
    return f"{base}.scratch_cleaned.json"


def ortho_segments_dir(cache_dir: str, entity_id: str) -> str:
    return os.path.join(cache_dir, "ortho", f"{entity_id}_segments")


def extracted_dir(cache_dir: str, entity_id: str) -> str:
    return os.path.join(cache_dir, "extracted", entity_id)


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

CACHE_SUBDIRS = ("downloads", "extracted", "stitched", "georef", "ortho")
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
