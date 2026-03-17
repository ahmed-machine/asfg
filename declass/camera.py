"""Camera system definitions for KH-4, KH-7, and KH-9."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraSystem:
    name: str           # Human-readable name
    program: str        # Program name (CORONA, GAMBIT, HEXAGON)
    entity_prefix: str  # Entity ID prefix (DS1, DZB, D3C)
    ee_dataset: str     # EarthExplorer dataset alias
    archive_format: str # "tgz" (multi-frame) or "tif" (single file)
    needs_stitching: bool
    has_stereo_pairs: bool  # Aft/Fore camera pairs
    expected_aspect_range: tuple = (1.5, 20.0)  # (min, max) width/height ratio


KH4 = CameraSystem(
    name="KH-4",
    program="CORONA",
    entity_prefix="DS1",
    ee_dataset="corona2",
    archive_format="tif",   # USGS ships as gzip-compressed single TIF
    needs_stitching=True,   # 4 sub-frame segments need horizontal stitching
    has_stereo_pairs=True,
    expected_aspect_range=(3.0, 15.0),
)

KH7 = CameraSystem(
    name="KH-7",
    program="GAMBIT",
    entity_prefix="DZB",
    ee_dataset="declassii",
    archive_format="tif",   # USGS ships as gzip-compressed single TIF
    needs_stitching=True,   # 2 sub-frame segments need horizontal stitching
    has_stereo_pairs=False,
    expected_aspect_range=(1.5, 8.0),
)

KH9 = CameraSystem(
    name="KH-9",
    program="HEXAGON",
    entity_prefix="D3C",
    ee_dataset="declassiii",
    archive_format="tgz",
    needs_stitching=True,
    has_stereo_pairs=True,
    expected_aspect_range=(2.0, 10.0),
)

ALL_SYSTEMS = [KH4, KH7, KH9]


def identify_camera(entity_id: str) -> CameraSystem:
    """Identify camera system from entity ID prefix."""
    for system in ALL_SYSTEMS:
        if entity_id.startswith(system.entity_prefix):
            return system
    raise ValueError(f"Unknown camera system for entity: {entity_id}")


def camera_from_dataset(dataset_alias: str) -> CameraSystem:
    """Get camera system from EarthExplorer dataset alias."""
    for system in ALL_SYSTEMS:
        if system.ee_dataset == dataset_alias:
            return system
    raise ValueError(f"Unknown dataset alias: {dataset_alias}")
