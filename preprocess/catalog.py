"""Camera system definitions, CSV parsing, and scene/strip grouping for USGS declassified satellite imagery."""

import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


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


@dataclass
class Scene:
    """A single satellite image scene from a USGS CSV catalog."""
    entity_id: str
    acquisition_date: str  # YYYY/MM/DD as in CSV
    mission: str
    frame: int
    camera_type: str       # "Aft", "Forward", camera designation, etc.
    camera_system: CameraSystem
    download_available: bool
    corners: dict          # {"NW": (lat, lon), "NE": ..., "SE": ..., "SW": ...}
    center: tuple          # (lat_dec, lon_dec)
    source_csv: str = ""   # Which CSV file this came from


@dataclass
class Strip:
    """An ordered group of scenes from the same mission/date/camera pass."""
    mission: str
    date: str
    camera_designation: str  # e.g. "A", "F" for KH-9 aft/fore
    camera_system: CameraSystem
    scenes: list = field(default_factory=list)

    @property
    def entity_ids(self) -> list:
        return [s.entity_id for s in self.scenes]

    @property
    def bbox(self) -> tuple:
        """Bounding box (west, south, east, north) in decimal degrees."""
        lats = []
        lons = []
        for s in self.scenes:
            for corner in s.corners.values():
                lats.append(corner[0])
                lons.append(corner[1])
        return (min(lons), min(lats), max(lons), max(lats))


# Column name mappings to handle variations across CSV formats
_COLUMN_ALIASES = {
    "entity_id": ["Entity ID"],
    "acquisition_date": ["Acquisition Date"],
    "mission": ["Mission"],
    "frame": ["Frame", "Frame Number"],
    "camera_type": ["Camera Type", "Camera"],
    "download_available": ["Download Available", "Down Load Available"],
    "center_lat": ["Center Latitude dec"],
    "center_lon": ["Center Longitude dec"],
    "nw_lat": ["NW Corner Lat dec", "NW Cormer Lat dec"],
    "nw_lon": ["NW Corner Long dec"],
    "ne_lat": ["NE Corner Lat dec"],
    "ne_lon": ["NE Corner Long dec"],
    "se_lat": ["SE Corner Lat dec"],
    "se_lon": ["SE Corner Long dec"],
    "sw_lat": ["SW Corner Lat dec"],
    "sw_lon": ["SW Corner Long dec"],
}


def _resolve_column(headers: list, field_name: str) -> str:
    """Find the actual column name for a logical field, handling aliases."""
    aliases = _COLUMN_ALIASES.get(field_name, [field_name])
    for alias in aliases:
        if alias in headers:
            return alias
    raise KeyError(f"No column found for '{field_name}' (tried: {aliases}) in headers: {headers}")


def _detect_dataset(csv_path: str) -> Optional[str]:
    """Detect EE dataset alias from CSV filename."""
    basename = os.path.basename(csv_path).lower()
    for alias in ["corona2", "declassii", "declassiii"]:
        if basename.startswith(alias):
            return alias
    return None


def _camera_designation(entity_id: str, camera_type: str, camera_system: CameraSystem) -> str:
    """Extract camera designation for strip grouping.

    KH-9: single letter from entity ID (e.g. 'A' from D3C1213-200346A003)
    KH-4: 'A' (Aft) or 'F' (Forward) from camera type column
    KH-7: 'H' (single camera)
    """
    if camera_system.entity_prefix == "D3C":
        # D3C1213-200346A003 → 'A'
        parts = entity_id.split("-")
        if len(parts) == 2:
            suffix = parts[1]
            for ch in suffix:
                if ch.isalpha():
                    return ch.upper()
    if camera_system.entity_prefix == "DS1":
        # DS1022-1024DA007 → 'A' (D=direction, A=aft)
        # DS1022-1024DF001 → 'F' (D=direction, F=forward)
        parts = entity_id.split("-")
        if len(parts) == 2:
            suffix = parts[1]
            for i, ch in enumerate(suffix):
                if ch == "D" and i + 1 < len(suffix):
                    next_ch = suffix[i + 1]
                    if next_ch in ("A", "F"):
                        return next_ch
        # Fallback to camera_type column
        ct = camera_type.strip().lower()
        if "aft" in ct:
            return "A"
        if "forward" in ct or "fore" in ct:
            return "F"
    if camera_system.entity_prefix == "DZB":
        return "H"
    return camera_type[:1].upper() if camera_type else "X"


def parse_csv(csv_path: str) -> list:
    """Parse a USGS EarthExplorer CSV into Scene objects."""
    dataset_alias = _detect_dataset(csv_path)

    scenes = []
    with open(csv_path, "r", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if not headers:
            return scenes

        # Resolve column names
        col = {}
        for field_name in _COLUMN_ALIASES:
            col[field_name] = _resolve_column(headers, field_name)

        for row in reader:
            entity_id = row[col["entity_id"]].strip()
            if not entity_id:
                continue

            camera_system = identify_camera(entity_id)
            camera_type = row[col["camera_type"]].strip()

            # Parse download availability
            dl = row[col["download_available"]].strip()
            download_available = dl in ("1", "Y", "Yes", "true", "True")

            # Parse decimal corners
            try:
                corners = {
                    "NW": (float(row[col["nw_lat"]]), float(row[col["nw_lon"]])),
                    "NE": (float(row[col["ne_lat"]]), float(row[col["ne_lon"]])),
                    "SE": (float(row[col["se_lat"]]), float(row[col["se_lon"]])),
                    "SW": (float(row[col["sw_lat"]]), float(row[col["sw_lon"]])),
                }
            except (ValueError, KeyError) as e:
                print(f"  WARNING: Could not parse corners for {entity_id}: {e}")
                continue

            center = (
                float(row[col["center_lat"]]),
                float(row[col["center_lon"]]),
            )

            scene = Scene(
                entity_id=entity_id,
                acquisition_date=row[col["acquisition_date"]].strip(),
                mission=row[col["mission"]].strip(),
                frame=int(row[col["frame"]]),
                camera_type=camera_type,
                camera_system=camera_system,
                download_available=download_available,
                corners=corners,
                center=center,
                source_csv=csv_path,
            )
            scenes.append(scene)

    print(f"  Parsed {len(scenes)} scenes from {os.path.basename(csv_path)}")
    return scenes


def parse_csvs(csv_paths: list) -> list:
    """Parse multiple CSVs and return all scenes."""
    scenes = []
    for path in csv_paths:
        scenes.extend(parse_csv(path))
    return scenes


def group_into_strips(scenes: list) -> list:
    """Group scenes into strips by (mission, date, camera designation).

    Scenes within a strip are sorted by frame number.
    """
    key_to_scenes = defaultdict(list)

    for scene in scenes:
        cam_des = _camera_designation(scene.entity_id, scene.camera_type, scene.camera_system)
        key = (scene.mission, scene.acquisition_date, cam_des)
        key_to_scenes[key].append(scene)

    strips = []
    for (mission, date, cam_des), strip_scenes in sorted(key_to_scenes.items()):
        # Sort by frame number within the strip
        strip_scenes.sort(key=lambda s: s.frame)
        strip = Strip(
            mission=mission,
            date=date,
            camera_designation=cam_des,
            camera_system=strip_scenes[0].camera_system,
            scenes=strip_scenes,
        )
        strips.append(strip)

    return strips



def filter_scenes(scenes: list, entity_ids: list = None,
                  camera_systems: list = None,
                  download_only: bool = True) -> list:
    """Filter scenes by entity IDs, camera systems, and download availability."""
    filtered = scenes
    if entity_ids:
        id_set = set(entity_ids)
        filtered = [s for s in filtered if s.entity_id in id_set]
    if camera_systems:
        sys_set = set(camera_systems)
        filtered = [s for s in filtered if s.camera_system in sys_set]
    if download_only:
        filtered = [s for s in filtered if s.download_available]
    return filtered


# ---------------------------------------------------------------------------
# Geographic coverage selection
# ---------------------------------------------------------------------------

def _rasterize_quad(corners: dict, grid_shape: tuple, bbox: tuple):
    """Rasterize a 4-corner quadrilateral onto a boolean grid.

    Args:
        corners: {"NW": (lat, lon), "NE": ..., "SE": ..., "SW": ...}
        grid_shape: (rows, cols) of the output grid
        bbox: (west, south, east, north) in decimal degrees

    Returns:
        Boolean numpy array of shape grid_shape, True inside the quad.
    """
    import numpy as np

    rows, cols = grid_shape
    west, south, east, north = bbox

    # Quad vertices in (lon, lat) order: NW → NE → SE → SW
    verts = [
        (corners["NW"][1], corners["NW"][0]),
        (corners["NE"][1], corners["NE"][0]),
        (corners["SE"][1], corners["SE"][0]),
        (corners["SW"][1], corners["SW"][0]),
    ]
    n_verts = len(verts)

    # Grid cell centres
    cx = np.linspace(west, east, cols, endpoint=False) + (east - west) / (2 * cols)
    cy = np.linspace(south, north, rows, endpoint=False) + (north - south) / (2 * rows)
    gx, gy = np.meshgrid(cx, cy)
    px = gx.ravel()
    py = gy.ravel()

    # Ray-casting point-in-polygon (vectorised)
    inside = np.zeros(len(px), dtype=bool)
    for i in range(n_verts):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n_verts]
        cond = ((y1 > py) != (y2 > py)) & (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-30) + x1)
        inside ^= cond

    return inside.reshape(grid_shape)


def _combined_coverage(scenes: list, target_bbox: tuple, grid_size: int = 500) -> float:
    """Compute fraction of target_bbox covered by union of scenes' footprints."""
    import numpy as np

    grid_shape = (grid_size, grid_size)
    covered = np.zeros(grid_shape, dtype=bool)
    for scene in scenes:
        covered |= _rasterize_quad(scene.corners, grid_shape, target_bbox)
    return float(covered.sum()) / (grid_size * grid_size)


def select_best_mission_coverage(
    scenes: list,
    target_bbox: tuple,
    min_scene_coverage: float = 0.01,
    camera_systems: list = None,
    prefer_camera: str = None,
) -> tuple:
    """Select the best single-mission+date group of scenes for target bbox coverage.

    Scenes are grouped by (mission, acquisition_date), then sub-grouped by camera
    designation. The group with the highest combined coverage is selected. Frames
    adding less than min_scene_coverage marginal contribution are dropped.

    Args:
        scenes: All candidate scenes (should already be filtered for download_available).
        target_bbox: (west, south, east, north) in decimal degrees.
        min_scene_coverage: Drop frames adding less than this fraction of marginal coverage.
        camera_systems: Optional list of CameraSystem to restrict to.
        prefer_camera: Optional camera designation preference ("A"/"F").

    Returns:
        (selected_scenes, coverage_fraction, metadata_dict)
    """
    import numpy as np

    # Pre-filter
    candidates = scenes
    if camera_systems:
        sys_set = set(cs.name for cs in camera_systems) if hasattr(camera_systems[0], 'name') else set(camera_systems)
        candidates = [s for s in candidates if s.camera_system.name in sys_set or s.camera_system in sys_set]
    candidates = [s for s in candidates if s.download_available]

    # Quick bbox overlap filter
    west, south, east, north = target_bbox
    filtered = []
    for s in candidates:
        lats = [c[0] for c in s.corners.values()]
        lons = [c[1] for c in s.corners.values()]
        s_west, s_east = min(lons), max(lons)
        s_south, s_north = min(lats), max(lats)
        if s_east > west and s_west < east and s_north > south and s_south < north:
            filtered.append(s)

    if not filtered:
        print("  WARNING: No scenes overlap the target bbox")
        return [], 0.0, {}

    # Group by (mission, date)
    mission_groups = defaultdict(list)
    for s in filtered:
        key = (s.mission, s.acquisition_date)
        mission_groups[key].append(s)

    # For each mission+date, sub-group by camera designation and evaluate
    grid_size = 500
    grid_shape = (grid_size, grid_size)
    best_scenes = []
    best_coverage = 0.0
    best_meta = {}

    for (mission, date), group_scenes in mission_groups.items():
        camera_system = group_scenes[0].camera_system

        # Sub-group by camera designation
        cam_subgroups = defaultdict(list)
        for s in group_scenes:
            cam_des = _camera_designation(s.entity_id, s.camera_type, s.camera_system)
            cam_subgroups[cam_des].append(s)

        # Evaluate each camera sub-group
        for cam_des, cam_scenes in cam_subgroups.items():
            if prefer_camera and cam_des != prefer_camera.upper():
                # Still evaluate but only pick if nothing better
                pass

            coverage = _combined_coverage(cam_scenes, target_bbox, grid_size)

            # Prefer the requested camera when coverage is comparable (within 5%)
            is_preferred = prefer_camera and cam_des == prefer_camera.upper()
            effective_coverage = coverage + (0.05 if is_preferred else 0.0)

            if effective_coverage > best_coverage or (
                effective_coverage == best_coverage and len(cam_scenes) < len(best_scenes)
            ):
                best_coverage = effective_coverage
                best_scenes = cam_scenes
                best_meta = {
                    "mission": mission,
                    "date": date,
                    "camera_system": camera_system.name,
                    "camera_designation": cam_des,
                }

    if not best_scenes:
        return [], 0.0, {}

    # Within best group, drop frames with negligible marginal contribution
    # Sort by individual coverage (descending) for greedy selection
    scene_coverages = []
    for s in best_scenes:
        mask = _rasterize_quad(s.corners, grid_shape, target_bbox)
        scene_coverages.append((s, mask))
    scene_coverages.sort(key=lambda x: -x[1].sum())

    selected = []
    covered = np.zeros(grid_shape, dtype=bool)
    total_cells = grid_size * grid_size

    for scene, mask in scene_coverages:
        marginal = float((mask & ~covered).sum()) / total_cells
        if marginal >= min_scene_coverage:
            selected.append(scene)
            covered |= mask

    # Sort selected by frame number for consistent ordering
    selected.sort(key=lambda s: s.frame)
    final_coverage = float(covered.sum()) / total_cells

    best_meta["scenes_selected"] = len(selected)
    best_meta["predicted_coverage"] = round(final_coverage, 3)

    print(f"  Selected {len(selected)} scenes from {best_meta['camera_system']} "
          f"mission {best_meta['mission']} ({best_meta['date']}) "
          f"cam={best_meta['camera_designation']} — {final_coverage*100:.1f}% coverage")
    for s in selected:
        print(f"    {s.entity_id} frame={s.frame}")

    return selected, final_coverage, best_meta
