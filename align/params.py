"""Camera-system profile loader and parameter management.

Provides AlignParams — a structured container for all tunable hyperparameters —
loaded from YAML profiles with inheritance.  Supports runtime overrides via the
``override()`` context manager (for Optuna trials).
"""

from __future__ import annotations

import copy
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Profile directory
# ---------------------------------------------------------------------------

_PROFILES_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "profiles")


# ---------------------------------------------------------------------------
# Parameter section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CoarseParams:
    template_radius_m: float = 6000.0
    search_margin_m: float = 300.0
    coarse_res: float = 15.0
    refine_res: float = 5.0
    min_ncc: float = 0.3


@dataclass
class ScaleRotationParams:
    grid_cols: int = 3
    grid_rows: int = 3
    detection_res: float = 5.0
    min_valid_frac: float = 0.30


@dataclass
class MatchingParams:
    roma_tile_size: int = 1024
    roma_tile_overlap: int = 512
    roma_size: int = 640
    roma_num_corresp: int = 600
    land_mask_frac_min: float = 0.30
    tile_joint_land_min: float = 0.03
    match_quota_grid_m: int = 2000
    match_quota_per_cell: int = 30
    ransac_reproj_threshold: float = 5.0
    estimation_method: str = "ransac"  # "ransac", "lmeds", "magsac"


@dataclass
class ValidationParams:
    anchor_inlier_threshold: int = 6
    cv_refit_threshold_m: float = 40.0
    tin_tarr_thresh: float = 1.5
    mad_sigma: float = 2.5
    mad_sigma_scaled: float = 3.5  # used when needs_scale_rotation=True


@dataclass
class GridOptimParams:
    pyramid_levels: List[List[int]] = field(
        default_factory=lambda: [[8, 300], [24, 300], [64, 500]])
    early_stop_threshold: float = 0.0003
    early_stop_window: int = 20
    lr: float = 0.002
    w_data: float = 1.0
    w_chamfer: float = 0.30
    w_arap: float = 0.5
    w_laplacian: float = 0.2
    w_disp: float = 0.05
    max_residual_norm: float = 0.03
    # Per-level multiplicative scales (indexed by level_idx)
    level_w_data_scale: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])
    level_w_disp_scale: List[float] = field(default_factory=lambda: [1.0, 0.8, 0.7])
    level_reg_scale: List[float] = field(default_factory=lambda: [1.0, 1.30, 1.50])
    level_chamfer_scale: List[float] = field(default_factory=lambda: [1.0, 0.667, 0.5])


@dataclass
class FlowParams:
    max_flow_bias_m: float = 15.0
    fb_consistency_px: float = 3.0
    coast_max_dim: int = 4096
    dis_variational_iters: int = 5
    sea_raft_tile_size: int = 1024
    max_correction_coarse_m: float = 75.0
    max_correction_fine_m: float = 30.0
    max_correction_combined_m: float = 100.0
    median_kernel: int = 3
    bilateral_d: int = 9
    bilateral_sigma_color: float = 3.0
    bilateral_sigma_space: float = 5.0

@dataclass
class NormalizationParams:
    clahe_clip_limit: float = 3.0
    wallis_matching: bool = True
    flow_joint_percentile: bool = True
    flow_percentile_lo: int = 1
    flow_percentile_hi: int = 99


@dataclass
class QaParams:
    """QA scoring weights and acceptance thresholds."""
    # Score formula weights
    grid_weight: float = 0.55
    patch_weight: float = 0.25
    stable_boundary_weight: float = 18.0
    shore_boundary_weight: float = 12.0
    # Acceptance thresholds
    accept_image_score_max: float = 140.0
    accept_holdout_median_max: float = 90.0
    accept_cv_mean_max: float = 90.0
    accept_coverage_min: float = 0.10
    # Quality grade thresholds (A/B/C/D)
    grade_a_score: float = 40.0
    grade_a_holdout: float = 30.0
    grade_b_score: float = 80.0
    grade_b_holdout: float = 60.0


@dataclass
class CameraParams:
    """Physical camera model parameters for OpticalBar pre-correction."""
    type: str = ""                          # "opticalbar" or empty (no correction)
    focal_length: float = 0.0              # metres
    pixel_pitch: float = 0.0               # metres (scan resolution)
    scan_time: float = 0.0                 # seconds
    speed: float = 0.0                     # m/s orbital velocity
    forward_tilt: float = 0.0             # radians (positive=forward, negative=aft)
    scan_dir: str = "right"               # "right" or "left"
    motion_compensation_factor: float = 1.0
    scan_arc_deg: float = 70.0            # total scan arc in degrees
    # When True, ASP bundle_adjust is called with --solve-intrinsics
    # --intrinsics-to-float focal_length. 2OC (Hou et al. 2023, Table 6)
    # reports fitted focal lengths drifting ~0.1% from nominal across
    # sub-images of the same strip; this per-strip adaptive fit is
    # responsible for ~30-45% of the paper's reported accuracy gain.
    bundle_adjust_solve_intrinsics: bool = False
    # When True, _maybe_generate_asp_ortho processes each USGS sub-image
    # (a, b, c, d for KH-4; a..g for KH-9 PC) independently — separate
    # cam_gen + mapproject per sub-frame, then a gdalbuildvrt mosaic of
    # the resulting orthos. 2OC §3.1: stitching first leaks per-segment
    # offsets into the single-camera fit. Default false; gate per profile.
    per_segment_ortho: bool = False

    @property
    def is_panoramic(self) -> bool:
        return self.type == "opticalbar" and self.focal_length > 0

    def to_dict(self) -> dict:
        """Convert to dict for preprocess.camera_model functions."""
        return {
            "focal_length": self.focal_length,
            "pixel_pitch": self.pixel_pitch,
            "scan_time": self.scan_time,
            "speed": self.speed,
            "forward_tilt": self.forward_tilt,
            "scan_dir": self.scan_dir,
            "motion_compensation_factor": self.motion_compensation_factor,
        }


@dataclass
class MetaParams:
    name: str = "base"
    description: str = ""
    cameras: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------

@dataclass
class AlignParams:
    """All tunable parameters, loaded from YAML profile."""
    meta: MetaParams = field(default_factory=MetaParams)
    camera: CameraParams = field(default_factory=CameraParams)
    coarse: CoarseParams = field(default_factory=CoarseParams)
    scale_rotation: ScaleRotationParams = field(default_factory=ScaleRotationParams)
    matching: MatchingParams = field(default_factory=MatchingParams)
    validation: ValidationParams = field(default_factory=ValidationParams)
    grid_optim: GridOptimParams = field(default_factory=GridOptimParams)
    flow: FlowParams = field(default_factory=FlowParams)
    normalization: NormalizationParams = field(default_factory=NormalizationParams)
    qa: QaParams = field(default_factory=QaParams)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_active: Optional[AlignParams] = None


# ---------------------------------------------------------------------------
# YAML loading with inheritance
# ---------------------------------------------------------------------------

def _resolve_profile_path(name: str) -> str:
    """Find a profile YAML by name, searching _PROFILES_DIR."""
    candidates = [
        os.path.join(_PROFILES_DIR, f"{name}.yaml"),
        os.path.join(_PROFILES_DIR, f"{name}.yml"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Allow absolute / relative paths
    if os.path.isfile(name):
        return name
    raise FileNotFoundError(
        f"Profile '{name}' not found. Searched: {candidates}")


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (base is mutated)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _load_profile_dict(name: str, _seen: set | None = None) -> dict:
    """Load a profile dict, recursively resolving ``inherits``."""
    if _seen is None:
        _seen = set()
    if name in _seen:
        raise ValueError(f"Circular profile inheritance: {name}")
    _seen.add(name)

    path = _resolve_profile_path(name)
    data = _load_yaml(path)

    parent_name = data.pop("inherits", None)
    if parent_name:
        parent = _load_profile_dict(parent_name, _seen)
        data = _deep_merge(parent, data)

    return data


def _dict_to_params(d: dict) -> AlignParams:
    """Convert a flat/nested dict to an AlignParams dataclass."""
    p = AlignParams()
    section_map = {
        "meta": (MetaParams, "meta"),
        "camera": (CameraParams, "camera"),
        "coarse": (CoarseParams, "coarse"),
        "scale_rotation": (ScaleRotationParams, "scale_rotation"),
        "matching": (MatchingParams, "matching"),
        "validation": (ValidationParams, "validation"),
        "grid_optim": (GridOptimParams, "grid_optim"),
        "flow": (FlowParams, "flow"),
        "normalization": (NormalizationParams, "normalization"),
        "qa": (QaParams, "qa"),
    }
    for section_key, (cls, attr_name) in section_map.items():
        if section_key in d and isinstance(d[section_key], dict):
            section_data = d[section_key]
            # Only pass keys that the dataclass expects
            valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
            filtered = {k: v for k, v in section_data.items() if k in valid_keys}
            setattr(p, attr_name, cls(**filtered))
    return p


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_profile(name: str = "_base") -> AlignParams:
    """Load a named profile (with inheritance) and return an AlignParams."""
    d = _load_profile_dict(name)
    return _dict_to_params(d)


def get_params() -> AlignParams:
    """Return the currently active parameters (loads _base on first call)."""
    global _active
    if _active is None:
        _active = load_profile("_base")
    return _active


def set_profile(name: str) -> AlignParams:
    """Load a profile and set it as the active singleton.  Returns it."""
    global _active
    _active = load_profile(name)
    _sync_constants(_active)
    return _active


@contextmanager
def override(**kwargs):
    """Temporarily patch active params for one Optuna trial.

    Keys use dotted notation: ``override(grid_optim__w_data=2.0)``
    maps to ``params.grid_optim.w_data = 2.0``.  Double-underscore
    separates section from field.
    """
    global _active
    params = get_params()
    backup = copy.deepcopy(params)

    try:
        for key, value in kwargs.items():
            parts = key.split("__")
            if len(parts) == 2:
                section, attr = parts
                obj = getattr(params, section)
                setattr(obj, attr, value)
            elif len(parts) == 1:
                # Flat key — search all sections
                found = False
                for section_name in ("coarse", "scale_rotation", "matching",
                                     "validation", "grid_optim", "flow",
                                     "normalization"):
                    obj = getattr(params, section_name)
                    if hasattr(obj, key):
                        setattr(obj, key, value)
                        found = True
                        break
                if not found:
                    raise KeyError(f"Unknown parameter: {key}")
            else:
                raise KeyError(f"Invalid key format: {key}")

        # Sync constants module
        _sync_constants(params)
        yield params
    finally:
        _active = backup
        _sync_constants(backup)


def update_profile_yaml(profile_name: str, updates: dict) -> str:
    """Write tuned parameters back to a profile YAML.

    *updates* uses dotted notation matching ``override()``::

        update_profile_yaml("kh9", {
            "matching__roma_size": 784,
            "grid_optim__w_data": 1.5,
        })

    Returns the path of the updated YAML file.
    """
    path = _resolve_profile_path(profile_name)
    data = _load_yaml(path)

    for key, value in updates.items():
        parts = key.split("__")
        if len(parts) == 2:
            section, attr = parts
            if section not in data:
                data[section] = {}
            data[section][attr] = value
        elif len(parts) == 1:
            # Try to find section
            for section_name in ("coarse", "scale_rotation", "matching",
                                 "validation", "grid_optim", "flow",
                                 "normalization"):
                section_data = data.get(section_name, {})
                if isinstance(section_data, dict) and key in section_data:
                    section_data[key] = value
                    break
        else:
            raise KeyError(f"Invalid key format: {key}")

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return path


def _sync_constants(params: AlignParams) -> None:
    """Push AlignParams values into align.constants module globals."""
    from . import constants as _c
    g = vars(_c)

    # Coarse
    g["DEFAULT_TEMPLATE_RADIUS_M"] = params.coarse.template_radius_m
    g["DEFAULT_SEARCH_MARGIN_M"] = params.coarse.search_margin_m
    g["COARSE_RES"] = params.coarse.coarse_res
    g["REFINE_RES"] = params.coarse.refine_res

    # Scale/Rotation
    g["SCALE_GRID_COLS"] = params.scale_rotation.grid_cols
    g["SCALE_GRID_ROWS"] = params.scale_rotation.grid_rows
    g["SCALE_DETECTION_RES"] = params.scale_rotation.detection_res
    g["SCALE_MIN_VALID_FRAC"] = params.scale_rotation.min_valid_frac

    # Matching
    g["ROMA_TILE_SIZE"] = params.matching.roma_tile_size
    g["ROMA_TILE_OVERLAP"] = params.matching.roma_tile_overlap
    g["ROMA_SIZE"] = params.matching.roma_size
    g["ROMA_NUM_CORRESP"] = params.matching.roma_num_corresp
    g["LAND_MASK_FRAC_MIN"] = params.matching.land_mask_frac_min
    g["TILE_JOINT_LAND_MIN"] = params.matching.tile_joint_land_min
    g["MATCH_QUOTA_GRID_M"] = params.matching.match_quota_grid_m
    g["MATCH_QUOTA_PER_CELL"] = params.matching.match_quota_per_cell
    g["RANSAC_REPROJ_THRESHOLD"] = params.matching.ransac_reproj_threshold

    # Validation
    g["ANCHOR_INLIER_THRESHOLD"] = params.validation.anchor_inlier_threshold
    g["CV_REFIT_THRESHOLD_M"] = params.validation.cv_refit_threshold_m
    g["TIN_TARR_THRESH"] = params.validation.tin_tarr_thresh
    g["MAD_SIGMA"] = params.validation.mad_sigma
    g["MAD_SIGMA_SCALED"] = params.validation.mad_sigma_scaled

    # Grid optim
    g["DEFAULT_PYRAMID_LEVELS"] = [
        tuple(lv) for lv in params.grid_optim.pyramid_levels]
    g["EARLY_STOP_THRESHOLD"] = params.grid_optim.early_stop_threshold
    g["EARLY_STOP_WINDOW"] = params.grid_optim.early_stop_window

    # Flow
    g["MAX_FLOW_BIAS_M"] = params.flow.max_flow_bias_m
    g["FB_CONSISTENCY_PX"] = params.flow.fb_consistency_px
    g["COAST_MAX_DIM"] = params.flow.coast_max_dim
    g["DIS_VARIATIONAL_ITERS"] = params.flow.dis_variational_iters
    g["SEA_RAFT_TILE_SIZE"] = params.flow.sea_raft_tile_size
    g["MAX_CORRECTION_COARSE_M"] = params.flow.max_correction_coarse_m
    g["MAX_CORRECTION_FINE_M"] = params.flow.max_correction_fine_m
    g["MAX_CORRECTION_COMBINED_M"] = params.flow.max_correction_combined_m
    g["FLOW_MEDIAN_KERNEL"] = params.flow.median_kernel
    g["FLOW_BILATERAL_D"] = params.flow.bilateral_d
    g["FLOW_BILATERAL_SIGMA_COLOR"] = params.flow.bilateral_sigma_color
    g["FLOW_BILATERAL_SIGMA_SPACE"] = params.flow.bilateral_sigma_space

    # Normalization
    g["CLAHE_CLIP_LIMIT"] = params.normalization.clahe_clip_limit


# ---------------------------------------------------------------------------
# Camera detection from filename
# ---------------------------------------------------------------------------

# Order matters: KH-7 pattern is a subset of general DZB, so check it first.
_CAMERA_PATTERNS = [
    (re.compile(r"DZB004\d+H", re.IGNORECASE), "kh7"),   # KH-7 GAMBIT
    (re.compile(r"D3C\d{4}", re.IGNORECASE), "kh9"),      # KH-9 Hexagon (mapping)
    (re.compile(r"DZB\d+", re.IGNORECASE), "kh9"),         # KH-9 Hexagon (survey)
    (re.compile(r"DS\d{4}", re.IGNORECASE), "kh4"),        # KH-4 Corona / LANYARD
]


def detect_camera(filename: str) -> str | None:
    """Detect camera profile from *filename* (basename or full path).

    Returns profile name ('kh9', 'kh4', 'kh7') or None if unrecognised.
    """
    basename = os.path.basename(filename)
    for pattern, profile in _CAMERA_PATTERNS:
        if pattern.search(basename):
            return profile
    return None
