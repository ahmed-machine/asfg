"""Named constants for the alignment pipeline."""

# Flow refinement
MAX_FLOW_BIAS_M = 15.0
FB_CONSISTENCY_PX = 3.0
DIS_VARIATIONAL_ITERS = 5
SEA_RAFT_TILE_SIZE = 1024
MAX_CORRECTION_COARSE_M = 75.0
MAX_CORRECTION_FINE_M = 30.0
MAX_CORRECTION_COMBINED_M = 100.0
FLOW_MEDIAN_KERNEL = 5
FLOW_SOFT_SIGMA = 12.0

# Coastline / DINOv3 feature extraction
COAST_MAX_DIM = 4096
DINO_MAX_DIM = 32768
FEAT_SPATIAL_MAX = 768  # max spatial dim for compressed features
FEAT_COVERAGE_RADIUS_NORM = 0.12  # normalised [-1,1] distance; ~12km at typical scale

# Grid optimizer default pyramid levels: (grid_size, iters)
DEFAULT_PYRAMID_LEVELS = [(8, 300), (24, 300), (64, 500)]

# Grid optimizer early stopping
EARLY_STOP_THRESHOLD = 0.0003  # 0.03% improvement
EARLY_STOP_WINDOW = 20

# RANSAC
RANSAC_REPROJ_THRESHOLD = 5.0

# Anchor matching
ANCHOR_INLIER_THRESHOLD = 6

# Validation
TIN_TARR_THRESH = 1.5
SKIP_FPP = False
MAD_SIGMA = 2.5
MAD_SIGMA_SCALED = 3.5

# CLAHE normalization
CLAHE_CLIP_LIMIT = 3.0

# RoMa tiled matching
ROMA_TILE_SIZE = 1024
ROMA_TILE_OVERLAP = 256
ROMA_SIZE = 640  # 16 * 40, multiple of 16 for DINOv3 backbone
ROMA_NUM_CORRESP = 600

# Minimum valid land fraction for tile/patch acceptance
LAND_MASK_FRAC_MIN = 0.30

# Joint land fraction: reject tiles where BOTH ref and off have < this land fraction
TILE_JOINT_LAND_MIN = 0.08

# Post-collection grid quota: cap matches per spatial cell for even distribution
MATCH_QUOTA_GRID_M = 2000   # cell size in projected metres
MATCH_QUOTA_PER_CELL = 30   # max matches kept per cell

# Cross-validation RANSAC refit threshold (metres)
CV_REFIT_THRESHOLD_M = 40.0

# Coarse offset detection defaults
DEFAULT_TEMPLATE_RADIUS_M = 6000
DEFAULT_SEARCH_MARGIN_M = 300
COARSE_RES = 15.0
REFINE_RES = 5.0

# Scale/rotation detection defaults
SCALE_GRID_COLS = 3
SCALE_GRID_ROWS = 3
SCALE_DETECTION_RES = 5.0
SCALE_MIN_VALID_FRAC = 0.30

# Output CRS
OUTPUT_CRS_EPSG = 3857


def reload_from_profile(profile_name: str = "_base") -> None:
    """Overwrite module-level constants from a named profile.

    All existing ``from .constants import X`` statements continue to see
    the values that were current at import time.  This function updates
    the *module* globals so that any code doing ``constants.X`` (attribute
    access) picks up the new values.  For a full sync (including already-
    imported names), use ``align.params.set_profile()`` which calls this
    internally.
    """
    from .params import load_profile, _sync_constants
    p = load_profile(profile_name)
    _sync_constants(p)
