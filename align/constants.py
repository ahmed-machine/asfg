"""Named constants for the alignment pipeline."""

# Flow refinement
MAX_FLOW_BIAS_M = 15.0
FB_CONSISTENCY_PX = 3.0
DIS_VARIATIONAL_ITERS = 5
SEA_RAFT_TILE_SIZE = 1024
MAX_CORRECTION_COARSE_M = 50.0
MAX_CORRECTION_FINE_M = 20.0
MAX_CORRECTION_COMBINED_M = 60.0
FLOW_MEDIAN_KERNEL = 3
FLOW_BILATERAL_D = 9
FLOW_BILATERAL_SIGMA_COLOR = 3.0
FLOW_BILATERAL_SIGMA_SPACE = 5.0
FLOW_SOFT_SIGMA = 6.0

# Coastline
COAST_MAX_DIM = 4096

# DINO feature extraction
DINO_MAX_DIM = 4096

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
MAD_SIGMA = 2.5
MAD_SIGMA_SCALED = 3.5

# CLAHE normalization
CLAHE_CLIP_LIMIT = 3.0

# RoMa tiled matching
ROMA_TILE_SIZE = 1024
ROMA_TILE_OVERLAP = 512
ROMA_SIZE = 640  # 16 * 40, multiple of 16 for DINOv3 backbone
ROMA_NUM_CORRESP = 600

# Minimum valid land fraction for tile/patch acceptance
LAND_MASK_FRAC_MIN = 0.30

# Joint land fraction: reject tiles where BOTH ref and off have < this land fraction
TILE_JOINT_LAND_MIN = 0.03

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
