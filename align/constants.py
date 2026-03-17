"""Named constants for the alignment pipeline."""

# Flow refinement
MAX_FLOW_BIAS_M = 15.0
FB_CONSISTENCY_PX = 3.0

# Coastline / DINOv3 feature extraction
COAST_MAX_DIM = 4096
DINO_MAX_DIM = 32768
FEAT_SPATIAL_MAX = 384  # max spatial dim for compressed DINOv3 features

# Grid optimizer default pyramid levels: (grid_size, iters)
DEFAULT_PYRAMID_LEVELS = [(8, 300), (24, 300), (64, 500)]

# Grid optimizer early stopping
EARLY_STOP_THRESHOLD = 0.0003  # 0.03% improvement
EARLY_STOP_WINDOW = 20

# RANSAC
RANSAC_REPROJ_THRESHOLD = 5.0

# Anchor matching
ANCHOR_INLIER_THRESHOLD = 6

# RoMa tiled matching
ROMA_TILE_SIZE = 1024
ROMA_TILE_OVERLAP = 256
ROMA_SIZE = 616  # 14 * 44, multiple of 14 for DINOv2 backbone
ROMA_NUM_CORRESP = 600

# Minimum valid land fraction for tile/patch acceptance
LAND_MASK_FRAC_MIN = 0.30

# Cross-validation RANSAC refit threshold (metres)
CV_REFIT_THRESHOLD_M = 40.0

# Coarse offset detection defaults
DEFAULT_TEMPLATE_RADIUS_M = 6000
DEFAULT_SEARCH_MARGIN_M = 300

# Output CRS
OUTPUT_CRS_EPSG = 3857
