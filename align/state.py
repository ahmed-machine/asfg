"""Pipeline state container passed between alignment steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from rasterio.crs import CRS

from .types import BBox, GCP, GlobalHypothesis, MaskProvider, MatchPair, MetadataPrior, QaReport


@dataclass
class AlignState:
    """Explicit state passed between pipeline steps."""
    input_path: str
    reference_path: str
    output_path: str
    work_crs: Optional[CRS] = None
    overlap: Optional[BBox] = None
    anchors_path: Optional[str] = None
    best: bool = True
    match_res: float = 5.0
    coarse_pass: int = 0
    yes: bool = False

    # Populated during setup
    offset_res_m: float = 0.0
    ref_res_m: float = 0.0
    expected_scale: float = 1.0

    # Updated during pipeline
    current_input: str = ""
    coarse_dx: float = 0.0
    coarse_dy: float = 0.0
    coarse_total: float = 0.0
    coarse_corr: float = 0.0
    precorrection_applied: bool = False
    precorrection_tmp: Optional[str] = None
    needs_scale_rotation: bool = False

    # Scale/rotation detection quality
    scale_valid_patches: int = 0
    scale_total_patches: int = 0
    scale_x_spread: float = 0.0
    scale_y_spread: float = 0.0

    matched_pairs: list[MatchPair] = field(default_factory=list)
    gcps: list[GCP] = field(default_factory=list)
    match_weights: Optional[np.ndarray] = None
    ransac_survivor_count: int = 0
    match_quality_residual: float = float('inf')
    boundary_gcps: list[GCP] = field(default_factory=list)
    geo_residuals: list[float] = field(default_factory=list)
    mean_residual: float = float('inf')
    max_residual: float = float('inf')
    gcp_coverage: float = 1.0
    used_neural: bool = False
    use_sift_refinement: bool = False

    was_corrected: bool = False
    needs_ortho: bool = False
    rough_georef_path: Optional[str] = None
    correction_outliers: List[str] = field(default_factory=list)

    M_geo: Optional[np.ndarray] = None
    cv_mean: Optional[float] = None
    model_cache: Optional[object] = None
    tin_tarr_thresh: float = 1.5
    skip_fpp: bool = False
    matcher_anchor: str = "roma"
    matcher_dense: str = "roma"
    grid_size: int = 20
    grid_iters: int = 300
    arap_weight: float = 1.0
    mask_provider: MaskProvider = MaskProvider.COASTAL_OBIA
    global_search: bool = True
    global_search_res: float = 40.0
    global_search_top_k: int = 3
    force_global: bool = False
    reference_window: Optional[tuple] = None
    metadata_prior_paths: List[str] = field(default_factory=list)
    metadata_priors: List[MetadataPrior] = field(default_factory=list)
    global_hypotheses: List[GlobalHypothesis] = field(default_factory=list)
    chosen_hypothesis: Optional[GlobalHypothesis] = None
    reference_bounds_work: Optional[tuple] = None
    target_bounds_work: Optional[tuple] = None
    qa_holdout_pairs: list[MatchPair] = field(default_factory=list)
    qa_reports: List[QaReport] = field(default_factory=list)
    qa_json_path: Optional[str] = None
    diagnostics_dir: Optional[str] = None
    allow_abstain: bool = False
    tps_fallback: bool = False
    abstained: bool = False
    temp_paths: List[str] = field(default_factory=list)
