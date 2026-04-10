"""Typed records shared across alignment stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# String enums for scattered literals
# ---------------------------------------------------------------------------

class MatcherType(str, Enum):
    """Dense feature matcher backend."""
    ROMA = "roma"


class MaskProvider(str, Enum):
    """Mask provider backend for pipeline-level mask_provider field."""
    HEURISTIC = "heuristic"
    COASTAL_OBIA = "coastal_obia"


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in projected coordinates."""
    left: float
    bottom: float
    right: float
    top: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.left, self.bottom, self.right, self.top)


# ---------------------------------------------------------------------------
# Result dataclasses for function returns (replacing bare tuples)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CoarseOffset:
    """Result of coarse offset detection."""
    dx_m: Optional[float]
    dy_m: Optional[float]
    correlation: float
    method: str = "land_mask_ncc"


@dataclass(frozen=True, slots=True)
class AffineEstimate:
    """Result of RANSAC or least-squares affine fitting."""
    M: np.ndarray
    residuals: list[float]
    n_inliers: int = 0


@dataclass(frozen=True, slots=True)
class OutlierRemovalResult:
    """Result of iterative outlier removal."""
    matched_pairs: list[Any]
    M_geo: np.ndarray
    geo_residuals: list[float]


@dataclass(frozen=True, slots=True)
class ReferenceOffsetCorrection:
    """Result of reference offset detection and correction."""
    matched_pairs: list[Any]
    M_geo: np.ndarray
    geo_residuals: list[float]
    was_corrected: bool


@dataclass(slots=True)
class MatchPair:
    """Typed representation of a matched correspondence."""

    ref_x: float
    ref_y: float
    off_x: float
    off_y: float
    confidence: float
    name: str
    precision: float = 1.0
    source: str = ""
    hypothesis_id: str = ""

    @property
    def is_anchor(self) -> bool:
        return self.name.startswith("anchor:")

    def with_confidence(self, confidence: float) -> "MatchPair":
        """Return a copy with updated confidence."""
        return MatchPair(
            ref_x=self.ref_x, ref_y=self.ref_y,
            off_x=self.off_x, off_y=self.off_y,
            confidence=confidence, name=self.name,
            precision=self.precision, source=self.source,
            hypothesis_id=self.hypothesis_id,
        )

    def ref_coords(self) -> tuple[float, float]:
        return (self.ref_x, self.ref_y)

    def off_coords(self) -> tuple[float, float]:
        return (self.off_x, self.off_y)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_legacy(cls, pair: Sequence[Any]) -> "MatchPair":
        if len(pair) < 6:
            raise ValueError(f"Expected 6 values for legacy match pair, got {len(pair)}")
        prec = float(pair[6]) if len(pair) > 6 else 1.0
        return cls(
            ref_x=float(pair[0]),
            ref_y=float(pair[1]),
            off_x=float(pair[2]),
            off_y=float(pair[3]),
            confidence=float(pair[4]),
            name=str(pair[5]),
            precision=prec,
        )


@dataclass(slots=True)
class GCP:
    """Ground control point: pixel (col, row) → geographic (gx, gy)."""

    col: float
    row: float
    gx: float
    gy: float
    synthetic: bool = False
    source: str = "match"
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_legacy(cls, gcp: Sequence[Any], *, synthetic: bool = False,
                    source: str = "match", name: str = "") -> "GCP":
        if len(gcp) < 4:
            raise ValueError(f"Expected 4 values for legacy GCP, got {len(gcp)}")
        return cls(
            col=float(gcp[0]),
            row=float(gcp[1]),
            gx=float(gcp[2]),
            gy=float(gcp[3]),
            synthetic=synthetic,
            source=source,
            name=name,
        )


@dataclass(slots=True)
class MetadataPrior:
    """External or derived geospatial prior."""

    source: str
    confidence: float = 0.5
    west: Optional[float] = None
    south: Optional[float] = None
    east: Optional[float] = None
    north: Optional[float] = None
    crs: str = "EPSG:4326"
    center_lon: Optional[float] = None
    center_lat: Optional[float] = None
    corners: dict[str, tuple[float, float]] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def has_bounds(self) -> bool:
        return None not in (self.west, self.south, self.east, self.north)

    def bounds(self) -> Optional[tuple[float, float, float, float]]:
        if not self.has_bounds:
            return None
        return (float(self.west), float(self.south), float(self.east), float(self.north))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GlobalHypothesis:
    """Candidate coarse localization against the reference image."""

    hypothesis_id: str
    score: float
    source: str
    left: float
    bottom: float
    right: float
    top: float
    dx_m: float = 0.0
    dy_m: float = 0.0
    scale_hint: float = 1.0
    rotation_hint_deg: float = 0.0
    work_crs: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2.0, (self.bottom + self.top) / 2.0)

    def bounds(self) -> tuple[float, float, float, float]:
        return (self.left, self.bottom, self.right, self.top)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QaReport:
    """Independent QA result for a candidate output."""

    candidate: str
    output_path: str
    total_score: float
    confidence: float
    accepted: bool
    image_metrics: dict[str, Any] = field(default_factory=dict)
    holdout_metrics: dict[str, Any] = field(default_factory=dict)
    coverage: float = 0.0
    cv_mean_m: Optional[float] = None
    hypothesis_id: str = ""
    reasons: list[str] = field(default_factory=list)
    quality_grade: str = "D"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AlignmentJob:
    """Single manifest-driven alignment job."""

    input_path: str
    reference_path: str
    output_path: Optional[str] = None
    anchors_path: Optional[str] = None
    metadata_priors: list[str] = field(default_factory=list)
    qa_json_path: Optional[str] = None
    diagnostics_dir: Optional[str] = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StripManifest:
    """Sequential strip-alignment manifest."""

    manifest_path: str
    jobs: list[AlignmentJob]
    shared_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BlockManifest:
    """Block-alignment manifest spanning multiple strips or jobs."""

    manifest_path: str
    jobs: list[AlignmentJob]
    shared_options: dict[str, Any] = field(default_factory=dict)


