"""Typed records shared across alignment stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional, Sequence


LegacyMatch = tuple[float, float, float, float, float, str]
LegacyGCP = tuple[float, float, float, float]


@dataclass(slots=True)
class MatchPair:
    """Typed representation of a matched correspondence."""

    ref_x: float
    ref_y: float
    off_x: float
    off_y: float
    confidence: float
    name: str
    source: str = ""
    semantic_class: str = "unknown"
    hypothesis_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_anchor(self) -> bool:
        return self.name.startswith("anchor:")

    def to_legacy(self) -> LegacyMatch:
        return (
            float(self.ref_x),
            float(self.ref_y),
            float(self.off_x),
            float(self.off_y),
            float(self.confidence),
            str(self.name),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_legacy(cls, pair: Sequence[Any]) -> "MatchPair":
        if len(pair) < 6:
            raise ValueError(f"Expected 6 values for legacy match pair, got {len(pair)}")
        return cls(
            ref_x=float(pair[0]),
            ref_y=float(pair[1]),
            off_x=float(pair[2]),
            off_y=float(pair[3]),
            confidence=float(pair[4]),
            name=str(pair[5]),
        )


@dataclass(slots=True)
class GCP:
    """Typed representation of a correction GCP."""

    col: float
    row: float
    gx: float
    gy: float
    synthetic: bool = False
    source: str = "match"
    name: str = ""

    def to_legacy(self) -> LegacyGCP:
        return (float(self.col), float(self.row), float(self.gx), float(self.gy))

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


def ensure_match_pair(value: MatchPair | Sequence[Any]) -> MatchPair:
    if isinstance(value, MatchPair):
        return value
    return MatchPair.from_legacy(value)


def ensure_gcp(value: GCP | Sequence[Any], *, synthetic: bool = False,
               source: str = "match", name: str = "") -> GCP:
    if isinstance(value, GCP):
        return value
    return GCP.from_legacy(value, synthetic=synthetic, source=source, name=name)


def match_pairs_from_legacy(values: Iterable[MatchPair | Sequence[Any]]) -> list[MatchPair]:
    return [ensure_match_pair(v) for v in values]


def match_pairs_to_legacy(values: Iterable[MatchPair | Sequence[Any]]) -> list[LegacyMatch]:
    return [ensure_match_pair(v).to_legacy() for v in values]


def gcps_from_legacy(values: Iterable[GCP | Sequence[Any]], *, synthetic: bool = False,
                     source: str = "match") -> list[GCP]:
    return [ensure_gcp(v, synthetic=synthetic, source=source) for v in values]


def gcps_to_legacy(values: Iterable[GCP | Sequence[Any]], *, synthetic: bool = False,
                   source: str = "match") -> list[LegacyGCP]:
    return [ensure_gcp(v, synthetic=synthetic, source=source).to_legacy() for v in values]
