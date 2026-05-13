"""Declarative table of mapbh.org tile layers.

Each ``MapbhLayer`` records the layer slug (used in tile URLs), the imagery
year, the camera kind ("kh" for declassified satellite, "modern" for
post-1984 imagery), and an inferred mission tag where possible.

Per project rule (CLAUDE.md): every pre-1984 layer is a KH-mission asset.
Post-1984 is "modern". The build-time discovery script
(``scripts/experimental/lora/build_mapbh_pairs.py``) consults this table for
mission classification — newly added catalogue entries should be appended
here so they're picked up automatically by the pair generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


LayerKind = Literal["kh", "modern"]
Mission = Literal[
    "kh-4", "kh-4a", "kh-4b", "kh-7", "kh-9-mc", "kh-9-pc", "modern", "unknown"
]


@dataclass(frozen=True)
class MapbhLayer:
    slug: str
    year: int
    kind: LayerKind
    mission: Mission
    tile_ext: str = "png"  # most TileServer GL endpoints serve PNG; some are JPG
    native_gsd_m: float | None = None  # nominal ground sample distance (metres)
    notes: str = ""

    @property
    def is_kh(self) -> bool:
        return self.kind == "kh"

    @property
    def is_modern(self) -> bool:
        return self.kind == "modern"


def _infer_mission_from_slug(slug: str, year: int) -> Mission:
    """Best-effort mission tag from slug prefix conventions.

    Conventions documented in memory/cross_kh_system_audit.md and
    memory/pinhole_cam_gen_wiring.md:
        DS{NNNN}        → KH-4 series; NNNN ∈ [9024..9069] = KH-4,
                          ∈ [1023..1052] = KH-4A, ∈ [1101..1117] = KH-4B
        DZB12{xx}       → KH-9 MC (Mapping Camera)
        DZB40{xx}       → KH-7
        D3C{xxxx}       → KH-9 PC (Panoramic Camera)
    """
    if year >= 1984:
        return "modern"

    upper = slug.upper()
    # Look for DS{NNNN} mission number anywhere in the slug.
    # Mission ranges follow the CORONA Atlas convention: KH-4A starts at
    # DS1018 (some references cite 1023; the Atlas treats 1018-1022 as
    # early KH-4A). KH-4B starts at DS1101.
    import re
    m = re.search(r"DS(\d{4})", upper)
    if m:
        n = int(m.group(1))
        if 9024 <= n <= 9069:
            return "kh-4"
        if 1018 <= n <= 1052:
            return "kh-4a"
        if 1101 <= n <= 1117:
            return "kh-4b"
    if re.search(r"DZB12\d{2}", upper):
        return "kh-9-mc"
    if re.search(r"DZB40\d{2}", upper):
        return "kh-7"
    if re.search(r"D3C\d{4}", upper):
        return "kh-9-pc"
    if re.search(r"-KH9-", upper):
        # explicit tag fallback: KH9 in slug w/o a recognised camera suffix
        return "kh-9-mc"

    return "unknown"


# Known SATELLITE imagery layers from the mapbh.org catalogue (verified
# 2026-05). The catalogue contains many additional scanned topographic maps
# and aerial photos (1862-1957 plus various Bahrain / Manama / Riffa
# region maps); those are NOT useful as cross-temporal training pairs for
# a satellite-imagery matcher and are deliberately excluded here.
#
# To extend: re-fetch the catalogue HTML and add any newly-uploaded
# satellite-imagery layers (DS / DZB / D3C / Sentinel / Landsat slugs).
LAYER_TABLE: tuple[MapbhLayer, ...] = (
    MapbhLayer(slug="1965-DS1022-1024DA", year=1965, kind="kh", mission=_infer_mission_from_slug("1965-DS1022-1024DA", 1965)),
    MapbhLayer(slug="1967-Bahrain", year=1967, kind="kh", mission=_infer_mission_from_slug("1967-Bahrain", 1967), notes="mission unverified — slug does not follow DS/DZB/D3C convention"),
    MapbhLayer(slug="1968-DS1104-1057DA", year=1968, kind="kh", mission=_infer_mission_from_slug("1968-DS1104-1057DA", 1968)),
    MapbhLayer(slug="1976-KH9-DZB1212", year=1976, kind="kh", mission=_infer_mission_from_slug("1976-KH9-DZB1212", 1976)),
    MapbhLayer(slug="1982-D3C1217", year=1982, kind="kh", mission=_infer_mission_from_slug("1982-D3C1217", 1982)),
    MapbhLayer(slug="20141112", year=2014, kind="modern", mission="modern"),
    MapbhLayer(slug="20201216", year=2020, kind="modern", mission="modern", tile_ext="jpg"),
)


def get_layer(slug: str) -> MapbhLayer | None:
    for layer in LAYER_TABLE:
        if layer.slug == slug:
            return layer
    return None


def kh_layers() -> tuple[MapbhLayer, ...]:
    return tuple(layer for layer in LAYER_TABLE if layer.is_kh)


def modern_layers() -> tuple[MapbhLayer, ...]:
    return tuple(layer for layer in LAYER_TABLE if layer.is_modern)


def cross_temporal_pairs(*, exclude_missions: Iterable[str] = ()) -> list[tuple[MapbhLayer, MapbhLayer]]:
    """Return all (kh, modern) and (kh_a, kh_b) layer pairs from the table.

    ``exclude_missions`` filters out KH layers whose mission tag matches any
    entry — used (e.g.) to hold out KH-9 if the experiment requires it.
    """
    excluded = set(exclude_missions)
    pairs: list[tuple[MapbhLayer, MapbhLayer]] = []
    kh = [l for l in kh_layers() if l.mission not in excluded]
    mods = list(modern_layers())
    for k in kh:
        for m in mods:
            pairs.append((k, m))
    for i, k1 in enumerate(kh):
        for k2 in kh[i + 1:]:
            pairs.append((k1, k2))
    if len(mods) >= 2:
        pairs.append((mods[0], mods[1]))
    return pairs
