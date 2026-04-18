"""Per-mission altitude prior for KH-4 / KH-7 / KH-9 reconnaissance satellites.

Parses the USGS entity ID, looks the mission up in ``data/kh_missions.yaml``,
propagates a Space-Track TLE with skyfield to find the satellite's altitude
at its closest pass over the scene centre on the acquisition date, and
returns the result. Callers use this as:

  * the ``Zs0`` seed for ASP ``cam_gen`` (see :mod:`preprocess.camera_model`)
  * the ``initial.Zs0`` prior for the 14-parameter LM fit
    (see :mod:`preprocess.kh_panoramic`)

When anything is missing — catalog entry, TLE file, skyfield dependency,
acquisition date, or when the satellite was not overhead on that day — the
function returns ``None`` (or an ``AltitudeResult`` with a ``from_catalog_
nominal`` / ``from_tle_mean`` source) and the caller falls through to its
existing 170 km nominal behaviour. The change is therefore always safe.

TLE files are populated by ``scripts/fetch_kh_tles.py`` (one-shot
Space-Track bootstrap). Without them the module still returns the catalog's
per-mission mean altitude, which for KH-4B's eccentric orbits is already a
much better prior than the universal 170 km nominal.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

__all__ = [
    "MissionRef",
    "AltitudeResult",
    "parse_entity_id",
    "load_mission_catalog",
    "altitude_m_at",
    "catalog_mean_altitude_m",
    "NOMINAL_ALTITUDE_M",
    "Z_S0_MIN_M",
    "Z_S0_MAX_M",
]


NOMINAL_ALTITUDE_M: float = 170_000.0
Z_S0_MIN_M: float = 140_000.0
Z_S0_MAX_M: float = 280_000.0

# Entity-ID formats across USGS declass datasets:
#   * KH-4/4A/4B (corona2):   DS<4-digit-mission>-<frame>      e.g. DS1022-1024DA007
#   * KH-7 (declassii):       DZB00<4-digit-mission><5-digit-ops>H<6-digit-frame>
#                                                              e.g. DZB00403600089H015001 → mission 4036
#   * KH-9 (declassii DZB form, rare): DZB<4-digit-mission>-<frame>  e.g. DZB1212-500104L017
#   * KH-9 panoramic (declassiii): D3C<4-digit-mission>-<frame>  e.g. D3C1213-200346A003
_MISSION_ID_RE = re.compile(
    r"^(?:"
    r"(?P<d3c>D3C)(?P<d3c_mid>\d{4})"              # D3C<mid>...
    r"|(?P<ds>DS)(?P<ds_mid>\d{4})"                # DS<mid>-...
    r"|(?P<dzb7>DZB)00(?P<dzb7_mid>\d{4})\d"       # KH-7: DZB00<mid><ops...>
    r"|(?P<dzb9>DZB)(?P<dzb9_mid>\d{4})-"          # KH-9 DZB: DZB<mid>-...
    r")"
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_CATALOG = os.path.join(_REPO_ROOT, "data", "kh_missions.yaml")
_DEFAULT_TLE_DIR = os.path.join(_REPO_ROOT, "data", "kh_tle")


@dataclass(frozen=True)
class MissionRef:
    system: str           # "KH-4" | "KH-4A" | "KH-4B" | "KH-7" | "KH-9"
    mission_id: str       # "1212", "1022", "4011", ...
    frame: str            # suffix after the first '-'


@dataclass(frozen=True)
class AltitudeResult:
    altitude_m: float
    source: str           # from_tle_at_closest_pass | from_tle_mean
                          # | from_catalog_nominal | unavailable
    tle_epoch_utc: Optional[str] = None
    subpoint_distance_km: Optional[float] = None


def _series_from_mission_id(prefix: str, mission_id: str) -> Optional[str]:
    """Derive the camera series from the USGS prefix + numeric mission ID.

    Mission-ID ranges (public record):
      * KH-4   : 9001-9018  (CORONA, 1962-1963)
      * KH-4A  : 1001-1052  (1963-1969)
      * KH-4B  : 1101-1117  (1967-1972)
      * KH-7   : 4001-4038  (GAMBIT, 1963-1967)
      * KH-9   : 1201-1220  (HEXAGON PC + Global Cam, 1971-1986)
    """
    try:
        mid = int(mission_id)
    except ValueError:
        return None
    if prefix == "D3C":
        return "KH-9"
    if prefix == "DZB":
        if 1201 <= mid <= 1220:
            return "KH-9"         # KH-9 Global Camera (Declass-II)
        if 4001 <= mid <= 4099:
            return "KH-7"
        return None
    if prefix == "DS":
        if 9000 <= mid <= 9099:
            return "KH-4"
        if 1001 <= mid <= 1099:
            return "KH-4A"
        if 1101 <= mid <= 1199:
            return "KH-4B"
        return None
    return None


def parse_entity_id(entity_id: str) -> Optional[MissionRef]:
    """Extract (system, mission_id, frame) from a USGS entity identifier.

    Examples
    --------
    >>> parse_entity_id("D3C1213-200346A003")
    MissionRef(system='KH-9', mission_id='1213', frame='200346A003')
    >>> parse_entity_id("DS1022-1024DA007")
    MissionRef(system='KH-4A', mission_id='1022', frame='1024DA007')
    >>> parse_entity_id("DZB1212-500104L017")
    MissionRef(system='KH-9', mission_id='1212', frame='500104L017')
    >>> parse_entity_id("DZB00403600089H015001")
    MissionRef(system='KH-7', mission_id='4036', frame='00089H015001')
    >>> parse_entity_id("unknown") is None
    True
    """
    if not entity_id:
        return None
    m = _MISSION_ID_RE.match(entity_id.strip())
    if not m:
        return None
    if m.group("d3c"):
        prefix, mid = "D3C", m.group("d3c_mid")
    elif m.group("ds"):
        prefix, mid = "DS", m.group("ds_mid")
    elif m.group("dzb7"):
        prefix, mid = "DZB", m.group("dzb7_mid")
    elif m.group("dzb9"):
        prefix, mid = "DZB", m.group("dzb9_mid")
    else:
        return None
    series = _series_from_mission_id(prefix, mid)
    if series is None:
        return None
    rest = entity_id[m.end():]
    frame = rest.lstrip("-")
    return MissionRef(system=series, mission_id=mid, frame=frame)


@lru_cache(maxsize=4)
def load_mission_catalog(path: Optional[str] = None) -> dict:
    """Load ``data/kh_missions.yaml`` once; cache the result.

    Returns an empty dict (not an error) when the catalog is missing — the
    downstream fallback to ``NOMINAL_ALTITUDE_M`` remains valid.
    """
    catalog_path = path or _DEFAULT_CATALOG
    if not os.path.isfile(catalog_path):
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        with open(catalog_path, "r") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as e:
        print(f"  [mission_altitude] Failed to load {catalog_path}: {e}")
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _lookup_mission(catalog: dict, mission_id: str) -> Optional[dict]:
    missions = catalog.get("missions") or {}
    entry = missions.get(mission_id) or missions.get(str(mission_id))
    if entry is None:
        return None
    if not isinstance(entry, dict):
        return None
    return entry


def _series_defaults(catalog: dict, series: Optional[str]) -> dict:
    if not series:
        return {}
    defaults = (catalog.get("series_defaults") or {}).get(series) or {}
    return defaults if isinstance(defaults, dict) else {}


def _catalog_nominal_altitude_m(
    catalog: dict, mission_entry: Optional[dict], series: Optional[str]
) -> Optional[float]:
    for src in (mission_entry or {}, _series_defaults(catalog, series)):
        km = src.get("nominal_altitude_km") if isinstance(src, dict) else None
        if km is None and isinstance(src, dict):
            per = src.get("perigee_km")
            apo = src.get("apogee_km")
            if per is not None and apo is not None:
                km = (float(per) + float(apo)) / 2.0
        if km is not None:
            try:
                return float(km) * 1000.0
            except (TypeError, ValueError):
                continue
    return None


def catalog_mean_altitude_m(
    mission_id: str,
    catalog_path: Optional[str] = None,
) -> Optional[float]:
    """Return ``(perigee + apogee) / 2`` for ``mission_id`` in metres.

    Phase 3c tiebreak candidate: when both cam_gen and TLE land in a
    wrong-altitude basin, catalog mean gives a third candidate grounded
    in the mission's published orbital extremes rather than in a single
    orbital pass. Falls through:

        1. mission entry's ``perigee_km`` + ``apogee_km``
        2. series defaults' ``perigee_km`` + ``apogee_km``
        3. None (no data)

    Unlike :func:`_catalog_nominal_altitude_m`, this function does NOT
    consult ``nominal_altitude_km`` — the point of the mean is to have
    a candidate distinct from the nominal seed and centred in the real
    orbit's altitude range.
    """
    if not mission_id:
        return None
    catalog = load_mission_catalog(catalog_path)
    if not catalog:
        return None
    mission_entry = _lookup_mission(catalog, mission_id)
    series = (mission_entry or {}).get("series")
    for src in (mission_entry or {}, _series_defaults(catalog, series)):
        if not isinstance(src, dict):
            continue
        per = src.get("perigee_km")
        apo = src.get("apogee_km")
        if per is None or apo is None:
            continue
        try:
            return (float(per) + float(apo)) / 2.0 * 1000.0
        except (TypeError, ValueError):
            continue
    return None


def _tle_path(catalog: dict, mission_entry: dict, mission_id: str) -> str:
    tle_dir = catalog.get("tle_dir")
    if not tle_dir:
        tle_dir = _DEFAULT_TLE_DIR
    elif not os.path.isabs(tle_dir):
        tle_dir = os.path.join(_REPO_ROOT, tle_dir)
    fname = mission_entry.get("tle_file") or f"{mission_id}.tle"
    return os.path.join(tle_dir, fname)


def _parse_tle_file(tle_path: str) -> list:
    """Return a list of (name, line1, line2) triples from a TLE file.

    Tolerant of comment lines / Space-Track ToU banners (anything that
    doesn't start with "1 " or "2 " is treated as a name line).
    """
    triples: list = []
    if not os.path.isfile(tle_path):
        return triples
    with open(tle_path, "r") as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    i = 0
    name: Optional[str] = None
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            triples.append((name or "UNNAMED", ln, lines[i + 1]))
            i += 2
            name = None
            continue
        if ln.strip():
            name = ln.strip()
        i += 1
    return triples


def _tle_epoch(line1: str) -> Optional[datetime]:
    """Parse the epoch field (positions 19-32) of a TLE line 1 as UTC."""
    if len(line1) < 32:
        return None
    try:
        yy = int(line1[18:20])
        day = float(line1[20:32])
    except ValueError:
        return None
    year = 2000 + yy if yy < 57 else 1900 + yy
    base = datetime(year, 1, 1, tzinfo=timezone.utc)
    return base + timedelta(days=day - 1.0)


def _pick_nearest_tle(triples: list, acq_dt: datetime) -> Optional[tuple]:
    """Return the (name, l1, l2, epoch) tuple closest in epoch to acq_dt."""
    best = None
    best_delta = None
    for name, l1, l2 in triples:
        epoch = _tle_epoch(l1)
        if epoch is None:
            continue
        delta = abs((epoch - acq_dt).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best = (name, l1, l2, epoch)
    return best


def _gc_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2.0 * 6371.0088 * math.asin(min(1.0, math.sqrt(a)))


def _propagate_closest_pass(
    triples: list,
    acq_date: date,
    lat: float,
    lon: float,
    search_window_hours: float,
) -> Optional[tuple]:
    """Return (altitude_m, tle_epoch_iso, subpoint_distance_km) at the
    satellite's minimum-subpoint-distance moment on the acquisition date,
    or ``None`` if skyfield or the TLE is unavailable / unusable.
    """
    try:
        from skyfield.api import EarthSatellite, load, wgs84
    except ImportError:
        return None
    if not triples:
        return None

    acq_utc = datetime(
        acq_date.year, acq_date.month, acq_date.day, 12, 0, 0, tzinfo=timezone.utc
    )
    nearest = _pick_nearest_tle(triples, acq_utc)
    if nearest is None:
        return None
    name, l1, l2, epoch = nearest

    ts = load.timescale()
    try:
        sat = EarthSatellite(l1, l2, name, ts)
    except Exception:
        return None

    window = max(60.0, float(search_window_hours)) * 3600.0
    # Coarse 30 s scan, then 1 s refinement around the minimum.
    coarse_step = 30.0
    n_steps = int(window / coarse_step) + 1
    start = acq_utc - timedelta(seconds=window / 2.0)
    best_i = -1
    best_d = float("inf")
    for i in range(n_steps):
        t = ts.from_datetime(start + timedelta(seconds=i * coarse_step))
        # geographic_position_of returns the satellite's geodetic position
        # (lat, lon, elevation-above-ellipsoid). subpoint_of returns the
        # ground point under the satellite — elevation would be zero.
        pos = wgs84.geographic_position_of(sat.at(t))
        d = _gc_distance_km(lat, lon, pos.latitude.degrees, pos.longitude.degrees)
        if d < best_d:
            best_d = d
            best_i = i
    if best_i < 0:
        return None

    # Fine search ±30 s around coarse best at 1 s resolution.
    fine_center = start + timedelta(seconds=best_i * coarse_step)
    fine_best_d = best_d
    fine_best_t = fine_center
    fine_best_alt_m = None
    for j in range(-30, 31):
        t_dt = fine_center + timedelta(seconds=j)
        t = ts.from_datetime(t_dt)
        pos = wgs84.geographic_position_of(sat.at(t))
        d = _gc_distance_km(lat, lon, pos.latitude.degrees, pos.longitude.degrees)
        if d < fine_best_d:
            fine_best_d = d
            fine_best_t = t_dt
            fine_best_alt_m = float(pos.elevation.m)
    if fine_best_alt_m is None:
        t = ts.from_datetime(fine_center)
        pos = wgs84.geographic_position_of(sat.at(t))
        fine_best_alt_m = float(pos.elevation.m)

    epoch_iso = epoch.isoformat().replace("+00:00", "Z")
    return (fine_best_alt_m, epoch_iso, fine_best_d)


def _tle_mean_altitude_m(triples: list, acq_date: date) -> Optional[float]:
    """Derive mean altitude from the nearest TLE's mean motion (n).

    h_mean = (mu / (2*pi*n / 86400)^2)^(1/3) - R_earth
    """
    import math
    if not triples:
        return None
    acq_utc = datetime(
        acq_date.year, acq_date.month, acq_date.day, 12, 0, 0, tzinfo=timezone.utc
    )
    nearest = _pick_nearest_tle(triples, acq_utc)
    if nearest is None:
        return None
    _, _, l2, _ = nearest
    if len(l2) < 63:
        return None
    try:
        n_rev_per_day = float(l2[52:63])
    except ValueError:
        return None
    if n_rev_per_day <= 0.0:
        return None
    mu = 3.986004418e14
    n_rad_s = n_rev_per_day * 2.0 * math.pi / 86400.0
    a = (mu / (n_rad_s ** 2)) ** (1.0 / 3.0)
    R_earth = 6378137.0
    return a - R_earth


def altitude_m_at(
    mission_id: str,
    acq_date: Optional[date],
    lat_deg: Optional[float],
    lon_deg: Optional[float],
    search_window_hours: float = 24.0,
    catalog_path: Optional[str] = None,
) -> Optional[AltitudeResult]:
    """Resolve per-mission altitude (metres above ellipsoid) for a frame.

    Fallback chain:
        1. TLE propagated to satellite's closest pass over (lat, lon) on
           ``acq_date`` → ``from_tle_at_closest_pass``.
        2. TLE mean-motion-derived mean altitude → ``from_tle_mean``
           (used when satellite was not overhead that day, e.g. coarse TLE
           or catalog/date mismatch).
        3. Catalog nominal / ``(perigee+apogee)/2`` → ``from_catalog_nominal``.
        4. ``None`` — mission unknown or no data.

    The returned altitude is always clamped to ``[140, 280]`` km; if the
    clamp activates, the source is downgraded to ``from_catalog_nominal``.
    """
    if not mission_id:
        return None
    catalog = load_mission_catalog(catalog_path)
    mission_entry = _lookup_mission(catalog, mission_id)
    if mission_entry is None and not catalog:
        return None

    series = (mission_entry or {}).get("series")
    catalog_nominal_m = _catalog_nominal_altitude_m(catalog, mission_entry, series)

    tle_triples: list = []
    if mission_entry is not None:
        tle_path = _tle_path(catalog, mission_entry, mission_id)
        tle_triples = _parse_tle_file(tle_path)

    # Need both an acquisition date and a scene centre to propagate.
    can_propagate = (
        acq_date is not None
        and lat_deg is not None
        and lon_deg is not None
        and tle_triples
    )

    # Respect a sparse-TLE hint if the catalog flags it.
    sparse = bool((mission_entry or {}).get("tle_quality") == "sparse")

    if can_propagate and not sparse:
        pass_res = _propagate_closest_pass(
            tle_triples, acq_date, float(lat_deg), float(lon_deg),
            search_window_hours,
        )
        if pass_res is not None:
            alt_m, epoch_iso, d_km = pass_res
            clamped = _clamp(alt_m)
            if clamped is None:
                return AltitudeResult(
                    altitude_m=catalog_nominal_m or NOMINAL_ALTITUDE_M,
                    source="from_catalog_nominal",
                    tle_epoch_utc=epoch_iso,
                    subpoint_distance_km=d_km,
                )
            if d_km > 500.0:
                # Satellite was not overhead → distrust per-time altitude.
                mean_m = _tle_mean_altitude_m(tle_triples, acq_date)
                mean_clamped = _clamp(mean_m) if mean_m is not None else None
                if mean_clamped is not None:
                    return AltitudeResult(
                        altitude_m=mean_clamped,
                        source="from_tle_mean",
                        tle_epoch_utc=epoch_iso,
                        subpoint_distance_km=d_km,
                    )
            else:
                return AltitudeResult(
                    altitude_m=clamped,
                    source="from_tle_at_closest_pass",
                    tle_epoch_utc=epoch_iso,
                    subpoint_distance_km=d_km,
                )

    if tle_triples and acq_date is not None:
        mean_m = _tle_mean_altitude_m(tle_triples, acq_date)
        mean_clamped = _clamp(mean_m) if mean_m is not None else None
        if mean_clamped is not None:
            return AltitudeResult(
                altitude_m=mean_clamped,
                source="from_tle_mean",
                tle_epoch_utc=None,
            )

    if catalog_nominal_m is not None:
        clamped = _clamp(catalog_nominal_m) or catalog_nominal_m
        return AltitudeResult(
            altitude_m=clamped,
            source="from_catalog_nominal",
        )

    return None


def _clamp(alt_m: float) -> Optional[float]:
    if alt_m is None:
        return None
    try:
        a = float(alt_m)
    except (TypeError, ValueError):
        return None
    if Z_S0_MIN_M <= a <= Z_S0_MAX_M:
        return a
    return None
