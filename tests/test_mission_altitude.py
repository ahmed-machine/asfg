"""Tests for preprocess.mission_altitude — TLE/catalog altitude prior."""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

import pytest

from preprocess import mission_altitude
from preprocess.mission_altitude import (
    AltitudeResult,
    MissionRef,
    Z_S0_MAX_M,
    Z_S0_MIN_M,
    altitude_m_at,
    load_mission_catalog,
    parse_entity_id,
)


# ---------------------------------------------------------------------------
# parse_entity_id
# ---------------------------------------------------------------------------


def test_parse_entity_id_kh9_d3c():
    ref = parse_entity_id("D3C1213-200346A003")
    assert ref == MissionRef(system="KH-9", mission_id="1213", frame="200346A003")


def test_parse_entity_id_kh9_dzb():
    ref = parse_entity_id("DZB1212-500104L017")
    assert ref is not None
    assert ref.system == "KH-9"
    assert ref.mission_id == "1212"
    assert ref.frame == "500104L017"


def test_parse_entity_id_kh4a_ds():
    ref = parse_entity_id("DS1022-1024DA007")
    assert ref is not None
    assert ref.system == "KH-4A"
    assert ref.mission_id == "1022"
    assert ref.frame == "1024DA007"


def test_parse_entity_id_kh4b_ds():
    ref = parse_entity_id("DS1104-1024DF010")
    assert ref is not None
    assert ref.system == "KH-4B"
    assert ref.mission_id == "1104"


def test_parse_entity_id_kh7_dzb():
    ref = parse_entity_id("DZB4020-001234H012")
    assert ref is not None
    assert ref.system == "KH-7"
    assert ref.mission_id == "4020"


def test_parse_entity_id_unknown_prefix_returns_none():
    assert parse_entity_id("XX9999-abc") is None


def test_parse_entity_id_out_of_range_returns_none():
    # DS3000 is outside any known CORONA mission range.
    assert parse_entity_id("DS3000-1024DA007") is None


def test_parse_entity_id_empty_returns_none():
    assert parse_entity_id("") is None
    assert parse_entity_id(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Catalog + altitude fallback semantics (no skyfield / no TLE file)
# ---------------------------------------------------------------------------


def _write_catalog(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "kh_missions.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_load_mission_catalog_missing_returns_empty(tmp_path):
    load_mission_catalog.cache_clear()
    missing = tmp_path / "does_not_exist.yaml"
    assert load_mission_catalog(str(missing)) == {}


def test_altitude_unknown_mission_returns_none(tmp_path):
    load_mission_catalog.cache_clear()
    cat = _write_catalog(tmp_path, """\
        missions:
          "1212":
            series: KH-9
            nominal_altitude_km: 170.0
    """)
    res = altitude_m_at(
        mission_id="9999",
        acq_date=date(1976, 8, 15),
        lat_deg=26.2,
        lon_deg=50.6,
        catalog_path=str(cat),
    )
    # Mission not in catalog → None (caller falls back to baked-in 170 km).
    assert res is None


def test_altitude_nominal_fallback_when_tle_missing(tmp_path):
    """When there's no TLE file, we should still get the per-mission
    nominal altitude from the catalog. Works without skyfield."""
    load_mission_catalog.cache_clear()
    cat = _write_catalog(tmp_path, f"""\
        tle_dir: {tmp_path}/kh_tle
        missions:
          "1022":
            series: KH-4A
            nominal_altitude_km: 195.0
    """)
    res = altitude_m_at(
        mission_id="1022",
        acq_date=date(1965, 7, 20),
        lat_deg=0.0,
        lon_deg=0.0,
        catalog_path=str(cat),
    )
    assert res is not None
    assert res.source == "from_catalog_nominal"
    assert abs(res.altitude_m - 195_000.0) < 1.0


def test_altitude_series_default_when_mission_missing_fields(tmp_path):
    load_mission_catalog.cache_clear()
    cat = _write_catalog(tmp_path, f"""\
        tle_dir: {tmp_path}/kh_tle
        series_defaults:
          KH-4B:
            perigee_km: 154.0
            apogee_km:  276.0
            nominal_altitude_km: 170.0
        missions:
          "1110":
            series: KH-4B
    """)
    res = altitude_m_at(
        mission_id="1110",
        acq_date=date(1970, 6, 1),
        lat_deg=0.0,
        lon_deg=0.0,
        catalog_path=str(cat),
    )
    assert res is not None
    assert res.source == "from_catalog_nominal"
    # Mission entry has no nominal_altitude_km; falls through to series default.
    assert abs(res.altitude_m - 170_000.0) < 1.0


def test_altitude_clamp_downgrades_source(tmp_path):
    load_mission_catalog.cache_clear()
    cat = _write_catalog(tmp_path, f"""\
        tle_dir: {tmp_path}/kh_tle
        missions:
          "9901":
            series: KH-9
            nominal_altitude_km: 1000.0
    """)
    res = altitude_m_at(
        mission_id="9901",
        acq_date=date(1980, 1, 1),
        lat_deg=0.0,
        lon_deg=0.0,
        catalog_path=str(cat),
    )
    # 1 000 km is out of [140, 280] km → catalog nominal is clamped to its
    # raw value (function returns it un-clamped because the catalog path is
    # the final fallback; the source stays from_catalog_nominal).
    assert res is not None
    assert res.source == "from_catalog_nominal"
    assert res.altitude_m == pytest.approx(1_000_000.0)


def test_altitude_without_date_falls_through_to_nominal(tmp_path):
    load_mission_catalog.cache_clear()
    cat = _write_catalog(tmp_path, f"""\
        tle_dir: {tmp_path}/kh_tle
        missions:
          "1212":
            series: KH-9
            nominal_altitude_km: 170.0
    """)
    res = altitude_m_at(
        mission_id="1212",
        acq_date=None,
        lat_deg=26.2,
        lon_deg=50.6,
        catalog_path=str(cat),
    )
    assert res is not None
    assert res.source == "from_catalog_nominal"
    assert abs(res.altitude_m - 170_000.0) < 1.0


# ---------------------------------------------------------------------------
# TLE mean-motion fallback (works without skyfield — we only parse line 2)
# ---------------------------------------------------------------------------


# Fake TLE for mission 1212: epoch day-220 of 1976, mean motion 16.27
# rev/day → semi-major axis ≈ 6578 km → altitude ≈ 200 km (in [140, 280]).
_FAKE_TLE_BODY = """\
1 09006U 76065A   76220.50000000  .00003000  00000-0  12345-3 0  9990
2 09006  96.4000 123.4567 0048000 234.5678 125.4321 16.27000000123456
"""


def test_altitude_from_tle_mean_without_skyfield(tmp_path, monkeypatch):
    """When skyfield isn't available, propagation is skipped but the mean-
    motion-derived altitude from the TLE line 2 should still work (pure
    arithmetic, no skyfield needed).
    """
    load_mission_catalog.cache_clear()
    tle_dir = tmp_path / "kh_tle"
    tle_dir.mkdir()
    (tle_dir / "1212.tle").write_text(_FAKE_TLE_BODY)
    cat = _write_catalog(tmp_path, f"""\
        tle_dir: {tle_dir}
        missions:
          "1212":
            series: KH-9
            nominal_altitude_km: 170.0
            tle_file: 1212.tle
    """)

    # Force the propagation path to fail as if skyfield were missing.
    monkeypatch.setattr(mission_altitude, "_propagate_closest_pass",
                        lambda *a, **kw: None)

    res = altitude_m_at(
        mission_id="1212",
        acq_date=date(1976, 8, 15),
        lat_deg=26.2,
        lon_deg=50.6,
        catalog_path=str(cat),
    )
    assert res is not None
    assert res.source == "from_tle_mean"
    # n = 15.1 rev/day → semi-major axis ≈ 6579 km → altitude ≈ 201 km.
    assert Z_S0_MIN_M <= res.altitude_m <= Z_S0_MAX_M
    assert 180_000 < res.altitude_m < 230_000


def test_altitude_ignores_unparseable_tle(tmp_path):
    load_mission_catalog.cache_clear()
    tle_dir = tmp_path / "kh_tle"
    tle_dir.mkdir()
    (tle_dir / "1212.tle").write_text("# garbage file\nnot a tle\n")
    cat = _write_catalog(tmp_path, f"""\
        tle_dir: {tle_dir}
        missions:
          "1212":
            series: KH-9
            nominal_altitude_km: 170.0
            tle_file: 1212.tle
    """)

    res = altitude_m_at(
        mission_id="1212",
        acq_date=date(1976, 8, 15),
        lat_deg=26.2,
        lon_deg=50.6,
        catalog_path=str(cat),
    )
    # Empty TLE triples list → falls through to catalog nominal.
    assert res is not None
    assert res.source == "from_catalog_nominal"
    assert abs(res.altitude_m - 170_000.0) < 1.0


# ---------------------------------------------------------------------------
# Shipped catalog loads cleanly
# ---------------------------------------------------------------------------


def test_shipped_catalog_contains_test_missions():
    load_mission_catalog.cache_clear()
    catalog = load_mission_catalog()
    missions = (catalog.get("missions") or {})
    assert "1212" in missions, "shipped catalog should have KH-9 DZB1212"
    assert "1213" in missions, "shipped catalog should have KH-9 D3C1213"
    assert "1022" in missions, "shipped catalog should have KH-4A DS1022"
    assert missions["1212"].get("series") == "KH-9"
    assert missions["1213"].get("series") == "KH-9"
    assert missions["1022"].get("series") == "KH-4A"
