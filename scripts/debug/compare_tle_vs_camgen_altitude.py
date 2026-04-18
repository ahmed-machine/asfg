"""Compare per-strip TLE altitude vs ASP cam_gen altitude.

Iterates over cached ``*_strip_opticalbar.tsai`` files under a segments
directory, parses the ECEF iC → altitude above WGS84 for the cam_gen
result, and calls :func:`preprocess.mission_altitude.altitude_m_at` for
the same (mission, acquisition-date, scene-centre). Prints a per-scene
row and flags disagreements > 10 km.

Read-only: no files are written, no subprocesses are launched.

Usage
-----
    python3 scripts/debug/compare_tle_vs_camgen_altitude.py \\
        --output-dir output/1213-fore \\
        --catalog-csv data/catalogs/declass3.csv

``--output-dir`` is the cache root that contains per-scene segments dirs
(``{scene}/seg/{scene}_strip_opticalbar.tsai``). ``--catalog-csv`` is a
USGS CSV that lists the scenes' acquisition dates + centre lat/lon.
Either can be supplied multiple times.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocess.camera_model import _ecef_to_geodetic, _parse_opticalbar_tsai
from preprocess.catalog import parse_csvs
from preprocess.mission_altitude import altitude_m_at, parse_entity_id


def _find_strip_tsai(output_dir: Path) -> list[tuple[str, Path]]:
    """Return a list of (scene_id, tsai_path) pairs under ``output_dir``."""
    hits: list[tuple[str, Path]] = []
    for tsai in output_dir.rglob("*_strip_opticalbar.tsai"):
        scene = tsai.name.replace("_strip_opticalbar.tsai", "")
        hits.append((scene, tsai))
    return sorted(hits)


def _parse_camgen_altitude_m(tsai_path: Path) -> float | None:
    parsed = _parse_opticalbar_tsai(str(tsai_path))
    if parsed is None:
        return None
    _, _, alt_m = _ecef_to_geodetic(parsed["iC"])
    return float(alt_m)


def _scene_index(csv_paths: list[str]) -> dict[str, object]:
    """Map entity_id → Scene. Uses existing ``preprocess.catalog.parse_csvs``."""
    idx: dict[str, object] = {}
    if not csv_paths:
        return idx
    scenes = parse_csvs([str(p) for p in csv_paths])
    for s in scenes:
        idx[s.entity_id] = s
    return idx


def _parse_date(raw: str):
    from datetime import datetime
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", action="append", default=[],
                    help="Pipeline cache root. May be supplied multiple times.")
    ap.add_argument("--catalog-csv", action="append", default=[],
                    help="USGS EarthExplorer CSV with acquisition dates + centre.")
    ap.add_argument("--warn-km", type=float, default=10.0,
                    help="Flag disagreements >= this many km (default 10).")
    args = ap.parse_args()

    if not args.output_dir:
        print("ERROR: supply at least one --output-dir", file=sys.stderr)
        return 2

    scenes = _scene_index(args.catalog_csv)
    if not scenes:
        print("WARNING: no --catalog-csv supplied; comparison will skip scenes "
              "where the acquisition date is unknown.")

    rows = []
    for root in args.output_dir:
        for scene_id, tsai in _find_strip_tsai(Path(root)):
            cam_alt = _parse_camgen_altitude_m(tsai)
            if cam_alt is None:
                continue
            scene = scenes.get(scene_id)
            acq_date = _parse_date(scene.acquisition_date) if scene else None
            ctr = getattr(scene, "center", None) if scene else None
            lat = lon = None
            if isinstance(ctr, (list, tuple)) and len(ctr) >= 2:
                lat, lon = float(ctr[0]), float(ctr[1])
            mref = parse_entity_id(scene_id)
            tle_alt = None
            source = "no_mission_or_date"
            if mref and acq_date is not None:
                res = altitude_m_at(mref.mission_id, acq_date, lat, lon)
                if res is not None:
                    tle_alt = float(res.altitude_m)
                    source = res.source
            rows.append((scene_id, cam_alt, tle_alt, source))

    if not rows:
        print("No *_strip_opticalbar.tsai files found.")
        return 0

    print(f"{'scene_id':<30} {'cam_gen_km':>10} {'tle_km':>10} {'Δ_km':>7}  source")
    print("-" * 80)
    disagreements = 0
    for scene_id, cam_alt, tle_alt, source in rows:
        cam_s = f"{cam_alt / 1000.0:>10.1f}"
        if tle_alt is None:
            print(f"{scene_id:<30} {cam_s} {'—':>10} {'—':>7}  {source}")
            continue
        delta_km = abs(cam_alt - tle_alt) / 1000.0
        flag = " <=="
        if delta_km < args.warn_km:
            flag = ""
        else:
            disagreements += 1
        print(f"{scene_id:<30} {cam_s} {tle_alt/1000.0:>10.1f} {delta_km:>7.1f}  {source}{flag}")

    print(f"\n{len(rows)} scenes; {disagreements} disagreements >= {args.warn_km} km")
    return 0 if disagreements == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
