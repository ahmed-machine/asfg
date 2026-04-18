"""Per-mission per-quarter altitude report for KH-4 / KH-7 / KH-9.

Walks every mission in ``data/kh_missions.yaml`` that has a TLE file
under ``data/kh_tle/``, samples the orbit at five evenly spaced points
(start, 25 %, 50 %, 75 %, end of the TLE epoch window), and prints a
markdown table per camera series with mean / perigee / apogee altitude.

Usage
-----
    python3 scripts/kh_altitude_report.py
    python3 scripts/kh_altitude_report.py --series KH-9
    python3 scripts/kh_altitude_report.py --output kh_altitudes.md

Altitude math (no skyfield dependency):
    n [rev/day] → n_rad_s = n * 2π / 86400
    a [m]       = (μ / n_rad_s²)^(1/3),  μ = 3.986004418e14
    perigee_m   = a*(1-e) - R_earth
    apogee_m    = a*(1+e) - R_earth
    altitude_m  = a - R_earth          (spherical-earth mean)

Only TLE lines are needed; the output is deterministic and repeatable.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocess.mission_altitude import (
    _parse_tle_file,
    _tle_epoch,
    load_mission_catalog,
)

MU = 3.986004418e14           # Earth GM, m³/s²
R_EARTH = 6378137.0           # WGS-84 equatorial radius, m


def _parse_tle_line2(l2: str) -> tuple[float, float] | None:
    """Return (eccentricity, mean_motion_rev_per_day) from TLE line 2."""
    if len(l2) < 63:
        return None
    try:
        # Eccentricity: 7 digits, implicit leading "0.", cols 27-33 (1-based).
        ecc = float("0." + l2[26:33].strip())
        n = float(l2[52:63])
    except ValueError:
        return None
    if n <= 0.0 or not (0.0 <= ecc < 1.0):
        return None
    return ecc, n


def _sample_altitudes(ecc: float, n_rev_per_day: float) -> tuple[float, float, float]:
    """(perigee_m, mean_m, apogee_m) above WGS-84 sphere."""
    n_rad_s = n_rev_per_day * 2.0 * math.pi / 86400.0
    a = (MU / (n_rad_s ** 2)) ** (1.0 / 3.0)
    perigee_m = a * (1.0 - ecc) - R_EARTH
    apogee_m = a * (1.0 + ecc) - R_EARTH
    mean_m = a - R_EARTH
    return perigee_m, mean_m, apogee_m


def _nearest_tle(triples: list, target: datetime) -> tuple | None:
    """Return (l1, l2, epoch_dt) of the TLE with epoch closest to ``target``."""
    best = None
    best_delta = None
    for name, l1, l2 in triples:
        epoch = _tle_epoch(l1)
        if epoch is None:
            continue
        delta = abs((epoch - target).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best = (l1, l2, epoch)
    return best


def _quarter_samples(start: datetime, end: datetime) -> list[tuple[str, datetime]]:
    """Return [(label, dt), ...] at 0 / 25 / 50 / 75 / 100 % of (end - start)."""
    if end <= start:
        return [("t=start", start)]
    total_s = (end - start).total_seconds()
    out = []
    for frac, label in [(0.0, "Q0_start"), (0.25, "Q1_25%"),
                        (0.50, "Q2_50%"), (0.75, "Q3_75%"),
                        (1.0,  "Q4_end")]:
        dt = start + timedelta(seconds=total_s * frac)
        out.append((label, dt))
    return out


def _load_satcat_sidecar(mid: str, tle_dir: Path) -> dict | None:
    p = tle_dir / f"{mid}.satcat.json"
    if not p.exists():
        return None
    try:
        import json
        return json.loads(p.read_text())
    except Exception:
        return None


def _satcat_row(mid: str, entry: dict, satcat: dict) -> dict | None:
    """Single-sample row synthesised from a satcat entry when gp_history is
    empty. Uses satcat PERIGEE / APOGEE (km above Earth's surface)."""
    try:
        per_km = float(satcat.get("PERIGEE"))
        apo_km = float(satcat.get("APOGEE"))
    except (TypeError, ValueError):
        return None
    mean_km = (per_km + apo_km) / 2.0
    a_m = (per_km + apo_km) / 2.0 * 1000.0 + R_EARTH
    ecc = (apo_km - per_km) * 1000.0 / (2.0 * a_m)
    n_rad_s = math.sqrt(MU / a_m ** 3)
    n_rev_per_day = n_rad_s * 86400.0 / (2.0 * math.pi)
    launch_raw = satcat.get("LAUNCH") or entry.get("launch_date")
    decay_raw = satcat.get("DECAY")
    try:
        launch_dt = datetime.fromisoformat(str(launch_raw)).replace(tzinfo=timezone.utc)
    except Exception:
        launch_dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
    try:
        decay_dt = datetime.fromisoformat(str(decay_raw)).replace(tzinfo=timezone.utc)
    except Exception:
        decay_dt = launch_dt
    return {
        "mid": mid,
        "sample": "satcat",
        "sample_dt": launch_dt,
        "tle_epoch": decay_dt,
        "perigee_km": per_km,
        "mean_km": mean_km,
        "apogee_km": apo_km,
        "ecc": ecc,
        "n_rev_per_day": n_rev_per_day,
    }


def _report_mission(mid: str, entry: dict, tle_dir: Path) -> tuple[list[dict], str] | None:
    """Return (rows, diagnostic) for a single mission, or None if nothing on disk."""
    fname = entry.get("tle_file") or f"{mid}.tle"
    tle_path = tle_dir / fname
    satcat = _load_satcat_sidecar(mid, tle_dir)
    triples = []
    if tle_path.exists() and tle_path.stat().st_size > 0:
        triples = _parse_tle_file(str(tle_path))
    if not triples:
        if satcat is not None:
            row = _satcat_row(mid, entry, satcat)
            if row is not None:
                return [row], f"satcat only (no TLEs) — decay {satcat.get('DECAY', '?')}"
        return None

    epochs = [_tle_epoch(l1) for _, l1, _ in triples]
    epochs = [e for e in epochs if e is not None]
    if len(epochs) < 2:
        return None
    first = min(epochs)
    last = max(epochs)

    rows = []
    for label, dt in _quarter_samples(first, last):
        nearest = _nearest_tle(triples, dt)
        if nearest is None:
            continue
        l1, l2, tle_epoch = nearest
        parsed = _parse_tle_line2(l2)
        if parsed is None:
            continue
        ecc, n = parsed
        perigee_m, mean_m, apogee_m = _sample_altitudes(ecc, n)
        rows.append({
            "mid": mid,
            "sample": label,
            "sample_dt": dt,
            "tle_epoch": tle_epoch,
            "perigee_km": perigee_m / 1000.0,
            "mean_km": mean_m / 1000.0,
            "apogee_km": apogee_m / 1000.0,
            "ecc": ecc,
            "n_rev_per_day": n,
        })
    diag = (
        f"{len(triples)} TLEs, span "
        f"{first.date().isoformat()} → {last.date().isoformat()} "
        f"({(last - first).days} d)"
    )
    return rows, diag


def _print_series_table(series: str, rows_by_mid: dict[str, list[dict]],
                        diagnostics: dict[str, str], missions: dict) -> str:
    lines = [f"\n## {series}\n"]
    if not rows_by_mid:
        lines.append("_No missions with TLE data in this series._\n")
        return "\n".join(lines)

    header = (
        "| mission | launch      | sample  | sample_date | "
        "perigee_km | mean_km | apogee_km | ecc      | n_rev/day | tle_epoch_date |"
    )
    sep = (
        "|---------|-------------|---------|-------------|"
        "------------|---------|-----------|----------|-----------|----------------|"
    )
    lines.append(header)
    lines.append(sep)
    for mid in sorted(rows_by_mid):
        rows = rows_by_mid[mid]
        launch = (missions.get(mid) or {}).get("launch_date")
        launch_str = str(launch) if launch else "—"
        for r in rows:
            lines.append(
                f"| {r['mid']} | {launch_str:<11s} | {r['sample']:<7s} | "
                f"{r['sample_dt'].date().isoformat()} | "
                f"{r['perigee_km']:>10.1f} | {r['mean_km']:>7.1f} | "
                f"{r['apogee_km']:>9.1f} | {r['ecc']:.5f} | "
                f"{r['n_rev_per_day']:.5f} | "
                f"{r['tle_epoch'].date().isoformat()}  |"
            )
        lines.append(f"|    _{mid} — {diagnostics[mid]}_  ||||||||||\n")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", action="append", default=[],
                    help="Restrict to specific series (repeatable).")
    ap.add_argument("--output", type=Path, default=None,
                    help="Write the report to this file (default: stdout).")
    ap.add_argument("--catalog", type=Path, default=None,
                    help="Override mission catalog path.")
    args = ap.parse_args()

    catalog = load_mission_catalog(str(args.catalog) if args.catalog else None)
    missions = catalog.get("missions") or {}
    tle_dir_raw = catalog.get("tle_dir") or "data/kh_tle"
    tle_dir = Path(tle_dir_raw)
    if not tle_dir.is_absolute():
        tle_dir = REPO_ROOT / tle_dir

    series_filter = set(args.series) if args.series else None
    by_series: dict[str, dict[str, list[dict]]] = {}
    diagnostics: dict[str, str] = {}
    missing: dict[str, list[str]] = {}

    for mid in sorted(missions):
        entry = missions[mid]
        series = entry.get("series") or "unknown"
        if series_filter and series not in series_filter:
            continue
        res = _report_mission(mid, entry, tle_dir)
        if res is None:
            missing.setdefault(series, []).append(mid)
            continue
        rows, diag = res
        by_series.setdefault(series, {})[mid] = rows
        diagnostics[mid] = diag

    out_lines = ["# KH Altitude Report — Per Mission / Per Quarter of Duration\n"]
    out_lines.append(
        f"Source: ``data/kh_missions.yaml`` + TLE files under ``{tle_dir}``. "
        "Altitudes derived from TLE mean motion and eccentricity at five "
        "evenly-spaced samples across the TLE epoch window.\n"
    )
    for series in ["KH-4", "KH-4A", "KH-4B", "KH-7", "KH-9"]:
        if series_filter and series not in series_filter:
            continue
        rows = by_series.get(series, {})
        out_lines.append(
            _print_series_table(series, rows, diagnostics, missions)
        )
        if series in missing and missing[series]:
            mids = ", ".join(missing[series])
            out_lines.append(
                f"\n_Missions without TLE data: {mids}_\n"
            )

    report = "\n".join(out_lines)
    if args.output is not None:
        args.output.write_text(report)
        print(f"Wrote {args.output} ({len(report):,} chars)")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
