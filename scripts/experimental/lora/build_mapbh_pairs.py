"""Phase 2.2 — Build cross-temporal training pairs from map.mapbh.org.

For an ROI polygon (default ``data/bahrain_boundary.geojson``) and the
satellite-imagery layers in ``preprocess/mapbh/layers.py::LAYER_TABLE``, this
script generates random crops at random spans (2, 5, or 10 km) and fetches
the same crop from two layers — every (KH, modern) pair plus every
(KH, KH) cross-mission/cross-era pair plus a small (modern, modern) sanity
slice.

Each pair is saved as:
  pair_NNNNN_ref.png      — modern layer (or "newer" KH layer in cross-era)
  pair_NNNNN_src_clean.png — historical KH layer (or "older" cross-era)
  pair_NNNNN_meta.json    — bbox, layer slugs, mission tags, span_km, kind

Pairs do NOT receive synthetic affine perturbation — the cross-temporal
change between the two layers IS the supervision signal. The downstream
``extract_mapbh_pseudo_labels.py`` produces correspondence labels by
running frozen base RoMa on each pair.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class CropSpec:
    layer_src: str
    layer_ref: str
    centre_lon: float
    centre_lat: float
    span_km: float
    zoom: int

    def bbox(self) -> tuple[float, float, float, float]:
        # Approximate degrees per km at this latitude
        deg_lat = self.span_km / 110.574
        deg_lon = self.span_km / (111.320 * np.cos(np.radians(self.centre_lat)))
        half_lat = deg_lat / 2
        half_lon = deg_lon / 2
        return (
            self.centre_lon - half_lon,
            self.centre_lat - half_lat,
            self.centre_lon + half_lon,
            self.centre_lat + half_lat,
        )


def _zoom_for_span(span_km: float) -> int:
    """Pick a zoom that gives roughly constant ~25 tiles per crop."""
    if span_km <= 2.5:
        return 16
    if span_km <= 6.0:
        return 15
    return 14


def _read_polygon(geojson_path: Path) -> tuple[float, float, float, float, list]:
    """Return (lon_w, lat_s, lon_e, lat_n) bbox + a ring of (lon, lat) points.

    Picks the first Polygon outer ring; the bahrain AOI is rectangular so the
    ring is sufficient for point-in-polygon rejection sampling.
    """
    gj = json.loads(geojson_path.read_text())

    # Find first Polygon (outer ring)
    geom = None
    if gj.get("type") == "FeatureCollection":
        for feat in gj["features"]:
            if feat["geometry"]["type"] == "Polygon":
                geom = feat["geometry"]
                break
    elif gj.get("type") == "Feature":
        if gj["geometry"]["type"] == "Polygon":
            geom = gj["geometry"]
    elif gj.get("type") == "Polygon":
        geom = gj
    if geom is None:
        raise ValueError(f"No Polygon found in {geojson_path}")

    ring = geom["coordinates"][0]  # outer ring
    lons = [pt[0] for pt in ring]
    lats = [pt[1] for pt in ring]
    return min(lons), min(lats), max(lons), max(lats), ring


def _point_in_ring(lon: float, lat: float, ring: list) -> bool:
    """Standard ray-casting point-in-polygon test."""
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > lat) != (y2 > lat):
            x_cross = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x_cross:
                inside = not inside
    return inside


def _random_crop(rng: random.Random, lon_w: float, lat_s: float, lon_e: float, lat_n: float, ring: list) -> tuple[float, float, float]:
    """Sample a random (centre_lon, centre_lat, span_km) inside the polygon."""
    for _ in range(200):
        lon = rng.uniform(lon_w, lon_e)
        lat = rng.uniform(lat_s, lat_n)
        if _point_in_ring(lon, lat, ring):
            span_km = rng.choice([2.0, 5.0, 10.0])
            return lon, lat, span_km
    raise RuntimeError("rejection sampling failed; ROI polygon may be empty")


def _enumerate_layer_pairs() -> list[tuple]:
    """Build the (src, ref) layer-pair list from LAYER_TABLE.

    Returns a list of (src_layer, ref_layer, kind) tuples where kind is one
    of: 'kh-modern', 'kh-kh', 'modern-modern'.
    """
    from preprocess.mapbh.layers import kh_layers, modern_layers

    kh = list(kh_layers())
    mods = list(modern_layers())

    pairs = []
    # KH ↔ modern (always: src=KH, ref=modern)
    for k in kh:
        for m in mods:
            pairs.append((k, m, "kh-modern"))
    # KH ↔ KH cross-mission/era (src is the older year)
    for i, k1 in enumerate(kh):
        for k2 in kh[i + 1:]:
            if k1.year <= k2.year:
                pairs.append((k1, k2, "kh-kh"))
            else:
                pairs.append((k2, k1, "kh-kh"))
    # Modern ↔ modern sanity slice (just the first ordered pair)
    if len(mods) >= 2:
        a, b = sorted(mods, key=lambda m: m.year)
        pairs.append((a, b, "modern-modern"))
    return pairs


def _next_pair_id(out_dir: Path) -> int:
    existing = sorted(out_dir.glob("pair_*_meta.json"))
    if not existing:
        return 0
    last = existing[-1].name
    n = int(last[len("pair_"):len("pair_") + 5])
    return n + 1


def _save_png(arr: np.ndarray, path: Path) -> None:
    import cv2
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def main() -> None:
    p = argparse.ArgumentParser(description="Build cross-temporal training pairs from mapbh.org")
    p.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "lora_pairs")
    p.add_argument("--aoi", type=Path, default=REPO_ROOT / "data" / "bahrain_boundary.geojson")
    p.add_argument("--target-pairs", type=int, default=1500)
    p.add_argument("--out-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", type=Path, default=Path("~/.cache/declass-process/mapbh"))
    p.add_argument("--max-attempts-multiplier", type=float, default=3.0,
                   help="Stop sampling after target_pairs * this many attempts (rejection cap)")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    from preprocess.mapbh.client import MapbhClient

    client = MapbhClient(cache_dir=args.cache_dir)
    rng = random.Random(args.seed)

    lon_w, lat_s, lon_e, lat_n, ring = _read_polygon(args.aoi)
    print(f"[build_mapbh_pairs] AOI bbox = ({lon_w:.3f}, {lat_s:.3f}, {lon_e:.3f}, {lat_n:.3f})")

    layer_pairs = _enumerate_layer_pairs()
    # Build a weighted list so we get roughly the targets in the plan
    weighted = []
    for src, ref, kind in layer_pairs:
        if kind == "kh-modern":
            weighted += [(src, ref, kind)] * 4
        elif kind == "kh-kh":
            weighted += [(src, ref, kind)] * 2
        else:  # modern-modern: rare sanity slice
            weighted += [(src, ref, kind)] * 1
    print(f"[build_mapbh_pairs] {len(layer_pairs)} layer-pair specs, weighted to {len(weighted)} buckets")

    pair_id = _next_pair_id(args.output)
    n_built = 0
    n_skipped_coverage = 0
    n_skipped_fetch = 0
    counts_by_kind: dict[str, int] = {"kh-modern": 0, "kh-kh": 0, "modern-modern": 0}
    counts_by_pair: dict[tuple[str, str], int] = {}
    t0 = time.time()
    target = args.target_pairs
    max_attempts = int(target * args.max_attempts_multiplier)
    attempts = 0

    while n_built < target and attempts < max_attempts:
        attempts += 1
        src_layer, ref_layer, kind = rng.choice(weighted)
        try:
            lon, lat, span_km = _random_crop(rng, lon_w, lat_s, lon_e, lat_n, ring)
        except RuntimeError as e:
            print(f"[build_mapbh_pairs] {e}")
            break
        zoom = _zoom_for_span(span_km)
        spec = CropSpec(
            layer_src=src_layer.slug, layer_ref=ref_layer.slug,
            centre_lon=lon, centre_lat=lat, span_km=span_km, zoom=zoom,
        )
        bbox = spec.bbox()

        # Coverage probe — sample one tile per layer at the bbox centre
        from preprocess.mapbh.client import lonlat_to_tile
        tx, ty = lonlat_to_tile(lon, lat, zoom)
        if not client.probe_tile(src_layer.slug, zoom, tx, ty):
            n_skipped_coverage += 1
            continue
        if not client.probe_tile(ref_layer.slug, zoom, tx, ty):
            n_skipped_coverage += 1
            continue

        # Fetch both layers
        try:
            src_arr, src_bbox_3857 = client.fetch_bbox(src_layer.slug, bbox, zoom, (args.out_size, args.out_size))
            ref_arr, ref_bbox_3857 = client.fetch_bbox(ref_layer.slug, bbox, zoom, (args.out_size, args.out_size))
        except Exception as e:
            print(f"[build_mapbh_pairs] fetch failed for {src_layer.slug}↔{ref_layer.slug}: {e}")
            n_skipped_fetch += 1
            continue

        # Quality gate: reject pairs with mostly-empty fetches (large 0 swathes
        # indicate partial coverage at the crop edge)
        if src_arr.mean() < 5 or ref_arr.mean() < 5:
            n_skipped_coverage += 1
            continue
        if (src_arr == 0).all(axis=2).mean() > 0.2 or (ref_arr == 0).all(axis=2).mean() > 0.2:
            n_skipped_coverage += 1
            continue

        # Save
        stem = f"pair_{pair_id:05d}"
        ref_path = args.output / f"{stem}_ref.png"
        src_path = args.output / f"{stem}_src_clean.png"
        meta_path = args.output / f"{stem}_meta.json"
        _save_png(ref_arr, ref_path)
        _save_png(src_arr, src_path)
        meta = {
            "kind": "mapbh",
            "pair_kind": kind,
            "layer_src": src_layer.slug,
            "layer_ref": ref_layer.slug,
            "mission_src": src_layer.mission,
            "mission_ref": ref_layer.mission,
            "year_src": src_layer.year,
            "year_ref": ref_layer.year,
            "centre_lonlat": [lon, lat],
            "span_km": span_km,
            "zoom": zoom,
            "bbox_lonlat": list(bbox),
            "bbox_3857_src": list(src_bbox_3857),
            "bbox_3857_ref": list(ref_bbox_3857),
            "out_size": args.out_size,
            # No synthetic perturbation; identity affine for compatibility with
            # downstream label loaders.
            "affine_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "georef_noise_m_p50": 15,
            "georef_noise_m_p95": 80,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        n_built += 1
        counts_by_kind[kind] += 1
        bucket = (src_layer.slug, ref_layer.slug)
        counts_by_pair[bucket] = counts_by_pair.get(bucket, 0) + 1
        pair_id += 1

        if n_built % 25 == 0:
            elapsed = time.time() - t0
            rate = n_built / elapsed if elapsed > 0 else 0
            eta = (target - n_built) / rate if rate > 0 else float("inf")
            print(f"[build_mapbh_pairs] {n_built}/{target} built ({attempts} attempts, {n_skipped_coverage} no-coverage, {n_skipped_fetch} fetch-fail) elapsed={elapsed:.0f}s eta={eta:.0f}s")

    summary = {
        "target_pairs": target,
        "n_built": n_built,
        "n_skipped_coverage": n_skipped_coverage,
        "n_skipped_fetch": n_skipped_fetch,
        "n_attempts": attempts,
        "wall_clock_s": time.time() - t0,
        "counts_by_kind": counts_by_kind,
        "counts_by_pair": {f"{a}__{b}": v for (a, b), v in sorted(counts_by_pair.items())},
        "out_size": args.out_size,
        "seed": args.seed,
    }
    summary_path = args.output / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[build_mapbh_pairs] wrote {summary_path}")
    print(f"[build_mapbh_pairs] done: {n_built} pairs built")
    print(f"[build_mapbh_pairs] by kind: {counts_by_kind}")


if __name__ == "__main__":
    main()
