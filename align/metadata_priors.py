"""Metadata-prior loading and extraction helpers."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from typing import Any, Iterable

from .types import MetadataPrior


EE_NS = {"ee": "http://earthexplorer.usgs.gov/eemetadata.xsd"}


def parse_bbox_xml(xml_path: str) -> MetadataPrior:
    """Extract a bbox prior from a USGS metadata XML."""

    tree = ET.parse(xml_path)
    root = tree.getroot()
    bounding = root.find(".//spdom/bounding")
    if bounding is None:
        raise ValueError(f"No USGS bounding box found in {xml_path}")
    west = float(bounding.findtext("westbc"))
    north = float(bounding.findtext("northbc"))
    east = float(bounding.findtext("eastbc"))
    south = float(bounding.findtext("southbc"))
    return MetadataPrior(
        source=os.path.basename(xml_path),
        confidence=0.15,
        west=west,
        south=south,
        east=east,
        north=north,
        crs="EPSG:4326",
        center_lon=(west + east) / 2.0,
        center_lat=(south + north) / 2.0,
        attributes={"format": "usgs_bbox_xml"},
    )


def parse_ee_metadata_xml(xml_path: str) -> MetadataPrior:
    """Extract an EarthExplorer corner prior from metadata XML."""

    tree = ET.parse(xml_path)
    root = tree.getroot()
    fields = {}
    for field in root.findall(".//ee:metadataField", EE_NS):
        name = field.get("name")
        value_el = field.find("ee:metadataValue", EE_NS)
        if name and value_el is not None and value_el.text:
            fields[name] = value_el.text.strip()

    corners = {}
    for corner in ("NW", "NE", "SE", "SW"):
        lat_key = f"{corner} Corner Lat dec"
        lon_key = f"{corner} Corner Long dec"
        if lat_key in fields and lon_key in fields:
            corners[corner] = (float(fields[lat_key]), float(fields[lon_key]))

    if len(corners) != 4:
        raise ValueError(f"No EarthExplorer corner set found in {xml_path}")

    lons = [corners[key][1] for key in ("NW", "NE", "SE", "SW")]
    lats = [corners[key][0] for key in ("NW", "NE", "SE", "SW")]
    return MetadataPrior(
        source=os.path.basename(xml_path),
        confidence=0.35,
        west=min(lons),
        south=min(lats),
        east=max(lons),
        north=max(lats),
        crs="EPSG:4326",
        center_lon=(min(lons) + max(lons)) / 2.0,
        center_lat=(min(lats) + max(lats)) / 2.0,
        corners=corners,
        attributes={
            "format": "earth_explorer_xml",
            "entity_id": fields.get("Entity ID"),
            "acquisition_date": fields.get("Acquisition Date"),
        },
    )


def _prior_from_dict(data: dict[str, Any], source: str) -> MetadataPrior:
    if "prior" in data and isinstance(data["prior"], dict):
        data = data["prior"]

    corners = data.get("corners") or {}
    clean_corners = {}
    for key, value in corners.items():
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            clean_corners[str(key)] = (float(value[0]), float(value[1]))

    return MetadataPrior(
        source=str(data.get("source") or source),
        confidence=float(data.get("confidence", 0.5)),
        west=_maybe_float(data.get("west") or data.get("left")),
        south=_maybe_float(data.get("south") or data.get("bottom")),
        east=_maybe_float(data.get("east") or data.get("right")),
        north=_maybe_float(data.get("north") or data.get("top")),
        crs=str(data.get("crs", "EPSG:4326")),
        center_lon=_maybe_float(data.get("center_lon") or data.get("lon")),
        center_lat=_maybe_float(data.get("center_lat") or data.get("lat")),
        corners=clean_corners,
        attributes={
            key: value
            for key, value in data.items()
            if key not in {
                "source", "confidence", "west", "left", "south", "bottom",
                "east", "right", "north", "top", "crs", "center_lon", "lon",
                "center_lat", "lat", "corners",
            }
        },
    )


def _maybe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def parse_metadata_prior_file(path: str) -> list[MetadataPrior]:
    """Parse a prior file in JSON or XML form."""

    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [_prior_from_dict(item, source=os.path.basename(path)) for item in payload]
        if isinstance(payload, dict) and isinstance(payload.get("priors"), list):
            return [_prior_from_dict(item, source=os.path.basename(path)) for item in payload["priors"]]
        if isinstance(payload, dict):
            return [_prior_from_dict(payload, source=os.path.basename(path))]
        raise ValueError(f"Unsupported metadata-prior JSON payload in {path}")

    if suffix == ".xml":
        errors = []
        for parser in (parse_ee_metadata_xml, parse_bbox_xml):
            try:
                return [parser(path)]
            except Exception as exc:
                errors.append(str(exc))
        raise ValueError(f"Could not parse metadata prior XML {path}: {'; '.join(errors)}")

    raise ValueError(f"Unsupported metadata prior format: {path}")


def find_sidecar_prior_paths(input_path: str, priors_dir: str | None = None) -> list[str]:
    """Return likely sidecar prior paths for *input_path*."""

    candidates = []
    stem = os.path.splitext(os.path.basename(input_path))[0]
    search_dirs = []
    if priors_dir:
        search_dirs.append(priors_dir)
    search_dirs.append(os.path.dirname(input_path) or ".")

    patterns = [
        f"{stem}.priors.json",
        f"{stem}.json",
        f"{stem}.xml",
        f"{stem}_metadata.xml",
    ]
    for directory in search_dirs:
        for name in patterns:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                candidates.append(path)
    return list(dict.fromkeys(candidates))


def load_metadata_priors(input_path: str, explicit_paths: Iterable[str] | None = None,
                         priors_dir: str | None = None) -> list[MetadataPrior]:
    """Load priors from explicit files and likely sidecars."""

    all_paths = []
    for path in explicit_paths or []:
        if path:
            all_paths.append(path)
    all_paths.extend(find_sidecar_prior_paths(input_path, priors_dir=priors_dir))

    priors = []
    seen = set()
    for path in all_paths:
        if path in seen:
            continue
        seen.add(path)
        try:
            priors.extend(parse_metadata_prior_file(path))
            print(f"  Loaded metadata prior: {path}")
        except Exception as exc:
            print(f"  WARNING: Failed to load metadata prior {path}: {exc}")
    return priors


def merge_prior_bounds(priors: Iterable[MetadataPrior]) -> tuple[float, float, float, float] | None:
    """Return the union of all prior bounds in their native CRS when compatible."""

    priors = [prior for prior in priors if prior.has_bounds]
    if not priors:
        return None
    crs_values = {prior.crs for prior in priors}
    if len(crs_values) != 1:
        return None
    west = min(float(prior.west) for prior in priors if prior.west is not None)
    south = min(float(prior.south) for prior in priors if prior.south is not None)
    east = max(float(prior.east) for prior in priors if prior.east is not None)
    north = max(float(prior.north) for prior in priors if prior.north is not None)
    return (west, south, east, north)
