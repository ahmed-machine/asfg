from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_bounds

STAGE_MARKERS = ("process", "manifest", "qa", "selection")


def write_test_raster(path: Path, *, bounds=(0.0, 0.0, 10.0, 10.0),
                      crs="EPSG:4326", width: int = 16, height: int = 16,
                      fill: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((height, width), fill, dtype=np.uint8)
    transform = from_bounds(*bounds, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


def write_array_raster(path: Path, data: np.ndarray, *, bounds=(0.0, 0.0, 10.0, 10.0),
                       crs="EPSG:4326") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {data.shape}")
    height, width = data.shape
    transform = from_bounds(*bounds, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


def make_synthetic_feature_image(size: int = 320) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    cv2.rectangle(img, (40, 50), (135, 215), 180, -1)
    cv2.circle(img, (240, 100), 35, 255, -1)
    cv2.line(img, (175, 225), (292, 284), 210, 8)
    cv2.ellipse(img, (118, 265), (42, 18), 25, 0, 360, 220, -1)
    cv2.putText(img, "KH4", (176, 198),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, 150, 3, cv2.LINE_AA)
    points = np.array([[255, 245], [300, 238], [292, 290]], dtype=np.int32)
    cv2.fillConvexPoly(img, points, 200)
    return img


def warp_synthetic_image(data: np.ndarray, *, scale: float = 1.0,
                         rotation_deg: float = 0.0, shift_x_px: float = 0.0,
                         shift_y_px: float = 0.0) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {data.shape}")
    height, width = data.shape
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, rotation_deg, scale)
    matrix[0, 2] += shift_x_px
    matrix[1, 2] += shift_y_px
    return cv2.warpAffine(
        data.astype(np.float32),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def make_scene(entity_id: str = "DS1104-1057DA024", *,
               corners: dict | None = None,
               camera_type: str = "aft",
               needs_stitching: bool = False):
    if corners is None:
        corners = {
            "NW": (8.0, 1.0),
            "NE": (8.0, 4.0),
            "SE": (5.0, 4.0),
            "SW": (5.0, 1.0),
        }
    camera_system = SimpleNamespace(
        entity_prefix="DS1",
        name="KH-4",
        program="CORONA",
        needs_stitching=needs_stitching,
    )
    return SimpleNamespace(
        entity_id=entity_id,
        camera_system=camera_system,
        camera_type=camera_type,
        corners=corners,
        acquisition_date="1968-01-01",
    )


def sanitize_nodeid(nodeid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", nodeid).strip("._")


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
