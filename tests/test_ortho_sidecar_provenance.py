"""Tests for the coarse-align ortho sidecar + provenance helpers.

The pipeline must never mutate the canonical ortho cache when applying a
reference-specific coarse shift. These tests exercise the provenance /
resolution helpers that enforce that contract without invoking ASP.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import paths
import process


def _touch(path: Path, payload: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_provenance_roundtrip(tmp_path):
    source = _touch(tmp_path / "ortho" / "EID_ortho.tif", b"source")
    reference = _touch(tmp_path / "ref.tif", b"ref")
    prov = paths.ortho_coarse_provenance_path(str(tmp_path), "EID")

    process._write_coarse_ortho_provenance(prov, str(source), str(reference))
    assert os.path.exists(prov)
    assert process._coarse_sidecar_provenance_matches(
        prov, str(source), str(reference)
    )


def test_provenance_invalidates_on_reference_change(tmp_path):
    source = _touch(tmp_path / "ortho" / "EID_ortho.tif", b"source")
    ref_a = _touch(tmp_path / "ref_a.tif", b"a")
    ref_b = _touch(tmp_path / "ref_b.tif", b"b-longer-payload")
    prov = paths.ortho_coarse_provenance_path(str(tmp_path), "EID")

    process._write_coarse_ortho_provenance(prov, str(source), str(ref_a))
    assert not process._coarse_sidecar_provenance_matches(
        prov, str(source), str(ref_b)
    )


def test_provenance_invalidates_on_source_modification(tmp_path):
    source = _touch(tmp_path / "ortho" / "EID_ortho.tif", b"source")
    reference = _touch(tmp_path / "ref.tif", b"ref")
    prov = paths.ortho_coarse_provenance_path(str(tmp_path), "EID")

    process._write_coarse_ortho_provenance(prov, str(source), str(reference))
    # Simulate canonical rebuild (mapproject rerun) → sidecar should be stale.
    source.write_bytes(b"source-v2-larger-payload")
    assert not process._coarse_sidecar_provenance_matches(
        prov, str(source), str(reference)
    )


def test_coarse_ortho_for_reference_requires_all_artifacts(tmp_path):
    cache_dir = str(tmp_path)
    entity = "EID"
    source = _touch(tmp_path / "ortho" / f"{entity}_ortho.tif", b"source")
    sidecar = _touch(Path(paths.ortho_coarse_path(cache_dir, entity)), b"sidecar")
    reference = _touch(tmp_path / "ref.tif", b"ref")
    prov = paths.ortho_coarse_provenance_path(cache_dir, entity)

    # Without provenance JSON, resolution returns None even if sidecar is there.
    assert process._coarse_ortho_for_reference(
        cache_dir, entity, str(source), str(reference)
    ) is None

    process._write_coarse_ortho_provenance(prov, str(source), str(reference))
    resolved = process._coarse_ortho_for_reference(
        cache_dir, entity, str(source), str(reference)
    )
    assert resolved == str(sidecar)

    # If the sidecar is removed, resolution should return None.
    os.remove(sidecar)
    assert process._coarse_ortho_for_reference(
        cache_dir, entity, str(source), str(reference)
    ) is None


@pytest.mark.parametrize("reference", [None, "", "/nonexistent/ref.tif"])
def test_coarse_ortho_for_reference_missing_reference(tmp_path, reference):
    cache_dir = str(tmp_path)
    assert process._coarse_ortho_for_reference(
        cache_dir, "EID", str(tmp_path / "ortho" / "EID_ortho.tif"), reference
    ) is None


def test_ortho_path_unchanged(tmp_path):
    """Regression guard: the canonical path must remain stable across any
    future refactor of the sidecar plumbing."""
    assert paths.ortho_path(str(tmp_path), "EID").endswith("/ortho/EID_ortho.tif")
    assert paths.ortho_coarse_path(str(tmp_path), "EID").endswith(
        "/ortho/EID_ortho.coarse.tif"
    )
    assert paths.ortho_coarse_provenance_path(str(tmp_path), "EID").endswith(
        "/ortho/EID_ortho.coarse.json"
    )
