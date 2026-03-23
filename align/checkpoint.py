"""Phase checkpoint save/load for the alignment pipeline.

Checkpoints capture AlignState at phase boundaries so that later phases
(grid_optim, flow) can be re-run without repeating expensive earlier work
(coarse offset, feature matching).

Serialisation: JSON for scalars + lists, companion ``.npz`` for numpy arrays.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .pipeline import AlignState

# Schema version — bump when checkpoint format changes
_SCHEMA_VERSION = 1

# Fields to skip (non-serialisable or recreated lazily)
_SKIP_FIELDS = {"model_cache", "temp_paths"}

# Fields stored in the .npz companion
_NUMPY_FIELDS = {"match_weights"}


def save_checkpoint(state: "AlignState", phase_id: str, checkpoint_dir: str) -> str:
    """Serialise *state* after a pipeline phase.

    Returns the path of the saved JSON file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    json_path = os.path.join(checkpoint_dir, f"{phase_id}.json")
    npz_path = os.path.join(checkpoint_dir, f"{phase_id}.npz")

    data: dict = {"_schema_version": _SCHEMA_VERSION, "_phase_id": phase_id}
    np_arrays: dict = {}

    for fname, fval in _iter_state_fields(state):
        if fname in _SKIP_FIELDS:
            continue

        if fname in _NUMPY_FIELDS and fval is not None:
            np_arrays[fname] = np.asarray(fval)
            continue

        data[fname] = _encode(fname, fval)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpySafeEncoder)

    if np_arrays:
        np.savez_compressed(npz_path, **np_arrays)
    elif os.path.exists(npz_path):
        os.remove(npz_path)

    print(f"  [Checkpoint] Saved {phase_id} → {json_path}", flush=True)
    return json_path


def load_checkpoint(phase_id: str, checkpoint_dir: str) -> "AlignState":
    """Restore an AlignState from a saved checkpoint."""
    from .pipeline import AlignState
    from .types import (QaReport, GlobalHypothesis, MetadataPrior, MatchPair, GCP)

    json_path = os.path.join(checkpoint_dir, f"{phase_id}.json")
    npz_path = os.path.join(checkpoint_dir, f"{phase_id}.npz")

    with open(json_path) as f:
        data = json.load(f)

    schema_v = data.pop("_schema_version", 0)
    if schema_v > _SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint schema v{schema_v} is newer than supported v{_SCHEMA_VERSION}")
    data.pop("_phase_id", None)

    # Load numpy arrays
    np_arrays = {}
    if os.path.exists(npz_path):
        with np.load(npz_path) as npz:
            for k in npz.files:
                np_arrays[k] = npz[k]

    # Reconstruct CRS
    if "work_crs" in data:
        crs_val = data.pop("work_crs")
        if crs_val is not None:
            from rasterio.crs import CRS
            data["work_crs"] = CRS.from_epsg(crs_val)

    # Reconstruct M_geo
    if "M_geo" in data:
        m = data.pop("M_geo")
        if m is not None:
            data["M_geo"] = np.array(m, dtype=np.float64)

    # Reconstruct typed lists — dicts come from to_dict(), lists from legacy
    data["matched_pairs"] = [
        MatchPair(**p) if isinstance(p, dict) else MatchPair.from_legacy(p)
        for p in data.get("matched_pairs", [])
    ]
    data["qa_holdout_pairs"] = [
        MatchPair(**p) if isinstance(p, dict) else MatchPair.from_legacy(p)
        for p in data.get("qa_holdout_pairs", [])
    ]
    data["gcps"] = [
        GCP(**g) if isinstance(g, dict) else GCP.from_legacy(g)
        for g in data.get("gcps", [])
    ]
    data["boundary_gcps"] = [
        GCP(**g) if isinstance(g, dict) else GCP.from_legacy(g)
        for g in data.get("boundary_gcps", [])
    ]
    data["geo_residuals"] = data.get("geo_residuals", [])
    data["correction_outliers"] = data.get("correction_outliers", [])

    # Reconstruct metadata_priors and global_hypotheses as plain tuples
    data["metadata_priors"] = [
        MetadataPrior(**{k: v for k, v in p.items()
                         if k in {f.name for f in __import__('dataclasses').fields(MetadataPrior)}})
        if isinstance(p, dict) else tuple(p)
        for p in data.get("metadata_priors", [])
    ]
    data["global_hypotheses"] = [
        GlobalHypothesis(**{k: v for k, v in h.items()
                            if k in {f.name for f in __import__('dataclasses').fields(GlobalHypothesis)}})
        if isinstance(h, dict) else tuple(h)
        for h in data.get("global_hypotheses", [])
    ]
    data["qa_reports"] = data.get("qa_reports", [])

    # Reconstruct chosen_hypothesis
    ch = data.get("chosen_hypothesis")
    if ch is not None:
        if isinstance(ch, dict):
            # Filter to only valid GlobalHypothesis fields
            import dataclasses as _dc
            valid = {f.name for f in _dc.fields(GlobalHypothesis)}
            data["chosen_hypothesis"] = GlobalHypothesis(
                **{k: v for k, v in ch.items() if k in valid})
        else:
            data["chosen_hypothesis"] = tuple(ch)

    # Overlap tuple
    ov = data.get("overlap")
    if ov is not None:
        data["overlap"] = tuple(ov)

    # Reference window tuple
    rw = data.get("reference_window")
    if rw is not None:
        data["reference_window"] = tuple(rw)

    # Reference bounds and target bounds
    for bk in ("reference_bounds_work", "target_bounds_work"):
        bv = data.get(bk)
        if bv is not None:
            data[bk] = tuple(bv)

    # Apply numpy arrays
    for k, v in np_arrays.items():
        data[k] = v

    # model_cache is not serialised — will be recreated lazily
    data.pop("model_cache", None)
    # temp_paths not serialised
    data.setdefault("temp_paths", [])

    # Fix stale temp paths: if current_input doesn't exist, fall back
    current = data.get("current_input")
    if current and not os.path.exists(current):
        fallback = data.get("input_path", "")
        if fallback and os.path.exists(fallback):
            data["current_input"] = fallback
        else:
            # Try the original input from the diagnostics dir
            diag_dir = os.path.dirname(os.path.dirname(json_path))
            for candidate in ("tune_baseline.tif",):
                p = os.path.join(diag_dir, candidate)
                if os.path.exists(p):
                    data["current_input"] = p
                    break

    # Build AlignState — only pass keys that are valid fields
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(AlignState)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}

    state = AlignState(**filtered)
    print(f"  [Checkpoint] Loaded {phase_id} ← {json_path}", flush=True)
    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types and dataclass objects."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "_asdict"):
            return obj._asdict()
        return super().default(obj)


def _iter_state_fields(state):
    """Yield (name, value) pairs from an AlignState dataclass."""
    import dataclasses
    for f in dataclasses.fields(state):
        yield f.name, getattr(state, f.name)


def _encode(name: str, value):
    """Encode a single field value to JSON-safe form."""
    if value is None:
        return None

    # CRS → EPSG int
    if name == "work_crs":
        try:
            return value.to_epsg()
        except Exception:
            return str(value)

    # numpy ndarray → list
    if isinstance(value, np.ndarray):
        return value.tolist()

    # Named tuples / dataclass-like objects
    if hasattr(value, "_asdict"):
        return value._asdict()
    if hasattr(value, "to_dict"):
        return value.to_dict()

    # numpy scalars
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)

    # Dicts — recurse to sanitize numpy types inside
    if isinstance(value, dict):
        return {k: _encode(k, v) for k, v in value.items()}

    # Lists of tuples (matched_pairs, gcps, etc.)
    if isinstance(value, (list, tuple)):
        return [_encode_element(v) for v in value]

    return value


def _encode_element(v):
    """Encode a single element within a list."""
    if isinstance(v, (list, tuple)):
        return list(v)
    if hasattr(v, "_asdict"):
        return v._asdict()
    if hasattr(v, "to_dict"):
        return v.to_dict()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v
