from __future__ import annotations

import numpy as np
import pytest
from rasterio.crs import CRS

from align import pipeline
from align.state import AlignState
from align.types import BBox, GCP, MatchPair


def _state_for_pairs(pairs: list[MatchPair]) -> AlignState:
    state = AlignState(
        input_path="target.tif",
        reference_path="reference.tif",
        output_path="aligned.tif",
        work_crs=CRS.from_epsg(3857),
        overlap=BBox(0.0, 0.0, 100.0, 100.0),
    )
    state.matched_pairs = pairs
    state.gcps = [
        GCP(col=p.off_x, row=p.off_y, gx=p.ref_x, gy=p.ref_y, name=p.name)
        for p in pairs
    ]
    state.M_geo = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    state.geo_residuals = [5.0 for _ in pairs]
    state.gcp_coverage = 1.0
    state.qa_holdout_pairs = []
    state.cv_mean = None
    return state


@pytest.mark.fast
@pytest.mark.qa
def test_prewarp_gate_rejects_high_leverage_local_fit():
    """A clustered local fit plus one distant point must abstain before warp.

    This is the DA024 failure mode: an affine/grid candidate can look
    plausible near the local cluster while the design matrix is dominated by
    a single high-leverage point. The gate should surface this explicitly
    instead of accepting a confidence-threshold false positive.
    """
    coords = [
        (10.0, 10.0), (11.0, 10.0), (10.0, 11.0), (11.0, 11.0),
        (12.0, 10.0), (10.0, 12.0), (12.0, 12.0), (90.0, 90.0),
    ]
    pairs = [
        MatchPair(x, y, x + 1.0, y + 1.0, 0.9, f"auto:{idx}")
        for idx, (x, y) in enumerate(coords)
    ]
    state = _state_for_pairs(pairs)

    reasons, diagnostics = pipeline._prewarp_gate_reasons(state)

    assert "gcp_leverage_failed" in reasons
    assert diagnostics["rank"] == 6
    assert diagnostics["max_leverage"] > 0.98
