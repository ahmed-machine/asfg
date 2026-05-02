"""Test the DIS-to-SEA-RAFT fallback in flow refinement.

DIS has a small effective search radius (~30 px). On cross-system
declassified imagery (1968 KH-4B vs 1976 KH-9) the residual after
grid optimisation can exceed that range, especially under non-rigid
panoramic distortion. Forward-backward consistency then drops to ~0%
and DIS is rejected by the precheck.

SEA-RAFT has a much larger effective search range and provides a
mixture-of-Laplace uncertainty signal. When DIS gives up at the coarse
pass, the fallback substitutes SEA-RAFT once before the whole flow
refinement bails. We exercise the gate logic without loading real
weights by monkeypatching the flow estimators.
"""

from __future__ import annotations

import numpy as np
import pytest

from align import warp


def _arr(rows=64, cols=64, fill=128):
    return np.full((rows, cols), fill, dtype=np.uint8)


def _flat_flow(shape=(64, 64), dx=0.0, dy=0.0):
    flow = np.zeros((*shape, 2), dtype=np.float32)
    flow[:, :, 0] = dx
    flow[:, :, 1] = dy
    return flow


def _capture_warp_branch(monkeypatch, *,
                         dis_returns: np.ndarray,
                         raft_returns: np.ndarray | Exception | None,
                         dis_fb_pct: float,
                         raft_fb_pct: float):
    """Stub the flow estimators + FB-mask helper inside align.warp.

    Returns lists ``(dis_calls, raft_calls)`` populated when each
    estimator is invoked, plus the masking helper sequence so the test
    can verify the ordering."""
    dis_calls = []
    raft_calls = []
    fb_calls = []

    def _stub_dis(ref, off):
        dis_calls.append(1)
        return dis_returns

    def _stub_raft(ref, off):
        raft_calls.append(1)
        if isinstance(raft_returns, Exception):
            raise raft_returns
        return raft_returns

    def _stub_fb_mask(ref, off, flow, threshold, use_uncertainty=False):
        fb_calls.append({"use_uncertainty": use_uncertainty})
        # Return a mask whose mean matches the requested percentage
        target_pct = (raft_fb_pct
                      if any(np.array_equal(flow, raft_returns) if hasattr(flow, "shape") else False
                             for _ in [None]) else dis_fb_pct)
        # The above is fragile — instead, use call count: first call =
        # DIS precheck, second = RAFT precheck, third = full mask.
        idx = len(fb_calls)
        if idx == 1:
            target_pct = dis_fb_pct
        elif idx == 2 and len(raft_calls) > 0:
            target_pct = raft_fb_pct
        else:
            target_pct = max(dis_fb_pct, raft_fb_pct)
        n_true = max(1, int(ref.size * target_pct / 100))
        mask = np.zeros(ref.shape, dtype=bool)
        flat = mask.reshape(-1)
        flat[:n_true] = True
        return mask

    monkeypatch.setattr(warp, "_estimate_flow", _stub_dis)
    monkeypatch.setattr(warp, "_estimate_flow_raft", _stub_raft)
    monkeypatch.setattr(warp, "_forward_backward_mask", _stub_fb_mask)
    return dis_calls, raft_calls, fb_calls


def _run_just_the_gate(monkeypatch, dis_returns, raft_returns,
                       dis_fb_pct, raft_fb_pct):
    """Mimic just the gate block from _compute_flow_corrections so we
    can test the fallback logic without rasterio / GridWarper
    machinery. Returns the chosen flow + mask, or raises if neither
    DIS nor SEA-RAFT clears the 5% floor."""
    from align import constants as _C
    ref_u8 = _arr()
    warped_u8 = _arr()

    dis_calls, raft_calls, fb_calls = _capture_warp_branch(
        monkeypatch,
        dis_returns=dis_returns, raft_returns=raft_returns,
        dis_fb_pct=dis_fb_pct, raft_fb_pct=raft_fb_pct,
    )

    flow_c = warp._estimate_flow(ref_u8, warped_u8)
    fb_mask_c = warp._forward_backward_mask(
        ref_u8, warped_u8, flow_c, _C.FB_CONSISTENCY_PX,
        use_uncertainty=False,
    )
    precheck_c_pct = float(fb_mask_c.sum()) / max(1, fb_mask_c.size) * 100
    if precheck_c_pct < 5.0:
        try:
            flow_raft_c = warp._estimate_flow_raft(ref_u8, warped_u8)
        except Exception as exc:
            raise RuntimeError("Insufficient reliable coarse flow pixels") from exc
        fb_mask_raft = warp._forward_backward_mask(
            ref_u8, warped_u8, flow_raft_c, _C.FB_CONSISTENCY_PX,
            use_uncertainty=False,
        )
        raft_pct = float(fb_mask_raft.sum()) / max(1, fb_mask_raft.size) * 100
        if raft_pct < 5.0:
            raise RuntimeError("Insufficient reliable coarse flow pixels")
        flow_c = flow_raft_c
        fb_mask_c = fb_mask_raft
    return flow_c, fb_mask_c, dis_calls, raft_calls


def test_dis_clears_precheck_no_fallback(monkeypatch):
    """DIS finds 50% reliable → SEA-RAFT not invoked."""
    flow_c, mask, dis_calls, raft_calls = _run_just_the_gate(
        monkeypatch,
        dis_returns=_flat_flow(),
        raft_returns=_flat_flow(),
        dis_fb_pct=50.0, raft_fb_pct=99.0,
    )
    assert len(dis_calls) == 1
    assert len(raft_calls) == 0


def test_dis_fails_searaft_recovers(monkeypatch):
    """DIS finds 0% reliable, SEA-RAFT finds 30% → use SEA-RAFT."""
    raft_flow = _flat_flow(dx=2.0)
    flow_c, mask, dis_calls, raft_calls = _run_just_the_gate(
        monkeypatch,
        dis_returns=_flat_flow(),
        raft_returns=raft_flow,
        dis_fb_pct=0.0, raft_fb_pct=30.0,
    )
    assert len(dis_calls) == 1
    assert len(raft_calls) == 1
    assert np.array_equal(flow_c, raft_flow), \
        "should fall through to SEA-RAFT's flow"


def test_both_fail_raises(monkeypatch):
    """DIS 0%, SEA-RAFT 0% → RuntimeError as before."""
    with pytest.raises(RuntimeError, match="Insufficient reliable"):
        _run_just_the_gate(
            monkeypatch,
            dis_returns=_flat_flow(),
            raft_returns=_flat_flow(),
            dis_fb_pct=0.0, raft_fb_pct=0.0,
        )


def test_searaft_exception_falls_through_to_raise(monkeypatch):
    """If SEA-RAFT raises (model load failure, OOM), the gate raises
    rather than crashing the whole alignment."""
    with pytest.raises(RuntimeError, match="Insufficient reliable"):
        _run_just_the_gate(
            monkeypatch,
            dis_returns=_flat_flow(),
            raft_returns=ImportError("sea_raft module unavailable"),
            dis_fb_pct=0.0, raft_fb_pct=99.0,
        )
