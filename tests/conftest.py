from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from .helpers import STAGE_MARKERS, make_scene, sanitize_nodeid, write_json, write_test_raster  # noqa: F401

_RESULTS_BY_NODEID: dict[str, dict] = {}


def _piecewise_run_dir() -> Path | None:
    value = os.environ.get("PIECEWISE_RUN_DIR")
    return Path(value) if value else None


def _stage_for_item(item: pytest.Item) -> str:
    for stage in STAGE_MARKERS:
        if item.get_closest_marker(stage):
            return stage
    return "unassigned"


class PiecewiseArtifactRecorder:
    def __init__(self, base_dir: Path | None, *, nodeid: str, stage: str):
        self.base_dir = base_dir
        self.nodeid = nodeid
        self.stage = stage
        if self.base_dir is not None:
            self.base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.base_dir is not None

    def artifact_dir(self) -> str | None:
        return str(self.base_dir.resolve()) if self.base_dir is not None else None

    def write_json(self, name: str, payload: dict) -> Path | None:
        if self.base_dir is None:
            return None
        return write_json(self.base_dir / name, payload)

    def write_text(self, name: str, payload: str) -> Path | None:
        if self.base_dir is None:
            return None
        path = self.base_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        return path

    def record_case_summary(self, payload: dict) -> Path | None:
        enriched = {
            "nodeid": self.nodeid,
            "stage": self.stage,
            **payload,
        }
        return self.write_json("case_summary.json", enriched)


@pytest.fixture
def piecewise_case(request: pytest.FixtureRequest) -> PiecewiseArtifactRecorder:
    run_dir = _piecewise_run_dir()
    stage = _stage_for_item(request.node)
    base_dir = None
    if run_dir is not None:
        base_dir = run_dir / "artifacts" / stage / sanitize_nodeid(request.node.nodeid)
    return PiecewiseArtifactRecorder(base_dir, nodeid=request.node.nodeid, stage=stage)


def pytest_sessionstart(session: pytest.Session) -> None:
    _RESULTS_BY_NODEID.clear()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()

    should_record = (
        report.when == "call"
        or (report.when == "setup" and (report.failed or report.skipped))
    )
    if not should_record:
        return

    status = "passed"
    if report.failed:
        status = "failed"
    elif report.skipped:
        status = "skipped"

    stage = _stage_for_item(item)
    artifact_dir = None
    run_dir = _piecewise_run_dir()
    if run_dir is not None:
        artifact_dir = str((run_dir / "artifacts" / stage / sanitize_nodeid(item.nodeid)).resolve())

    _RESULTS_BY_NODEID[item.nodeid] = {
        "nodeid": item.nodeid,
        "name": item.name,
        "stage": stage,
        "status": status,
        "duration_s": float(getattr(report, "duration", 0.0)),
        "artifact_dir": artifact_dir,
    }


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    run_dir = _piecewise_run_dir()
    if run_dir is None:
        return
    payload = {
        "selected_stages": os.environ.get("PIECEWISE_SELECTED_STAGES", "").split(","),
        "exitstatus": exitstatus,
        "tests": [
            _RESULTS_BY_NODEID[nodeid]
            for nodeid in sorted(_RESULTS_BY_NODEID)
        ],
    }
    path = run_dir / "piecewise_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
