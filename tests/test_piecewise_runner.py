from __future__ import annotations

import json

import pytest

from scripts.test import piecewise_harness as harness


@pytest.mark.fast
def test_build_pytest_command_includes_fast_marker_and_selected_stages(tmp_path):
    cmd = harness.build_pytest_command(["process", "qa"], tmp_path, pytest_args=["-q"])

    assert cmd[:3] == [harness.sys.executable, "-m", "pytest"]
    assert "fast and (process or qa)" in cmd
    assert "--junitxml" in cmd
    assert str(tmp_path / "junit.xml") in cmd
    assert cmd[-1] == "-q"


@pytest.mark.fast
def test_build_summary_groups_stage_results(tmp_path):
    run_dir = tmp_path / "piecewise_latest"
    run_dir.mkdir(parents=True)
    (run_dir / "piecewise_results.json").write_text(
        json.dumps(
            {
                "tests": [
                    {"nodeid": "tests/test_process_piecewise.py::test_a", "stage": "process", "status": "passed", "duration_s": 0.1, "artifact_dir": "a"},
                    {"nodeid": "tests/test_process_piecewise.py::test_b", "stage": "process", "status": "failed", "duration_s": 0.2, "artifact_dir": "b"},
                    {"nodeid": "tests/test_alignment_piecewise.py::test_c", "stage": "qa", "status": "skipped", "duration_s": 0.05, "artifact_dir": "c"},
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = harness.build_summary(
        "piecewise_latest",
        run_dir,
        ["process", "qa"],
        ["python", "-m", "pytest"],
        exit_code=1,
        wall_clock_s=2.345,
    )

    assert summary["run_id"] == "piecewise_latest"
    assert summary["exit_code"] == 1
    assert summary["results"]["process"]["passed"] == 1
    assert summary["results"]["process"]["failed"] == 1
    assert summary["results"]["qa"]["skipped"] == 1
    assert summary["results"]["process"]["tests"][0]["nodeid"].endswith("test_a")
    assert summary["results"]["qa"]["tests"][0]["artifact_dir"] == "c"
