#!/usr/bin/env python3
"""QA tests for the multi-image tuning infrastructure.

Covers: camera detection, profile loading/updating, case config,
checkpoint serialization, score extraction, score aggregation,
parameter mapping, and orchestrator utilities.

Run:  python3 scripts/test/test_tuning.py
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

_passed = 0
_failed = 0


def _check(name: str, condition: bool, detail: str = ""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


# -----------------------------------------------------------------------
# 1. Camera detection
# -----------------------------------------------------------------------

def test_camera_detect():
    print("\n=== Camera Detection ===")
    from align.params import detect_camera

    _check("KH-9 D3C prefix",
           detect_camera("D3C1217-100109F007.tif") == "kh9")
    _check("KH-9 DZB prefix",
           detect_camera("DZB1212-500236L002001.tif") == "kh9")
    _check("KH-4 DS prefix",
           detect_camera("DS1052-1073DA157.tif") == "kh4")
    _check("KH-7 DZB004 H suffix",
           detect_camera("DZB00403600089H016001.tif") == "kh7")
    _check("Full path extraction",
           detect_camera("/some/path/1982-05-23 - Bahrain - D3C1217-100109F007.tif") == "kh9")
    _check("Unknown returns None",
           detect_camera("random_image.tif") is None)
    _check("KH-7 before general DZB (order matters)",
           detect_camera("DZB00401234H001.tif") == "kh7")
    _check("Case insensitive",
           detect_camera("d3c1217-100109f007.TIF") == "kh9")


# -----------------------------------------------------------------------
# 2. Profile loading
# -----------------------------------------------------------------------

def test_profile_loading():
    print("\n=== Profile Loading ===")
    from align.params import load_profile, set_profile, get_params

    base = load_profile("_base")
    _check("Base profile loads", base is not None)
    _check("Base roma_size", base.matching.roma_size == 640)
    _check("Base w_data", base.grid_optim.w_data == 1.0)

    kh4 = load_profile("kh4")
    _check("KH-4 inherits from base", kh4 is not None)
    _check("KH-4 roma_size override", kh4.matching.roma_size == 784)
    _check("KH-4 inherits flow params", kh4.flow.max_flow_bias_m == 15.0)
    _check("KH-4 clahe override", kh4.normalization.clahe_clip_limit == 2.0)

    kh9 = load_profile("kh9")
    _check("KH-9 loads", kh9 is not None)
    _check("KH-9 roma_size override", kh9.matching.roma_size == 784)

    # set_profile sets the singleton
    p = set_profile("kh4")
    _check("set_profile returns params", p.matching.roma_size == 784)
    active = get_params()
    _check("get_params returns active", active.matching.roma_size == 784)

    # Reset
    set_profile("_base")


# -----------------------------------------------------------------------
# 3. Profile override context manager
# -----------------------------------------------------------------------

def test_profile_override():
    print("\n=== Profile Override ===")
    from align.params import set_profile, get_params, override

    set_profile("_base")

    with override(grid_optim__w_data=2.5, matching__roma_size=784):
        p = get_params()
        _check("Override w_data", p.grid_optim.w_data == 2.5)
        _check("Override roma_size", p.matching.roma_size == 784)

    p = get_params()
    _check("Restored w_data after context", p.grid_optim.w_data == 1.0)
    _check("Restored roma_size after context", p.matching.roma_size == 640)


# -----------------------------------------------------------------------
# 4. Profile YAML update
# -----------------------------------------------------------------------

def test_profile_yaml_update():
    print("\n=== Profile YAML Update ===")
    import yaml
    from align.params import update_profile_yaml, load_profile

    # Work on a temp copy
    src = PROJECT_ROOT / "data" / "profiles" / "kh9.yaml"
    with open(src) as f:
        original = f.read()

    try:
        update_profile_yaml("kh9", {"matching__roma_size": 999})
        reloaded = load_profile("kh9")
        _check("YAML update persists", reloaded.matching.roma_size == 999)
    finally:
        # Restore original
        with open(src, "w") as f:
            f.write(original)

    restored = load_profile("kh9")
    _check("Restored original after test", restored.matching.roma_size == 784)


# -----------------------------------------------------------------------
# 5. Tune cases config
# -----------------------------------------------------------------------

def test_tune_cases_config():
    print("\n=== Tune Cases Config ===")
    import yaml

    with open(PROJECT_ROOT / "data" / "tune_cases.yaml") as f:
        config = yaml.safe_load(f)

    cases = config.get("cases", {})
    _check("Has cases", len(cases) >= 2)

    for cid, case in cases.items():
        _check(f"{cid} has target", "target" in case)
        _check(f"{cid} has reference", "reference" in case)
        _check(f"{cid} has profile", "profile" in case)
        _check(f"{cid} has weight", "weight" in case)

    groups = config.get("param_groups", {})
    _check("Has profile_specific group", len(groups.get("profile_specific", [])) > 0)
    _check("Has base_shared group", len(groups.get("base_shared", [])) > 0)

    # No overlap between groups
    ps = set(groups.get("profile_specific", []))
    bs = set(groups.get("base_shared", []))
    _check("No overlap between param groups", len(ps & bs) == 0)


# -----------------------------------------------------------------------
# 6. Score extraction
# -----------------------------------------------------------------------

def test_score_extraction():
    print("\n=== Score Extraction ===")
    from scripts.tune.tune import _extract_score, _clamp_score
    from types import SimpleNamespace

    # Test with total_score in qa_reports dict
    state = SimpleNamespace(
        qa_reports=[{"total_score": 65.2, "image_metrics": {"score": 55.0}}],
        qa_json_path=None,
    )
    score = _extract_score(state, "/nonexistent")
    _check("Extracts total_score from dict report", score == 65.2)

    # Test with image_metrics.score fallback
    state2 = SimpleNamespace(
        qa_reports=[{"image_metrics": {"score": 55.0}}],
        qa_json_path=None,
    )
    score2 = _extract_score(state2, "/nonexistent")
    _check("Extracts image_metrics.score", score2 == 55.0)

    # Test with direct score key
    state3 = SimpleNamespace(
        qa_reports=[{"score": 42.0}],
        qa_json_path=None,
    )
    score3 = _extract_score(state3, "/nonexistent")
    _check("Extracts direct score", score3 == 42.0)

    # Test with aligned_qa.json file
    with tempfile.TemporaryDirectory() as td:
        qa_path = os.path.join(td, "aligned_qa.json")
        with open(qa_path, "w") as f:
            json.dump({"reports": [{"total_score": 77.3}]}, f)
        state4 = SimpleNamespace(qa_reports=[], qa_json_path=None)
        score4 = _extract_score(state4, td)
        _check("Extracts from aligned_qa.json", score4 == 77.3)

    # Test empty state returns 999
    state5 = SimpleNamespace(qa_reports=[], qa_json_path=None)
    score5 = _extract_score(state5, "/nonexistent")
    _check("Empty state returns 999", score5 == 999.0)

    # Test clamp
    _check("Clamp inf → 999", _clamp_score(float("inf")) == 999.0)
    _check("Clamp -inf → 999", _clamp_score(float("-inf")) == 999.0)
    _check("Clamp nan → 999", _clamp_score(float("nan")) == 999.0)
    _check("Clamp None → 999", _clamp_score(None) == 999.0)
    _check("Clamp normal passes through", _clamp_score(65.2) == 65.2)
    _check("Clamp zero passes through", _clamp_score(0.0) == 0.0)


# -----------------------------------------------------------------------
# 9. Checkpoint serialization (numpy types)
# -----------------------------------------------------------------------

def test_checkpoint_numpy_encoder():
    print("\n=== Checkpoint Numpy Encoder ===")
    from align.checkpoint import _NumpySafeEncoder

    data = {
        "float32": np.float32(1.5),
        "float64": np.float64(2.5),
        "int32": np.int32(42),
        "int64": np.int64(99),
        "bool_": np.bool_(True),
        "array": np.array([1.0, 2.0]),
        "nested": {"inner": np.float32(3.14)},
        "list": [np.float32(1.0), np.int64(2)],
    }

    try:
        result = json.dumps(data, cls=_NumpySafeEncoder)
        parsed = json.loads(result)
        _check("Serializes without error", True)
        _check("float32 preserved", abs(parsed["float32"] - 1.5) < 0.01)
        _check("int64 preserved", parsed["int64"] == 99)
        _check("bool preserved", parsed["bool_"] is True)
        _check("array → list", parsed["array"] == [1.0, 2.0])
        _check("nested float32", abs(parsed["nested"]["inner"] - 3.14) < 0.01)
    except Exception as e:
        _check("Serialization", False, str(e))


# -----------------------------------------------------------------------
# 10. Checkpoint dataclass reconstruction
# -----------------------------------------------------------------------

def test_checkpoint_dataclass_reconstruction():
    print("\n=== Checkpoint Dataclass Reconstruction ===")
    from align.types import GlobalHypothesis, MetadataPrior
    from align.checkpoint import _encode, _NumpySafeEncoder

    # GlobalHypothesis round-trip
    hyp = GlobalHypothesis(
        hypothesis_id="test", score=0.9, source="test",
        left=50.0, bottom=26.0, right=51.0, top=27.0,
        diagnostics={"key": np.float32(1.5)},
    )
    encoded = _encode("chosen_hypothesis", hyp)
    _check("GlobalHypothesis encodes to dict", isinstance(encoded, dict))
    _check("hypothesis_id preserved", encoded["hypothesis_id"] == "test")

    # Verify it JSON-serializes
    try:
        s = json.dumps(encoded, cls=_NumpySafeEncoder)
        _check("GlobalHypothesis JSON serializable", True)
    except Exception as e:
        _check("GlobalHypothesis JSON serializable", False, str(e))

    # MetadataPrior round-trip
    prior = MetadataPrior(
        source="test.xml", confidence=0.35,
        west=50.0, south=26.0, east=51.0, north=27.0,
    )
    encoded_p = _encode("metadata_priors", [prior])
    _check("MetadataPrior list encodes", isinstance(encoded_p, list))
    _check("MetadataPrior item is dict", isinstance(encoded_p[0], dict))


# -----------------------------------------------------------------------
# 11. Parameter mapping (Optuna → profile keys)
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# 12. Plateau detection
# -----------------------------------------------------------------------

def test_plateau_detection():
    print("\n=== Plateau Detection ===")
    from scripts.tune.tune import check_plateau
    from types import SimpleNamespace

    # Mock study with trials
    class MockStudy:
        def __init__(self, values):
            self.trials = [
                SimpleNamespace(
                    value=v,
                    state=SimpleNamespace(name="COMPLETE"),
                    number=i,
                )
                for i, v in enumerate(values)
            ]

    # Need to mock TrialState
    import optuna
    from optuna.trial import TrialState
    for t in MockStudy([100, 95, 90, 85, 80, 75, 70]).trials:
        t.state = TrialState.COMPLETE

    study_improving = MockStudy([100, 95, 90, 85, 80, 75, 70])
    for t in study_improving.trials:
        t.state = TrialState.COMPLETE
    _check("No plateau when improving",
           not check_plateau(study_improving, patience=5))

    study_flat = MockStudy([100, 95, 95.5, 95.3, 95.4, 95.2, 95.1])
    for t in study_flat.trials:
        t.state = TrialState.COMPLETE
    _check("Plateau detected when flat",
           check_plateau(study_flat, patience=5))


# -----------------------------------------------------------------------
# 13. Phase checkpoint ID mapping
# -----------------------------------------------------------------------

def test_phase_checkpoint_ids():
    print("\n=== Phase Checkpoint IDs ===")
    from scripts.tune.tune import _phase_checkpoint_id

    _check("matching → post_scale_rotation",
           _phase_checkpoint_id("matching") == "post_scale_rotation")
    _check("validation → post_match",
           _phase_checkpoint_id("validation") == "post_match")
    _check("grid_optim → post_validate",
           _phase_checkpoint_id("grid_optim") == "post_validate")
    _check("flow → post_validate",
           _phase_checkpoint_id("flow") == "post_validate")
    _check("normalization → post_validate",
           _phase_checkpoint_id("normalization") == "post_validate")


# -----------------------------------------------------------------------
# 14. No-CRS resolution estimation
# -----------------------------------------------------------------------

def test_no_crs_resolution():
    print("\n=== No-CRS Resolution Estimation ===")
    from align.geo import get_native_resolution_m
    from align.types import MetadataPrior
    from types import SimpleNamespace

    # Mock a no-CRS dataset
    mock_src = SimpleNamespace(
        crs=None, width=10000, height=5000,
        transform=None, bounds=None,
    )

    prior = MetadataPrior(
        source="test", confidence=0.5,
        west=50.0, south=26.0, east=51.0, north=26.5,
    )
    res = get_native_resolution_m(mock_src, priors=[prior])
    _check("No-CRS returns positive resolution", res > 0)
    # ~111km wide at lat 26.25, 10000px → ~11 m/px
    _check("Resolution in reasonable range", 5 < res < 20,
           f"got {res:.1f}")

    # No priors → sentinel
    res_no_prior = get_native_resolution_m(mock_src, priors=None)
    _check("No priors returns sentinel 1.0", res_no_prior == 1.0)


# -----------------------------------------------------------------------
# 15. auto-align.py --profile arg exists
# -----------------------------------------------------------------------

def test_auto_align_profile_arg():
    print("\n=== auto-align.py --profile ===")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "auto-align.py"), "--help"],
        capture_output=True, text=True,
    )
    _check("--profile in help", "--profile" in result.stdout)
    _check("auto-detected in help", "Auto-detected" in result.stdout)


# -----------------------------------------------------------------------
# 17. tune.py --case CLI parses
# -----------------------------------------------------------------------

def test_tune_case_cli():
    print("\n=== tune.py --case ===")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "tune" / "tune.py"), "--help"],
        capture_output=True, text=True,
    )
    _check("--case in help", "--case" in result.stdout)
    _check("--target in help", "--target" in result.stdout)
    _check("--reference in help", "--reference" in result.stdout)
    _check("--anchors in help", "--anchors" in result.stdout)


# -----------------------------------------------------------------------
# 18. Checkpoint round-trip (save → JSON → load → usable object)
# -----------------------------------------------------------------------

def test_checkpoint_round_trip():
    print("\n=== Checkpoint Round-Trip ===")
    from align.state import AlignState
    from align.types import GlobalHypothesis, MetadataPrior
    from align.checkpoint import save_checkpoint, load_checkpoint

    with tempfile.TemporaryDirectory() as td:
        # Create a realistic AlignState with dataclass objects
        hyp = GlobalHypothesis(
            hypothesis_id="hyp_0", score=0.85, source="ncc_grid",
            left=50.0, bottom=26.0, right=51.0, top=27.0,
            diagnostics={"ncc": np.float32(0.85), "rank": np.int64(1)},
        )
        prior = MetadataPrior(
            source="test.xml", confidence=0.35,
            west=50.0, south=26.0, east=51.0, north=27.0,
        )

        # Create temp file to act as current_input
        input_file = os.path.join(td, "input.tif")
        with open(input_file, "wb") as f:
            f.write(b"fake")
        ref_file = os.path.join(td, "ref.tif")
        with open(ref_file, "wb") as f:
            f.write(b"fake")

        state = AlignState(
            input_path=input_file,
            reference_path=ref_file,
            output_path=os.path.join(td, "out.tif"),
            current_input=input_file,
            global_hypotheses=[hyp],
            chosen_hypothesis=hyp,
            metadata_priors=[prior],
            diagnostics_dir=td,
        )

        ckpt_dir = os.path.join(td, "checkpoints")
        save_checkpoint(state, "test_phase", ckpt_dir)

        loaded = load_checkpoint("test_phase", ckpt_dir)

        # A1: GlobalHypothesis survives round-trip
        _check("chosen_hypothesis is GlobalHypothesis",
               isinstance(loaded.chosen_hypothesis, GlobalHypothesis))
        _check("chosen_hypothesis.hypothesis_id accessible",
               loaded.chosen_hypothesis.hypothesis_id == "hyp_0")
        _check("chosen_hypothesis.score preserved",
               abs(loaded.chosen_hypothesis.score - 0.85) < 0.01)

        # A2: global_hypotheses list items are GlobalHypothesis
        _check("global_hypotheses non-empty", len(loaded.global_hypotheses) > 0)
        _check("global_hypotheses[0] is GlobalHypothesis",
               isinstance(loaded.global_hypotheses[0], GlobalHypothesis))
        _check("global_hypotheses[0].hypothesis_id accessible",
               loaded.global_hypotheses[0].hypothesis_id == "hyp_0")

        # A3: MetadataPrior survives round-trip
        _check("metadata_priors non-empty", len(loaded.metadata_priors) > 0)
        _check("metadata_priors[0] is MetadataPrior",
               isinstance(loaded.metadata_priors[0], MetadataPrior))
        _check("metadata_priors[0].has_bounds accessible",
               loaded.metadata_priors[0].has_bounds is True)

        # A4: numpy scalars inside diagnostics survive
        diag = loaded.chosen_hypothesis.diagnostics
        _check("diagnostics dict preserved", isinstance(diag, dict))
        _check("numpy float32 survives round-trip",
               isinstance(diag.get("ncc"), (int, float)))
        _check("numpy int64 survives round-trip",
               isinstance(diag.get("rank"), (int, float)))


# -----------------------------------------------------------------------
# 19. Stale temp path fallback
# -----------------------------------------------------------------------

def test_stale_temp_path_fallback():
    print("\n=== Stale Temp Path Fallback ===")
    from align.state import AlignState
    from align.checkpoint import save_checkpoint, load_checkpoint

    with tempfile.TemporaryDirectory() as td:
        # Create real files for initial save
        real_input = os.path.join(td, "input.tif")
        with open(real_input, "wb") as f:
            f.write(b"fake")
        ref_file = os.path.join(td, "ref.tif")
        with open(ref_file, "wb") as f:
            f.write(b"fake")

        state = AlignState(
            input_path=real_input,
            reference_path=ref_file,
            output_path=os.path.join(td, "out.tif"),
            current_input=real_input,
            diagnostics_dir=td,
        )

        ckpt_dir = os.path.join(td, "checkpoints")
        save_checkpoint(state, "stale_test", ckpt_dir)

        # B1: Both exist → keeps original
        loaded1 = load_checkpoint("stale_test", ckpt_dir)
        _check("Existing path kept", loaded1.current_input == real_input)

        # B2: Patch checkpoint JSON to point to a nonexistent temp path,
        # input_path still valid → falls back to input_path
        import json as _json
        ckpt_json = os.path.join(ckpt_dir, "stale_test.json")
        with open(ckpt_json) as f:
            data = _json.load(f)
        data["current_input"] = "/tmp/nonexistent_stale_test_xyz.tif"
        data["input_path"] = real_input
        with open(ckpt_json, "w") as f:
            _json.dump(data, f)

        loaded2 = load_checkpoint("stale_test", ckpt_dir)
        _check("Stale path falls back to input_path",
               loaded2.current_input == real_input)

        # B3: Both stale, but tune_baseline.tif exists in diagnostics dir
        baseline = os.path.join(td, "tune_baseline.tif")
        with open(baseline, "wb") as f:
            f.write(b"fake")
        data["current_input"] = "/tmp/nonexistent_xyz.tif"
        data["input_path"] = "/tmp/also_nonexistent_xyz.tif"
        with open(ckpt_json, "w") as f:
            _json.dump(data, f)

        loaded3 = load_checkpoint("stale_test", ckpt_dir)
        _check("Falls back to tune_baseline.tif",
               loaded3.current_input == baseline)


# -----------------------------------------------------------------------
# 20. _resolve_case routing logic
# -----------------------------------------------------------------------

def test_resolve_case():
    print("\n=== _resolve_case Routing ===")
    import argparse
    from scripts.tune.tune import _resolve_case

    # C1: Explicit --target/--reference override config
    args1 = argparse.Namespace(
        target="/explicit/target.tif",
        reference="/explicit/ref.tif",
        anchors="/explicit/anchors.json",
        case=None, profile="_base",
    )
    t, r, a = _resolve_case(args1)
    _check("Explicit target", t == "/explicit/target.tif")
    _check("Explicit reference", r == "/explicit/ref.tif")
    _check("Explicit anchors", a == "/explicit/anchors.json")

    # C2: Explicit target/ref, no anchors → default anchors
    args2 = argparse.Namespace(
        target="/explicit/target.tif",
        reference="/explicit/ref.tif",
        anchors=None,
        case=None, profile="_base",
    )
    _, _, a2 = _resolve_case(args2)
    _check("Default anchors when not specified",
           a2.endswith("bahrain_anchor_gcps.json"))

    # C3: --case kh9_1982 loads correct paths
    args3 = argparse.Namespace(
        target=None, reference=None, anchors=None,
        case="kh9_1982", profile="_base",
    )
    t3, r3, a3 = _resolve_case(args3)
    _check("Case kh9_1982 target set", t3 is not None and len(t3) > 0)
    _check("Case kh9_1982 reference set", r3 is not None and len(r3) > 0)
    _check("Case kh9_1982 sets profile",
           args3.profile == "kh9")

    # C4: Default (no --case) picks first case
    args4 = argparse.Namespace(
        target=None, reference=None, anchors=None,
        case=None, profile="_base",
    )
    t4, r4, a4 = _resolve_case(args4)
    _check("Default case returns valid target", t4 is not None and len(t4) > 0)

    # C5: Unknown case ID exits (test via exception catch)
    args5 = argparse.Namespace(
        target=None, reference=None, anchors=None,
        case="nonexistent_case_xyz", profile="_base",
    )
    try:
        # _resolve_case calls sys.exit(1) on unknown case
        _resolve_case(args5)
        _check("Unknown case fails", False, "should have exited")
    except SystemExit as e:
        _check("Unknown case fails", e.code == 1)


# -----------------------------------------------------------------------
# 21. Constants sync after profile change
# -----------------------------------------------------------------------

def test_constants_sync():
    print("\n=== Constants Sync ===")
    from align.params import set_profile, override, _sync_constants, get_params
    from align import constants

    # D1: _sync_constants explicitly pushes values into constants module
    set_profile("kh4")
    _sync_constants(get_params())
    _check("KH-4 ROMA_SIZE synced to constants",
           constants.ROMA_SIZE == 784)
    _check("KH-4 CLAHE_CLIP_LIMIT synced",
           abs(constants.CLAHE_CLIP_LIMIT - 2.0) < 0.01)

    # D2: override() context manager syncs constants automatically
    with override(matching__roma_size=999):
        _check("Override ROMA_SIZE in constants",
               constants.ROMA_SIZE == 999)

    # D3: After override exits, constants restored to pre-override values
    _check("Restored ROMA_SIZE after override",
           constants.ROMA_SIZE == 784)

    # Reset to base
    set_profile("_base")
    _sync_constants(get_params())
    _check("Base ROMA_SIZE after reset",
           constants.ROMA_SIZE == 640)


# -----------------------------------------------------------------------
# 22. apply_best param filtering
# -----------------------------------------------------------------------

def test_apply_best_filtering():
    print("\n=== apply_best Param Filtering ===")
    import yaml
    from align.params import load_profile, update_profile_yaml

    # Load config to get param groups
    config_path = PROJECT_ROOT / "data" / "tune_cases.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    param_groups = config.get("param_groups", {})
    profile_specific_keys = set(param_groups.get("profile_specific", []))
    base_shared_keys = set(param_groups.get("base_shared", []))

    # Test that filtering correctly separates keys
    mixed_params = {
        "matching__roma_size": 784,        # profile_specific
        "grid_optim__w_data": 2.0,         # profile_specific
        "grid_optim__w_arap": 0.8,         # base_shared
        "grid_optim__lr": 0.005,           # base_shared
        "flow__max_flow_bias_m": 20.0,     # base_shared
    }

    # E1: Per-profile filtering keeps only profile_specific
    per_profile = {k: v for k, v in mixed_params.items()
                   if k in profile_specific_keys}
    _check("Per-profile keeps roma_size",
           "matching__roma_size" in per_profile)
    _check("Per-profile keeps w_data",
           "grid_optim__w_data" in per_profile)
    _check("Per-profile excludes w_arap",
           "grid_optim__w_arap" not in per_profile)
    _check("Per-profile excludes lr",
           "grid_optim__lr" not in per_profile)
    _check("Per-profile excludes flow param",
           "flow__max_flow_bias_m" not in per_profile)

    # E2: Cross-profile filtering keeps only base_shared
    cross_profile = {k: v for k, v in mixed_params.items()
                     if k in base_shared_keys}
    _check("Cross-profile keeps w_arap",
           "grid_optim__w_arap" in cross_profile)
    _check("Cross-profile keeps lr",
           "grid_optim__lr" in cross_profile)
    _check("Cross-profile keeps flow param",
           "flow__max_flow_bias_m" in cross_profile)
    _check("Cross-profile excludes roma_size",
           "matching__roma_size" not in cross_profile)
    _check("Cross-profile excludes w_data",
           "grid_optim__w_data" not in cross_profile)

    # E3: Union covers all keys
    _check("All mixed keys classified",
           set(mixed_params.keys()) == per_profile.keys() | cross_profile.keys())


# -----------------------------------------------------------------------
# 23. Trial cleanup
# -----------------------------------------------------------------------

def test_trial_cleanup():
    print("\n=== Trial Cleanup ===")
    from scripts.tune.tune import _cleanup_trial, _cleanup_old_trials

    # F1: _cleanup_trial removes .tif/.npz/.jpg but keeps .json
    with tempfile.TemporaryDirectory() as td:
        # Create fake trial files
        for ext in (".tif", ".npz", ".jpg", ".png", ".json"):
            with open(os.path.join(td, f"output{ext}"), "w") as f:
                f.write("fake")
        # Also a checkpoint subdir
        ckpt_sub = os.path.join(td, "checkpoints")
        os.makedirs(ckpt_sub)
        with open(os.path.join(ckpt_sub, "post_validate.json"), "w") as f:
            f.write("{}")

        _cleanup_trial(td, keep_summary=True)

        remaining = set(os.listdir(td))
        _check("JSON kept", "output.json" in remaining)
        _check("TIF removed", "output.tif" not in remaining)
        _check("NPZ removed", "output.npz" not in remaining)
        _check("JPG removed", "output.jpg" not in remaining)
        _check("PNG removed", "output.png" not in remaining)
        _check("Checkpoints subdir removed",
               not os.path.isdir(ckpt_sub))

    # F2: _cleanup_old_trials keeps N best by score, removes rest
    with tempfile.TemporaryDirectory() as td:
        # Create 5 trial dirs with different scores
        for i, score in enumerate([80.0, 60.0, 90.0, 70.0, 50.0]):
            trial_dir = os.path.join(td, f"trial_{i}")
            os.makedirs(trial_dir)
            with open(os.path.join(trial_dir, "summary.json"), "w") as f:
                json.dump({"score": score}, f)

        _cleanup_old_trials(td, keep_best_n=2)

        surviving = sorted(d for d in os.listdir(td) if d.startswith("trial_"))
        _check("Kept 2 trial dirs", len(surviving) == 2)

        # Read surviving scores to verify they're the best
        surviving_scores = []
        for d in surviving:
            with open(os.path.join(td, d, "summary.json")) as f:
                surviving_scores.append(json.load(f)["score"])
        _check("Kept best scores",
               sorted(surviving_scores) == [50.0, 60.0])

    # F3: _cleanup_old_trials with fewer trials than keep_best_n does nothing
    with tempfile.TemporaryDirectory() as td:
        for i in range(2):
            trial_dir = os.path.join(td, f"trial_{i}")
            os.makedirs(trial_dir)
            with open(os.path.join(trial_dir, "summary.json"), "w") as f:
                json.dump({"score": 70.0 + i}, f)

        _cleanup_old_trials(td, keep_best_n=5)
        surviving = [d for d in os.listdir(td) if d.startswith("trial_")]
        _check("No removal when under threshold", len(surviving) == 2)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    test_camera_detect()
    test_profile_loading()
    test_profile_override()
    test_profile_yaml_update()
    test_tune_cases_config()
    test_score_extraction()
    test_checkpoint_numpy_encoder()
    test_checkpoint_dataclass_reconstruction()
    test_plateau_detection()
    test_phase_checkpoint_ids()
    test_no_crs_resolution()
    test_auto_align_profile_arg()
    test_tune_case_cli()
    test_checkpoint_round_trip()
    test_stale_temp_path_fallback()
    test_resolve_case()
    test_constants_sync()
    test_apply_best_filtering()
    test_trial_cleanup()

    print(f"\n{'='*60}")
    print(f"Results: {_passed} passed, {_failed} failed")
    print(f"{'='*60}")
    sys.exit(1 if _failed > 0 else 0)
