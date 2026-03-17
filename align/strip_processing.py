"""Manifest-driven strip processing helpers."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

from .models import ModelCache, get_torch_device
from .types import AlignmentJob, StripManifest


def load_strip_manifest(path: str) -> StripManifest:
    """Load a strip manifest JSON file."""

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    shared = dict(payload.get("shared", {}))
    raw_jobs = payload.get("jobs") or payload.get("frames") or []
    jobs = []
    for item in raw_jobs:
        jobs.append(
            AlignmentJob(
                input_path=item["input"],
                reference_path=item.get("reference") or shared.get("reference"),
                output_path=item.get("output"),
                anchors_path=item.get("anchors") or shared.get("anchors"),
                metadata_priors=_normalize_path_list(item.get("metadata_priors") or shared.get("metadata_priors")),
                qa_json_path=item.get("qa_json") or shared.get("qa_json"),
                diagnostics_dir=item.get("diagnostics_dir") or shared.get("diagnostics_dir"),
                options={
                    key: value
                    for key, value in item.items()
                    if key not in {"input", "reference", "output", "anchors", "metadata_priors", "qa_json", "diagnostics_dir"}
                },
            )
        )
    return StripManifest(manifest_path=path, jobs=jobs, shared_options=shared)


def _normalize_path_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _job_to_namespace(job: AlignmentJob, shared_options: dict):
    options = dict(shared_options)
    options.update(job.options)
    return SimpleNamespace(
        input=job.input_path,
        reference=job.reference_path,
        output=job.output_path,
        anchors=job.anchors_path,
        metadata_priors=job.metadata_priors,
        qa_json=job.qa_json_path,
        diagnostics_dir=job.diagnostics_dir,
        coarse_pass=int(options.pop("coarse_pass", 0)),
        yes=bool(options.pop("yes", True)),
        device=options.pop("device", "auto"),
        match_res=float(options.pop("match_res", 5.0)),
        tin_tarr_thresh=float(options.pop("tin_tarr_thresh", 1.5)),
        skip_fpp=bool(options.pop("skip_fpp", False)),
        matcher_anchor=options.pop("matcher_anchor", "roma"),
        matcher_dense=options.pop("matcher_dense", "roma"),
        grid_size=int(options.pop("grid_size", 20)),
        grid_iters=int(options.pop("grid_iters", 300)),
        arap_weight=float(options.pop("arap_weight", 1.0)),
        global_search=bool(options.pop("global_search", True)),
        global_search_res=float(options.pop("global_search_res", 40.0)),
        global_search_top_k=int(options.pop("global_search_top_k", 3)),
        metadata_priors_dir=options.pop("metadata_priors_dir", None),
        reference_window=options.pop("reference_window", None),
        mask_provider=options.pop("mask_provider", "coastal_obia"),
        allow_abstain=bool(options.pop("allow_abstain", False)),
        force_global=bool(options.pop("force_global", False)),
    )


def run_strip_manifest(path: str, runner, *, model_cache=None):
    """Run all jobs in a strip manifest with shared model cache reuse."""

    manifest = load_strip_manifest(path)
    created_cache = False
    if model_cache is None:
        device = get_torch_device(manifest.shared_options.get("device", "auto"))
        model_cache = ModelCache(device)
        created_cache = True

    outputs = []
    try:
        for job in manifest.jobs:
            args = _job_to_namespace(job, manifest.shared_options)
            outputs.append(runner(args, model_cache=model_cache))
    finally:
        if created_cache:
            model_cache.close()
    return outputs
