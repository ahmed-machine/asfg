"""Manifest-driven block processing helpers."""

from __future__ import annotations

import json
import os

from .models import ModelCache, get_torch_device
from .strip_processing import run_strip_manifest
from .types import AlignmentJob, BlockManifest


def load_block_manifest(path: str) -> BlockManifest:
    """Load a block manifest JSON file."""

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    shared = dict(payload.get("shared", {}))
    jobs = []
    for item in payload.get("jobs", []):
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
    return BlockManifest(manifest_path=path, jobs=jobs, shared_options=shared)


def _normalize_path_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def run_block_manifest(path: str, runner, *, model_cache=None):
    """Run all jobs or strip manifests in a block manifest."""

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    shared = dict(payload.get("shared", {}))
    strip_manifests = payload.get("strips", [])

    created_cache = False
    if model_cache is None:
        device = get_torch_device(shared.get("device", "auto"))
        model_cache = ModelCache(device)
        created_cache = True

    outputs = []
    try:
        for strip in strip_manifests:
            strip_path = strip if isinstance(strip, str) else strip.get("manifest")
            if not strip_path:
                continue
            strip_path = os.path.join(os.path.dirname(path), strip_path) if not os.path.isabs(strip_path) else strip_path
            outputs.extend(run_strip_manifest(strip_path, runner, model_cache=model_cache))

        if payload.get("jobs"):
            manifest = load_block_manifest(path)
            from .strip_processing import _job_to_namespace  # local import to avoid a bigger public API

            for job in manifest.jobs:
                args = _job_to_namespace(job, manifest.shared_options)
                outputs.append(runner(args, model_cache=model_cache))
    finally:
        if created_cache:
            model_cache.close()
    return outputs
