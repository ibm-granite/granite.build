"""The results.json every step emits as its ``granite_results`` artifact.

One file per benchmark run, one schema across benchmarks, so a recipe-level
report can join them without knowing any benchmark's internals.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from granite_evals import __version__
from granite_evals.registry import Benchmark

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
RESULTS_FILE = "results.json"
GENERATION_FILE = "generation.json"
"""What ``--phase generate`` writes instead: the same record with ``value``
None, for the score phase to check its output dir against."""


def _package_versions(names: list[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def write_results(
    benchmark: Benchmark,
    outcome: dict[str, Any],
    *,
    started: float,
    served_model_name: str,
    generation: dict[str, Any] | None = None,
) -> Path:
    config = benchmark.config
    record = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark.id,
        "metric": benchmark.metric,
        "phase": config.phase,
        "value": outcome.pop("value", None),
        "n": outcome.pop("n"),
        "repeats": benchmark.repeats,
        "smoke": config.limit is not None,
        "limit": config.limit,
        "model": config.model,
        "served_model_name": served_model_name,
        "dataset": outcome.pop("dataset", config.dataset),
        "dataset_revision": outcome.pop("dataset_revision", config.dataset_revision),
        "options": config.options,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "duration_s": round(time.time() - started, 1),
        "versions": {
            "granite-evals": __version__,
            "python": platform.python_version(),
            **_package_versions(["vllm", "datasets", *benchmark.harness_packages]),
        },
        "details": outcome,
    }
    if generation is not None:
        record["generation"] = {
            k: generation.get(k) for k in ("served_model_name", "options", "started_at", "duration_s", "versions", "job")
        }
    if config.phase == "generate":
        record["job"] = os.environ.get("LSB_JOBID", "")
    path = config.output_dir / (GENERATION_FILE if config.phase == "generate" else RESULTS_FILE)
    path.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n")
    return path


def read_generation(benchmark: Benchmark) -> dict[str, Any]:
    """The generation record the score phase grades, checked against this run."""
    config = benchmark.config
    path = config.output_dir / GENERATION_FILE
    if not path.exists():
        raise SystemExit(f"{benchmark.id}: no {path}; run --phase generate on this output dir first")
    record = json.loads(path.read_text())
    ours = {
        "benchmark": benchmark.id,
        "model": config.model,
        "limit": config.limit,
        "repeats": benchmark.repeats,
        "dataset": config.dataset,
        "dataset_revision": config.dataset_revision,
    }
    # A benchmark may resolve the dataset itself (record: its pin); only an explicit
    # override has to match.
    if not config.dataset:
        ours.pop("dataset")
        ours.pop("dataset_revision")
    wrong = {k: (record.get(k), v) for k, v in ours.items() if record.get(k) != v}
    if wrong:
        diff = ", ".join(f"{k}: generated {g!r}, scoring {s!r}" for k, (g, s) in wrong.items())
        raise SystemExit(f"{benchmark.id}: {path} was generated with other settings ({diff})")
    # Options may differ on purpose (a scoring timeout, another judge), so only warn:
    # a generation-side one (agent=gold, sampling) makes the score describe other outputs.
    generated = record.get("options") or {}
    if changed := sorted(k for k in {*generated, *config.options} if generated.get(k) != config.options.get(k)):
        log.warning("%s: options differ from the generation's: %s", benchmark.id,
                    ", ".join(f"{k} {generated.get(k)!r} -> {config.options.get(k)!r}" for k in changed))
    return record
