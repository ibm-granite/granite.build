# SPDX-License-Identifier: Apache-2.0

"""swebench generic-bridge tests: fakes only (no GPU, no network, no harness)."""

import pytest
from granite_evals.benchmarks import generic
from granite_evals.registry import RunConfig, get


def _config(tmp_path, dataset="", **options):
    return RunConfig(model="m", output_dir=tmp_path, dataset=dataset, options=options)


def test_dataset_is_required(tmp_path):
    with pytest.raises(SystemExit, match="dataset= is required"):
        generic.SWEBenchSet(_config(tmp_path))


def test_harness_pin_and_override(tmp_path):
    bench = generic.SWEBenchSet(_config(tmp_path, dataset="org/custom-set"))
    assert bench.dataset_source() == ("org/custom-set", None)
    assert bench.dataset.startswith("https://pypi.org/project/swebench/")
    assert bench.dataset_revision.startswith("sha256:")


def test_registry():
    assert get("swebench") is generic.SWEBenchSet
