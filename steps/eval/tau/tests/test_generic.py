# SPDX-License-Identifier: Apache-2.0

"""tau generic-bridge tests: fakes only (no GPU, no network, no harness)."""

import pytest
from granite_evals.benchmarks import generic
from granite_evals.registry import RunConfig, get


def _config(tmp_path, **options):
    return RunConfig(model="m", output_dir=tmp_path, options=options)


def test_domains_is_required(tmp_path):
    with pytest.raises(SystemExit, match="domains= is required"):
        generic.Tau3Domains(_config(tmp_path))


def test_domains_parsed_and_set(tmp_path):
    bench = generic.Tau3Domains(_config(tmp_path, domains="airline, retail,airline"))
    assert bench.domains == ("airline", "retail")


def test_registry():
    assert get("tau3") is generic.Tau3Domains
