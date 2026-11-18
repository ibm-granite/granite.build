# SPDX-License-Identifier: Apache-2.0

"""mcpmark generic-bridge tests: fakes only (no GPU, no network, no harness)."""

import pytest
from granite_evals.benchmarks import generic
from granite_evals.registry import RunConfig, get


def _config(tmp_path, **options):
    return RunConfig(model="m", output_dir=tmp_path, options=options)


def test_service_is_required(tmp_path):
    with pytest.raises(SystemExit, match="service= is required"):
        generic.McpmarkService(_config(tmp_path))


def test_service_set(tmp_path):
    bench = generic.McpmarkService(_config(tmp_path, service="filesystem"))
    assert bench.service == "filesystem"


def test_registry():
    assert get("mcpmark") is generic.McpmarkService
