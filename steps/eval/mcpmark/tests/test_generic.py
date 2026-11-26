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


def test_declares_its_metric():
    # `granite-evals list` (run at image build) prints every id's metric; the
    # Mcpmark base leaves it to the per-service subclasses.
    assert generic.McpmarkService.metric == "pass@1"


@pytest.mark.parametrize(
    "service", ["github", "notion", "insforge", "supabase", "playwright_webarena"]
)
def test_services_the_image_cannot_run_are_refused(tmp_path, service):
    """The pinned checkout ships these, but the image sets none up and has no
    pinned task count for them (N_TASKS): refused up front, not a KeyError."""
    with pytest.raises(SystemExit, match="filesystem, playwright, postgres"):
        generic.McpmarkService(_config(tmp_path, service=service))


@pytest.mark.parametrize("service", ["filesystem", "postgres", "playwright"])
def test_the_curated_services_are_accepted(tmp_path, service):
    assert generic.McpmarkService(_config(tmp_path, service=service)).service == service
    assert generic.SERVICES == ("filesystem", "playwright", "postgres")
