# SPDX-License-Identifier: Apache-2.0

"""tau generic-bridge tests: fakes only (no GPU, no network, no harness)."""

import pytest
from granite_evals.benchmarks import generic
from granite_evals.registry import RunConfig, get


def _config(tmp_path, **options):
    return RunConfig(model="m", output_dir=tmp_path, options=options)


SHIPPED = ["airline", "banking_knowledge", "retail", "telecom"]
REAL_SHIPPED = generic.shipped_domains  # the autouse fixture fakes it


@pytest.fixture(autouse=True)
def _shipped(monkeypatch):
    monkeypatch.setattr(generic, "shipped_domains", lambda source, revision: list(SHIPPED))


def test_an_unknown_domain_is_refused_at_startup(tmp_path):
    with pytest.raises(SystemExit) as e:
        generic.Tau3Domains(_config(tmp_path, domains="airline,airlines"))
    msg = str(e.value)
    assert "domain(s) airlines are not in the pinned data" in msg
    assert "ships airline, banking_knowledge, retail, telecom" in msg


def test_shipped_domains_reads_the_data_dirs(tmp_path, monkeypatch):
    root = tmp_path / "data" / "tau2" / "domains"
    for name in ("retail", "airline"):
        (root / name).mkdir(parents=True)
    (root / "README.md").write_text("not a domain")
    seen = []
    monkeypatch.setattr(
        generic,
        "resolve_data_dir",
        lambda source, revision: seen.append((source, revision)) or tmp_path / "data",
    )
    assert REAL_SHIPPED("repo", "c0ffee") == ["airline", "retail"]
    assert seen == [("repo", "c0ffee")]


def test_domains_is_required(tmp_path):
    with pytest.raises(SystemExit, match="domains= is required"):
        generic.Tau3Domains(_config(tmp_path))


def test_domains_parsed_and_set(tmp_path):
    bench = generic.Tau3Domains(_config(tmp_path, domains="airline, retail,airline"))
    assert bench.domains == ("airline", "retail")


def test_registry():
    assert get("tau3") is generic.Tau3Domains
