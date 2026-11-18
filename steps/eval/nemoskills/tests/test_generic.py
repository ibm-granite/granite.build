# SPDX-License-Identifier: Apache-2.0

"""nemoskills generic-bridge tests: fakes only (no GPU, no network, no ns)."""

import json

import pytest
from granite_evals.benchmarks import generic
from granite_evals.benchmarks.nemo_skills import NS_COMMIT, NS_REPO
from granite_evals.registry import RunConfig, get


def _config(tmp_path, dataset="", **options):
    return RunConfig(model="m", output_dir=tmp_path, dataset=dataset, options=options)


def test_benchmark_is_required(tmp_path):
    with pytest.raises(SystemExit, match="benchmark= is required"):
        generic.Nemoskills(_config(tmp_path))


def test_ns_benchmark_and_metric_option(tmp_path):
    bench = generic.Nemoskills(_config(tmp_path, **{"benchmark": "math-500"}))
    assert bench.ns_benchmark == "math-500"
    assert bench.ns_metric == "symbolic_correct"
    bench = generic.Nemoskills(
        _config(tmp_path, **{"benchmark": "ifbench", "metric": "if"})
    )
    assert bench.ns_metric == "if"


def test_dataset_must_be_local(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="local path"):
        generic.Nemoskills(
            _config(tmp_path, dataset="some/hub-id", **{"benchmark": "math-500"})
        )
    data = tmp_path / "prepared.jsonl"
    data.write_text(json.dumps({"problem": "1+1", "expected_answer": "2"}) + "\n")
    # split() reads the ns module (not installed here); the local-path branch
    # only needs the name it composes.
    monkeypatch.setattr(generic.Nemoskills, "split", lambda self: "test", raising=False)
    bench = generic.Nemoskills(
        _config(tmp_path, dataset=str(data), **{"benchmark": "math-500"})
    )
    prepared, prov = bench.prepare_data()
    assert prepared == data and prov["source"] == "local"


def test_pins():
    assert generic.Nemoskills.dataset == NS_REPO
    assert generic.Nemoskills.dataset_revision == NS_COMMIT


def test_registry():
    assert get("nemoskills") is generic.Nemoskills
