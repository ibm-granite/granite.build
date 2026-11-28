# SPDX-License-Identifier: Apache-2.0

"""tbench generic-bridge tests: fakes only (no GPU, no network, no harness)."""

import pytest
from granite_evals.benchmarks import generic, tbench
from granite_evals.registry import RunConfig, get


def _config(tmp_path, dataset="", **options):
    return RunConfig(model="m", output_dir=tmp_path, dataset=dataset, options=options)


def test_dataset_is_required(tmp_path):
    with pytest.raises(SystemExit, match="dataset= is required"):
        generic.TerminalBenchSet(_config(tmp_path))


def test_harness_pin_and_override(tmp_path):
    bench = generic.TerminalBenchSet(
        _config(tmp_path, dataset="harborframework/terminal-bench-2.0")
    )
    assert bench.dataset_source() == ("harborframework/terminal-bench-2.0", None)
    assert bench.dataset.startswith("https://pypi.org/project/harbor/")
    assert bench.dataset_revision.startswith("sha256:")


def test_registry():
    assert get("terminal-bench") is generic.TerminalBenchSet


def test_load_tasks_registry_less_layout(tmp_path):
    """The terminal-bench-2.0 mirror has no registry.json: one task directory
    per task at the root, each with its task.toml."""
    root = tmp_path / "mirror"
    for name in ("b-task", "a-task", "not-a-task"):
        (root / name).mkdir(parents=True)
    (root / "a-task" / "task.toml").write_text("[task]\n")
    (root / "b-task" / "task.toml").write_text("[task]\n")
    (root / "not-a-task" / "other.txt").write_text("x")
    got_root, rows = tbench.load_tasks(str(root), None, tmp_path / "dest")
    assert [r["name"] for r in rows] == ["a-task", "b-task"]
    assert got_root == root


def test_load_tasks_requires_tasks(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(SystemExit, match="no registry.json"):
        tbench.load_tasks(str(root), None, tmp_path / "dest")
