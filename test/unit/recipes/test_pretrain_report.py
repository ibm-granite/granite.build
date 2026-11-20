#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""recipes/pretrain/report.py, run as a subprocess over fixture result trees.

The script joins the ``results.json`` files the granite-evals steps write (schema:
``steps/eval/src/core/granite_evals/results.py``, ``write_results``) under
``<root>/ckpt_<N>/<entry>/results.json`` into summary.csv, per_task.csv and
summary.md.

The fixtures below are written as real files so the schema under test is visible
here: ``_record`` mirrors ``write_results``'s record key for key, and the two
per-task detail shapes are the ones that really exist in this repo --
``details.per_task`` as ``{task: {"accuracy": ..., "n": ...}}`` (nemoskills RULER,
steps/eval/nemoskills/src/nemo_skills_ruler.py) and ``details.per_language`` /
``details.per_subject`` as ``{key: {"exact_match": ..., "n": ...}}`` (the lmeval
family, steps/eval/lmeval/src/chat_if.py ``summarize``).

The script is run the way the eval-combined target will run it -- a bare CPython
with nothing installed -- so it is invoked with ``-I`` and must never import
anything outside the standard library.
"""

import csv
import json
import pathlib
import subprocess
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "recipes" / "pretrain" / "report.py"


def _record(benchmark, metric, value, n, details=None):
    """One granite-evals results.json record, keyed as write_results writes it."""
    return {
        "schema_version": 1,
        "benchmark": benchmark,
        "metric": metric,
        "phase": "all",
        "value": value,
        "n": n,
        "repeats": 1,
        "smoke": False,
        "limit": None,
        "model": "/runs/exp/run/export/ckpt_2/hf",
        "served_model_name": "hf",
        "dataset": "li-lab/MMLU-ProX-Lite",
        "dataset_revision": "0c0ffee",
        "options": {},
        "started_at": "2026-01-01T00:00:00Z",
        "duration_s": 12.5,
        "versions": {"granite-evals": "0.1.0", "python": "3.12.8"},
        "details": {} if details is None else details,
    }


def _write(root, rung, entry, record):
    out = root / rung / entry
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    if isinstance(record, str):  # malformed on purpose
        path.write_text(record)
    else:
        path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def _run(root, *args, expect=0):
    proc = subprocess.run(
        [sys.executable, "-I", str(_SCRIPT), str(root), *args],
        capture_output=True,
        text=True,
    )
    assert (
        proc.returncode == expect
    ), f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    return proc


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


# --- the plan's three cases ------------------------------------------------


@pytest.fixture
def tree(tmp_path):
    """Two rungs x two entries; one entry carries details.per_task, one does not."""
    root = tmp_path / "run"
    for rung, (core, prox) in {
        "ckpt_2": (0.25, 0.1),
        "ckpt_10": (0.5, 0.2),
    }.items():
        _write(
            root,
            rung,
            "base-core",
            _record(
                "base-core",
                "accuracy",
                core,
                400,
                details={
                    "per_task": {
                        "arc_easy": {"accuracy": core + 0.05, "n": 200},
                        "hellaswag": {"accuracy": core - 0.05, "n": 200},
                    }
                },
            ),
        )
        _write(
            root,
            rung,
            "mmlu-prox-lite",
            _record("mmlu-prox-lite", "exact match", prox, 60),
        )
    return root


def test_summary_csv_is_rungs_by_entries(tree):
    _run(tree)
    rows = _csv(tree / "summary.csv")
    assert rows[0] == ["rung", "base-core", "mmlu-prox-lite"]
    assert rows[1:] == [["2", "0.25", "0.1"], ["10", "0.5", "0.2"]]


def test_per_task_csv_carries_tasks_and_n(tree):
    _run(tree)
    rows = _csv(tree / "per_task.csv")
    assert rows[0] == ["rung", "entry", "task", "metric", "value", "n"]
    body = rows[1:]
    assert ["2", "base-core", "arc_easy", "accuracy", "0.3", "200"] in body
    assert ["10", "base-core", "hellaswag", "accuracy", "0.45", "200"] in body
    # The headline of every entry is in the long form too, so per_task.csv alone
    # carries every value and every n.
    assert ["2", "mmlu-prox-lite", "(headline)", "exact match", "0.1", "60"] in body
    # An entry with no per-task detail contributes its headline row only.
    assert [r for r in body if r[1] == "mmlu-prox-lite" and r[2] != "(headline)"] == []


def test_summary_md_has_the_headline_table(tree):
    _run(tree)
    text = (tree / "summary.md").read_text()
    assert "| rung | base-core | mmlu-prox-lite |" in text
    assert "| 10 | 0.5 | 0.2 |" in text
    # The metric each column is measured in is named, since entries differ.
    assert "exact match" in text and "accuracy" in text


def test_rungs_sort_numerically(tmp_path):
    root = tmp_path / "run"
    for n in (2, 10, 1000, 30):
        _write(root, f"ckpt_{n}", "mmlu", _record("mmlu", "accuracy", n / 10000.0, 8))
    _run(root)
    assert [r[0] for r in _csv(root / "summary.csv")[1:]] == ["2", "10", "30", "1000"]
    assert [r[0] for r in _csv(root / "per_task.csv")[1:]] == ["2", "10", "30", "1000"]


def test_missing_results_is_a_blank_cell(tmp_path):
    root = tmp_path / "run"
    _write(root, "ckpt_2", "mmlu", _record("mmlu", "accuracy", 0.4, 8))
    _write(root, "ckpt_2", "gsm8k", _record("gsm8k", "accuracy", 0.1, 8))
    _write(root, "ckpt_4", "mmlu", _record("mmlu", "accuracy", 0.6, 8))
    (root / "ckpt_4" / "gsm8k").mkdir(parents=True)  # dispatched, nothing written
    proc = _run(root)
    rows = _csv(root / "summary.csv")
    assert rows[0] == ["rung", "gsm8k", "mmlu"]
    assert rows[1:] == [["2", "0.1", "0.4"], ["4", "", "0.6"]]
    assert "gsm8k" in proc.stderr  # the gap is reported, not hidden
    assert "no results.json" in (root / "summary.md").read_text()


# --- tolerated oddities ----------------------------------------------------


def test_empty_root_gives_valid_empty_outputs(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    _run(root)
    assert _csv(root / "summary.csv") == [["rung"]]
    assert _csv(root / "per_task.csv") == [
        ["rung", "entry", "task", "metric", "value", "n"]
    ]
    assert "no ckpt_" in (root / "summary.md").read_text()


def test_root_that_does_not_exist_is_an_error_not_a_traceback(tmp_path):
    proc = _run(tmp_path / "nope", expect=2)
    assert "Traceback" not in proc.stderr
    assert "nope" in proc.stderr


def test_malformed_json_is_tolerated_and_flagged(tmp_path):
    root = tmp_path / "run"
    _write(root, "ckpt_2", "mmlu", _record("mmlu", "accuracy", 0.4, 8))
    _write(root, "ckpt_2", "gsm8k", "{not json at all")
    proc = _run(root)
    assert _csv(root / "summary.csv")[1:] == [["2", "", "0.4"]]
    assert "gsm8k" in proc.stderr
    assert "unreadable" in (root / "summary.md").read_text()


def test_rung_with_no_entry_directories(tmp_path):
    root = tmp_path / "run"
    _write(root, "ckpt_2", "mmlu", _record("mmlu", "accuracy", 0.4, 8))
    (root / "ckpt_4").mkdir()
    _run(root)
    assert _csv(root / "summary.csv")[1:] == [["2", "0.4"], ["4", ""]]


def test_empty_per_task_section(tmp_path):
    root = tmp_path / "run"
    _write(
        root,
        "ckpt_2",
        "mmlu",
        _record("mmlu", "accuracy", 0.4, 8, details={"per_task": {}}),
    )
    _run(root)
    assert _csv(root / "per_task.csv")[1:] == [
        ["2", "mmlu", "(headline)", "accuracy", "0.4", "8"]
    ]


def test_string_value_is_carried_verbatim(tmp_path):
    root = tmp_path / "run"
    _write(root, "ckpt_2", "mmlu", _record("mmlu", "accuracy", "n/a", None))
    _run(root)
    assert _csv(root / "summary.csv")[1:] == [["2", "n/a"]]
    assert _csv(root / "per_task.csv")[1:] == [
        ["2", "mmlu", "(headline)", "accuracy", "n/a", ""]
    ]


def test_null_value_is_a_blank_cell(tmp_path):
    # phase generate leaves value None; so does a half-written record.
    root = tmp_path / "run"
    _write(root, "ckpt_2", "mmlu", _record("mmlu", "accuracy", None, 8))
    proc = _run(root)
    assert _csv(root / "summary.csv")[1:] == [["2", ""]]
    assert "no value" in proc.stderr


def test_commas_and_quotes_are_csv_quoted(tmp_path):
    root = tmp_path / "run"
    entry = 'odd,"entry"'
    _write(
        root,
        "ckpt_2",
        entry,
        _record(
            entry,
            "accuracy",
            0.5,
            4,
            details={"per_task": {'a,b "c"': {"accuracy": 0.5, "n": 4}}},
        ),
    )
    _run(root)
    assert _csv(root / "summary.csv") == [["rung", entry], ["2", "0.5"]]
    assert ["2", entry, 'a,b "c"', "accuracy", "0.5", "4"] in _csv(
        root / "per_task.csv"
    )


def test_per_language_and_per_subject_details(tmp_path):
    # The lmeval family's shape: no per_task, but per_language and per_subject.
    root = tmp_path / "run"
    _write(
        root,
        "ckpt_2",
        "mmlu-prox-lite",
        _record(
            "mmlu-prox-lite",
            "exact match",
            0.3,
            20,
            details={
                "per_language": {"en": {"exact_match": 0.4, "n": 10}},
                "per_subject": {"math": {"exact_match": 0.2, "n": 10}},
            },
        ),
    )
    _run(root)
    body = _csv(root / "per_task.csv")[1:]
    assert ["2", "mmlu-prox-lite", "language:en", "exact_match", "0.4", "10"] in body
    assert ["2", "mmlu-prox-lite", "subject:math", "exact_match", "0.2", "10"] in body


def test_benchmark_name_mismatch_is_flagged_but_kept(tmp_path):
    # A crashed eval can leave a SIBLING eval's results.json in the dir; the value
    # stays in summary.csv (an entry name need not equal a benchmark id) but the
    # mismatch is named in stderr and in summary.md.
    root = tmp_path / "run"
    _write(root, "ckpt_2", "mmlu", _record("gsm8k", "accuracy", 0.9, 8))
    proc = _run(root)
    assert _csv(root / "summary.csv")[1:] == [["2", "0.9"]]
    assert "gsm8k" in proc.stderr
    md = (root / "summary.md").read_text()
    assert "gsm8k" in md and "0.9" in md


# --- where the outputs land ------------------------------------------------


def test_out_dir_option(tmp_path):
    root = tmp_path / "run"
    _write(root, "ckpt_2", "mmlu", _record("mmlu", "accuracy", 0.4, 8))
    out = tmp_path / "report"
    _run(root, "--out", str(out))
    for name in ("summary.csv", "per_task.csv", "summary.md"):
        assert (out / name).is_file()
        assert not (root / name).exists()


def test_stdlib_only(tmp_path):
    """No repo import, and nothing a bare python:3.12-slim lacks."""
    text = _SCRIPT.read_text()
    assert text.startswith("# SPDX-License-Identifier: Apache-2.0")
    for banned in ("pandas", "yaml", "numpy", "gbserver", "gbcli", "gbcommon"):
        assert banned not in text, banned


def test_heredoc_safe(tmp_path):
    """The eval-combined target ships this file inline in a shell heredoc."""
    text = _SCRIPT.read_text()
    assert "\t" not in text
    for hazard in ("$", "`", "\\\\"):
        assert hazard not in text, hazard
