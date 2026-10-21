# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the score step's bundled Python (``../src/lcb_score.py``).

These run in Mode 1 only (``make test`` in this directory), under the repo venv,
which has neither LiveCodeBench nor ``datasets``. So they cover the logic this
step owns: reading the generate step's output, refusing a mismatched run, and the
summary. LiveCodeBench's scorer itself is exercised by the build tests.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"


def _load(name):
    """Import ``src/<name>.py`` by path (``src/`` is not a package)."""
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    spec = importlib.util.spec_from_file_location(name, _SRC / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score = _load("lcb_score")
IDS = ["abc400_a", "abc400_b", "abc400_c"]
WINDOW = ("release_v6", "2025-04-05", "2025-04-05")


def _meta(ids=IDS, release="release_v6", start="2025-04-05", end="2025-04-05"):
    return {
        "release_version": release,
        "start_date": start,
        "end_date": end,
        "question_ids_sha256": hashlib.sha256(
            "\n".join(sorted(ids)).encode()
        ).hexdigest(),
    }


def _completions(ids=IDS):
    return [{"question_id": i, "code_list": ["print(1)"]} for i in ids]


def _write(dirpath, completions, meta):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / score.COMPLETIONS).write_text(json.dumps(completions))
    (dirpath / score.META).write_text(json.dumps(meta))
    return dirpath


def test_hash_ignores_order():
    assert score.question_ids_sha256(["b", "a"]) == score.question_ids_sha256(
        ["a", "b"]
    )
    assert score.question_ids_sha256(["a"]) != score.question_ids_sha256(["a", "b"])


def test_matching_run_passes():
    score.check_matches(
        _meta(), _completions(), *WINDOW, benchmark_ids=list(reversed(IDS))
    )


@pytest.mark.parametrize(
    "meta_kw, expected",
    [
        ({"release": "release_v5"}, "release_version"),
        ({"start": "2025-04-06"}, "start_date"),
        ({"end": None}, "end_date"),
    ],
)
def test_window_mismatch_is_refused(meta_kw, expected):
    with pytest.raises(ValueError, match=expected):
        score.check_matches(
            _meta(**meta_kw), _completions(), *WINDOW, benchmark_ids=IDS
        )


def test_empty_dates_equal_none():
    # Config passes "" for an unset date; generate records None. Both mean "no bound".
    score.check_matches(
        _meta(start=None, end=None),
        _completions(),
        "release_v6",
        "",
        "",
        benchmark_ids=IDS,
    )


def test_problem_set_mismatch_names_the_difference():
    with pytest.raises(ValueError) as e:
        score.check_matches(_meta(), _completions(IDS[:2]), *WINDOW, benchmark_ids=IDS)
    assert "2 completions vs 3 problems" in str(e.value) and "abc400_c" in str(e.value)


def test_duplicate_question_id_is_refused():
    dup = _completions() + [{"question_id": "abc400_a", "code_list": ["x"]}]
    with pytest.raises(ValueError, match="more than once"):
        score.check_matches(_meta(), dup, *WINDOW, benchmark_ids=IDS)


def test_load_completions_reads_the_generate_output(tmp_path):
    d = _write(tmp_path / "c", _completions(), _meta())
    completions, meta = score.load_completions(d)
    assert [c["question_id"] for c in completions] == IDS and meta[
        "release_version"
    ] == "release_v6"


@pytest.mark.parametrize("missing", ["completions.json", "generation_meta.json"])
def test_load_completions_missing_file(tmp_path, missing):
    d = _write(tmp_path / "c", _completions(), _meta())
    (d / missing).unlink()
    with pytest.raises(ValueError, match="eval/livecodebench-generate"):
        score.load_completions(d)


@pytest.mark.parametrize(
    "bad", [[], {"a": 1}, [{"question_id": "x"}], [{"code_list": []}]]
)
def test_load_completions_bad_shape(tmp_path, bad):
    d = _write(tmp_path / "c", bad, _meta())
    with pytest.raises(ValueError):
        score.load_completions(d)


def test_summary_by_difficulty():
    eval_all = [
        {"question_id": "a", "difficulty": "easy", "pass@1": 1.0},
        {"question_id": "b", "difficulty": "easy", "pass@1": 0.5},
        {"question_id": "c", "difficulty": "hard", "pass@1": 0.0},
    ]
    s = score.summarize(
        eval_all, {"model": "m", "n": 2}, {"release_version": "release_v6"}
    )
    assert s["problems"] == 3 and s["pass@1"] == pytest.approx(0.5)
    assert s["by_difficulty"]["easy"] == {"problems": 2, "pass@1": 0.75}
    assert s["by_difficulty"]["hard"]["pass@1"] == 0.0
    assert s["generation"]["model"] == "m" and s["release_version"] == "release_v6"


def test_dataset_dir_check(tmp_path):
    ds = _load("lcb_dataset")
    with pytest.raises(ValueError, match="not found"):
        ds.check_dataset_dir(str(tmp_path / "nope"))
    with pytest.raises(ValueError, match=ds.LOADING_SCRIPT):
        ds.check_dataset_dir(str(tmp_path))
    (tmp_path / ds.LOADING_SCRIPT).write_text("")
    assert ds.check_dataset_dir(str(tmp_path)) == tmp_path


def test_shim_copies_are_identical():
    """Each step ships its own copy of lcb_dataset.py (src/ is published per step)."""
    other = (
        _SRC.parents[2]
        / "livecodebench-generate"
        / "skypilot"
        / "src"
        / "lcb_dataset.py"
    )
    assert (_SRC / "lcb_dataset.py").read_bytes() == other.read_bytes()
