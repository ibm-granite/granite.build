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
    ds = _load("lcb_dataset")
    assert ds.question_ids_sha256(["b", "a"]) == ds.question_ids_sha256(["a", "b"])
    assert ds.question_ids_sha256(["a"]) != ds.question_ids_sha256(["a", "b"])


def test_scripts_use_the_one_shared_hash():
    """generate and score must hash with lcb_dataset's function, not their own."""
    for script in (
        _SRC / "lcb_score.py",
        _SRC.parents[2]
        / "livecodebench-generate"
        / "skypilot"
        / "src"
        / "lcb_generate.py",
    ):
        assert "def question_ids_sha256" not in script.read_text(), script


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


# question_ids() must give the same ids, in the same order, as decoding the window,
# without decoding it. Run against the committed fixture with the real `datasets`
# library; only LiveCodeBench's module (which loads the dataset) is stubbed.
_FIXTURE = Path(__file__).resolve().parent.parent / "test-data" / "fixture"


@pytest.fixture
def fixture_loader(monkeypatch, tmp_path):
    datasets = pytest.importorskip("datasets")
    # The step pins datasets==3.5.0; 4.x dropped loading scripts, which the
    # LiveCodeBench dataset is. Under 4.x this check can't run, so skip it there.
    if int(datasets.__version__.split(".")[0]) >= 4:
        pytest.skip(
            f"needs the step's datasets 3.x for loading scripts, not {datasets.__version__}"
        )
    import types

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    script = str(_FIXTURE / "code_generation_lite" / "code_generation_lite.py")

    def load_dataset(path, *args, **kwargs):
        assert path == "livecodebench/code_generation_lite"
        return datasets.load_dataset(script, *args, **kwargs)

    class CodeGenerationProblem:
        def __init__(self, **row):
            self.question_id = row["question_id"]

    module = types.ModuleType("lcb_runner.benchmarks.code_generation")
    module.load_dataset = load_dataset
    module.CodeGenerationProblem = CodeGenerationProblem
    for name in ("lcb_runner", "lcb_runner.benchmarks"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "lcb_runner.benchmarks.code_generation", module)
    return module


@pytest.mark.parametrize(
    "start, end, expected",
    [
        (None, None, ["fx_sum", "fx_max", "fx_rev", "fx_square"]),
        ("2099-01-01", "2099-01-01", ["fx_sum", "fx_max", "fx_rev", "fx_square"]),
        ("2025-04-05", "2025-04-05", []),
    ],
)
def test_question_ids_match_a_full_decode(fixture_loader, start, end, expected):
    dataset = _load("lcb_dataset")
    ids = dataset.question_ids("release_v6", start_date=start, end_date=end)
    decoded = dataset._windowed_loader(fixture_loader)("release_v6", start, end)
    assert ids == [p.question_id for p in decoded] == expected


def test_main_reads_ids_not_problems(monkeypatch, tmp_path):
    """main() reads only the window's ids; LiveCodeBench's scorer decodes the problems.

    lcb_dataset is stubbed: its ids function is called, and any attempt to decode the
    benchmark from main() (through LiveCodeBench's loader) fails the test.
    """
    import types

    ids = ["abc400_a", "abc400_b"]
    calls = []
    stub = types.SimpleNamespace(
        install=lambda d: calls.append(("install", d)),
        question_ids=lambda *a, **k: calls.append(("question_ids", a, k)) or ids,
        question_ids_sha256=_load("lcb_dataset").question_ids_sha256,
    )
    monkeypatch.setattr(score, "lcb_dataset", stub)
    loader = types.ModuleType("lcb_runner.benchmarks.code_generation")

    def must_not_decode(*a, **k):
        raise AssertionError("main() decoded the benchmark itself")

    loader.load_code_generation_dataset = must_not_decode
    for name in ("lcb_runner", "lcb_runner.benchmarks"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "lcb_runner.benchmarks.code_generation", loader)

    out = tmp_path / "out"

    def fake_scorer(module, run_name):
        assert module == "lcb_runner.runner.custom_evaluator"
        rows = [{"question_id": i, "difficulty": "easy", "pass@1": 1.0} for i in ids]
        (out / f"completions_{score.SCENARIO}_output_eval_all.json").write_text(
            json.dumps(rows)
        )

    monkeypatch.setattr(score.runpy, "run_module", fake_scorer)
    comp = _write(tmp_path / "comp", _completions(ids), _meta(ids))
    rc = score.main(
        [
            "--completions-dir",
            str(comp),
            "--dataset-dir",
            str(tmp_path),
            "--output-dir",
            str(out),
            "--release-version",
            "release_v6",
            "--start-date",
            "2025-04-05",
            "--end-date",
            "2025-04-05",
        ]
    )
    assert rc == 0
    assert [c[0] for c in calls] == ["install", "question_ids"]
