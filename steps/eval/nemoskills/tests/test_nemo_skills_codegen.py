"""HumanEval and MBPP (``nemo_skills_codegen``): registration, pins, the ns row
mapping, upstream-exact program assembly, the ported HumanEval execution
harness, and grading through the ``unsafe-local`` sandbox (no container, no
nemoskills extra); with the extra, the phases split and the gold run end to
end. The two dataset rows below are verbatim from the pinned revisions."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from granite_evals import registry
from granite_evals.benchmarks import nemo_skills as nsb
from granite_evals.benchmarks import nemo_skills_codegen as nsc
from granite_evals.registry import RunConfig

IDS = {
    "human-eval": ("pass@1 accuracy", 1, "passing_tests"),
    "mbpp": ("pass@1 accuracy", 1, "passing_tests"),
}


def bench(bid, tmp_path, **kw):
    return registry.get(bid)(RunConfig(model="m", output_dir=tmp_path, **kw))


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# openai/openai_humaneval @ HUMANEVAL_REVISION, HumanEval/0, verbatim.
HE0 = {
    "task_id": "HumanEval/0",
    "prompt": (
        "from typing import List\n"
        "\n"
        "\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        '    """ Check if in given list of numbers, are any two numbers closer to each other than\n'
        "    given threshold.\n"
        "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
        "    False\n"
        "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
        "    True\n"
        '    """\n'
    ),
    "canonical_solution": (
        "    for idx, elem in enumerate(numbers):\n"
        "        for idx2, elem2 in enumerate(numbers):\n"
        "            if idx != idx2:\n"
        "                distance = abs(elem - elem2)\n"
        "                if distance < threshold:\n"
        "                    return True\n"
        "\n"
        "    return False\n"
    ),
    "test": (
        "\n"
        "\n"
        "METADATA = {\n"
        "    'author': 'jt',\n"
        "    'dataset': 'test'\n"
        "}\n"
        "\n"
        "\n"
        "def check(candidate):\n"
        "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n"
        "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n"
        "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n"
        "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n"
        "    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n"
        "    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n"
        "    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n"
    ),
    "entry_point": "has_close_elements",
}

# google-research-datasets/mbpp @ MBPP_REVISION, config sanitized, task_id 11.
MBPP11 = {
    "source_file": "Benchmark Questions Verification V2.ipynb",
    "task_id": 11,
    "prompt": "Write a python function to remove first and last occurrence of a given character from the string.",
    "code": (
        "def remove_Occ(s,ch): \n"
        "    for i in range(len(s)): \n"
        "        if (s[i] == ch): \n"
        "            s = s[0 : i] + s[i + 1:] \n"
        "            break\n"
        "    for i in range(len(s) - 1,-1,-1):  \n"
        "        if (s[i] == ch): \n"
        "            s = s[0 : i] + s[i + 1:] \n"
        "            break\n"
        "    return s "
    ),
    "test_imports": [],
    "test_list": [
        'assert remove_Occ("hello","l") == "heo"',
        'assert remove_Occ("abcda","a") == "bcd"',
        'assert remove_Occ("PHP","P") == "H"',
    ],
}


def _fence(gen):
    """``extract_completion`` for tests without ns: the fenced block, fence off."""
    return gen.removeprefix("```python\n").removesuffix("\n```")


# -- registration / config ---------------------------------------------------------


@pytest.mark.parametrize("bid", IDS)
def test_registered_as_in_suite(bid, tmp_path):
    import yaml

    cls = registry.get(bid)
    assert issubclass(cls, nsb.NemoSkillsBenchmark) and issubclass(
        cls, nsc.CodegenBenchmark
    )
    assert (cls.metric, cls.default_repeats, cls.ns_metric) == IDS[bid]
    assert cls.extra == "nemoskills" and "nemo-skills" in cls.harness_packages
    assert cls.splittable and len(cls.dataset_revision) == 40
    suite = yaml.safe_load(
        (Path(registry.__file__).parent / "suites" / "extended.yaml").read_text()
    )
    assert {b["id"]: b["metric"] for b in suite["benchmarks"]}[bid] == cls.metric


@pytest.mark.parametrize("bid", IDS)
def test_no_paid_api(bid, tmp_path, monkeypatch):
    """No judge: nothing here may reach a paid endpoint or the meter."""
    from granite_evals import meter

    monkeypatch.setattr(meter, "metered", lambda *a, **k: pytest.fail("paid API used"))
    b = bench(bid, tmp_path)
    assert b.judge is None and not b.judge_args and not b.judge_generation_module
    assert "judge_base_url" not in b.config.options


def test_tools_variants_ride_the_codegen_run(tmp_path):
    from granite_evals.benchmarks import tools_agentic as ta

    for bid, parent in (("human-eval-tools", nsc.HumanEval), ("mbpp-tools", nsc.MBPP)):
        cls = registry.get(bid)
        assert issubclass(cls, parent) and issubclass(cls, ta.ToolBenchmark)
        b = bench(bid, tmp_path)
        assert b.tool_names() == ["python"] and b.metric == "pass@1 accuracy"
        args = b.generation_args([])
        assert "++prompt_config=generic/codegen" in args and "++eval_type=null" in args
        assert "++tool_modules=[granite_evals.agent_tools::SandboxPythonTool]" in args


def test_pins(tmp_path):
    assert bench("human-eval", tmp_path).pins() == {
        nsc.HUMANEVAL_DATASET: nsc.HUMANEVAL_REVISION
    }
    assert bench("mbpp", tmp_path).pins() == {nsc.MBPP_DATASET: nsc.MBPP_REVISION}
    assert (nsc.HUMANEVAL_N, nsc.MBPP_N) == (164, 257)
    assert nsc.MBPP_CONFIG == "sanitized" and nsc.GRADE_IMAGE.startswith("python:")
    assert len(nsc.HUMANEVAL_EXEC_COMMIT) == 40


def test_module_args(tmp_path):
    for bid in IDS:
        b = bench(bid, tmp_path)
        args = b.generation_args([])
        assert (
            args[0] == "++prompt_config=generic/codegen"
            and args.count("++eval_type=null") == 1
        )
        assert "++inference.temperature=null" in args
        assert (
            b.split() == "test"
            and b.generation_module() == "nemo_skills.inference.generate"
        )
        assert not b.uses_judge() and not b.uses_sandbox()
        assert (
            b.metrics_type()
            == "granite_evals.benchmarks.nemo_skills_codegen::PassingTestsMetrics"
        )


# -- prepare -----------------------------------------------------------------------


def _fake_load(monkeypatch, rows_by_source, reads):
    def load_split(source, *, revision=None, split="test", name=None):
        reads.append((source, revision, split, name))
        return rows_by_source[source]

    monkeypatch.setattr(nsc.data, "load_split", load_split)


def test_prepare_maps_the_pinned_hf_rows(tmp_path, monkeypatch):
    reads = []
    _fake_load(monkeypatch, {nsc.HUMANEVAL_DATASET: [HE0]}, reads)
    monkeypatch.setattr(nsc.HumanEval, "expected_rows", 1)  # the fake split's size
    b = bench("human-eval", tmp_path / "he")
    path, prov = b.prepare_data()
    assert reads == [(nsc.HUMANEVAL_DATASET, nsc.HUMANEVAL_REVISION, "test", None)]
    row = nsc.humaneval_row(HE0)
    assert _rows(path) == [row]
    assert row["question"] == row["prompt"].replace("    ", "\t")  # ns's tab quirk
    assert prov == {
        "source": "raw HF dataset, ns row shape",
        "dataset": nsc.HUMANEVAL_DATASET,
        "revision": nsc.HUMANEVAL_REVISION,
        "config": None,
        "rows": 1,
        "sha256": nsb._sha256(path),
    }
    assert path == tmp_path / "he" / "ns-data" / "human-eval" / "test.jsonl"

    reads.clear()
    _fake_load(monkeypatch, {nsc.MBPP_DATASET: [MBPP11]}, reads)
    monkeypatch.setattr(nsc.MBPP, "expected_rows", 1)  # the fake split's size
    path, prov = bench("mbpp", tmp_path / "mb").prepare_data()
    assert reads == [(nsc.MBPP_DATASET, nsc.MBPP_REVISION, "test", "sanitized")]
    row = nsc.mbpp_row(MBPP11)
    assert _rows(path) == [row]
    # evalplus's MBPP prompt: the docstring it builds from the problem + first test
    assert row["prompt"] == f'"""\n{MBPP11["prompt"]}\n{MBPP11["test_list"][0]}\n"""\n'
    assert row["canonical_solution"] == MBPP11["code"]
    assert row["test"] == "\n".join(MBPP11["test_list"])
    assert (
        prov["config"] == "sanitized"
        and prov["rows"] == 1
        and prov["dataset"] == nsc.MBPP_DATASET
    )


def test_prepare_checks_the_pinned_count(tmp_path, monkeypatch):
    sources = {nsc.HUMANEVAL_DATASET: [HE0], nsc.MBPP_DATASET: [MBPP11]}
    _fake_load(monkeypatch, sources, [])
    monkeypatch.setattr(nsc.HumanEval, "expected_rows", 164)
    with pytest.raises(RuntimeError, match="1 rows, expected 164"):
        bench("human-eval", tmp_path / "a").prepare_data()
    monkeypatch.setattr(nsc.MBPP, "expected_rows", 257)
    with pytest.raises(RuntimeError, match="1 rows, expected 257"):
        bench("mbpp", tmp_path / "b").prepare_data()
    # an --dataset hub override drops the pin (and its count check)
    reads = []
    _fake_load(monkeypatch, {**sources, "other/mbpp": [MBPP11, MBPP11]}, reads)
    path, prov = bench("mbpp", tmp_path / "c", dataset="other/mbpp").prepare_data()
    assert reads == [("other/mbpp", None, "test", "sanitized")]
    assert (
        len(_rows(path)) == 2
        and prov["dataset"] == "other/mbpp"
        and prov["revision"] is None
    )


def test_prepare_accepts_a_local_jsonl(tmp_path):
    src = _write(tmp_path / "he.jsonl", [nsc.humaneval_row(HE0)])
    path, prov = bench("human-eval", tmp_path, dataset=str(src)).prepare_data()
    assert (
        path == src and prov["source"] == "local" and prov["sha256"] == nsb._sha256(src)
    )


def test_gold_answers_are_fenced_reference_solutions(tmp_path):
    for bid, raw in (("human-eval", HE0), ("mbpp", MBPP11)):
        b = bench(bid, tmp_path)
        row = b.ns_row(raw)
        server = nsb._GoldServer([row], b.gold_generation)
        assert server.lookup(f"prompt\n\n{row['question']}\n\nanswer") == (
            f"```python\n{row['canonical_solution']}\n```"
        )
        assert server.lookup("nothing matches this") == ""


# -- program assembly ----------------------------------------------------------------


def test_humaneval_program_is_upstreams_assembly():
    row = nsc.humaneval_row(HE0)
    body = row["canonical_solution"]
    # the gold body continues the prompt's signature: upstream's exact concat
    assert nsc.humaneval_program(body, row) == (
        row["prompt"] + body + "\n" + row["test"] + "\ncheck(has_close_elements)"
    )
    # a whole-function completion (evalplus's "solution" mode) is used alone
    whole = "def has_close_elements(numbers, threshold):\n    return False\n"
    assert (
        nsc.humaneval_program(whole, row)
        == whole + "\n" + row["test"] + "\ncheck(has_close_elements)"
    )
    assert nsc._standalone(whole, "has_close_elements")
    assert nsc._standalone(
        "async def has_close_elements():\n    ...\n", "has_close_elements"
    )
    assert not nsc._standalone(body, "has_close_elements")  # a body, not a definition
    assert not nsc._standalone("def other(x):\n    return x\n", "has_close_elements")
    assert not nsc._standalone("def broken(:", "has_close_elements")  # not parseable


def test_mbpp_program_is_evalpluss_assembly():
    row = nsc.mbpp_row(MBPP11)
    fn = row["canonical_solution"]
    assert nsc.mbpp_program(fn, row) == row["prompt"] + fn + "\n" + row["test"]


# -- the grading harness ---------------------------------------------------------------


def _run_grader(tmp_path, programs, mode="on", workers=2, timeout=10.0):
    """The harness as it runs in the sandbox: a script, argv and all."""
    rows = _write(
        tmp_path / "rows.jsonl",
        [{"task_id": str(i), "program": p} for i, p in enumerate(programs)],
    )
    out = tmp_path / "out.jsonl"
    script = tmp_path / "granite_grade.py"
    script.write_text(nsc.GRADE_RUNNER)
    p = subprocess.run(
        [
            sys.executable,
            str(script),
            mode,
            str(rows),
            str(out),
            str(workers),
            str(timeout),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert p.returncode == 0, p.stderr
    lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    return lines[0], lines[1:]


def test_grade_runner_passes_fails_and_times_out(tmp_path):
    row = nsc.humaneval_row(HE0)
    hello, results = _run_grader(
        tmp_path,
        [
            nsc.humaneval_program(row["canonical_solution"], row),  # the gold program
            nsc.humaneval_program(
                "    return True\n", row
            ),  # a wrong body: an assert fails
            "import time\ntime.sleep(30)\n",
            "def broken(:\n",
        ],
        timeout=2.0,
    )
    assert hello["ready"] and hello["network"] == "not requested" and hello["python"]
    assert [r["task_id"] for r in results] == ["0", "1", "2", "3"]
    assert [r["passed"] for r in results] == [True, False, False, False]
    assert results[0]["result"] == "passed"
    assert results[1]["result"].startswith("failed")  # AssertionError has no message
    assert results[2]["result"] == "timed out"
    assert results[3]["result"].startswith("failed")  # a syntax error


def test_grade_runner_runs_the_gold_mbpp_program(tmp_path):
    row = nsc.mbpp_row(MBPP11)
    hello, (r,) = _run_grader(
        tmp_path, [nsc.mbpp_program(row["canonical_solution"], row)]
    )
    assert hello["ready"] and r["passed"] and r["result"] == "passed"


@pytest.mark.skipif(
    sys.platform != "darwin", reason="Linux may or may not allow unshare here"
)
def test_grade_runner_offline_requires_isolation(tmp_path):
    hello, results = _run_grader(tmp_path, ["1\n"], mode="off")
    assert not hello["ready"] and hello["network"].startswith("failed") and not results


# -- grading through the sandbox --------------------------------------------------------


def _he_rows():
    row = nsc.humaneval_row(HE0)
    return [
        {**row, "generation": f"```python\n{row['canonical_solution']}```"},
        {
            **row,
            "task_id": "HumanEval/0-wrong",
            "generation": "```python\n    return True\n```",
        },
        {
            **row,
            "task_id": "HumanEval/0-whole",
            "generation": "```python\ndef has_close_elements(n, t):\n    return False\n```",
        },
    ]


def test_grade_rows_in_the_local_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(nsc, "extract_completion", _fence)
    rows = _he_rows()
    b = bench(
        "human-eval",
        tmp_path,
        phase="score",
        options={"sandbox": "unsafe-local", "grade_network": "on"},
    )
    with b.grading_sandbox():
        b._grade_rows(rows)
    assert rows[0]["is_correct"] and rows[0]["grade_status"] == "passed"
    assert not rows[1]["is_correct"] and rows[1]["grade_status"].startswith("failed")
    assert not rows[2]["is_correct"] and rows[2]["grade_status"].startswith(
        "failed"
    )  # whole-function branch
    r = b._grade_report
    assert (r["backend"], r["image"], r["network_mode"]) == (
        "unsafe-local",
        nsc.GRADE_IMAGE,
        "on",
    )
    assert r["network_isolation"] == "not requested" and r["graded"] == 3
    assert (r["workers"], r["timeout_s"]) == (8, 10.0) and "execution.py" in r[
        "harness"
    ]


def test_grade_network_mode_is_validated(tmp_path):
    b = bench(
        "human-eval",
        tmp_path,
        phase="score",
        options={"sandbox": "unsafe-local", "grade_network": "maybe"},
    )
    with pytest.raises(SystemExit, match="off, best-effort or on"):
        with b.grading_sandbox():
            pass


@pytest.mark.skipif(
    sys.platform != "darwin", reason="Linux may or may not allow unshare here"
)
def test_grading_refuses_an_unisolated_network(tmp_path, monkeypatch):
    monkeypatch.setattr(nsc, "extract_completion", _fence)
    rows = _he_rows()[:1]
    b = bench(
        "human-eval", tmp_path, phase="score", options={"sandbox": "unsafe-local"}
    )  # grade_network off
    with b.grading_sandbox():
        with pytest.raises(SystemExit, match="cannot isolate the grading sandbox"):
            b._grade_rows(rows)
    assert "is_correct" not in rows[0]


# -- with the harness ---------------------------------------------------------------


@pytest.fixture
def ns():
    return pytest.importorskip("nemo_skills")


def _split_run(tmp_path, monkeypatch, phase, events, rows):
    """A human-eval run on fake generations: ns's generate is faked, the grading
    sandbox is the unsafe-local one, everything else (input check, grading,
    metrics) is real."""
    prepared = _write(tmp_path / "data.jsonl", rows)

    def fake_generate(cmd, out, *, log_path, what):
        events.append(("generate", cmd[-1]))
        gens = [f"```python\n{r['canonical_solution']}```" for r in rows]
        nsb._write_jsonl(
            out,
            [
                {**r, "generation": g, "num_generated_tokens": 1}
                for r, g in zip(rows, gens)
            ],
        )

    monkeypatch.setattr(nsb, "_run_ns", fake_generate)
    b = bench(
        "human-eval",
        tmp_path,
        phase=phase,
        options={"sandbox": "unsafe-local", "grade_network": "on"},
    )
    monkeypatch.setattr(b, "prepare_data", lambda: (prepared, {"sha256": "x"}))
    return b.run("" if phase == "score" else "http://127.0.0.1:9/v1", "m")


def test_generate_then_score_and_phase_all(ns, tmp_path, monkeypatch):
    row = nsc.humaneval_row(HE0)
    rows = [row, {**row, "task_id": "HumanEval/0b"}]
    gen = tmp_path / "generation" / "output-rs0.jsonl"

    g = _split_run(tmp_path, monkeypatch, "generate", events := [], rows)
    assert events == [
        ("generate", "++eval_type=null")
    ]  # deferred eval on top of the baked-in null
    assert nsb._unscored(gen).exists() and all(
        "is_correct" not in r for r in _rows(gen)
    )
    assert g["n"] == 2 and "value" not in g and "grading_sandbox" not in g

    s = _split_run(tmp_path, monkeypatch, "score", events := [], rows)
    assert events == []  # nothing generated; only graded
    assert (s["value"], s["n"]) == (1.0, 2)
    assert not nsb._unscored(gen).exists() and all(r["is_correct"] for r in _rows(gen))
    again = _split_run(tmp_path, monkeypatch, "score", events := [], rows)
    assert events == [] and (again["value"], again["n"]) == (
        1.0,
        2,
    )  # graded rows keep their grade

    # phase all: no marker is ever written (eval_type=null is baked in), yet the
    # file is still graded — the base's marker check would leave it unscored
    out = tmp_path / "all"
    a = _split_run(out, monkeypatch, "all", events := [], rows)
    assert events == [("generate", "++inference.tokens_to_generate=null")]
    assert (a["value"], a["n"]) == (1.0, 2)
    assert not nsb._unscored(out / "generation" / "output-rs0.jsonl").exists()
    assert all(r["is_correct"] for r in _rows(out / "generation" / "output-rs0.jsonl"))


@pytest.mark.parametrize(("bid", "raw"), [("human-eval", HE0), ("mbpp", MBPP11)])
def test_gold_end_to_end(ns, tmp_path, bid, raw):
    """Reference answers through ns's real codegen prompt -> generate ->
    extraction -> program assembly -> the ported harness: 100%."""
    data = _write(tmp_path / "data.jsonl", [registry.get(bid).ns_row(raw)])
    kw = {} if bid == "mbpp" else {"repeats": 2}  # both aggregation shapes
    b = bench(
        bid,
        tmp_path / "out",
        dataset=str(data),
        options={"answers": "gold", "sandbox": "unsafe-local", "grade_network": "on"},
        **kw,
    )
    r = b.run("", "")
    assert (r["value"], r["n"], r["answers"]) == (1.0, 1, "gold")
    assert r["ns_metric"]["key"] == "passing_tests" and r["ns_metric"]["raw"] == 100.0
    assert r["pass_at_k"]["pass_at_1"] == 1.0 and r["pass_at_k"]["k"] == b.repeats
    rows = _rows(tmp_path / "out" / "generation" / "output-rs0.jsonl")
    assert all(x["is_correct"] and x["grade_status"] == "passed" for x in rows)
    assert r["grading_sandbox"]["network_isolation"] == "not requested"
    assert r["grading_sandbox"]["graded"] == len(rows)
