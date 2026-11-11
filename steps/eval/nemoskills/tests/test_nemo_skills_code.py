"""NeMo-Skills code benchmarks (LiveCodeBench v6, SciCode): config, data handling,
gold servers, sandbox checks, and ns's metrics/args (with the nemoskills extra)."""

import hashlib
import json
import sys
from pathlib import Path

import httpx
import pytest

from granite_evals import registry
from granite_evals.benchmarks import nemo_skills as nsb
from granite_evals.benchmarks import nemo_skills_code as nsc
from granite_evals.benchmarks import scicode_prefill as scp
from granite_evals.registry import RunConfig

IDS = {
    "livecodebench-v6": ("pass@1 accuracy", 1, "accuracy"),
    "scicode": ("pass@1 subtask accuracy", 1, "subtask_accuracy"),
}


def bench(bid, tmp_path, **kw):
    return registry.get(bid)(RunConfig(model="m", output_dir=tmp_path, **kw))


@pytest.mark.parametrize("bid", IDS)
def test_registered_as_in_suite(bid):
    cls = registry.get(bid)
    assert issubclass(cls, nsb.NemoSkillsBenchmark)
    assert (cls.metric, cls.default_repeats, cls.ns_metric) == IDS[bid]
    assert cls.extra == "nemoskills" and "nemo-skills" in cls.harness_packages
    assert len(cls.dataset_revision) == 40


@pytest.mark.parametrize("bid", IDS)
def test_no_paid_api(bid, tmp_path, monkeypatch):
    """No judge: nothing here may reach a paid endpoint or the meter."""
    from granite_evals import meter

    monkeypatch.setattr(meter, "metered", lambda *a, **k: pytest.fail("paid API used"))
    b = bench(bid, tmp_path)
    assert b.judge is None and not b.judge_args and not b.judge_generation_module
    assert "judge_base_url" not in b.config.options


def test_pins(tmp_path):
    assert bench("livecodebench-v6", tmp_path).pins() == {nsc.LCB_DATASET: nsc.LCB_REVISION}
    assert bench("scicode", tmp_path).pins() == {nsc.SCICODE_DATASET: nsc.SCICODE_REVISION}


# -- LiveCodeBench -------------------------------------------------------------


def _lcb_rows():
    return [
        {
            "task_id": tid,
            "question_content": f"problem {tid}",
            "starter_code": "",
            "public_test_cases": "[big]",
            "private_test_cases": "big",
            "metadata": "{}",
            "release_version": "v6",
        }
        for tid in ("abc365_a", "zzz999_a", "3533")
    ]


def _write(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_lcb_tests_file_and_slim_input(tmp_path):
    b = bench("livecodebench-v6", tmp_path, limit=2)
    rows = b.load_rows(_write(tmp_path / "in.jsonl", _lcb_rows()))
    assert [r["task_id"] for r in rows] == ["abc365_a", "zzz999_a"]  # --limit: first N
    assert all(not set(nsc.LCB_TEST_COLUMNS) & set(r) for r in rows)
    full = [json.loads(line) for line in b.tests_file().read_text().splitlines()]
    assert [r["task_id"] for r in full] == ["abc365_a", "zzz999_a"] and full[0]["private_test_cases"] == "big"
    assert "question" not in rows[0]


def test_lcb_gold_uses_hand_solutions(tmp_path):
    b = bench("livecodebench-v6", tmp_path, options={"answers": "gold"})
    rows = b.load_rows(_write(tmp_path / "in.jsonl", _lcb_rows()))
    assert sorted(r["task_id"] for r in rows) == ["3533", "abc365_a"]
    server = nsb._GoldServer(rows, b.gold_generation)
    text = server.lookup("system...\nproblem 3533\n\nformat")
    assert text.startswith("```python\n") and "finalPositionOfSnake" in text and text.endswith("```")


def test_lcb_hand_solutions_run():
    """The reference programs themselves, on the problems' sample tests."""
    import subprocess

    samples = {
        "abc365_a": [("2023\n", "365"), ("1992\n", "366"), ("1800\n", "365"), ("1600\n", "366")],
        "abc365_b": [("4\n8 2 5 1\n", "3"), ("8\n1 2 3 4 5 10 9 11\n", "6")],
        "abc366_b": [("3\nabc\nde\nfghi\n", "fda\ngeb\nh*c\ni")],
    }
    for tid, cases in samples.items():
        for stdin, want in cases:
            out = subprocess.run([sys.executable, "-c", nsc.LCB_GOLD[tid]], input=stdin, capture_output=True, text=True)
            assert out.stdout.strip() == want, (tid, stdin, out.stderr)
    ns: dict = {}
    exec(nsc.LCB_GOLD["3533"], ns)
    assert ns["Solution"]().finalPositionOfSnake(2, ["RIGHT", "DOWN"]) == 3
    assert ns["Solution"]().finalPositionOfSnake(3, ["DOWN", "RIGHT", "UP"]) == 1
    exec(nsc.LCB_GOLD["3519"], ns)
    assert ns["Solution"]().winningPlayerCount(4, [[0, 0], [1, 0], [1, 0], [2, 1], [2, 1], [2, 0]]) == 2
    assert ns["Solution"]().winningPlayerCount(5, [[1, 1], [1, 2], [1, 3], [1, 4]]) == 0


# -- SciCode -------------------------------------------------------------------


def _step(desc, header, code):
    return {"step_description_prompt": desc, "function_header": header, "ground_truth_code": code, "return_line": "return x"}


def test_scicode_gold_server_answers_the_next_step():
    rows = [
        {"problem_id": "1", "sub_steps": [_step("Compute A.", "def a(x):", "def a(x):\n    return 1"), _step("Compute B.", "def b(x):", "def b(x):\n    return 2")]},
        {"problem_id": "2", "sub_steps": [_step("Compute C.", "def c(x):", "def c(x):\n    return 3")]},
    ]
    s = nsc.SciCodeGoldServer(rows)
    prompt = f"steps:\nCompute A.\ndef a(x):\n    return 1\n\n{nsc.NEXT_STEP_MARKER}\n\nCompute B.\n\ndef b(x):\n\nreturn x\nDEPENDENCIES"
    assert s.lookup(prompt) == "```python\ndef b(x):\n    return 2\n```"
    assert s.lookup(f"{nsc.NEXT_STEP_MARKER}\nCompute C.\ndef c(x):") == "```python\ndef c(x):\n    return 3\n```"
    assert s.lookup(f"{nsc.NEXT_STEP_MARKER}\nsomething else") == ""


def test_scicode_gold_uses_dev_split(tmp_path, ns):
    assert bench("scicode", tmp_path).split() == "test"
    assert bench("scicode", tmp_path, options={"answers": "gold"}).split() == "dev"


def test_scicode_requires_sandbox_env(tmp_path):
    b = bench("scicode", tmp_path, options={"sandbox_env": str(tmp_path / "nope")})
    with pytest.raises(SystemExit, match="no sandbox env"):
        b.sandbox_python()


def _fake_env(tmp_path):
    """A 'sandbox env' that is this interpreter."""
    env = tmp_path / "env" / "bin"
    env.mkdir(parents=True)
    (env / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (env / "python").chmod(0o755)
    (env / "pip").write_text("#!/bin/sh\necho shim\n")
    (env / "pip").chmod(0o755)
    return str(env.parent)


def test_scicode_sandbox_report(tmp_path, monkeypatch):
    h5 = tmp_path / "test_data.h5"
    h5.write_bytes(b"targets")
    monkeypatch.setattr(nsc, "SCICODE_H5", str(h5))
    monkeypatch.setattr(nsc, "SCICODE_H5_SHA256", hashlib.sha256(b"targets").hexdigest())
    b = bench("scicode", tmp_path, options={"sandbox_env": _fake_env(tmp_path)})
    rows = [{"required_dependencies": "import json\nimport math"}, {"required_dependencies": "import no_such_module_x"}]
    r = b.sandbox_report(rows)
    assert r["h5_sha256_ok"] and r["dependency_blocks"] == 2
    assert list(r["import_failures"]) == ["import no_such_module_x"]
    monkeypatch.setattr(nsc, "SCICODE_H5_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="sha256"):
        b.sandbox_report(rows)


# -- with the harness ---------------------------------------------------------


@pytest.fixture
def ns():
    return pytest.importorskip("nemo_skills")


def test_module_args(ns, tmp_path):
    lcb = bench("livecodebench-v6", tmp_path)
    args = lcb.generation_args([])
    assert args[:2] == ["++prompt_config=eval/livecodebench/default_reasoning", "++eval_type=livecodebench"]
    assert args[-1] == f"++eval_config.test_file={tmp_path / 'lcb-tests.jsonl'}"
    assert lcb.split() == "test_v6_2408_2505" and lcb.metrics_type() == "livecodebench"
    sc = bench("scicode", tmp_path)
    assert sc.generation_module() == "granite_evals.benchmarks.scicode_prefill"  # wraps ns's module
    assert bench("scicode", tmp_path, options={"prefill_fixes": "false"}).generation_module() == scp.NS_SCICODE_MODULE
    assert sc.uses_sandbox() and sc.metrics_type() == "scicode"
    assert "++prompt_config=eval/scicode/background" in sc.generation_args([])
    for b in (lcb, sc):
        assert not b.uses_judge()


def test_lcb_gold_code_survives_ns_extraction(ns):
    from nemo_skills.evaluation.evaluator.code import preprocess_code

    for tid, code in nsc.LCB_GOLD.items():
        out = preprocess_code({"generation": f"```python\n{code}```", "task_id": tid}, "python")
        assert out["completion"].strip() == code.strip(), tid


def test_scicode_gold_code_survives_ns_extraction(ns):
    from nemo_skills.inference.eval.scicode_utils import extract_python_script

    code = "import numpy as np\ndef f(x):\n    return np.sum(x)\n"
    assert extract_python_script(f"```python\n{code}\n```").strip() == "def f(x):\n    return np.sum(x)"


def test_scicode_metrics_map_to_value(ns, tmp_path):
    b = bench("scicode", tmp_path, repeats=2)

    def rows(name, statuses):
        return _write(
            tmp_path / name,
            [{"eval_status": [{"process_status": s} for s in st], "num_generated_tokens": 1} for st in statuses],
        )

    files = [
        rows("output-rs0.jsonl", [["completed", "error"], ["completed", "completed", "completed"]]),
        rows("output-rs1.jsonl", [["completed", "completed"], ["error", "error", "error"]]),
    ]
    m = b.compute_metrics(files)
    # repeat 0: 4/5 subtasks, repeat 1: 2/5 -> 60%
    assert m["_all_"][b.aggregation()]["subtask_accuracy"] == pytest.approx(60.0)


def test_scicode_sandbox_server_runs(ns, tmp_path):
    """ns's server under the sandbox-env hook (here: this interpreter, if it has flask)."""
    pytest.importorskip("flask")
    pytest.importorskip("IPython")
    if sys.platform != "linux":
        pytest.skip("ns's sandbox server sets RLIMIT_AS, which only Linux allows")
    b = bench("scicode", tmp_path, options={"sandbox_env": _fake_env(tmp_path)})
    with b.ns_sandbox() as args:
        port = args[-1].split("=")[1]
        r = httpx.post(
            f"http://127.0.0.1:{port}/execute",
            json={"generated_code": "echo hi; pip install scipy==1.10.1", "language": "shell", "timeout": 10},
            timeout=60,
        )
        # the env's bin comes first on the sandbox's PATH: its pip shim answers
        assert r.status_code == 200 and r.json()["stdout"].split() == ["hi", "shim"]


def test_scicode_sandbox_app_is_outside_the_job_venv(ns, tmp_path):
    """flask puts the top of an --app file's package on sys.path, so the server runs
    from a lone copy: the job venv's site-packages must not shadow the sandbox env's."""
    b = bench("scicode", tmp_path)
    app = b.sandbox_app()
    assert app.is_file() and not (app.parent / "__init__.py").exists()
    assert "site-packages" not in str(app)
    assert "TerminalInteractiveShell" in app.read_text()


def test_sandbox_env_drops_the_job_python_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/job/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/job")
    monkeypatch.setenv("VIRTUAL_ENV", "/job/.venv")
    env = nsc.sandbox_env(tmp_path / "bin" / "python")
    assert not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} & set(env)
    assert env["PYTHONNOUSERSITE"] == "1" and env["PATH"].startswith(f"{tmp_path / 'bin'}:")


@pytest.mark.skipif(not Path(nsc.SANDBOX_ENV, "bin", "python").exists(), reason="no image sandbox env")
def test_image_sandbox_runs_its_own_scientific_stack(ns, tmp_path):
    """In the image: the real sandbox env serves, and code it runs gets the env's
    Python 3.10 with numpy 1.26.4 / scipy 1.10.1 (not the job venv's)."""
    if sys.platform != "linux":
        pytest.skip("ns's sandbox server sets RLIMIT_AS, which only Linux allows")
    b = bench("scicode", tmp_path)
    deps = "import numpy as np\nfrom scipy.special import erfc\nimport scipy.linalg"
    report = b.sandbox_report([{"required_dependencies": deps}])
    assert report["packages"]["numpy"] == "1.26.4" and report["packages"]["scipy"] == "1.10.1"
    assert report["import_failures"] == {} and report["python"].startswith("3.10.")
    code = "import sys, numpy, scipy.integrate\nprint(tuple(sys.version_info[:2]), numpy.__version__, scipy.__version__)"
    with b.ns_sandbox() as args:
        port = args[-1].split("=")[1]
        r = httpx.post(
            f"http://127.0.0.1:{port}/execute",
            json={"generated_code": code, "language": "python", "timeout": 60},
            timeout=120,
        )
    assert r.status_code == 200, r.text
    assert r.json()["stdout"].strip() == "(3, 10) 1.26.4 1.10.1", r.json()


# -- SciCode prefilled steps (scicode_prefill) ---------------------------------


def test_prefill_originals_are_pinned():
    for key, text in scp.ORIGINAL.items():
        assert hashlib.sha256(text.encode()).hexdigest() == scp.ORIGINAL_SHA256[key]
    assert scp.ORIGINAL.keys() == {("13", 5), ("62", 0)}
    assert "class Maxwell:" in scp.ORIGINAL["13", 5]
    assert "class Block:" in scp.ORIGINAL["62", 0] and "class EnlargedBlock:" in scp.ORIGINAL["62", 0]


def test_misplaced_methods():
    assert scp.misplaced_methods("def __init__(self, n):\n    self.n = n\n") == ["__init__"]
    assert scp.misplaced_methods("def make(cls):\n    return cls()\n") == ["make"]
    assert scp.misplaced_methods("class A:\n    def __init__(self):\n        pass\n") == []
    assert scp.misplaced_methods("def f(x):\n    return x\n") == []
    assert scp.misplaced_methods("def (:") == []


def test_prefill_plan_on_synthetic():
    fixed, rec = scp.plan({("13", 5): "def __init__(self, n):\n    self.n = n\n", ("76", 2): "def f(x):\n    return x\n"})
    assert fixed == {("13", 5): scp.ORIGINAL["13", 5]}
    assert rec[0]["step"] == "13.6" and rec[0]["restored_classes"] == ["Maxwell"]
    assert rec[0]["source"].endswith(f"{scp.SCICODE_DATA_COMMIT}/eval/data/13.6.txt")
    with pytest.raises(RuntimeError, match="9.2.*no original"):
        scp.plan({("9", 1): "def run(self):\n    pass\n"})


def _norm(node):
    """The node's AST with trailing whitespace stripped from docstring lines (ns's
    13.6 text drops the original's indentation on one blank docstring line)."""
    import ast

    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            n.value = "\n".join(line.rstrip() for line in n.value.split("\n"))
    return ast.dump(node)


def test_prefill_plan_on_ns(ns):
    """Every ns prefilled step is scanned: exactly 13.6 and 62.1 are methods cut
    out of their class, and the restored text holds the same methods."""
    import ast

    from nemo_skills.inference.eval.scicode_utils import prefilled_steps_code

    fixed, rec = scp.plan(prefilled_steps_code)
    assert [r["step"] for r in rec] == ["13.6", "62.1"]
    assert [r["restored_classes"] for r in rec] == [["Maxwell"], ["Block", "EnlargedBlock"]]
    for key, new in fixed.items():
        ns_funcs = [_norm(n) for n in ast.parse(prefilled_steps_code[key]).body if isinstance(n, ast.FunctionDef)]
        methods = [_norm(m) for c in ast.parse(new).body if isinstance(c, ast.ClassDef) for m in c.body]
        assert ns_funcs and all(f in methods for f in ns_funcs)  # ns's code is a method of the original
    unfixed = {k for k in prefilled_steps_code if k not in fixed}
    assert unfixed and not any(scp.misplaced_methods(prefilled_steps_code[k]) for k in unfixed)


def test_prefill_apply_reaches_ns_generation_module(ns):
    """ns's generation module holds the very dict apply() updates, so the wrapper's
    fix reaches it (it imports the dict by name)."""
    import importlib

    from nemo_skills.inference.eval import scicode_utils

    gen = importlib.import_module(scp.NS_SCICODE_MODULE)
    assert gen.prefilled_steps_code is scicode_utils.prefilled_steps_code
    saved = dict(scicode_utils.prefilled_steps_code)
    try:
        assert [r["step"] for r in scp.apply()] == ["13.6", "62.1"]
        assert gen.prefilled_steps_code["13", 5] == scp.ORIGINAL["13", 5]
        assert gen.prefilled_steps_code["62", 0] == scp.ORIGINAL["62", 0]
        assert scp.plan(gen.prefilled_steps_code) == ({}, [])  # nothing left to fix
    finally:
        scicode_utils.prefilled_steps_code.clear()
        scicode_utils.prefilled_steps_code.update(saved)


def test_prefill_wrapper_runs_ns_main(ns):
    """``python -m`` the wrapper: fixes reported, then ns's Hydra entry point runs."""
    import subprocess

    p = subprocess.run([sys.executable, "-m", "granite_evals.benchmarks.scicode_prefill", "--help"], capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stderr
    assert "restored ['Maxwell']" in p.stderr and "restored ['Block', 'EnlargedBlock']" in p.stderr
    assert "prompt_config" in p.stdout


def test_prefill_record(ns, tmp_path):
    r = bench("scicode", tmp_path).prefill_record()
    assert r["prefill_fixes_enabled"] and [f["step"] for f in r["prefill_fixes"]] == ["13.6", "62.1"]
    assert bench("scicode", tmp_path, options={"prefill_fixes": "false"}).prefill_record() == {
        "prefill_fixes": [],
        "prefill_fixes_enabled": False,
    }


# -- phases --------------------------------------------------------------------


def test_lcb_score_grades_with_the_tests_file(ns, tmp_path):
    b = bench("livecodebench-v6", tmp_path, phase="score")
    overrides = nsb._eval_overrides(b.generation_args([]))
    assert overrides == ["++eval_type=livecodebench", f"++eval_config.test_file={tmp_path / 'lcb-tests.jsonl'}"]
    b.load_rows(_write(tmp_path / "in.jsonl", _lcb_rows()))
    assert len(nsb._read_jsonl(b.tests_file())) == 3  # rewritten in the score phase too


def _scicode_split(tmp_path, monkeypatch, phase, events):
    """A SciCode run on 2 fake problems: fake generation, fake sandbox, a fake
    evaluator that grades each marked file as ns's would (eval_status)."""
    import contextlib

    prepared = _write(tmp_path / "prepared.jsonl", [{"problem_id": str(i), "sub_steps": []} for i in range(2)])

    def fake_generate(cmd, out, *, log_path, what):
        events.append(("generate", cmd[-1]))
        graded = {} if cmd[-1] == "++eval_type=null" else {"eval_status": [{"process_status": "completed"}]}
        rows = [{"problem_id": str(i), "generation": "g", "num_generated_tokens": 1, **graded} for i in range(2)]
        nsb._write_jsonl(out, rows)

    @contextlib.contextmanager
    def fake_sandbox(self):
        events.append(("sandbox",))
        yield ["++sandbox.port=1"]

    def fake_evaluate(self, output, args, what):
        if not output.exists():
            raise self.not_generated(what)
        if nsb._unscored(output).exists():
            events.append(("evaluate", output.name))
            rows = [{**r, "eval_status": [{"process_status": "completed"}]} for r in nsb._read_jsonl(output)]
            nsb._write_jsonl(output, rows)
            nsb._unscored(output).unlink()

    monkeypatch.setattr(nsb, "_run_ns", fake_generate)
    monkeypatch.setattr(nsc.SciCode, "ns_sandbox", fake_sandbox)
    monkeypatch.setattr(nsc.SciCode, "sandbox_report", lambda self, rows: events.append(("report",)) or {"ok": True})
    monkeypatch.setattr(nsb.NemoSkillsBenchmark, "_evaluate", fake_evaluate)
    b = bench("scicode", tmp_path, phase=phase, repeats=2)
    monkeypatch.setattr(b, "prepare_data", lambda: (prepared, {"sha256": "x"}))
    return b.run("http://127.0.0.1:9/v1" if phase != "score" else "", "m")


def test_scicode_generate_then_score(ns, tmp_path, monkeypatch):
    events = []
    g = _scicode_split(tmp_path, monkeypatch, "generate", events)
    assert events == [("generate", "++eval_type=null")] * 2  # no sandbox, no report
    assert g["n"] == 2 and "value" not in g and "sandbox" not in g and g["prefill_fixes_enabled"]
    events.clear()
    (tmp_path / "generation" / "output-rs1.jsonl").unlink()
    with pytest.raises(RuntimeError, match="no generation for repeat 1"):
        _scicode_split(tmp_path, monkeypatch, "score", events)
    assert events == [("report",), ("sandbox",), ("evaluate", "output-rs0.jsonl")]  # graded what is there
    _scicode_split(tmp_path, monkeypatch, "generate", events := [])
    assert events == [("generate", "++eval_type=null")]  # only the missing repeat
    s = _scicode_split(tmp_path, monkeypatch, "score", events := [])
    assert events == [("report",), ("sandbox",), ("evaluate", "output-rs1.jsonl")]  # rs0 kept its grades
    assert (s["value"], s["n"], s["sandbox"]) == (1.0, 2, {"ok": True}) and s["prefill_fixes_enabled"]


def test_scicode_phase_all_unchanged(ns, tmp_path, monkeypatch):
    r = _scicode_split(tmp_path, monkeypatch, "all", events := [])
    # ns evaluates inside generation, with the sandbox up
    assert events == [("report",), ("sandbox",), ("generate", "++sandbox.port=1"), ("generate", "++sandbox.port=1")]
    assert (r["value"], r["sandbox"]) == (1.0, {"ok": True})
