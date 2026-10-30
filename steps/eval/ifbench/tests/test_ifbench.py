"""IFBench family tests: fakes only (no GPU, no network, no server)."""

from granite_evals.benchmarks import ifbench as chat_if
from granite_evals.registry import RunConfig
from pathlib import Path
from types import SimpleNamespace
import json
import os
import pytest


# --- IFBench (NeMo-Skills) ------------------------------------------------------


def test_ifbench_registered_and_pinned(tmp_path):
    from granite_evals.registry import get

    cls = get("ifbench")
    assert cls is chat_if.IFBench
    assert cls.metric == "pass@1 prompt loose / strict accuracy" and cls.default_repeats == 5
    assert cls.extra == "ifbench" and cls.ns_metric == "prompt_loose_accuracy"
    assert cls.ns_report_metrics == ("prompt_strict_accuracy", "instruction_loose_accuracy", "instruction_strict_accuracy")
    b = cls(RunConfig(model="m", output_dir=tmp_path))
    assert b.repeats == 5 and b.aggregation() == "pass@1[avg-of-5]"
    assert cls(RunConfig(model="m", output_dir=tmp_path, repeats=2)).aggregation() == "pass@1[avg-of-2]"
    assert b.pins() == {}  # a GitHub file, pinned by URL + sha256, not an HF repo
    pinned, sha = cls.pinned_urls[chat_if.IFBENCH_TEST_URL]
    assert chat_if.IFBENCH_DATA_COMMIT in pinned and len(sha) == 64
    assert len(cls.prepared_sha256) == 64  # the prepared test.jsonl is checked too


def test_ifbench_has_no_gold_mode(tmp_path):
    b = chat_if.IFBench(RunConfig(model="m", output_dir=tmp_path, options={"answers": "gold"}))
    with pytest.raises(SystemExit, match="no answers=gold"):
        b.run("", "")


def test_ifbench_pin_covers_what_ns_prepare_reads(tmp_path):
    pytest.importorskip("nemo_skills")
    b = chat_if.IFBench(RunConfig(model="m", output_dir=tmp_path))
    prepare = (Path(b.ns_module().__file__).parent / "prepare.py").read_text()
    assert f'URL = "{chat_if.IFBENCH_TEST_URL}"' in prepare
    args = b.generation_args([])
    assert {"++prompt_config=generic/default", "++eval_type=ifbench", "++generation_key=response"} <= set(args)
    assert b.metrics_type() == "if"


def _if_rows(path, loose, strict):
    """ns generation rows after ns's ifbench evaluator merged IFBench's results."""
    rows = []
    for i, (lo, st) in enumerate(zip(loose, strict, strict=True)):
        rows.append(
            {
                "key": str(i),
                "prompt": f"p{i}",
                "response": "r",
                "instruction_id_list": ["a", "b"],
                "num_generated_tokens": 3,
                "loose_eval": {"follow_all_instructions": all(lo), "follow_instruction_list": lo},
                "strict_eval": {"follow_all_instructions": all(st), "follow_instruction_list": st},
            }
        )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_ifbench_value_is_prompt_loose_accuracy_avg_of_2(tmp_path):
    pytest.importorskip("nemo_skills")
    b = chat_if.IFBench(RunConfig(model="m", output_dir=tmp_path, repeats=2))
    files = [
        _if_rows(tmp_path / "output-rs0.jsonl", [[True, True], [True, False]], [[True, False], [False, False]]),
        _if_rows(tmp_path / "output-rs1.jsonl", [[True, True], [True, True]], [[True, True], [False, False]]),
    ]
    m = b.compute_metrics(files)["_all_"][b.aggregation()]
    assert m["prompt_loose_accuracy"] == pytest.approx(75.0)  # (1/2 + 2/2) / 2
    assert m["prompt_strict_accuracy"] == pytest.approx(25.0)
    assert m["instruction_loose_accuracy"] == pytest.approx(87.5)
    assert m["instruction_strict_accuracy"] == pytest.approx(37.5)
    # pass@2: a prompt counts if either generation followed all its instructions
    assert b.pass_at_k(b.compute_metrics(files), 2) == {"k": 2, "pass_at_1": 0.75, "pass_at_k": 1.0, "n": 2, "how": "ns metrics"}


def test_ifbench_puts_the_pinned_nltk_data_first(tmp_path, monkeypatch):
    """IFBench's verifiers nltk.download() at build time; ~/nltk_data would win over
    the image's pinned copy unless NLTK_DATA names it (inherited by run_eval)."""
    from granite_evals.benchmarks.nemo_skills import NemoSkillsBenchmark

    seen = {}
    monkeypatch.setattr(NemoSkillsBenchmark, "run", lambda self, *a: seen.setdefault("NLTK_DATA", __import__("os").environ["NLTK_DATA"]))
    monkeypatch.setenv("NLTK_DATA", f"/elsewhere:{chat_if.IFBENCH_NLTK_DATA}")
    chat_if.IFBench(RunConfig(model="m", output_dir=tmp_path)).run("", "")
    assert seen["NLTK_DATA"].split(":") == [chat_if.IFBENCH_NLTK_DATA, "/elsewhere"]


def _ifbench_phase(tmp_path, monkeypatch, phase, events):
    """IFBench on 2 prompts with ns's generation and evaluator faked: the evaluator
    grades each marked file (prompt 0 followed, prompt 1 not)."""
    from granite_evals.benchmarks import nemo_skills as nsb

    prepared = tmp_path / "prepared.jsonl"
    nsb._write_jsonl(prepared, [{"key": str(i), "prompt": f"p{i}", "instruction_id_list": ["a"]} for i in range(2)])

    def fake_generate(cmd, out, *, log_path, what):
        events.append(("generate", cmd[-1]))
        rows = nsb._read_jsonl(prepared)
        nsb._write_jsonl(out, [{**r, "response": "r", "num_generated_tokens": 3} for r in rows])

    def fake_evaluate(self, output, args, what):
        if not output.exists():
            raise self.not_generated(what)
        if nsb._unscored(output).exists():
            events.append(("evaluate", nsb._eval_overrides(args), os.environ["NLTK_DATA"].split(os.pathsep)[0]))
            ev = [{"follow_all_instructions": i == 0, "follow_instruction_list": [i == 0]} for i in range(2)]
            nsb._write_jsonl(output, [{**r, "loose_eval": e, "strict_eval": e} for r, e in zip(nsb._read_jsonl(output), ev)])
            nsb._unscored(output).unlink()

    monkeypatch.setattr(nsb, "_run_ns", fake_generate)
    monkeypatch.setattr(nsb.NemoSkillsBenchmark, "_evaluate", fake_evaluate)
    b = chat_if.IFBench(RunConfig(model="m", output_dir=tmp_path, phase=phase, repeats=1))
    monkeypatch.setattr(b, "prepare_data", lambda: (prepared, {"sha256": "x"}))
    return b.run("http://127.0.0.1:9/v1" if phase == "generate" else "", "m")


def test_ifbench_generate_then_score(tmp_path, monkeypatch):
    pytest.importorskip("nemo_skills")
    monkeypatch.setenv("NLTK_DATA", "/elsewhere")
    g = _ifbench_phase(tmp_path, monkeypatch, "generate", events := [])
    assert g["n"] == 2 and "value" not in g and events == [("generate", "++eval_type=null")]
    assert os.environ["NLTK_DATA"] == "/elsewhere"  # only scoring needs the pinned nltk data
    s = _ifbench_phase(tmp_path, monkeypatch, "score", events := [])
    graded = ("evaluate", ["++eval_type=ifbench"], chat_if.IFBENCH_NLTK_DATA)
    assert events == [graded] and (s["value"], s["n"]) == (pytest.approx(0.5), 2)
    assert s["metrics"] == {
        "prompt_loose_accuracy": 0.5,
        "prompt_strict_accuracy": 0.5,
        "instruction_loose_accuracy": 0.5,
        "instruction_strict_accuracy": 0.5,
    }
    assert s["pass_at_k"]["k"] == 1 and s["pass_at_k"]["pass_at_k"] == pytest.approx(0.5)
    assert _ifbench_phase(tmp_path, monkeypatch, "score", events := [])["value"] == pytest.approx(0.5)
    assert events == []  # re-scored from the graded files
    (tmp_path / "generation" / "output-rs0.jsonl").unlink()
    with pytest.raises(RuntimeError, match="no generation for repeat 0"):
        _ifbench_phase(tmp_path, monkeypatch, "score", [])


def test_ifbench_is_splittable():
    assert chat_if.IFBench.splittable


# Every instruction of the pinned IFBench test data (newer than the IFBench checkout
# ns pins, whose own data/ has 294 prompts) through its verifier, as ns's evaluator
# runs them (cwd = the IFBench checkout, the job's python), with the exceptions
# ns's patch would swallow (scored "not followed") reported instead. nltk.download is
# a no-op here: the pinned data must be enough, and a build-time test must not
# change it.
IFBENCH_VERIFY = r"""
import json, sys
import nltk
nltk.download = lambda *a, **k: True
import instructions_registry as reg
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
response = (
    "Birds sing at dawn. Do you hear them? I walked to the river with Maria and John, and the water was cold.\n\n"
    "* Peter met them in Paris on 3 May 2024.\n* They counted 42 boats, 7 bridges and 1 dog!\n\n"
    "Run, because the rain is coming; however, nobody ran. P.S. The end."
)
seen, failed = set(), {}
for row in rows:
    for iid, kw in zip(row["instruction_id_list"], row["kwargs"]):
        seen.add(iid)
        try:
            inst = reg.INSTRUCTION_DICT[iid](iid)
            inst.build_description(**{k: v for k, v in kw.items() if v is not None})
            args = inst.get_instruction_args()
            if args and "prompt" in args:
                inst.build_description(prompt=row["prompt"])
            inst.check_following(response)
        except Exception as e:
            failed.setdefault(iid, repr(e)[:300])
found = {p: str(nltk.data.find(p)) for p in ("tokenizers/punkt", "tokenizers/punkt_tab", "corpora/stopwords", "taggers/averaged_perceptron_tagger_eng")}
print(json.dumps({"rows": len(rows), "instructions": sorted(seen), "failed": failed, "nltk": found}))
"""


def test_ifbench_verifiers_run_in_the_image(tmp_path):
    import os
    import subprocess
    import sys

    import hashlib
    import urllib.request

    if not (chat_if.IFBENCH_DIR / "run_eval.py").is_file():
        pytest.skip("no IFBench checkout (built by docker/extras/ifbench.sh)")
    url, sha = chat_if.IFBench.pinned_urls[chat_if.IFBENCH_TEST_URL]
    data = tmp_path / "IFBench_test.jsonl"
    with urllib.request.urlopen(url, timeout=60) as f:
        data.write_bytes(f.read())
    assert hashlib.sha256(data.read_bytes()).hexdigest() == sha
    env = dict(os.environ, NLTK_DATA=chat_if.IFBENCH_NLTK_DATA, HOME=str(tmp_path))
    r = subprocess.run(
        [sys.executable, "-c", IFBENCH_VERIFY, str(data)], cwd=chat_if.IFBENCH_DIR, env=env,
        capture_output=True, text=True, timeout=900,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr[-3000:]
    report = json.loads(r.stdout.strip().splitlines()[-1])
    assert report["rows"] == 300 and len(report["instructions"]) > 50
    assert report["failed"] == {}
    assert all(p.startswith(chat_if.IFBENCH_NLTK_DATA + "/") for p in report["nltk"].values()), report["nltk"]
