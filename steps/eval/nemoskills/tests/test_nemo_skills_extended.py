"""Extended NeMo-Skills benchmarks (nemo_skills_extended): config, judge parsing,
the think-tag check, and end-to-end runs through ns's real generate / judge /
evaluate / metrics against local fakes (gold server, judge, CritPt API, COMET)."""

import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from granite_evals import comet_score, registry
from granite_evals.benchmarks import nemo_skills as nsb
from granite_evals.benchmarks import nemo_skills_extended as g5
from granite_evals.registry import RunConfig

IDS = {
    "hle": ("pass@1 judge_correct", 1),
    "omniscience": ("pass@1 judge_correct", 8),
    "omniscience-hallucination": ("pass@1 judge_omni_hallucination", 8),
    "critpt": ("Challenge Accuracy", 1),
    "aa-lcr": ("pass@1 judge correct", 16),
    "wmt24pp": ("en→xx comet", 1),
}
KEY_ENV = "GRANITE_EVALS_TEST_FAKE_JUDGE_KEY"


def bench(bid, out, **kw):
    return registry.get(bid)(RunConfig(model="m", output_dir=out, **kw))


@pytest.fixture
def ns():
    return pytest.importorskip("nemo_skills")


@pytest.fixture
def mecab():  # sacrebleu[ja], in the nemoskills extra only (the bird image has ns without it)
    return pytest.importorskip("MeCab")


@pytest.mark.parametrize("bid", IDS)
def test_registered_as_in_suite(bid):
    cls = registry.get(bid)
    assert issubclass(cls, nsb.NemoSkillsBenchmark) and cls.splittable
    assert (cls.metric, cls.default_repeats) == IDS[bid]
    assert len(cls.dataset_revision) == 40 and cls.dataset
    suite = yaml.safe_load((Path(registry.__file__).parent / "suites" / "extended.yaml").read_text())
    assert {b["id"]: b["metric"] for b in suite["benchmarks"]}[bid] == cls.metric


def test_pins(tmp_path):
    assert bench("hle", tmp_path).pins() == {"cais/hle": g5.HLE_REVISION}
    assert bench("aa-lcr", tmp_path).pins() == {"ArtificialAnalysis/AA-LCR": g5.LCR_REVISION}
    assert bench("omniscience-hallucination", tmp_path).pins() == {g5.OMNI_DATASET: g5.OMNI_REVISION}
    for ref in (g5.COMET_MODEL, g5.COMET_ENCODER):
        assert len(ref.partition("@")[2]) == 40


def test_judge_defaults_are_the_granite_judge(ns, tmp_path):
    for bid in ("hle", "omniscience", "omniscience-hallucination", "aa-lcr"):
        b = bench(bid, tmp_path)
        ep = b.judge_endpoint("http://v/v1", "m")
        assert (ep["model"], ep["base_url"]) == (nsb.JUDGE_MODEL, nsb.JUDGE_BASE_URL)
        assert b.uses_judge() and b.gold_judged and b.official_judge
        assert "++inference.temperature=0.0" in b.judge_overrides()
    assert not bench("critpt", tmp_path).uses_judge()


def test_generation_args_send_no_sampling(ns, tmp_path):
    for bid in IDS:
        args = bench(bid, tmp_path).generation_args([])
        for k in ("temperature", "top_p", "tokens_to_generate"):
            assert f"++inference.{k}=null" in args, (bid, k)
    omni = bench("omniscience", tmp_path).generation_args([])
    # ns's parse_reasoning=True first, ours after it: the last one wins
    assert omni.index("++parse_reasoning=False") > omni.index("++parse_reasoning=True")
    assert "++prompt_config=generic/hle" in bench("hle", tmp_path).generation_args([])
    c = bench("critpt", tmp_path, options={"critpt_api_key_env": "K", "critpt_api_url": "http://x"})
    assert c.generation_module() == "granite_evals.ns_critpt"
    assert {"++eval_type=critpt", "++eval_config.api_key_env=K", "++eval_config.api_url=http://x"} <= set(
        c.generation_args([])
    )
    w = bench("wmt24pp", tmp_path, options={"languages": "de_DE,ja_JP"})
    assert "++prompt_config=multilingual/segment-translation" in w.generation_args([])
    assert w.ns_prepare_args == ("--target_languages", "de_DE", "ja_JP")
    assert bench("wmt24pp", tmp_path).ns_prepare_args[1:] == g5.WMT_LANGUAGES
    with pytest.raises(SystemExit):
        bench("wmt24pp", tmp_path, options={"languages": "german"}).languages()


def test_judge_parsers(ns, tmp_path):
    hle = bench("hle", tmp_path)
    assert hle.judge_failures({"judgement": "extracted_final_answer: 4\nJudgement: no\nConfidence: 90"}) == (0, 1)
    assert hle.judge_failures({"judgement": "**Judgement**: Yes"}) == (0, 1)
    assert hle.judge_failures({"judgement": "I cannot tell"}) == (1, 1)
    assert hle.judge_failures({"judgement": ""}) == (1, 1)
    omni = bench("omniscience", tmp_path)
    assert [omni.judge_failures({"judgement": j})[0] for j in ("A", " d\n", "B", "E", "A. correct", "")] == [
        0, 0, 0, 1, 1, 1,
    ]  # fmt: skip
    lcr = bench("aa-lcr", tmp_path)
    assert [lcr.judge_failures({"judgement": j})[0] for j in ("CORRECT", "incorrect", "Correct.", "yes", None)] == [
        0, 0, 0, 1, 1,
    ]  # fmt: skip


def test_think_tags_fail_the_run(tmp_path):
    gen = tmp_path / "output-rs0.jsonl"
    nsb._write_jsonl(gen, [{"generation": "4"}, {"generation": "<think>hmm</think>4"}])
    with pytest.raises(SystemExit, match="1/2 generations .* <think>"):
        bench("hle", tmp_path)._evaluate(gen, [], "rs0")
    nsb._write_jsonl(gen, [{"generation": "ok", "intermediate": "x</think>y"}])
    bench("hle", tmp_path)._evaluate(gen, [], "rs0")  # hle scores only generation (unmarked: nothing to do)
    with pytest.raises(SystemExit, match="think"):
        bench("critpt", tmp_path)._evaluate(gen, [], "rs0")  # critpt's turn-1 answer goes back in turn 2


def test_invalid_judgements_left_out_of_the_score(ns, tmp_path):
    judged = tmp_path / "output-rs0.jsonl"
    rows = [
        {"problem": "p", "expected_answer": "1", "generation": "Answer: 1", "judgement": j}
        for j in ("Judgement: yes", "Judgement: yes", "Judgement: no", "no verdict at all")
    ]
    nsb._write_jsonl(judged, rows)
    m = bench("hle", tmp_path).compute_metrics([judged])
    assert m["_all_"]["pass@1"]["judge_correct"] == pytest.approx(200 / 3)
    assert m["_all_"]["pass@1"]["num_entries"] == 3 and m["judge_invalid_left_out"] == 1
    assert len(nsb._read_jsonl(judged)) == 4  # the judged file itself is kept whole (re-judged on resume)


# -- end to end: gold answers -> ns generate -> ns judge -> metrics ------------------


class _Judge(nsb._Server):
    """A fake judge endpoint (OpenAI chat); ``verdict(prompt)`` is its reply."""

    def __init__(self, verdict):
        self.verdict, self.requests = verdict, []

    def handle(self, method, path, headers, body):
        if method == "GET":
            return nsb._json_response({"object": "list", "data": [{"id": "judge", "object": "model"}]})
        req = json.loads(body)
        self.requests.append(req)
        text = self.verdict("\n".join(nsb._content_text(m.get("content")) for m in req["messages"]))
        msg = {"role": "assistant", "content": text}
        return nsb._json_response({
            "id": "j", "object": "chat.completion", "created": 0, "model": req.get("model", "judge"),
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        })  # fmt: skip


def _between(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def hle_verdict(prompt):
    ok = _between(prompt, "[correct_answer]:", "\n") in _between(prompt, "[response]:", "Your judgement")
    return f"extracted_final_answer: x\nReasoning: r\nJudgement: {'yes' if ok else 'no'}\nConfidence: 100"


def lcr_verdict(prompt):
    same = _between(prompt, "The OFFICIAL ANSWER:", "\n") == _between(prompt, "CANDIDATE ANSWER TO ASSESS:", "\n")
    return "CORRECT" if same else "INCORRECT"


def _local_data(monkeypatch, cls, path, rows):
    nsb._write_jsonl(path, rows)
    monkeypatch.setattr(cls, "prepare_data", lambda self: (path, {"source": "test", "sha256": nsb._sha256(path)}))


HLE_ROWS = [
    {"id": f"h{i}", "problem": f"What is {i} + {i}?", "expected_answer": str(2 * i), "answer_type": "exactMatch",
     "subset_for_metrics": "Math"}
    for i in range(1, 5)
]  # fmt: skip
OMNI_ROWS = [
    {"id": f"o{i}", "domain": "Science", "topic": "Physics", "question": f"Omniscience question {i}?",
     "expected_answer": f"answer {i}"}
    for i in range(4)
]  # fmt: skip
LCR_ROWS = [
    {"index": i, "question": f"Documents ... long text {i} ... Question: what is item {i}?",
     "original_question": f"what is item {i}?", "expected_answer": f"item-{i}", "document_category": "Legal",
     "input_tokens": 1000 * (i + 1)}
    for i in range(3)
]  # fmt: skip


def _gold(bid, out, **kw):
    options = {"answers": "gold", "judge_base_url": kw.pop("judge_url"), "judge_api_key_env": KEY_ENV}
    return bench(bid, out, workers=4, options={**options, **kw.pop("options", {})}, **kw)


@pytest.mark.parametrize(
    "bid, cls, rows, verdict",
    [("hle", g5.HLE, HLE_ROWS, hle_verdict), ("aa-lcr", g5.AALCR, LCR_ROWS, lcr_verdict)],
)
def test_gold_end_to_end_through_the_judge(ns, tmp_path, monkeypatch, bid, cls, rows, verdict):
    monkeypatch.setenv(KEY_ENV, "fake-key")
    _local_data(monkeypatch, cls, tmp_path / "data.jsonl", rows)
    with _Judge(verdict) as judge:
        r = _gold(bid, tmp_path / "out", repeats=2, judge_url=judge.base_url).run("", "")
    assert (r["value"], r["n"]) == (1.0, len(rows))
    assert r["ns_metric"]["aggregation"] == "pass@1[avg-of-2]"
    assert r["pass_at_k"] == {"k": 2, "pass_at_1": 1.0, "pass_at_k": 1.0, "n": len(rows), "how": "ns metrics"}
    assert [p["judge_correct"] for p in r["per_repeat"]] == [1.0, 1.0]
    assert (r["judge_invalid"], r["judge_total"], r["incomplete"]) == (0, 2 * len(rows), False)
    assert r["judge_model"] == nsb.JUDGE_MODEL and r["judge_usage"]["requests"] == 2 * len(rows)
    assert len(judge.requests) == 2 * len(rows)
    assert all(q["temperature"] == 0.0 and q["model"] == nsb.JUDGE_MODEL for q in judge.requests)


def test_hle_judge_scores_a_wrong_answer_wrong(ns, tmp_path, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "fake-key")
    _local_data(monkeypatch, g5.HLE, tmp_path / "data.jsonl", HLE_ROWS)
    monkeypatch.setattr(g5.HLE, "gold_generation", lambda self, row: "Answer: 1\nConfidence: 50%")
    with _Judge(hle_verdict) as judge:
        r = _gold("hle", tmp_path / "out", judge_url=judge.base_url).run("", "")
    assert r["value"] == 0.0 and r["pass_at_k"]["k"] == 1 and r["pass_at_k"]["pass_at_k"] == 0.0


def omni_verdict(prompt):
    # Graded by question: 0, 1 correct; 2 incorrect; 3 not attempted.
    for i, v in ((0, "A"), (1, "A"), (2, "B"), (3, "D")):
        if f"Omniscience question {i}?" in prompt:
            return v
    return "?"


def test_omniscience_and_its_hallucination_rate_from_one_run(ns, tmp_path, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "fake-key")
    _local_data(monkeypatch, g5.Omniscience, tmp_path / "data.jsonl", OMNI_ROWS)
    with _Judge(omni_verdict) as judge:
        r = _gold("omniscience", tmp_path / "omni", judge_url=judge.base_url).run("", "")
    assert r["value"] == 0.5 and r["ns_metrics"]["_all_"]["pass@1"]["judge_omni_hallucination"] == 50.0
    gen = nsb._read_jsonl(tmp_path / "omni" / "generation" / "output-rs0.jsonl")
    assert [g["generation"] for g in gen] == [f"answer {i}" for i in range(4)]  # parse_reasoning off: kept

    # The hallucination rate of that run: nothing generated or judged again.
    monkeypatch.setattr(nsb, "_run_ns", lambda *a, **k: pytest.fail("generated or judged again"))
    opts = {"generations_from": str(tmp_path / "omni"), "judge_api_key_env": KEY_ENV, "answers": "gold"}
    h = bench("omniscience-hallucination", tmp_path / "hall", options=opts)
    assert not h.needs_server() and not h.generating
    hr = h.run("", "")
    assert (hr["value"], hr["n"]) == (0.5, 4) and hr["generations_from"] == str(tmp_path / "omni")
    assert hr["ns_metric"]["key"] == "judge_omni_hallucination"
    with pytest.raises(SystemExit, match="nothing to generate"):
        bench("omniscience-hallucination", tmp_path / "hall", phase="generate", options=opts).run("", "")
    with pytest.raises(SystemExit, match="differs from what was generated"):  # another --limit: not that run
        bench("omniscience-hallucination", tmp_path / "hall2", limit=2, options=opts).run("", "")


def test_invalid_judgement_left_out_and_rejudged(ns, tmp_path, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "fake-key")
    _local_data(monkeypatch, g5.Omniscience, tmp_path / "data.jsonl", OMNI_ROWS)
    flaky = {"n": 0}

    def verdict(prompt):  # question 2 gets no valid verdict the first time
        if "Omniscience question 2?" in prompt and not flaky["n"]:
            flaky["n"] += 1
            return "The answer is wrong."
        return omni_verdict(prompt)

    kw = {"options": {"max_judge_invalid_frac": "0.5"}}
    with _Judge(verdict) as judge:
        r = _gold("omniscience", tmp_path / "out", judge_url=judge.base_url, **kw).run("", "")
        assert (r["judge_invalid"], r["incomplete"]) == (1, True)
        assert r["value"] == pytest.approx(2 / 3) and r["ns_metrics"]["judge_invalid_left_out"] == 1
        n = len(judge.requests)
        r2 = _gold("omniscience", tmp_path / "out", judge_url=judge.base_url, **kw).run("", "")
        assert len(judge.requests) == n + 1  # only the invalid one, again
    assert (r2["judge_invalid"], r2["incomplete"], r2["value"]) == (0, False, 0.5)


def test_judged_gold_generate_then_score(ns, tmp_path, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "fake-key")
    _local_data(monkeypatch, g5.HLE, tmp_path / "data.jsonl", HLE_ROWS)
    with _Judge(hle_verdict) as judge:
        g = _gold("hle", tmp_path / "out", phase="generate", judge_url=judge.base_url).run("", "")
        assert "value" not in g and not judge.requests  # no judging in the generate phase
        s = _gold("hle", tmp_path / "out", phase="score", judge_url=judge.base_url).run("", "")
    assert (s["value"], s["n"]) == (1.0, 4) and len(judge.requests) == 4


# -- CritPt: the external grading API -------------------------------------------------


class _CritPtAPI(nsb._Server):
    def __init__(self, accuracy):
        self.accuracy, self.calls = accuracy, []

    def handle(self, method, path, headers, body):
        self.calls.append((path, {k.lower(): v for k, v in headers.items()}.get("x-api-key"), json.loads(body)))
        return nsb._json_response({
            "accuracy": self.accuracy, "timeout_rate": 0.0, "server_timeout_count": 0, "judge_error_count": 1,
        })  # fmt: skip


CRITPT_ROWS = [
    {"problem_id": f"p{i}", "problem": f"Physics problem {i}", "code_template": "def answer():\n    ...",
     "expected_answer": "", "metadata": {}}
    for i in range(g5.CRITPT_PROBLEMS)
]  # fmt: skip


def _fake_critpt_generate(cmds):
    def run(cmd, out, *, log_path, what):
        cmds.append(cmd)
        args = dict(a[2:].split("=", 1) for a in cmd if a.startswith("++"))
        rows = nsb._read_jsonl(Path(args["input_file"]))
        nsb._write_jsonl(out, [
            {**r, "generation": "```python\ndef answer():\n    return 1\n```", "intermediate": "solved",
             "num_generated_tokens": 5, "finish_reason": "stop"}
            for r in rows
        ])  # fmt: skip

    return run


def test_critpt_generate_then_score_with_the_api(ns, tmp_path, monkeypatch):
    _local_data(monkeypatch, g5.CritPt, tmp_path / "data.jsonl", CRITPT_ROWS)
    monkeypatch.setattr(nsb, "_run_ns", _fake_critpt_generate(cmds := []))
    monkeypatch.delenv(g5.CRITPT_API_KEY_ENV, raising=False)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="needs an Artificial Analysis API key"):  # phase all: before generating
        bench("critpt", out).run("http://127.0.0.1:9/v1", "m")
    assert not cmds
    g = bench("critpt", out, phase="generate").run("http://127.0.0.1:9/v1", "m")  # no key needed
    assert g["n"] == 70 and cmds[0][-1] == "++eval_type=null"
    assert nsb._unscored(out / "generation" / "output-rs0.jsonl").exists()

    monkeypatch.setenv(g5.CRITPT_API_KEY_ENV, "aa-test-key")
    with _CritPtAPI(0.3) as api:
        s = bench("critpt", out, phase="score", options={"critpt_api_url": api.base_url + "/critpt"}).run("", "")
        assert len(api.calls) == 1
        path, key, payload = api.calls[0]
        assert path == "/v1/critpt" and key == "aa-test-key" and len(payload["submissions"]) == 70
        assert payload["submissions"][0]["generated_code"].startswith("```python\ndef answer")
        again = bench("critpt", out, phase="score", options={"critpt_api_url": api.base_url}).run("", "")
        assert len(api.calls) == 1  # graded once; the response is cached next to the file
    assert (s["value"], s["n"]) == (pytest.approx(0.3), 70) and again["value"] == s["value"]
    assert s["critpt_api"][0]["judge_error_count"] == 1 and s["incomplete"]


def test_critpt_limits(ns, tmp_path, monkeypatch):
    _local_data(monkeypatch, g5.CritPt, tmp_path / "data.jsonl", CRITPT_ROWS)
    monkeypatch.setattr(nsb, "_run_ns", _fake_critpt_generate([]))
    monkeypatch.setenv(g5.CRITPT_API_KEY_ENV, "k")
    with pytest.raises(SystemExit, match="exactly 70"):
        bench("critpt", tmp_path / "a", limit=5).run("http://127.0.0.1:9/v1", "m")
    assert bench("critpt", tmp_path / "a", limit=5, phase="generate").run("http://127.0.0.1:9/v1", "m")["n"] == 5
    with pytest.raises(SystemExit, match="no answers=gold"):
        bench("critpt", tmp_path / "b", options={"answers": "gold"}).run("", "")


# -- WMT24++ ---------------------------------------------------------------------------

WMT_ROWS = [
    {"source": src, "reference": ref, "source_language": "en", "target_language": lang,
     "source_lang_name": "English", "target_lang_name": name}
    for lang, name, pairs in (
        ("de_DE", "German", [("Good morning.", "Guten Morgen."), ("The cat sleeps on the mat.", "Die Katze schläft auf der Matte.")]),
        ("ja_JP", "Japanese", [("Good morning.", "おはようございます。"), ("The weather is nice today.", "今日は天気が良いです。")]),
    )
    for src, ref in pairs
]  # fmt: skip


def test_wmt_gold_server_matches_source_and_language():
    s = g5.WMTGoldServer(WMT_ROWS)
    prompt = "Translate the following segment into {}, without additional explanation.\n\n{}"
    assert s.lookup(prompt.format("Japanese", "Good morning.")) == "おはようございます。"
    assert s.lookup(prompt.format("German", "Good morning.")) == "Guten Morgen."
    assert s.lookup(prompt.format("French", "Good morning.")) == ""


def test_wmt_gold_bleu_end_to_end(ns, mecab, tmp_path, monkeypatch):
    """Reference translations through ns's real prompt -> generate -> BLEU (ja-mecab
    for Japanese, with ns's run-time pip install patched out)."""
    _local_data(monkeypatch, g5.WMT24pp, tmp_path / "data.jsonl", WMT_ROWS)
    import nemo_skills.evaluation.metrics.translation_metrics as tm

    run = tm.subprocess.run

    def no_pip(cmd, *a, **k):
        assert "pip" not in " ".join(map(str, cmd)), cmd
        return run(cmd, *a, **k)

    monkeypatch.setattr(tm.subprocess, "run", no_pip)
    b = bench("wmt24pp", tmp_path / "out", options={"answers": "gold", "score_metric": "bleu"})
    assert not b.uses_judge() and b.ns_metric == "bleu"
    r = b.run("", "")
    assert (r["value"], r["n"]) == (pytest.approx(1.0), 4)
    assert r["ns_metrics"]["_all_"]["en->ja_JP"]["bleu"] == pytest.approx(100.0)
    assert r["per_repeat"][0]["bleu"] == pytest.approx(1.0)


def test_wmt_comet_step(ns, mecab, tmp_path, monkeypatch):
    _local_data(monkeypatch, g5.WMT24pp, tmp_path / "data.jsonl", WMT_ROWS)
    specs = []

    def fake_comet(self, cmd, work, env):  # stands in for comet_score.py in /opt/comet
        assert cmd[:2] == [sys.executable, comet_score.__file__] and env["HF_HUB_OFFLINE"] == "1"
        assert "PYTHONPATH" not in env and "VIRTUAL_ENV" not in env
        spec = json.loads(Path(cmd[2]).read_text())
        specs.append(spec)
        for inp, out in spec["files"]:
            nsb._write_jsonl(Path(out), [{**r, "comet": 0.9} for r in nsb._read_jsonl(Path(inp))])
            Path(out + ".done").touch()

    monkeypatch.setattr(g5.WMT24pp, "_run_comet", fake_comet)
    monkeypatch.setattr(g5.WMT24pp, "comet_model_dir", lambda self: (tmp_path / "m", {"model": "x@1"}))
    opts = {"answers": "gold", "comet_python": sys.executable}
    r = bench("wmt24pp", tmp_path / "out", repeats=2, options=opts).run("", "")
    assert r["value"] == pytest.approx(0.9) and r["ns_metric"] == {
        "aggregation": "en->xx", "key": "comet", "raw": pytest.approx(90.0), "scale": 0.01,
    }  # fmt: skip
    assert r["ns_metrics"]["_all_"]["en->xx"]["bleu"] == pytest.approx(100.0)  # BLEU alongside
    assert [p["comet"] for p in r["per_repeat"]] == [pytest.approx(0.9)] * 2
    assert r["comet"]["model"] == "x@1" and r["comet"]["precision"] == "bf16"
    (spec,) = specs
    assert Path(spec["ns_comet"]).name == "comet.py" and Path(spec["ns_comet"]).exists()
    assert [Path(o).name for _, o in spec["files"]] == ["output-rs0.jsonl", "output-rs1.jsonl"]
    assert spec["checkpoint"] == str(tmp_path / "m" / "checkpoints" / "model.ckpt")
    bench("wmt24pp", tmp_path / "out", repeats=2, options=opts).run("", "")
    assert len(specs) == 1  # scored files (.done) are not scored again


def test_comet_model_dir_points_hparams_at_the_local_encoder(tmp_path):
    model, enc = tmp_path / "xcomet", tmp_path / "xlmr"
    (model / "checkpoints").mkdir(parents=True)
    (model / "checkpoints" / "model.ckpt").write_text("weights")
    (model / "hparams.yaml").write_text(
        "nr_frozen_epochs: 0.3\npretrained_model: facebook/xlm-roberta-xxl\nlayer: mix\n"
    )
    enc.mkdir()
    b = bench("wmt24pp", tmp_path / "out", options={"comet_model": str(model), "comet_encoder": str(enc)})
    local, refs = b.comet_model_dir()
    assert refs["model"] == str(model) and refs["encoder"] == str(enc)
    assert (local / "hparams.yaml").read_text() == f"nr_frozen_epochs: 0.3\npretrained_model: {enc}\nlayer: mix\n"
    assert (local / "checkpoints" / "model.ckpt").read_text() == "weights"
    assert b.comet_model_dir()[0] == local  # re-runnable
    with pytest.raises(SystemExit, match="repo@revision"):
        bench("wmt24pp", tmp_path / "o2", options={"comet_model": "Unbabel/XCOMET-XXL"}).comet_model_dir()
    with pytest.raises(SystemExit, match="no COMET env"):
        bench("wmt24pp", tmp_path / "o3", options={"comet_python": str(tmp_path / "nope")}).comet_python()


def test_comet_score_script_drives_ns_scorer(tmp_path, monkeypatch):
    """comet_score.py: ns's comet.py (by path) with the local checkpoint, offline."""
    loaded = []

    class Model:
        def to(self, dtype):
            loaded.append(("dtype", dtype))
            return self

    fake_comet = types.ModuleType("comet")
    fake_comet.load_from_checkpoint = lambda path, **kw: loaded.append((path, kw)) or Model()
    monkeypatch.setitem(sys.modules, "comet", fake_comet)
    ns_comet = tmp_path / "comet.py"
    ns_comet.write_text(
        "import json\nfrom pathlib import Path\n_PRECISION_TO_DTYPE = {'bf16': 'BF16', 'fp32': None}\n"
        "def process_file(i, o, model, batch_size):\n"
        "    o.write_text(json.dumps([str(i), batch_size]))\n"
    )
    spec = tmp_path / "spec.json"
    files = [[str(tmp_path / f"in{k}"), str(tmp_path / f"out{k}")] for k in range(2)]
    spec.write_text(json.dumps({"ns_comet": str(ns_comet), "checkpoint": "/m/checkpoints/model.ckpt",
                                "precision": "bf16", "batch_size": 16, "files": files}))  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["comet_score.py", str(spec)])
    comet_score.main()
    assert loaded == [
        ("/m/checkpoints/model.ckpt", {"reload_hparams": True, "local_files_only": True}),
        ("dtype", "BF16"),
    ]
    assert json.loads((tmp_path / "out1").read_text()) == [str(tmp_path / "in1"), 16]
