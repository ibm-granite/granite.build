"""RULER through NeMo-Skills: config, source pinning, per-task args (thinking on:
chat format, cap - prompt budgets; off: ns's text/prefill), the per-sample budget
and failure log, gold server, context check, the think-tag guard, and the ns
metrics/ruler_score path (with the nemoskills extra)."""

import hashlib
import json
from pathlib import Path

import pytest

from granite_evals import registry
from granite_evals.benchmarks import nemo_skills as nsb
from granite_evals.benchmarks import nemo_skills_ruler as nsr
from granite_evals.registry import RunConfig

IDS = {"ruler-128k": 131072, "ruler-64k": 65536, "ruler-256k": 262144, "ruler-512k": 524288, "ruler-1m": 1048576}

ARGS = (  # ns prepare --data_format default (thinking off)
    '"++prompt_config=generic/default ++eval_type=ruler ++eval_config.match_type=all '
    '++inference.tokens_to_generate=128 ++start_assistant_response_key=generation ++inference.endpoint_type=text "'
)
CHAT_ARGS = '"++prompt_config=generic/default ++eval_type=ruler ++eval_config.match_type=all "'  # --data_format chat
OFF = {"enable_thinking": "false"}


def bench(bid, tmp_path, **kw):
    kw.setdefault("model", "/models/granite")
    return registry.get(bid)(RunConfig(output_dir=tmp_path, **kw))


@pytest.mark.parametrize("bid", IDS)
def test_registered_as_in_suite(bid):
    cls = registry.get(bid)
    assert issubclass(cls, nsb.NemoSkillsBenchmark)
    assert (cls.metric, cls.default_repeats, cls.max_seq_length) == ("accuracy", 1, IDS[bid])
    assert cls.extra == "nemoskills" and cls.dataset_revision == nsr.RULER_COMMIT
    assert cls.dataset.startswith("https://github.com/NVIDIA/RULER/tree/")


def test_no_paid_api(tmp_path, monkeypatch):
    from granite_evals import meter

    monkeypatch.setattr(meter, "metered", lambda *a, **k: pytest.fail("paid API used"))
    b = bench("ruler-128k", tmp_path)
    assert b.judge is None and not b.judge_args


def test_tasks_option(tmp_path):
    assert bench("ruler-64k", tmp_path).tasks() == list(nsr.TASKS) and len(nsr.TASKS) == 13
    assert bench("ruler-64k", tmp_path, options={"tasks": "vt,qa_1"}).tasks() == ["vt", "qa_1"]
    with pytest.raises(SystemExit, match="unknown RULER tasks"):
        bench("ruler-64k", tmp_path, options={"tasks": "vt,nope"}).tasks()


def test_needs_a_tokenizer(tmp_path):
    with pytest.raises(SystemExit, match="tokenizer"):
        bench("ruler-64k", tmp_path, model="none").prepare_data()


def _ruler_src(tmp_path, commit=nsr.RULER_COMMIT):
    root = tmp_path / "ruler"
    j = root / "scripts/data/synthetic/json"
    j.mkdir(parents=True)
    for name in nsr.RULER_JSON_SHA256:
        (j / name).write_text(name)
    (root / "GRANITE_EVALS_COMMIT").write_text(commit + "\n")
    return root


def test_ruler_source_checked(tmp_path, monkeypatch):
    root = _ruler_src(tmp_path)
    b = bench("ruler-128k", tmp_path, options={"ruler_dir": str(root)})
    with pytest.raises(SystemExit, match="sha256"):
        b.check_ruler_source()
    monkeypatch.setattr(nsr, "RULER_JSON_SHA256", {n: hashlib.sha256(n.encode()).hexdigest() for n in nsr.RULER_JSON_SHA256})
    assert set(b.check_ruler_source()) == set(nsr.RULER_JSON_SHA256)
    (root / "GRANITE_EVALS_COMMIT").write_text("0" * 40)
    with pytest.raises(SystemExit, match="is not RULER"):
        b.check_ruler_source()


def _setup(tmp_path, tasks=nsr.TASKS, n=nsr.NUM_SAMPLES, args=CHAT_ARGS):
    d = tmp_path / "setup"
    for t in tasks:
        (d / t).mkdir(parents=True)
        (d / t / "__init__.py").write_text(f"METRICS_TYPE = 'ruler'\nGENERATION_ARGS = ({args})\n")
        rows = [
            {"index": i, "question": f"{t} context {i} ... what is the value {i}?", "expected_answer": [f"v{i}", "w"], "length": 10}
            for i in range(n)
        ]
        (d / t / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_task_generation_args_thinking_off_is_ns(tmp_path):
    d = _setup(tmp_path, tasks=["vt"], args=ARGS)
    b = bench("ruler-128k", tmp_path, options=OFF)
    assert (b.data_format(), b.tokens_to_generate("vt"), b.setup_name()) == ("default", 30, "granite_131072")
    args = b.task_generation_args(d, "vt")
    assert args[:6] == [
        "++prompt_config=generic/default",
        "++eval_type=ruler",
        "++eval_config.match_type=all",
        "++inference.tokens_to_generate=128",
        "++start_assistant_response_key=generation",
        "++inference.endpoint_type=text",
    ]
    assert "++tokenizer=/models/granite" in args and "++inference.temperature=null" in args
    assert sum(a.startswith("++inference.tokens_to_generate=") for a in args) == 1  # the task's own
    assert args[-1] == "++chat_template_kwargs.enable_thinking=false"
    assert b.required_context() == 131072 and b.departures(131072) == []
    b = bench("ruler-128k", tmp_path, options={"max_tokens": "64", **OFF, "tokenizer": "/tok"})
    args = b.task_generation_args(d, "vt")
    assert args[-2:] == ["++inference.tokens_to_generate=64", "++chat_template_kwargs.enable_thinking=false"]
    assert "++tokenizer=/tok" in args


def test_task_generation_args_thinking_on(tmp_path):
    d = _setup(tmp_path, tasks=["vt", "niah_single_1", "qa_2"])
    b = bench("ruler-64k", tmp_path)  # enable_thinking unset: the chat template's default (on)
    assert (b.thinking(), b.data_format(), b.setup_name()) == (True, "chat", "granite_65536_chat")
    args = b.task_generation_args(d, "vt")
    assert not any(a.startswith("++start_assistant_response_key") for a in args)  # no answer-prefix prefill
    assert not any("enable_thinking" in a for a in args)
    # no fixed budget: each sample gets cap - prompt tokens (capped_generation_task)
    assert args[-2:] == ["++inference.endpoint_type=chat", "++inference.tokens_to_generate=null"]
    assert {t: b.answer_budget(t) for t in ("vt", "niah_single_1", "qa_2")} == {"vt": 30, "niah_single_1": 128, "qa_2": 32}
    assert b.required_context() == 65536
    assert len(b.departures(65536)) == 1 and "cap 65536 - prompt tokens" in b.departures(65536)[0]
    b = bench("ruler-64k", tmp_path, options={"enable_thinking": "true"})
    assert b.task_generation_args(d, "vt")[-1] == "++chat_template_kwargs.enable_thinking=true"
    # max_tokens bounds cap - prompt
    b = bench("ruler-64k", tmp_path, options={"max_tokens": "4096"})
    assert b.task_generation_args(d, "qa_2")[-1] == "++inference.tokens_to_generate=4096" and b.required_context() == 65536
    assert b.budget_rule() == "context cap - prompt tokens, at most 4096"


def test_data_format_must_match_thinking(tmp_path):
    text, chat = _setup(tmp_path / "t", tasks=["vt"], args=ARGS), _setup(tmp_path / "c", tasks=["vt"])
    with pytest.raises(SystemExit, match="not ns's 'chat' RULER format"):
        bench("ruler-64k", tmp_path).task_generation_args(text, "vt")
    with pytest.raises(SystemExit, match="not ns's 'default' RULER format"):
        bench("ruler-64k", tmp_path, options=OFF).task_generation_args(chat, "vt")


def test_sample_length(tmp_path):
    b = bench("ruler-128k", tmp_path, options={"sample_length": "114688"})
    assert (b.sample_length(), b.setup_name(), b.required_context()) == (114688, "granite_114688_chat", 114688)
    assert any("114688 tokens, not 131072" in d for d in b.departures(131072))
    with pytest.raises(SystemExit, match="sample_length"):
        bench("ruler-64k", tmp_path, options={"sample_length": "70000"}).sample_length()


def test_prepare_passes_format_and_length(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(nsr.RulerBenchmark, "check_ruler_source", lambda self: {})
    monkeypatch.setattr(nsr.RulerBenchmark, "_task_files", lambda self, d, tasks: {"task_files": {}})
    monkeypatch.setattr(nsr.RulerBenchmark, "_run_prepare", lambda self, ddir, tasks: calls.append(self.setup_name()) or (ddir / self.setup_name()).mkdir(parents=True))
    for opts in ({}, OFF, {"sample_length": "60000"}):
        _, prov = bench("ruler-64k", tmp_path, options={"tasks": "vt", **opts}).prepare_data()
        assert prov["spec"]["data_format"] == ("default" if opts == OFF else "chat")
        assert prov["spec"]["max_seq_length"] == int(opts.get("sample_length", 65536))
    assert calls == ["granite_65536_chat", "granite_65536", "granite_60000_chat"]


def test_task_files_must_be_complete(tmp_path):
    d = _setup(tmp_path, tasks=["vt"], n=3)
    with pytest.raises(RuntimeError, match="expected 100"):
        bench("ruler-128k", tmp_path)._task_files(d, ["vt"])


def test_gold_server_matches_by_context_tail():
    rows = [
        {"question": "x" * 5000 + " needle A?", "expected_answer": ["1234"]},
        {"question": "x" * 5000 + " needle B?", "expected_answer": ["alpha", "beta"]},
    ]
    s = nsr.RulerGoldServer(rows)
    prompt = "<|start|>" + rows[1]["question"] + "<|end|><|assistant|><think>\nAnswer:"
    assert s.lookup(prompt) == "alpha beta"
    assert s.lookup("unrelated") == ""


class _Models(nsb._Server):
    def __init__(self, n):
        self.n = n

    def handle(self, method, path, headers, body):
        return nsb._json_response({"data": [{"id": "served", "max_model_len": self.n}]})


def test_context_check(tmp_path):
    b = bench("ruler-128k", tmp_path, options=OFF)
    with _Models(131072) as s:
        assert b.check_context(s.base_url, "served") == 131072
    with _Models(32768) as s, pytest.raises(SystemExit, match="--max-model-len 131072 .*sample_length=N"):
        b.check_context(s.base_url, "served")
    assert b.check_context("http://127.0.0.1:9/v1", "served") is None  # unknown: not fatal


def test_context_check_thinking(tmp_path):
    """Thinking fills the cap: a server holding a full sample is enough."""
    with _Models(131072) as s:
        assert bench("ruler-128k", tmp_path).check_context(s.base_url, "served") == 131072
        assert bench("ruler-64k", tmp_path).check_context(s.base_url, "served") == 131072
    with _Models(131072) as s, pytest.raises(SystemExit, match="max_model_len=131072 < 262144"):
        bench("ruler-256k", tmp_path).check_context(s.base_url, "served")


def test_context_cap(tmp_path):
    b = bench("ruler-128k", tmp_path)
    assert b.context_cap(131072) == 131072 and b.context_cap(262144) == 262144  # the served max_model_len
    with pytest.raises(SystemExit, match="context_cap=N"):  # thinking needs to know it
        b.context_cap(None)
    assert bench("ruler-128k", tmp_path, options=OFF).context_cap(None) == 131072  # ns RULER's sizing
    assert bench("ruler-128k", tmp_path, options={"answers": "gold"}).context_cap(None) == 131072
    assert bench("ruler-64k", tmp_path, options={"context_cap": "100000"}).context_cap(131072) == 100000
    with pytest.raises(SystemExit, match="> the served max_model_len"):
        bench("ruler-64k", tmp_path, options={"context_cap": "200000"}).context_cap(131072)
    with pytest.raises(SystemExit, match="< sample_length"):
        bench("ruler-128k", tmp_path, options={"context_cap": "65536"}).context_cap(None)
    with pytest.raises(SystemExit, match="thinking_budget was replaced"):
        bench("ruler-128k", tmp_path, options={"thinking_budget": "1000"}).context_cap(131072)


def test_context_checked_before_data(tmp_path, monkeypatch):
    monkeypatch.setattr(nsr.RulerBenchmark, "prepare_data", lambda self: pytest.fail("built data first"))
    with _Models(32768) as s, pytest.raises(SystemExit, match="max_model_len=32768"):
        bench("ruler-64k", tmp_path).run(s.base_url, "served")


SPEC = {"task": "vt", "repeat": 0, "cap": 1000, "answer_budget": 30, "thinking": True, "failures_file": ""}


def test_sample_budget():
    assert nsr.sample_budget(SPEC, 900, None) == (100, None)  # cap - prompt
    assert nsr.sample_budget(SPEC, 900, 64) == (64, None)  # bounded by max_tokens
    assert nsr.sample_budget(SPEC, 970, None) == (30, None)  # exactly the answer budget fits
    assert nsr.sample_budget(SPEC, 971, None) == (29, "prompt_exceeds_cap")
    off = {**SPEC, "thinking": False}
    assert nsr.sample_budget(off, 970, 30) == (30, None)  # ns's task budget, unchanged
    assert nsr.sample_budget(off, 971, 30) == (30, "prompt_exceeds_cap")


def test_classify_result():
    ctx = {"generation": "", "error": "context_window_exceeded", "detailed_error": "No strategy configured.", "finish_reason": "error"}
    assert nsr.classify_result(ctx) == "context_length_error"
    vllm = {"generation": "", "error": "See detailed_error", "finish_reason": "error",
            "detailed_error": "This model's maximum context length is 131072 tokens. However, you requested 131100"}  # fmt: skip
    assert nsr.classify_result(vllm) == "context_length_error"
    assert nsr.classify_result({"generation": "", "error": "See detailed_error", "detailed_error": "timeout"}) is None
    assert nsr.classify_result({"generation": "", "finish_reason": "length"}) == "length_before_answer"
    assert nsr.classify_result({"generation": " 12", "finish_reason": "length"}) is None  # an answer, cut: ns scores it
    assert nsr.classify_result({"generation": "12", "finish_reason": "stop"}) is None


class _Tok:
    """One token per character; the chat template wraps each message in 2 tokens
    (and kwargs add one)."""

    def encode(self, text, add_special_tokens):
        return list(text) + ([0] if add_special_tokens else [])

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        return [0] * (sum(len(m["content"]) + 2 for m in messages) + add_generation_prompt + len(kwargs))


def test_count_prompt_tokens():
    assert nsr.count_prompt_tokens(_Tok(), "abc", None) == 4
    msgs = [{"role": "user", "content": "abcd"}]
    assert nsr.count_prompt_tokens(_Tok(), msgs, None) == 7
    assert nsr.count_prompt_tokens(_Tok(), msgs, {"enable_thinking": True}) == 8


def test_think_tags_in_generation_fail(tmp_path):
    b = bench("ruler-64k", tmp_path)
    f = tmp_path / "out.jsonl"
    f.write_text(json.dumps({"index": 0, "generation": "The value is 12.", "reasoning_content": "needle 12"}) + "\n")
    b.check_generations(f)
    for gen in ("<think>needle 12</think>The value is 12.", "needle 12</think>12"):
        f.write_text(json.dumps({"index": 3, "generation": gen}) + "\n")
        with pytest.raises(RuntimeError, match=r"think tags .* samples \[3\]"):
            b.check_generations(f)


def test_reset_if_input_changed(tmp_path):
    d = tmp_path / "vt"
    d.mkdir()
    (d / "input.jsonl").write_text("a\n")
    nsr._reset_task_if_input_changed(d)
    (d / "output-rs0.jsonl").write_text("{}\n")
    nsr._reset_task_if_input_changed(d)
    assert (d / "output-rs0.jsonl").exists()
    (d / "input.jsonl").write_text("b\n")
    nsr._reset_task_if_input_changed(d)
    assert not (d / "output-rs0.jsonl").exists()


# -- with the harness ---------------------------------------------------------


@pytest.fixture
def ns():
    return pytest.importorskip("nemo_skills")


def _fake_generate(correct_every, cut_every=0):
    """Rows ``index % cut_every == 1`` stop at the length cap inside their thinking."""

    def gen(self, setup_dir, task, input_file, base_url, served, k, cap):
        rows = [json.loads(line) for line in Path(input_file).read_text().splitlines()]
        out = Path(input_file).parent / f"output-rs{k}.jsonl"
        lines = []
        for r in rows:
            cut = bool(cut_every) and r["index"] % cut_every == 1
            g = {"generation": "", "reasoning_content": "v1 ...", "finish_reason": "length", "num_generated_tokens": 100,
                 "granite_failure": "length_before_answer"}  # fmt: skip
            if not cut:
                g = {"generation": "g", "reasoning_content": "r", "finish_reason": "stop", "num_generated_tokens": 10 + r["index"]}
            g.update(granite_prompt_tokens=40, granite_max_tokens=1000 - 40)
            lines.append(json.dumps({**r, **g, "is_correct": not cut and r["index"] % correct_every == 0}) + "\n")
        out.write_text("".join(lines))

    return gen


def test_run_scores_with_ns_ruler_score(ns, tmp_path, monkeypatch):
    d = _setup(tmp_path)
    b = bench("ruler-128k", tmp_path / "out", limit=4, options={"answers": "gold"})
    monkeypatch.setattr(nsr.RulerBenchmark, "prepare_data", lambda self: (d, {"source": "fake"}))
    monkeypatch.setattr(nsr.RulerBenchmark, "_generate_task", _fake_generate(2))
    r = b.run("", "")
    assert r["n"] == 13 * 4 and r["value"] == pytest.approx(0.5)  # indices 0..3: 0 and 2 correct
    assert r["per_task"]["vt"] == {"accuracy": pytest.approx(0.5), "n": 4, "statuses": {"stop": 4}}
    assert not r["partial_task_set"] and r["ns_metric"]["score"] == "ruler_score"
    assert r["statuses"] == {"stop": 52}
    assert r["pass_at_k"] == {"k": 1, "pass_at_1": pytest.approx(0.5), "pass_at_k": pytest.approx(0.5), "n": 52, "how": "ns metrics, ruler_score"}
    th = r["thinking"]
    assert (th["enabled"], th["budget"], th["data_format"], th["scoring_source"]) == (
        True,
        "context cap - prompt tokens",
        "chat",
        nsr.SCORING_SOURCES["chat"],
    )
    assert th["per_task"]["vt"]["answer_budget"] == 30 and th["per_task"]["vt"]["tokens_to_generate"] is None
    assert th["per_task"]["vt"]["generated_tokens"] == {"p50": 12, "p95": 13, "max": 13}
    assert th["per_task"]["vt"]["max_tokens"] == {"p50": 1000 - 40, "p95": 1000 - 40, "max": 1000 - 40}
    assert r["sample_length"] == 131072 and r["required_context"] == 131072 and r["context_cap"] == 131072
    assert r["failures"] == {"file": nsr.FAILURES_FILE, "total": 0, "by_reason": dict.fromkeys(nsr.FAILURE_REASONS, 0), "per_task": {}}
    assert len(r["departures"]) == 1 and r["generation_args"]["vt"][-1] == "++inference.tokens_to_generate=null"


def test_run_pass_at_k(ns, tmp_path, monkeypatch):
    """k = 2, each repeat right on the other half of the samples: pass@1 0.5, pass@2 1."""
    d = _setup(tmp_path, tasks=["vt", "cwe"])
    b = bench("ruler-64k", tmp_path / "out", limit=4, repeats=2, options={"answers": "gold", "tasks": "vt,cwe"})
    monkeypatch.setattr(nsr.RulerBenchmark, "prepare_data", lambda self: (d, {"source": "fake"}))
    gen = _fake_generate(2)

    def shifted(self, setup_dir, task, input_file, base_url, served, k, cap):
        gen(self, setup_dir, task, input_file, base_url, served, k, cap)
        out = Path(input_file).parent / f"output-rs{k}.jsonl"
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        out.write_text("".join(json.dumps({**r, "is_correct": (r["index"] + k) % 2 == 0}) + "\n" for r in rows))

    monkeypatch.setattr(nsr.RulerBenchmark, "_generate_task", shifted)
    r = b.run("", "")
    assert r["value"] == pytest.approx(0.5) and r["ns_metric"]["aggregation"] == "pass@1[avg-of-2]"
    assert r["pass_at_k"] == {"k": 2, "pass_at_1": pytest.approx(0.5), "pass_at_k": pytest.approx(1.0), "n": 8, "how": "ns metrics, ruler_score"}


def test_run_cut_off_thinking_scores_zero(ns, tmp_path, monkeypatch):
    d = _setup(tmp_path, tasks=["vt"])
    b = bench("ruler-64k", tmp_path / "out", limit=4, options={"answers": "gold", "tasks": "vt"})
    monkeypatch.setattr(nsr.RulerBenchmark, "prepare_data", lambda self: (d, {"source": "fake"}))
    monkeypatch.setattr(nsr.RulerBenchmark, "_generate_task", _fake_generate(1, cut_every=2))
    r = b.run("", "")
    assert r["value"] == pytest.approx(0.5)  # indices 1 and 3 cut off inside the thinking
    vt = r["thinking"]["per_task"]["vt"]
    assert (vt["length_no_content"], vt["length_with_content"], vt["with_reasoning"]) == (2, 0, 4)
    assert r["failures"]["total"] == 2 and r["failures"]["by_reason"]["length_before_answer"] == 2
    assert r["failures"]["per_task"] == {"vt": {"length_before_answer": 2}}


def test_run_thinking_off_record(ns, tmp_path, monkeypatch):
    d = _setup(tmp_path, tasks=["vt"], args=ARGS)
    b = bench("ruler-64k", tmp_path / "out", limit=2, options={"answers": "gold", "tasks": "vt", **OFF})
    monkeypatch.setattr(nsr.RulerBenchmark, "prepare_data", lambda self: (d, {"source": "fake"}))
    monkeypatch.setattr(nsr.RulerBenchmark, "_generate_task", _fake_generate(1))
    r = b.run("", "")
    assert r["thinking"]["enabled"] is False and r["thinking"]["budget"] == "ns/RULER task budget" and r["departures"] == []
    assert r["thinking"]["per_task"]["vt"]["tokens_to_generate"] == 30
    assert r["thinking"]["scoring_source"] == nsr.SCORING_SOURCES["default"]
    assert r["thinking"]["endpoint"] == "text (answer prefix prefilled)"


def test_scoring_source_by_mode(tmp_path):
    """Thinking on scores the post-parser message content; thinking off (text
    endpoint, no reasoning parser) scores the raw completion."""
    on = bench("ruler-64k", tmp_path)
    off = bench("ruler-64k", tmp_path, options=OFF)
    assert on.scoring_source() == nsr.SCORING_SOURCES["chat"] and "reasoning parser" in on.scoring_source()
    assert off.scoring_source() == nsr.SCORING_SOURCES["default"]
    assert "raw text completion" in off.scoring_source() and "no reasoning parser" in off.scoring_source()


def test_run_task_subset(ns, tmp_path, monkeypatch):
    d = _setup(tmp_path, tasks=["vt", "cwe"])
    b = bench("ruler-64k", tmp_path / "out", limit=3, options={"answers": "gold", "tasks": "vt,cwe"})
    monkeypatch.setattr(nsr.RulerBenchmark, "prepare_data", lambda self: (d, {"source": "fake"}))
    monkeypatch.setattr(nsr.RulerBenchmark, "_generate_task", _fake_generate(1))
    r = b.run("", "")
    assert r["value"] == 1.0 and r["partial_task_set"] and r["tasks"] == ["vt", "cwe"]


def _word_tokenizer(path):
    """A local HF tokenizer: one token per word; the chat template adds 2 per message and 1 to prompt."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    template = "{% for m in messages %}[UNK] {{ m['content'] }} [UNK] {% endfor %}{% if add_generation_prompt %}[UNK]{% endif %}"
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", chat_template=template).save_pretrained(path)
    return str(path)


class _CapServer(nsb._Server):
    """A chat endpoint: ``LENGTH`` samples stop at the cap in the thinking, ``CTX``
    samples get vLLM's context-length 400; others answer. Records max_tokens."""

    def __init__(self):
        self.max_tokens = {}

    def handle(self, method, path, headers, body):
        if not path.rstrip("/").endswith("/completions"):
            return nsb._json_response({"data": [{"id": "m"}]})
        req = json.loads(body)
        q = req["messages"][-1]["content"]
        self.max_tokens[q.split()[0]] = req.get("max_tokens") or req.get("max_completion_tokens")
        if "CTX" in q:
            msg = "This model's maximum context length is 1000 tokens. However, you requested 1001 tokens."
            return nsb._json_response({"object": "error", "message": msg, "type": "BadRequestError", "code": 400}, 400)
        content, finish = ("", "length") if "LENGTH" in q else ("v0 w", "stop")
        choice = {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}
        usage = {"prompt_tokens": 1, "completion_tokens": 7, "total_tokens": 8}
        return nsb._json_response({"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": [choice], "usage": usage})


def test_capped_generation_through_ns(ns, tmp_path):
    """ns generate with the per-sample budget, end to end (Hydra, ns's chat model,
    soft fail): max_tokens = cap - prompt; context failures score 0 and are logged."""
    pytest.importorskip("transformers")
    tok = _word_tokenizer(tmp_path / "tok")
    setup = _setup(tmp_path, tasks=["vt"], n=4)
    questions = ["s0 short", "s1 LENGTH", "s2 " + "word " * 980, "s3 CTX"]  # s2: 981 words + 3 template tokens
    rows = [{"index": i, "question": q, "expected_answer": ["v0", "w"]} for i, q in enumerate(questions)]
    out = tmp_path / "out"
    b = bench("ruler-64k", out, options={"tokenizer": tok, "tasks": "vt"})
    inp = out / "ruler" / "vt" / "input.jsonl"
    nsb._write_jsonl(inp, rows)
    with _CapServer() as s:
        b._generate_task(setup, "vt", inp, s.base_url, "m", 0, 1000)
    got = {r["index"]: r for r in nsb._read_jsonl(inp.parent / "output-rs0.jsonl")}
    assert [got[i]["granite_failure"] for i in range(4)] == [None, "length_before_answer", "prompt_exceeds_cap", "context_length_error"]
    assert got[0]["granite_prompt_tokens"] == 2 + 3 and got[0]["granite_max_tokens"] == 1000 - 5
    assert s.max_tokens == {"s0": 995, "s1": 995, "s3": 995}  # s2 is never sent
    assert got[2]["granite_max_tokens"] == 1000 - 984 and got[2]["generation"] == ""
    log = [json.loads(line) for line in (out / nsr.FAILURES_FILE).read_text().splitlines()]
    assert sorted((r["index"], r["reason"]) for r in log) == [(1, "length_before_answer"), (2, "prompt_exceeds_cap"), (3, "context_length_error")]
    assert {r["task"] for r in log} == {"vt"} and next(r for r in log if r["index"] == 2)["prompt_tokens"] == 984
    assert next(r for r in log if r["index"] == 1)["generated_tokens"] == 7
    assert b.failures_record(out / "ruler", ["vt"])["by_reason"] == {
        "prompt_exceeds_cap": 1, "context_length_error": 1, "length_before_answer": 1}  # fmt: skip


def test_ns_ruler_evaluator_accepts_gold_answers(ns, tmp_path):
    """The gold server's answer passes ns's own RULER match (all / part)."""
    from nemo_skills.evaluation.evaluator import evaluate

    rows = [{"generation": "alpha beta", "expected_answer": ["alpha", "beta"]}, {"generation": "x", "expected_answer": ["alpha"]}]
    f = tmp_path / "out.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    evaluate("ruler", {"input_file": str(f), "match_type": "all"})
    got = [json.loads(line)["is_correct"] for line in f.read_text().splitlines()]
    assert got == [1.0, 0.0]


# Imported only inside functions that the hf tokenizer path never calls.
_RULER_LAZY = {"nemo", "google", "tiktoken", "manifest_utils", "constants", "template", "tokenizer"}


@pytest.mark.skipif(not Path(nsr.RULER_DIR, "scripts", "data").is_dir(), reason="no image RULER checkout")
def test_image_has_ruler_generator_imports():
    """In the image: every module RULER's generators import at the top is installed
    (the job's prepare runs them with this interpreter)."""
    import ast
    import importlib.util
    import sys

    root = Path(nsr.RULER_DIR, "scripts", "data")
    local = {p.stem for p in root.rglob("*.py")}
    missing = {}
    for p in root.rglob("*.py"):
        if p.name.startswith("download_"):  # build-time only (docker/extras/nemoskills.sh)
            continue
        for node in ast.parse(p.read_text()).body:
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for n in names:
                top = n.split(".")[0]
                if top in local or top in _RULER_LAZY or top in sys.stdlib_module_names:
                    continue
                if importlib.util.find_spec(top) is None:
                    missing.setdefault(top, []).append(str(p.relative_to(root)))
    assert not missing, missing


# -- phases --------------------------------------------------------------------


def _fake_ns(calls):
    """Stands in for ns generation: answers even samples right; grades them with
    ns's RULER match in-process unless the generation runs with eval_type=null."""
    from nemo_skills.evaluation.evaluator import evaluate

    def run(cmd, out, *, log_path, what):
        calls.append(what)
        args = dict(a[2:].split("=", 1) for a in cmd if a.startswith("++"))  # the last override wins
        rows = nsb._read_jsonl(Path(args["input_file"]))
        gen = [
            {**r, "generation": " ".join(r["expected_answer"]) if r["index"] % 2 == 0 else "x", "reasoning_content": "r",
             "finish_reason": "stop", "num_generated_tokens": 10 + r["index"]}
            for r in rows
        ]  # fmt: skip
        nsb._write_jsonl(out, gen)
        if args["eval_type"] != "null":
            evaluate(args["eval_type"], {"input_file": str(out), "match_type": args["eval_config.match_type"]})

    return run


def _prepared(monkeypatch):
    """prepare_data's cache as the generate phase leaves it (no RULER build)."""

    def run_prepare(b, ddir, tasks):
        _setup(ddir, tasks=tasks).rename(ddir / b.setup_name())

    monkeypatch.setattr(nsr.RulerBenchmark, "check_ruler_source", lambda b: {})
    monkeypatch.setattr(nsr.RulerBenchmark, "_run_prepare", run_prepare)


def test_run_generate_then_score(ns, tmp_path, monkeypatch):
    kw = {"limit": 4, "options": {"tasks": "vt,qa_1"}}
    _prepared(monkeypatch)
    monkeypatch.setattr(nsr, "_run_ns", _fake_ns(calls := []))
    monkeypatch.setattr(nsr.RulerBenchmark, "check_context", lambda b, url, served: 262144)
    out = tmp_path / "split"
    g = bench("ruler-128k", out, phase="generate", **kw).run("http://127.0.0.1:9/v1", "m")
    assert g["n"] == 8 and "value" not in g and g["served_max_model_len"] == g["context_cap"] == 262144
    assert g["thinking"]["scoring_source"] == nsr.SCORING_SOURCES["chat"] and set(g["generation_args"]) == {"vt", "qa_1"}
    vt = out / "ruler" / "vt" / "output-rs0.jsonl"
    assert all("is_correct" not in r for r in nsb._read_jsonl(vt)) and nsb._unscored(vt).exists()
    (out / "generation.json").write_text(json.dumps({"details": g}))  # as the CLI records it

    for name in ("_run_ns", "RulerGoldServer"):
        monkeypatch.setattr(nsr, name, lambda *a, **k: pytest.fail("score phase generated"))
    for name in ("check_context", "tokenizer_fingerprint", "_run_prepare"):
        monkeypatch.setattr(nsr.RulerBenchmark, name, lambda *a, **k: pytest.fail(f"score phase ran {a}"))
    s = bench("ruler-128k", out, phase="score", **kw).run("", "m")
    assert (s["value"], s["n"]) == (pytest.approx(0.5), 8) and not nsb._unscored(vt).exists()
    assert s["served_max_model_len"] == g["served_max_model_len"] and s["context_cap"] == 262144
    assert "cap 262144" in s["departures"][0] and s["failures"] == g["failures"]
    assert s["thinking"]["per_task"]["vt"]["generated_tokens"] == {"p50": 12, "p95": 13, "max": 13}
    monkeypatch.undo()

    _prepared(monkeypatch)
    monkeypatch.setattr(nsr, "_run_ns", _fake_ns(calls_all := []))
    monkeypatch.setattr(nsr.RulerBenchmark, "check_context", lambda b, url, served: 262144)
    a = bench("ruler-128k", tmp_path / "all", **kw).run("http://127.0.0.1:9/v1", "m")
    strip = ("data_provenance", "generation_args")  # paths under each output dir
    assert {k: v for k, v in a.items() if k not in strip} == {k: v for k, v in s.items() if k not in strip}
    assert len(calls_all) == len(calls) == 2


def test_score_without_generation(ns, tmp_path, monkeypatch):
    kw = {"limit": 2, "options": {"tasks": "vt,cwe", "answers": "gold"}}
    with pytest.raises(RuntimeError, match="no generation for RULER data"):
        bench("ruler-64k", tmp_path, phase="score", **kw).run("", "gold")
    _prepared(monkeypatch)
    monkeypatch.setattr(nsr, "_run_ns", _fake_ns([]))
    bench("ruler-64k", tmp_path, phase="generate", **kw).run("", "")
    (tmp_path / "ruler" / "cwe" / "output-rs0.jsonl").unlink()
    with pytest.raises(RuntimeError, match="no generation for cwe repeat 0"):
        bench("ruler-64k", tmp_path, phase="score", **kw).run("", "gold")
    with pytest.raises(SystemExit, match="differs from what was generated"):
        bench("ruler-64k", tmp_path, phase="score", **{**kw, "limit": 1}).run("", "gold")
