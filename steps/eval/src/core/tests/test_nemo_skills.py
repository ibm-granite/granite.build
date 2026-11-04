"""NeMo-Skills adapter: config, pinning, judge metering (always), and ns's own
evaluate/metrics plus a gold end-to-end run (with the nemoskills extra)."""

import json

import httpx
import pytest

from granite_evals import registry
from granite_evals.benchmarks import nemo_skills as nsb
from granite_evals.registry import RunConfig

IDS = {
    "aime25": ("pass@1 symbolic correct", 1),
    "hmmt-feb25": ("pass@1 symbolic correct", 1),
    "gpqa": ("pass@1 symbolic correct", 2),
    "mmlu-pro": ("5-shot CoT symbolic correct", 1),
    "arena-hard-v2": ("win rate", 1),
}


def bench(bid, tmp_path, **kw):
    return registry.get(bid)(RunConfig(model="m", output_dir=tmp_path, **kw))


@pytest.mark.parametrize("bid", IDS)
def test_registered_as_in_suite(bid):
    cls = registry.get(bid)
    assert issubclass(cls, nsb.NemoSkillsBenchmark)
    assert (cls.metric, cls.default_repeats) == IDS[bid]
    assert cls.extra == "nemoskills" and "nemo-skills" in cls.harness_packages
    assert len(cls.dataset_revision) == 40


def test_pins_include_primary_dataset(tmp_path):
    b = bench("gpqa", tmp_path)
    assert b.pins() == {"Idavidrein/gpqa": "83022cefff930aea54f654c0b282e74b9eeda5c6"}
    assert bench("aime25", tmp_path).pins() == {}  # bundled in the pinned ns commit
    urls = registry.get("arena-hard-v2").pinned_urls
    assert len(urls) == 3
    for orig, (pinned, sha) in urls.items():
        assert "/main/" in orig and nsb._ARENA_HARD_COMMIT in pinned and len(sha) == 64


def test_sampling_defaults_to_generation_config(tmp_path):
    assert bench("aime25", tmp_path).sampling() == {"temperature": None, "top_p": None, "top_k": -1, "max_tokens": None}
    b = bench("aime25", tmp_path, options={"temperature": "0.6", "max_tokens": "1000"})
    assert b.sampling()["temperature"] == 0.6 and b.sampling()["max_tokens"] == 1000
    assert nsb._hydra(None) == "null"


def test_passthrough_options():
    opts = {"ns.chat_template_kwargs.enable_thinking": "false", "judge.x": "1", "temperature": "1"}
    assert nsb._passthrough(opts, "ns.") == ["++chat_template_kwargs.enable_thinking=false"]
    assert nsb._passthrough(opts, "judge.") == ["++x=1"]


def test_aggregation_follows_repeats(tmp_path):
    assert bench("aime25", tmp_path).aggregation() == "pass@1"  # one sample by default
    assert bench("aime25", tmp_path, repeats=4).aggregation() == "pass@1[avg-of-4]"


def test_gold_disables_server(tmp_path):
    assert bench("aime25", tmp_path).needs_server()
    assert not bench("aime25", tmp_path, options={"answers": "gold"}).needs_server()


def test_gold_generation_formats(tmp_path):
    row = {"expected_answer": "C", "problem": "p"}
    assert bench("gpqa", tmp_path).gold_generation(row) == "Answer: C"
    assert bench("mmlu-pro", tmp_path).gold_generation(row) == "\\boxed{C}"  # 5-shot CoT
    assert bench("mmlu-pro", tmp_path, options={"shots": "0"}).gold_generation(row) == "Answer: C"
    assert bench("aime25", tmp_path).gold_generation({"expected_answer": "70"}) == "\\boxed{70}"


def test_gold_server_matches_longest_problem():
    rows = [{"problem": "Find x.", "a": "1"}, {"problem": "Find x. Then find y.", "a": "2"}]
    with nsb._GoldServer(rows, lambda r: r["a"]) as s:
        r = httpx.post(
            s.base_url + "/chat/completions",
            json={"model": "gold", "messages": [{"role": "user", "content": "Solve: Find x. Then find y. Box it."}]},
        )
        assert r.json()["choices"][0]["message"]["content"] == "2"
        assert httpx.get(s.base_url + "/models").json()["data"][0]["id"] == "gold"


def test_usage_fields_openai_and_anthropic_style():
    assert nsb.usage_fields(
        {"prompt_tokens": 10, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 4}}
    ) == {"prompt_tokens": 10, "completion_tokens": 3, "cached_tokens": 4, "cache_creation_tokens": 0}
    assert nsb.usage_fields(
        {"input_tokens": 7, "output_tokens": 2, "cache_read_input_tokens": 5, "cache_creation_input_tokens": 1}
    ) == {"prompt_tokens": 7, "completion_tokens": 2, "cached_tokens": 5, "cache_creation_tokens": 1}


class _Upstream(nsb._Server):
    def __init__(self):
        self.auth = []

    def handle(self, method, path, headers, body):
        self.auth.append((path, {k.lower(): v for k, v in headers.items()}.get("authorization")))
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 60}}
        return nsb._json_response({"choices": [{"message": {"content": "[[A>B]]"}}], "usage": usage, "model": "j"})


def test_metering_proxy_injects_key_and_totals_usage(tmp_path):
    log = tmp_path / "judge" / "usage.jsonl"
    with _Upstream() as up, nsb._MeteringProxy(up.base_url, "sekret", log) as proxy:
        for _ in range(3):
            r = httpx.post(proxy.base_url + "/chat/completions", json={"model": "j"}, headers={"Authorization": "x"})
            assert r.status_code == 200
    assert up.auth == [("/v1/chat/completions", "Bearer sekret")] * 3
    assert "sekret" not in log.read_text()
    u = nsb.judge_usage(log, n_examples=3)
    assert (u["requests"], u["prompt_tokens"], u["completion_tokens"], u["cached_tokens"]) == (3, 300, 60, 180)
    assert u["per_example"]["prompt_tokens"] == 100 and u["failed_requests"] == 0


def test_judge_endpoint_defaults_and_self(tmp_path):
    ep = bench("arena-hard-v2", tmp_path).judge_endpoint("http://v/v1", "g")
    assert ep == {
        "base_url": nsb.JUDGE_BASE_URL,
        "model": "aws/claude-sonnet-5",
        "api_key_env": "GRANITE_EVALS_JUDGE_API_KEY",
        "is_self": False,
    }
    ep = bench("arena-hard-v2", tmp_path, options={"judge_model": "self"}).judge_endpoint("http://v/v1", "g")
    assert ep["base_url"] == "http://v/v1" and ep["model"] == "g" and ep["is_self"]


def test_statuses_count_soft_failures(tmp_path):
    f = tmp_path / "o.jsonl"
    f.write_text('{"finish_reason": "stop"}\n{"finish_reason": "error"}\n{"error": "x"}\n')
    assert nsb._statuses(f) == {"stop": 1, "error": 2}


# -- with the harness ---------------------------------------------------------


@pytest.fixture
def ns():
    return pytest.importorskip("nemo_skills")


def test_module_generation_args(ns, tmp_path):
    assert bench("aime25", tmp_path).generation_args([])[:2] == ["++prompt_config=generic/math", "++eval_type=math"]
    gpqa = bench("gpqa", tmp_path)
    assert gpqa.split() == "diamond" and gpqa.metrics_type() == "multichoice"
    assert "++prompt_config=eval/aai/mcq-4choices" in gpqa.generation_args([])
    assert bench("mmlu-pro", tmp_path).split() == "test"
    arena = bench("arena-hard-v2", tmp_path)
    assert arena.uses_judge() and not bench("aime25", tmp_path).uses_judge()
    assert arena.judge_module() == "nemo_skills.inference.eval.arena_judge"
    assert "++inference.top_p=null" in arena.judge_overrides()


def test_mmlu_pro_shots(ns, tmp_path):
    five = bench("mmlu-pro", tmp_path).generation_args([])
    assert five[:2] == ["++prompt_config=generic/general-boxed", "++examples_type='{examples_type}'"]
    assert "++eval_type=multichoice" in five and sum(a.startswith("++prompt_config=") for a in five) == 1
    zero = bench("mmlu-pro", tmp_path, options={"shots": "0"}).generation_args([])
    assert zero[0] == "++prompt_config=eval/aai/mcq-10choices" and not any("examples_type" in a for a in zero)
    with pytest.raises(SystemExit, match="use 0 or 5"):
        bench("mmlu-pro", tmp_path, options={"shots": "3"}).shots()


def test_mmlu_pro_five_shot_prompt_is_ns_per_category(ns, tmp_path):
    """The args reach ns as a per-row examples_type: Hydra keeps the string, and
    ns fills the row's category's 5 validation CoT examples before the question."""
    from hydra import compose, initialize
    from nemo_skills.dataset.utils import get_mcq_fields
    from nemo_skills.inference import generate  # noqa: F401  registers base_generation_config
    from nemo_skills.prompt.few_shot_examples.mmlu_pro import examples_map
    from nemo_skills.prompt.utils import get_prompt

    args = bench("mmlu-pro", tmp_path).generation_args([])
    with initialize(version_base=None):
        cfg = compose(config_name="base_generation_config", overrides=[a for a in args if a.split("=")[0] in ("++prompt_config", "++examples_type")])
    assert cfg.examples_type == "{examples_type}"
    prompt = get_prompt(prompt_config=cfg.prompt_config, examples_type=cfg.examples_type)
    row = {"examples_type": "mmlu_pro_few_shot_computer_science", **get_mcq_fields("What is 2+2?", ["3", "4"])}
    text = prompt.fill(row)[-1]["content"]
    shots = examples_map["mmlu_pro_few_shot_computer_science"]
    assert len(shots) == 5 and all(s["solution"] in text for s in shots)
    assert text.count("The answer is \\boxed{") == 5 and text.endswith("What is 2+2?\n\nA) 3\nB) 4")


def _math_rows(tmp_path, name, correct):
    f = tmp_path / name
    rows = [
        {"problem": f"p{i}", "expected_answer": "1", "predicted_answer": "1", "symbolic_correct": c, "num_generated_tokens": 5}
        for i, c in enumerate(correct)
    ]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return f


def test_ns_math_metrics_map_to_value(ns, tmp_path):
    b = bench("aime25", tmp_path, repeats=2)
    files = [_math_rows(tmp_path, "output-rs0.jsonl", [True, False]), _math_rows(tmp_path, "output-rs1.jsonl", [True, True])]
    m = b.compute_metrics(files)
    assert m["_all_"][b.aggregation()]["symbolic_correct"] == pytest.approx(75.0)
    assert m["_all_"]["pass@2"]["symbolic_correct"] == pytest.approx(100.0)
    assert b.pass_at_k(m, 2) == {"k": 2, "pass_at_1": 0.75, "pass_at_k": 1.0, "n": 2, "how": "ns metrics"}
    one = bench("aime25", tmp_path)
    assert one.pass_at_k(one.compute_metrics(files[:1]), 2) == {"k": 1, "pass_at_1": 0.5, "pass_at_k": 0.5, "n": 2, "how": "ns metrics"}


def _arena_file(tmp_path, verdicts):
    f = tmp_path / "judged.jsonl"
    rows = [
        {"judgement-gen-base": f"[[{a}]]", "judgement-base-gen": f"[[{b}]]", "category": "hard_prompt", "generation": "x"}
        for a, b in verdicts
    ]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return f


def test_arena_one_sided_sample_falls_back(ns, tmp_path):
    b = bench("arena-hard-v2", tmp_path)
    m = b.compute_metrics([_arena_file(tmp_path, [("A>B", "B>A")] * 3)])  # candidate wins every game
    assert m["_all_"]["pass@1"]["score"] == 100.0
    assert "one-sided" in m["win_rate_method"]


def test_arena_mixed_sample_uses_ns_fit(ns, tmp_path):
    b = bench("arena-hard-v2", tmp_path)
    verdicts = [("A>B", "B>A"), ("B>A", "A>B"), ("A=B", "A=B"), ("A>>B", "B>A"), ("B>>A", "A>B")]
    m = b.compute_metrics([_arena_file(tmp_path, verdicts)])
    assert 0 < m["_all_"]["pass@1"]["score"] < 100
    assert m["win_rate_method"] == "bootstrapped Bradley-Terry (ns)"


def test_aime25_gold_end_to_end_and_resume(ns, tmp_path):
    """Reference answers through ns's real prompt -> generate -> math grading."""
    b = bench("aime25", tmp_path, limit=3, repeats=2, workers=3, options={"answers": "gold"})
    r = b.run("", "")
    assert (r["value"], r["n"]) == (1.0, 3)
    assert r["pass_at_k"] == {"k": 2, "pass_at_1": 1.0, "pass_at_k": 1.0, "n": 3, "how": "ns metrics"}
    assert r["metrics"] == {"symbolic_correct": 1.0}
    assert r["data_provenance"]["sha256"] == nsb.AIME25.prepared_sha256
    assert r["statuses"] == {"stop": 6}
    gen = tmp_path / "generation" / "output-rs0.jsonl"
    first = [json.loads(line) for line in gen.read_text().splitlines()]
    assert [row["id"] for row in first] == ["aime25-0", "aime25-1", "aime25-2"]
    assert all(row["symbolic_correct"] for row in first)
    mtime = gen.stat().st_mtime_ns
    assert b.run("", "")["value"] == 1.0 and gen.stat().st_mtime_ns == mtime  # resumed, not regenerated
    b2 = bench("aime25", tmp_path, limit=2, repeats=1, options={"answers": "gold"})
    assert b2.run("", "")["n"] == 2 and len(gen.read_text().splitlines()) == 2  # new input: regenerated


def test_paid_judge_goes_through_meter(tmp_path, monkeypatch):
    from granite_evals import meter

    calls = []
    monkeypatch.setattr(meter, "metered", lambda url, role: calls.append((url, role)) or "http://127.0.0.1:9/metered")
    b = bench("arena-hard-v2", tmp_path)
    assert b.judge_upstream(b.judge_endpoint("http://v/v1", "g")) == "http://127.0.0.1:9/metered"
    assert calls == [(nsb.JUDGE_BASE_URL, "judge")]
    # the CLI has metered a judge_base_url option already; self-judging is unpaid
    opt = bench("arena-hard-v2", tmp_path, options={"judge_base_url": "http://127.0.0.1:8/v1"})
    assert opt.judge_upstream(opt.judge_endpoint("", "")) == "http://127.0.0.1:8/v1"
    me = bench("arena-hard-v2", tmp_path, options={"judge_model": "self"})
    assert me.judge_upstream(me.judge_endpoint("http://v/v1", "g")) == "http://v/v1"
    assert len(calls) == 1


def test_judge_calls_reach_the_meter_proxy(tmp_path, monkeypatch):
    """End to end through the real granite meter: judge proxy -> meter -> upstream."""
    from granite_evals import meter

    monkeypatch.setenv("GRANITE_EVALS_SPEND_LEDGER", str(tmp_path / "ledger.jsonl"))
    with _Upstream() as up, meter.Meters({}, {"benchmark": "arena-hard-v2"}) as meters:
        monkeypatch.setattr(nsb, "JUDGE_BASE_URL", up.base_url)
        b = bench("arena-hard-v2", tmp_path)
        upstream = b.judge_upstream(b.judge_endpoint("", ""))
        assert upstream.startswith("http://127.0.0.1:") and upstream != up.base_url
        with nsb._MeteringProxy(upstream, "k", tmp_path / "usage.jsonl") as proxy:
            httpx.post(proxy.base_url + "/chat/completions", json={"model": "aws/claude-sonnet-5"})
        assert meters.summary()["calls"] == 1
    assert up.auth == [("/v1/chat/completions", "Bearer k")]
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 1


class _Broke(nsb._Server):
    def handle(self, method, path, headers, body):
        return nsb._json_response({"error": {"type": "budget_exhausted"}}, 402)


def test_proxy_flags_budget_exhausted(tmp_path):
    with _Broke() as up, nsb._MeteringProxy(up.base_url, "", tmp_path / "u.jsonl") as proxy:
        assert httpx.post(proxy.base_url + "/chat/completions", json={}).status_code == 402
        assert proxy.budget_exhausted


def _fake_arena_judge(verdicts, calls):
    """Stands in for ns's arena judge: judges the rows its resume file lacks,
    verdicts[i] for row i, as ns with ++skip_filled does (generate.py:559-590)."""

    def run(cmd, out, *, log_path, what):
        args = dict(a[2:].split("=", 1) for a in cmd if a.startswith("++"))
        rows = nsb._read_jsonl(nsb.Path(args["input_file"]))
        resume = out.with_name(out.name + "-async")
        done = {r.pop("_async_position"): r for r in nsb._read_jsonl(resume)}
        for i, row in enumerate(rows):
            if i not in done:
                calls.append(i)
                a, b = verdicts[i]
                done[i] = {**row, "judgement-gen-base": a, "judgement-base-gen": b, "judgement": ""}
        nsb._write_jsonl(out, [done[i] for i in range(len(rows))])
        resume.unlink(missing_ok=True)

    return run


def _judge(tmp_path, monkeypatch, verdicts, calls, **options):
    gen = tmp_path / "generation" / "output-rs0.jsonl"
    nsb._write_jsonl(gen, [{"generation": f"g{i}", "category": "hard_prompt"} for i in range(len(verdicts))])
    monkeypatch.setattr(nsb, "_run_ns", _fake_arena_judge(verdicts, calls))
    b = bench("arena-hard-v2", tmp_path, options={"judge_model": "self", **options})
    return b, b._judge_all(tmp_path / "generation", tmp_path / "judged", "http://127.0.0.1:9/v1", "g")


def test_arena_all_valid_judgements(ns, tmp_path, monkeypatch):
    _, d = _judge(tmp_path, monkeypatch, [("[[A>B]]", "[[B>A]]")] * 10, calls := [])
    assert (d["judge_invalid"], d["judge_total"], d["incomplete"]) == (0, 20, False)
    assert len(calls) == 10


def test_arena_invalid_judgements_above_threshold_fail(ns, tmp_path, monkeypatch):
    verdicts = [("[[A>B]]", "[[B>A]]")] * 9 + [("", "no verdict")]  # 2/20 invalid
    with pytest.raises(SystemExit, match="judge_invalid 2/20"):
        _judge(tmp_path, monkeypatch, verdicts, [])
    (tmp_path / "judged" / "output-rs0.jsonl").unlink()
    ok = [("[[A>B]]", "[[B>A]]")] * 99 + [("[[A>B]]", "")]  # 1/200 invalid
    assert _judge(tmp_path, monkeypatch, ok, [])[1]["judge_invalid"] == 1
    with pytest.raises(SystemExit):  # 0: any invalid judgement fails
        _judge(tmp_path, monkeypatch, ok, [], max_judge_invalid_frac="0")


def test_arena_invalid_judgements_below_threshold_counted_and_rejudged(ns, tmp_path, monkeypatch):
    verdicts = [("[[A>B]]", "[[B>A]]")] * 19 + [("[[A>B]] or [[B>A]]", "[[A=B]]")]  # conflicting: invalid
    b, d = _judge(tmp_path, monkeypatch, verdicts, calls := [])
    assert (d["judge_invalid"], d["judge_total"], d["incomplete"]) == (1, 40, True)
    judged = tmp_path / "judged" / "output-rs0.jsonl"
    assert b.compute_metrics([judged])["_all_"]["pass@1"]["invalid_scores"] == 1  # ns agrees
    # resume re-judges only the invalid row, keeping the others and their order
    verdicts[19] = ("[[A>B]]", "[[B>A]]")
    calls.clear()
    _, d = _judge(tmp_path, monkeypatch, verdicts, calls)
    assert calls == [19] and (d["judge_invalid"], d["incomplete"]) == (0, False)
    assert [r["generation"] for r in nsb._read_jsonl(judged)] == [f"g{i}" for i in range(20)]


def test_arena_budget_exhausted_stops_and_keeps_the_file(ns, tmp_path, monkeypatch):
    rows = [{"generation": "g", "category": "hard_prompt"}] * 2
    for k in range(2):
        nsb._write_jsonl(tmp_path / "generation" / f"output-rs{k}.jsonl", rows)
    judge = _fake_arena_judge([("", "")] * 2, calls := [])

    def refused(cmd, out, **kw):  # every call answered 402 by the meter, soft-failed by ns
        base = next(a.split("=", 1)[1] for a in cmd if a.startswith("++server.base_url="))
        assert httpx.post(base + "/chat/completions", json={}).status_code == 402
        judge(cmd, out, **kw)

    monkeypatch.setattr(nsb, "_run_ns", refused)
    b = bench("arena-hard-v2", tmp_path, repeats=2, options={"judge_model": "self"})
    with _Broke() as up, pytest.raises(SystemExit, match="402"):
        b._judge_all(tmp_path / "generation", tmp_path / "judged", up.base_url, "g")
    assert (tmp_path / "judged" / "output-rs0.jsonl").exists() and len(calls) == 2  # stopped before rs1
    assert not (tmp_path / "judged" / "output-rs1.jsonl").exists()


# -- phases --------------------------------------------------------------------


def test_eval_overrides_keep_only_the_evaluation():
    args = ["++prompt_config=x", "++eval_type=math", "++eval_config.timeout=5", "+eval_config.a.b=1", "++inference.top_k=-1"]
    assert nsb._eval_overrides(args) == ["++eval_type=math", "++eval_config.timeout=5", "+eval_config.a.b=1"]


def test_phase_support(ns, tmp_path):
    assert all(registry.get(bid).splittable for bid in IDS)
    assert not bench("arena-hard-v2", tmp_path).score_needs_server()
    assert bench("arena-hard-v2", tmp_path, options={"judge_model": "self"}).score_needs_server()
    assert not bench("aime25", tmp_path, options={"judge_model": "self"}).score_needs_server()  # no judge


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_aime25_gold_generate_then_score(ns, tmp_path, monkeypatch):
    """Generate leaves the files ungraded and marked; score grades them with ns's
    batch evaluation, without a model, to the same rows as --phase all."""
    kw = {"limit": 3, "repeats": 2, "workers": 3, "options": {"answers": "gold"}}
    split = tmp_path / "split"
    r = bench("aime25", split, phase="generate", **kw).run("", "")
    assert r["n"] == 3 and "value" not in r and r["statuses"] == {"stop": 6}
    gen = split / "generation" / "output-rs0.jsonl"
    assert all("symbolic_correct" not in row for row in _rows(gen))
    assert nsb._unscored(gen).exists()

    monkeypatch.setattr(nsb, "_run_ns", lambda *a, **k: pytest.fail("score phase generated"))
    monkeypatch.setattr(nsb, "_GoldServer", lambda *a, **k: pytest.fail("score phase served"))
    s = bench("aime25", split, phase="score", **kw).run("", "gold")
    assert (s["value"], s["n"], s["statuses"]) == (1.0, 3, {"stop": 6})
    assert not nsb._unscored(gen).exists()
    graded = _rows(gen)
    mtime = gen.stat().st_mtime_ns
    assert bench("aime25", split, phase="score", **kw).run("", "gold")["value"] == 1.0  # re-runnable
    assert gen.stat().st_mtime_ns == mtime  # already graded: left as is
    monkeypatch.undo()

    a = bench("aime25", tmp_path / "all", **kw).run("", "")
    assert {k: v for k, v in a.items() if k not in ("ns_metrics",)} == {k: v for k, v in s.items() if k not in ("ns_metrics",)}
    # the same rows but for timings (key order aside)
    timing = ("interleaved_eval_single_time_s", "generation_start_time", "generation_end_time", "generation_time")
    in_gen = _rows(tmp_path / "all" / "generation" / "output-rs0.jsonl")
    assert [{k: v for k, v in r.items() if k not in timing} for r in graded] == [
        {k: v for k, v in r.items() if k not in timing} for r in in_gen
    ]
    assert not nsb._unscored(tmp_path / "all" / "generation" / "output-rs0.jsonl").exists()


def test_score_without_generation(ns, tmp_path):
    kw = {"limit": 2, "repeats": 2, "options": {"answers": "gold"}}
    with pytest.raises(RuntimeError, match="no generation for input"):
        bench("aime25", tmp_path, phase="score", **kw).run("", "gold")
    bench("aime25", tmp_path, phase="generate", **kw).run("", "")
    (tmp_path / "generation" / "output-rs1.jsonl").unlink()
    with pytest.raises(RuntimeError, match="no generation for repeat 1"):
        bench("aime25", tmp_path, phase="score", **kw).run("", "gold")
    with pytest.raises(SystemExit, match="differs from what was generated"):
        bench("aime25", tmp_path, phase="score", **{**kw, "limit": 1}).run("", "gold")
    assert (tmp_path / "generation" / "output-rs0.jsonl").exists()  # score never discards


def test_phase_all_grades_a_marked_file(ns, tmp_path):
    """A file left by an interrupted generate phase is graded by --phase all too."""
    kw = {"limit": 2, "repeats": 1, "options": {"answers": "gold"}}
    bench("aime25", tmp_path, phase="generate", **kw).run("", "")
    assert bench("aime25", tmp_path, **kw).run("", "")["value"] == 1.0
    assert not nsb._unscored(tmp_path / "generation" / "output-rs0.jsonl").exists()


def test_arena_generate_needs_no_judge(ns, tmp_path, monkeypatch):
    prepared = tmp_path / "prepared.jsonl"
    nsb._write_jsonl(prepared, [{"question": f"q{i}", "category": "hard_prompt"} for i in range(3)])
    cmds = []

    def fake_generate(cmd, out, *, log_path, what):
        cmds.append(cmd)
        args = dict(a[2:].split("=", 1) for a in cmd if a.startswith("++"))
        nsb._write_jsonl(out, [{**r, "generation": "g"} for r in nsb._read_jsonl(nsb.Path(args["input_file"]))])

    monkeypatch.setattr(nsb, "_run_ns", fake_generate)
    monkeypatch.setattr(nsb.NemoSkillsBenchmark, "_judge_all", lambda *a, **k: pytest.fail("judged in generate"))
    monkeypatch.setattr(nsb, "_MeteringProxy", lambda *a, **k: pytest.fail("judge endpoint in generate"))
    b = bench("arena-hard-v2", tmp_path, phase="generate")
    monkeypatch.setattr(b, "prepare_data", lambda: (prepared, {"sha256": "x"}))
    r = b.run("http://127.0.0.1:9/v1", "m")
    assert r["n"] == 3 and "value" not in r
    assert len(cmds) == 1 and cmds[0][-1] == "++eval_type=null"
    assert len(_rows(tmp_path / "generation" / "output-rs0.jsonl")) == 3
