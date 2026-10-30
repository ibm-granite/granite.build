"""ProfBench / GDPval adapter tests with fake judges, a fake policy model and a
fake sandbox (no GPU, no network, no containers).

The ProfBench scoring tests run upstream's own scripts; they need the pinned
files locally (the judged image has them in /opt/profbench/<commit>, a checkout
can point GRANITE_EVALS_PROFBENCH_DIR at them) and are skipped otherwise.
"""

import asyncio
import io
import json
import logging
import shutil
import types
from pathlib import Path

import pytest

pytest.importorskip("openai")

from granite_evals import data, meter  # noqa: E402
from granite_evals.benchmarks import judge_general as jg  # noqa: E402
from granite_evals.registry import RunConfig, get  # noqa: E402

# -- fakes ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _placeholder_judge_endpoint(monkeypatch):
    """The module has no built-in judge URL (sites pass judge_base_url); tests that take the default path need one."""
    monkeypatch.setattr(jg, "DEFAULT_JUDGE_BASE_URL", "https://judge.example.test/v1")



class Usage:
    def __init__(self, prompt=100, completion=5, cached=0, cache_write=0):
        self.prompt_tokens, self.completion_tokens = prompt, completion
        self._d = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "prompt_tokens_details": {"cached_tokens": cached},
            "cache_creation_input_tokens": cache_write,
        }

    def model_dump(self):
        return self._d


def completion(content, usage=None):
    msg = types.SimpleNamespace(content=content, reasoning_content=None)
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=msg, finish_reason="stop")], usage=usage or Usage()
    )


class FakeJudgeClient:
    """Answers with ``answer(request) -> str``; records every request."""

    def __init__(self, answer, reject=()):
        self.answer, self.reject, self.requests = answer, set(reject), []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        bad = self.reject & set(request)
        if bad:
            err = Exception(f"Error code: 400 - {sorted(bad)[0]} is not supported with this model")
            err.status_code = 400
            raise err
        return completion(self.answer(request), Usage(prompt=1000, completion=10, cached=900))


def user_text(request):
    content = request["messages"][-1]["content"]
    return content if isinstance(content, str) else "".join(b["text"] for b in content)


def fake_judge(answer, **kw):
    return jg.Judge(base_url="http://judge/v1", model="judge-m", api_key="k", is_self=False,
                    client=FakeJudgeClient(answer, kw.pop("reject", ())), **kw)


# -- judge client -------------------------------------------------------------


def test_prompt_caching_splits_the_shared_response_prefix():
    j = fake_judge(lambda r: "Yes", cache_split=jg._pb_cache_split)
    prompt = "Response:\n\nREPORT" + jg._PB_SPLIT + "criterion X. Only answer Yes or No."
    j.chat.completions.create(model="ignored", messages=[{"role": "user", "content": prompt}])
    req = j._client.requests[0]
    assert req["model"] == "judge-m"
    blocks = req["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "Response:\n\nREPORT", "cache_control": {"type": "ephemeral"}}
    assert "".join(b["text"] for b in blocks) == prompt
    assert j.last_usage() == {"prompt_tokens": 1000, "completion_tokens": 10, "cached_tokens": 900, "cache_write_tokens": 0,
                            "reasoning_tokens": 0}


def test_self_judge_sends_plain_prompts():
    j = jg.Judge(base_url="http://x/v1", model="served", api_key="EMPTY", is_self=True,
                 cache_split=jg._pb_cache_split, client=FakeJudgeClient(lambda r: "No"))
    j.chat.completions.create(model="m", messages=[{"role": "user", "content": "Response:\n\nR" + jg._PB_SPLIT + "c"}])
    assert isinstance(j._client.requests[0]["messages"][0]["content"], str)
    assert j.describe()["judge_is_self"] is True


def test_rejected_parameters_are_dropped_once_and_recorded():
    j = fake_judge(lambda r: "Yes", reject={"top_p"})
    for _ in range(2):
        j.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}], temperature=0.6, top_p=0.95)
    sent = j._client.requests
    assert "top_p" in sent[0] and all("top_p" not in r for r in sent[1:])
    assert len(sent) == 3  # one rejected, then one per call
    assert j.dropped == ["top_p"] and "top_p" in j.adaptations[0]


def test_a_400_that_names_no_parameter_is_raised_with_the_request_intact():
    class TooLong(FakeJudgeClient):
        def create(self, **request):
            self.requests.append(request)
            err = Exception("Error code: 400 - prompt is too long: 250000 tokens > 200000 maximum")
            err.status_code = 400
            raise err

    j = jg.Judge(base_url="http://judge/v1", model="judge-m", api_key="k", is_self=False,
                 client=TooLong(lambda r: "Yes"), extra_body={"thinking": {"type": "adaptive"}})
    with pytest.raises(Exception, match="prompt is too long"):
        j.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}],
                                  temperature=0.6, top_p=0.95, reasoning_effort="high")
    (sent,) = j._client.requests  # no retry with a parameter dropped
    assert (sent["temperature"], sent["top_p"], sent["reasoning_effort"]) == (0.6, 0.95, "high")
    assert sent["extra_body"] == {"thinking": {"type": "adaptive"}}
    assert j.dropped == [] and j.adaptations == []


def test_usage_totals_and_cost():
    recs = [jg.usage_record(Usage(1000, 10, cached=900)), jg.usage_record(Usage(1000, 10, cache_write=1000))]
    t = jg.total_usage(recs, n_examples=2)
    assert t["prompt_tokens"] == 2000 and t["cached_tokens"] == 900 and t["calls"] == 2
    # 100 uncached + 900 cached*0.1 + 1000 written*1.25, at $2/M in; 20 out at $10/M
    assert t["estimated_cost_usd"] == pytest.approx((100 * 2 + 900 * 0.2 + 1000 * 2.5 + 20 * 10) / 1e6, abs=1e-4)
    assert t["per_example"]["prompt_tokens"] == 1000
    anthropic_style = jg.usage_record({"prompt_tokens": 5, "completion_tokens": 1, "cache_read_input_tokens": 3})
    assert anthropic_style["cached_tokens"] == 3


def test_default_judge_goes_through_the_meter(tmp_path, monkeypatch):
    monkeypatch.setattr(jg, "DEFAULT_JUDGE_BASE_URL", "https://judge.example.test/v1")  # no built-in endpoint: use a placeholder
    bench = jg.ProfBench(RunConfig(model="m", output_dir=tmp_path))
    monkeypatch.setenv("GRANITE_EVALS_JUDGE_API_KEY", "k")
    seen = []
    monkeypatch.setattr(meter, "metered", lambda url, role: seen.append((url, role)) or "http://127.0.0.1:1/v1")
    j = bench.make_judge("http://policy/v1", "served")
    assert seen == [(jg.DEFAULT_JUDGE_BASE_URL, "judge")]
    assert j.base_url == "http://127.0.0.1:1/v1" and j.model == jg.DEFAULT_JUDGE_MODEL
    assert str(j._client.base_url).startswith("http://127.0.0.1:1/v1")


def test_default_judge_url_is_proxied_by_an_active_meter(tmp_path, monkeypatch):
    monkeypatch.setattr(jg, "DEFAULT_JUDGE_BASE_URL", "https://judge.example.test/v1")  # no built-in endpoint: use a placeholder
    monkeypatch.setenv("GRANITE_EVALS_JUDGE_API_KEY", "k")
    monkeypatch.delenv("GRANITE_EVALS_SPEND_LEDGER", raising=False)
    bench = jg.GDPval(RunConfig(model="m", output_dir=tmp_path))
    with meter.Meters(bench.config.options, {"benchmark": "gdpval"}) as meters:
        j = bench.make_judge("http://policy/v1", "served")
        assert j.base_url.startswith("http://127.0.0.1:")
        assert (jg.DEFAULT_JUDGE_BASE_URL.rstrip("/"), "judge") in meters._meters


def test_judge_base_url_option_is_not_metered_twice(tmp_path, monkeypatch):
    monkeypatch.setenv("GRANITE_EVALS_JUDGE_API_KEY", "k")
    monkeypatch.setattr(meter, "metered", lambda *a, **k: pytest.fail("the CLI meters the option already"))
    opts = {"judge_base_url": "http://127.0.0.1:9/v1", "judge_model": "claude-haiku-4-5-20251001"}
    j = jg.ProfBench(RunConfig(model="m", output_dir=tmp_path, options=opts)).make_judge("http://p/v1", "s")
    assert j.base_url == "http://127.0.0.1:9/v1" and j.model == "claude-haiku-4-5-20251001"


def test_real_judge_needs_its_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GRANITE_EVALS_JUDGE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="GRANITE_EVALS_JUDGE_API_KEY"):
        jg.ProfBench(RunConfig(model="m", output_dir=tmp_path)).make_judge("http://p/v1", "s")
    self_judge = jg.ProfBench(RunConfig(model="m", output_dir=tmp_path, options={"judge_model": "self"}))
    j = self_judge.make_judge("http://p/v1", "served")
    assert j.is_self and j.model == "served" and j.base_url == "http://p/v1"


def test_judge_thinking_options_go_into_every_request_body(tmp_path, monkeypatch):
    monkeypatch.setenv("GRANITE_EVALS_JUDGE_API_KEY", "k")
    monkeypatch.setattr(meter, "metered", lambda url, role: "http://127.0.0.1:1/v1")
    opts = {"judge_thinking": "adaptive", "judge_effort": "max", "judge_extra_body": '{"x": 1}'}
    bench = jg.ProfBench(RunConfig(model="m", output_dir=tmp_path, options=opts))
    assert bench.judge_extra_body() == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "max"}, "x": 1}
    budget = jg.ProfBench(RunConfig(model="m", output_dir=tmp_path, options={"judge_thinking": "enabled:4096"}))
    assert budget.judge_extra_body() == {"thinking": {"type": "enabled", "budget_tokens": 4096}}
    with pytest.raises(SystemExit, match="judge_thinking"):
        jg.ProfBench(RunConfig(model="m", output_dir=tmp_path, options={"judge_thinking": "max"})).judge_extra_body()
    assert bench.make_judge("http://p/v1", "s").extra_body["output_config"] == {"effort": "max"}
    j = fake_judge(lambda r: "Yes", extra_body={"thinking": {"type": "adaptive"}}, reject={"temperature"})
    j.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}], temperature=0.6)
    assert all(r["extra_body"] == {"thinking": {"type": "adaptive"}} for r in j._client.requests)
    assert j.dropped == ["temperature"] and j.describe()["judge_extra_body"] == {"thinking": {"type": "adaptive"}}
    assert jg.usage_record({"completion_tokens": 900, "completion_tokens_details": {"reasoning_tokens": 850}})[
        "reasoning_tokens"] == 850


def test_registered_ids_metrics_and_pins():
    assert get("profbench").metric == "overall" and get("gdpval").metric == "Elo"
    for cls in (jg.ProfBench, jg.GDPval):
        assert cls.extra == "judged" and len(cls.dataset_revision) == 40


# -- ProfBench ------------------------------------------------------------------


def _profbench_dir():
    import os

    for d in (os.environ.get("GRANITE_EVALS_PROFBENCH_DIR", ""), f"/opt/profbench/{jg.PROFBENCH_COMMIT}",
              str(Path.home() / ".cache/granite/profbench" / jg.PROFBENCH_COMMIT)):
        if d and all((Path(d) / n).exists() for n in jg.PROFBENCH_FILES):
            return d
    return None


needs_upstream = pytest.mark.skipif(_profbench_dir() is None, reason="ProfBench scripts not available locally")


def pb_task(task_id, domain, crits):
    return {
        "task_id": task_id,
        "domain": domain,
        "prompt": f"Write the {task_id} report.",
        "o3_response": f"o3 report for {task_id}",
        "grok4_response": "g",
        "r1-0528_response": "r",
        "rubrics": [
            {"criterion_description": d, "criterion_weight": w, "criterion_type": t,
             "o3_fulfilment": f, "grok4_fulfilment": False, "r1-0528_fulfilment": False}
            for d, w, t, f in crits
        ],
    }


PB_ROWS = [
    pb_task("Fin-0", "Finance MBA", [("mentions GOOD", "Critical", ["Reasoning"], True),
                                      ("mentions BAD", "Minor", ["Style"], False)]),
    pb_task("Phys-0", "Physics PhD", [("mentions GOOD", "Major", ["Extraction (recall)"], True)]),
]


@pytest.fixture
def pb_env(tmp_path, monkeypatch):
    d = _profbench_dir()
    if d:
        monkeypatch.setenv("GRANITE_EVALS_PROFBENCH_DIR", d)
    monkeypatch.setattr(data, "load_split", lambda *a, **k: [dict(r) for r in PB_ROWS])
    policy = []

    class FakeOpenAI:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        def create(self, **request):
            policy.append(request)
            return completion("My report says GOOD things.", Usage(50, 200))

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    return types.SimpleNamespace(policy=policy)


def rubric_judge(request):
    """Yes iff the criterion's keyword is in the response."""
    text = user_text(request)
    response, criterion = text.split(jg._PB_SPLIT)
    keyword = criterion.split("mentions ")[1].split(".")[0]
    return "Yes" if keyword in response else "No"


def make_pb(tmp_path, monkeypatch, judge, limit=None, **options):
    bench = jg.ProfBench(RunConfig(model="m", output_dir=tmp_path, options=options, workers=2, limit=limit))
    monkeypatch.setattr(bench, "make_judge", lambda *a, **k: judge)
    return bench


@needs_upstream
def test_profbench_scores_with_the_upstream_scorer(tmp_path, monkeypatch, pb_env):
    judge = fake_judge(rubric_judge, cache_split=jg._pb_cache_split)
    out = make_pb(tmp_path, monkeypatch, judge).run("http://policy/v1", "served")
    # lite: Fin-0 x4, Phys-0 x4; every response is "GOOD": Fin 4/(4+2), Phys 3/3.
    assert out["n"] == 8 and len(pb_env.policy) == 8
    assert out["scores"]["Finance MBA"] == pytest.approx(66.7)
    assert out["scores"]["Physics PhD"] == 100.0
    assert out["value"] == pytest.approx((0.667 + 1.0) / 2, abs=1e-3)
    assert out["pass_at_k"]["k"] == 1 and out["pass_at_k"]["pass_at_k"] == out["pass_at_k"]["pass_at_1"] == out["value"]
    assert out["judge_usage"]["calls"] == 12 and out["judge_usage"]["cached_tokens"] == 12 * 900
    # Upstream's mixed reasoning: high for Physics PhD or Style criteria.
    efforts = {(user_text(r).rsplit(": ", 1)[1][:13], r.get("reasoning_effort")) for r in judge._client.requests}
    assert ("mentions GOOD", "high") in efforts and ("mentions GOOD", "low") in efforts
    assert out["statuses"] == {"ok": 8}
    assert pb_env.policy[0]["messages"] == [{"role": "user", "content": "Write the Fin-0 report."}]
    assert pb_env.policy[0]["max_tokens"] == 64000 and "temperature" not in pb_env.policy[0]


@needs_upstream
def test_profbench_resumes_and_limits(tmp_path, monkeypatch, pb_env):
    make_pb(tmp_path, monkeypatch, fake_judge(rubric_judge), limit=None).run("http://p/v1", "s")
    judge = fake_judge(lambda r: pytest.fail("judged again"))
    pb_env.policy.clear()
    bench = jg.ProfBench(RunConfig(model="m", output_dir=tmp_path, limit=1))
    monkeypatch.setattr(bench, "make_judge", lambda *a, **k: judge)
    out = bench.run("http://p/v1", "s")
    assert pb_env.policy == [] and out["tasks"] == ["Fin-0"] and out["n"] == 4


@needs_upstream
def test_profbench_failed_criteria_are_retried_on_resume(tmp_path, monkeypatch, pb_env):
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("gateway hiccup")
        return rubric_judge(request)

    with pytest.raises(SystemExit, match="criteria_failed 1/2"):  # above the default 0.05
        make_pb(tmp_path, monkeypatch, fake_judge(flaky), limit=1, samples="1", judge_parallel=1).run("http://p/v1", "s")
    calls["n"] = 0
    (tmp_path / "samples" / "Fin-0" / "0" / "judgments.json").unlink()
    out = make_pb(tmp_path, monkeypatch, fake_judge(flaky), limit=1, samples="1", judge_parallel=1,
                  max_failed_frac="0.5").run("http://p/v1", "s")
    assert out["statuses"] == {"judge_incomplete": 1} and out["criteria_judged"] == 1
    assert (out["criteria_failed"], out["criteria_total"], out["incomplete"], out["n"]) == (1, 2, True, 1)
    out = make_pb(tmp_path, monkeypatch, fake_judge(rubric_judge), limit=1, samples="1").run("http://p/v1", "s")
    assert out["statuses"] == {"ok": 1} and out["criteria_judged"] == 2
    assert (out["criteria_failed"], out["criteria_total"], out["incomplete"]) == (0, 2, False)


@needs_upstream
def test_profbench_failed_generation_is_dropped_not_zero(tmp_path, monkeypatch, pb_env):
    import openai

    class OnePhysFails:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        def create(self, **request):
            if "Phys-0" in request["messages"][0]["content"] and request["seed"] == 2:
                raise ConnectionError("policy server gone")
            return completion("My report says GOOD things.", Usage(50, 200))

    monkeypatch.setattr(openai, "OpenAI", OnePhysFails)
    with pytest.raises(SystemExit, match="criteria_failed 1/12"):
        make_pb(tmp_path, monkeypatch, fake_judge(rubric_judge)).run("http://p/v1", "s")
    out = make_pb(tmp_path, monkeypatch, fake_judge(rubric_judge), max_failed_frac="0.1").run("http://p/v1", "s")
    # Phys-0's three generated reports all score 100; the failed one is left out, not a 0
    assert out["scores"]["Physics PhD"] == 100.0 and out["n"] == 7
    assert (out["criteria_failed"], out["criteria_total"], out["incomplete"]) == (1, 12, True)
    assert out["statuses"]["generation_failed"] == 1


@needs_upstream
def test_profbench_human_gold_needs_no_model(tmp_path, monkeypatch, pb_env):
    bench = jg.ProfBench(RunConfig(model="none", output_dir=tmp_path,
                                   options={"responses": "o3", "judge_model": "human"}))
    assert not bench.needs_server()
    out = bench.run("", "none")
    assert pb_env.policy == [] and out["n"] == 2 and out["judge_model"] == "human"
    assert out["scores"]["Finance MBA"] == pytest.approx(66.7) and out["scores"]["Physics PhD"] == 100.0


@needs_upstream
def test_profbench_judge_agreement_on_provided_reports(tmp_path, monkeypatch, pb_env):
    # A judge that says Yes to everything disagrees with the human "No" on Fin-0's 2nd criterion.
    out = make_pb(tmp_path, monkeypatch, fake_judge(lambda r: "Yes"), responses="o3").run("http://p/v1", "s")
    assert out["judge_agreement"]["mean_pred_minus_human"] == pytest.approx(33.3)
    assert 0 < out["judge_agreement"]["macro_f1"] < 100


@needs_upstream
def test_profbench_empty_response_scores_zero_without_judging(tmp_path, monkeypatch, pb_env):
    import openai

    class Empty:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(
                create=lambda **r: completion("", Usage(5, 32000))))

    monkeypatch.setattr(openai, "OpenAI", Empty)
    judge = fake_judge(lambda r: pytest.fail("nothing to judge"))
    out = make_pb(tmp_path, monkeypatch, judge, version="debug", limit=1).run("http://p/v1", "s")
    assert out["value"] == 0.0 and out["statuses"] == {"empty_response": 1}


@needs_upstream
def test_profbench_max_effort_judge_verdict_is_the_content_only(tmp_path, monkeypatch, pb_env):
    """judge_reasoning_effort=max on every criterion, thinking fields in the body,
    temperature dropped when rejected; the thinking text never becomes the verdict."""

    class Thinking(FakeJudgeClient):
        def create(self, **request):
            self.requests.append(request)
            if "temperature" in request:
                err = Exception("Error code: 400 - temperature may only be set to 1 when thinking is enabled")
                err.status_code = 400
                raise err
            c = completion("No")
            c.choices[0].message.reasoning_content = "Yes. The criterion is clearly met, so: Yes"
            c.usage = Usage(1000, 900)
            c.usage._d["completion_tokens_details"] = {"reasoning_tokens": 850}
            return c

    # A default run first: its ratings stay apart from the tagged re-judging.
    make_pb(tmp_path, monkeypatch, fake_judge(rubric_judge), limit=1, samples="1").run("http://p/v1", "s")
    client = Thinking(None)
    judge = jg.Judge(base_url="http://judge/v1", model="judge-m", api_key="k", is_self=False, client=client,
                     extra_body={"thinking": {"type": "adaptive"}, "output_config": {"effort": "max"}})
    out = make_pb(tmp_path, monkeypatch, judge, limit=1, samples="1", judge_reasoning_effort="max",
                  judge_tag="max", judge_parallel=1).run("http://p/v1", "s")
    sent = [r for r in client.requests if "temperature" not in r]
    assert len(sent) == 2 and all(r["reasoning_effort"] == "max" for r in client.requests)
    assert all(r["extra_body"]["output_config"] == {"effort": "max"} and "top_p" in r for r in sent)
    assert judge.dropped == ["temperature"]
    assert out["value"] == 0.0 and out["judge_unparsed_ratings"] == 0  # every verdict "No", from content
    assert out["judge_reasoning_effort"] == "max" and "reasoning_effort max for every criterion" in out["judge_request"]
    assert out["judge_usage"]["reasoning_tokens"] == 2 * 850 and out["judge_tag"] == "max"
    sdir = tmp_path / "samples" / "Fin-0" / "0"
    assert json.loads((sdir / "judgments.json").read_text())["0"]["judge_rating"] == "Yes"
    assert json.loads((sdir / "judgments-max.json").read_text())["0"]["judge_rating"] == "No"
    probe = make_pb(tmp_path, monkeypatch, fake_judge(lambda r: "Yes"), limit=1, samples="1",
                    judge_tag="probe", judge_max_criteria="1").run("http://p/v1", "s")
    assert probe["criteria_judged"] == 1 and probe["judge_max_criteria"] == 1


def _phase(bench, phase):
    bench.config.phase = phase
    return bench


@needs_upstream
def test_profbench_generate_then_score(tmp_path, monkeypatch, pb_env):
    monkeypatch.delenv("GRANITE_EVALS_JUDGE_API_KEY", raising=False)  # generating needs no judge key
    gen = _phase(jg.ProfBench(RunConfig(model="m", output_dir=tmp_path, workers=2)), "generate")
    assert gen.needs_server() and not gen.score_needs_server()
    out = gen.run("http://p/v1", "s")
    assert (out["n"], out["generations_failed"], out["statuses"]) == (8, 0, {"ok": 8}) and "value" not in out
    assert len(pb_env.policy) == 8 and not list(tmp_path.glob("samples/*/*/judgments*.json"))
    pb_env.policy.clear()
    judge = fake_judge(rubric_judge)
    scored = _phase(make_pb(tmp_path, monkeypatch, judge), "score").run("", "s")
    assert pb_env.policy == [] and scored["n"] == 8 and scored["judge_usage"]["calls"] == 12
    # the same reports judged by --phase all give the same result
    whole = make_pb(tmp_path / "all", monkeypatch, fake_judge(rubric_judge)).run("http://p/v1", "s")
    assert whole["value"] == scored["value"] and whole["scores"] == scored["scores"]
    # score again: resumes, nothing is judged twice
    again = _phase(make_pb(tmp_path, monkeypatch, fake_judge(lambda r: pytest.fail("judged again"))), "score")
    assert again.run("", "s")["value"] == scored["value"]


@needs_upstream
def test_profbench_score_without_a_generation_is_a_failed_sample(tmp_path, monkeypatch, pb_env):
    _phase(jg.ProfBench(RunConfig(model="m", output_dir=tmp_path, options={"samples": "1"})), "generate").run("http://p/v1", "s")
    (tmp_path / "samples" / "Phys-0" / "0" / "response.json").unlink()
    pb_env.policy.clear()
    bench = _phase(make_pb(tmp_path, monkeypatch, fake_judge(rubric_judge), samples="1"), "score")
    with pytest.raises(SystemExit, match="criteria_failed 1/3"):
        bench.run("", "s")
    out = _phase(make_pb(tmp_path, monkeypatch, fake_judge(rubric_judge), samples="1", max_failed_frac="0.5"),
                 "score").run("", "s")
    assert pb_env.policy == [] and out["n"] == 1 and out["statuses"] == {"ok": 1, "not_generated": 1}
    assert not (tmp_path / "samples" / "Phys-0" / "0" / "response.json").exists()  # never generated in score


@needs_upstream
def test_profbench_human_gold_in_both_phases(tmp_path, monkeypatch, pb_env):
    opts = {"responses": "o3", "judge_model": "human"}
    gen = _phase(jg.ProfBench(RunConfig(model="none", output_dir=tmp_path, options=opts)), "generate")
    assert not gen.needs_server() and not gen.score_needs_server()
    assert gen.run("", "none")["statuses"] == {"dataset": 2}
    out = _phase(jg.ProfBench(RunConfig(model="none", output_dir=tmp_path, options=opts)), "score").run("", "none")
    assert pb_env.policy == [] and out["n"] == 2 and out["scores"]["Physics PhD"] == 100.0


def test_profbench_self_judge_cannot_score_alone(tmp_path):
    bench = jg.ProfBench(RunConfig(model="m", output_dir=tmp_path, options={"judge_model": "self"}))
    assert bench.score_needs_server() and bench.splittable


# -- GDPval ---------------------------------------------------------------------


def test_verdict_parsing():
    assert jg.parse_verdict("A is better.\nVERDICT: B") == "B"
    assert jg.parse_verdict("VERDICT: A ... on reflection\n**VERDICT: TIE**") == "TIE"
    assert jg.parse_verdict("verdict - a") == "A"
    assert jg.parse_verdict("no decision") is None and jg.parse_verdict(None) is None


def test_elo_style_score():
    assert jg.elo_style(5, 10, 1000) == pytest.approx(1000)
    assert jg.elo_style(10, 10, 1000) > 1000 > jg.elo_style(0, 10, 1000)
    assert jg.elo_style(0, 10, 1000) - 1000 == pytest.approx(1000 - jg.elo_style(10, 10, 1000))
    assert jg.elo_style(0, 10, 1000) == pytest.approx(1000 + 400 * __import__("math").log10(0.5 / 10.5))


def _docx_bytes(text):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph(text)
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "k", "v"
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_render_office_files():
    openpyxl = pytest.importorskip("openpyxl")
    pptx = pytest.importorskip("pptx")
    assert "Quarterly memo" in jg.render_file("m.docx", _docx_bytes("Quarterly memo"))
    assert "k | v" in jg.render_file("m.docx", _docx_bytes("x"))

    wb = openpyxl.Workbook()
    wb.active["A1"], wb.active["A2"] = 3, "=A1*2"  # openpyxl stores no cached value
    buf = io.BytesIO()
    wb.save(buf)
    text = jg.render_file("b.xlsx", buf.getvalue())
    assert "3" in text and "=A1*2" in text

    p = pptx.Presentation()
    s = p.slides.add_slide(p.slide_layouts[1])
    s.shapes.title.text = "Revenue plan"
    buf = io.BytesIO()
    p.save(buf)
    assert "Revenue plan" in jg.render_file("d.pptx", buf.getvalue())

    zbuf = io.BytesIO()
    import zipfile

    with zipfile.ZipFile(zbuf, "w") as z:
        z.writestr("inner/notes.txt", "hello zip")
    assert "hello zip" in jg.render_file("a.zip", zbuf.getvalue())
    assert "not shown" in jg.render_file("x.png", b"\x89PNG")
    assert "could not be read" in jg.render_file("bad.docx", b"junk")
    assert "truncated" in jg.render_file("t.txt", b"x" * 100, limit=10)


def test_render_pdf():
    pypdf = pytest.importorskip("pypdf")
    w = pypdf.PdfWriter()
    w.add_blank_page(72, 72)
    buf = io.BytesIO()
    w.write(buf)
    assert "page 1" in jg.render_file("r.pdf", buf.getvalue())


# A sandbox that runs commands in a local directory (the provider's path handling,
# file transfer and env scrubbing; no container).
class LocalDirSandbox:
    def __init__(self, root):
        self.root, self.started, self.closed, self.commands = root, False, False, []

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def _run_env(self):
        return {"PATH": "/usr/bin:/bin", "GRANITE_EVALS_JUDGE_API_KEY": "secret", "HF_TOKEN": "t", "HOME": "/h"}

    def execute(self, command, *, cwd="/", timeout=None, env=None, stdin=None):
        import subprocess

        from granite_evals.sandbox import ExecResult

        self.commands.append(command)
        command = command.replace("/workspace", str(self.root))
        cwd = str(self.root) if cwd.startswith("/workspace") else cwd
        p = subprocess.run(["bash", "-c", f"cd {cwd} && {command}"], input=stdin, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return ExecResult(p.stdout, p.returncode)


def test_sandbox_provider_file_transfer_and_env(tmp_path):
    pytest.importorskip("stirrup")
    sb = LocalDirSandbox(tmp_path)
    provider = jg.sandbox_provider_class()("img", setup="echo ready", factory=lambda: sb)

    async def go():
        async with provider:
            assert sb.started and provider.setup_output.strip() == "ready"
            assert "GRANITE_EVALS_JUDGE_API_KEY" not in sb._run_env() and "HF_TOKEN" not in sb._run_env()
            assert sb._run_env()["HOME"] == "/h"
            blob = bytes(range(256)) * 10
            await provider.write_file_bytes("out/data.bin", blob)
            assert await provider.read_file_bytes("/workspace/out/data.bin") == blob
            assert await provider.file_exists("out/data.bin")
            assert not await provider.file_exists("nope.txt")
            assert await provider.is_directory("out")
            assert await provider.list_files(".") == ["out/data.bin"]
            with pytest.raises(FileNotFoundError):
                await provider.read_file_bytes("missing")
            if shutil.which("timeout"):
                r = await provider.run_command("echo hi && exit 3")
                assert r.exit_code == 3 and "hi" in r.stdout
        assert sb.closed

    asyncio.run(go())
    assert any(c.startswith("timeout --kill-after=5s") for c in sb.commands) or not shutil.which("timeout")


GDP_TASKS = [
    {"task_id": "t1", "sector": "Finance", "occupation": "Analyst", "prompt": "Write memo.docx.",
     "reference_files": ["reference_files/h1/input.txt"], "deliverable_files": ["deliverable_files/h1/memo.docx"],
     "rubric_pretty": "+2 mentions revenue"},
    {"task_id": "t2", "sector": "Health", "occupation": "Nurse", "prompt": "Write a plan.",
     "reference_files": [], "deliverable_files": [], "rubric_pretty": "x"},
    {"task_id": "t3", "sector": "Media", "occupation": "Editor", "prompt": "Make a video.",
     "reference_files": [], "deliverable_files": ["deliverable_files/h3/clip.mp4"], "rubric_pretty": "y"},
]


@pytest.fixture
def gdp_env(tmp_path, monkeypatch):
    ds = tmp_path / "dataset"
    for rel, content in (("reference_files/h1/input.txt", b"Q3 revenue 12M"),
                         ("deliverable_files/h1/memo.docx", _docx_bytes("Expert memo: revenue 12M")),
                         ("deliverable_files/h3/clip.mp4", b"\x00\x00")):
        (ds / rel).parent.mkdir(parents=True, exist_ok=True)
        (ds / rel).write_bytes(content)
    monkeypatch.setattr(data, "load_split", lambda *a, **k: [dict(t) for t in GDP_TASKS])
    return ds


def model_prefers(side):
    """A judge that picks the submission whose text contains ``side``."""

    def answer(request):
        text = user_text(request)
        a = text.split("=== SUBMISSION A ===")[1].split("=== SUBMISSION B ===")[0]
        return "VERDICT: A" if side in a else "VERDICT: B"

    return answer


def make_gdp(tmp_path, monkeypatch, gdp_env, judge, **options):
    bench = jg.GDPval(RunConfig(model="m", output_dir=tmp_path / "run", dataset=str(gdp_env),
                                options=options, workers=2))
    monkeypatch.setattr(bench, "make_judge", lambda *a, **k: judge)
    return bench


def test_gdpval_expert_vs_expert_is_even_and_flagged(tmp_path, monkeypatch, gdp_env, caplog):
    judge = fake_judge(lambda r: "Both equal.\nVERDICT: TIE", cache_split=jg._gdp_cache_split)
    bench = make_gdp(tmp_path, monkeypatch, gdp_env, judge, deliverables="expert")
    assert not bench.needs_server()
    with caplog.at_level(logging.WARNING):
        out = bench.run("", "none")
    assert out["value"] == pytest.approx(1290) and out["win_rate"] == 0.5 and out["n"] == 1
    assert out["elo_is_approximation"] is True and "APPROXIMATION" in out["metric_note"]
    assert "published GDPval" in out["metric_note"]
    assert out["pass_at_k"] == {"k": 1, "pass_at_1": 0.5, "pass_at_k": None, "n": 1,
                                "how": "pass_at_1 is the win rate the Elo comes from; no pass@k for an Elo"}
    assert "APPROXIMATION" in caplog.text
    assert out["statuses"] == {"judged": 1, "excluded:no_expert_deliverable": 1,
                               "excluded:expert_deliverable_not_text": 1}
    assert out["judge_usage"]["calls"] == 2
    # Both orders share the task+rubric prefix, which is marked for caching.
    prefixes = [r["messages"][0]["content"][0]["text"] for r in judge._client.requests]
    assert prefixes[0] == prefixes[1] and "+2 mentions revenue" in prefixes[0]
    assert "Expert memo: revenue 12M" in user_text(judge._client.requests[0])


def test_gdpval_model_run_judges_both_orders_and_resumes(tmp_path, monkeypatch, gdp_env):
    agents = []

    async def fake_agent(self, task, refs, out_dir, base_url, served):
        agents.append((task["task_id"], [p.name for p in refs]))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "memo.docx").write_bytes(_docx_bytes("MODEL memo"))
        return {"finished": True, "note": "Done.", "paths": ["memo.docx"], "turns": 3,
                "token_usage": {"input": 10, "output": 5, "reasoning": 1}, "failed_outputs": []}

    monkeypatch.setattr(jg.GDPval, "_agent", fake_agent)
    judge = fake_judge(model_prefers("MODEL memo"))
    out = make_gdp(tmp_path, monkeypatch, gdp_env, judge).run("http://p/v1", "s")
    assert agents == [("t1", ["input.txt"])]
    assert out["win_rate"] == 1.0 and out["value"] == pytest.approx(jg.elo_style(1, 1, 1290), abs=0.1)
    assert out["position_consistency"] == 1.0 and len(judge._client.requests) == 2
    report = json.loads((tmp_path / "run/tasks/t1/report.json").read_text())
    assert report["verdicts"] == [1.0, 1.0] and report["agent"]["turns"] == 3
    # Resume: nothing is re-run or re-judged.
    again = make_gdp(tmp_path, monkeypatch, gdp_env, fake_judge(lambda r: pytest.fail("rejudged"))).run("http://p/v1", "s")
    assert again["win_rate"] == 1.0 and len(agents) == 1


def test_gdpval_no_submission_is_a_loss_without_a_judge_call(tmp_path, monkeypatch, gdp_env):
    async def gave_up(self, task, refs, out_dir, base_url, served):
        return {"finished": False, "note": None, "paths": [], "turns": 250,
                "token_usage": {"input": 1, "output": 1, "reasoning": 0}, "failed_outputs": []}

    monkeypatch.setattr(jg.GDPval, "_agent", gave_up)
    judge = fake_judge(lambda r: pytest.fail("no judge call"))
    out = make_gdp(tmp_path, monkeypatch, gdp_env, judge, tasks="t1").run("http://p/v1", "s")
    assert out["statuses"] == {"no_submission": 1} and out["win_rate"] == 0.0 and out["value"] < 1290


def test_gdpval_broken_task_does_not_sink_the_run(tmp_path, monkeypatch, gdp_env):
    async def boom(self, *a):
        raise RuntimeError("sandbox died")

    monkeypatch.setattr(jg.GDPval, "_agent", boom)
    # the other tasks are excluded, so t1 failing is 1/1 failed: the run fails, not a 0
    with pytest.raises(SystemExit, match="tasks_failed 1/1"):
        make_gdp(tmp_path, monkeypatch, gdp_env, fake_judge(lambda r: "VERDICT: A")).run("http://p/v1", "s")
    with pytest.raises(SystemExit, match="no task was graded"):
        make_gdp(tmp_path, monkeypatch, gdp_env, fake_judge(lambda r: "VERDICT: A"),
                 max_failed_frac="1").run("http://p/v1", "s")
    assert not (tmp_path / "run/tasks/t1/report.json").exists()  # retried on resume


def test_gdpval_failed_task_is_left_out_not_a_loss(tmp_path, monkeypatch, gdp_env):
    async def agent(self, task, refs, out_dir, base_url, served):
        if task["task_id"] == "t1b":
            raise RuntimeError("sandbox died")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "memo.docx").write_bytes(_docx_bytes("MODEL memo"))
        return {"finished": True, "note": "Done.", "paths": ["memo.docx"], "turns": 3,
                "token_usage": {"input": 1, "output": 1, "reasoning": 0}, "failed_outputs": []}

    tasks = GDP_TASKS + [{**GDP_TASKS[0], "task_id": "t1b"}]
    monkeypatch.setattr(data, "load_split", lambda *a, **k: [dict(t) for t in tasks])
    monkeypatch.setattr(jg.GDPval, "_agent", agent)
    out = make_gdp(tmp_path, monkeypatch, gdp_env, fake_judge(model_prefers("MODEL memo")),
                   max_failed_frac="0.5").run("http://p/v1", "s")
    assert out["win_rate"] == 1.0 and out["n"] == 1 and out["statuses"]["error:RuntimeError"] == 1
    assert (out["tasks_failed"], out["tasks_total"], out["incomplete"]) == (1, 2, True)


def test_gdpval_judge_invalid_is_a_failure_and_rejudged_on_resume(tmp_path, monkeypatch, gdp_env):
    agents = []

    async def fake_agent(self, task, refs, out_dir, base_url, served):
        agents.append(task["task_id"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "memo.docx").write_bytes(_docx_bytes("MODEL memo"))
        return {"finished": True, "note": "Done.", "paths": ["memo.docx"], "turns": 3,
                "token_usage": {"input": 1, "output": 1, "reasoning": 0}, "failed_outputs": []}

    monkeypatch.setattr(jg.GDPval, "_agent", fake_agent)
    with pytest.raises(SystemExit, match="tasks_failed 1/1"):
        make_gdp(tmp_path, monkeypatch, gdp_env, fake_judge(lambda r: "no verdict here")).run("http://p/v1", "s")
    report = json.loads((tmp_path / "run/tasks/t1/report.json").read_text())
    assert report["status"] == "judge_invalid" and "score" not in report
    judge = fake_judge(model_prefers("MODEL memo"))
    out = make_gdp(tmp_path, monkeypatch, gdp_env, judge).run("http://p/v1", "s")
    # re-judged with the saved deliverables; the agent is not run again
    assert agents == ["t1"] and len(judge._client.requests) == 2
    assert out["statuses"]["judged"] == 1 and out["win_rate"] == 1.0
    assert (out["tasks_failed"], out["tasks_total"], out["incomplete"]) == (0, 1, False)


def _memo_agent(agents):
    async def fake_agent(self, task, refs, out_dir, base_url, served):
        agents.append(task["task_id"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "memo.docx").write_bytes(_docx_bytes("MODEL memo"))
        return {"finished": True, "note": "Done.", "paths": ["memo.docx"], "turns": 3,
                "token_usage": {"input": 1, "output": 1, "reasoning": 0}, "failed_outputs": []}

    return fake_agent


def test_gdpval_generate_then_score(tmp_path, monkeypatch, gdp_env):
    agents = []
    monkeypatch.setattr(jg.GDPval, "_agent", _memo_agent(agents))
    monkeypatch.delenv("GRANITE_EVALS_JUDGE_API_KEY", raising=False)  # generating needs no judge key
    gen = jg.GDPval(RunConfig(model="m", output_dir=tmp_path / "run", dataset=str(gdp_env), workers=2,
                              phase="generate"))
    assert gen.needs_server() and not gen.score_needs_server()
    out = gen.run("http://p/v1", "s")
    assert agents == ["t1"] and out["n"] == 1 and "value" not in out
    assert out["statuses"] == {"submitted": 1, "excluded:no_expert_deliverable": 1,
                               "excluded:expert_deliverable_not_text": 1}
    assert (tmp_path / "run/tasks/t1/agent.json").exists() and not list(tmp_path.glob("run/tasks/*/report.json"))
    judge = fake_judge(model_prefers("MODEL memo"))
    score = make_gdp(tmp_path, monkeypatch, gdp_env, judge)
    score.config.phase = "score"
    scored = score.run("", "s")
    assert agents == ["t1"] and len(judge._client.requests) == 2 and scored["judge_usage"]["calls"] == 2
    # the same metrics as --phase all on a fresh dir
    whole = make_gdp(tmp_path / "all", monkeypatch, gdp_env, fake_judge(model_prefers("MODEL memo"))).run("http://p/v1", "s")
    keys = ("value", "n", "win_rate", "statuses", "tasks_failed", "tasks_total", "per_sector")
    assert {k: scored[k] for k in keys} == {k: whole[k] for k in keys}
    # score again: resumes from report.json
    again = make_gdp(tmp_path, monkeypatch, gdp_env, fake_judge(lambda r: pytest.fail("rejudged")))
    again.config.phase = "score"
    assert again.run("", "s")["value"] == scored["value"]


def test_gdpval_score_without_a_generation_is_a_failure(tmp_path, monkeypatch, gdp_env):
    agents = []
    monkeypatch.setattr(jg.GDPval, "_agent", _memo_agent(agents))
    bench = make_gdp(tmp_path, monkeypatch, gdp_env, fake_judge(lambda r: pytest.fail("nothing to judge")))
    bench.config.phase = "score"
    with pytest.raises(SystemExit, match="tasks_failed 1/1"):
        bench.run("", "s")
    assert agents == [] and not (tmp_path / "run/tasks/t1/report.json").exists()  # retried on resume
    lenient = make_gdp(tmp_path, monkeypatch, gdp_env, fake_judge(lambda r: "VERDICT: A"), max_failed_frac="1")
    lenient.config.phase = "score"
    with pytest.raises(SystemExit, match="no task was graded"):
        lenient.run("", "s")
    report = lenient._task(GDP_TASKS[0], tmp_path / "run/tasks", "", "s", None, str(gdp_env), None)
    assert report["status"] == "error:not_generated" and "run --phase generate" in report["error"]


def test_gdpval_expert_gold_in_both_phases(tmp_path, monkeypatch, gdp_env):
    judge = fake_judge(lambda r: "VERDICT: TIE")
    gen = make_gdp(tmp_path, monkeypatch, gdp_env, fake_judge(lambda r: pytest.fail("no judge")), deliverables="expert")
    gen.config.phase = "generate"
    assert not gen.needs_server() and gen.run("", "none")["n"] == 1
    score = make_gdp(tmp_path, monkeypatch, gdp_env, judge, deliverables="expert")
    score.config.phase = "score"
    out = score.run("", "none")
    assert out["value"] == pytest.approx(1290) and out["n"] == 1 and len(judge._client.requests) == 2
    assert jg.GDPval(RunConfig(model="m", output_dir=tmp_path, options={"judge_model": "self"})).score_needs_server()


@pytest.mark.parametrize("cls", [jg.ProfBench, jg.GDPval])
def test_one_response_per_task(tmp_path, cls):
    with pytest.raises(SystemExit, match="use --repeats 1"):
        cls(RunConfig(model="m", output_dir=tmp_path, repeats=2)).run("", "m")
