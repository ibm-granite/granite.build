"""τ³-bench: scoring, aggregation, endpoints and metering, resume, limit.

The end-to-end tests run the real harness (tau2) on its pinned task data against
a local fake OpenAI-compatible server that plays agent, user simulator and
judge (no GPU, no network). They need the `tau` extra and the data (baked into
the image at /opt/tau2-data, or GRANITE_EVALS_TAU2_TEST_DATA / the fetch cache).
"""

import importlib.util
import json
import os
import random
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from granite_evals import meter, registry
from granite_evals.benchmarks import tau
from granite_evals.registry import RunConfig

SUITE = {
    "tau3-bench": "pass@1 (avg of 3)",
    "tau3-airline": "pass@1",
    "tau3-retail": "pass@1",
    "tau3-telecom": "pass@1",
    "tau3-banking-knowledge": "pass@1",
}


@pytest.fixture(autouse=True)
def _placeholder_gateway(monkeypatch):
    """The module has no built-in gateway URL (sites pass *_base_url); tests that take the default path need one."""
    monkeypatch.setattr(tau, "DEFAULT_GATEWAY", "https://gateway.example.test/v1")



def _bench(bid, tmp_path, **kw):
    options = kw.pop("options", {})
    return registry.get(bid)(RunConfig(model="m", output_dir=tmp_path, options=options, **kw))


# -- pure logic ----------------------------------------------------------------


def test_ids_metrics_and_pins():
    for bid, metric in SUITE.items():
        cls = registry.get(bid)
        assert cls.metric == metric
        assert cls.dataset == tau.TAU2_REPO and cls.dataset_revision == tau.TAU2_COMMIT
        assert cls.extra == "tau" and cls.default_repeats == 1
    assert registry.get("tau3-bench").domains == ("airline", "retail", "telecom")
    assert registry.get("tau3-banking-knowledge").domains == ("banking_knowledge",)


def test_trial_seeds_match_harness():
    random.seed(300)
    expected = [random.randint(0, 1000000) for _ in range(4)]
    assert tau.trial_seeds(300, 4) == expected


def test_pass_hat_k_and_summary():
    assert tau.pass_hat_k(4, 2, 1) == 0.5
    assert tau.pass_hat_k(4, 2, 2) == pytest.approx(1 / 6)
    assert tau.pass_at_k(4, 2, 1) == 0.5 and tau.pass_at_k(4, 2, 2) == pytest.approx(5 / 6) and tau.pass_at_k(4, 2, 3) == 1.0
    recs = [
        {"task_id": "a", "trial": 0, "reward": 1.0, "status": "success"},
        {"task_id": "a", "trial": 1, "reward": 0.0, "status": "fail"},
        {"task_id": "b", "trial": 0, "reward": 1.0, "status": "success"},
        {"task_id": "b", "trial": 1, "reward": 1.0, "status": "success"},
    ]
    s = tau.domain_summary(recs, 2)
    assert s["pass_hat_1"] == 0.75 and s["pass_hat_2"] == 0.5
    assert s["pass_at_1"] == 0.75 and s["pass_at_2"] == 1.0  # a and b each succeed in some trial
    assert s["per_trial_pass_1"] == [1.0, 0.5]
    assert s["statuses"] == {"fail": 1, "success": 3}
    assert (s["simulations_failed"], s["simulations_total"]) == (0, 4)


def test_failed_simulations_are_left_out_as_the_harness_does():
    recs = [
        {"task_id": "a", "trial": 0, "reward": 1.0, "status": "success"},
        {"task_id": "a", "trial": 1, "reward": 0.0, "status": "error:infrastructure"},
        {"task_id": "b", "trial": 0, "reward": 1.0, "status": "success"},
        {"task_id": "b", "trial": 1, "reward": 0.0, "status": "fail"},
        {"task_id": "c", "trial": 0, "reward": 0.0, "status": "error:ConnectionError"},
        {"task_id": "c", "trial": 1, "reward": 0.0, "status": "error:budget_exhausted"},
    ]
    s = tau.domain_summary(recs, 2)
    # a: 1/1, b: 1/2, c dropped (no trial ran); pass^2 undefined with a at one trial
    assert s["pass_hat_1"] == 0.75 and "pass_hat_2" not in s and "pass_at_2" not in s
    assert (s["n_tasks"], s["n_simulations"], s["simulations_failed"], s["simulations_total"]) == (2, 3, 3, 6)
    assert s["avg_reward"] == pytest.approx(2 / 3) and s["per_trial_pass_1"] == [1.0, 0.0]


def _run_with(tmp_path, monkeypatch, fail, **options):
    b = _bench("tau3-airline", tmp_path, repeats=2, options={"user_model": "self", **options})
    monkeypatch.setattr(b, "load_tasks", lambda: ({"airline": [_T(i) for i in range(10)]}, tau.TAU2_REPO, "c"))
    monkeypatch.setattr(tau, "install_hooks", lambda: None)

    def sim(domain, task, trial, seed, agent, user, judge):
        bad = (task.id, trial) in fail
        return {"domain": domain, "task_id": task.id, "trial": trial, "reward": 0.0 if bad else 1.0,
                "status": "error:infrastructure" if bad else "success", "usage": tau._zero_usage()}

    monkeypatch.setattr(b, "_simulation", sim)
    return b.run("http://vllm/v1", "served")


def test_failure_threshold_and_incomplete(tmp_path, monkeypatch):
    out = _run_with(tmp_path, monkeypatch, set())
    assert (out["value"], out["simulations_failed"], out["simulations_total"], out["incomplete"]) == (1.0, 0, 20, False)
    out = _run_with(tmp_path, monkeypatch, {("3", 1)}, max_failed_frac="0.1")
    # the failed trial is no loss: every scored simulation succeeded
    assert (out["value"], out["n"], out["simulations_failed"], out["incomplete"]) == (1.0, 10, 1, True)
    assert out["domains"]["airline"]["simulations_failed"] == 1
    assert _run_with(tmp_path, monkeypatch, {("3", 1)})["incomplete"]  # 1/20: at the default 0.05
    with pytest.raises(SystemExit, match="simulations_failed 2/20"):
        _run_with(tmp_path, monkeypatch, {("3", 1), ("4", 0)})
    with pytest.raises(SystemExit):
        _run_with(tmp_path, monkeypatch, {("3", 1)}, max_failed_frac="0")
    with pytest.raises(SystemExit, match="no simulation scored"):
        _run_with(tmp_path, monkeypatch, {(str(i), k) for i in range(10) for k in range(2)}, max_failed_frac="1")


def test_usage_counts_cached_tokens():
    class M:
        usage = {"prompt_tokens": 100, "completion_tokens": 10}
        raw_data = {"usage": {"prompt_tokens_details": {"cached_tokens": 60}, "cache_creation_input_tokens": 5}}

    acc = tau.empty_usage()
    tau.add_usage(acc, M())
    tau.add_usage(acc, M())
    assert acc == {"calls": 2, "prompt_tokens": 200, "completion_tokens": 20, "cached_tokens": 120,
                   "cache_creation_tokens": 10}


def test_task_filename_is_safe_and_unique():
    a = tau.task_filename("[mms_issue]a|b[PERSONA:Hard]")
    b = tau.task_filename("[mms_issue]a|b[PERSONA:None]")
    assert "/" not in a and "|" not in a and a != b


class _T:
    evaluation_criteria = None

    def __init__(self, i):
        self.id = str(i)


def test_aggregate_is_mean_of_core_domains_and_limit(tmp_path, monkeypatch):
    b = _bench("tau3-bench", tmp_path, limit=2, repeats=2, options={"user_model": "self"})
    seen = []

    def load_tasks():
        return {d: [_T(i) for i in range(2)] for d in b.domains}, tau.TAU2_REPO, tau.TAU2_COMMIT

    rewards = {"airline": 1.0, "retail": 0.0, "telecom": 1.0}

    def sim(domain, task, trial, seed, agent, user, judge):
        seen.append((domain, task.id, trial, seed))
        r = rewards[domain] if task.id == "0" else 0.0
        return {"domain": domain, "task_id": task.id, "trial": trial, "reward": r,
                "status": "success" if r else "fail", "usage": tau._zero_usage()}

    monkeypatch.setattr(b, "load_tasks", load_tasks)
    monkeypatch.setattr(b, "_simulation", sim)
    monkeypatch.setattr(tau, "install_hooks", lambda: None)
    out = b.run("http://vllm/v1", "served")
    # per domain pass^1: airline .5, retail 0, telecom .5 -> mean 1/3
    assert out["value"] == pytest.approx(1 / 3)
    how = "mean over domains; a task counts if any of its k trials has reward 1"
    pak = {"k": 2, "pass_at_1": pytest.approx(1 / 3), "pass_at_k": pytest.approx(1 / 3), "n": 6, "how": how}
    assert out["pass_at_k"] == {**pak, "pass_hat_k": pytest.approx(1 / 3)}
    assert set(out["domains"]) == {"airline", "retail", "telecom"}
    assert out["n"] == 6 and len(seen) == 12
    assert {s[3] for s in seen} == set(tau.trial_seeds(300, 2))


# -- endpoints and metering --------------------------------------------------


def test_self_endpoints_use_the_served_model(tmp_path):
    b = _bench("tau3-retail", tmp_path, options={"user_model": "self", "judge_model": "self"})
    u = b.llm_endpoint("user", "http://vllm/v1", "granite", temperature=0.0)
    assert u.model == "hosted_vllm/granite" and u.kwargs["api_base"] == "http://vllm/v1" and u.is_self


def test_paid_endpoint_needs_its_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GRANITE_EVALS_USER_API_KEY", raising=False)
    b = _bench("tau3-airline", tmp_path)
    with pytest.raises(SystemExit, match="GRANITE_EVALS_USER_API_KEY"):
        b.llm_endpoint("user", "http://vllm/v1", "granite", temperature=0.0)


def test_default_gateway_goes_through_the_meter(tmp_path, monkeypatch):
    monkeypatch.setenv("GRANITE_EVALS_USER_API_KEY", "k")
    monkeypatch.setenv("GRANITE_EVALS_JUDGE_API_KEY", "k")
    b = _bench("tau3-retail", tmp_path)
    with meter.Meters(b.config.options, {}) as meters:
        u = b.llm_endpoint("user", "http://vllm/v1", "granite", temperature=0.0)
        j = b.llm_endpoint("judge", "http://vllm/v1", "granite", temperature=0.0)
        proxy = meter.metered(tau.DEFAULT_GATEWAY, role="user")
        # Same gateway, one meter per role, so the ledger tells simulator from judge.
        assert u.kwargs["api_base"] == proxy != j.kwargs["api_base"]
        assert proxy.startswith("http://127.0.0.1:") and j.kwargs["api_base"].startswith("http://127.0.0.1:")
        assert sorted(r for _, r in meters._meters) == ["judge", "user"]
    assert u.model == "openai/aws/claude-sonnet-5" and u.record()["base_url"] == tau.DEFAULT_GATEWAY


def test_given_base_url_is_metered_once(tmp_path, monkeypatch):
    monkeypatch.setenv("GRANITE_EVALS_USER_API_KEY", "k")
    b = _bench("tau3-airline", tmp_path, options={"user_base_url": "https://gw.example/v1"})
    with meter.Meters(b.config.options, {}) as meters:
        u = b.llm_endpoint("user", "http://vllm/v1", "granite", temperature=0.0)
        assert u.kwargs["api_base"].startswith("http://127.0.0.1:")
        assert list(meters._meters) == [("https://gw.example/v1", "user")]


# -- phases (fake harness) ------------------------------------------------------


class _RI:
    def __init__(self, reward):
        self.reward = reward

    def model_dump(self, mode):
        return {"reward": self.reward}


class _Sim:
    """A SimulationRun stand-in: what _simulation reads and saves."""

    termination_reason = "user_stop"

    def __init__(self, task_id, reward_info=None):
        self.task_id, self.reward_info, self.messages = task_id, reward_info, ["m1", "m2"]

    def model_dump(self, mode):
        return {"task_id": self.task_id, "reward_info": self.reward_info and self.reward_info.model_dump(mode)}


class _NLTask(_T):
    class evaluation_criteria:  # noqa: N801
        reward_basis = ["NL_ASSERTION"]


def _phase_bench(tmp_path, monkeypatch, phase, bid="tau3-retail", **options):
    """10 retail tasks graded on NL assertions x 2 trials; the model, the user
    simulator and the grader are fakes that record their calls."""
    b = _bench(bid, tmp_path, repeats=2, phase=phase, options=options)
    calls = {"simulate": [], "grade": [], "gold": []}
    monkeypatch.setattr(b, "load_tasks", lambda: ({d: [_NLTask(i) for i in range(10)] for d in b.domains},
                                                  tau.TAU2_REPO, "c"))
    monkeypatch.setattr(tau, "install_hooks", lambda: None)

    def simulate(domain, task, seed, agent, user):
        calls["simulate"].append((domain, task.id))
        # --phase all: graded by the harness in the run; generate: not graded
        return _Sim(task.id, _RI(1.0 if int(task.id) % 2 == 0 else 0.0) if b.scoring else None)

    def grade(domain, task, sim):
        calls["grade"].append((domain, task.id, b.gold))
        assert sim.task_id == task.id and sim.reward_info is None
        return _RI(1.0 if b.gold or int(task.id) % 2 == 0 else 0.0)

    def gold(domain, task):
        calls["gold"].append((domain, task.id))
        return _Sim(task.id)

    monkeypatch.setattr(b, "_simulate", simulate)
    monkeypatch.setattr(b, "_grade", grade)
    monkeypatch.setattr(b, "_gold", gold)
    monkeypatch.setattr(b, "_load_simulation", lambda dump: _Sim(dump["task_id"], None))
    return b, calls


def _saved(tmp_path, domain="retail"):
    return [json.loads(p.read_text()) for p in sorted((tmp_path / domain).rglob("*.json"))]


def test_generate_saves_ungraded_simulations_without_a_judge(tmp_path, monkeypatch):
    monkeypatch.delenv("GRANITE_EVALS_JUDGE_API_KEY", raising=False)
    b, calls = _phase_bench(tmp_path, monkeypatch, "generate", user_model="self")
    out = b.run("http://vllm/v1", "served")
    assert "value" not in out and out["n"] == 10 and out["simulations_generated"] == 20
    assert len(calls["simulate"]) == 20 and not calls["grade"]
    saved = _saved(tmp_path)
    assert len(saved) == 20
    assert all(r["status"] == tau.GENERATED and r["reward"] is None and r["reward_info"] is None for r in saved)
    assert all(r["simulation"]["reward_info"] is None for r in saved)
    # rerun: nothing is simulated again
    b2, calls2 = _phase_bench(tmp_path, monkeypatch, "generate", user_model="self")
    assert b2.run("http://vllm/v1", "served")["simulations_generated"] == 20 and not calls2["simulate"]


def test_score_grades_saved_simulations_without_the_model(tmp_path, monkeypatch):
    b, _ = _phase_bench(tmp_path, monkeypatch, "generate", user_model="self")
    b.run("http://vllm/v1", "served")
    # the user simulator's key is not needed to score; the judge's is
    monkeypatch.delenv("GRANITE_EVALS_USER_API_KEY", raising=False)
    monkeypatch.setenv("GRANITE_EVALS_JUDGE_API_KEY", "k")
    b, calls = _phase_bench(tmp_path, monkeypatch, "score")
    out = b.run("", "served")
    assert not calls["simulate"] and len(calls["grade"]) == 20
    assert out["value"] == 0.5 and out["n"] == 10 and out["simulations_failed"] == 0
    assert out["judge"]["model"] == tau.DEFAULT_SIM_MODEL and out["user_simulator"]["model"] == tau.DEFAULT_SIM_MODEL
    saved = _saved(tmp_path)
    assert {r["status"] for r in saved} == {"success", "fail"}
    assert all(r["reward_info"] == {"reward": r["reward"]} for r in saved)
    # rerun: nothing is graded again
    b2, calls2 = _phase_bench(tmp_path, monkeypatch, "score")
    assert b2.run("", "served")["value"] == 0.5 and not calls2["grade"]


def test_score_and_all_agree(tmp_path, monkeypatch):
    monkeypatch.setenv("GRANITE_EVALS_JUDGE_API_KEY", "k")
    b, _ = _phase_bench(tmp_path / "all", monkeypatch, "all", user_model="self")
    whole = b.run("http://vllm/v1", "served")
    b, _ = _phase_bench(tmp_path / "split", monkeypatch, "generate", user_model="self")
    b.run("http://vllm/v1", "served")
    b, _ = _phase_bench(tmp_path / "split", monkeypatch, "score", user_model="self")
    split = b.run("", "served")
    assert split["value"] == whole["value"] and split["domains"] == whole["domains"]


def test_score_counts_a_missing_simulation_as_failed(tmp_path, monkeypatch):
    b, _ = _phase_bench(tmp_path, monkeypatch, "generate", user_model="self")
    b.run("http://vllm/v1", "served")
    next((tmp_path / "retail" / "trial-1").glob("*.json")).unlink()
    monkeypatch.setenv("GRANITE_EVALS_JUDGE_API_KEY", "k")
    b, calls = _phase_bench(tmp_path, monkeypatch, "score")
    out = b.run("", "served")  # 1/20 missing: at the default max_failed_frac 0.05
    assert not calls["simulate"] and len(calls["grade"]) == 19
    assert out["incomplete"] and out["domains"]["retail"]["statuses"]["error:not_generated"] == 1
    assert len(_saved(tmp_path)) == 19  # score writes no simulation
    b, _ = _phase_bench(tmp_path, monkeypatch, "score", max_failed_frac="0")
    with pytest.raises(SystemExit, match="simulations_failed 1/20"):
        b.run("", "served")


def test_a_grading_error_keeps_the_generation(tmp_path, monkeypatch):
    b, _ = _phase_bench(tmp_path, monkeypatch, "generate", user_model="self")
    b.run("http://vllm/v1", "served")
    monkeypatch.setenv("GRANITE_EVALS_JUDGE_API_KEY", "k")
    b, _ = _phase_bench(tmp_path, monkeypatch, "score", max_failed_frac="1")
    monkeypatch.setattr(b, "_grade", lambda d, t, s: 1 / 0 if t.id == "3" else _RI(1.0))
    out = b.run("", "served")
    assert out["domains"]["retail"]["statuses"]["error:ZeroDivisionError"] == 2
    assert sum(r["status"] == tau.GENERATED for r in _saved(tmp_path)) == 2  # graded on the next run
    b, calls = _phase_bench(tmp_path, monkeypatch, "score")
    assert b.run("", "served")["simulations_failed"] == 0 and len(calls["grade"]) == 2


def test_all_grades_a_generated_simulation_without_simulating_it(tmp_path, monkeypatch):
    b, _ = _phase_bench(tmp_path, monkeypatch, "generate", user_model="self")
    b.run("http://vllm/v1", "served")
    monkeypatch.setenv("GRANITE_EVALS_JUDGE_API_KEY", "k")
    b, calls = _phase_bench(tmp_path, monkeypatch, "all", user_model="self")
    assert b.run("http://vllm/v1", "served")["value"] == 0.5
    assert not calls["simulate"] and len(calls["grade"]) == 20


def test_all_is_graded_in_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("GRANITE_EVALS_JUDGE_API_KEY", "k")
    b, calls = _phase_bench(tmp_path, monkeypatch, "all", user_model="self")
    out = b.run("http://vllm/v1", "served")
    assert out["value"] == 0.5 and len(calls["simulate"]) == 20 and not calls["grade"]
    assert out["judge"]["model"] == tau.DEFAULT_SIM_MODEL


def test_gold_in_every_phase(tmp_path, monkeypatch):
    b, calls = _phase_bench(tmp_path / "all", monkeypatch, "all", bid="tau3-bench", agent="gold")
    assert not b.needs_server() and not b.score_needs_server()
    assert b.run("", "")["value"] == 1.0 and len(calls["gold"]) == 60 == len(calls["grade"])
    b, calls = _phase_bench(tmp_path / "split", monkeypatch, "generate", bid="tau3-bench", agent="gold")
    assert b.run("", "")["n"] == 30 and len(calls["gold"]) == 60 and not calls["grade"]
    b, calls = _phase_bench(tmp_path / "split", monkeypatch, "score", bid="tau3-bench", agent="gold")
    out = b.run("", "")
    assert out["value"] == 1.0 and set(out["domains"]) == set(tau.CORE_DOMAINS)
    assert not calls["gold"] and len(calls["grade"]) == 60 and all(g for _, _, g in calls["grade"])


def test_score_needs_server_only_for_a_self_judge_on_nl_domains(tmp_path):
    def needs(bid, **options):
        return _bench(bid, tmp_path, options=options).score_needs_server()

    assert needs("tau3-bench", judge_model="self") and needs("tau3-banking-knowledge", judge_model="self")
    assert not needs("tau3-airline", judge_model="self") and not needs("tau3-telecom", judge_model="self")
    assert not needs("tau3-retail") and not needs("tau3-retail", judge_model="self", agent="gold")
    assert all(registry.get(bid).splittable for bid in SUITE)


# -- end to end with the harness -----------------------------------------------


def _data_dir():
    cands = [os.environ.get("GRANITE_EVALS_TAU2_TEST_DATA", ""), tau.BAKED_DATA_ROOT / tau.TAU2_COMMIT / "data",
             Path(os.environ.get("GRANITE_EVALS_TAU2_DATA_CACHE", Path.home() / ".cache/granite/tau2-data"))
             / tau.TAU2_COMMIT / "data"]
    for c in cands:
        if c and (Path(c) / "tau2" / "domains").is_dir():
            return Path(c)
    return None


harness = pytest.mark.skipif(
    importlib.util.find_spec("tau2") is None or _data_dir() is None,
    reason="needs the tau extra and the pinned tau2-bench data",
)


class _FakeLLM(BaseHTTPRequestHandler):
    """Agent (served model), user simulator and judge in one OpenAI-compatible
    server. The agent looks up a user once, then answers; the user asks once,
    then stops; the judge answers in a markdown fence."""

    protocol_version = "HTTP/1.1"
    calls: list[dict] = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeLLM.calls.append({"path": self.path, "model": body["model"], "auth": self.headers.get("Authorization")})
        msgs = body["messages"]
        model = body["model"]
        msg = {"role": "assistant", "content": "ok"}
        if model == "granite":
            if msgs[-1]["role"] == "user" and not any(m["role"] == "tool" for m in msgs):
                msg = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "get_user_details", "arguments": json.dumps({"user_id": "nobody"})}}]}
            else:
                msg["content"] = "I could not find you. Anything else?"
        elif model == "judge-model":
            msg["content"] = '```json\n{"results": []}\n```'
        else:  # user simulator: flipped roles, so its own turns are "assistant"
            turns = sum(m["role"] == "assistant" for m in msgs)
            msg["content"] = "Hi, please change my flight." if turns == 0 else "###STOP###"
        out = json.dumps({
            "id": "x", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
                      "prompt_tokens_details": {"cached_tokens": 40}},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("x-litellm-response-cost", "0.001")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def fake_llm():
    _FakeLLM.calls = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


@harness
def test_airline_end_to_end_metered_and_resumable(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setenv("GRANITE_EVALS_USER_API_KEY", "user-key")
    monkeypatch.setenv("GRANITE_EVALS_SPEND_LEDGER", str(tmp_path / "ledger.jsonl"))
    opts = {"user_base_url": fake_llm, "user_model": "aws/claude-sonnet-5", "max_retries": "0"}
    b = _bench("tau3-airline", tmp_path / "run", limit=1, repeats=1, dataset=str(_data_dir()), options=dict(opts))
    with meter.Meters(b.config.options, {"benchmark": b.id}) as meters:
        out = b.run(fake_llm, "granite")
        spend = meters.summary()
    rec = out["domains"]["airline"]
    # graded by the harness on the final DB; the fake agent changes nothing
    assert rec["n_simulations"] == 1 and sum(rec["statuses"].values()) == 1
    assert set(rec["statuses"]) <= {"success", "fail"}
    assert rec["termination_reasons"] == {"user_stop": 1}
    # the user simulator went through the meter, with its key; the agent did not
    user_calls = [c for c in _FakeLLM.calls if c["model"] == "aws/claude-sonnet-5"]
    assert len(user_calls) == 2 and all(c["auth"] == "Bearer user-key" for c in user_calls)
    assert spend["calls"] == 2 and spend["usd"] == pytest.approx(0.002)
    assert out["user_sim_usage"] == {"calls": 2, "prompt_tokens": 200, "completion_tokens": 20,
                                     "cached_tokens": 80, "cache_creation_tokens": 0}
    assert out["agent_usage"]["calls"] == 2 and out["judge"] is None
    saved = next((tmp_path / "run" / "airline" / "trial-0").glob("*.json"))
    sim = json.loads(saved.read_text())["simulation"]
    assert any(tc["name"] == "get_user_details" for m in sim["messages"] for tc in (m.get("tool_calls") or []))

    # resume: nothing is simulated again
    n = len(_FakeLLM.calls)
    b2 = _bench("tau3-airline", tmp_path / "run", limit=1, repeats=1, dataset=str(_data_dir()), options=dict(opts))
    assert b2.run(fake_llm, "granite")["domains"]["airline"]["pass_hat_1"] == rec["pass_hat_1"]
    assert len(_FakeLLM.calls) == n


@harness
def test_retail_nl_assertions_use_the_configured_judge(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setenv("GRANITE_EVALS_JUDGE_API_KEY", "judge-key")
    monkeypatch.setenv("GRANITE_EVALS_USER_API_KEY", "user-key")
    opts = {"user_base_url": fake_llm, "user_model": "user-model", "judge_base_url": fake_llm,
            "judge_model": "judge-model", "max_retries": "0"}
    # the judge runs only for tasks graded on NL assertions: pick the first one
    everything = _bench("tau3-retail", tmp_path, dataset=str(_data_dir()), options=dict(opts)).load_tasks()[0]
    task = next(t for t in everything["retail"]
                if "NL_ASSERTION" in tau._basis(t) and t.evaluation_criteria.nl_assertions)
    opts["tasks"] = "^" + re.escape(task.id) + "$"
    b = _bench("tau3-retail", tmp_path, limit=1, repeats=1, dataset=str(_data_dir()), options=opts)
    out = b.run(fake_llm, "granite")
    assert out["tasks"] == {"retail": [task.id]}
    judge_calls = [c for c in _FakeLLM.calls if c["model"] == "judge-model"]
    assert len(judge_calls) == 1 and judge_calls[0]["auth"] == "Bearer judge-key"
    assert out["judge_usage"]["calls"] == 1 and not out["judge_is_self"]
    assert out["judge"]["model"] == "judge-model"
    saved = json.loads(next((tmp_path / "retail" / "trial-0").glob("*.json")).read_text())
    assert saved["reward_info"]["nl_assertions"] == []  # fenced JSON parsed


@harness
def test_budget_exhausted_stops_paid_calls(tmp_path, fake_llm, monkeypatch):
    monkeypatch.setenv("GRANITE_EVALS_USER_API_KEY", "k")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"cost_usd": 100.0}) + "\n")
    monkeypatch.setenv("GRANITE_EVALS_SPEND_LEDGER", str(ledger))
    monkeypatch.setenv("GRANITE_EVALS_SPEND_BUDGET_USD", "50")
    opts = {"user_base_url": fake_llm, "max_retries": "0"}
    b = _bench("tau3-airline", tmp_path / "run", limit=2, repeats=1, dataset=str(_data_dir()), options=opts)
    with meter.Meters(b.config.options, {}), pytest.raises(SystemExit, match="402.*2 simulations not run"):
        b.run(fake_llm, "granite")
    assert not [c for c in _FakeLLM.calls if c["model"] == "aws/claude-sonnet-5"]
    assert not list((tmp_path / "run").rglob("*.json"))  # nothing saved: rerunnable


@harness
def test_gold_scores_every_task(tmp_path):
    for bid in ("tau3-airline", "tau3-telecom", "tau3-banking-knowledge"):
        b = _bench(bid, tmp_path / bid, limit=3, repeats=1, dataset=str(_data_dir()), options={"agent": "gold"})
        assert b.run("", "")["value"] == 1.0


@harness
def test_generate_then_score_matches_all(tmp_path, fake_llm, monkeypatch):
    """The live grading and the after-the-fact grading of the same kind of
    conversation (the fake is deterministic) give the same reward_info; the
    judge is called only when scoring."""
    monkeypatch.setenv("GRANITE_EVALS_JUDGE_API_KEY", "judge-key")
    monkeypatch.setenv("GRANITE_EVALS_USER_API_KEY", "user-key")
    opts = {"user_base_url": fake_llm, "user_model": "user-model", "judge_base_url": fake_llm,
            "judge_model": "judge-model", "max_retries": "0"}
    everything = _bench("tau3-retail", tmp_path, dataset=str(_data_dir()), options=dict(opts)).load_tasks()[0]
    task = next(t for t in everything["retail"]
                if "NL_ASSERTION" in tau._basis(t) and t.evaluation_criteria.nl_assertions)
    opts["tasks"] = "^" + re.escape(task.id) + "$"

    def bench(out, phase):
        return _bench("tau3-retail", out, limit=1, repeats=1, dataset=str(_data_dir()), options=dict(opts),
                      phase=phase)

    whole = bench(tmp_path / "all", "all").run(fake_llm, "granite")
    _FakeLLM.calls = []
    gen = bench(tmp_path / "split", "generate").run(fake_llm, "granite")
    assert gen["n"] == 1 and not [c for c in _FakeLLM.calls if c["model"] == "judge-model"]
    _FakeLLM.calls = []
    split = bench(tmp_path / "split", "score").run("", "granite")
    assert [c["model"] for c in _FakeLLM.calls] == ["judge-model"]
    assert split["domains"]["retail"]["pass_hat_1"] == whole["domains"]["retail"]["pass_hat_1"]

    def reward_info(out):
        return json.loads(next((out / "retail" / "trial-0").glob("*.json")).read_text())["reward_info"]

    assert reward_info(tmp_path / "split") == reward_info(tmp_path / "all")


@harness
def test_gold_generate_then_score(tmp_path):
    for bid in ("tau3-airline", "tau3-banking-knowledge"):
        out = tmp_path / bid
        kw = dict(limit=3, repeats=1, dataset=str(_data_dir()), options={"agent": "gold"})
        assert _bench(bid, out, phase="generate", **kw).run("", "")["n"] == 3
        assert _bench(bid, out, phase="score", **kw).run("", "")["value"] == 1.0
