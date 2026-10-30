"""MultiChallenge adapter tests with a fake policy model and fake judges
(no GPU, no network, no paid calls; synthetic conversations)."""

import json
import types

import pytest

pytest.importorskip("openai")

from test_judge_general import Usage, completion, fake_judge, user_text  # noqa: E402

from granite_evals.benchmarks import multi_challenge as mc  # noqa: E402
from granite_evals.registry import RunConfig, get  # noqa: E402


def q(qid, axis, rubric):
    return {"QUESTION_ID": qid, "AXIS": axis, "TARGET_QUESTION": rubric, "PASS_CRITERIA": "YES",
            "CONVERSATION": [{"role": "user", "content": f"hi {qid}"}, {"role": "assistant", "content": "hello"},
                             {"role": "user", "content": f"now answer {qid}"}]}


ROWS = [q("b2", "INFERENCE_MEMORY", "Does it say PASS?"), q("a1", "INFERENCE_MEMORY", "Does it say PASS?"),
        q("c3", "SELF_COHERENCE", "Does it say PASS?")]
SHIPPED = [{"QUESTION_ID": "a1", "RESPONSE": ["PASS"]}, {"QUESTION_ID": "b2", "RESPONSE": ["nope"]},
           {"QUESTION_ID": "c3", "RESPONSE": ["PASS"]}]


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(mc.MultiChallenge, "_fetch", staticmethod(lambda rel, sha: rel))
    monkeypatch.setattr(mc, "read_rows", lambda p: [dict(r) for r in (SHIPPED if "final_model" in str(p) else ROWS)])
    state = types.SimpleNamespace(policy=[], answer=lambda r: "PASS")

    class FakeOpenAI:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        def create(self, **request):
            state.policy.append(request)
            return completion(state.answer(request), Usage(20, 30))

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    return state


def make(tmp_path, monkeypatch, judge, phase="all", repeats=None, limit=None, **options):
    bench = get("multi-challenge")(RunConfig(model="m", output_dir=tmp_path, options=options, workers=2,
                                             repeats=repeats, limit=limit))
    bench.config.phase = phase
    monkeypatch.setattr(bench, "make_judge", lambda *a, **k: judge)
    return bench


def structured_judge(request):
    response = user_text(request).split("<MODEL_RESPONSE>\n")[1].split("\n</MODEL_RESPONSE>")[0]
    return json.dumps({"reasoning": "checked", "verdict": "YES" if "PASS" in response else "NO"})


def test_registration():
    cls = get("multi-challenge")
    assert cls.extra == "judged" and cls.splittable and cls.default_repeats == 1 and len(cls.dataset_revision) == 40
    assert set(cls.gold_modes) == set(mc.MC_RESPONSES) == set(mc.MC_PAPER_AUTO)


def test_verdict_parsing():
    assert mc.parse_verdict('{"reasoning": "NO way", "verdict": "YES"}') == "YES"
    assert mc.parse_verdict('```json\n{"reasoning": "r", "verdict": "NO"}\n```') == "NO"
    assert mc.parse_verdict("Reasoning: it says YES early.\nVerdict: no") == "NO"
    assert mc.parse_verdict("It fails. NO") == "NO"
    assert mc.parse_verdict("I can't grade this.") is None


def test_macro_over_axes():
    value, axes = mc.macro({"a": 1.0, "b": 0.0, "c": 1.0}, {"a": "X", "b": "X", "c": "Y"})
    assert axes == {"X": 0.5, "Y": 1.0} and value == 0.75  # unweighted over axes, not 2/3


def test_conversation_is_sent_and_judged(tmp_path, monkeypatch, env):
    env.answer = lambda r: "nope" if r["messages"][-1]["content"].endswith("b2") else "PASS"
    judge = fake_judge(structured_judge)
    out = make(tmp_path, monkeypatch, judge).run("http://p/v1", "s")
    assert out["value"] == 0.75 and out["axis_scores"] == {"INFERENCE_MEMORY": 0.5, "SELF_COHERENCE": 1.0}
    assert out["metric_name"] == "pass@1" and out["n"] == 3
    assert (out["pass_at_k"]["k"], out["pass_at_k"]["pass_at_1"], out["pass_at_k"]["pass_at_k"]) == (1, 0.75, 0.75)
    first = sorted(env.policy, key=lambda r: r["messages"][-1]["content"])[0]
    assert [m["role"] for m in first["messages"]] == ["user", "assistant", "user"]
    assert set(first) == {"model", "messages", "seed"}
    req = judge._client.requests[0]
    assert req["temperature"] == 0 and req["response_format"] == mc.MC_RESPONSE_FORMAT
    assert len(req["messages"]) == 1 and "Be VERY STRICT!" in user_text(req)
    assert "hi " not in user_text(req)  # the judge sees the final answer and the rubric only
    assert out["judge_structured_output"] and out["deviations"]


def test_repeats_give_avg_of_k_and_pass_at_k(tmp_path, monkeypatch, env):
    # a1 passes on seed 0 only; b2 never; c3 always
    def answer(r):
        qid = r["messages"][-1]["content"].split()[-1]
        return "PASS" if qid == "c3" or (qid == "a1" and r["seed"] == 0) else "nope"

    env.answer = answer
    out = make(tmp_path, monkeypatch, fake_judge(structured_judge), repeats=2).run("http://p/v1", "s")
    # avg-of-2: IM = (0.5 + 0) / 2 = 0.25, SC = 1 -> 0.625; pass@2: IM 0.5, SC 1 -> 0.75
    assert out["value"] == 0.625 and out["pass_at_k"]["pass_at_k"] == 0.75 and out["pass_at_k"]["k"] == 2
    assert out["metric_name"] == "pass@1[avg-of-2]" and out["per_repeat"] == [0.75, 0.5]


def test_rejected_response_format_falls_back_to_text_once(tmp_path, monkeypatch, env):
    def text_judge(request):
        return "The response says PASS.\nVerdict: YES"

    judge = fake_judge(text_judge, reject=("response_format",))
    out = make(tmp_path, monkeypatch, judge).run("http://p/v1", "s")
    assert out["value"] == 1.0 and not out["judge_structured_output"]
    assert len(out["judge_request_adaptations"]) == 1
    assert sum("response_format" in r for r in judge._client.requests) >= 1
    assert "response_format" not in judge._client.requests[-1]


def test_judge_failures_and_empty_answers(tmp_path, monkeypatch, env):
    env.answer = lambda r: "" if r["messages"][-1]["content"].endswith("c3") else "PASS"
    judge = fake_judge(lambda r: "I can't help with evaluating that." if "PASS" in user_text(r) else "YES")
    with pytest.raises(SystemExit, match="responses_failed 2/3"):
        make(tmp_path, monkeypatch, judge).run("http://p/v1", "s")
    out = make(tmp_path, monkeypatch, fake_judge(structured_judge)).run("http://p/v1", "s")
    # resumed: a1/b2 judged now, c3's empty answer failed without a judge call
    assert out["value"] == 0.5 and out["empty_responses_not_judged"] == 1 and not out["incomplete"]


def test_shipped_responses_gold(tmp_path, monkeypatch, env):
    bench = make(tmp_path, monkeypatch, fake_judge(structured_judge), responses="o1-preview")
    assert not bench.needs_server()
    out = bench.run("", "none")
    assert env.policy == [] and out["value"] == 0.75 and out["paper_reference"]["average"] == 35.73
    assert "data/final_model_responses/o1-preview_responses.jsonl" in out["dataset_sha256"]
    with pytest.raises(SystemExit, match="one attempt"):
        make(tmp_path / "k", monkeypatch, fake_judge(structured_judge), responses="o1-preview", repeats=2).run("", "")


def test_generate_then_score_and_limit(tmp_path, monkeypatch, env):
    monkeypatch.delenv("GRANITE_EVALS_JUDGE_API_KEY", raising=False)
    gen = make(tmp_path, monkeypatch, None, phase="generate", limit=2).run("http://p/v1", "s")
    assert gen["n"] == 2 and sorted(p.name for p in (tmp_path / "items").iterdir()) == ["a1", "b2"]
    env.policy.clear()
    out = make(tmp_path, monkeypatch, fake_judge(structured_judge), phase="score", limit=2).run("", "s")
    assert env.policy == [] and out["value"] == 1.0 and out["n"] == 2
