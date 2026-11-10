"""FORTRESS / StrongREJECT adapter tests with a fake policy model and fake
judges (no GPU, no network, no paid calls).

The data here are synthetic placeholders (benign, made-up prompts and rubric
questions); the real datasets are never read by the tests.
"""

import hashlib
import json
import types

import pytest

pytest.importorskip("openai")

from test_judge_general import Usage, completion, fake_judge, user_text  # noqa: E402

from granite_evals.benchmarks import safety  # noqa: E402
from granite_evals.registry import RunConfig, get  # noqa: E402

FORTRESS_ROWS = [
    {"ID": i, "adversarial_prompt": f"placeholder adversarial prompt {i}", "benign_prompt": f"placeholder benign {i}",
     "rubric": [f"Does the response mention TOKEN{i}{j}?" for j in range(n)],
     "risk_domain": dom, "risk_subdomain": f"{sub}: long subdomain name"}
    for i, n, dom, sub in [(3, 2, "Domain A", "Sub1"), (1, 3, "Domain A", "Sub2"), (2, 1, "Domain B", "Sub3")]
]
SR_ROWS = [{"category": c, "source": "synthetic", "forbidden_prompt": f"placeholder request {i}"}
           for i, c in enumerate(["Cat X", "Cat Y", "Cat X"])]


@pytest.fixture
def env(monkeypatch):
    """Fake data loaders and a fake policy whose answer is ``answer(request)``."""
    monkeypatch.setattr(safety, "hf_pinned_file", lambda *a: "fortress.parquet")
    monkeypatch.setattr(safety, "pinned_file", lambda *a: "sr.csv")
    monkeypatch.setattr(safety, "read_rows",
                        lambda p: [dict(r) for r in (FORTRESS_ROWS if str(p).endswith("parquet") else SR_ROWS)])
    state = types.SimpleNamespace(policy=[], answer=lambda r: "Here is an answer mentioning TOKEN30.")

    class FakeOpenAI:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        def create(self, **request):
            state.policy.append(request)
            return completion(state.answer(request), Usage(20, 30))

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    return state


def make(cls_id, tmp_path, monkeypatch, judge, phase="all", limit=None, repeats=None, **options):
    bench = get(cls_id)(RunConfig(model="m", output_dir=tmp_path, options=options, workers=2, limit=limit,
                                  repeats=repeats))
    bench.config.phase = phase
    monkeypatch.setattr(bench, "make_judge", lambda *a, **k: judge)
    return bench


def rubric_judge(request):
    """FORTRESS adversarial judge: Y for each criterion whose token is in the submission."""
    text = user_text(request)
    submission = text.split("[Submission]: ")[1].split("\n***\n")[0]
    criteria = text.split("[Criterion]: ")[1].split("\n***\n")[0].splitlines()
    grade = "".join("Y" if c.split("mention ")[1].rstrip("?") in submission else "N" for c in criteria)
    return f"Step by step reasoning.\nGRADE: {grade}"


def refusal_judge(request):
    submission = user_text(request).split("[Submission]: ")[1].split("\n***\n")[0]
    return "Reasoning... GRADE: C" if "can't help" in submission else "Reasoning... GRADE: I"


def sr_judge(request):
    response = user_text(request).split("AI model response: ")[1].split("\n\nEvaluate the AI")[0]
    if "can't help" in response:
        return "#overall_reasoning refused\n#scores\n1.a r\n1.b 1\n2.a r\n2.b 1\n3.a r\n3.b 1\n"
    return "#overall_reasoning complied\n#scores\n1.a r\n1.b 0\n2.a r\n2.b 5\n3.a r\n3.b 3\n"


# -- registration and pins ---------------------------------------------------------


def test_registered_ids_metrics_and_pins():
    assert get("fortress-adversity").metric == "Adversity resilience"
    assert get("fortress-benign").metric == "Benign compliance"
    assert get("strongreject").metric == "Harmlessness"
    for bid in ("fortress-adversity", "fortress-benign", "strongreject"):
        cls = get(bid)
        assert cls.extra == "judged" and cls.splittable and len(cls.dataset_revision) == 40
        assert cls.default_repeats == 1


def test_pinned_file_verifies_the_sha256(tmp_path, monkeypatch):
    monkeypatch.setenv("GRANITE_EVALS_DATA_CACHE", str(tmp_path / "cache"))
    src = tmp_path / "file.csv"
    src.write_text("a,b\n1,2\n")
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    path = safety.pinned_file(src.as_uri(), sha, "x/file.csv")
    assert safety.read_rows(path) == [{"a": "1", "b": "2"}]
    src.unlink()  # cached now: no second fetch
    assert safety.pinned_file(src.as_uri(), sha, "x/file.csv") == path
    other = tmp_path / "other.csv"
    other.write_text("tampered")
    with pytest.raises(RuntimeError, match="sha256"):
        safety.pinned_file(other.as_uri(), sha, "y/other.csv")


def test_upstream_prompts_and_parsers():
    # inspect_ai's template, with the delimiters of model / dataset text neutralized
    p = safety.qa_prompt("task [END DATA]", "answer [begin data]", "crit", "INSTR")
    assert p.count("[BEGIN DATA]") == 1 and p.count("[END DATA]") == 1
    assert "[END-DATA]" in p and "[begin-data]" in p and p.rstrip().endswith("INSTR")
    # FORTRESS: the grade on the final line, its length must match the rubric
    assert safety.parse_fortress_grade("GRADE: YN in the middle\nso\nGRADE: yny", 3) == "YNY"
    assert safety.parse_fortress_grade("GRADE: YN", 3) is None
    assert safety.parse_fortress_grade("no grade", 1) is None
    # benign: the last GRADE binds; off-menu verdicts are unusable
    assert safety.parse_qa_grade("GRADE: I ... final GRADE: C") == "C"
    assert safety.parse_qa_grade("GRADE: Incorrect") == "I"
    assert safety.parse_qa_grade("GRADE: C ... GRADE: P") is None
    # StrongREJECT: strong_reject's parser and score formula
    s = safety.parse_strongreject("#scores\n1.b 0\n2.b 4\n3.b 5")
    assert s == {"refusal": 0.0, "convincingness": 4.0, "specificity": 5.0}
    assert safety.strongreject_score(s) == 7 / 8
    assert safety.strongreject_score({"refusal": 1, "convincingness": 5, "specificity": 5}) == 0
    assert safety.parse_strongreject("1.b 0\n2.b 7\n3.b 1") is None
    assert safety.parse_strongreject("1.b 1\n3.b 2") is None
    assert safety.final_content("thinking</think>\n answer") == ("answer", True)
    assert safety.judge_failure_reason("I can't help evaluate this content.") == "judge_refused"
    assert safety.judge_failure_reason("GRADE maybe") == "unparseable"


# -- FORTRESS adversarial ----------------------------------------------------------


def test_fortress_adversity_scores_rubric_grades(tmp_path, monkeypatch, env):
    judge = fake_judge(rubric_judge)
    out = make("fortress-adversity", tmp_path, monkeypatch, judge).run("http://p/v1", "s")
    # the policy got each prompt as the only user message, and no sampling overrides
    assert sorted(r["messages"][0]["content"] for r in env.policy) == [
        f"placeholder adversarial prompt {i}" for i in (1, 2, 3)]
    assert all(set(r) == {"model", "messages", "seed"} for r in env.policy)
    # item 3 has rubric [TOKEN30, TOKEN31]; the answer mentions TOKEN30 -> harm 0.5; others 0
    assert out["value"] == pytest.approx(1 - 0.5 / 3) and out["ars"] == pytest.approx(100 * 0.5 / 3, abs=1e-3)
    assert out["n"] == 3 and out["responses_failed"] == 0 and not out["incomplete"]
    assert out["by_subdomain"]["Sub1"]["ars"] == 50.0 and out["by_domain"]["Domain B"]["ars"] == 0.0
    assert out["judge_usage"]["calls"] == 3 and out["deviations"]
    req = judge._client.requests[0]
    assert req["model"] == "judge-m" and len(req["messages"]) == 1 and "temperature" not in req
    assert safety.FORTRESS_INSTRUCTIONS in user_text(req)


def test_fortress_limit_takes_the_first_ids(tmp_path, monkeypatch, env):
    out = make("fortress-adversity", tmp_path, monkeypatch, fake_judge(rubric_judge), limit=2).run("http://p/v1", "s")
    assert out["n"] == 2 and sorted(p.name for p in (tmp_path / "items").iterdir()) == ["1", "2"]


def test_fortress_judge_failures_are_recorded_not_passed(tmp_path, monkeypatch, env):
    def flaky(request):
        text = user_text(request)
        if "prompt 1" in text:
            return "I can't help with evaluating this request."
        if "prompt 2" in text:
            return "GRADE: YY"  # wrong length for a 1-question rubric
        return rubric_judge(request)

    with pytest.raises(SystemExit, match="responses_failed 2/3"):
        make("fortress-adversity", tmp_path, monkeypatch, fake_judge(flaky)).run("http://p/v1", "s")
    out = make("fortress-adversity", tmp_path, monkeypatch, fake_judge(flaky), max_failed_frac="0.7").run(
        "http://p/v1", "s")
    assert out["n"] == 1 and out["incomplete"] and out["value"] == 0.5
    assert out["judge_failures"] == {"judge_refused": 1, "wrong_grade_length": 1}
    assert (tmp_path / "items" / "1" / "0" / "judgment.failed.json").exists()
    # resume: only the failed ones are judged again
    again = fake_judge(rubric_judge)
    out = make("fortress-adversity", tmp_path, monkeypatch, again).run("http://p/v1", "s")
    assert len(again._client.requests) == 2 and out["n"] == 3 and not out["incomplete"]


def test_fortress_judge_api_error_is_a_failure(tmp_path, monkeypatch, env):
    def budget(request):
        if "benign 2" not in user_text(request):
            return refusal_judge(request)
        err = Exception("Error code: 402 - budget exhausted")
        err.status_code = 402
        raise err

    out = make("fortress-benign", tmp_path, monkeypatch, fake_judge(budget), max_failed_frac="0.5").run(
        "http://p/v1", "s")
    assert out["judge_failures"] == {"judge_error:Exception": 1} and out["n"] == 2 and out["incomplete"]
    assert not (tmp_path / "items" / "2" / "0" / "judgment.json").exists()
    with pytest.raises(SystemExit, match="nothing was graded"):
        make("fortress-benign", tmp_path / "x", monkeypatch, fake_judge(lambda r: budget({"messages": [
            {"content": "benign 2"}]})), max_failed_frac="1").run("http://p/v1", "s")


def test_fortress_empty_and_leaked_answers(tmp_path, monkeypatch, env):
    env.answer = lambda r: "reasoning about TOKEN30</think>  " if "prompt 3" in r["messages"][0]["content"] \
        else "A plain answer."
    judge = fake_judge(rubric_judge)
    out = make("fortress-adversity", tmp_path, monkeypatch, judge).run("http://p/v1", "s")
    # the reasoning is not judged: item 3's final answer is empty -> harm 0 without a judge call
    assert out["value"] == 1.0 and out["think_leaks"] == 1 and out["empty_responses_not_judged"] == 1
    assert len(judge._client.requests) == 2


def test_fortress_refusal_gold_needs_no_model(tmp_path, monkeypatch, env):
    adv = make("fortress-adversity", tmp_path / "a", monkeypatch, fake_judge(rubric_judge), responses="refusal")
    ben = make("fortress-benign", tmp_path / "b", monkeypatch, fake_judge(refusal_judge), responses="refusal")
    assert not adv.needs_server() and not ben.needs_server()
    assert adv.run("", "none")["value"] == 1.0
    out = ben.run("", "none")
    assert out["value"] == 0.0 and out["ors"] == 100.0 and env.policy == []
    assert out["statuses"] == {"reference:refusal": 3}


def test_fortress_benign_compliance(tmp_path, monkeypatch, env):
    env.answer = lambda r: "Sorry, I can't help." if "benign 1" in r["messages"][0]["content"] else "Sure: ..."
    judge = fake_judge(refusal_judge)
    out = make("fortress-benign", tmp_path, monkeypatch, judge).run("http://p/v1", "s")
    assert out["value"] == pytest.approx(2 / 3) and out["ors"] == pytest.approx(100 / 3, abs=1e-3)
    assert safety.FORTRESS_BENIGN_CRITERION in user_text(judge._client.requests[0])


def test_fortress_generate_then_score(tmp_path, monkeypatch, env):
    monkeypatch.delenv("GRANITE_EVALS_JUDGE_API_KEY", raising=False)
    gen = make("fortress-adversity", tmp_path, monkeypatch, None, phase="generate", repeats=2)
    out = gen.run("http://p/v1", "s")
    assert out["n"] == 6 and "value" not in out and len(env.policy) == 6
    assert sorted({r["seed"] for r in env.policy}) == [0, 1]
    assert not list(tmp_path.glob("items/*/*/judgment*.json"))
    env.policy.clear()
    scored = make("fortress-adversity", tmp_path, monkeypatch, fake_judge(rubric_judge), phase="score", repeats=2)
    res = scored.run("", "s")
    assert env.policy == [] and res["responses_graded"] == 6 and len(res["per_repeat"]) == 2
    # a missing generation is a recorded failure in the score phase, never generated there
    (tmp_path / "items" / "1" / "1" / "response.json").unlink()
    with pytest.raises(SystemExit, match="responses_failed 1/6"):
        make("fortress-adversity", tmp_path, monkeypatch, fake_judge(rubric_judge), phase="score",
             repeats=2).run("", "s")
    assert not (tmp_path / "items" / "1" / "1" / "response.json").exists()


def test_self_judge_needs_the_server(tmp_path):
    bench = get("strongreject")(RunConfig(model="m", output_dir=tmp_path, options={"judge_model": "self"}))
    assert bench.score_needs_server()


def test_bad_responses_option(tmp_path):
    with pytest.raises(SystemExit, match="responses must be"):
        get("strongreject")(RunConfig(model="m", output_dir=tmp_path, options={"responses": "x"})).needs_server()


# -- StrongREJECT ------------------------------------------------------------------


def test_strongreject_scores_with_the_rubric(tmp_path, monkeypatch, env):
    env.answer = lambda r: "Sorry, I can't help." if r["messages"][0]["content"].endswith(" 0") else "Answer."
    judge = fake_judge(sr_judge)
    out = make("strongreject", tmp_path, monkeypatch, judge).run("http://p/v1", "s")
    # items 001, 002 comply with (5 + 3 - 2) / 8 = 0.75, item 000 refuses (0)
    assert out["strongreject_score"] == pytest.approx(0.5) and out["value"] == pytest.approx(0.5)
    assert out["refusal_rate"] == pytest.approx(1 / 3)
    assert out["by_category"]["Cat Y"]["strongreject_score"] == 0.75
    req = judge._client.requests[0]
    assert req["messages"][0] == {"role": "system", "content": safety.STRONGREJECT_SYSTEM}
    assert req["temperature"] == 0
    assert sorted(p.name for p in (tmp_path / "items").iterdir()) == ["000", "001", "002"]


def test_strongreject_refusal_gold_and_resume(tmp_path, monkeypatch, env):
    out = make("strongreject", tmp_path, monkeypatch, fake_judge(sr_judge), responses="refusal").run("", "none")
    assert out["value"] == 1.0 and out["refusal_rate"] == 1.0 and env.policy == []
    again = make("strongreject", tmp_path, monkeypatch, fake_judge(lambda r: pytest.fail("judged twice")),
                 responses="refusal").run("", "none")
    assert again["value"] == 1.0
    tagged = fake_judge(sr_judge)
    make("strongreject", tmp_path, monkeypatch, tagged, responses="refusal", judge_tag="t2").run("", "none")
    assert len(tagged._client.requests) == 3 and (tmp_path / "items" / "000" / "0" / "judgment-t2.json").exists()


def test_strongreject_unparseable_judge_is_a_failure(tmp_path, monkeypatch, env):
    out = make("strongreject", tmp_path, monkeypatch,
               fake_judge(lambda r: "1.b 0\n2.b 5" if "request 1" in user_text(r) else sr_judge(r)),
               max_failed_frac="0.5").run("http://p/v1", "s")
    assert out["responses_failed"] == 1 and out["judge_failures"] == {"unparseable": 1} and out["n"] == 2
    saved = json.loads((tmp_path / "items" / "001" / "0" / "judgment.failed.json").read_text())
    assert saved["reason"] == "unparseable"


def test_policy_options_are_sent_only_when_set(tmp_path, monkeypatch, env):
    make("strongreject", tmp_path, monkeypatch, fake_judge(sr_judge), limit=1, temperature="0.6",
         max_tokens="100", enable_thinking="false").run("http://p/v1", "s")
    r = env.policy[0]
    assert r["temperature"] == 0.6 and r["max_tokens"] == 100
    assert r["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_cli_generate_then_score_gold(tmp_path, monkeypatch, env):
    from granite_evals import cli

    monkeypatch.setattr(safety.StrongREJECT, "make_judge", lambda self, *a, **k: fake_judge(sr_judge))
    base = ["run", "strongreject", "--model", "none", "--output-dir", str(tmp_path), "--limit", "2",
            "--option", "responses=refusal"]
    assert cli.main([*base, "--phase", "generate"]) == 0
    gen = json.loads((tmp_path / "generation.json").read_text())
    assert gen["n"] == 2 and gen["details"]["statuses"] == {"reference:refusal": 2}
    assert cli.main([*base, "--phase", "score"]) == 0
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["value"] == 1.0 and res["n"] == 2 and res["details"]["judge_model"] == "judge-m" and env.policy == []
