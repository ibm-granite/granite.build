# SPDX-License-Identifier: Apache-2.0

"""generic lm-eval bridge tests: fakes only (no GPU, no network, no server,
no lm-eval install -- the harness surface is faked at the module's seams)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from granite_evals.benchmarks import chat_if, generic
from granite_evals.registry import RunConfig, get


def _task(output_type="multiple_choice", metric="acc"):
    return SimpleNamespace(
        OUTPUT_TYPE=output_type,
        config=SimpleNamespace(metric_list=[{"metric": metric}]),
    )


def _config(tmp_path: Path, **options):
    return RunConfig(model="m", output_dir=tmp_path, options=options)


def test_parse_tasks():
    assert generic.parse_tasks("a, b,a ") == ["a", "b"]
    with pytest.raises(SystemExit, match="tasks= is required"):
        generic.parse_tasks("")
    with pytest.raises(SystemExit, match="tasks= is required"):
        generic.parse_tasks(" , ")


def test_validate_tasks():
    known = ["arc_easy", "hellaswag", "mmlu"]
    generic.validate_tasks(["hellaswag", "mmlu"], known)  # no error
    with pytest.raises(SystemExit, match="helaswag"):
        generic.validate_tasks(["helaswag"], known)
    with pytest.raises(SystemExit, match="did you mean 'hellaswag'"):
        generic.validate_tasks(["helaswag"], known)


def test_parse_gen_kwargs():
    assert generic.parse_gen_kwargs(
        "temperature=0.6,max_gen_toks=2048,until=%3Cx%3E"
    ) == {
        "temperature": 0.6,
        "max_gen_toks": 2048,
        "until": "<x>",
    }
    with pytest.raises(SystemExit, match="key=value"):
        generic.parse_gen_kwargs("temperature")


def test_resolve_modes():
    types = {"a": "generate_until", "b": "multiple_choice", "c": "loglikelihood"}
    assert generic.resolve_modes(types, "auto") == {"base": ["b", "c"], "chat": ["a"]}
    assert generic.resolve_modes(types, "chat") == {"chat": ["a", "b", "c"]}
    with pytest.raises(SystemExit, match="mode must be one of"):
        generic.resolve_modes(types, "both")
    with pytest.raises(SystemExit, match="no auto protocol"):
        generic.resolve_modes({"a": "perplexity"}, "auto")


def test_metric_keys():
    assert generic.metric_keys("acc") == ["acc,none", "acc"]


RESULTS = {
    "results": {
        "hellaswag": {"acc,none": 0.5, "acc_stderr,none": 0.01, "acc_norm,none": 0.6},
        "arc_easy": {"acc,none": 0.7, "acc_stderr,none": 0.02},
        "mmlu": {"acc,none": 0.55},  # a group row: never in the tasks list
    },
    "n-samples": {
        "hellaswag": {"effective": 100, "original": 100},
        "arc_easy": {"effective": 50, "original": 50},
    },
    "samples": {
        "hellaswag": [{"doc_id": 0, "acc": 1.0}],
        "arc_easy": [{"doc_id": 0, "acc": 0.0}],
    },
    "groups": {"mmlu": {"acc,none": 0.55}},
}


def test_summarize_results():
    out = generic.summarize_results(
        RESULTS, ["arc_easy", "hellaswag"], {"arc_easy": "acc", "hellaswag": "acc"}
    )
    assert out["value"] == pytest.approx(0.6)  # macro mean, group row excluded
    assert out["n"] == 150
    assert out["per_task"]["hellaswag"] == {"metric": "acc", "value": 0.5, "n": 100}


def test_summarize_results_metric_override():
    out = generic.summarize_results(RESULTS, ["hellaswag"], {"hellaswag": "acc_norm"})
    assert out["value"] == pytest.approx(0.6)


def test_summarize_results_missing_metric():
    out = generic.summarize_results(
        RESULTS, ["arc_easy", "hellaswag"], {"arc_easy": "f1", "hellaswag": "acc"}
    )
    assert out["value"] == pytest.approx(0.5)  # the task without the key is left out
    assert out["per_task"]["arc_easy"]["value"] is None


def test_init_requires_tasks(tmp_path):
    with pytest.raises(SystemExit, match="tasks= is required"):
        generic.LMEval(_config(tmp_path))


def test_init_validates_names(tmp_path, monkeypatch):
    monkeypatch.setattr(generic, "task_index", lambda: ["hellaswag"])
    with pytest.raises(SystemExit, match="unknown task"):
        generic.LMEval(_config(tmp_path, tasks="helaswag"))
    generic.LMEval(_config(tmp_path, tasks="hellaswag"))  # no error


def test_init_rejects_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(generic, "task_index", lambda: ["hellaswag"])
    config = RunConfig(
        model="m",
        output_dir=tmp_path,
        dataset="some/data",
        options={"tasks": "hellaswag"},
    )
    with pytest.raises(SystemExit, match="do not apply"):
        generic.LMEval(config)


def test_init_parses_options(tmp_path, monkeypatch):
    monkeypatch.setattr(generic, "task_index", lambda: ["hellaswag", "gsm8k"])
    bench = generic.LMEval(
        _config(
            tmp_path,
            tasks="hellaswag,gsm8k",
            mode="chat",
            num_fewshot="5",
            gen_kwargs="temperature=0.6",
            metric="acc_norm",
        )
    )
    assert bench.task_list == ["hellaswag", "gsm8k"]
    assert bench.mode == "chat"
    assert bench.num_fewshot == 5
    assert bench.gen_kwargs == {"temperature": 0.6}
    assert bench.metric_override == "acc_norm"


def test_primary_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(generic, "task_index", lambda: ["a", "b"])
    bench = generic.LMEval(_config(tmp_path, tasks="a,b"))
    tasks = {"a": _task(metric="acc"), "b": _task(metric="exact_match")}
    assert bench.primary_metrics(tasks) == {"a": "acc", "b": "exact_match"}
    bench = generic.LMEval(_config(tmp_path, tasks="a,b", metric="acc_norm"))
    assert bench.primary_metrics(tasks) == {"a": "acc_norm", "b": "acc_norm"}


def test_registry():
    assert get("lm-eval") is generic.LMEval


def test_run_end_to_end(tmp_path, monkeypatch):
    """The orchestration, with the harness faked at its three seams: task
    names, task objects, simple_evaluate. One base-mode task pair and one
    chat-mode task, so the mixed-list two-pass path runs."""
    monkeypatch.setattr(
        generic, "task_index", lambda: ["hellaswag", "arc_easy", "gsm8k"]
    )
    monkeypatch.setattr(
        generic,
        "load_tasks",
        lambda tasks: (
            {
                "hellaswag": _task("multiple_choice", "acc"),
                "arc_easy": _task("multiple_choice", "acc"),
                "gsm8k": _task("generate_until", "exact_match"),
            },
            {"lm-eval-test-group": SimpleNamespace()},
        ),
    )
    calls = []

    def fake_simple_evaluate(**kwargs):
        calls.append(kwargs)
        if "gsm8k" in kwargs["tasks"]:
            return {
                "results": {"gsm8k": {"exact_match,none": 0.9}},
                "n-samples": {"gsm8k": {"effective": 30, "original": 30}},
                "samples": {"gsm8k": [{"doc_id": 0, "exact_match": 1.0}]},
                "groups": {},
            }
        return RESULTS

    monkeypatch.setattr(generic, "run_simple_evaluate", fake_simple_evaluate)
    monkeypatch.setattr(
        generic.LMEval,
        "make_lm",
        lambda self, mode, base_url, served, seed: SimpleNamespace(mode=mode),
    )
    monkeypatch.setattr(chat_if, "drop_errors_from_cache", lambda cache_dir: 0)

    bench = generic.LMEval(_config(tmp_path, tasks="hellaswag,arc_easy,gsm8k"))
    outcome = bench.run("http://server/v1", "m")

    assert outcome["value"] == pytest.approx((0.5 + 0.7 + 0.9) / 3)
    assert outcome["n"] == 180
    assert outcome["mode"] == {"base": 2, "chat": 1}
    assert set(outcome["per_task"]) == {"hellaswag", "arc_easy", "gsm8k"}
    assert len(outcome["per_repeat"]) == 1

    # Two passes, one per mode, chat flags only on the chat pass.
    assert [tuple(c["tasks"]) for c in calls] == [("hellaswag", "arc_easy"), ("gsm8k",)]
    assert [c["apply_chat_template"] for c in calls] == [False, True]
    assert [c["fewshot_as_multiturn"] for c in calls] == [False, True]

    # Artifacts: merged lm_eval_results.json and per-task sample rows.
    keep = json.loads((tmp_path / "repeat-0" / "lm_eval_results.json").read_text())
    assert keep["leaf_tasks"] == ["arc_easy", "gsm8k", "hellaswag"]
    assert set(keep["results"]) == {"hellaswag", "arc_easy", "gsm8k", "mmlu"}
    rows = (tmp_path / "repeat-0" / "samples" / "gsm8k.jsonl").read_text().splitlines()
    assert json.loads(rows[0])["exact_match"] == 1.0
