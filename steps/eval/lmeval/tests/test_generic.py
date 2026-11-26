# SPDX-License-Identifier: Apache-2.0

"""generic lm-eval bridge tests: fakes only (no GPU, no network, no server,
no lm-eval install -- the harness surface is faked at the module's seams)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from granite_evals.benchmarks import chat_if, generic
from granite_evals.registry import RunConfig, get


class _FakeTask(SimpleNamespace):
    def set_config(self, key, value, update=False):
        """lm-eval's Task.set_config: update merges a dict into the key."""
        if update and isinstance(getattr(self.config, key, None), dict):
            getattr(self.config, key).update(value)
        else:
            setattr(self.config, key, value)


def _task(
    output_type="multiple_choice",
    metric="acc",
    filters=None,
    name=None,
    generation_kwargs=None,
    metrics=None,
):
    return _FakeTask(
        task_name=name,
        OUTPUT_TYPE=output_type,
        config=SimpleNamespace(
            metric_list=[{"metric": m} for m in (metrics or [metric])],
            filter_list=(
                [{"name": f, "filter": []} for f in filters] if filters else None
            ),
            generation_kwargs=dict(generation_kwargs or {}),
        ),
    )


class _Group:
    """lm-eval's Group surface the bridge uses: name, direct children in
    yaml order (iteration), and every leaf (get_all_tasks)."""

    def __init__(self, name, *children):
        self.name = name
        self.children = list(children)

    def __iter__(self):
        return iter(self.children)

    def get_all_tasks(self):
        out = []
        for child in self.children:
            out += child.get_all_tasks() if isinstance(child, _Group) else [child]
        return out


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
    assert generic.metric_keys("exact_match", ["strict-match", "flexible-extract"]) == [
        "exact_match,none",
        "exact_match",
        "exact_match,strict-match",
        "exact_match,flexible-extract",
    ]
    # An explicit <metric>,<filter> is exactly that key.
    assert generic.metric_keys("exact_match,flexible-extract", ["strict-match"]) == [
        "exact_match,flexible-extract"
    ]


def test_task_filters():
    assert generic.task_filters(_task()) == []
    assert generic.task_filters(
        _task(filters=["strict-match", "flexible-extract"])
    ) == ["strict-match", "flexible-extract"]
    # filter_list entries as objects (a parsed config) work the same way.
    task = SimpleNamespace(
        config=SimpleNamespace(filter_list=[SimpleNamespace(name="maj@1")])
    )
    assert generic.task_filters(task) == ["maj@1"]


# gsm8k's row shape at lm-eval 0.4.13: one key per filter_list entry, no ",none".
GSM8K_ROW = {
    "alias": "gsm8k",
    "exact_match,strict-match": 0.25,
    "exact_match_stderr,strict-match": 0.01,
    "exact_match,flexible-extract": 0.75,
    "exact_match_stderr,flexible-extract": 0.02,
}
GSM8K_FILTERS = {"gsm8k": ["strict-match", "flexible-extract"]}


def test_summarize_results_filtered_metric():
    results = {
        "results": {"gsm8k": GSM8K_ROW},
        "n-samples": {"gsm8k": {"effective": 4, "original": 4}},
    }
    # The task's first filter, in filter_list order (not dict order).
    reordered = {
        "results": {"gsm8k": dict(reversed(list(GSM8K_ROW.items())))},
        "n-samples": results["n-samples"],
    }
    for res in (results, reordered):
        out = generic.summarize_results(
            res, ["gsm8k"], {"gsm8k": "exact_match"}, GSM8K_FILTERS
        )
        assert out["value"] == pytest.approx(0.25)
        assert out["per_task"]["gsm8k"] == {
            "metric": "exact_match",
            "key": "exact_match,strict-match",
            "value": 0.25,
            "n": 4,
        }
    # metric=<metric>,<filter> picks another filter.
    out = generic.summarize_results(
        results,
        ["gsm8k"],
        {"gsm8k": "exact_match,flexible-extract"},
        GSM8K_FILTERS,
    )
    assert out["value"] == pytest.approx(0.75)


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
    assert out["per_task"]["hellaswag"] == {
        "metric": "acc",
        "key": "acc,none",
        "value": 0.5,
        "n": 100,
    }


def test_summarize_results_metric_override():
    out = generic.summarize_results(RESULTS, ["hellaswag"], {"hellaswag": "acc_norm"})
    assert out["value"] == pytest.approx(0.6)


def test_summarize_results_missing_metric():
    """A metric the row does not carry fails the run, naming what it has,
    rather than writing a null value into results.json."""
    with pytest.raises(SystemExit, match=r"'arc_easy': no 'f1' \(has acc,none\)"):
        generic.summarize_results(
            RESULTS, ["arc_easy", "hellaswag"], {"arc_easy": "f1", "hellaswag": "acc"}
        )
    # Without the task's filter names, a filtered-only row is a miss too.
    with pytest.raises(
        SystemExit, match="exact_match,flexible-extract, exact_match,strict-match"
    ):
        generic.summarize_results(
            {"results": {"gsm8k": GSM8K_ROW}}, ["gsm8k"], {"gsm8k": "exact_match"}
        )


def test_plan_passes_keeps_objects():
    """Requested groups reach simple_evaluate as the loaded Group objects
    (their overrides and aggregates apply), plain tasks as Task objects --
    never as names, which the harness would reload standalone."""
    arc = _task(name="arc_challenge")
    hs = _task(name="hellaswag")
    wino = _task(name="winogrande")
    group = _Group("mc_group", arc, hs)
    tasks = {"arc_challenge": arc, "hellaswag": hs, "winogrande": wino}
    mode_of = dict.fromkeys(tasks, "base")
    passes, split = generic.plan_passes(
        ["mc_group", "winogrande"], tasks, {"mc_group": group}, mode_of
    )
    assert split == []
    assert list(passes) == ["base"]
    assert passes["base"][0] is group
    assert passes["base"][1] is wino
    assert len(passes["base"]) == 2  # the group's leaves are not passed again


def test_plan_passes_splits_a_mixed_group():
    """openllm's shape: a group over multiple-choice tasks, a multiple-choice
    subgroup, and one generation task. Under mode=auto it cannot run as one
    pass: the uniform subgroup still goes whole (it aggregates), the leaves
    go as their (override-carrying) objects, the parent is reported split."""
    arc = _task(name="arc_challenge")
    anatomy = _task(name="mmlu_anatomy")
    astronomy = _task(name="mmlu_astronomy")
    gsm8k = _task("generate_until", "exact_match", name="gsm8k")
    mmlu = _Group("mmlu", anatomy, astronomy)
    openllm = _Group("openllm", arc, mmlu, gsm8k)
    tasks = {t.task_name: t for t in (arc, anatomy, astronomy, gsm8k)}
    groups = {"openllm": openllm, "mmlu": mmlu}
    mode_of = {**dict.fromkeys(tasks, "base"), "gsm8k": "chat"}
    passes, split = generic.plan_passes(["openllm"], tasks, groups, mode_of)
    assert split == ["openllm"]
    assert passes == {"base": [arc, mmlu], "chat": [gsm8k]}
    # mode=chat (or base) puts every leaf in one mode: the group runs whole.
    passes, split = generic.plan_passes(
        ["openllm"], tasks, groups, dict.fromkeys(tasks, "chat")
    )
    assert split == [] and passes == {"chat": [openllm]}


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


def _gsm8k(**kw):
    return _task(
        "generate_until",
        "exact_match",
        filters=["strict-match", "flexible-extract"],
        name="gsm8k",
        **kw,
    )


HELLASWAG = _task("multiple_choice", "acc", name="hellaswag")
ARC_EASY = _task("multiple_choice", "acc", name="arc_easy")
GSM8K = _gsm8k()


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
                "hellaswag": HELLASWAG,
                "arc_easy": ARC_EASY,
                "gsm8k": GSM8K,
            },
            {},
        ),
    )
    calls = []

    def fake_simple_evaluate(**kwargs):
        calls.append(kwargs)
        if GSM8K in kwargs["tasks"]:
            return {
                "results": {
                    "gsm8k": {
                        "exact_match,strict-match": 0.9,
                        "exact_match,flexible-extract": 0.1,
                    }
                },
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
    assert outcome["per_task"]["gsm8k"]["key"] == "exact_match,strict-match"
    assert len(outcome["per_repeat"]) == 1

    # Two passes, one per mode, chat flags only on the chat pass; each gets
    # the loaded Task objects, not names.
    assert [c["tasks"] for c in calls] == [[HELLASWAG, ARC_EASY], [GSM8K]]
    assert outcome["split_groups"] == []
    assert [c["apply_chat_template"] for c in calls] == [False, True]
    assert [c["fewshot_as_multiturn"] for c in calls] == [False, True]

    # Artifacts: merged lm_eval_results.json and per-task sample rows.
    keep = json.loads((tmp_path / "repeat-0" / "lm_eval_results.json").read_text())
    assert keep["leaf_tasks"] == ["arc_easy", "gsm8k", "hellaswag"]
    assert set(keep["results"]) == {"hellaswag", "arc_easy", "gsm8k", "mmlu"}
    rows = (tmp_path / "repeat-0" / "samples" / "gsm8k.jsonl").read_text().splitlines()
    assert json.loads(rows[0])["exact_match"] == 1.0


def test_run_passes_the_loaded_group(tmp_path, monkeypatch):
    """tasks=<group>: simple_evaluate gets the Group object the bridge
    loaded (whose children carry the group's overrides), so the harness
    computes the group's aggregate; the headline stays the leaf mean."""
    arc = _task(name="arc_challenge")
    arc.config.num_fewshot = 25  # the group's override, set at load time
    hs = _task(name="hellaswag")
    group = _Group("mc_group", arc, hs)
    monkeypatch.setattr(generic, "task_index", lambda: ["mc_group"])
    monkeypatch.setattr(
        generic,
        "load_tasks",
        lambda tasks: ({"arc_challenge": arc, "hellaswag": hs}, {"mc_group": group}),
    )
    calls = []

    def fake_simple_evaluate(**kwargs):
        calls.append(kwargs)
        (item,) = kwargs["tasks"]
        assert item is group
        assert item.children[0].config.num_fewshot == 25
        return {
            "results": {
                "mc_group": {"acc,none": 0.5},
                "arc_challenge": {"acc,none": 0.4},
                "hellaswag": {"acc,none": 0.8},
            },
            "groups": {"mc_group": {"acc,none": 0.5}},
            "n-samples": {
                "arc_challenge": {"effective": 3},
                "hellaswag": {"effective": 1},
            },
        }

    monkeypatch.setattr(generic, "run_simple_evaluate", fake_simple_evaluate)
    monkeypatch.setattr(
        generic.LMEval,
        "make_lm",
        lambda self, mode, base_url, served, seed: SimpleNamespace(mode=mode),
    )
    monkeypatch.setattr(chat_if, "drop_errors_from_cache", lambda cache_dir: 0)

    outcome = generic.LMEval(_config(tmp_path, tasks="mc_group")).run(
        "http://s/v1", "m"
    )
    assert len(calls) == 1
    assert outcome["value"] == pytest.approx((0.4 + 0.8) / 2)
    assert outcome["groups"] == ["mc_group"] and outcome["split_groups"] == []
    keep = json.loads((tmp_path / "repeat-0" / "lm_eval_results.json").read_text())
    assert keep["groups"] == {"mc_group": {"acc,none": 0.5}}


def test_thinking_budget_is_chat_ifs():
    assert generic.THINKING_BUDGET == chat_if.CARD_SAMPLING["max_tokens"] == 32768


def test_init_rejects_bad_thinking(tmp_path, monkeypatch):
    monkeypatch.setattr(generic, "task_index", lambda: ["gsm8k"])
    with pytest.raises(SystemExit, match="thinking must be one of on, off"):
        generic.LMEval(_config(tmp_path, tasks="gsm8k", thinking="no"))


def test_chat_gen_kwargs(tmp_path, monkeypatch):
    monkeypatch.setattr(generic, "task_index", lambda: ["gsm8k"])
    bench = generic.LMEval(_config(tmp_path, tasks="gsm8k"))
    assert bench.chat_gen_kwargs() is None
    bench = generic.LMEval(
        _config(tmp_path, tasks="gsm8k", thinking="off", gen_kwargs="temperature=0.6")
    )
    assert bench.chat_gen_kwargs() == {
        "temperature": 0.6,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _mixed_run(tmp_path, monkeypatch, tasks, samples=None, **options):
    """run() over hellaswag (base) and ``tasks`` (chat), simple_evaluate
    faked: returns the outcome and the simple_evaluate calls."""
    hs = _task("multiple_choice", "acc", name="hellaswag")
    objects = {"hellaswag": hs, **{t.task_name: t for t in tasks}}
    monkeypatch.setattr(generic, "task_index", lambda: list(objects))
    monkeypatch.setattr(generic, "load_tasks", lambda names: (dict(objects), {}))
    calls = []

    def fake_simple_evaluate(**kwargs):
        calls.append(kwargs)
        names = [t.task_name for t in kwargs["tasks"]]
        return {
            "results": {
                n: {"acc,none": 0.5, "exact_match,strict-match": 0.5} for n in names
            },
            "n-samples": {n: {"effective": 2} for n in names},
            "samples": {n: (samples or {}).get(n, []) for n in names},
        }

    monkeypatch.setattr(generic, "run_simple_evaluate", fake_simple_evaluate)
    monkeypatch.setattr(
        generic.LMEval,
        "make_lm",
        lambda self, mode, base_url, served, seed: SimpleNamespace(
            mode=mode, seed=seed
        ),
    )
    monkeypatch.setattr(chat_if, "drop_errors_from_cache", lambda cache_dir: 0)
    bench = generic.LMEval(_config(tmp_path, tasks=",".join(objects), **options))
    return bench.run("http://s/v1", "m"), calls


def test_thinking_off_reaches_the_chat_pass_only(tmp_path, monkeypatch):
    gsm8k = _gsm8k()
    outcome, calls = _mixed_run(tmp_path, monkeypatch, [gsm8k], thinking="off")
    by_mode = {c["model"].mode: c for c in calls}
    assert by_mode["chat"]["gen_kwargs"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert by_mode["base"]["gen_kwargs"] is None  # gen_kwargs never reach base mode
    assert gsm8k.config.generation_kwargs == {}  # no thinking budget when off
    assert outcome["thinking"] == "off"
    assert outcome["thinking_budget"]["tasks"] == []


def test_gen_kwargs_never_reach_base_mode(tmp_path, monkeypatch):
    _, calls = _mixed_run(
        tmp_path, monkeypatch, [_gsm8k()], gen_kwargs="temperature=0.6"
    )
    by_mode = {c["model"].mode: c for c in calls}
    assert by_mode["base"]["gen_kwargs"] is None
    assert by_mode["chat"]["gen_kwargs"] == {"temperature": 0.6}


def test_thinking_on_gives_the_budget(tmp_path, monkeypatch):
    """thinking on (the default): a chat task with no max_gen_toks of its
    own gets the thinking budget; a task that sets one keeps it; base tasks
    are untouched."""
    bare = _gsm8k()
    own = _task(
        "generate_until",
        "exact_match",
        name="own",
        filters=["strict-match"],
        generation_kwargs={"max_gen_toks": 64},
    )
    outcome, _ = _mixed_run(tmp_path, monkeypatch, [bare, own])
    assert bare.config.generation_kwargs == {"max_gen_toks": generic.THINKING_BUDGET}
    assert own.config.generation_kwargs == {"max_gen_toks": 64}
    assert outcome["thinking_budget"] == {"max_gen_toks": 32768, "tasks": ["gsm8k"]}


def test_gen_kwargs_max_gen_toks_wins_over_the_budget(tmp_path, monkeypatch):
    bare = _gsm8k()
    _, calls = _mixed_run(tmp_path, monkeypatch, [bare], gen_kwargs="max_gen_toks=1024")
    assert bare.config.generation_kwargs == {}
    assert {c["model"].mode: c for c in calls}["chat"]["gen_kwargs"] == {
        "max_gen_toks": 1024
    }


def test_response_counts():
    rows_chat = [
        {"resps": [["the answer"]]},
        {"resps": [[""]]},
        {"resps": [["   "]]},
        {"resps": [[chat_if.ERROR_MARKER]]},
    ]
    rows_base = [
        {"resps": [[(-1.0, True)], [(-2.0, False)]]},
        {"resps": [[(-1e9, False)], [(-2.0, False)]]},
        {"resps": [[[-1e9, False]], [[-2.0, False]]]},
    ]
    counts = generic.response_counts({"c": rows_chat, "b": rows_base}, {"c"})
    assert counts == {
        "c": {"docs": 4, "empty": 2, "errors": 1},
        "b": {"docs": 3, "empty": 0, "errors": 2},
    }


def test_empty_responses_fail_the_run(tmp_path, monkeypatch):
    """The cluster smoke's shape: every gsm8k response empty (the thought
    ran out of tokens, content null). Counted, and over max_failed_frac the
    run fails instead of reporting 0.0."""
    empty = {"gsm8k": [{"resps": [[""]]}, {"resps": [[""]]}]}
    with pytest.raises(SystemExit, match="responses_failed 2/2"):
        _mixed_run(tmp_path, monkeypatch, [_gsm8k()], samples=empty)


def test_few_empty_responses_are_recorded(tmp_path, monkeypatch):
    rows = [{"resps": [["#### 4"]]}] * 99 + [{"resps": [[""]]}]
    outcome, _ = _mixed_run(tmp_path, monkeypatch, [_gsm8k()], samples={"gsm8k": rows})
    assert outcome["responses"]["gsm8k"] == {"docs": 100, "empty": 1, "errors": 0}
    assert outcome["responses_failed"] == 1 and outcome["responses_total"] == 100
    assert outcome["incomplete"] is True
    with pytest.raises(SystemExit, match="above the allowed fraction 0.0"):
        _mixed_run(
            tmp_path / "strict",
            monkeypatch,
            [_gsm8k()],
            samples={"gsm8k": rows},
            max_failed_frac="0",
        )
