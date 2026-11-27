# SPDX-License-Identifier: Apache-2.0

"""generic lm-eval bridge tests: fakes only (no GPU, no network, no server,
no lm-eval install -- the harness surface is faked at the module's seams)."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
from granite_evals.benchmarks import chat_if, generic, include
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


REAL_PROBE = generic.chat_template_thinking  # the autouse fixture fakes it


@pytest.fixture(autouse=True)
def _fake_harness_load(monkeypatch):
    """The bridge loads the selected tasks when it is constructed (before
    the server starts). By default each name loads as a multiple-choice
    task; a test that needs other tasks fakes load_tasks itself."""
    monkeypatch.setattr(
        generic,
        "load_tasks",
        lambda names: ({name: _task(name=name) for name in names}, {}),
    )
    monkeypatch.setattr(
        generic, "chat_template_thinking", lambda model: dict(GRANITE_PROBE)
    )


GRANITE_PROBE = {
    "switch": "enable_thinking",
    "switches": {"enable_thinking": {"default": "on"}},
    "think_tokens": [],
    "error": None,
}
NO_SWITCH_PROBE = {
    "switch": None,
    "switches": {},
    "think_tokens": ["<think>"],
    "error": None,
}


def test_tasks_are_loaded_and_checked_before_the_server_starts(tmp_path, monkeypatch):
    # Construction is what the CLI does before it starts vLLM: the tasks are
    # loaded and their protocol resolved there, so run() only serves.
    monkeypatch.setattr(generic, "task_index", lambda: ["hellaswag"])
    loaded = []
    monkeypatch.setattr(
        generic,
        "load_tasks",
        lambda names: loaded.append(names)
        or ({"hellaswag": _task(name="hellaswag")}, {}),
    )
    bench = generic.LMEval(_config(tmp_path, tasks="hellaswag"))
    assert loaded == [["hellaswag"]]
    assert bench.mode_groups == {"base": ["hellaswag"]}


@pytest.mark.parametrize(
    "kind", ["multiple_choice", "loglikelihood", "loglikelihood_rolling"]
)
def test_mode_chat_on_a_loglikelihood_task_is_refused_at_startup(
    tmp_path, monkeypatch, kind
):
    monkeypatch.setattr(generic, "task_index", lambda: ["gsm8k", "hellaswag"])
    monkeypatch.setattr(
        generic,
        "load_tasks",
        lambda names: (
            {"gsm8k": _gsm8k(), "hellaswag": _task(kind, name="hellaswag")},
            {},
        ),
    )
    with pytest.raises(SystemExit) as e:
        generic.LMEval(_config(tmp_path, tasks="gsm8k,hellaswag", mode="chat"))
    msg = str(e.value)
    assert "mode=chat runs generation tasks only" in msg and "hellaswag" in msg
    assert "gsm8k" not in msg.split("but", 1)[1]
    assert "mode=auto" in msg and "mode=base" in msg


def test_a_task_without_a_metric_is_refused_at_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(generic, "task_index", lambda: ["x"])
    task = _task(name="x")
    task.config.metric_list = []
    monkeypatch.setattr(generic, "load_tasks", lambda names: ({"x": task}, {}))
    with pytest.raises(SystemExit, match="declares no metric_list"):
        generic.LMEval(_config(tmp_path, tasks="x"))


def test_mode_chat_on_generation_tasks_is_fine(tmp_path, monkeypatch):
    monkeypatch.setattr(generic, "task_index", lambda: ["gsm8k"])
    monkeypatch.setattr(generic, "load_tasks", lambda names: ({"gsm8k": _gsm8k()}, {}))
    bench = generic.LMEval(_config(tmp_path, tasks="gsm8k", mode="chat"))
    assert bench.mode_groups == {"chat": ["gsm8k"]}


def test_no_message_recommends_mode_chat_for_a_mixed_group():
    # mode=chat cannot run a group with multiple-choice tasks; the docs and
    # messages point at mode=base.
    import inspect

    src = inspect.getsource(generic)
    assert "mode=chat or mode=base" not in src
    assert "``mode=chat`` or ``mode=base``" not in src


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
    assert generic.resolve_modes(types, "base") == {"base": ["a", "b", "c"]}
    assert generic.resolve_modes({"a": "generate_until"}, "chat") == {"chat": ["a"]}
    with pytest.raises(SystemExit, match="generation tasks only"):
        generic.resolve_modes(types, "chat")
    with pytest.raises(SystemExit, match="mode must be one of"):
        generic.resolve_modes(types, "both")
    with pytest.raises(SystemExit, match="no auto protocol; set mode=base"):
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
            mode="base",
            num_fewshot="5",
            gen_kwargs="temperature=0.6",
            metric="acc_norm",
        )
    )
    assert bench.task_list == ["hellaswag", "gsm8k"]
    assert bench.mode == "base"
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
    monkeypatch.setattr(generic, "drop_failed_requests", lambda cache_dir: 0)

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
    monkeypatch.setattr(generic, "drop_failed_requests", lambda cache_dir: 0)

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


def test_thinking_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(generic, "task_index", lambda: ["gsm8k"])
    monkeypatch.setattr(generic, "load_tasks", lambda names: ({"gsm8k": _gsm8k()}, {}))
    bench = generic.LMEval(_config(tmp_path, tasks="gsm8k"))
    assert bench.thinking == "off"
    assert bench.chat_gen_kwargs() == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    bench = generic.LMEval(
        _config(tmp_path, tasks="gsm8k", thinking="on", gen_kwargs="temperature=0.6")
    )
    assert bench.chat_gen_kwargs() == {
        "temperature": 0.6,
        "chat_template_kwargs": {"enable_thinking": True},
    }


@pytest.mark.parametrize(
    ("probe", "thinking", "kwargs"),
    [
        (GRANITE_PROBE, "off", {"enable_thinking": False}),
        (
            {
                "switch": "thinking",
                "switches": {"thinking": {"default": "off"}},
                "think_tokens": [],
                "error": None,
            },
            "on",
            {"thinking": True},
        ),
        (NO_SWITCH_PROBE, "off", {}),
        (
            {"switch": None, "switches": {}, "think_tokens": [], "error": "OSError: x"},
            "off",
            {"enable_thinking": False},
        ),
        (
            {"switch": None, "switches": {}, "think_tokens": [], "error": "OSError: x"},
            "on",
            {"enable_thinking": True},
        ),
    ],
)
def test_requests_set_the_detected_switch(
    tmp_path, monkeypatch, probe, thinking, kwargs
):
    monkeypatch.setattr(generic, "task_index", lambda: ["gsm8k"])
    monkeypatch.setattr(generic, "load_tasks", lambda names: ({"gsm8k": _gsm8k()}, {}))
    monkeypatch.setattr(generic, "chat_template_thinking", lambda model: dict(probe))
    bench = generic.LMEval(_config(tmp_path, tasks="gsm8k", thinking=thinking))
    assert bench.thinking_kwargs() == kwargs
    assert bench.chat_gen_kwargs() == (
        {"chat_template_kwargs": kwargs} if kwargs else None
    )


@pytest.mark.parametrize(
    ("probe", "options", "line"),
    [
        (
            GRANITE_PROBE,
            {},
            "thinking=off (default); this checkpoint's chat template has a thinking switch "
            "(enable_thinking), pass thinking=on to use it",
        ),
        (
            GRANITE_PROBE,
            {"thinking": "off"},
            "thinking=off; this checkpoint's chat template has a thinking switch",
        ),
        (
            NO_SWITCH_PROBE,
            {},
            "thinking=off (default); this checkpoint's chat template has no thinking switch "
            "(its tokenizer has <think>, but the template renders the same either way)",
        ),
        (
            GRANITE_PROBE,
            {"thinking": "on"},
            "thinking=on; requests set chat_template_kwargs.enable_thinking=true",
        ),
        (
            {
                "switch": None,
                "switches": {},
                "think_tokens": [],
                "error": "OSError: no such dir",
            },
            {},
            "thinking=off (default); could not read this checkpoint's chat template "
            "(OSError: no such dir)",
        ),
    ],
)
def test_the_startup_status_line(tmp_path, monkeypatch, caplog, probe, options, line):
    monkeypatch.setattr(generic, "task_index", lambda: ["gsm8k"])
    monkeypatch.setattr(generic, "load_tasks", lambda names: ({"gsm8k": _gsm8k()}, {}))
    monkeypatch.setattr(generic, "chat_template_thinking", lambda model: dict(probe))
    with caplog.at_level("INFO", logger=generic.log.name):
        bench = generic.LMEval(_config(tmp_path, tasks="gsm8k", **options))
    assert line in caplog.text
    assert bench.thinking_switch == probe


def test_no_chat_task_no_probe(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(generic, "task_index", lambda: ["hellaswag"])
    probed = []
    monkeypatch.setattr(
        generic, "chat_template_thinking", lambda model: probed.append(model)
    )
    with caplog.at_level("INFO", logger=generic.log.name):
        bench = generic.LMEval(_config(tmp_path, tasks="hellaswag"))
    assert probed == [] and bench.thinking_switch is None
    assert "thinking does not apply" in caplog.text
    with pytest.raises(SystemExit, match="thinking=on applies to chat-mode tasks"):
        generic.LMEval(_config(tmp_path, tasks="hellaswag", thinking="on"))


def test_thinking_on_without_a_switch_is_refused_at_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(generic, "task_index", lambda: ["gsm8k"])
    monkeypatch.setattr(generic, "load_tasks", lambda names: ({"gsm8k": _gsm8k()}, {}))
    monkeypatch.setattr(
        generic, "chat_template_thinking", lambda model: dict(NO_SWITCH_PROBE)
    )
    with pytest.raises(SystemExit, match="has no thinking switch.*nothing to turn on"):
        generic.LMEval(_config(tmp_path, tasks="gsm8k", thinking="on"))


class _FakeTokenizer:
    """A chat template as transformers renders it: Granite 4.x-style (an
    enable_thinking switch, on by default), Granite 3.x-style (thinking,
    off by default), or none."""

    def __init__(self, style):
        self.style = style

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kw):
        assert tokenize is False and add_generation_prompt is True
        text = (
            "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
            + "<|assistant|>"
        )
        if self.style == "granite4" and kw.get("enable_thinking", True):
            text += "<think>"
        if self.style == "granite3" and kw.get("thinking", False):
            text = "<|system|>think first" + text
        return text

    def get_vocab(self):
        return {"<think>": 1, "</think>": 2} if self.style != "plain" else {"a": 1}


@pytest.mark.parametrize(
    ("style", "switch", "switches", "tokens"),
    [
        (
            "granite4",
            "enable_thinking",
            {"enable_thinking": {"default": "on"}},
            ["<think>", "</think>"],
        ),
        (
            "granite3",
            "thinking",
            {"thinking": {"default": "off"}},
            ["<think>", "</think>"],
        ),
        ("tokens-only", None, {}, ["<think>", "</think>"]),
        ("plain", None, {}, []),
    ],
)
def test_chat_template_thinking_renders_the_template(
    monkeypatch, style, switch, switches, tokens
):
    import sys
    from types import ModuleType

    fake = ModuleType("transformers")
    loaded = []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model, trust_remote_code):
            loaded.append(model)
            return _FakeTokenizer(style)

    fake.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake)
    probe = REAL_PROBE("/models/x")
    assert loaded == ["/models/x"]
    assert probe == {
        "switch": switch,
        "switches": switches,
        "think_tokens": tokens,
        "error": None,
    }


def test_chat_template_thinking_never_raises(monkeypatch):
    import sys
    from types import ModuleType

    fake = ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model, trust_remote_code):
            raise OSError("not a model dir")

    fake.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake)
    probe = REAL_PROBE("x")
    assert probe["switch"] is None and probe["error"] == "OSError: not a model dir"


def _mixed_run(tmp_path, monkeypatch, tasks, samples=None, repeats=None, **options):
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
    if not getattr(generic.drop_failed_requests, "_recorder", False):
        monkeypatch.setattr(generic, "drop_failed_requests", lambda cache_dir: 0)
    config = RunConfig(
        model="m",
        output_dir=tmp_path,
        repeats=repeats,
        options={"tasks": ",".join(objects), **options},
    )
    bench = generic.LMEval(config)
    return bench.run("http://s/v1", "m"), calls


def test_thinking_off_reaches_the_chat_pass_only(tmp_path, monkeypatch):
    gsm8k = _gsm8k()
    outcome, calls = _mixed_run(tmp_path, monkeypatch, [gsm8k])  # off: the default
    by_mode = {c["model"].mode: c for c in calls}
    assert by_mode["chat"]["gen_kwargs"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert outcome["thinking_switch"] == GRANITE_PROBE
    assert outcome["thinking_kwargs"] == {"enable_thinking": False}
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
    assert by_mode["chat"]["gen_kwargs"] == {
        "temperature": 0.6,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_thinking_on_gives_the_budget(tmp_path, monkeypatch):
    """thinking on: a chat task with no max_gen_toks of its
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
    outcome, _ = _mixed_run(tmp_path, monkeypatch, [bare, own], thinking="on")
    assert bare.config.generation_kwargs == {"max_gen_toks": generic.THINKING_BUDGET}
    assert own.config.generation_kwargs == {"max_gen_toks": 64}
    assert outcome["thinking_budget"] == {"max_gen_toks": 32768, "tasks": ["gsm8k"]}


@pytest.mark.parametrize("where", ["gen_kwargs", "task"])
def test_a_max_tokens_of_its_own_wins_over_the_budget(tmp_path, monkeypatch, where):
    # lm-eval sends max_tokens over max_gen_toks: the budget would not apply,
    # so it is neither set nor recorded.
    bare = _gsm8k(generation_kwargs={"max_tokens": 512} if where == "task" else None)
    options = {"gen_kwargs": "max_tokens=512"} if where == "gen_kwargs" else {}
    outcome, _ = _mixed_run(tmp_path, monkeypatch, [bare], thinking="on", **options)
    assert "max_gen_toks" not in bare.config.generation_kwargs
    assert outcome["thinking_budget"]["tasks"] == []


def test_gen_kwargs_max_gen_toks_wins_over_the_budget(tmp_path, monkeypatch):
    bare = _gsm8k()
    _, calls = _mixed_run(
        tmp_path, monkeypatch, [bare], gen_kwargs="max_gen_toks=1024", thinking="on"
    )
    assert bare.config.generation_kwargs == {}
    assert {c["model"].mode: c for c in calls}["chat"]["gen_kwargs"] == {
        "max_gen_toks": 1024,
        "chat_template_kwargs": {"enable_thinking": True},
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


def test_response_counts_count_each_doc_once():
    """lm-eval logs one sample row per doc per filter (gsm8k: strict-match and
    flexible-extract, the same raw resps): the second gsm8k cluster smoke
    reported docs 20 for --limit 10. Each doc counts once, by doc_id."""
    rows = []
    for doc_id, resp in enumerate(["#### 18", "", chat_if.ERROR_MARKER]):
        for filt in ("strict-match", "flexible-extract"):
            rows.append({"doc_id": doc_id, "filter": filt, "resps": [[resp]]})
    assert generic.response_counts({"gsm8k": rows}, {"gsm8k"}) == {
        "gsm8k": {"docs": 3, "empty": 1, "errors": 1}
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


def test_text_stops():
    assert generic.text_stops(["Question:", "</s>", "<|im_end|>"]) == ["Question:"]
    assert generic.text_stops("Q:") == ["Q:"]
    assert generic.text_stops(["<s>", "<|eot_id|>", ""]) == []
    assert generic.text_stops(None) == []
    # only a whole special token is one: text around it is still text
    assert generic.text_stops(["x</s>", "<|a|>b"]) == ["x</s>", "<|a|>b"]


GSM8K_UNTIL = {"until": ["Question:", "</s>", "<|im_end|>"]}


@pytest.mark.parametrize(
    ("until", "gen_kwargs", "named"),
    [
        (GSM8K_UNTIL, "", "gsm8k stops on ['Question:']"),
        ({"until": ["</s>"]}, "until=Q:", "gsm8k stops on ['Q:']"),
    ],
)
def test_thinking_on_with_text_stops_is_refused_at_startup(
    tmp_path, monkeypatch, until, gen_kwargs, named
):
    """The 2026-11-27 smoke's shape (gsm8k stops on Question:): refused at
    construction, before the server starts, naming the stops and thinking=off."""
    monkeypatch.setattr(generic, "task_index", lambda: ["gsm8k"])
    gsm8k = _gsm8k(generation_kwargs=dict(until))
    monkeypatch.setattr(generic, "load_tasks", lambda names: ({"gsm8k": gsm8k}, {}))
    options = {"gen_kwargs": gen_kwargs} if gen_kwargs else {}
    with pytest.raises(SystemExit) as e:
        generic.LMEval(_config(tmp_path, tasks="gsm8k", thinking="on", **options))
    msg = str(e.value)
    assert msg.startswith("lm-eval: thinking=on, but " + named)
    assert "Use thinking=off" in msg and "gen_kwargs=until=" in msg


@pytest.mark.parametrize(
    ("options", "ok"),
    [
        ({}, True),
        ({"thinking": "off"}, True),
        ({"thinking": "on", "gen_kwargs": "until=</s>"}, True),
    ],
)
def test_text_stops_are_fine_with_thinking_off_or_special_stops(
    tmp_path, monkeypatch, options, ok
):
    monkeypatch.setattr(generic, "task_index", lambda: ["gsm8k"])
    gsm8k = _gsm8k(generation_kwargs=dict(GSM8K_UNTIL))
    monkeypatch.setattr(generic, "load_tasks", lambda names: ({"gsm8k": gsm8k}, {}))
    generic.LMEval(_config(tmp_path, tasks="gsm8k", **options))  # no error


def test_the_old_stop_hint_is_gone():
    # Refused at startup instead of diagnosed after the run.
    assert not hasattr(generic, "thinking_stop_hint")


@pytest.mark.parametrize("seed", [0, 7])
def test_tasks_are_built_after_seeding(tmp_path, monkeypatch, seed):
    """gpqa's process_docs shuffles each question's choices with random at
    task construction: the bridge seeds random as simple_evaluate does
    (the run's seed) before load_tasks, so every run draws the same order."""
    drawn = []

    def load_tasks(names):
        drawn.append(random.random())
        return {"hellaswag": _task(name="hellaswag")}, {}

    monkeypatch.setattr(generic, "task_index", lambda: ["hellaswag"])
    monkeypatch.setattr(generic, "load_tasks", load_tasks)
    monkeypatch.setattr(
        generic,
        "run_simple_evaluate",
        lambda **kw: {"results": {"hellaswag": {"acc,none": 1.0}}},
    )
    monkeypatch.setattr(
        generic.LMEval, "make_lm", lambda self, mode, base_url, served, seed: None
    )
    monkeypatch.setattr(generic, "drop_failed_requests", lambda cache_dir: 0)
    config = RunConfig(
        model="m", output_dir=tmp_path, seed=seed, options={"tasks": "hellaswag"}
    )
    random.seed(12345)  # whatever state the process is in
    generic.LMEval(config).run("http://s/v1", "m")
    assert drawn == [random.Random(seed).random()]


@pytest.mark.parametrize("split", ["test", "validation"])
def test_freeze_eval_docs_computes_them_once(split):
    calls = []

    class Task:
        def has_test_docs(self):
            return split == "test"

        def has_validation_docs(self):
            return True

        def test_docs(self):
            calls.append("test")
            return [random.random()]

        def validation_docs(self):
            calls.append("validation")
            return [random.random()]

    task = Task()
    generic.freeze_eval_docs(task)
    first = getattr(task, f"{split}_docs")()
    assert getattr(task, f"{split}_docs")() is first
    assert calls == [split]


def test_seed_like_simple_evaluate_seeds_numpy_too():
    numpy = pytest.importorskip("numpy")
    generic.seed_like_simple_evaluate(3)
    first = (random.random(), numpy.random.random())
    generic.seed_like_simple_evaluate(3)
    assert (random.random(), numpy.random.random()) == first


def test_failed_requests_are_dropped_in_both_modes(tmp_path, monkeypatch):
    """Each repeat's cache cleanup drops chat_if's ERROR_MARKER and
    include's ERROR_LL answers, so a resumed run retries them."""
    seen = []
    monkeypatch.setattr(
        chat_if, "drop_errors_from_cache", lambda d: seen.append(("chat", d)) or 1
    )
    monkeypatch.setattr(
        include, "drop_errors_from_cache", lambda d: seen.append(("base", d)) or 2
    )
    assert generic.drop_failed_requests(tmp_path) == 3
    assert seen == [("chat", tmp_path), ("base", tmp_path)]

    cleaned = []

    def recorder(cache_dir):
        cleaned.append(cache_dir)
        return 0

    recorder._recorder = True
    monkeypatch.setattr(generic, "drop_failed_requests", recorder)
    _mixed_run(tmp_path / "run", monkeypatch, [_gsm8k()], repeats=2)
    assert cleaned == [
        tmp_path / "run" / f"repeat-{k}" / "lm_cache" / generic.cache_key("m", "m", 1234 + k)
        for k in range(2)
    ]


def test_drop_failed_requests_on_a_real_cache(tmp_path):
    sqlitedict = pytest.importorskip("sqlitedict")
    with sqlitedict.SqliteDict(str(tmp_path / "cache_rank0.db"), autocommit=True) as db:
        db["ll"] = include.ERROR_LL
        db["gen"] = chat_if.ERROR_MARKER
        db["ok"] = (-1.5, True)
    assert generic.drop_failed_requests(tmp_path) == 2
    with sqlitedict.SqliteDict(str(tmp_path / "cache_rank0.db")) as db:
        assert dict(db) == {"ok": (-1.5, True)}


def test_primary_metric_is_the_first_metric_list_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(generic, "task_index", lambda: ["gpqa"])
    bench = generic.LMEval(_config(tmp_path, tasks="gpqa"))
    task = _task(metrics=["acc", "acc_norm"], name="gpqa")
    assert bench.primary_metrics({"gpqa": task}) == {"gpqa": "acc"}


def _repeated_run(tmp_path, monkeypatch, values, **config):
    """run() over one base task with simple_evaluate faked to answer the
    k-th call with values[k]; returns the outcome and the calls."""
    task = _task(name="hellaswag")
    monkeypatch.setattr(generic, "task_index", lambda: ["hellaswag"])
    monkeypatch.setattr(generic, "load_tasks", lambda names: ({"hellaswag": task}, {}))
    calls = []

    def fake_simple_evaluate(**kwargs):
        calls.append(kwargs)
        return {
            "results": {"hellaswag": {"acc,none": values[len(calls) - 1]}},
            "n-samples": {"hellaswag": {"effective": 10}},
        }

    monkeypatch.setattr(generic, "run_simple_evaluate", fake_simple_evaluate)
    monkeypatch.setattr(
        generic.LMEval,
        "make_lm",
        lambda self, mode, base_url, served, seed: SimpleNamespace(
            mode=mode, seed=seed
        ),
    )
    monkeypatch.setattr(generic, "drop_failed_requests", lambda cache_dir: 0)
    run_config = RunConfig(
        model=config.pop("model", "m"),
        output_dir=tmp_path,
        options={"tasks": "hellaswag", **config.pop("options", {})},
        **config,
    )
    return generic.LMEval(run_config).run("http://s/v1", "m"), calls


def test_repeats_average_and_reseed(tmp_path, monkeypatch):
    """repeats=k: k passes, each with its own LM seed (seed + 1234 + k) and
    the run's random_seed; the headline averages all k."""
    outcome, calls = _repeated_run(
        tmp_path, monkeypatch, [0.2, 0.4, 0.9], repeats=3, seed=5
    )
    assert outcome["value"] == pytest.approx((0.2 + 0.4 + 0.9) / 3)
    assert [r["value"] for r in outcome["per_repeat"]] == [0.2, 0.4, 0.9]
    assert [c["model"].seed for c in calls] == [5 + 1234, 5 + 1234 + 1, 5 + 1234 + 2]
    assert {c["random_seed"] for c in calls} == {5}
    assert [c["use_cache"] for c in calls] == [
        str(
            tmp_path
            / f"repeat-{k}"
            / "lm_cache"
            / generic.cache_key("m", "m", 5 + 1234 + k)
            / "cache"
        )
        for k in range(3)
    ]


def test_the_response_cache_is_keyed_on_model_and_seed(tmp_path, monkeypatch):
    """The harness keys cached responses by request only: a second model, or
    another seed, in the same run dir must not replay the first one's."""
    dirs = []

    def run(model, seed, served="m"):
        _, calls = _repeated_run(tmp_path, monkeypatch, [0.5], model=model, seed=seed)
        (call,) = calls
        dirs.append(Path(call["use_cache"]).parent)
        return dirs[-1]

    a = run("org/model-a", 0)
    assert run("org/model-a", 0) == a  # a resume finds its cache
    b = run("org/model-b", 0)
    c = run("org/model-a", 1)
    assert len({a, b, c}) == 3
    assert all(d.parent == tmp_path / "repeat-0" / "lm_cache" for d in (a, b, c))
    assert a.name == generic.cache_key("org/model-a", "m", 0 + 1234)


def test_limit_and_num_fewshot_reach_the_harness(tmp_path, monkeypatch):
    _, calls = _repeated_run(
        tmp_path, monkeypatch, [0.5], limit=7, options={"num_fewshot": "3"}
    )
    (call,) = calls
    assert call["limit"] == 7
    assert call["num_fewshot"] == 3
    _, calls = _repeated_run(tmp_path / "default", monkeypatch, [0.5])
    assert calls[0]["limit"] is None and calls[0]["num_fewshot"] is None
