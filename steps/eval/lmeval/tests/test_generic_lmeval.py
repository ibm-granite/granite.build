# SPDX-License-Identifier: Apache-2.0

"""The lm-eval bridge's chat path against the real lm-eval (skipped where
lm-eval is not installed; it runs in the lmeval image build): a local
gsm8k-shaped task (gsm8k.yaml's prompt, stops, filters and metric over a local
json dataset) and a stub of vLLM's chat endpoint serving a thinking model
behind a reasoning parser -- the thought, then </think>, then the answer;
max_tokens and stop strings apply to the whole text; no </think> in the
output means content null (the Granite 4.2 cluster smoke's 0.0)."""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("lm_eval")

from granite_evals.benchmarks import generic  # noqa: E402
from granite_evals.registry import RunConfig  # noqa: E402

THINK_TOKENS = 300

TASK_YAML = """task: gsm8k_local
dataset_path: json
dataset_kwargs:
  data_files:
    train: {data}/train.jsonl
    test: {data}/test.jsonl
output_type: generate_until
training_split: train
fewshot_split: train
test_split: test
doc_to_text: "Question: {{{{question}}}}\\nAnswer:"
doc_to_target: "{{{{answer}}}}"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: false
    regexes_to_ignore: [",", "\\\\$", "(?s).*#### ", "\\\\.$"]
generation_kwargs:
  until: {until}
  do_sample: false
  temperature: 0.0
{extra}num_fewshot: 2
filter_list:
  - name: "strict-match"
    filter:
      - function: "regex"
        regex_pattern: "#### (\\\\-?[0-9\\\\.\\\\,]+)"
      - function: "take_first"
"""


class _Stub(BaseHTTPRequestHandler):
    log: list = []

    def do_POST(self):  # noqa: N802 - http.server's name
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.log.append(body)
        thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking", True)
        q = body["messages"][-1]["content"]
        m = re.search(r"has (\d+) apples and (\d+) pears", q)
        ans = int(m.group(1)) + int(m.group(2)) if m else 0
        thought = ["hmm"] * (THINK_TOKENS if thinking else 0)
        tokens = thought + ["</think>"] + f"The answer is {ans}.\n#### {ans}".split(" ")
        max_tokens = body.get("max_tokens", 16)
        text = " ".join(tokens[:max_tokens])
        for stop in body.get("stop") or []:
            if (i := text.find(stop)) >= 0:
                text = text[:i]
        content = text.split("</think>", 1)[1].lstrip() if "</think>" in text else None
        out = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _Stub.log = []
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()


QUESTION_STOP = '["Question:", "</s>", "<|im_end|>"]'  # gsm8k's own
SPECIAL_STOPS = '["</s>", "<|im_end|>"]'


def _bridge(tmp_path, monkeypatch, own_max_gen_toks=None, until=SPECIAL_STOPS, **options):
    data, tasks = tmp_path / "data", tmp_path / "tasks"
    data.mkdir()
    tasks.mkdir()
    for split, n in (("train", 4), ("test", 3)):
        with open(data / f"{split}.jsonl", "w") as f:
            for i in range(n):
                a, b = 10 + i, 20 + i
                f.write(
                    json.dumps(
                        {
                            "question": f"Tom has {a} apples and {b} pears; how many fruits?",
                            "answer": f"{a}+{b}={a + b}\n#### {a + b}",
                        }
                    )
                    + "\n"
                )
    extra = f"  max_gen_toks: {own_max_gen_toks}\n" if own_max_gen_toks else ""
    (tasks / "gsm8k_local.yaml").write_text(
        TASK_YAML.format(data=data, extra=extra, until=until)
    )

    def load_tasks(names):
        from lm_eval.tasks import TaskManager

        loaded = TaskManager(include_path=str(tasks)).load(names)
        return dict(loaded["tasks"]), dict(loaded["groups"])

    monkeypatch.setattr(generic, "task_index", lambda: ["gsm8k_local"])
    monkeypatch.setattr(generic, "load_tasks", load_tasks)
    # A Granite 4.x template: an enable_thinking switch (the stub reads it).
    monkeypatch.setattr(
        generic,
        "chat_template_thinking",
        lambda model: {
            "switch": "enable_thinking",
            "switches": {"enable_thinking": {"default": "on"}},
            "think_tokens": [],
            "error": None,
        },
    )
    config = RunConfig(
        model="m",
        output_dir=tmp_path / "out",
        workers=2,
        options={
            "tasks": "gsm8k_local",
            "metric": "exact_match,strict-match",
            **options,
        },
    )
    return generic.LMEval(config)


def test_thinking_on_gets_the_budget_and_answers(tmp_path, monkeypatch, stub):
    outcome = _bridge(tmp_path, monkeypatch, thinking="on").run(stub, "granite")
    assert all(
        r.get("chat_template_kwargs") == {"enable_thinking": True} for r in _Stub.log
    )
    assert {r["max_tokens"] for r in _Stub.log} == {generic.THINKING_BUDGET}
    assert outcome["value"] == pytest.approx(1.0)
    assert outcome["responses"]["gsm8k_local"] == {"docs": 3, "empty": 0, "errors": 0}


def test_thinking_off_sends_enable_thinking_false(tmp_path, monkeypatch, stub):
    # off is the default, and the task's own Question: stop is fine without a thought
    outcome = _bridge(tmp_path, monkeypatch, until=QUESTION_STOP).run(stub, "granite")
    assert all(
        r.get("chat_template_kwargs") == {"enable_thinking": False} for r in _Stub.log
    )
    assert outcome["value"] == pytest.approx(1.0)


def test_a_cut_thought_fails_the_run(tmp_path, monkeypatch, stub):
    """The task's own small max_gen_toks with thinking on: every content is
    null, lm-eval scores "", and the run fails instead of reporting 0.0."""
    bench = _bridge(tmp_path, monkeypatch, own_max_gen_toks=16, thinking="on")
    with pytest.raises(SystemExit, match="responses_failed 3/3"):
        bench.run(stub, "granite")
    rows = (tmp_path / "out" / "repeat-0" / "samples" / "gsm8k_local.jsonl").read_text()
    assert all(json.loads(r)["resps"] == [[""]] for r in rows.splitlines())


def test_thinking_on_with_the_tasks_question_stop_is_refused(tmp_path, monkeypatch):
    # The real harness's view of the task's until, before any server starts.
    with pytest.raises(SystemExit, match=r"thinking=on, but gsm8k_local stops on \['Question:'\]"):
        _bridge(tmp_path, monkeypatch, until=QUESTION_STOP, thinking="on")


SHUFFLE_YAML = """task: shuffle_local
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data}
output_type: multiple_choice
test_split: test
process_docs: !function utils.process_docs
doc_to_text: "{{{{question}}}}"
doc_to_choice: "{{{{choices}}}}"
doc_to_target: label
metric_list:
  - metric: acc
"""

# gpqa's own shape: the choices shuffled with random inside a datasets map.
SHUFFLE_UTILS = """import random


def process_docs(dataset):
    def _shuffle(doc):
        choices = [doc["right"], doc["w1"], doc["w2"], doc["w3"]]
        random.shuffle(choices)
        return {"choices": choices, "label": choices.index(doc["right"])}

    return dataset.map(_shuffle)
"""


def _shuffle_task(tmp_path):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    data = tmp_path / "q.jsonl"
    with open(data, "w") as f:
        for i in range(8):
            row = {"question": f"q{i}", "right": f"r{i}", "w1": f"a{i}", "w2": f"b{i}", "w3": f"c{i}"}
            f.write(json.dumps(row) + "\n")
    (tasks / "shuffle_local.yaml").write_text(SHUFFLE_YAML.format(data=data))
    (tasks / "utils.py").write_text(SHUFFLE_UTILS)
    return tasks


def _orders(tmp_path, monkeypatch, runs):
    """For each HF cache: a warm-up build (seed 99) that prepares the data
    and, with caching on, writes the map cache; then one build per seed in
    ``runs[cache]``. Returns {(cache, seed): the docs' choice orders}."""
    datasets = pytest.importorskip("datasets")
    tasks = _shuffle_task(tmp_path)
    out = {}
    for cache, seeds in runs.items():
        monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", str(tmp_path / cache))
        for seed in (99, *seeds):
            generic.seed_like_simple_evaluate(seed)
            objs, _ = generic.load_tasks(["shuffle_local"], include_path=str(tasks))
            task = objs["shuffle_local"]
            out[(cache, seed)] = [doc["choices"] for doc in task.test_docs()]
            # the same order every time the harness reads the docs in this run
            assert [doc["choices"] for doc in task.eval_docs] == out[(cache, seed)]
    return out


def test_a_shuffled_order_follows_the_seed_not_the_datasets_cache(tmp_path, monkeypatch):
    """Runs on one HF_HOME in either order: each gets its own seed's order,
    not the one an earlier run left in the datasets map cache (gpqa's choice
    order, 2026-11-27 review)."""
    datasets = pytest.importorskip("datasets")
    o = _orders(tmp_path, monkeypatch, {"a": (1, 0), "b": (0, 1)})
    assert o[("a", 0)] != o[("a", 1)]  # the seeds give different orders
    assert o[("a", 0)] == o[("b", 0)] and o[("a", 1)] == o[("b", 1)]
    assert datasets.is_caching_enabled()  # restored after each build


def test_without_fresh_maps_the_first_run_fixes_the_order(tmp_path, monkeypatch):
    # The harness alone: the warm-up run's map cache decides every later order.
    monkeypatch.setattr(generic, "fresh_dataset_maps", __import__("contextlib").nullcontext)
    monkeypatch.setattr(generic, "freeze_eval_docs", lambda task: None)
    o = _orders(tmp_path, monkeypatch, {"a": (1, 0)})
    assert o[("a", 99)] == o[("a", 1)] == o[("a", 0)]


def test_the_real_tasks_mean_metric_is_recognised(tmp_path, monkeypatch):
    # The leave-out rescoring needs the harness's resolved aggregation.
    bench = _bridge(tmp_path, monkeypatch)
    assert generic.is_mean_metric(bench.task_objects["gsm8k_local"], "exact_match")
