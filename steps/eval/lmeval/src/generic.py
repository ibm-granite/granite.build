# SPDX-License-Identifier: Apache-2.0

"""LMEval family: the generic bridge to any task in the pinned harness.

lm-eval
-------
One registered id for every task lm-evaluation-harness ships (14k+ task
names at the pinned 0.4.13, plus groups): ``granite-evals run lm-eval
--option tasks=hellaswag,arc_easy``. The curated benchmarks in this family
(``mmlu-prox-lite``, ``include-44``) stay as they are -- fixed protocols,
curated aggregation; this one is the escape hatch that makes a task a
recipe's config, not a module. It exists so the other harnesses can grow
the same shape: one generic id per family whose options carry the
harness's own selection surface (see steps/eval/README.md, "The generic
bridge").

Options (all through ``--option`` / the step's ``options`` config):

* ``tasks`` (required): comma-separated lm-eval task or group names, as
  lm-evaluation-harness's own task index knows them. Unknown names fail
  fast, before vLLM starts, with near-miss suggestions.
* ``mode``: ``auto`` (default), ``chat`` or ``base``. ``chat`` is
  ``local-chat-completions`` with ``apply_chat_template`` and
  ``fewshot_as_multiturn`` (the mmlu-prox-lite protocol); ``base`` is
  ``local-completions`` loglikelihood (the include-44 protocol). ``auto``
  picks per task from its ``output_type``: ``generate_until`` runs chat,
  ``multiple_choice``/``loglikelihood`` run base. A task list that mixes
  them is fine: the modes run as two passes over the same output dir.
* ``num_fewshot``: the few-shot count passed to the harness. Unset keeps
  each task's own default (which may be 0 or its yaml's ``num_fewshot``).
* ``gen_kwargs``: comma-separated ``key=value`` overrides of the tasks'
  generation kwargs, e.g. ``gen_kwargs=temperature=0.6,max_gen_toks=2048``.
  Values are int/float when they parse, else percent-decoded strings
  (``%2C`` a comma, ``%20`` a space -- the ``stop=`` convention); applies
  to chat-mode generation only.
* ``metric``: the primary metric to report, for every task (e.g.
  ``acc_norm``). Unset uses each task's own primary (its config's first
  ``metric_list`` entry).

Semantics: the headline ``value`` is the macro mean over the resolved
tasks of each task's primary metric (a single task: its metric), ``n`` the
sum of the tasks' effective sizes. ``details.per_task`` keeps each task's
metric, value and size, and ``lm_eval_results.json`` under
``repeat-<k>/`` preserves the harness's full output. Groups given in
``tasks`` are expanded by the harness; only leaf tasks are averaged (their
group aggregates stay in the artifacts).

``--limit N`` is lm-eval's own: the first N docs per task. It is not the
curated benchmarks' cross-language selection; smoke runs on multilingual
groups cover the first languages' tasks only. ``--repeats k`` runs the
whole set k times and averages; there is no pass@k here, because
per-example scores have no meaning for every metric family (the curated
benchmarks, whose metrics are 0/1, keep theirs).

Resume: the harness response cache (``repeat-<k>/lm_cache``) keeps every
finished request, so a restarted run only sends the missing ones; requests
that keep failing are answered with an error marker and dropped from the
cache for the next run (the curated modules' adapters and cache care --
``chat_if``/``include`` -- reused as is).

Not available generically: ``dataset``/``dataset_revision`` overrides (each
task pins its own data; setting them fails fast) and ``answers=gold`` (the
gold responses are task-specific). The class-level pin every benchmark
carries is the harness repo itself, at the release the pyproject extra
pins -- the task yamls and their dataset pins live there. Protocol
decisions a curated benchmark freezes -- sampling, stop strings, thinking
mode -- are the tasks' own here unless ``gen_kwargs`` overrides them.
"""

from __future__ import annotations

import difflib
import json
import logging
import time
import urllib.parse
from pathlib import Path
from typing import Any

from granite_evals.registry import Benchmark, register

log = logging.getLogger(__name__)

MODES = ("auto", "chat", "base")
# A task's output_type -> the protocol that runs it (mode=auto).
OUTPUT_TYPE_MODE = {
    "generate_until": "chat",
    "multiple_choice": "base",
    "loglikelihood": "base",
}
# lm-eval results rows key metrics as "<metric>,<aggregation>" (acc,none);
# a few bare (older task configs). Try both, in that order.
METRIC_KEY_FORMATS = ("{m},none", "{m}")


def parse_tasks(spec: str) -> list[str]:
    """``--option tasks=``: comma-separated, order kept, duplicates dropped."""
    tasks = [t.strip() for t in spec.split(",") if t.strip()]
    if not tasks:
        raise SystemExit("lm-eval: tasks= is required, e.g. tasks=hellaswag,arc_easy")
    return list(dict.fromkeys(tasks))


def validate_tasks(tasks: list[str], known: list[str]) -> None:
    """Fail fast on unknown names, before anything is served or fetched."""
    unknown = [t for t in tasks if t not in known]
    if unknown:
        hints = []
        for name in unknown:
            if near := difflib.get_close_matches(name, known, n=2, cutoff=0.6):
                hints.append(f"{name!r} (did you mean {', '.join(map(repr, near))}?)")
            else:
                hints.append(repr(name))
        raise SystemExit(
            f"lm-eval: unknown task(s): {', '.join(hints)}; "
            "names are lm-evaluation-harness's own task and group ids"
        )


def parse_gen_kwargs(spec: str) -> dict[str, Any]:
    """``--option gen_kwargs=k=v,k2=v2``: int/float when they parse, else a
    percent-decoded string (no spaces or commas unencoded -- the options
    value is one word)."""
    out: dict[str, Any] = {}
    for pair in spec.split(","):
        key, sep, value = pair.partition("=")
        if not (key and sep):
            raise SystemExit(
                f"lm-eval: gen_kwargs expects key=value pairs, got {pair!r}"
            )
        value = urllib.parse.unquote(value)
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
        out[key] = value
    return out


def resolve_modes(output_types: dict[str, str], mode: str) -> dict[str, list[str]]:
    """Tasks grouped by the protocol that runs them. ``mode`` picks every
    task's protocol (auto: by output_type); mixed lists split into one group
    per mode, in a stable order."""
    if mode not in MODES:
        raise SystemExit(
            f"lm-eval: mode must be one of {', '.join(MODES)}; got {mode!r}"
        )
    if mode != "auto":
        return {mode: list(output_types)}
    unknown = sorted({t for t in output_types.values() if t not in OUTPUT_TYPE_MODE})
    if unknown:
        raise SystemExit(
            f"lm-eval: task output_type(s) {unknown} have no auto protocol; set mode=chat or mode=base"
        )
    groups: dict[str, list[str]] = {}
    for task, output_type in output_types.items():
        groups.setdefault(OUTPUT_TYPE_MODE[output_type], []).append(task)
    return dict(sorted(groups.items()))


def metric_keys(name: str) -> list[str]:
    """The results-row keys a metric may appear under."""
    return [fmt.format(m=name) for fmt in METRIC_KEY_FORMATS]


def summarize_results(
    results: dict, tasks: list[str], primary: dict[str, str]
) -> dict[str, Any]:
    """The headline and per-task numbers from one (or merged) simple_evaluate
    output: macro mean over ``tasks`` of each task's primary metric, ``n``
    summed effective sizes. Group rows are left out (only leaf tasks are
    averaged); a task whose metric key is missing counts in ``per_task`` with
    a null value but not in the mean."""
    per_task: dict[str, dict[str, Any]] = {}
    values, n = [], 0
    rows = results.get("results") or {}
    sizes = results.get("n-samples") or {}
    for task in tasks:
        row = rows.get(task) or {}
        name = primary[task]
        value = next((row[k] for k in metric_keys(name) if k in row), None)
        size = (sizes.get(task) or {}).get("effective")
        per_task[task] = {"metric": name, "value": value, "n": size}
        if value is not None:
            values.append(value)
        if isinstance(size, int):
            n += size
    return {
        "value": sum(values) / len(values) if values else None,
        "n": n,
        "per_task": per_task,
    }


def task_index() -> list[str]:
    """Every task/group name the pinned harness knows (its yaml index; no
    dataset is touched). Separate function so tests can fake it."""
    from lm_eval.tasks import TaskManager

    return list(TaskManager().all_tasks)


def load_tasks(tasks: list[str]) -> tuple[dict, dict]:
    """The harness's view of ``tasks``: (leaf task objects, group objects).
    Constructs tasks, which fetches their datasets -- the run needs them
    anyway. Separate function so tests can fake it."""
    from lm_eval.tasks import TaskManager

    loaded = TaskManager().load(tasks)
    return dict(loaded["tasks"]), dict(loaded["groups"])


def run_simple_evaluate(**kwargs):
    """lm-eval's ``simple_evaluate``, imported lazily (tests fake this)."""
    from lm_eval import simple_evaluate

    return simple_evaluate(**kwargs)


@register
class LMEval(Benchmark):
    id = "lm-eval"
    metric = "macro mean over tasks of each task's primary metric"
    extra = "lmeval"
    harness_packages = ("lm-eval",)
    # The pin every benchmark must carry (registry contract): the tasks and
    # their own dataset pins live in the harness repo, at the release the
    # pyproject extra pins (lm-eval==0.4.13); results.json records the wheel
    # version under "versions" beside this pin.
    dataset = "git+https://github.com/EleutherAI/lm-evaluation-harness"
    dataset_revision = "eb4678c395c2c1f157d462b8f3a09831160c861a"  # tag v0.4.13

    def __init__(self, config):
        super().__init__(config)
        if config.dataset or config.dataset_revision:
            raise SystemExit(
                "lm-eval: dataset/dataset_revision do not apply here (every task pins its own "
                "data); the curated benchmarks take them, or fork the task yaml"
            )
        self.task_list = parse_tasks(config.options.get("tasks", ""))
        validate_tasks(self.task_list, task_index())
        self.mode = config.options.get("mode", "auto")
        self.num_fewshot = (
            int(config.options["num_fewshot"])
            if "num_fewshot" in config.options
            else None
        )
        self.gen_kwargs = (
            parse_gen_kwargs(config.options["gen_kwargs"])
            if "gen_kwargs" in config.options
            else {}
        )
        self.metric_override = config.options.get("metric", "")

    def opt(self, key: str, default: Any) -> Any:
        value = self.config.options.get(key)
        return default if value is None else type(default)(value)

    def make_lm(self, mode: str, base_url: str, served: str, seed: int):
        """The curated modules' robust adapters, as is: bounded retries per
        request, then an error marker instead of an exception, plus progress
        log lines. Base mode tokenizes loglikelihood requests with the
        checkpoint's tokenizer, like include-44."""
        if mode == "chat":
            from granite_evals.benchmarks.chat_if import _chat_lm_class

            return _chat_lm_class()(
                base_url=f"{base_url.rstrip('/')}/chat/completions",
                model=served,
                num_concurrent=self.config.workers,
                max_retries=self.opt("max_retries", 5),
                timeout=self.opt("request_timeout", 3600),
                tokenized_requests=False,
                tokenizer_backend=None,
                seed=seed,
                benchmark_id=self.id,
            )
        from granite_evals.benchmarks.include import _lm_class

        return _lm_class()(
            base_url=f"{base_url.rstrip('/')}/completions",
            model=served,
            tokenizer=self.config.model,
            num_concurrent=self.config.workers,
            max_retries=self.opt("max_retries", 5),
            timeout=self.opt("request_timeout", 3600),
            tokenized_requests=False,
            tokenizer_backend="huggingface",
            seed=seed,
            benchmark_id=self.id,
        )

    def primary_metrics(self, task_objects: dict) -> dict[str, str]:
        """Each task's primary metric: its config's first ``metric_list``
        entry, unless ``metric=`` overrides every task with one name."""
        if self.metric_override:
            return {name: self.metric_override for name in task_objects}
        out = {}
        for name, task in task_objects.items():
            metric_list = (
                getattr(getattr(task, "config", None), "metric_list", None) or []
            )
            if not metric_list:
                raise SystemExit(
                    f"lm-eval: task {name!r} declares no metric_list; pass metric=<name>"
                )
            out[name] = metric_list[0]["metric"]
        return out

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        task_objects, groups = load_tasks(self.task_list)
        output_types = {name: task.OUTPUT_TYPE for name, task in task_objects.items()}
        primary = self.primary_metrics(task_objects)
        mode_groups = resolve_modes(output_types, self.mode)
        for mode, tasks in mode_groups.items():
            log.info(
                "lm-eval: %d task(s) in %s mode (%s)",
                len(tasks),
                mode,
                ", ".join(sorted(tasks)[:5]) + ("..." if len(tasks) > 5 else ""),
            )
        log.info(
            "lm-eval: %d group(s) expanded, %d leaf task(s) total",
            len(groups),
            len(task_objects),
        )

        per_repeat = []
        for k in range(self.repeats):
            per_repeat.append(
                {
                    "repeat": k,
                    **self._repeat(
                        k,
                        mode_groups,
                        task_objects,
                        primary,
                        base_url,
                        served_model_name,
                    ),
                }
            )
            log.info(
                "lm-eval repeat %d: %s over %d",
                k,
                per_repeat[-1]["value"],
                per_repeat[-1]["n"],
            )

        values = [r["value"] for r in per_repeat if r["value"] is not None]
        return {
            "value": sum(values) / len(values) if values else None,
            "n": per_repeat[0]["n"],
            "tasks": self.task_list,
            "leaf_tasks": sorted(task_objects),
            "groups": sorted(groups),
            "mode": (
                self.mode
                if self.mode != "auto"
                else {mode: len(tasks) for mode, tasks in mode_groups.items()}
            ),
            "num_fewshot": self.num_fewshot,
            "gen_kwargs": self.gen_kwargs,
            "per_task": per_repeat[0]["per_task"],
            "per_repeat": per_repeat,
        }

    def _repeat(
        self,
        k: int,
        mode_groups: dict,
        task_objects: dict,
        primary: dict,
        base_url: str,
        served: str,
    ) -> dict[str, Any]:
        repeat_dir = self.config.output_dir / f"repeat-{k}"
        cache_dir = repeat_dir / "lm_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()

        # One simple_evaluate pass per mode (a mixed task list runs chat and
        # base in separate passes, same repeat dir and cache), merged into
        # one result: the cache keys by request, so the modes cannot
        # collide.
        merged: dict[str, Any] = {}
        for mode, tasks in mode_groups.items():
            results = run_simple_evaluate(
                model=self.make_lm(
                    mode, base_url, served, seed=self.config.seed + 1234 + k
                ),
                tasks=tasks,
                task_manager=None,
                use_cache=str(cache_dir / "cache"),
                num_fewshot=self.num_fewshot,
                limit=self.config.limit,
                apply_chat_template=(mode == "chat"),
                fewshot_as_multiturn=(mode == "chat"),
                gen_kwargs=(self.gen_kwargs or None) if mode == "chat" else None,
                log_samples=True,
                bootstrap_iters=0,
                random_seed=self.config.seed,
            )
            for key, value in results.items():
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key].update(value)
                else:
                    merged[key] = value
        from granite_evals.benchmarks.chat_if import drop_errors_from_cache

        if dropped := drop_errors_from_cache(cache_dir):
            log.warning(
                "lm-eval repeat %d: %d failed requests dropped from the cache for retry",
                k,
                dropped,
            )
        self._write(repeat_dir, merged, task_objects)
        summary = summarize_results(merged, sorted(task_objects), primary)
        summary["duration_s"] = round(time.time() - started, 1)
        return summary

    def _write(self, repeat_dir: Path, results: dict, task_objects: dict) -> None:
        """Per-example artifacts (``samples/<task>.jsonl``) and the harness's
        own output (``lm_eval_results.json``), the curated modules' shape."""
        sdir = repeat_dir / "samples"
        sdir.mkdir(exist_ok=True)
        for task, rows in (results.get("samples") or {}).items():
            with (sdir / f"{task}.jsonl").open("w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        keep = {
            key: results.get(key)
            for key in (
                "results",
                "groups",
                "configs",
                "versions",
                "n-shot",
                "n-samples",
                "config",
            )
        }
        keep["leaf_tasks"] = sorted(task_objects)
        (repeat_dir / "lm_eval_results.json").write_text(
            json.dumps(keep, indent=2, ensure_ascii=False, default=str)
        )
