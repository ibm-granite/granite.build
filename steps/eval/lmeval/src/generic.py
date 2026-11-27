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
  ``fewshot_as_multiturn`` (mmlu-prox-lite's adapter, but each task's own
  generation kwargs, not mmlu-prox-lite's card sampling); ``base`` is
  ``local-completions`` loglikelihood (include-44's adapter). ``auto``
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
* ``thinking``: ``on`` (default: the chat template's own default; Granite
  4.x thinks) or ``off`` (``chat_template_kwargs.enable_thinking=false``, as
  mmlu-prox-lite's ``thinking=off``). Chat mode only. With thinking on, a
  chat task that sets no ``max_gen_toks`` (and none in ``gen_kwargs``) gets
  a 32768-token budget (chat_if's ``CARD_SAMPLING``): lm-eval's default of
  256 ends a Granite thought before its ``</think>``, the server's reasoning
  parser then returns no content, and lm-eval scores an empty answer. vLLM
  applies the task's stop strings to the whole generation, thought
  included (its detokenizer checks them before the reasoning parser
  splits the text), so a stop the thought happens to contain (gsm8k's
  ``Question:``) also ends it with no content. ``thinking=off`` is the
  harness's own protocol for the lm-eval generation tasks. With thinking
  on, the run warns at startup naming every chat task that stops on text
  (a stop other than a special token such as ``</s>`` or ``<|im_end|>``),
  records them in ``details.thinking_stops``, and a response check that
  fails on empty responses from such a task names this cause and the
  remedies. The requests themselves are unchanged.
* ``max_failed_frac`` (default 0.05): see the response check below.
* ``metric``: the primary metric to report, for every task (e.g.
  ``acc_norm``), or one exact results key ``<metric>,<filter>`` (e.g.
  ``exact_match,strict-match``) to pick a filter. Unset uses each task's own
  primary (its config's first ``metric_list`` entry).

A task's metric is read from its results row as ``<metric>,none`` (a task
without a ``filter_list``), then bare ``<metric>``, then ``<metric>,<filter>``
for each of the task's ``filter_list`` names in their yaml order -- so gsm8k's
``exact_match`` reads ``exact_match,strict-match``, its first filter. The key
read lands in ``details.per_task`` (``key``). A task whose row has none of
these keys fails the run (after its artifacts are written), naming the keys
it does have: a missing metric is never reported as a null score.

Response check: per task the run counts docs whose chat response is
empty (no content: a thought cut off, or stopped inside the thought) and
docs whose request failed (the adapters' error markers: chat_if's
``ERROR_MARKER``, include's ``ERROR_LL``). They stay in lm-eval's score as
wrong answers; ``details.responses`` records the counts per task, and the
run fails (``failure_policy``) when they exceed ``max_failed_frac`` of all
docs over all repeats. Failed requests are dropped from the response cache
in both modes, so a rerun retries them.

Semantics: the headline ``value`` is the macro mean over the resolved
tasks of each task's primary metric (a single task: its metric), ``n`` the
sum of the tasks' effective sizes. ``details.per_task`` keeps each task's
metric, value and size, and ``lm_eval_results.json`` under
``repeat-<k>/`` preserves the harness's full output. Groups given in
``tasks`` reach the harness as the group objects it loaded, not as their
leaf names: a group's per-task overrides (openllm's 25-shot arc_challenge,
10-shot hellaswag) apply, and the harness computes the group's aggregate
(where its yaml declares one) into the artifacts' ``results``/``groups``.
Only leaf tasks are averaged into the headline. A group whose leaves span
both modes (openllm: gsm8k generates, the rest are multiple choice) cannot
run as one pass under ``mode=auto``: its single-mode subgroups and its
tasks run in their own modes, overrides kept, but the harness computes no
aggregate for the group itself (``details.split_groups`` names it; set
``mode=chat`` or ``mode=base`` to run it whole).

``--limit N`` is lm-eval's own: the first N docs per task. It is not the
curated benchmarks' cross-language selection; smoke runs on multilingual
groups cover the first languages' tasks only. ``--repeats k`` runs the
whole set k times and averages; there is no pass@k here, because
per-example scores have no meaning for every metric family (the curated
benchmarks, whose metrics are 0/1, keep theirs).

Reproducible docs: the bridge builds the tasks itself, so it seeds
``random`` (the run's seed), numpy and torch (1234) first, as lm-eval's
simple_evaluate does before it builds them; a task whose ``process_docs``
shuffles (gpqa's choice order) then gets the same order on every run.
``datasets`` caches that mapped dataset under ``hf_home`` and reuses it,
keyed by the map function, not the seed: a cache written by an older,
unseeded bridge run keeps that run's order. Use a fresh ``hf_home``, or
delete the dataset's ``cache-*.arrow`` files under ``$HF_HOME/datasets``,
to get the seeded order.

Resume: the harness response cache (``repeat-<k>/lm_cache``) keeps every
finished request, so a restarted run only sends the missing ones; requests
that keep failing are answered with an error marker and dropped from the
cache for the next run (the curated modules' adapters and cache care --
``chat_if``/``include`` -- reused as is).

Not available generically: ``dataset``/``dataset_revision`` overrides (each
task names its own dataset; setting them fails fast) and ``answers=gold``
(the gold responses are task-specific). The class-level pin every benchmark
carries is the harness repo itself, at the release the pyproject extra
pins -- the task yamls live there. One caveat the curated benchmarks do
not have: the harness's task yamls name their datasets but do not pin
revisions, so a bridge run reads each dataset at the hub's current
snapshot (results.json records the harness pin, not per-task dataset
revisions). Protocol decisions a curated benchmark freezes -- sampling,
stop strings, thinking mode -- are the tasks' own here unless
``gen_kwargs`` overrides them.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any

from granite_evals.registry import Benchmark, failure_policy, register

log = logging.getLogger(__name__)

MODES = ("auto", "chat", "base")
# A task's output_type -> the protocol that runs it (mode=auto).
OUTPUT_TYPE_MODE = {
    "generate_until": "chat",
    "multiple_choice": "base",
    "loglikelihood": "base",
}
# lm-eval results rows key metrics as "<metric>,<filter>": "none" for a task
# without a filter_list (acc,none), else one key per filter_list entry
# (gsm8k: exact_match,strict-match and exact_match,flexible-extract); a few
# bare (older task configs). The unfiltered forms first, then the task's
# filters in their yaml order.
METRIC_KEY_FORMATS = ("{m},none", "{m}")
THINKING = ("on", "off")
# chat_if.CARD_SAMPLING["max_tokens"]: Granite 4.2's thinking budget (a test
# keeps the two equal).
THINKING_BUDGET = 32768
# Stop strings that are special tokens (</s>, <|im_end|>, <|eot_id|>): a
# thought is text and does not write them. Any other stop can occur in a
# thought, and vLLM matches stops against the thought too.
SPECIAL_TOKEN_STOP = re.compile(r"</?s>|<\|[^<>|]*\|>")


def text_stops(until: Any) -> list[str]:
    """The stop strings in a task's ``until`` that are text, not a special
    token (``until`` is a string, a list of them, or unset)."""
    if not until:
        return []
    stops = [until] if isinstance(until, str) else list(until)
    return [
        s
        for s in stops
        if isinstance(s, str) and s and not SPECIAL_TOKEN_STOP.fullmatch(s)
    ]


def thinking_stop_hint(
    thinking_stops: dict[str, list[str]], responses: dict[str, dict[str, int]]
) -> str:
    """The cause the response check names when empty responses come from
    chat tasks that stop on text with thinking on; "" when none do."""
    hit = {
        task: stops
        for task, stops in thinking_stops.items()
        if responses.get(task, {}).get("empty")
    }
    if not hit:
        return ""
    empty = sum(responses[task]["empty"] for task in hit)
    named = "; ".join(
        f"{task} stops on {stops!r}" for task, stops in sorted(hit.items())
    )
    return (
        f"{empty} of them are empty responses with thinking on from tasks that stop on "
        f"text ({named}). vLLM matches stop strings against the whole generation, thought "
        "included, so a thought that writes one ends the response before its answer. "
        "Rerun with thinking=off (the harness's protocol for its generation tasks), or "
        "replace the task's stop strings with gen_kwargs=until=<stop> (lm-eval cannot "
        "clear them)"
    )


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


def metric_keys(name: str, filters: list[str] | tuple[str, ...] = ()) -> list[str]:
    """The results-row keys a metric may appear under, in lookup order: the
    unfiltered forms, then one per filter in ``filters`` (the task's
    ``filter_list`` names, yaml order). A name that already carries a
    filter (``exact_match,strict-match``) is that one key."""
    if "," in name:
        return [name]
    keys = [fmt.format(m=name) for fmt in METRIC_KEY_FORMATS]
    keys += [f"{name},{f}" for f in filters if f"{name},{f}" not in keys]
    return keys


def task_filters(task: Any) -> list[str]:
    """A task's ``filter_list`` names in yaml order (empty: no filter_list,
    i.e. the harness's implicit ``none``)."""
    filter_list = getattr(getattr(task, "config", None), "filter_list", None) or []
    names = []
    for entry in filter_list:
        name = (
            entry.get("name")
            if isinstance(entry, dict)
            else getattr(entry, "name", None)
        )
        if name:
            names.append(name)
    return names


def summarize_results(
    results: dict,
    tasks: list[str],
    primary: dict[str, str],
    filters: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """The headline and per-task numbers from one (or merged) simple_evaluate
    output: macro mean over ``tasks`` of each task's primary metric, ``n``
    summed effective sizes. Group rows are left out (only leaf tasks are
    averaged). ``filters`` gives each task's filter names (see
    :func:`metric_keys`); a task whose row has none of its metric's keys
    fails with SystemExit rather than report a null."""
    filters = filters or {}
    per_task: dict[str, dict[str, Any]] = {}
    values, n = [], 0
    rows = results.get("results") or {}
    sizes = results.get("n-samples") or {}
    missing = []
    for task in tasks:
        row = rows.get(task) or {}
        name = primary[task]
        key = next(
            (k for k in metric_keys(name, filters.get(task, ())) if k in row), None
        )
        if key is None:
            have = sorted(k for k in row if "_stderr" not in k and k != "alias")
            missing.append(
                f"{task!r}: no {name!r} (has {', '.join(have) or 'no metrics'})"
            )
            continue
        value = row[key]
        size = (sizes.get(task) or {}).get("effective")
        per_task[task] = {"metric": name, "key": key, "value": value, "n": size}
        values.append(value)
        if isinstance(size, int):
            n += size
    if missing:
        raise SystemExit(
            "lm-eval: metric not in the results for "
            + "; ".join(missing)
            + ". Pass metric=<metric> or metric=<metric>,<filter> (lm_eval_results.json "
            "under repeat-<k>/ has the full rows)"
        )
    return {
        "value": sum(values) / len(values) if values else None,
        "n": n,
        "per_task": per_task,
    }


def is_group(item: Any) -> bool:
    """An lm-eval Group (it has leaves), as opposed to a Task."""
    return hasattr(item, "get_all_tasks")


def plan_passes(
    task_list: list[str],
    task_objects: dict,
    groups: dict,
    mode_of: dict[str, str],
) -> tuple[dict[str, list], list[str]]:
    """What each mode's simple_evaluate pass is given: the loaded objects,
    never names (a name reloads the task standalone, dropping the overrides
    a group set on it, and no group aggregate is computed).

    A requested group whose leaves all run in one mode goes to that pass
    whole. A group that spans modes is split into its direct children (in
    yaml order), recursively, so its single-mode subgroups still aggregate;
    its name is returned in the second element. Leaf tasks not under a
    requested group (plain task names, tags' tasks) go as Task objects.
    Returns ({mode: [objects]}, [split group names])."""
    passes: dict[str, list] = {}
    split: list[str] = []
    covered: set[str] = set()

    def place(item: Any) -> None:
        if not is_group(item):
            covered.add(item.task_name)
            passes.setdefault(mode_of[item.task_name], []).append(item)
            return
        leaves = [task.task_name for task in item.get_all_tasks()]
        covered.update(leaves)
        modes = {mode_of[name] for name in leaves}
        if len(modes) == 1:
            passes.setdefault(modes.pop(), []).append(item)
        elif modes:
            split.append(item.name)
            for child in item:
                place(child)

    for name in task_list:
        if name in groups:
            place(groups[name])
    for name, task in task_objects.items():
        if name not in covered:
            place(task)
    return dict(sorted(passes.items())), split


def is_empty_response(text: Any) -> bool:
    """A chat response with no content: lm-eval turns a null ``content`` into
    ``LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER`` (default empty)."""
    import os

    text = "" if text is None else str(text)
    placeholder = os.environ.get("LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER", "")
    return not text.strip() or (bool(placeholder) and text == placeholder)


def response_counts(samples: dict, chat_tasks: set[str]) -> dict[str, dict[str, int]]:
    """Per task: docs, docs with an empty chat response, docs with a failed
    request (chat_if's ERROR_MARKER, include's ERROR_LL), from lm-eval's
    logged samples. lm-eval logs one row per doc per filter (gsm8k: two), all
    with the same raw ``resps``: each doc counts once, by its ``doc_id``."""
    from granite_evals.benchmarks.chat_if import ERROR_MARKER
    from granite_evals.benchmarks.include import ERROR_LL

    out: dict[str, dict[str, int]] = {}
    for task, rows in (samples or {}).items():
        docs = empty = errors = 0
        seen: set = set()
        for index, row in enumerate(rows):
            doc = row.get("doc_id", ("row", index))
            if doc in seen:
                continue
            seen.add(doc)
            docs += 1
            resps = [r for choice in row.get("resps") or [] for r in choice]
            if task in chat_tasks:
                if any(r == ERROR_MARKER for r in resps):
                    errors += 1
                elif resps and all(is_empty_response(r) for r in resps):
                    empty += 1
            elif any(
                isinstance(r, (list, tuple)) and list(r) == list(ERROR_LL)
                for r in resps
            ):
                errors += 1
        out[task] = {"docs": docs, "empty": empty, "errors": errors}
    return out


def task_index() -> list[str]:
    """Every task/group name the pinned harness knows (its yaml index; no
    dataset is touched). Separate function so tests can fake it."""
    from lm_eval.tasks import TaskManager

    return list(TaskManager().all_tasks)


def seed_like_simple_evaluate(seed: int) -> None:
    """Seed as lm-eval's simple_evaluate does before it builds tasks:
    ``random`` with the run's seed, numpy and torch with lm-eval's
    DEFAULT_OTHER_SEED (1234). The bridge builds the tasks itself
    (load_tasks, before simple_evaluate), and task construction runs
    ``process_docs``, which may shuffle (gpqa shuffles each question's
    choices with ``random``): unseeded, every fresh dataset cache gets a
    different answer order."""
    import random

    other = 1234
    try:
        from lm_eval.defaults import DEFAULT_OTHER_SEED as other
    except ImportError:
        pass
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(other)
    except ImportError:
        pass
    try:
        from lm_eval.utils import set_torch_seed
    except ImportError:
        return
    set_torch_seed(other)


def load_tasks(tasks: list[str]) -> tuple[dict, dict]:
    """The harness's view of ``tasks``: (leaf task objects, group objects).
    Constructs tasks, which fetches their datasets -- the run needs them
    anyway. Separate function so tests can fake it."""
    from lm_eval.tasks import TaskManager

    loaded = TaskManager().load(tasks)
    return dict(loaded["tasks"]), dict(loaded["groups"])


def drop_failed_requests(cache_dir: Path) -> int:
    """Drop both adapters' failed-request answers from the response cache, so
    a rerun retries them instead of scoring them: chat_if's ERROR_MARKER
    (chat mode) and include's ERROR_LL (base mode, otherwise scored as the
    worst choice)."""
    from granite_evals.benchmarks import chat_if, include

    return chat_if.drop_errors_from_cache(cache_dir) + include.drop_errors_from_cache(
        cache_dir
    )


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
        self.thinking = config.options.get("thinking", "on")
        if self.thinking not in THINKING:
            raise SystemExit(
                f"lm-eval: thinking must be one of {', '.join(THINKING)}; got {self.thinking!r}"
            )

    def chat_gen_kwargs(self) -> dict[str, Any] | None:
        """The chat pass's gen_kwargs: ``gen_kwargs=`` plus, with
        ``thinking=off``, ``chat_template_kwargs.enable_thinking=false``."""
        kw = dict(self.gen_kwargs)
        if self.thinking == "off":
            kw["chat_template_kwargs"] = {"enable_thinking": False}
        return kw or None

    def apply_thinking_budget(
        self, task_objects: dict, chat_tasks: list[str]
    ) -> list[str]:
        """With thinking on, give every chat task that sets no max_gen_toks
        (and none in gen_kwargs) the thinking budget; the tasks given it."""
        if self.thinking != "on" or "max_gen_toks" in self.gen_kwargs:
            return []
        given = []
        for name in chat_tasks:
            task = task_objects[name]
            own = (
                getattr(getattr(task, "config", None), "generation_kwargs", None) or {}
            )
            if "max_gen_toks" not in own:
                task.set_config(
                    key="generation_kwargs",
                    value={"max_gen_toks": THINKING_BUDGET},
                    update=True,
                )
                given.append(name)
        if given:
            log.info(
                "lm-eval: thinking on: max_gen_toks=%d for %s",
                THINKING_BUDGET,
                ", ".join(given),
            )
        return given

    def thinking_stops(
        self, task_objects: dict, chat_tasks: list[str]
    ) -> dict[str, list[str]]:
        """With thinking on, each chat task's text stop strings (``until``
        from gen_kwargs, else the task's own); tasks without any are left
        out. Empty with thinking off."""
        if self.thinking != "on":
            return {}
        out = {}
        for name in chat_tasks:
            if "until" in self.gen_kwargs:
                until = self.gen_kwargs["until"]
            else:
                own = (
                    getattr(
                        getattr(task_objects[name], "config", None),
                        "generation_kwargs",
                        None,
                    )
                    or {}
                )
                until = own.get("until")
            stops = text_stops(until)
            if stops:
                out[name] = stops
        return out

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
        seed_like_simple_evaluate(self.config.seed)
        task_objects, groups = load_tasks(self.task_list)
        output_types = {name: task.OUTPUT_TYPE for name, task in task_objects.items()}
        primary = self.primary_metrics(task_objects)
        mode_groups = resolve_modes(output_types, self.mode)
        budget_tasks = self.apply_thinking_budget(
            task_objects, mode_groups.get("chat", [])
        )
        stop_tasks = self.thinking_stops(task_objects, mode_groups.get("chat", []))
        if stop_tasks:
            log.warning(
                "lm-eval: thinking on, and %d chat task(s) stop on text (%s): vLLM matches "
                "stop strings against the whole generation, thought included, so a thought "
                "that writes one ends the response with no answer (an empty response); "
                "thinking=off avoids it",
                len(stop_tasks),
                "; ".join(f"{t} on {s!r}" for t, s in sorted(stop_tasks.items())[:5])
                + ("; ..." if len(stop_tasks) > 5 else ""),
            )
        passes, split = plan_passes(
            self.task_list,
            task_objects,
            groups,
            {task: mode for mode, tasks in mode_groups.items() for task in tasks},
        )
        for name in split:
            log.warning(
                "lm-eval: group %s spans chat and base tasks: its parts run in their own "
                "modes (overrides kept), but no aggregate is computed for it; set mode=chat "
                "or mode=base to run it whole",
                name,
            )
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
                        passes,
                        task_objects,
                        primary,
                        base_url,
                        served_model_name,
                        set(mode_groups.get("chat", [])),
                    ),
                }
            )
            log.info(
                "lm-eval repeat %d: %s over %d",
                k,
                per_repeat[-1]["value"],
                per_repeat[-1]["n"],
            )

        responses: dict[str, dict[str, int]] = {}
        for rep in per_repeat:
            for task, counts in rep["responses"].items():
                into = responses.setdefault(task, dict.fromkeys(counts, 0))
                for key, n in counts.items():
                    into[key] += n
        try:
            policy = failure_policy(
                self.id,
                "responses",
                sum(c["empty"] + c["errors"] for c in responses.values()),
                sum(c["docs"] for c in responses.values()),
                self.opt("max_failed_frac", 0.05),
            )
        except SystemExit as e:
            hint = thinking_stop_hint(stop_tasks, responses)
            if not hint:
                raise
            raise SystemExit(f"{e}. {hint}") from None
        values = [r["value"] for r in per_repeat if r["value"] is not None]
        return {
            "value": sum(values) / len(values) if values else None,
            "n": per_repeat[0]["n"],
            "tasks": self.task_list,
            "leaf_tasks": sorted(task_objects),
            "groups": sorted(groups),
            "split_groups": split,
            "mode": (
                self.mode
                if self.mode != "auto"
                else {mode: len(tasks) for mode, tasks in mode_groups.items()}
            ),
            "num_fewshot": self.num_fewshot,
            "gen_kwargs": self.gen_kwargs,
            "thinking": self.thinking,
            "thinking_budget": {"max_gen_toks": THINKING_BUDGET, "tasks": budget_tasks},
            "thinking_stops": stop_tasks,
            "responses": responses,
            **policy,
            "per_task": per_repeat[0]["per_task"],
            "per_repeat": per_repeat,
        }

    def _repeat(
        self,
        k: int,
        passes: dict,
        task_objects: dict,
        primary: dict,
        base_url: str,
        served: str,
        chat_tasks: set[str] = frozenset(),
    ) -> dict[str, Any]:
        repeat_dir = self.config.output_dir / f"repeat-{k}"
        cache_dir = repeat_dir / "lm_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()

        # One simple_evaluate pass per mode (a mixed task list runs chat and
        # base in separate passes, same repeat dir and cache), merged into
        # one result: the cache keys by request, so the modes cannot
        # collide. Each pass gets the loaded Task/Group objects (see
        # plan_passes), so group overrides and aggregates apply.
        merged: dict[str, Any] = {}
        for mode, items in passes.items():
            results = run_simple_evaluate(
                model=self.make_lm(
                    mode, base_url, served, seed=self.config.seed + 1234 + k
                ),
                tasks=items,
                task_manager=None,
                use_cache=str(cache_dir / "cache"),
                num_fewshot=self.num_fewshot,
                limit=self.config.limit,
                apply_chat_template=(mode == "chat"),
                fewshot_as_multiturn=(mode == "chat"),
                gen_kwargs=self.chat_gen_kwargs() if mode == "chat" else None,
                log_samples=True,
                bootstrap_iters=0,
                random_seed=self.config.seed,
            )
            for key, value in results.items():
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key].update(value)
                else:
                    merged[key] = value
        if dropped := drop_failed_requests(cache_dir):
            log.warning(
                "lm-eval repeat %d: %d failed requests dropped from the cache for retry",
                k,
                dropped,
            )
        self._write(repeat_dir, merged, task_objects)
        summary = summarize_results(
            merged,
            sorted(task_objects),
            primary,
            {name: task_filters(task) for name, task in task_objects.items()},
        )
        summary["responses"] = response_counts(merged.get("samples") or {}, chat_tasks)
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
