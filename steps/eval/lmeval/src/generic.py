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
  (``%2C`` a comma, ``%20`` a space -- the ``stop=`` convention),
  ``do_sample`` as ``true``/``false``; applies to chat-mode generation only.
  ``model``, ``messages``, ``prompt`` and ``stop`` are refused: the request
  sets them itself (a task's stop strings are ``until=``).
* ``thinking``: ``off`` (default) or ``on``, chat mode only. Before the
  server starts, the bridge renders the checkpoint's chat template on a
  fixed message with each known switch (``enable_thinking``, Granite 3.x's
  ``thinking``) set true and false; a switch is a kwarg whose two renders
  differ (the tokenizer's think tokens are recorded as a hint only). It
  prints one status line saying which, e.g. ``thinking=off (default); this
  checkpoint's chat template has a thinking switch (enable_thinking), pass
  thinking=on to use it``, and records the probe in
  ``details.thinking_switch``. Requests then set that switch
  (``chat_template_kwargs.<switch>=false``, or ``true`` with ``thinking=on``);
  a template without one gets no kwarg, and ``thinking=on`` is refused
  there (nothing to turn on). A template that cannot be read falls back to
  ``enable_thinking``. With thinking on, a chat task that sets no
  ``max_gen_toks`` or ``max_tokens`` (and none in ``gen_kwargs``) gets a
  32768-token budget (chat_if's ``CARD_SAMPLING``): lm-eval's default of 256
  ends a Granite thought before its ``</think>``, and the server's
  reasoning parser then returns no content. ``thinking=on`` with a task
  that stops on text (any stop but a special token such as ``</s>`` or
  ``<|im_end|>``) is refused at startup: vLLM applies stop strings to the
  whole generation, thought included (its detokenizer checks them before
  the reasoning parser splits the text), so a thought that writes one
  (gsm8k's ``Question:``) ends the response with no content. ``thinking=off``
  is the harness's own protocol for its generation tasks.
* ``max_failed_frac`` (default 0.05, in [0, 1]; ``num_fewshot`` and it are
  checked at startup): see the response check below.
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
``ERROR_MARKER``, include's ``ERROR_LL``). They are left out of the score,
as registry.failure_policy has it (a failed item is never scored as a model
loss): a mean metric is recomputed from the per-doc sample rows over the
docs left (``details.per_task``: ``value``, ``n``, ``left_out``, and the
harness's own number as ``harness_value``); a metric that does not average
over docs cannot drop any, so such a task with a failed doc fails the run.
``details.responses`` records the counts per task, and the run fails
(``failure_policy``) when they exceed ``max_failed_frac`` of all docs over
all repeats. Failed requests are dropped from the response cache
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
``mode=base`` to run it whole, its generation tasks then through the
completions endpoint without the chat template). ``mode=chat`` runs
generation tasks only: the chat endpoint returns no loglikelihoods, so a
selection with a multiple-choice or loglikelihood task is refused at startup.

``--limit N`` is lm-eval's own: the first N docs per task. It is not the
curated benchmarks' cross-language selection; smoke runs on multilingual
groups cover the first languages' tasks only. ``--repeats k`` runs the
whole set k times and averages; there is no pass@k here, because
per-example scores have no meaning for every metric family (the curated
benchmarks, whose metrics are 0/1, keep theirs).

Reproducible docs: the bridge builds the tasks itself, so it seeds
``random`` (the run's seed), numpy and torch (1234) first, as lm-eval's
simple_evaluate does before it builds them, and builds them with the
``datasets`` map cache off: a task whose ``process_docs`` shuffles in a map
(gpqa's choice order) then gets its seed's order on every run, whichever
run first wrote that dataset's cache under ``hf_home``.

Resume: the harness response cache (``repeat-<k>/lm_cache/<key>``, the key a
hash of the model, its served name and the repeat's LM seed, so another
model or seed never replays these answers) keeps every finished request, so a
restarted run only sends the missing ones; requests
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

import contextlib
import difflib
import hashlib
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
# Below this many tokens of context beside the budget, a few-shot prompt may
# not fit: the run warns.
PROMPT_ROOM = 8192
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


# The chat-template kwargs that switch thinking, in the order they are
# tried: Granite 4.x (and most templates) read enable_thinking, Granite 3.x
# reads thinking.
THINKING_SWITCHES = ("enable_thinking", "thinking")
THINK_TOKENS = ("<think>", "</think>")
PROBE_MESSAGES = [{"role": "user", "content": "What is 2 + 2?"}]


def chat_template_thinking(model: str) -> dict[str, Any]:
    """Whether ``model``'s chat template has a thinking switch: the template
    rendered on a fixed message with each of THINKING_SWITCHES set True and
    False; a switch is a kwarg whose two renders differ, and ``default`` says
    which of them the template renders with no kwarg. The tokenizer's think
    tokens are recorded as a hint only (a template may carry them with no
    switch). Never raises: an unreadable template is recorded in ``error``.
    Separate function so tests can fake it."""
    out: dict[str, Any] = {
        "switch": None,
        "switches": {},
        "think_tokens": [],
        "error": None,
    }
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        def render(**kwargs):
            return tok.apply_chat_template(
                PROBE_MESSAGES, tokenize=False, add_generation_prompt=True, **kwargs
            )

        default = render()
        for kwarg in THINKING_SWITCHES:
            on, off = render(**{kwarg: True}), render(**{kwarg: False})
            if on != off:
                out["switches"][kwarg] = {
                    "default": (
                        "on"
                        if default == on
                        else "off" if default == off else "neither"
                    )
                }
        vocab = tok.get_vocab()
        out["think_tokens"] = [t for t in THINK_TOKENS if t in vocab]
    except Exception as e:  # noqa: BLE001 - recorded; the run falls back
        out["error"] = f"{type(e).__name__}: {e}"[:300]
        return out
    out["switch"] = next(iter(out["switches"]), None)
    return out


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


# Payload fields lm-eval's API models set themselves; gen_kwargs is spread
# over them, so these would silently replace the served model, the prompt or
# the task's stop strings (set until= for stops).
RESERVED_GEN_KWARGS = ("model", "messages", "prompt", "stop")
BOOL_WORDS = {"true": True, "false": False}


def parse_gen_kwargs(spec: str) -> dict[str, Any]:
    """``--option gen_kwargs=k=v,k2=v2``: int/float when they parse,
    ``do_sample`` as a bool (true/false), else a percent-decoded string (no
    spaces or commas unencoded -- the options value is one word). Keys the
    request payload owns (RESERVED_GEN_KWARGS) are refused."""
    out: dict[str, Any] = {}
    for pair in spec.split(","):
        key, sep, value = pair.partition("=")
        if not (key and sep):
            raise SystemExit(
                f"lm-eval: gen_kwargs expects key=value pairs, got {pair!r}"
            )
        if key in RESERVED_GEN_KWARGS:
            raise SystemExit(
                f"lm-eval: gen_kwargs cannot set {key!r}: the request sets it itself"
                + (" (set the task's stop strings with until=)" if key == "stop" else "")
            )
        value = urllib.parse.unquote(value)
        if key == "do_sample":
            if value.lower() not in BOOL_WORDS:
                raise SystemExit(f"lm-eval: gen_kwargs do_sample must be true or false, got {value!r}")
            out[key] = BOOL_WORDS[value.lower()]
            continue
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
        out[key] = value
    return out


MAX_FAILED_FRAC = 0.05


def parse_count(key: str, value: Any) -> int:
    """A non-negative integer option, or a SystemExit naming it."""
    try:
        n = int(str(value))
    except ValueError:
        raise SystemExit(f"lm-eval: {key} must be a whole number, got {value!r}") from None
    if n < 0:
        raise SystemExit(f"lm-eval: {key} must be 0 or more, got {n}")
    return n


def parse_fraction(key: str, value: Any) -> float:
    """A fraction option in [0, 1], or a SystemExit naming it."""
    try:
        frac = float(str(value))
    except ValueError:
        raise SystemExit(f"lm-eval: {key} must be a number in [0, 1], got {value!r}") from None
    if not 0 <= frac <= 1:
        raise SystemExit(f"lm-eval: {key} must be in [0, 1], got {frac}")
    return frac


def resolve_modes(output_types: dict[str, str], mode: str) -> dict[str, list[str]]:
    """Tasks grouped by the protocol that runs them. ``mode`` picks every
    task's protocol (auto: by output_type); mixed lists split into one group
    per mode, in a stable order."""
    if mode not in MODES:
        raise SystemExit(
            f"lm-eval: mode must be one of {', '.join(MODES)}; got {mode!r}"
        )
    if mode == "chat":
        # local-chat-completions generates only: it has no loglikelihood, so
        # a multiple-choice task would fail after the server is up.
        scored = sorted(
            t for t, kind in output_types.items() if kind != "generate_until"
        )
        if scored:
            raise SystemExit(
                f"lm-eval: mode=chat runs generation tasks only (the chat endpoint returns "
                f"no loglikelihoods), but {', '.join(scored[:5])}"
                f"{' ...' if len(scored) > 5 else ''} score by loglikelihood; use mode=auto "
                "(each task its own protocol) or mode=base"
            )
    if mode != "auto":
        return {mode: list(output_types)}
    unknown = sorted({t for t in output_types.values() if t not in OUTPUT_TYPE_MODE})
    if unknown:
        raise SystemExit(
            f"lm-eval: task output_type(s) {unknown} have no auto protocol; set mode=base "
            "(the completions endpoint serves every output type)"
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


def failed_docs(samples: dict, generation_tasks: set[str]) -> dict[str, dict[str, set]]:
    """Per task, the doc_ids with an empty response and those whose request
    failed, from lm-eval's logged samples. A generation task (chat mode, or
    base mode through the completions endpoint) fails with ERROR_MARKER and
    can be empty; a loglikelihood task fails with include's ERROR_LL. lm-eval
    logs one row per doc per filter (gsm8k: two), all with the same raw
    ``resps``: each doc counts once, by its ``doc_id``. ``docs`` is every
    doc_id seen."""
    from granite_evals.benchmarks.chat_if import ERROR_MARKER
    from granite_evals.benchmarks.include import ERROR_LL

    out: dict[str, dict[str, set]] = {}
    for task, rows in (samples or {}).items():
        seen: set = set()
        empty: set = set()
        errors: set = set()
        for index, row in enumerate(rows):
            # every row of a doc carries the same raw resps: the sets count it once
            doc = row.get("doc_id", ("row", index))
            seen.add(doc)
            resps = [r for choice in row.get("resps") or [] for r in choice]
            if task in generation_tasks:
                if any(r == ERROR_MARKER for r in resps):
                    errors.add(doc)
                elif resps and all(is_empty_response(r) for r in resps):
                    empty.add(doc)
            elif any(
                isinstance(r, (list, tuple)) and list(r) == list(ERROR_LL)
                for r in resps
            ):
                errors.add(doc)
        out[task] = {"docs": seen, "empty": empty, "errors": errors}
    return out


def response_counts(
    samples: dict, generation_tasks: set[str]
) -> dict[str, dict[str, int]]:
    """Per task: docs, docs with an empty response, docs with a failed
    request (see failed_docs)."""
    return {
        task: {key: len(ids) for key, ids in found.items()}
        for task, found in failed_docs(samples, generation_tasks).items()
    }


def is_mean_metric(task: Any, metric: str) -> bool:
    """Whether ``metric``'s aggregation over docs is the mean, so the score
    over the docs left in is the mean of their per-doc values: the task's
    ``aggregation()`` (the harness's resolved functions), else its
    ``metric_list`` entry's ``aggregation``."""
    try:
        fn = task.aggregation().get(metric)
    except Exception:  # noqa: BLE001 - a task without the method: its yaml
        fn = None
    if fn is not None:
        return getattr(fn, "__name__", "") == "mean"
    for entry in getattr(getattr(task, "config", None), "metric_list", None) or []:
        if entry.get("metric") == metric:
            return entry.get("aggregation") == "mean"
    return False


def score_without(
    rows: list[dict], key: str, left_out: set
) -> tuple[float | None, int]:
    """The mean of ``key``'s per-doc values (``<metric>,<filter>``: the
    sample rows of that filter, ``none`` for a bare metric) over the docs
    not in ``left_out``, and how many there are."""
    metric, _, filt = key.partition(",")
    filt = filt or "none"
    values = [
        float(row[metric])
        for row in rows
        if row.get("filter", "none") == filt
        and row.get("doc_id") not in left_out
        and isinstance(row.get(metric), (int, float, bool))
    ]
    return (sum(values) / len(values) if values else None), len(values)


def leave_out_failed(
    summary: dict[str, Any],
    samples: dict,
    task_objects: dict,
    found: dict[str, dict[str, set]],
) -> None:
    """Rescore, in place, each task with failed or empty responses over the
    docs left (registry.failure_policy: a failed item is never scored as a
    model loss). A mean metric is recomputed from the per-doc sample rows;
    any other aggregation cannot drop docs, so the run fails, naming it.
    The harness's own number stays in ``harness_value``."""
    values, n = [], 0
    for task, entry in summary["per_task"].items():
        out = found.get(task, {})
        left = (out.get("empty") or set()) | (out.get("errors") or set())
        entry["left_out"] = len(left)
        if left:
            metric = entry["key"].partition(",")[0]
            if not is_mean_metric(task_objects[task], metric):
                # Raised by run() once the failure threshold has had its say.
                summary.setdefault("unscorable", []).append(
                    f"{task}: {len(left)} docs failed or came back empty, and {metric!r} does "
                    "not average over docs, so they cannot be left out of its score"
                )
                entry["harness_value"], entry["value"] = entry["value"], None
                continue
            entry["harness_value"] = entry["value"]
            entry["value"], entry["n"] = score_without(samples.get(task) or [], entry["key"], left)
        if entry["value"] is not None:
            values.append(entry["value"])
        if isinstance(entry["n"], int):
            n += entry["n"]
    summary["value"] = sum(values) / len(values) if values else None
    summary["n_selected"] = summary["n"]  # the harness's sizes, before any left out
    summary["n"] = n


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


@contextlib.contextmanager
def fresh_dataset_maps():
    """No ``datasets`` map cache while tasks are built (and their eval docs
    frozen, see freeze_eval_docs). ``datasets`` caches a
    map's output under HF_HOME keyed by its fingerprint, which hashes the
    function, not the ``random`` state it draws from: gpqa's ``process_docs``
    shuffles each question's choices in a map, so whichever run first wrote
    that cache fixed the order for every later run on the same HF_HOME,
    whatever its seed. With caching off the map is recomputed from the
    seeded state; the downloaded and prepared data are still reused."""
    try:
        import datasets
    except ImportError:
        yield
        return
    was = datasets.is_caching_enabled()
    datasets.disable_caching()
    try:
        yield
    finally:
        if was:
            datasets.enable_caching()


def load_tasks(tasks: list[str], include_path: str | None = None) -> tuple[dict, dict]:
    """The harness's view of ``tasks``: (leaf task objects, group objects).
    Constructs tasks, which fetches their datasets -- the run needs them
    anyway -- with no map cache, and freezes each task's eval docs (see
    fresh_dataset_maps and freeze_eval_docs). ``include_path``
    adds a directory of task yamls (tests). Tests can fake this."""
    from lm_eval.tasks import TaskManager

    with fresh_dataset_maps():
        loaded = TaskManager(include_path=include_path).load(tasks)
        for task in loaded["tasks"].values():
            freeze_eval_docs(task)
    return dict(loaded["tasks"]), dict(loaded["groups"])


def freeze_eval_docs(task: Any) -> None:
    """Compute a task's eval docs once, now (seeded, no map cache), and have
    every later call return those: the harness re-runs ``process_docs`` each
    time it reads them (building requests, then scoring), and a shuffling
    ``process_docs`` would draw a new order each time with the cache off, or
    the cached order of whichever run wrote it with the cache on."""
    for name, has in (("test_docs", "has_test_docs"), ("validation_docs", "has_validation_docs")):
        if getattr(task, has, lambda: False)():
            docs = getattr(task, name)()
            setattr(task, name, lambda docs=docs: docs)
            return


def cache_key(model: str, served: str, seed: int) -> str:
    """The response cache's subdirectory for one model and LM seed: the
    harness keys cached responses by request (prompt and generation kwargs)
    only, so without it a run over another model, or a repeat with another
    seed, in the same dir would replay the first one's answers."""
    blob = json.dumps({"model": model, "served": served, "seed": seed}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def drop_failed_requests(cache_dir: Path) -> int:
    """Drop both adapters' failed-request answers from the response cache, so
    a rerun retries them instead of scoring them: chat_if's ERROR_MARKER
    (chat mode) and include's ERROR_LL (base mode, otherwise scored as the
    worst choice)."""
    from granite_evals.benchmarks import chat_if, include

    return chat_if.drop_errors_from_cache(cache_dir) + include.drop_errors_from_cache(
        cache_dir
    )


def _base_lm_class():
    """include's robust local-completions adapter, with one change for the
    bridge's base-mode generation tasks: a failed generation is answered with
    chat_if's ERROR_MARKER (a string the task's filters read as a wrong
    answer, and the response check counts as failed), not include's
    loglikelihood tuple ERROR_LL, which a generation task's regex filters
    would be handed."""
    from granite_evals.benchmarks.chat_if import ERROR_MARKER
    from granite_evals.benchmarks.include import ERROR_LL, _lm_class

    class BridgeCompletions(_lm_class()):
        async def amodel_call(self, session, sem, messages, *, generate=True, **kwargs):
            out = await super().amodel_call(session, sem, messages, generate=generate, **kwargs)
            if generate and isinstance(out, list) and out and all(o == ERROR_LL for o in out):
                return [ERROR_MARKER] * len(out)
            return out

    return BridgeCompletions


def served_max_model_len(base_url: str, served: str) -> int | None:
    """The served model's context length as vLLM reports it (``/models``),
    or None when the endpoint does not say. Tests fake this."""
    import httpx

    try:
        models = httpx.get(f"{base_url.rstrip('/')}/models", timeout=30).json()["data"]
    except Exception:  # noqa: BLE001 - an endpoint without it: no check
        return None
    for entry in models:
        if entry.get("id") == served and isinstance(entry.get("max_model_len"), int):
            return entry["max_model_len"]
    return None


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
            parse_count("num_fewshot", config.options["num_fewshot"])
            if "num_fewshot" in config.options
            else None
        )
        self.max_failed_frac = parse_fraction(
            "max_failed_frac", config.options.get("max_failed_frac", MAX_FAILED_FRAC)
        )
        self.gen_kwargs = (
            parse_gen_kwargs(config.options["gen_kwargs"])
            if "gen_kwargs" in config.options
            else {}
        )
        self.metric_override = config.options.get("metric", "")
        self.thinking = config.options.get("thinking", "off")
        if self.thinking not in THINKING:
            raise SystemExit(
                f"lm-eval: thinking must be one of {', '.join(THINKING)}; got {self.thinking!r}"
            )
        # Built here, before the server starts (the CLI constructs the
        # benchmark first): a selection the harness cannot run fails in
        # seconds, not after vLLM is up. Seeded first, as simple_evaluate
        # seeds before it builds tasks (see seed_like_simple_evaluate).
        seed_like_simple_evaluate(config.seed)
        self.task_objects, self.groups = load_tasks(self.task_list)
        self.mode_groups = resolve_modes(
            {name: task.OUTPUT_TYPE for name, task in self.task_objects.items()},
            self.mode,
        )
        self.primary = self.primary_metrics(self.task_objects)
        self.thinking_switch = self.check_thinking()

    def thinking_kwargs(self) -> dict[str, bool]:
        """The chat-template kwarg the requests set: the detected switch, on
        or off as ``thinking`` says; none for a template without one; the
        ``enable_thinking`` fallback when the template could not be read."""
        probe = self.thinking_switch or {}
        if probe.get("error"):
            return {"enable_thinking": self.thinking == "on"}
        if probe.get("switch"):
            return {probe["switch"]: self.thinking == "on"}
        return {}

    def chat_gen_kwargs(self) -> dict[str, Any] | None:
        """The chat pass's gen_kwargs: ``gen_kwargs=`` plus the thinking
        switch's ``chat_template_kwargs`` (see thinking_kwargs)."""
        kw = dict(self.gen_kwargs)
        if switch := self.thinking_kwargs():
            kw["chat_template_kwargs"] = switch
        return kw or None

    def check_thinking(self) -> dict[str, Any] | None:
        """Before the server starts: probe the chat template (chat tasks
        only), print the status line, and refuse thinking=on where it cannot
        work -- a template with no switch, or a task that stops on text.
        Returns the probe (None without a chat task)."""
        chat = self.mode_groups.get("chat", [])
        if not chat:
            if self.thinking == "on":
                raise SystemExit(
                    "lm-eval: thinking=on applies to chat-mode tasks, and this selection has none"
                )
            log.info(
                "lm-eval: thinking does not apply (no task in this selection runs in chat mode)"
            )
            return None
        probe = chat_template_thinking(self.config.model)
        given = "thinking" in self.config.options
        head = f"thinking={self.thinking}{'' if given else ' (default)'}"
        if probe["error"]:
            log.warning(
                "lm-eval: %s; could not read this checkpoint's chat template (%s), so requests "
                "set chat_template_kwargs.enable_thinking=%s",
                head,
                probe["error"],
                str(self.thinking == "on").lower(),
            )
        elif probe["switch"] is None:
            hint = (
                f" (its tokenizer has {' and '.join(probe['think_tokens'])}, but the template "
                "renders the same either way)"
                if probe["think_tokens"]
                else ""
            )
            if self.thinking == "on":
                raise SystemExit(
                    f"lm-eval: thinking=on, but this checkpoint's chat template has no thinking "
                    f"switch{hint}: there is nothing to turn on; drop thinking=on"
                )
            log.info(
                "lm-eval: %s; this checkpoint's chat template has no thinking switch%s",
                head,
                hint,
            )
        elif self.thinking == "off":
            log.info(
                "lm-eval: %s; this checkpoint's chat template has a thinking switch (%s), "
                "pass thinking=on to use it",
                head,
                probe["switch"],
            )
        else:
            log.info(
                "lm-eval: %s; requests set chat_template_kwargs.%s=true",
                head,
                probe["switch"],
            )
        if self.thinking == "on" and (
            stops := self.thinking_stops(self.task_objects, chat)
        ):
            named = "; ".join(
                f"{t} stops on {s!r}" for t, s in sorted(stops.items())[:5]
            )
            raise SystemExit(
                f"lm-eval: thinking=on, but {named}"
                f"{'; ...' if len(stops) > 5 else ''}: vLLM matches stop strings against the "
                "whole generation, thought included, so a thought that writes one ends the "
                "response with no answer. Use thinking=off (the harness's protocol for its "
                "generation tasks), or replace the stops with gen_kwargs=until=<stop>"
            )
        return probe

    def apply_thinking_budget(
        self, task_objects: dict, chat_tasks: list[str]
    ) -> list[str]:
        """With thinking on, give every chat task that sets no max_gen_toks
        or max_tokens (and none in gen_kwargs) the thinking budget; the tasks
        given it."""
        # lm-eval sends max_tokens over max_gen_toks when a task or gen_kwargs
        # sets it, so either one is the run's own budget.
        if self.thinking != "on" or {"max_gen_toks", "max_tokens"} & set(self.gen_kwargs):
            return []
        given = []
        for name in chat_tasks:
            task = task_objects[name]
            own = (
                getattr(getattr(task, "config", None), "generation_kwargs", None) or {}
            )
            if not {"max_gen_toks", "max_tokens"} & set(own):
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

    def check_context(self, max_model_len: int | None) -> None:
        """With the thinking budget applied: refuse a served context that
        cannot hold it (every request would be rejected), and warn when
        little room is left for the prompt."""
        if max_model_len is None:
            return
        room = max_model_len - THINKING_BUDGET
        if room <= 0:
            raise SystemExit(
                f"lm-eval: thinking=on gives chat tasks a {THINKING_BUDGET}-token budget, but the "
                f"server's max_model_len is {max_model_len}: every request would be rejected; raise "
                "max_model_len, set gen_kwargs=max_gen_toks=<n>, or use thinking=off"
            )
        if room < PROMPT_ROOM:
            log.warning(
                "lm-eval: max_model_len %d leaves %d tokens for the prompt beside the %d-token "
                "thinking budget; a longer prompt (few-shot turns) is rejected and counts as failed",
                max_model_len,
                room,
                THINKING_BUDGET,
            )

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
        return _base_lm_class()(
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
        task_objects, groups = self.task_objects, self.groups
        primary = self.primary
        mode_groups = self.mode_groups
        budget_tasks = self.apply_thinking_budget(
            task_objects, mode_groups.get("chat", [])
        )
        if budget_tasks:
            self.check_context(served_max_model_len(base_url, served_model_name))
        passes, split = plan_passes(
            self.task_list,
            task_objects,
            groups,
            {task: mode for mode, tasks in mode_groups.items() for task in tasks},
        )
        for name in split:
            log.warning(
                "lm-eval: group %s spans chat and base tasks: its parts run in their own "
                "modes (overrides kept), but no aggregate is computed for it; set mode=base "
                "to run it whole (generation then without the chat template)",
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
        policy = failure_policy(
            self.id,
            "responses",
            sum(c["empty"] + c["errors"] for c in responses.values()),
            sum(c["docs"] for c in responses.values()),
            self.max_failed_frac,
        )
        if unscorable := [m for r in per_repeat for m in r.get("unscorable", [])]:
            raise SystemExit(
                "lm-eval: " + "; ".join(dict.fromkeys(unscorable)) + "; rerun to retry the failed requests"
            )
        values = [r["value"] for r in per_repeat if r["value"] is not None]
        return {
            "value": sum(values) / len(values) if values else None,
            # the selection's size (the same every repeat); each repeat's own
            # n counts the docs it scored
            "n": per_repeat[0]["n_selected"],
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
            "thinking_switch": self.thinking_switch,
            "thinking_kwargs": (
                self.thinking_kwargs() if self.thinking_switch is not None else {}
            ),
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
    ) -> dict[str, Any]:
        repeat_dir = self.config.output_dir / f"repeat-{k}"
        lm_seed = self.config.seed + 1234 + k
        cache_dir = repeat_dir / "lm_cache" / cache_key(self.config.model, served, lm_seed)
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
                model=self.make_lm(mode, base_url, served, seed=lm_seed),
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
        found = failed_docs(
            merged.get("samples") or {},
            {n for n, t in task_objects.items() if t.OUTPUT_TYPE == "generate_until"},
        )
        summary["responses"] = {
            task: {key: len(ids) for key, ids in counts.items()}
            for task, counts in found.items()
        }
        leave_out_failed(summary, merged.get("samples") or {}, task_objects, found)
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
