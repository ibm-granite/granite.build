"""Chat family: MMLU-ProX lite (IFBench lives in the ifbench family).

MMLU-ProX lite: exact match (custom-extract)
---------------------------------------------
lm-evaluation-harness's own ``mmlu_prox_lite_{lang}_{subject}`` tasks (5-shot
CoT prompts, the ``custom-extract`` regex filter, ``exact_match``), sent to the
served model in lm-eval's chat mode, as NeMo Evaluator's lm-eval container runs
its ``mmlu_prox_*_chat`` tasks and as Granite 4.2's card says its numbers were
produced ("an evaluation framework based on NeMo Evaluator SDK"):
``local-chat-completions --apply_chat_template --fewshot_as_multiturn``.
The task's few-shot target is empty (each shot's text carries its worked
answer), so lm-eval sends the description as the system turn, the 5 shots as 5
user turns and the question as the last user turn; that is the harness's
behaviour, kept as is.

The default language scope is Nemo-Gym's official 6 (en, de, es, fr, it, ja),
i.e. 6 languages x 588 test questions. ``--option languages=extended`` selects the
earlier 11-language reading (adds pt, ar, cs, ko, zh) kept for comparison;
``--option languages=all`` (or a comma list) changes it further.

The headline value is the mean over languages of each language's exact match
(lm-eval's ``mmlu_prox_lite_{lang}`` group: size-weighted over subjects). All
languages have the same 588 questions, so this is also the micro average.
With ``--repeats k`` it is the mean of the k repeats' values (pass@1[avg-of-k])
and ``details.pass_at_k`` adds pass@k (an example matched in any repeat), both
averaged over languages the same way.

Generation uses Granite 4.2's card settings for thinking mode, not the task
configs: ``temperature=1.0``, ``top_p=0.95``, ``max_tokens=8192``, thinking left
on (see ``CARD_SAMPLING``). This deviates from NeMo Evaluator's lm-eval chat
protocol (the task's ``max_gen_toks: 2048``, ``temperature 1e-7``/``top_p
0.9999999``, the task's stop strings): with thinking on, 2048 tokens cut off
nearly every reasoning trace, so that protocol scores the 3b near 0 against the
card's 27.78. ``--option temperature=/top_p=/max_tokens=`` override the card
values (``max_tokens=2048 temperature=0.0000001 top_p=0.9999999`` is the NeMo
lm-eval chat protocol); ``--option thinking=off`` sends
``chat_template_kwargs.enable_thinking=false``.

``--option stop=`` replaces the task's stop strings (lm-eval's ``until``):
``stop=task`` (default) keeps each language's own four (``</s>``, ``Q:``, the
language's "Question:" word, ``<|im_end|>``); ``stop=none`` sends none (only
the tokenizer's EOS, which lm-eval always appends); anything else is a
comma-separated list of at most 4 strings, each percent-decoded, so the value
has no spaces or shell characters and survives bv-smoke's space-split OPTIONS
and its inner ``bash -c``: ``stop=%3C/s%3E,Q:`` is ``["</s>", "Q:"]``, ``%2C``
is a comma, ``%20`` a space. The stops the task actually used (per language)
are recorded in results.json as ``generation_kwargs``.
Per request, the samples record what the server said about the generation
(``granite_request``: ``finish_reason`` "length" or "stop", vLLM's
``stop_reason``, i.e. the stop string that matched, or None for EOS,
``completion_tokens``, which include the thinking, and the length of the
reasoning the parser split off). They come from ``repeat-<k>/requests.jsonl``,
appended as responses arrive, so a resumed run keeps them for cached
responses. results.json summarises them per repeat (``requests``). Only
lm-eval's concurrent path records them (``--workers`` > 1, the default 64);
with one worker the rows say None.

The served model's reasoning parser keeps the thinking out of ``content``, which
is what the regex sees. A thought cut off at ``max_tokens`` leaves no content
and scores as wrong, like in the reference protocol.

Resume: lm-eval's response cache (``repeat-<k>/lm_cache``) keeps every
finished request, so a restarted run only sends the missing ones. A request
that keeps failing is answered with an error marker, scored wrong, counted in
``statuses`` and dropped from the cache, so the next run retries it.
``--option answers=gold`` answers every question with its reference letter in
the language's answer phrase: it checks data, prompts, extraction and scoring
without a model and must score 1.0.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any, ClassVar

from granite_evals import data
from granite_evals.registry import Benchmark, pass_at_k, pass_at_k_record, register

log = logging.getLogger(__name__)

# Granite 4.x supported languages that MMLU-ProX has (no nl), in suite order.
EXTENDED_LANGUAGES = ("en", "de", "es", "fr", "ja", "pt", "ar", "cs", "it", "ko", "zh")
# Nemo-Gym's official MMLU-ProX scope: 6 languages, not the extended 11.
OFFICIAL_LANGUAGES = ("en", "de", "es", "fr", "it", "ja")
ALL_LANGUAGES = (
    "af", "ar", "bn", "cs", "de", "en", "es", "fr", "hi", "hu", "id", "it", "ja", "ko", "mr",
    "ne", "pt", "ru", "sr", "sw", "te", "th", "uk", "ur", "vi", "wo", "yo", "zh", "zu",
)  # fmt: skip
# lm-eval task suffix -> the dataset's ``category`` value (lm_eval/tasks/mmlu_prox/*/utils.py).
SUBJECTS = {
    "biology": "biology",
    "business": "business",
    "chemistry": "chemistry",
    "computer_science": "computer science",
    "economics": "economics",
    "engineering": "engineering",
    "health": "health",
    "history": "history",
    "law": "law",
    "math": "math",
    "other": "other",
    "philosophy": "philosophy",
    "physics": "physics",
    "psychology": "psychology",
}
FILTER = "custom-extract"
METRIC = "exact_match"
ERROR_MARKER = "[granite-evals: request failed]"
PROGRESS_EVERY = 100
RETRY_BACKOFF_S = 1.0  # doubles per attempt, capped at 30 s
# Granite 4.2 recommended sampling for thinking mode (thinking is on by default):
# the 3b model card, huggingface.co/ibm-granite/granite-4.2-3b README.md at
# e459acce, lines 331-341 (temperature 1.0, top_p 0.95, max_new_tokens 8192 in
# thinking mode, 2048 in non-thinking mode) and line 347 (thinking on by default).
# NeMo Evaluator's lm-eval chat defaults differ (nvidia-lm-eval 26.3
# core_evals/lm_evaluation_harness/framework.yml:21-23: max_new_tokens null, i.e.
# the task's 2048, temperature 1e-7, top_p 0.9999999); see the module docstring.
CARD_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 8192}
MAX_STOPS = 4  # lm-eval's chat payload keeps until[:4] (openai_completions.py)


def task_name(lang: str, subject: str) -> str:
    return f"mmlu_prox_lite_{lang}_{subject}"


def parse_task(name: str) -> tuple[str, str]:
    """``mmlu_prox_lite_de_computer_science`` -> ("de", "computer_science")."""
    rest = name.removeprefix("mmlu_prox_lite_")
    lang, _, subject = rest.partition("_")
    return lang, subject


def select_examples(rows_by_lang: dict[str, list[dict]], limit: int | None) -> dict[str, list[int]]:
    """lm-eval ``samples``: per task, the doc indices of the first ``limit``
    examples, ordered by (question_id, language order). An example is one
    question in one language, so a small limit still covers every language.
    Indices are positions in the task's docs: the language's test split
    filtered to the subject, order kept (lm-eval's ``process_docs``)."""
    langs = list(rows_by_lang)
    examples = []
    for lang, rows in rows_by_lang.items():
        seen: dict[str, int] = {}
        for row in rows:
            subject = next(s for s, c in SUBJECTS.items() if c == row["category"])
            idx = seen.get(subject, 0)
            seen[subject] = idx + 1
            examples.append({"order": (row["question_id"], langs.index(lang)), "task": task_name(lang, subject), "idx": idx})
    samples: dict[str, list[int]] = {}
    for ex in data.take(examples, limit, key="order"):
        samples.setdefault(ex["task"], []).append(ex["idx"])
    return {t: sorted(v) for t, v in sorted(samples.items())}


def sample_status(response: str, extracted: str) -> str:
    if response == ERROR_MARKER:
        return "error"
    if not response or response.startswith("LMEVAL_MODEL_NONE_ANSWER"):
        return "no_content"  # e.g. the thought ran into max_tokens
    if extracted == "[invalid]":
        return "no_answer"
    return "answered"


def summarize(samples: dict[str, list[dict]]) -> dict[str, Any]:
    """Per-language and headline exact match from lm-eval's logged samples
    (``custom-extract`` filter only)."""
    per_lang: dict[str, list[float]] = {}
    per_subject: dict[str, list[float]] = {}
    statuses: dict[str, int] = {}
    for task, rows in samples.items():
        lang, subject = parse_task(task)
        for row in rows:
            if row.get("filter") != FILTER:
                continue
            score = float(row[METRIC])
            per_lang.setdefault(lang, []).append(score)
            per_subject.setdefault(subject, []).append(score)
            resp = row["resps"][0][0] if row.get("resps") else ""
            status = sample_status(resp, row["filtered_resps"][0] if row.get("filtered_resps") else "")
            statuses[status] = statuses.get(status, 0) + 1
    by_lang = {lang: {"exact_match": sum(v) / len(v), "n": len(v)} for lang, v in per_lang.items()}
    return {
        "value": sum(x["exact_match"] for x in by_lang.values()) / len(by_lang) if by_lang else 0.0,
        "n": sum(x["n"] for x in by_lang.values()),
        "per_language": by_lang,
        "per_subject": {s: {"exact_match": sum(v) / len(v), "n": len(v)} for s, v in sorted(per_subject.items())},
        "statuses": statuses,
    }


def sample_scores(samples: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    """Per language, each example's exact match (``custom-extract``), keyed
    ``<task>/<doc_id>``: one repeat's input to pass@k."""
    out: dict[str, dict[str, float]] = {}
    for task, rows in samples.items():
        lang, _ = parse_task(task)
        for row in rows:
            if row.get("filter") == FILTER:
                out.setdefault(lang, {})[f"{task}/{row['doc_id']}"] = float(row[METRIC])
    return out


def prox_pass_at_k(repeats: list[dict[str, dict[str, float]]], k: int) -> dict[str, Any]:
    """``details.pass_at_k`` over repeats' ``sample_scores``, macro-averaged over
    languages as the headline is: pass@1[avg-of-k] and pass@k (matched in any
    repeat) per language, then their mean."""
    langs = sorted({lang for r in repeats for lang in r})
    recs = []
    for lang in langs:
        keys = sorted({key for r in repeats for key in r.get(lang, {})})
        recs.append(pass_at_k([[r[lang][key] for r in repeats if key in r.get(lang, {})] for key in keys], k))
    how = "mean over languages; an example counts if it matched in any repeat"
    if not recs:
        return pass_at_k_record(k, None, None, 0, how)
    return pass_at_k_record(k, sum(x["pass_at_1"] for x in recs) / len(recs), sum(x["pass_at_k"] for x in recs) / len(recs),
                            sum(x["n"] for x in recs), how)  # fmt: skip


def effective_generation_kwargs(configs: dict[str, dict]) -> dict[str, dict]:
    """Per language, the generation_kwargs lm-eval ran the tasks with (the task
    config after the overrides; ``until`` is the stop list sent, before
    lm-eval appends the tokenizer's EOS and keeps the first 4). One entry per
    language unless its subject tasks disagree (then also per task)."""
    out: dict[str, dict] = {}
    for task, cfg in sorted(configs.items()):
        if not task.startswith("mmlu_prox_lite_"):
            continue
        lang, _ = parse_task(task)
        gk = cfg.get("generation_kwargs") or {}
        if lang not in out:
            out[lang] = gk
        elif out[lang] != gk:
            out[task] = gk
    return out


def gold_response(lang: str, answer: str) -> str:
    from lm_eval.tasks.mmlu_prox.lang_libs import LANG_LIBS

    return LANG_LIBS[lang][5].format(answer)


REQUESTS_LOG = "requests.jsonl"
_response_meta: contextvars.ContextVar[list | None] = contextvars.ContextVar("granite_response_meta", default=None)


def response_meta(out: dict) -> list[dict]:
    """Per choice of one chat completion: how the generation ended and how
    long it was."""
    usage = out.get("usage") or {}
    metas = []
    for choice in out.get("choices") or []:
        msg = choice.get("message") or {}
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        metas.append({
            "finish_reason": choice.get("finish_reason"),
            "stop_reason": choice.get("stop_reason"),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "reasoning_chars": len(reasoning),
            "content_chars": len(msg.get("content") or ""),
        })  # fmt: skip
    return metas


def request_key(args) -> str:
    """lm-eval's cache key of a generate_until request, ``(context, gen_kwargs)``:
    the same for the request the LM sends and the sample row that logs it."""
    from lm_eval.api.model import hash_args

    return hash_args("generate_until", args)


def load_requests_log(path: Path) -> dict[str, dict]:
    """``requests.jsonl`` by request key; the latest record wins (a retried
    error is replaced by its answer)."""
    out: dict[str, dict] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec.pop("key")] = rec
    return out


def summarize_requests(samples: dict[str, list[dict]]) -> dict[str, Any]:
    """Counts of finish_reason (and of stop_reason for the stopped ones),
    finish_reason of the no-content answers, and completion token stats,
    over the scored rows (``custom-extract``, one per question). Rows with no
    record count as "unrecorded"."""
    finish: dict[str, int] = {}
    stops: dict[str, int] = {}
    no_content: dict[str, int] = {}
    tokens: list[int] = []
    for rows in samples.values():
        for row in rows:
            if row.get("filter") != FILTER:
                continue
            meta = row.get("granite_request")
            reason = str(meta.get("finish_reason")) if meta else "unrecorded"
            finish[reason] = finish.get(reason, 0) + 1
            if meta and meta.get("finish_reason") == "stop":
                sr = meta.get("stop_reason")
                label = "eos" if sr is None else str(sr)
                stops[label] = stops.get(label, 0) + 1
            resp = row["resps"][0][0] if row.get("resps") else ""
            if sample_status(resp, "") == "no_content":
                no_content[reason] = no_content.get(reason, 0) + 1
            if meta and isinstance(meta.get("completion_tokens"), int):
                tokens.append(meta["completion_tokens"])
    return {
        "finish_reason": finish,
        "stop_reason": stops,
        "no_content_finish_reason": no_content,
        "completion_tokens": {
            "n": len(tokens),
            "mean": round(sum(tokens) / len(tokens), 1) if tokens else None,
            "max": max(tokens) if tokens else None,
        },
    }


def _chat_lm_class():
    """lm-eval's local-chat-completions, made robust: bounded retries per
    request inside the call, then an error marker instead of an exception (one
    broken request must not sink ~7k), plus progress log lines."""
    from lm_eval.models.openai_completions import LocalChatCompletion

    class GraniteChatCompletions(LocalChatCompletion):
        def __init__(self, *args, benchmark_id: str = "", requests_log: Path | None = None, **kwargs):
            super().__init__(*args, **kwargs)
            self.benchmark_id = benchmark_id
            self.requests_log = requests_log
            self.done = self.errors = 0

        def parse_generations(self, outputs, **kwargs):
            sink = _response_meta.get()
            if sink is not None:
                for out in outputs if isinstance(outputs, list) else [outputs]:
                    sink.extend(response_meta(out))
            return super().parse_generations(outputs, **kwargs)

        def _log_requests(self, cache_keys, metas: list[dict]) -> None:
            """One requests.jsonl line per request (async path: one request
            per call)."""
            if self.requests_log is None or not cache_keys:
                return
            with self.requests_log.open("a") as f:
                for args, meta in zip(cache_keys, metas, strict=False):
                    f.write(json.dumps({"key": request_key(args), **meta}, ensure_ascii=False) + "\n")

        def _progress(self) -> None:
            if self.done % PROGRESS_EVERY == 0:
                log.info("%s: %d requests done, %d errors", self.benchmark_id, self.done, self.errors)

        async def amodel_call(self, session, sem, messages, *, generate=True, cache_keys=None, ctxlens=None, gen_kwargs=None, **kwargs):
            last: BaseException | None = None
            attempts = max(1, self.max_retries)
            for attempt in range(attempts):
                sink: list[dict] = []
                token = _response_meta.set(sink)
                try:
                    out = await super().amodel_call(
                        session, sem, messages, generate=generate, cache_keys=cache_keys,
                        ctxlens=ctxlens, gen_kwargs=gen_kwargs, **kwargs,
                    )  # fmt: skip
                    self._log_requests(cache_keys, sink)
                    self.done += 1
                    self._progress()
                    return out
                except Exception as e:  # noqa: BLE001 - retried, then scored as failed
                    last = e
                    if attempt + 1 < attempts:
                        await asyncio.sleep(min(RETRY_BACKOFF_S * 2**attempt, 30))
                finally:
                    _response_meta.reset(token)
            self._log_requests(cache_keys, [{"finish_reason": "error", "error": repr(last)}] * len(cache_keys or []))
            log.warning("%s: request failed after %d attempts: %r", self.benchmark_id, self.max_retries, last)
            self.done += 1
            self.errors += 1
            self._progress()
            return [ERROR_MARKER]

    return GraniteChatCompletions


def _gold_lm_class():
    """Answers each request with its reference letter (``answers=gold``)."""
    from lm_eval.api.model import LM

    class GoldLM(LM):
        def generate_until(self, requests, disable_tqdm: bool = False):
            out = []
            for req in requests:
                lang, _ = parse_task(req.task_name)
                out.append(gold_response(lang, req.doc["answer"]))
            return out

        def loglikelihood(self, requests, disable_tqdm: bool = False):
            raise NotImplementedError

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
            raise NotImplementedError

        def apply_chat_template(self, chat_history, add_generation_prompt: bool = True):
            return json.dumps(chat_history, ensure_ascii=False)

        @property
        def tokenizer_name(self) -> str:
            return "gold"

    return GoldLM


def drop_errors_from_cache(cache_dir: Path) -> int:
    """Delete cached error markers so a restart retries those requests."""
    from sqlitedict import SqliteDict

    dropped = 0
    for db in cache_dir.glob("*.db"):
        with SqliteDict(str(db), autocommit=True) as d:
            for key in [k for k, v in d.items() if v == ERROR_MARKER]:
                del d[key]
                dropped += 1
    return dropped


@register
class MMLUProXLite(Benchmark):
    id = "mmlu-prox-lite"
    metric = "exact match (custom-extract)"
    default_repeats = 1
    extra = "lmeval"
    harness_packages = ("lm-eval",)
    dataset = "li-lab/MMLU-ProX-Lite"
    dataset_revision = "e82aafb9460529687d3c7e51b401d8dd1dd309dd"
    split: ClassVar[str] = "test"

    def opt(self, key: str, default: Any) -> Any:
        value = self.config.options.get(key)
        return default if value is None else type(default)(value)

    @property
    def gold(self) -> bool:
        return self.opt("answers", "model") == "gold"

    def needs_server(self) -> bool:
        return not self.gold

    def languages(self) -> list[str]:
        spec = self.opt("languages", "official")
        if spec == "official":
            return list(OFFICIAL_LANGUAGES)
        if spec == "extended":
            return list(EXTENDED_LANGUAGES)
        if spec == "all":
            return list(ALL_LANGUAGES)
        langs = [x.strip() for x in spec.split(",") if x.strip()]
        unknown = sorted(set(langs) - set(ALL_LANGUAGES))
        if unknown:
            raise SystemExit(f"{self.id}: unknown languages {unknown}; MMLU-ProX has {', '.join(ALL_LANGUAGES)}")
        return langs

    def task_spec(self, tasks: list[str], source: str, revision: str | None) -> dict:
        """One inline lm-eval group over the upstream subject tasks, each
        pinned to ``source``@``revision`` (lm-eval passes ``dataset_kwargs`` to
        ``datasets.load_dataset``)."""
        override: dict[str, Any] = {"dataset_kwargs": {"revision": revision} if revision else {}}
        if source != self.dataset:
            override["dataset_path"] = source
        return {"group": "granite_mmlu_prox_lite", "task": [{"task": t, **override} for t in tasks]}

    def stop(self) -> list[str] | None:
        """``--option stop=``: None keeps the task's own stop strings."""
        spec = self.opt("stop", "task")
        if spec == "task":
            return None
        if spec == "none":
            return []
        stops = [urllib.parse.unquote(x) for x in spec.split(",")]
        if any(not x for x in stops):
            raise SystemExit(f"{self.id}: empty stop string in stop={spec!r}; use stop=none for no stops")
        if len(stops) > MAX_STOPS:
            raise SystemExit(f"{self.id}: {len(stops)} stop strings; lm-eval sends at most {MAX_STOPS}")
        return stops

    def gen_kwargs(self) -> dict[str, Any]:
        """Overrides of the task's generation_kwargs: the card's sampling
        (``CARD_SAMPLING``) unless an option overrides it, plus ``stop`` and
        ``thinking``. Gold mode ignores them.

        ``do_sample`` stays the task's ``false``: local-chat-completions drops
        it from the request (the server samples per ``temperature``), and
        lm-eval's response cache skips every ``do_sample=True`` request, which
        would make a restarted run resend all of them. The cache is per
        repeat (``repeat-<k>/lm_cache``), so repeats still sample afresh."""
        kw: dict[str, Any] = {
            "temperature": float(self.config.options.get("temperature", CARD_SAMPLING["temperature"])),
            "top_p": float(self.config.options.get("top_p", CARD_SAMPLING["top_p"])),
            "max_gen_toks": int(self.config.options.get("max_tokens", CARD_SAMPLING["max_tokens"])),
        }
        if (stops := self.stop()) is not None:
            kw["until"] = stops
        if self.opt("thinking", "default") == "off":
            kw["chat_template_kwargs"] = {"enable_thinking": False}
        return kw

    def make_lm(self, base_url: str, served: str, seed: int, requests_log: Path | None = None):
        if self.gold:
            return _gold_lm_class()()
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
            requests_log=requests_log,
        )

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        source, revision = self.dataset_source()
        langs = self.languages()
        samples = None
        if self.config.limit is not None:
            rows = {lang: data.load_split(source, revision=revision, split=self.split, name=lang) for lang in langs}
            samples = select_examples(rows, self.config.limit)
            tasks = list(samples)
        else:
            tasks = [task_name(lang, s) for lang in langs for s in SUBJECTS]
        log.info("%s: %d languages, %d tasks, limit=%s", self.id, len(langs), len(tasks), self.config.limit)

        per_repeat, scores = [], []
        for k in range(self.repeats):
            summary = self._repeat(k, tasks, samples, source, revision, base_url, served_model_name)
            scores.append(summary.pop("sample_scores"))
            per_repeat.append({"repeat": k, **summary})
            log.info("%s repeat %d: exact_match %.4f over %d (%s)", self.id, k, summary["value"], summary["n"], summary["statuses"])

        return {
            "value": sum(r["value"] for r in per_repeat) / len(per_repeat),
            "n": per_repeat[0]["n"],
            "dataset": source,
            "dataset_revision": revision,
            "languages": langs,
            "mode": "gold" if self.gold else "local-chat-completions, apply_chat_template, fewshot_as_multiturn",
            "gen_kwargs_overrides": self.gen_kwargs(),
            "stop": "task" if self.stop() is None else self.stop(),
            "generation_kwargs": per_repeat[0]["generation_kwargs"],
            "pass_at_k": prox_pass_at_k(scores, self.repeats),
            "per_repeat": per_repeat,
        }

    def _repeat(self, k, tasks, samples, source, revision, base_url, served) -> dict[str, Any]:
        from lm_eval import simple_evaluate
        from lm_eval.tasks import TaskManager

        repeat_dir = self.config.output_dir / f"repeat-{k}"
        cache_dir = repeat_dir / "lm_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        results = simple_evaluate(
            model=self.make_lm(base_url, served, seed=self.config.seed + 1234 + k, requests_log=repeat_dir / REQUESTS_LOG),
            tasks=[self.task_spec(tasks, source, revision)],
            task_manager=TaskManager(),
            samples=samples,
            use_cache=None if self.gold else str(cache_dir / "cache"),
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            gen_kwargs=self.gen_kwargs(),
            log_samples=True,
            bootstrap_iters=0,
            random_seed=self.config.seed,
        )
        if not self.gold:
            if dropped := drop_errors_from_cache(cache_dir):
                log.warning("%s repeat %d: %d failed requests dropped from the cache for retry", self.id, k, dropped)
        self._write(repeat_dir, results)
        summary = summarize(results["samples"])
        summary["sample_scores"] = sample_scores(results["samples"])
        if not self.gold:
            summary["requests"] = summarize_requests(results["samples"])
        summary["generation_kwargs"] = effective_generation_kwargs(results.get("configs") or {})
        summary["duration_s"] = round(time.time() - started, 1)
        return summary

    def _write(self, repeat_dir: Path, results: dict) -> None:
        """Per-example artifacts: ``samples/<task>.jsonl`` (each row with its
        ``granite_request`` record, None if the log has none) and lm-eval's
        per-task/group results as ``lm_eval_results.json``."""
        sdir = repeat_dir / "samples"
        sdir.mkdir(exist_ok=True)
        records = load_requests_log(repeat_dir / REQUESTS_LOG)
        for task, rows in results["samples"].items():
            with (sdir / f"{task}.jsonl").open("w") as f:
                for row in rows:
                    if records and row.get("arguments"):
                        row["granite_request"] = records.get(request_key(row["arguments"][0]))
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        keep = {key: results.get(key) for key in ("results", "groups", "configs", "versions", "n-shot", "n-samples", "config")}
        (repeat_dir / "lm_eval_results.json").write_text(json.dumps(keep, indent=2, ensure_ascii=False, default=str))
