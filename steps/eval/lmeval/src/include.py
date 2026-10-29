"""LMEval family: INCLUDE-base-44 (MMLU-ProX lite lives in chat_if.py).

INCLUDE-base-44: accuracy (multiple-choice loglikelihood)
---------------------------------------------------------
Cohere Labs' INCLUDE-base-44, run through lm-evaluation-harness's own
``include_base_44_{lang}_{domain}`` tasks (44 languages x 11 domains, 178
tasks, ~500-550 test questions per language) in the harness's base mode: no
chat template, one loglikelihood request per option letter, ``acc`` = the
argmax letter against the row's ``answer`` index (lm-eval's standard
multiple-choice protocol, what ``lm_eval --model local-completions`` runs on
these tasks). The prompt is the task's per-domain description, the question
with its four options and ``Answer:``; the scored continuations are
``" A"``-``" D"``.

The shipped task yamls read ``CohereForAI/include-base-44``, a repo since
renamed to ``CohereLabs/include-base-44``; the inline group below re-points
every task at the current repo and pins the revision (``dataset_path`` +
``dataset_kwargs``), so nothing depends on the stale upstream id.

The default scope is all 44 languages. The headline is the mean over
languages of each language's accuracy; the languages have different sizes,
so the micro average is also reported (``micro``), and ``per_domain`` breaks
the same questions down by domain. With ``--repeats k`` the headline is
pass@1[avg-of-k] and ``details.pass_at_k`` adds pass@k (an example correct
in any repeat), both averaged over languages the same way.

``--limit N`` picks the first N examples ordered by (in-task index, language
order): every language's first questions first, so a small limit covers the
first languages' tasks (LIMIT=5 sees Albanian's five). Indices are positions
in each task's docs: the language's test split filtered to the domain, order
kept (lm-eval's ``process_docs``).

Resume: lm-eval's response cache (``repeat-<k>/lm_cache``) keeps every
finished request, so a restarted run only sends the missing ones. A request
that keeps failing is answered with an error tuple, scored as a very
unlikely choice (so the question is usually wrong; if all four options fail,
the argmax falls to A, which can luck into the right answer), counted in
``statuses`` and dropped from the cache, so the next run retries it.

``--option languages=`` takes a comma list of the language slugs to subset
the default 44. ``--option answers=gold`` answers each option letter with
its reference loglikelihood (the answer's letter 0.0, the others -100): it
checks data, prompts and scoring without a model and must score 1.0.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, ClassVar

from granite_evals import data
from granite_evals.registry import Benchmark, pass_at_k, pass_at_k_record, register

log = logging.getLogger(__name__)

# lm-eval task suffix -> the dataset's ``domain`` value (lm_eval/tasks/include/utils.py).
DOMAINS = {
    "applied_science": "Applied Science",
    "arts_humanities": "Arts & Humanities",
    "business_commerce": "Business & Commerce",
    "driving_license": "Driving License",
    "general_knowledge": "General knowledge",
    "health_oriented_education": "Health oriented education",
    "marine_license": "Marine License",
    "medical_license": "Medical License",
    "professional_certification": "Professional certification",
    "stem": "STEM",
    "social_science": "Social Science",
}
DOMAIN_OF_CATEGORY = {category: slug for slug, category in DOMAINS.items()}
# Language slug -> (dataset config, domains with a task), from lm-eval's include
# task yamls: 44 languages, 178 language x domain tasks.
LANGUAGES = {
    "albanian": ("Albanian", ("arts_humanities", "business_commerce", "health_oriented_education", "social_science", "stem")),
    "arabic": ("Arabic", ("arts_humanities", "business_commerce", "driving_license", "general_knowledge", "social_science", "stem")),
    "armenian": ("Armenian", ("arts_humanities", "driving_license", "social_science", "stem")),
    "azerbaijani": (
        "Azerbaijani",
        ("applied_science", "arts_humanities", "business_commerce", "health_oriented_education", "social_science", "stem"),
    ),
    "basque": ("Basque", ("professional_certification",)),
    "belarusian": ("Belarusian", ("arts_humanities", "social_science", "stem")),
    "bengali": ("Bengali", ("arts_humanities", "general_knowledge", "professional_certification", "stem")),
    "bulgarian": ("Bulgarian", ("arts_humanities", "social_science", "stem")),
    "chinese": (
        "Chinese",
        ("applied_science", "arts_humanities", "business_commerce", "driving_license", "health_oriented_education",
         "professional_certification", "social_science", "stem"),
    ),
    "croatian": ("Croatian", ("arts_humanities", "social_science", "stem")),
    "dutch": ("Dutch", ("applied_science", "arts_humanities", "health_oriented_education", "social_science", "stem")),
    "estonian": ("Estonian", ("applied_science", "arts_humanities", "health_oriented_education", "social_science", "stem")),
    "finnish": ("Finnish", ("applied_science", "arts_humanities", "health_oriented_education", "social_science", "stem")),
    "french": ("French", ("arts_humanities", "driving_license", "health_oriented_education", "social_science", "stem")),
    "georgian": ("Georgian", ("arts_humanities",)),
    "german": ("German", ("driving_license", "social_science", "stem")),
    "greek": (
        "Greek",
        ("arts_humanities", "business_commerce", "health_oriented_education", "medical_license", "professional_certification",
         "social_science", "stem"),
    ),
    "hebrew": ("Hebrew", ("arts_humanities", "driving_license")),
    "hindi": (
        "Hindi",
        ("applied_science", "arts_humanities", "driving_license", "general_knowledge", "health_oriented_education",
         "professional_certification", "social_science", "stem"),
    ),
    "hungarian": ("Hungarian", ("applied_science", "social_science", "stem")),
    "indonesian": ("Indonesian", ("applied_science", "arts_humanities", "professional_certification", "social_science", "stem")),
    "italian": ("Italian", ("applied_science", "arts_humanities", "health_oriented_education", "professional_certification", "social_science", "stem")),
    "japanese": ("Japanese", ("driving_license", "medical_license", "professional_certification")),
    "kazakh": ("Kazakh", ("arts_humanities",)),
    "korean": ("Korean", ("professional_certification", "social_science")),
    "lithuanian": ("Lithuanian", ("arts_humanities", "business_commerce", "professional_certification", "social_science", "stem")),
    "malay": ("Malay", ("arts_humanities", "business_commerce", "social_science")),
    "malayalam": (
        "Malayalam",
        ("arts_humanities", "general_knowledge", "health_oriented_education", "marine_license", "social_science", "stem"),
    ),
    "nepali": ("Nepali", ("driving_license", "professional_certification")),
    "north_macedonian": (
        "North Macedonian",
        ("arts_humanities", "business_commerce", "social_science", "stem"),
    ),
    "persian": ("Persian", ("arts_humanities", "driving_license", "professional_certification", "social_science", "stem")),
    "polish": ("Polish", ("professional_certification", "social_science", "stem")),
    "portuguese": (
        "Portuguese",
        ("applied_science", "arts_humanities", "business_commerce", "health_oriented_education", "social_science", "stem"),
    ),
    "russian": (
        "Russian",
        ("applied_science", "arts_humanities", "business_commerce", "driving_license", "health_oriented_education",
         "marine_license", "social_science", "stem"),
    ),
    "serbian": ("Serbian", ("arts_humanities", "social_science", "stem")),
    "spanish": ("Spanish", ("arts_humanities", "health_oriented_education", "social_science", "stem")),
    "tagalog": ("Tagalog", ("arts_humanities", "driving_license")),
    "tamil": ("Tamil", ("general_knowledge", "stem")),
    "telugu": ("Telugu", ("applied_science", "arts_humanities", "social_science", "stem")),
    "turkish": ("Turkish", ("arts_humanities", "business_commerce", "social_science", "stem")),
    "ukrainian": ("Ukrainian", ("arts_humanities", "social_science", "stem")),
    "urdu": ("Urdu", ("arts_humanities", "health_oriented_education", "stem")),
    "uzbek": ("Uzbek", ("arts_humanities", "medical_license", "social_science", "stem")),
    "vietnamese": ("Vietnamese", ("arts_humanities", "social_science", "stem")),
}
LETTERS = "ABCD"
FILTER = "none"  # lm-eval's only filter for multiple-choice tasks (take_first)
METRIC = "acc"
ERROR_LL = (-1e9, False)  # answered instead of raising; scored as the worst choice
PROGRESS_EVERY = 100
RETRY_BACKOFF_S = 1.0  # doubles per attempt, capped at 30 s


def task_name(lang: str, domain: str) -> str:
    return f"include_base_44_{lang}_{domain}"


def parse_task(name: str) -> tuple[str, str]:
    """``include_base_44_north_macedonian_social_science`` -> ("north_macedonian", "social_science")."""
    rest = name.removeprefix("include_base_44_")
    for domain in sorted(DOMAINS, key=len, reverse=True):  # no domain is another's suffix
        if rest.endswith("_" + domain):
            return rest[: -len(domain) - 1], domain
    raise ValueError(f"not an INCLUDE task name: {name}")


def select_examples(rows_by_lang: dict[str, list[dict]], limit: int | None) -> dict[str, list[int]]:
    """lm-eval ``samples``: per task, the doc indices of the first ``limit``
    examples, ordered by (in-task index, language order): every language's
    first questions first, so a small limit covers the first languages'
    tasks. Indices are positions in the task's docs: the language's test
    split filtered to the domain, order kept (lm-eval's ``process_docs``)."""
    langs = list(rows_by_lang)
    examples = []
    for lang, rows in rows_by_lang.items():
        seen: dict[str, int] = {}
        for row in rows:
            domain = DOMAIN_OF_CATEGORY[row["domain"]]
            idx = seen.get(domain, 0)
            seen[domain] = idx + 1
            examples.append({"order": (idx, langs.index(lang)), "task": task_name(lang, domain), "idx": idx})
    samples: dict[str, list[int]] = {}
    for ex in data.take(examples, limit, key="order"):
        samples.setdefault(ex["task"], []).append(ex["idx"])
    return {t: sorted(v) for t, v in sorted(samples.items())}


def sample_status(row: dict) -> str:
    """``error`` if any of the four option requests failed, else ``answered``."""
    resps = row.get("filtered_resps") or []
    if any(r == ERROR_LL for r in resps):
        return "error"
    return "answered"


def summarize(samples: dict[str, list[dict]]) -> dict[str, Any]:
    """Per-language, per-domain and headline accuracy from lm-eval's logged
    samples (the ``none`` filter)."""
    per_lang: dict[str, list[float]] = {}
    per_domain: dict[str, list[float]] = {}
    statuses: dict[str, int] = {}
    for task, rows in samples.items():
        lang, domain = parse_task(task)
        for row in rows:
            if row.get("filter") != FILTER:
                continue
            score = float(row[METRIC])
            per_lang.setdefault(lang, []).append(score)
            per_domain.setdefault(domain, []).append(score)
            status = sample_status(row)
            statuses[status] = statuses.get(status, 0) + 1
    scores = [s for v in per_lang.values() for s in v]
    by_lang = {lang: {"acc": sum(v) / len(v), "n": len(v)} for lang, v in per_lang.items()}
    return {
        "value": sum(x["acc"] for x in by_lang.values()) / len(by_lang) if by_lang else 0.0,
        "n": len(scores),
        "micro": {"acc": sum(scores) / len(scores) if scores else 0.0, "n": len(scores)},
        "per_language": by_lang,
        "per_domain": {d: {"acc": sum(v) / len(v), "n": len(v)} for d, v in sorted(per_domain.items())},
        "statuses": statuses,
    }


def sample_scores(samples: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    """Per language, each example's accuracy, keyed ``<task>/<doc_id>``: one
    repeat's input to pass@k."""
    out: dict[str, dict[str, float]] = {}
    for task, rows in samples.items():
        lang, _ = parse_task(task)
        for row in rows:
            if row.get("filter") == FILTER:
                out.setdefault(lang, {})[f"{task}/{row['doc_id']}"] = float(row[METRIC])
    return out


def include_pass_at_k(repeats: list[dict[str, dict[str, float]]], k: int) -> dict[str, Any]:
    """``details.pass_at_k`` over repeats' ``sample_scores``, macro-averaged over
    languages as the headline is: pass@1[avg-of-k] and pass@k (correct in any
    repeat) per language, then their mean."""
    langs = sorted({lang for r in repeats for lang in r})
    recs = []
    for lang in langs:
        keys = sorted({key for r in repeats for key in r.get(lang, {})})
        recs.append(pass_at_k([[r[lang][key] for r in repeats if key in r.get(lang, {})] for key in keys], k))
    how = "mean over languages; an example counts if it was correct in any repeat"
    if not recs:
        return pass_at_k_record(k, None, None, 0, how)
    return pass_at_k_record(k, sum(x["pass_at_1"] for x in recs) / len(recs), sum(x["pass_at_k"] for x in recs) / len(recs),
                            sum(x["n"] for x in recs), how)  # fmt: skip


def _lm_class():
    """lm-eval's local-completions, made robust: bounded retries per request
    inside the call, then an error tuple instead of an exception (one broken
    request must not sink ~20k), plus progress log lines."""
    from lm_eval.models.openai_completions import LocalCompletionsAPI

    class GraniteCompletions(LocalCompletionsAPI):
        def __init__(self, *args, benchmark_id: str = "", **kwargs):
            super().__init__(*args, **kwargs)
            self.benchmark_id = benchmark_id
            self.done = self.errors = 0

        def _progress(self) -> None:
            if self.done % PROGRESS_EVERY == 0:
                log.info("%s: %d requests done, %d errors", self.benchmark_id, self.done, self.errors)

        async def amodel_call(self, session, sem, messages, *, generate=True, cache_keys=None, ctxlens=None, gen_kwargs=None, **kwargs):
            last: BaseException | None = None
            attempts = max(1, self.max_retries)
            for attempt in range(attempts):
                try:
                    out = await super().amodel_call(
                        session, sem, messages, generate=generate, cache_keys=cache_keys,
                        ctxlens=ctxlens, gen_kwargs=gen_kwargs, **kwargs,
                    )  # fmt: skip
                    self.done += 1
                    self._progress()
                    return out
                except Exception as e:  # noqa: BLE001 - retried, then scored as failed
                    last = e
                    if attempt + 1 < attempts:
                        await asyncio.sleep(min(RETRY_BACKOFF_S * 2**attempt, 30))
            log.warning("%s: request failed after %d attempts: %r", self.benchmark_id, self.max_retries, last)
            self.done += 1
            self.errors += 1
            self._progress()
            return [ERROR_LL] * len(messages)

    return GraniteCompletions


def _gold_lm_class():
    """Answers each option letter with its reference loglikelihood
    (``answers=gold``): the answer's letter is the argmax. The other letters
    score -100, not the error tuple, so gold rows do not count as errors."""
    from lm_eval.api.model import LM

    class GoldLM(LM):
        def generate_until(self, requests, disable_tqdm: bool = False):
            raise NotImplementedError

        def loglikelihood(self, requests, disable_tqdm: bool = False):
            out = []
            for req in requests:
                gold = LETTERS[int(req.doc["answer"])]
                out.append((0.0, True) if req.args[1].strip() == gold else (-100.0, False))
            return out

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
            raise NotImplementedError

        @property
        def tokenizer_name(self) -> str:
            return "gold"

    return GoldLM


def drop_errors_from_cache(cache_dir: Path) -> int:
    """Delete cached error tuples so a restart retries those requests."""
    from sqlitedict import SqliteDict

    dropped = 0
    for db in cache_dir.glob("*.db"):
        with SqliteDict(str(db), autocommit=True) as d:
            for key in [k for k, v in d.items() if v == ERROR_LL]:
                del d[key]
                dropped += 1
    return dropped


@register
class Include44(Benchmark):
    id = "include-44"
    metric = "accuracy (0-shot loglikelihood)"
    default_repeats = 1
    extra = "lmeval"
    harness_packages = ("lm-eval",)
    dataset = "CohereLabs/include-base-44"
    dataset_revision = "d2e1f6015f67a43c02a9a68db98e2298e2d6a660"
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
        """``--option languages=``: all 44 by default, or a comma list of slugs."""
        spec = self.opt("languages", "all")
        if spec == "all":
            return list(LANGUAGES)
        langs = [x.strip() for x in spec.split(",") if x.strip()]
        unknown = sorted(set(langs) - set(LANGUAGES))
        if unknown:
            raise SystemExit(f"{self.id}: unknown languages {unknown}; INCLUDE-base-44 has {len(LANGUAGES)}: {', '.join(LANGUAGES)}")
        return langs

    def task_spec(self, tasks: list[str], source: str, revision: str | None) -> dict:
        """One inline lm-eval group over the upstream language x domain tasks,
        each pinned to ``source``@``revision``: the shipped yamls read the
        dataset's old ``CohereForAI`` org (renamed to ``CohereLabs``), so
        every task carries the ``dataset_path`` override (lm-eval passes
        ``dataset_kwargs`` to ``datasets.load_dataset``)."""
        override: dict[str, Any] = {
            "dataset_path": source,
            "dataset_kwargs": {"revision": revision} if revision else {},
        }
        return {"group": "granite_include_44", "task": [{"task": t, **override} for t in tasks]}

    def make_lm(self, base_url: str, served: str, seed: int):
        if self.gold:
            return _gold_lm_class()()
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

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        source, revision = self.dataset_source()
        langs = self.languages()
        samples = None
        if self.config.limit is not None:
            rows = {lang: data.load_split(source, revision=revision, split=self.split, name=LANGUAGES[lang][0]) for lang in langs}
            samples = select_examples(rows, self.config.limit)
            tasks = list(samples)
        else:
            tasks = [task_name(lang, dom) for lang in langs for dom in LANGUAGES[lang][1]]
        log.info("%s: %d languages, %d tasks, limit=%s", self.id, len(langs), len(tasks), self.config.limit)

        per_repeat, scores = [], []
        for k in range(self.repeats):
            summary = self._repeat(k, tasks, samples, source, revision, base_url, served_model_name)
            scores.append(summary.pop("sample_scores"))
            per_repeat.append({"repeat": k, **summary})
            log.info("%s repeat %d: acc %.4f over %d (%s)", self.id, k, summary["value"], summary["n"], summary["statuses"])

        return {
            "value": sum(r["value"] for r in per_repeat) / len(per_repeat),
            "n": per_repeat[0]["n"],
            "dataset": source,
            "dataset_revision": revision,
            "languages": langs,
            "mode": "gold" if self.gold else "local-completions loglikelihood, no chat template",
            "pass_at_k": include_pass_at_k(scores, self.repeats),
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
            model=self.make_lm(base_url, served, seed=self.config.seed + 1234 + k),
            tasks=[self.task_spec(tasks, source, revision)],
            task_manager=TaskManager(),
            samples=samples,
            use_cache=None if self.gold else str(cache_dir / "cache"),
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
        summary["duration_s"] = round(time.time() - started, 1)
        return summary

    def _write(self, repeat_dir: Path, results: dict) -> None:
        """Per-example artifacts: ``samples/<task>.jsonl`` and lm-eval's
        per-task/group results as ``lm_eval_results.json``."""
        sdir = repeat_dir / "samples"
        sdir.mkdir(exist_ok=True)
        for task, rows in results["samples"].items():
            with (sdir / f"{task}.jsonl").open("w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        keep = {key: results.get(key) for key in ("results", "groups", "configs", "versions", "n-shot", "n-samples", "config")}
        (repeat_dir / "lm_eval_results.json").write_text(json.dumps(keep, indent=2, ensure_ascii=False, default=str))
