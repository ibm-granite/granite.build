"""BIRD text-to-SQL (Granite: BirdBench), metric "pass@1 execution match".

Protocol: NeMo-Skills' ``birdbench`` benchmark (the NeMo Evaluator backend the
Granite 4.2 model card's BirdBench numbers come from), on the official BIRD dev
set (``dev_20240627``: 1534 questions over 11 SQLite databases):

- data: the official ``dev.zip`` (not on HF with its databases), pinned by
  sha256 and verified on download; cached, never mirrored;
- prompt: NeMo-Skills' ``generic/text_to_sql`` with its schema context (the
  database's SQL dump, INSERT runs cut to 10, as its ``prepare.py`` builds it);
  no evidence ("external knowledge"), as NeMo-Skills (``--option evidence=true``
  prefixes it to the question instead);
- sampling: NeMo-Skills' birdbench GENERATION_ARGS (max_tokens 10000,
  temperature 0.6, top_p 0.95, top_k 20), each overridable by ``--option``;
- extraction: NeMo-Skills' ``BirdEvaluator._extract_answer`` (last ```sql block);
- scoring: BIRD's execution accuracy, ``set(pred rows) == set(gold rows)``, the
  predicted then the gold query on one connection within one 30 s budget, a
  timeout or error scoring 0 (DAMO-ConvAI ``evaluation.py`` / NeMo-Skills
  ``execute_sql``). Executed here in-process with ``sqlite3``: read-only (a
  model's DROP/UPDATE can't corrupt the database for later questions) and with
  the timeout enforced by a progress handler that aborts the query, instead of
  ``func_timeout``, which abandons a running query in a background thread.

``--option sql=gold`` scores the gold SQL against itself (no model), checking
data, databases and scoring end to end. Per-question results are written under
``<output_dir>/repeat-<k>/<question_id>.json`` and reused on restart.

Phases: ``--phase generate`` writes each question's file with the predicted SQL
and ``status: "generated"`` (no execution); ``--phase score`` executes those and
rewrites them with the match, never calling the model. Both need the dev set:
generation builds its prompts' schema context from the databases.

A question whose request fails (after the client's retries) is not a wrong
answer: it is left out of the accuracy, counted in ``details.questions_failed``
/ ``questions_total``, and retried on restart; so is a question the score phase
finds without a generation (``status: "error:not_generated"``). Above ``--option
max_failed_frac`` (default 0.05; 0 = any failure) the run fails; below it the
result is flagged ``incomplete``.

Repeats: 1 by default (pass@1). With ``--repeats k`` the value is the mean of
the k repeats' accuracy (pass@1[avg-of-k]) and ``details.pass_at_k`` adds
pass@k, the fraction of questions matched in any of their scored repeats.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import sqlite3
import time
import zipfile
from pathlib import Path
from typing import Any

from granite_evals import data
from granite_evals.registry import Benchmark, failure_policy, pass_at_k, register

log = logging.getLogger(__name__)

DEV_ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
DEV_ZIP_SHA256 = "cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630"
DEV_DIR = "dev_20240627"
PROMPT_CONFIG = "generic/text_to_sql"
TIMEOUT_S = 30.0  # BIRD's meta_time_out, NeMo-Skills' BirdEvaluatorConfig.timeout
DIFFICULTIES = ("simple", "moderate", "challenging")
NS_SAMPLING = {"max_tokens": 10000, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
"""NeMo-Skills' birdbench GENERATION_ARGS (nemo_skills/dataset/birdbench/__init__.py:18-26):
tokens_to_generate, temperature, top_p, top_k."""
GENERATED = "generated"  # the status of a question generated but not yet executed


@register
class BirdBench(Benchmark):
    id = "birdbench"
    metric = "pass@1 execution match"
    default_repeats = 1
    extra = "bird"
    harness_packages = ("nemo-skills",)
    splittable = True
    dataset = DEV_ZIP_URL
    dataset_revision = f"sha256:{DEV_ZIP_SHA256}"

    def opt(self, key: str, default: Any) -> Any:
        value = self.config.options.get(key)
        if value is None:
            return default
        if isinstance(default, bool):
            return str(value).lower() in ("1", "true", "yes")
        return type(default)(value)

    @property
    def gold(self) -> bool:
        return self.opt("sql", "model") == "gold"

    def needs_server(self) -> bool:
        return not self.gold

    def sampling(self) -> dict[str, Any]:
        """NS_SAMPLING, each overridable by an option (``top_k<=0`` = not sent)."""
        return {key: self.opt(key, default) for key, default in NS_SAMPLING.items()}

    def request_args(self) -> dict[str, Any]:
        """sampling() as chat-completion arguments: top_k is a vLLM extension, sent
        in extra_body and only when positive, as ns does (inference/model/vllm.py:114-127)."""
        args = self.sampling()
        top_k = args.pop("top_k")
        return {**args, "extra_body": {"top_k": top_k}} if top_k > 0 else args

    # -- entry point -------------------------------------------------------

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        dev = self.fetch()
        rows = json.loads((dev / "dev.json").read_text())
        questions = data.take(rows, self.config.limit, key="question_id")
        model = self.generating and not self.gold  # this run asks the model
        schemas = self.schemas(dev) if model else {}
        timeout = self.opt("timeout", TIMEOUT_S)
        log.info("%s: %d questions x %d repeats%s%s", self.id, len(questions), self.repeats,
                 " (gold SQL)" if self.gold else "", "" if self.phase == "all" else f" ({self.phase} phase)")
        client = _client(base_url) if model else None
        extract = _extractor(dev) if model else None
        prompt = _prompt() if model else None

        per_repeat = []
        by_question: dict[Any, list[bool]] = {}  # scored repeats' matches, for pass@k
        for k in range(self.repeats):
            rdir = self.config.output_dir / f"repeat-{k}"
            rdir.mkdir(parents=True, exist_ok=True)

            def one(q: dict) -> dict:
                return self._question(q, rdir, dev, schemas, timeout, client, served_model_name, extract, prompt, k)

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.workers) as pool:
                recs = list(pool.map(one, questions))
            scored = [r for r in recs if not r["status"].startswith("error:")]
            for r in scored:
                by_question.setdefault(r["question_id"], []).append(bool(r.get("correct")))
            per_repeat.append(summarize(scored) | {"repeat": k, "failed": len(recs) - len(scored)})
            log.info("%s repeat %d: %d/%d correct, %d failed", self.id, k, per_repeat[-1]["correct"], len(scored),
                     len(recs) - len(scored))  # fmt: skip

        failed = sum(r["failed"] for r in per_repeat)
        policy = failure_policy(self.id, "questions", failed, len(questions) * self.repeats,
                                self.opt("max_failed_frac", 0.05))  # fmt: skip
        if not self.scoring:
            return {"n": min(r["n"] for r in per_repeat), **policy, "mode": "gold" if self.gold else "model",
                    "sampling": {} if self.gold else self.sampling(),
                    "per_repeat": [{"repeat": r["repeat"], "generated": r["n"], "failed": r["failed"]}
                                   for r in per_repeat]}  # fmt: skip
        if any(r["n"] == 0 for r in per_repeat):
            raise SystemExit(f"{self.id}: a repeat has no scored question; rerun to retry the failed ones")
        return {
            "value": sum(r["accuracy"] for r in per_repeat) / len(per_repeat),
            "n": min(r["n"] for r in per_repeat),  # fewest scored in a repeat: failed questions are left out
            **policy,
            "dataset": DEV_ZIP_URL,
            "dataset_revision": self.dataset_revision,
            "split": f"{DEV_DIR}/dev.json",
            "mode": "gold" if self.gold else "model",
            "evidence": self.opt("evidence", False),
            "timeout_s": timeout,
            "prompt_config": PROMPT_CONFIG,
            "sampling": {} if self.gold else self.sampling(),
            "pass_at_k": pass_at_k(by_question.values(), self.repeats, "matched in any scored repeat"),
            "per_repeat": per_repeat,
        }

    # -- one question ------------------------------------------------------

    def _question(self, q, rdir, dev, schemas, timeout, client, served, extract, prompt, k) -> dict:
        qid = q["question_id"]
        path = rdir / f"{qid}.json"
        rec = json.loads(path.read_text()) if path.exists() else None
        if rec is not None and (rec["status"] != GENERATED or not self.scoring):
            return rec  # scored, or generated and this run does not score
        base = {"question_id": qid, "db_id": q["db_id"], "difficulty": q["difficulty"], "gold_sql": q["SQL"]}
        if rec is None and not self.generating:
            log.warning("%s repeat %d: question %s has no generation", self.id, k, qid)
            return {**base, "correct": False, "status": "error:not_generated"}
        try:
            if rec is not None:
                pass  # generated by an earlier --phase generate: score it
            elif self.gold:
                rec = {**base, "pred_sql": q["SQL"]}
            else:
                messages = prompt.fill({"question": question_text(q, self.opt("evidence", False)),
                                        "sql_context": schemas[q["db_id"]]})
                t0 = time.time()
                resp = client.chat.completions.create(
                    model=served, messages=messages, seed=self.config.seed + k, **self.request_args())
                msg = resp.choices[0].message
                text = msg.content or ""
                rec = {
                    **base,
                    "generation": text,
                    "reasoning": getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None),
                    "finish_reason": resp.choices[0].finish_reason,
                    "usage": resp.usage.model_dump() if resp.usage else None,
                    "gen_s": round(time.time() - t0, 1),
                    "pred_sql": extract(text),
                }
            if self.scoring:
                db = dev / "dev_databases" / q["db_id"] / f"{q['db_id']}.sqlite"
                rec.update(execution_match(rec["pred_sql"], q["SQL"], db, timeout))
            else:
                rec["status"] = GENERATED
        except Exception as e:  # one broken question must not sink the run
            log.exception("%s: question %s failed", self.id, qid)
            # Not persisted: an infrastructure error is retried on resume.
            return {**base, "correct": False, "status": f"error:{type(e).__name__}"}
        path.write_text(json.dumps(rec, indent=2))
        log.info("%s repeat %d: question %s %s", self.id, k, qid, rec["status"])
        return rec

    # -- data --------------------------------------------------------------

    def cache_dir(self) -> Path:
        root = os.environ.get("GRANITE_EVALS_DATA_CACHE") or (
            Path(os.environ["HF_HOME"]) / "granite-data" if os.environ.get("HF_HOME") else Path.home() / ".cache" / "granite-data"
        )
        return Path(root) / "birdbench" / DEV_ZIP_SHA256[:16]

    def fetch(self) -> Path:
        """The extracted dev set (``dev_20240627/``), downloaded and verified once."""
        if self.config.dataset:
            return Path(self.config.dataset)  # a local, already extracted dev dir
        cache = self.cache_dir()
        dev = cache / DEV_DIR
        if (cache / ".complete").exists():
            return dev
        cache.mkdir(parents=True, exist_ok=True)
        with _locked(cache / ".lock"):
            if (cache / ".complete").exists():
                return dev
            zpath = cache / "dev.zip"
            if not zpath.exists() or sha256(zpath) != DEV_ZIP_SHA256:
                download(DEV_ZIP_URL, zpath, DEV_ZIP_SHA256)
            extract_dev(zpath, cache)
            (cache / ".complete").write_text(DEV_ZIP_SHA256 + "\n")
        return dev

    def schemas(self, dev: Path) -> dict[str, str]:
        """NeMo-Skills' per-database schema context, computed once and cached."""
        path = dev.parent / "sql_context.json"
        if path.exists():
            return json.loads(path.read_text())
        from nemo_skills.dataset.birdbench.prepare import read_tables_file

        with _locked(dev.parent / ".lock"):
            if not path.exists():
                log.info("%s: building schema context (sqlite dump) for %s", self.id, dev)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(read_tables_file(str(dev))))
                tmp.replace(path)
        return json.loads(path.read_text())


# -- helpers (no harness import) ------------------------------------------------


def question_text(q: dict, evidence: bool) -> str:
    if evidence and q.get("evidence"):
        return f"{q['evidence']}\n{q['question']}"
    return q["question"]


class _Timeout(Exception):
    pass


def execution_match(pred_sql: str, gold_sql: str, db: Path, timeout: float) -> dict:
    """BIRD execution accuracy: pred then gold on one read-only connection within
    one ``timeout`` budget; correct iff the row sets are equal."""
    deadline = time.monotonic() + timeout
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, check_same_thread=False)
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    t0 = time.monotonic()
    try:
        cur = conn.cursor()
        try:
            pred = cur.execute(pred_sql).fetchall()
        except sqlite3.Error as e:
            if time.monotonic() > deadline:
                return {"correct": False, "status": "timeout", "exec_s": round(time.monotonic() - t0, 2)}
            return {"correct": False, "status": "pred_error", "error": str(e)[:500]}
        try:
            gold = cur.execute(gold_sql).fetchall()
        except sqlite3.Error as e:
            if time.monotonic() > deadline:
                return {"correct": False, "status": "timeout", "exec_s": round(time.monotonic() - t0, 2)}
            return {"correct": False, "status": "gold_error", "error": str(e)[:500]}
    finally:
        conn.close()
    correct = set(pred) == set(gold)
    return {
        "correct": correct,
        "status": "correct" if correct else "wrong",
        "exec_s": round(time.monotonic() - t0, 2),
        "pred_rows": len(pred),
        "gold_rows": len(gold),
    }


def summarize(recs: list[dict]) -> dict:
    correct = sum(bool(r.get("correct")) for r in recs)
    by_diff = {}
    for d in DIFFICULTIES:
        sub = [r for r in recs if r.get("difficulty") == d]
        if sub:
            c = sum(bool(r.get("correct")) for r in sub)
            by_diff[d] = {"n": len(sub), "correct": c, "accuracy": c / len(sub)}
    statuses: dict[str, int] = {}
    for r in recs:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return {
        "n": len(recs),
        "correct": correct,
        "accuracy": correct / len(recs) if recs else 0.0,
        "by_difficulty": by_diff,
        "statuses": dict(sorted(statuses.items())),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected_sha256: str) -> None:
    import httpx

    log.info("downloading %s", url)
    tmp = dest.with_suffix(".partial")
    h = hashlib.sha256()
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                h.update(chunk)
                f.write(chunk)
    if h.hexdigest() != expected_sha256:
        tmp.unlink()
        raise RuntimeError(f"{url}: sha256 {h.hexdigest()} != pinned {expected_sha256}")
    tmp.replace(dest)


def _junk(name: str) -> bool:
    return "__MACOSX" in name or name.rsplit("/", 1)[-1].startswith("._") or name.endswith(".DS_Store")


def extract_dev(zpath: Path, cache: Path) -> None:
    """dev.zip holds dev_20240627/ with a nested dev_databases.zip; extract both,
    skipping macOS metadata entries."""
    with zipfile.ZipFile(zpath) as z:
        z.extractall(cache, members=[m for m in z.namelist() if not _junk(m)])
    dev = cache / DEV_DIR
    inner = dev / "dev_databases.zip"
    with zipfile.ZipFile(inner) as z:
        z.extractall(dev, members=[m for m in z.namelist() if not _junk(m)])
    inner.unlink()


@contextlib.contextmanager
def _locked(path: Path):
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _client(base_url: str):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY", timeout=3600.0, max_retries=3)


def _extractor(dev: Path):
    from nemo_skills.evaluation.evaluator.bird import BirdEvaluator

    return BirdEvaluator({"data_dir": str(dev.parent)})._extract_answer


def _prompt():
    from nemo_skills.prompt.utils import get_prompt

    return get_prompt(PROMPT_CONFIG)
