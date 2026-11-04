"""τ³-bench family (Granite: τ³-bench, Airline, Retail, Telecom, Banking Knowledge).

Harness: sierra-research/tau2-bench v1.0.1 (the τ³ release), pinned by commit in
the ``tau`` extra. Task data is not part of the wheel; it is read from the same
commit (``data/tau2``), baked into the image by docker/extras/tau.sh or fetched
and verified on first use.

The agent is the served model, driven by the harness's own ``llm_agent`` with
tools called through the OpenAI API (vLLM's tool parser). The user simulator is
the harness's ``user_simulator`` on an OpenAI-compatible endpoint
(``user_base_url`` / ``user_model`` / ``user_api_key_env``; ``user_model=self``
uses the served model). Retail's NL-assertion checks call an LLM judge, which
the harness hard-codes to gpt-4.1; here it is ``judge_*`` (same pattern).
Both default to aws/claude-sonnet-5 on the configured gateway, not the harness's
gpt-4.1 default nor the actual official judge/user-sim (Azure gpt-4o-ncf,
confirmed from the gateway's cost logs), so scores are not
directly comparable with published τ³ numbers.

Metric, per domain: the harness's pass^1, i.e. the mean over tasks of the
fraction of trials with reward 1 (``tau2.metrics.agent_metrics``). The
``tau3-bench`` aggregate is the unweighted mean of pass^1 over the three core
domains (airline, retail, telecom), as the τ³ leaderboard computes "Overall";
banking is scored separately (web/leaderboard/src/components/Leaderboard.jsx).

Trials: 1 per task by default (pass^1 = pass@1 on one sample); the published
protocol runs 4 (``--repeats 4``). With k trials the value stays pass^1 (the
mean success over the k, i.e. pass@1[avg-of-k]); each domain adds the
harness's pass^k (all of k trials succeed) and pass@k (at least one does,
``pass_at_k``: 1 - C(n-c, k)/C(n, k)), and ``details.pass_at_k`` carries the
same mean over domains of pass^1, pass@k and pass^k (``pass_hat_k``).

Everything is written under ``<output_dir>/<domain>/trial-<k>/<task>.json``;
finished simulations are skipped on restart.

Phases: ``--phase generate`` runs the conversations (agent and user simulator)
and saves each ungraded (status ``generated``); ``--phase score`` grades the
saved ones with the harness's ``evaluate_simulation``, as the live run does
(DB, env assertions, actions, communicate; NL assertions via the judge), and
never simulates: a missing simulation is a failed one (``error:not_generated``).

A simulation that fails (an exception, the harness's infrastructure_error) is
not a model failure: as the harness does, it is left out of pass^k and the
average reward, and it is not saved, so a restart retries it. Each domain
reports ``simulations_failed`` / ``simulations_total``; above ``--option
max_failed_frac`` (default 0.05; 0 = any failure) in any domain the run fails,
below it the result is flagged ``incomplete``. A 402 (paid budget exhausted)
fails the run.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar

from granite_evals import data, meter
from granite_evals.registry import Benchmark, failure_policy, pass_at_k_record, register

log = logging.getLogger(__name__)

TAU2_REPO = "https://github.com/sierra-research/tau2-bench"
TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"  # tag v1.0.1
# Where docker/extras/tau.sh bakes data/ of TAU2_COMMIT: <root>/<commit>/data.
BAKED_DATA_ROOT = Path("/opt/tau2-data")
DATA_PATHS = ("data/tau2/domains", "data/tau2/user_simulator")

CORE_DOMAINS = ("airline", "retail", "telecom")
# Published protocol (docs/leaderboard-submission.md, config.py).
DEFAULT_SEED = 300
DEFAULT_MAX_STEPS = 200
DEFAULT_MAX_ERRORS = 10
DEFAULT_TRIALS = 1
"""One trial per task (granite's pass@1 default); the published protocol's 4 is ``--repeats 4``."""
# User decision 2026-09-28: judges and user simulators on the configured LiteLLM gateway.
DEFAULT_GATEWAY = ""  # set the role's *_base_url option (OpenAI-compatible endpoint)
DEFAULT_SIM_MODEL = "aws/claude-sonnet-5"
SELF = "self"
# Domains with tasks graded on NL assertions (an LLM judge): retail, and one
# banking_knowledge task (task_102) at TAU2_COMMIT.
NL_DOMAINS = frozenset({"retail", "banking_knowledge"})
# Status of a saved simulation not graded yet (--phase generate).
GENERATED = "generated"


# -- data ------------------------------------------------------------------


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def fetch_data(commit: str, dest: Path, repo: str = TAU2_REPO) -> Path:
    """Sparse-fetch ``data/tau2`` of ``repo`` at ``commit`` into ``dest`` (the
    repo root); verifies HEAD is exactly ``commit``. Returns ``dest/data``."""
    if (dest / ".granite-commit").exists():
        if (dest / ".granite-commit").read_text().strip() != commit:
            raise SystemExit(f"{dest} holds another tau2-bench commit")
        return dest / "data"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".tau2-", dir=dest.parent))
    try:
        _git("init", "-q", cwd=tmp)
        _git("remote", "add", "origin", repo, cwd=tmp)
        _git("sparse-checkout", "set", *DATA_PATHS, cwd=tmp)
        _git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit, cwd=tmp)
        _git("checkout", "-q", "FETCH_HEAD", cwd=tmp)
        head = _git("rev-parse", "HEAD", cwd=tmp)
        if head != commit:
            raise SystemExit(f"tau2-bench: fetched {head}, expected {commit}")
        shutil.rmtree(tmp / ".git")
        (tmp / ".granite-commit").write_text(commit + "\n")
        tmp.rename(dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest / "data"


def resolve_data_dir(source: str, revision: str | None) -> Path:
    """TAU2_DATA_DIR for a dataset source: a local path (a tau2-bench checkout or
    its ``data`` dir), else the pinned commit, baked or fetched."""
    local = Path(source)
    if local.exists():
        for cand in (local, local / "data"):
            if (cand / "tau2" / "domains").is_dir():
                return cand
        raise SystemExit(f"{source}: no tau2/domains under it (want a tau2-bench checkout or its data/)")
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit(f"tau2-bench data must be pinned to a 40-char commit, got {revision!r}")
    baked = BAKED_DATA_ROOT / revision
    if (baked / ".granite-commit").exists():
        return baked / "data"
    cache = Path(os.environ.get("GRANITE_EVALS_TAU2_DATA_CACHE", Path.home() / ".cache" / "granite" / "tau2-data"))
    return fetch_data(revision, cache / revision, repo=source)


def _import_harness(data_dir: Path) -> None:
    """tau2 reads TAU2_DATA_DIR once, at import; set it first and check it took."""
    os.environ["TAU2_DATA_DIR"] = str(data_dir)
    import tau2.utils.utils as tu

    if Path(tu.DATA_DIR).resolve() != data_dir.resolve():
        raise RuntimeError(f"tau2 was imported with DATA_DIR={tu.DATA_DIR}, need {data_dir}")
    from loguru import logger

    logger.remove()  # the harness logs every turn at DEBUG
    logger.add(lambda m: log.warning("tau2: %s", m.rstrip()), level=os.environ.get("GRANITE_EVALS_TAU2_LOG_LEVEL", "ERROR"))


# -- endpoints and usage accounting ----------------------------------------


@dataclass
class Endpoint:
    """An LLM the harness calls through litellm: model name + call kwargs."""

    model: str
    kwargs: dict[str, Any]
    name: str  # as recorded in results
    is_self: bool
    upstream: str = ""  # the real endpoint behind a metering proxy

    def record(self) -> dict[str, Any]:
        base = self.upstream or self.kwargs.get("api_base")
        return {"model": self.name, "base_url": base, "is_self": self.is_self,
                "args": {k: v for k, v in self.kwargs.items() if k not in ("api_key", "api_base")}}


def empty_usage() -> dict[str, int]:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cache_creation_tokens": 0}


def add_usage(acc: dict[str, int], message: Any) -> None:
    """Add one harness message's token usage (prompt, completion, and cached
    prompt tokens when the provider returns them) to ``acc``."""
    acc["calls"] += 1
    usage = getattr(message, "usage", None) or {}
    acc["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
    acc["completion_tokens"] += int(usage.get("completion_tokens") or 0)
    raw = (getattr(message, "raw_data", None) or {}).get("usage") or {}
    details = raw.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or raw.get("cache_read_input_tokens") or 0
    acc["cached_tokens"] += int(cached)
    acc["cache_creation_tokens"] += int(raw.get("cache_creation_input_tokens") or 0)


def _zero_usage() -> dict[str, dict[str, int]]:
    return {"agent": empty_usage(), "user_sim": empty_usage(), "judge": empty_usage()}


def sum_usage(items: list[dict[str, int]]) -> dict[str, int]:
    out = empty_usage()
    for u in items:
        for k in out:
            out[k] += int(u.get(k, 0))
    return out


class _Ctx(threading.local):
    """Per-worker-thread state: each simulation runs synchronously in one thread."""

    usage: dict[str, dict[str, int]] | None = None
    judge: Endpoint | None = None


_ctx = _Ctx()
_hooks_lock = threading.Lock()
# Set once the metering proxy answers HTTP 402: no further paid calls this run.
budget_exhausted = threading.Event()


def _guarded(orig: Callable, *args, **kwargs):
    if budget_exhausted.is_set():
        raise BudgetExhausted("granite spend budget exhausted")
    try:
        return orig(*args, **kwargs)
    except Exception as e:
        if "budget_exhausted" in str(e) or getattr(e, "status_code", None) == 402:
            budget_exhausted.set()
            raise BudgetExhausted(str(e)) from e
        raise


class BudgetExhausted(RuntimeError):
    pass


def _counting(orig: Callable, role: str) -> Callable:
    def generate(*args, **kwargs):
        message = _guarded(orig, *args, **kwargs)
        if _ctx.usage is not None:
            add_usage(_ctx.usage[role], message)
        return message

    generate._granite = True  # type: ignore[attr-defined]
    return generate


def _judging(orig: Callable) -> Callable:
    """The NL-assertion evaluator calls ``generate(model=<gpt-4.1>, ...)`` and
    ``json.loads`` the reply; swap in the configured judge, and strip markdown
    fences around the JSON (with the harness's own extractor)."""
    from tau2.utils.llm_utils import extract_json_from_llm_response

    def generate(*args, model=None, messages=None, call_name=None, **kwargs):
        judge = _ctx.judge
        if judge is None:
            raise RuntimeError("this task needs an NL-assertion judge; set judge_model")
        message = _guarded(orig, model=judge.model, messages=messages, call_name=call_name, **judge.kwargs)
        if _ctx.usage is not None:
            add_usage(_ctx.usage["judge"], message)
        content = message.content or ""
        try:
            json.loads(content)
        except ValueError:
            message.content = extract_json_from_llm_response(content)
        return message

    generate._granite = True  # type: ignore[attr-defined]
    return generate


def install_hooks() -> None:
    import tau2.agent.llm_agent as agent_mod
    import tau2.evaluator.evaluator_nl_assertions as nl_mod
    import tau2.user.user_simulator as user_mod
    from tau2.utils import llm_utils

    with _hooks_lock:
        if getattr(user_mod.generate, "_granite", False):
            return
        agent_mod.generate = _counting(llm_utils.generate, "agent")
        user_mod.generate = _counting(llm_utils.generate, "user_sim")
        nl_mod.generate = _judging(llm_utils.generate)


# -- scoring ---------------------------------------------------------------


def is_success(reward: float) -> bool:
    return abs(reward - 1.0) <= 1e-6  # tau2.metrics.agent_metrics.is_successful


def pass_hat_k(num_trials: int, successes: int, k: int) -> float:
    """tau2.metrics.agent_metrics.pass_hat_k: C(s, k) / C(n, k)."""
    from math import comb

    return comb(successes, k) / comb(num_trials, k) if num_trials >= k else float("nan")


def pass_at_k(num_trials: int, successes: int, k: int) -> float:
    """pass@k, the chance that k of the n trials hold a success: 1 - C(n-c, k) / C(n, k)."""
    from math import comb

    return 1 - comb(num_trials - successes, k) / comb(num_trials, k) if num_trials >= k else float("nan")


def is_failed(record: dict) -> bool:
    """A simulation that never ran to a reward: an exception here, or the
    harness's INFRASTRUCTURE_ERROR (which tau2's run_with_retry makes of one,
    runner/progress.py:19-51), a 402 included. Never persisted, so retried."""
    return record["status"].startswith("error:")


def domain_summary(records: list[dict], trials: int) -> dict[str, Any]:
    """pass^k and average reward over the (task, trial) records of one domain,
    as tau2's get_metrics_df: failed simulations are left out
    (agent_metrics.py:138-145), and pass^k goes up to the fewest trials a task
    has left (:158-165); a task with none left drops out. pass@k
    (``pass_at_<k>``, not in the harness) over the same tasks and k."""
    scored = [r for r in records if not is_failed(r)]
    by_task: dict[str, list[dict]] = {}
    for r in scored:
        by_task.setdefault(r["task_id"], []).append(r)
    rewards = [r["reward"] for r in scored]
    out: dict[str, Any] = {
        "n_tasks": len(by_task),
        "n_simulations": len(scored),
        "simulations_failed": len(records) - len(scored),
        "simulations_total": len(records),
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
    }
    max_k = min((len(rs) for rs in by_task.values()), default=0)
    for k in range(1, max_k + 1):
        vals = [pass_hat_k(len(rs), sum(is_success(r["reward"]) for r in rs), k) for rs in by_task.values()]
        out[f"pass_hat_{k}"] = sum(vals) / len(vals)
        anyk = [pass_at_k(len(rs), sum(is_success(r["reward"]) for r in rs), k) for rs in by_task.values()]
        out[f"pass_at_{k}"] = sum(anyk) / len(anyk)
    out["per_trial_pass_1"] = [
        _mean([is_success(r["reward"]) for r in scored if r["trial"] == t]) for t in range(trials)
    ]
    out["statuses"] = _count(r["status"] for r in records)
    out["termination_reasons"] = _count(r.get("termination_reason") or "none" for r in records)
    return out


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in items:
        out[s] = out.get(s, 0) + 1
    return dict(sorted(out.items()))


def trial_seeds(seed: int, trials: int) -> list[int]:
    """The harness's per-trial seeds (tau2.runner.batch.run_tasks)."""
    rng = random.Random(seed)
    return [rng.randint(0, 1000000) for _ in range(trials)]


def task_filename(task_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)[:80]
    return f"{safe}-{hashlib.sha1(task_id.encode()).hexdigest()[:8]}.json"


# -- the benchmarks ----------------------------------------------------------


class Tau3(Benchmark):
    metric = "pass@1"
    default_repeats = DEFAULT_TRIALS
    extra = "tau"
    harness_packages = ("tau2", "litellm")
    dataset = TAU2_REPO
    dataset_revision = TAU2_COMMIT
    splittable = True
    domains: ClassVar[tuple[str, ...]]

    def opt(self, key: str, default: Any) -> Any:
        value = self.config.options.get(key)
        return default if value is None else type(default)(value)

    @property
    def gold(self) -> bool:
        """``--option agent=gold`` replays each task's reference actions (and
        says its communicate_info) instead of running the model: checks data,
        environments and the DB / env-assertion / action / communicate grading
        without a model or simulator (every task should score 1). NL assertions
        are LLM-judged and skipped in this mode."""
        return self.opt("agent", "model") == "gold"

    def needs_server(self) -> bool:
        return not self.gold

    def score_needs_server(self) -> bool:
        """``judge_model=self`` grades NL assertions with the served model."""
        return (not self.gold and self.opt("judge_model", DEFAULT_SIM_MODEL) == SELF
                and bool(NL_DOMAINS & set(self.domains)))

    # -- endpoints -----------------------------------------------------------

    def agent_endpoint(self, base_url: str, served: str) -> Endpoint:
        kwargs: dict[str, Any] = {"api_base": base_url, "api_key": "EMPTY"}
        # Unset sampling options fall back to the checkpoint's generation_config
        # (vLLM's default); the harness's own default would be temperature 0.
        for key, cast in (("temperature", float), ("top_p", float), ("max_tokens", int)):
            if key in self.config.options:
                kwargs[key] = cast(self.config.options[key])
        if "enable_thinking" in self.config.options:
            flag = str(self.config.options["enable_thinking"]).lower() in ("1", "true", "yes")
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": flag}}
        return Endpoint(f"hosted_vllm/{served}", kwargs, served, True)

    def llm_endpoint(self, role: str, base_url: str, served: str, *, temperature: float,
                     called: bool = True) -> Endpoint:
        """``<role>_model`` / ``<role>_base_url`` / ``<role>_api_key_env`` (role =
        user or judge); ``<role>_model=self`` is the served model at base_url.
        ``called=False``: only recorded (the score phase's user simulator), so
        no key and no meter."""
        model = self.opt(f"{role}_model", DEFAULT_SIM_MODEL)
        kwargs: dict[str, Any] = {"temperature": self.opt(f"{role}_temperature", temperature)}
        if effort := self.opt(f"{role}_reasoning_effort", ""):
            kwargs["reasoning_effort"] = effort
        if self.opt(f"{role}_max_tokens", 0):
            kwargs["max_tokens"] = self.opt(f"{role}_max_tokens", 0)
        if model == SELF:
            return Endpoint(f"hosted_vllm/{served}", {**kwargs, "api_base": base_url, "api_key": "EMPTY"}, served, True)
        key_env = self.opt(f"{role}_api_key_env", f"GRANITE_EVALS_{role.upper()}_API_KEY")
        key = os.environ.get(key_env, "")
        if not called:
            given = self.config.options.get(f"{role}_base_url")
            return Endpoint(f"openai/{model}", {**kwargs, "api_base": given or DEFAULT_GATEWAY, "api_key": key},
                            model, False)
        if not key:
            raise SystemExit(
                f"{self.id}: {role}_model={model} needs an API key in ${key_env} "
                f"(or {role}_model=self for a smoke run)"
            )
        # A given *_base_url option is already the CLI's metering proxy; the class
        # default is metered here. Either way every paid call is ledgered.
        given = self.config.options.get(f"{role}_base_url")
        if not (given or DEFAULT_GATEWAY):
            raise SystemExit(f"{self.id}: no {role} endpoint; pass --option {role}_base_url=<OpenAI-compatible URL> (or {role}_model=self)")
        url = given or meter.metered(DEFAULT_GATEWAY, role=role)
        # openai/ = litellm's generic OpenAI-compatible client; the name after it
        # is the gateway's model id.
        upstream = DEFAULT_GATEWAY if not given else "(metered " + role + "_base_url)"
        return Endpoint(f"openai/{model}", {**kwargs, "api_base": url, "api_key": key}, model, False, upstream)

    # -- entry point -----------------------------------------------------------

    def load_tasks(self) -> tuple[dict[str, list], str, str | None]:
        source, revision = self.dataset_source()
        data_dir = resolve_data_dir(source, revision)
        _import_harness(data_dir)
        from tau2.runner.helpers import get_tasks

        split = self.opt("task_split", "base")
        tasks = {}
        for domain in self.domains:
            rows = [{"order": i, "task": t} for i, t in enumerate(get_tasks(domain, task_split_name=split))]
            if pattern := self.opt("tasks", ""):
                rows = [r for r in rows if re.search(pattern, r["task"].id)]
            tasks[domain] = [r["task"] for r in data.take(rows, self.config.limit, key="order")]
        return tasks, source, revision

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        tasks, source, revision = self.load_tasks()
        install_hooks()
        budget_exhausted.clear()
        # The judge grades; the agent and user simulator only converse (in the
        # score phase they are recorded, never called).
        needs_judge = self.scoring and not self.gold and any(
            "NL_ASSERTION" in _basis(t) for ts in tasks.values() for t in ts
        )
        agent = user = judge = None
        if not self.gold:
            agent = self.agent_endpoint(base_url, served_model_name)
            user = self.llm_endpoint("user", base_url, served_model_name, temperature=0.0,
                                     called=self.generating)
            if needs_judge:
                judge = self.llm_endpoint("judge", base_url, served_model_name, temperature=0.0)
        seed = DEFAULT_SEED + self.config.seed
        seeds = trial_seeds(seed, self.repeats)
        jobs = [(d, k, t) for d, ts in tasks.items() for k in range(self.repeats) for t in ts]
        log.info(
            "%s: %s tasks x %d trials (%d simulations), user=%s judge=%s",
            self.id, {d: len(ts) for d, ts in tasks.items()}, self.repeats, len(jobs),
            user.name if user else "-", judge.name if judge else "-",
        )
        done = [0]
        lock = threading.Lock()

        def one(job):
            d, k, t = job
            r = self._simulation(d, t, k, seeds[k], agent, user, judge)
            with lock:
                done[0] += 1
                reward = "-" if r["reward"] is None else f"{r['reward']:.2f}"
                log.info("%s %s trial %d: %s reward=%s %s (%d/%d)",
                         self.id, d, k, r["task_id"], reward, r["status"], done[0], len(jobs))
            return r

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.workers) as pool:
            records = list(pool.map(one, jobs))
        if budget_exhausted.is_set():
            # The simulations cut short are not saved: a rerun with budget retries them.
            raise SystemExit(f"{self.id}: the paid API budget is exhausted (HTTP 402); "
                             f"{sum(map(is_failed, records))} simulations not run, nothing scored")
        if not self.scoring:
            return self._generated(records, tasks, source, revision, seed, seeds, agent, user)

        per_domain = {}
        max_frac = self.opt("max_failed_frac", 0.05)
        for d in tasks:
            summary = domain_summary([r for r in records if r["domain"] == d], self.repeats)
            per_domain[d] = summary
            if summary["simulations_failed"]:
                log.warning("%s %s: %d/%d simulations failed, left out of the score; rerun to retry them",
                            self.id, d, summary["simulations_failed"], summary["simulations_total"])
            # per domain: failures in one domain must not hide behind another's successes
            failure_policy(f"{self.id} {d}", "simulations", summary["simulations_failed"],
                           summary["simulations_total"], max_frac)  # fmt: skip
            if not summary["n_simulations"]:
                raise SystemExit(f"{self.id} {d}: no simulation scored")
            log.info("%s %s: pass^1=%.4f over %d tasks", self.id, d, summary["pass_hat_1"], summary["n_tasks"])
        policy = failure_policy(self.id, "simulations", sum(map(is_failed, records)), len(records), max_frac)
        value = _mean([per_domain[d]["pass_hat_1"] for d in tasks])
        k = self.repeats

        def over_domains(key: str) -> float | None:  # None if a domain lost a task's k-th trial
            vals = [per_domain[d].get(key) for d in tasks]
            return None if None in vals else _mean(vals)

        pak = pass_at_k_record(k, value, over_domains(f"pass_at_{k}"),
                               sum(per_domain[d]["n_tasks"] for d in tasks),
                               "mean over domains; a task counts if any of its k trials has reward 1")  # fmt: skip
        pak["pass_hat_k"] = over_domains(f"pass_hat_{k}")
        usage = {role: sum_usage([r["usage"][role] for r in records]) for role in ("agent", "user_sim", "judge")}
        n_sims = len(records)
        return {
            "value": value,
            "n": sum(per_domain[d]["n_tasks"] for d in tasks),  # tasks with a scored simulation
            **policy,
            "dataset": source,
            "dataset_revision": revision,
            "harness": {"repo": TAU2_REPO, "commit": TAU2_COMMIT, "version": "1.0.1"},
            "aggregate": "mean of per-domain pass^1" if len(tasks) > 1 else "pass^1",
            "pass_at_k": pak,
            "domains": per_domain,
            "trials": self.repeats,
            "seed": seed,
            "trial_seeds": seeds,
            "max_steps": self.opt("max_steps", DEFAULT_MAX_STEPS),
            "max_errors": self.opt("max_errors", DEFAULT_MAX_ERRORS),
            "task_split": self.opt("task_split", "base"),
            "retrieval_config": self.retrieval_config() if "banking_knowledge" in tasks else None,
            "mode": "gold" if self.gold else "model",
            "agent": agent.record() if agent else None,
            "user_simulator": user.record() if user else None,
            "user_is_self": bool(user and user.is_self),
            "judge": judge.record() if judge else None,
            "judge_is_self": bool(judge and judge.is_self),
            "user_sim_usage": usage["user_sim"],
            "judge_usage": usage["judge"],
            "agent_usage": usage["agent"],
            "user_sim_usage_per_simulation": {k: v / n_sims for k, v in usage["user_sim"].items()} if n_sims else {},
            "tasks": {d: [t.id for t in ts] for d, ts in tasks.items()},
        }

    def _generated(self, records, tasks, source, revision, seed, seeds, agent, user) -> dict[str, Any]:
        """The generate phase's outcome: simulations saved for --phase score. A
        failed one is not saved; the score phase counts it as failed."""
        saved = [r for r in records if not is_failed(r)]
        usage = {role: sum_usage([r["usage"][role] for r in records]) for role in ("agent", "user_sim")}
        return {
            "n": len({(r["domain"], r["task_id"]) for r in saved}),  # tasks with a saved simulation
            "simulations_generated": len(saved),
            "simulations_failed": len(records) - len(saved),
            "simulations_total": len(records),
            "dataset": source,
            "dataset_revision": revision,
            "harness": {"repo": TAU2_REPO, "commit": TAU2_COMMIT, "version": "1.0.1"},
            "trials": self.repeats,
            "seed": seed,
            "trial_seeds": seeds,
            "mode": "gold" if self.gold else "model",
            "agent": agent.record() if agent else None,
            "user_simulator": user.record() if user else None,
            "user_sim_usage": usage["user_sim"],
            "agent_usage": usage["agent"],
            "tasks": {d: [t.id for t in ts] for d, ts in tasks.items()},
        }

    def retrieval_config(self) -> str:
        return self.opt("retrieval_config", "bm25_grep")

    # -- one simulation --------------------------------------------------------

    def _simulation(self, domain, task, trial, seed, agent, user, judge) -> dict:
        path = self.config.output_dir / domain / f"trial-{trial}" / task_filename(task.id)
        saved = json.loads(path.read_text()) if path.exists() else None
        # Done: graded, or generated and this phase does not grade.
        if saved is not None and not (self.scoring and saved["status"] == GENERATED):
            return {k: v for k, v in saved.items() if k != "simulation"}
        base = {"domain": domain, "task_id": task.id, "trial": trial, "seed": seed}
        if saved is None and not self.generating:
            # The score phase never simulates: counted as a failed simulation.
            return {**base, "reward": 0.0, "status": "error:not_generated",
                    "termination_reason": None, "usage": _zero_usage()}
        path.parent.mkdir(parents=True, exist_ok=True)
        if budget_exhausted.is_set():
            return {**base, "reward": 0.0, "status": "error:budget_exhausted",
                    "termination_reason": None, "usage": saved["usage"] if saved else _zero_usage()}
        # Grading a saved simulation adds its judge calls to the generation's usage.
        _ctx.usage = saved["usage"] if saved else _zero_usage()
        _ctx.judge = judge
        started = time.time()
        try:
            if saved is not None:
                sim = self._load_simulation(saved["simulation"])
                sim.reward_info = self._grade(domain, task, sim)
            elif self.gold:
                sim = self._gold(domain, task)
                if self.scoring:
                    sim.reward_info = self._grade(domain, task, sim)
            else:
                sim = self._simulate(domain, task, seed, agent, user)
            usage = _ctx.usage
        except Exception as e:  # one broken simulation must not sink the run
            log.exception("%s: %s %s trial %d failed", self.id, domain, task.id, trial)
            # Not persisted: an infrastructure error is retried on resume (a
            # grading error keeps the saved simulation, to be graded again).
            return {**base, "reward": 0.0, "status": f"error:{type(e).__name__}",
                    "termination_reason": None, "usage": _ctx.usage}
        finally:
            _ctx.usage, _ctx.judge = None, None
        termination = getattr(sim.termination_reason, "value", str(sim.termination_reason))
        duration = time.time() - started + (saved.get("duration_s", 0.0) if saved else 0.0)
        rec = {
            **base,
            "reward": None,
            "status": GENERATED,
            "termination_reason": termination,
            "duration_s": round(duration, 1),
            "num_messages": len(sim.messages or []),
            "usage": usage,
            "reward_info": None,
        }
        if self.scoring:
            reward_info = sim.reward_info
            rec["reward"] = float(reward_info.reward) if reward_info else 0.0
            rec["status"] = "success" if is_success(rec["reward"]) else "fail"
            rec["reward_info"] = reward_info.model_dump(mode="json") if reward_info else None
        if termination == "infrastructure_error":
            rec["reward"] = 0.0
            rec["status"] = "error:budget_exhausted" if budget_exhausted.is_set() else "error:infrastructure"
            return rec  # not persisted: retried on resume
        path.write_text(json.dumps({**rec, "simulation": sim.model_dump(mode="json")}, indent=1))
        return rec

    def run_config(self, domain: str, agent: Endpoint | None, user: Endpoint | None):
        from tau2.data_model.simulation import TextRunConfig

        kw: dict[str, Any] = {}
        if domain == "banking_knowledge":
            kw["retrieval_config"] = self.retrieval_config()
        return TextRunConfig(
            domain=domain,
            agent="llm_agent",
            user="user_simulator",
            llm_agent=agent.model if agent else "none",
            llm_args_agent=dict(agent.kwargs) if agent else {},
            llm_user=user.model if user else "none",
            llm_args_user=dict(user.kwargs) if user else {},
            max_steps=self.opt("max_steps", DEFAULT_MAX_STEPS),
            max_errors=self.opt("max_errors", DEFAULT_MAX_ERRORS),
            seed=DEFAULT_SEED + self.config.seed,
            **kw,
        )

    def _simulate(self, domain, task, seed, agent, user):
        """The conversation, graded (--phase all, as the harness runs it) or
        not (--phase generate: run_single_task's layers minus the evaluation,
        which ``_grade`` does later exactly as run_simulation would)."""
        import uuid

        from tau2.evaluator.evaluator import EvaluationType
        from tau2.runner.batch import run_single_task
        from tau2.runner.build import build_orchestrator
        from tau2.runner.progress import run_with_retry

        config = self.run_config(domain, agent, user)

        def converse():
            orchestrator = build_orchestrator(config, task, seed=seed, simulation_id=str(uuid.uuid4()))
            sim = orchestrator.run()
            sim.policy = orchestrator.environment.get_policy()  # as run_simulation
            return sim

        def graded():
            return run_single_task(config, task, seed=seed, evaluation_type=EvaluationType.ALL)

        return run_with_retry(
            graded if self.scoring else converse,
            task=task,
            trial=0,
            seed=seed,
            max_retries=self.opt("max_retries", 3),
            retry_delay=self.opt("retry_delay", 5.0),
            console_display=False,
        )

    def _grade(self, domain, task, sim):
        """The harness's live grading (runner/simulation.py run_simulation) of a
        text simulation: same task, env kwargs, solo_mode and replay; the gold
        trajectory with every non-LLM check."""
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
        from tau2.runner.build import _build_env_kwargs

        env_kwargs = _build_env_kwargs(self.run_config(domain, None, None), task)
        kind = EvaluationType.ALL_IGNORE_BASIS if self.gold else EvaluationType.ALL
        return evaluate_simulation(sim, task, kind, solo_mode=False, domain=domain, env_kwargs=env_kwargs or None)

    @staticmethod
    def _load_simulation(dump: dict):
        from tau2.data_model.simulation import SimulationRun

        return SimulationRun.model_validate(dump)

    def _gold(self, domain, task):
        """A trajectory that performs the task's reference actions, then says
        its communicate_info (graded by ``_grade``)."""
        import uuid

        from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
        from tau2.data_model.simulation import SimulationRun, TerminationReason
        from tau2.registry import registry
        from tau2.runner.build import _build_env_kwargs
        from tau2.utils.utils import get_now

        env_kwargs = _build_env_kwargs(self.run_config(domain, None, None), task)
        env = registry.get_env_constructor(domain)(solo_mode=False, **env_kwargs)
        init = task.initial_state
        history = list(init.message_history or []) if init else []
        env.set_state(
            initialization_data=init.initialization_data if init else None,
            initialization_actions=init.initialization_actions if init else None,
            message_history=history,
        )
        messages = list(history)
        criteria = task.evaluation_criteria
        for i, action in enumerate((criteria.actions if criteria else None) or []):
            call = ToolCall(id=f"gold-{i}", name=action.name, arguments=action.arguments, requestor=action.requestor)
            if action.requestor == "user":
                messages.append(UserMessage(role="user", tool_calls=[call]))
            else:
                messages.append(AssistantMessage(role="assistant", tool_calls=[call]))
            messages.append(env.get_response(call))
        info = (criteria.communicate_info if criteria else None) or []
        messages.append(AssistantMessage(role="assistant", content=" ".join(info) or "Done."))
        now = get_now()
        return SimulationRun(
            id=str(uuid.uuid4()), task_id=task.id, timestamp=now, start_time=now, end_time=now,
            duration=0.0, termination_reason=TerminationReason.AGENT_STOP, messages=messages,
        )


def _basis(task) -> set[str]:
    crit = task.evaluation_criteria
    return {getattr(b, "value", str(b)) for b in (crit.reward_basis or [])} if crit else set()


@register
class Tau3Bench(Tau3):
    id = "tau3-bench"
    metric = "pass@1 (avg of 3)"
    domains = CORE_DOMAINS


@register
class Tau3Airline(Tau3):
    id = "tau3-airline"
    domains = ("airline",)


@register
class Tau3Retail(Tau3):
    id = "tau3-retail"
    domains = ("retail",)


@register
class Tau3Telecom(Tau3):
    id = "tau3-telecom"
    domains = ("telecom",)


@register
class Tau3BankingKnowledge(Tau3):
    id = "tau3-banking-knowledge"
    domains = ("banking_knowledge",)
