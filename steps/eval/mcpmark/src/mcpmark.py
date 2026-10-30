"""MCPMark (Granite: agentic tool use over MCP services).

Metric: pass@1 task success rate per service. One agent run per task
(``--repeats 1``; ``details.pass_at_k`` is the k = 1 record).

The harness is MCPMark itself, baked into the image at a pinned commit
(:data:`MCPMARK_COMMIT`, Apache-2.0; ``mcpmark/extras.sh`` verifies the
codeload tarball sha256 and the extracted tree digest) and run as a
subprocess (``python -m pipeline``) from a writable per-run copy of its
checkout -- its wheel omits ``pipeline.py``, and its state managers write
downloads and task backups under the project root. Each task: the service's
state manager sets up the initial state (filesystem: a copy-on-write backup
of the category's CDN-zip tree; postgres: a ``CREATE DATABASE ... WITH
TEMPLATE`` clone of the category DB), the MCPMark agent drives the served
model against the service's MCP server (an npx / pipx stdio subprocess), and
the task's ``verify.py`` scores the final state.

Model access: litellm's openai provider with ``OPENAI_BASE_URL`` pointed at
the suite's in-job vLLM and ``OPENAI_API_KEY=EMPTY``
(``--models openai/<served_model_name>``; unknown model names default to
exactly that path in MCPMark's model_config). Never ``anthropic/...`` names
(a separate native-protocol code path) and never a ';'-joined base URL.

Task state (filesystem zips, postgres backups) is NEVER vendored or baked:
the state managers download it from storage.mcpmark.ai at run time into the
writable checkout, and aggregation records a digest of what was downloaded
(:data:`tree_digest`) so CDN drift is detectable across runs. Point
``--dataset`` at an existing MCPMark checkout to run against one fetched by
other means.

Resume: the harness's results tree under ``<output_dir>/mcpmark/`` IS the
resume store -- a task with a success or a non-retryable failure is skipped
on re-run (upstream's own rule, mirrored in :meth:`Mcpmark._done` so a
finished task costs no pipeline startup), a retryable one (rate limit, state
duplication, ...) is re-run (the pipeline wipes and retries it). The same
tree survives job retries, so a re-dispatched target continues where it
stopped. A fresh re-run is a new RUN_NAME, never a wiped tree.

Failures vs errors: a task that ran and failed verification scores 0; a task
whose error matches the infra patterns, or that never produced a meta.json,
is excluded from the score, counted in ``incomplete``, and retried on resume
(the failure_policy contract). Timeouts are deliberately NOT retryable
upstream: a timed-out task is a scored 0.

Services: one benchmark id per service (the τ³-bench shape), because they
need different infrastructure and have different task counts. The first
scope is the three that need no container runtime and no external accounts:
``mcpmark-filesystem`` (node/npx; state zips auto-download),
``mcpmark-postgres`` (a per-job postgres 17 cluster the adapter starts with
initdb/pg_ctl on a node-locked port; the harness restores the five category
DBs into it itself) and ``mcpmark-playwright`` (in-image chromium; the four
standard tasks drive the live public web, so expect run-to-run variance).
The github/notion/webarena services are deferred follow-ups (container
runtimes, account secrets, unrebuildable images).

Options: ``task_suite=standard|easy``, ``timeout=3600``,
``compaction_token`` (upstream default disables compaction),
``reasoning_effort`` (left unset for vLLM by default), ``max_failed_frac=0.05``.
``workers`` is validated to 1: upstream's pipeline runs tasks serially per
service, and its results tree merges whatever is on disk, so same-service
concurrency is upstream-unsupported (cross-service parallelism = separate
targets).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from granite_evals.registry import Benchmark, failure_policy, pass_at_k, register
from granite_evals.sandbox.nodelock import NodeLock

log = logging.getLogger(__name__)

# -- upstream pins ----------------------------------------------------------------
#
# MCPMark (github.com/eval-sys/mcpmark, Apache-2.0) at :data:`MCPMARK_COMMIT`.
# mcpmark/extras.sh bakes the checkout at /opt/mcpmark/<commit>, verifying the
# codeload tarball's sha256 (:data:`TARBALL_SHA256`, provenance only -- GitHub
# regenerates tarball bytes) and the digest of the extracted tree
# (:data:`TREE_SHA256`, the enforced pin, same scheme as :func:`tree_digest`),
# then writes the .granite-commit marker this module checks at run time.

REPO = "eval-sys/mcpmark"
MCPMARK_COMMIT = "cd45b7f57923b9b3985467f5139927575f83141c"
TARBALL_SHA256 = "cd1e67b1684e9cfc2354c023f1b3f61fc3d690af8dd5b69cd5e5fe1e6cf753c5"
TREE_SHA256 = "f825fd368508329f7e51a1f512828e36ff996506fbe09207ab31449f0cde8c23"

MCPMARK_HOME = Path(f"/opt/mcpmark/{MCPMARK_COMMIT}")
MCPMARK_PY = MCPMARK_HOME / ".venv" / "bin" / "python"
MCP_CONSTRAINTS = Path("/opt/pipx/mcp_constraints.txt")

# Task counts at the pinned commit, from the commit's git tree
# (tasks/<service>/<suite>/<category>/<task>/); the run checks them.
N_TASKS = {
    ("filesystem", "standard"): 30,
    ("filesystem", "easy"): 10,
    ("postgres", "standard"): 21,
    ("postgres", "easy"): 10,
    ("playwright", "standard"): 4,
    ("playwright", "easy"): 0,
}

# Infra errors the harness itself treats as retryable (src/errors.py at the
# pinned commit; matched as a substring against the lower-cased error message).
# A non-retryable error is a scored 0, not an infra failure; timeouts are
# deliberately NOT in this list upstream.
RETRYABLE_PATTERNS = (
    "ratelimit",
    "agent execution failed",
    "unavailable",
    "internal server error",
    "network error",
    "quota",
    "account balance",
    "mcp network error",
    "state duplication error",
    "thought_signature",
    "overloaded.",
)

SUITES = ("standard", "easy")
EXP_NAME = "run"
"""Stable --exp-name: keys the resume store inside <output_dir>/mcpmark/."""
DEFAULT_TIMEOUT_S = 3600
POSTGRES_MCP = "postgres-mcp==0.3.0"
PG_BINDIR = Path("/usr/lib/postgresql/17/bin")
PG_PORTS = range(15432, 15532)
"""Node-locked port scan for the per-job postgres cluster (the tbench
fixed-port pattern: two jobs on one node never share a port)."""


def tree_digest(root: Path) -> str:
    """sha256 over every file under ``root``: sorted posix relpath, NUL, the
    file's sha256 digest. Used to fingerprint the run-time-downloaded state
    payloads (the same scheme as extras.sh's build-time tree check)."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(b"\0")
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def model_slug(served_model_name: str) -> str:
    """The results-dir segment for a served model: upstream's evaluator.py at
    the pin builds it from the --models string ("openai/<served>") with every
    "." replaced by "-", so the slash stays a directory separator."""
    return f"openai/{served_model_name}".replace(".", "-")


def is_retryable(error: str | None) -> bool:
    """Upstream's errors.py at the pin: a substring match of
    :data:`RETRYABLE_PATTERNS` in the lower-cased error message."""
    if not error:
        return False
    lowered = error.lower()
    return any(p in lowered for p in RETRYABLE_PATTERNS)


def retry_tag(error: str | None) -> str:
    """The first :data:`RETRYABLE_PATTERNS` entry an error matches ("" if none)."""
    if not error:
        return ""
    lowered = error.lower()
    for pattern in RETRYABLE_PATTERNS:
        if pattern in lowered:
            return pattern
    return ""


class Mcpmark(Benchmark):
    """Shared implementation; subclasses pin the service."""

    service: ClassVar[str]
    splittable = False
    dataset = REPO
    dataset_revision = MCPMARK_COMMIT

    # -- options -----------------------------------------------------------

    def opt(self, key: str, default: Any) -> Any:
        """A ``--option key=value`` (always a string on the CLI), cast like ``default``."""
        value = self.config.options.get(key)
        return default if value is None else type(default)(value)

    @property
    def task_suite(self) -> str:
        suite = self.opt("task_suite", "standard")
        if suite not in SUITES:
            raise SystemExit(f"{self.id}: task_suite must be one of {', '.join(SUITES)}, not {suite!r}")
        return suite

    @property
    def results_root(self) -> Path:
        return self.config.output_dir / "mcpmark"

    def _service_dir(self) -> str:
        """evaluator.py at the pin: "<service>", suffixed "-<suite>" unless standard."""
        return self.service + ("" if self.task_suite == "standard" else f"-{self.task_suite}")

    def _run_dir(self, served_model_name: str) -> Path:
        """The harness's own results tree for this model and service. The
        model slug is upstream's: the whole --models string ("openai/<served>")
        with every "." replaced by "-", so its slash is a directory separator
        and the task dirs sit under ``.../<exp>/openai/<slug>__<service>/run-1``."""
        return (
            self.results_root / EXP_NAME / f"{model_slug(served_model_name)}__{self._service_dir()}" / "run-1"
        )

    # -- harness checkout: a writable per-run copy --------------------------

    def _checkout(self) -> Path:
        """The pinned checkout to run from: a ``--dataset`` dir as-is (trusted,
        like every benchmark's local-dataset override), else a writable copy of
        the baked one under the output dir (persistent across job retries, so
        CDN downloads and the resume store survive). The copy excludes the
        baked ``.venv`` (~1 GB the harness never writes to): the pipeline runs
        with cwd and PYTHONPATH at the copy but the interpreter and installed
        packages stay :data:`MCPMARK_PY` / the baked venv."""
        if self.config.dataset:
            root = Path(self.config.dataset)
            if not (root / "pipeline.py").is_file():
                raise SystemExit(f"{self.id}: --dataset {root}: not an MCPMark checkout (no pipeline.py)")
            return root
        dest = self.config.output_dir / "mcpmark-checkout"
        if not (dest / "pipeline.py").is_file():
            shutil.rmtree(dest, ignore_errors=True)
            partial = dest.with_name(dest.name + ".partial")
            shutil.rmtree(partial, ignore_errors=True)
            log.info("%s: copying the pinned MCPMark checkout to %s", self.id, dest)
            shutil.copytree(MCPMARK_HOME, partial, ignore=shutil.ignore_patterns(".venv"))
            partial.rename(dest)
        marker = dest / ".granite-commit"
        if not marker.is_file() or marker.read_text().strip() != MCPMARK_COMMIT:
            raise SystemExit(f"{self.id}: {dest} is not MCPMark {MCPMARK_COMMIT} (bad or missing .granite-commit)")
        return dest

    def _task_ids(self, checkout: Path) -> list[str]:
        """Sorted "category/task" ids for the service and suite, from the
        checkout's tasks/ tree (the same source of truth the pipeline reads)."""
        root = checkout / "tasks" / self.service / self.task_suite
        if not root.is_dir():
            return []
        return sorted(
            f"{category.name}/{task.name}"
            for category in root.iterdir()
            if category.is_dir()
            for task in category.iterdir()
            if task.is_dir()
        )

    # -- one pipeline invocation --------------------------------------------

    def _pipeline_env(self, checkout: Path, base_url: str, pg_env: dict[str, str] | None = None) -> dict[str, str]:
        """The pipeline subprocess env: litellm's openai provider resolution
        (OPENAI_BASE_URL at the in-job vLLM, the dummy key it requires), the
        writable checkout on PYTHONPATH, the baked venv first on PATH (it
        ships pipx, the postgres MCP server's launcher), and the mcp<2
        constraint guarding a cold re-resolve of postgres-mcp (an
        unconstrained one installs mcp 2.0.0, which dropped
        mcp.server.fastmcp and fails every postgres task with "Connection
        closed" -- reading as a model failure)."""
        env = {
            **os.environ,
            "OPENAI_BASE_URL": base_url,
            "OPENAI_API_KEY": "EMPTY",
            "PYTHONPATH": str(checkout),
            "PATH": f"{MCPMARK_HOME}/.venv/bin{os.pathsep}{os.environ.get('PATH', '')}",
            "UV_CONSTRAINT": str(MCP_CONSTRAINTS),
            "PIP_CONSTRAINT": str(MCP_CONSTRAINTS),
        }
        if self.service == "filesystem":
            # Where the state manager builds the per-category test trees
            # (default: under the project root; keep them with the run).
            env["FILESYSTEM_TEST_ROOT"] = str(self.config.output_dir / "test_environments")
        if pg_env:
            env.update(pg_env)
        return env

    def _run_pipeline(self, checkout: Path, env: dict[str, str], served_model_name: str, tasks: str) -> None:
        """One upstream pipeline invocation. ``tasks`` is "all", a category, or
        "category/task" (upstream's only granularities). check=False: a
        non-zero exit (some task failed) must not stop the loop; the meta.json
        tree is the source of truth and aggregation decides."""
        argv = [
            str(MCPMARK_PY),
            "-m",
            "pipeline",
            "--mcp",
            self.service,
            "--models",
            f"openai/{served_model_name}",
            "--exp-name",
            EXP_NAME,
            "--task-suite",
            self.task_suite,
            "--tasks",
            tasks,
            "--k",
            "1",
            "--timeout",
            str(self.opt("timeout", DEFAULT_TIMEOUT_S)),
            "--output-dir",
            str(self.results_root),
        ]
        if token := self.opt("compaction_token", 0):
            argv += ["--compaction-token", str(token)]
        if effort := self.opt("reasoning_effort", ""):
            argv += ["--reasoning-effort", effort]
        log.info("%s: %s", self.id, " ".join(argv))
        subprocess.run(argv, cwd=checkout, env=env, check=False)

    # -- resume: upstream's own rule, mirrored --------------------------------

    def _done(self, run_dir: Path, task_id: str) -> bool:
        """The harness's resume rule (evaluator.py at the pin): a task whose
        meta.json shows a success or a non-retryable error is finished; a
        retryable one is re-run (the pipeline wipes and retries it). Mirrored
        here so a finished task costs no pipeline startup on a resumed run."""
        meta = run_dir / task_id.replace("/", "__") / "meta.json"
        if not meta.is_file():
            return False
        try:
            record = json.loads(meta.read_text())
            result = record["execution_result"]
        except (OSError, ValueError, KeyError, TypeError):
            return False  # a partial write: re-run it
        return not is_retryable(result.get("error_message"))

    # -- postgres: a per-job cluster on a node-locked port --------------------

    @contextlib.contextmanager
    def _postgres_server(self):
        """initdb + pg_ctl over node-local scratch on the first free
        node-locked port. The harness's state manager restores the five
        category databases into it on first use and clones/drops a per-task
        DB, so one job-local server is fully isolated per run."""
        for port in PG_PORTS:
            lock = NodeLock(f"mcpmark-postgres-{port}")
            if not lock.try_acquire():
                continue
            log.info("%s: postgres cluster on port %d", self.id, port)
            data = Path(tempfile.mkdtemp(prefix="mcpmark-pg-"))
            try:
                self._run_checked(
                    [str(PG_BINDIR / "initdb"), "-D", str(data), "-U", "postgres", "--auth=trust"],
                    f"initdb (port {port})",
                )
                self._run_checked(
                    [
                        str(PG_BINDIR / "pg_ctl"),
                        "-D",
                        str(data),
                        "-w",
                        "-o",
                        f"-p {port} -k {data}",
                        "-l",
                        str(data / "postgres.log"),
                        "start",
                    ],
                    f"pg_ctl start (port {port})",
                )
                try:
                    yield {
                        "POSTGRES_HOST": "localhost",
                        "POSTGRES_PORT": str(port),
                        "POSTGRES_USERNAME": "postgres",
                        "POSTGRES_PASSWORD": "mcpmark",
                    }
                finally:
                    self._run_checked(
                        [str(PG_BINDIR / "pg_ctl"), "-D", str(data), "-m", "immediate", "-w", "stop"],
                        f"pg_ctl stop (port {port})",
                        check=False,
                    )
            finally:
                shutil.rmtree(data, ignore_errors=True)
                lock.release()
            return
        raise SystemExit(f"{self.id}: no free postgres port in [{PG_PORTS[0]}, {PG_PORTS[-1]}] on this node")

    @staticmethod
    def _run_checked(argv: list[str], what: str, *, check: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        if check and result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()[-500:]
            raise RuntimeError(f"{what} failed (exit {result.returncode}): {argv}\n{tail}")
        return result

    def _postgres_preflight(self, env: dict[str, str]) -> None:
        """Prove the pipx'd postgres-mcp starts before the node is spent: the
        baked venv (mcp<2, extras.sh) must answer --help. A broken resolve
        fails every task in ~3 s with "Connection closed", which reads as a
        model failure -- exactly the misreport this preflight prevents."""
        result = subprocess.run(
            ["pipx", "run", POSTGRES_MCP, "--help"], env=env, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()[-500:]
            raise SystemExit(
                f"{self.id}: {POSTGRES_MCP} preflight failed (exit {result.returncode}): {tail}; "
                f"the pipx venv is baked with mcp<2 ({MCP_CONSTRAINTS}) -- an unconstrained "
                "resolve installs mcp 2.0.0, which dropped mcp.server.fastmcp"
            )

    # -- entry point ----------------------------------------------------------

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        if self.repeats != 1:
            raise SystemExit(f"{self.id}: one agent run per task (repeats 1; upstream's --k is what varies samples)")
        if self.config.workers != 1:
            raise SystemExit(f"{self.id}: the pipeline runs tasks serially per service (workers 1; cross-service parallelism = separate targets)")
        checkout = self._checkout()
        ids = self._task_ids(checkout)
        pinned = not self.config.dataset
        if not ids:
            raise SystemExit(f"{self.id}: no {self.service} tasks in the {self.task_suite} suite at {MCPMARK_COMMIT}")
        if pinned and len(ids) != N_TASKS[(self.service, self.task_suite)]:
            expected = N_TASKS[(self.service, self.task_suite)]
            raise SystemExit(f"{self.id}: {len(ids)} {self.service}/{self.task_suite} tasks at {MCPMARK_COMMIT}, not {expected}")
        if self.config.limit:
            ids = ids[: self.config.limit]  # first N, sorted: comparable smokes
        run_dir = self._run_dir(served_model_name)
        source, revision = self.dataset_source()
        log.info("%s: %d %s/%s tasks (service %s)", self.id, len(ids), self.service, self.task_suite, self.service)
        cm = self._postgres_server() if self.service == "postgres" else contextlib.nullcontext(None)
        with cm as pg_env:
            env = self._pipeline_env(checkout, base_url, pg_env)
            if self.service == "postgres":
                self._postgres_preflight(env)
            for task_id in ids:
                if self._done(run_dir, task_id):
                    log.info("%s: %s already run; skipping (harness resume)", self.id, task_id)
                    continue
                self._run_pipeline(checkout, env, served_model_name, task_id)
        return self._aggregate(checkout, run_dir, ids, source, revision)

    # -- results.json from the harness's meta.json tree ------------------------

    def _state_digests(self, checkout: Path) -> dict[str, str]:
        """Digests of the run-time-downloaded state payloads (license
        unstated, content not commit-pinned), so storage.mcpmark.ai drift is
        detectable across runs: the filesystem test trees as restored after
        the run, and the postgres category backups."""
        candidates = {
            "filesystem_state": self.config.output_dir / "test_environments",
            "postgres_state": checkout / "postgres_state",
        }
        return {
            name: tree_digest(root)
            for name, root in candidates.items()
            if root.is_dir() and any(root.iterdir())
        }

    def _aggregate(
        self, checkout: Path, run_dir: Path, ids: list[str], source: str, revision: str | None
    ) -> dict[str, Any]:
        rows = []
        for task_id in ids:
            meta = run_dir / task_id.replace("/", "__") / "meta.json"
            row: dict[str, Any] = {"task": task_id, "success": False, "error": None, "missing": True}
            if meta.is_file():
                try:
                    result = json.loads(meta.read_text())["execution_result"]
                    row = {
                        "task": task_id,
                        "success": bool(result.get("success")),
                        "error": result.get("error_message"),
                        "missing": False,
                    }
                except (OSError, ValueError, KeyError, TypeError):
                    pass  # a partial write: infra failure, retried on resume
            rows.append(row)
        # A retryable error or a missing meta.json is infrastructure, not a
        # model loss: left out of the score, counted in task_failed.
        scored = [r for r in rows if not r["missing"] and not is_retryable(r["error"])]
        infra = [r for r in rows if r["missing"] or is_retryable(r["error"])]

        def status(row: dict[str, Any]) -> str:
            if row["missing"]:
                return "missing"
            if row["success"]:
                return "pass"
            if is_retryable(row["error"]):
                return f"retryable:{retry_tag(row['error'])}"
            return "fail"

        return {
            "value": (sum(r["success"] for r in scored) / len(scored)) if scored else 0.0,
            "n": len(scored),
            "n_total": len(ids),
            "dataset": source,
            "dataset_revision": revision,
            "mcpmark_commit": MCPMARK_COMMIT,
            "service": self.service,
            "task_suite": self.task_suite,
            "timeout": self.opt("timeout", DEFAULT_TIMEOUT_S),
            "state_digest": self._state_digests(checkout),
            "pass_at_k": pass_at_k(
                ([r["success"]] for r in scored),
                1,
                "one agent run per task (upstream --k 1)",
            ),
            **failure_policy(self.id, "task", len(infra), len(rows), self.opt("max_failed_frac", 0.05)),
            "per_task": {r["task"]: float(r["success"]) for r in scored},
            "per_task_status": {r["task"]: status(r) for r in rows},
            "tasks": list(ids),
        }


@register
class McpmarkFilesystem(Mcpmark):
    id = "mcpmark-filesystem"
    metric = "pass@1"
    extra = "mcpmark"
    harness_packages = ()
    service = "filesystem"


@register
class McpmarkPostgres(Mcpmark):
    id = "mcpmark-postgres"
    metric = "pass@1"
    extra = "mcpmark"
    harness_packages = ()
    service = "postgres"


@register
class McpmarkPlaywright(Mcpmark):
    id = "mcpmark-playwright"
    metric = "pass@1"
    extra = "mcpmark"
    harness_packages = ()
    service = "playwright"
