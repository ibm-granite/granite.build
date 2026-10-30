"""NL2RepoBench (Granite: agentic-coding, metric "test pass rate").

Upstream (github.com/multimodal-art-projection/NL2RepoBench, :data:`COMMIT`)
ships 104 tasks under ``test_files/<proName>/``: each is the ``start.md``
spec of one real PyPI project (aiofiles, six, scikit-learn, ...), its test
case count, the commands that install and test an implementation, and the
paths of the project's own test files. An agent implements the whole project
from ``start.md``; the grade runs the task's commands over the agent's
workspace inside the project's own test image and counts the pytest cases
that pass. The headline is the mean over tasks of upstream's
``success_rate`` = ``min(passed / test_case_count, 1)``.

LICENSE BLOCKER: upstream has NO license. Nothing from the repo is vendored,
baked into an image, mirrored or pushed. Only the task data (``test_files/``)
is used, and it is fetched at run start -- before any task is processed --
from the pinned commit as a tarball, verified (a sha256 over the extracted
tree plus the 104-task count) and cached under
``~/.cache/granite/nl2repobench/<commit>``. Point ``GRANITE_EVALS_NL2REPO_DIR``
at an existing ``test_files`` tree, or pass ``--dataset <dir>``, to use one
fetched by other means. The harness logic is a port in this file; the
OpenHands app image and the per-task GHCR test images are pulled from their
upstream registries at run time via enroot and never re-distributed. See
USAGE.md ("Why this shape is safe"); the runtime fetch of unlicensed data
is pending OSPO sign-off.

What runs, all pinned:

- Data: ``test_files/`` of :data:`REPO` at :data:`COMMIT`, verified against
  :data:`TASKS_SHA256` (:data:`N_TASKS` tasks). The tarball's own sha256 at
  pin time is recorded in :data:`TARBALL_SHA256`; the enforced pin is the
  digest of the extracted tree, which survives GitHub regenerating the
  tarball bytes.
- Agent: the OpenHands 0.56 app image (:data:`OPENHANDS_IMAGE`) under enroot,
  with OpenHands's *local* runtime -- the 0.56 name for what later versions
  call the ``process`` runtime, selected by ``RUNTIME=local`` (env or
  ``[core] runtime``). ``LocalRuntime`` runs the action execution server as a
  local subprocess and adds no isolation of its own; the enroot container is
  the isolation boundary. The workspace is a directory inside the sandbox
  (``[sandbox] volumes`` -> ``workspace_base``), seeded with ``start.md``;
  the task prompt is upstream's exact ``-t`` string (:data:`TASK_PROMPT`).
  Upstream instead runs one OpenHands *container* per task with the Docker
  runtime and its own ``runtime:0.56-nikolaik`` sandbox image; the generated
  ``config.toml`` here reproduces the effective settings of upstream's
  ``template/config.template.toml`` at the pin (agent tools, noop condenser,
  500 max iterations, trajectory saving), written by us rather than copied.
- Grade: per task, a fresh sandbox from the project's test image
  ``ghcr.io/multimodal-art-projection/nl2repobench/<proName>:1.0``
  (:data:`IMAGE_PREFIX`, :data:`IMAGE_TAG`), the agent's workspace stripped
  of packaging files (:data:`PACKAGE_FILES`) and of the gold test paths
  (``test_files.json``), then extracted over the image's ``/workspace``
  (upstream: a Docker image build whose ``COPY workspace /workspace`` leaves
  the image's own copies of the gold tests in place) with
  ``PYTHONPATH=/workspace``, and the task's ``test_commands.json`` run with
  workdir ``/workspace``. pytest summary lines are parsed exactly as
  upstream's ``analyze_pytest_results`` does (sums of ``N passed`` /
  ``N failed`` / ``N error`` over pytest-containing commands).

Deviations from upstream (also in ``details.deviations``):

- enroot + the local runtime instead of Docker: no Docker socket, and
  ``SANDBOX_RUNTIME_CONTAINER_IMAGE`` (upstream sets it for the Docker
  runtime) is inert, so it is not set. The agent runs in the app image
  rather than upstream's nikolaik runtime image; ``SKIP_DEPENDENCY_CHECK=1``
  skips LocalRuntime's jupyter/tmux/browser dependency probe.
- The task commands run through ``bash -c``, fixing upstream's
  ``shlex.split``-then-exec bug (upstream PR #14): the pinned task data
  contains commands with ``&&``, ``&`` and ``VAR=... cmd`` that upstream's
  runner cannot execute as written.
- Command output is captured even when a command exits non-zero (upstream's
  docker-exec wrapper returned the exception text), so the pytest summary is
  parsed from the real output.
- Wall-clock bounds upstream has none: ``eval_timeout`` (default 1800 s) per
  grade command and ``agent_timeout`` (default 14400 s, 0 = none) per agent
  run. A timed-out grade is persisted (status ``eval_timeout``) with the
  score parsed from the partial output; a timed-out agent still has its
  workspace so far graded.
- A workspace larger than ``ws_max_mb`` (default 512) grades as 0
  (``workspace_too_large``).
- ``agent=gold`` (no model): grades each task image with an untouched
  workspace (``start.md`` only) -- a model-free check of the fetch, images,
  sandbox and grading. Upstream has no equivalent; how much passes depends
  on what the images themselves ship.
- An infrastructure failure (fetch, image pull, sandbox) is not persisted
  and is retried on resume (registry.failure_policy), where upstream records
  the task as failed.

Layout: ``<output_dir>/repeat-<k>/<task>/`` holds ``workspace.tar.gz`` (the
agent's workspace, the generation a score phase grades), ``openhands.log``,
``trajectory.tar.gz``, ``meta.json`` and, once graded, ``report.json`` and
``test_output.txt``; finished tasks are skipped on restart. Splittable:
``--phase generate`` needs the model and one sandbox per task, ``--phase
score`` only the task images.

Options: ``agent=model|gold``, ``tasks`` (regex), ``openhands_image``,
``image_prefix``, ``image_tag``, ``max_iterations=500``,
``enable_browsing=true``, ``temperature`` / ``top_p`` / ``max_tokens``,
``agent_timeout=14400``, ``eval_timeout=1800``, ``ws_max_mb=512``,
``sandbox`` (enroot), ``max_failed_frac=0.05``.
"""

from __future__ import annotations

import base64
import concurrent.futures
import functools
import hashlib
import io
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

from granite_evals.registry import Benchmark, failure_policy, pass_at_k, register
from granite_evals.sandbox import make_sandbox

log = logging.getLogger(__name__)

# -- upstream pins -------------------------------------------------------------
#
# Upstream has no license and no tags; the commit is the only pin. The
# enforced integrity check is TASKS_SHA256 over the extracted test_files/
# tree (see tree_digest); TARBALL_SHA256 is recorded for provenance only, so
# GitHub regenerating tarball bytes cannot break the fetch.

REPO = "multimodal-art-projection/NL2RepoBench"
COMMIT = "781a1da1ee41fb8edb0bed22f586d69111610edf"
TARBALL_URL = f"https://github.com/{REPO}/archive/{COMMIT}.tar.gz"
TARBALL_SHA256 = "00bef9ac9838e1ca955d2441b436f32abf17ac963bd06076c5b03f62802cbd9e"
TASKS_SHA256 = "1de36960a0fc944c560dcb012adc4a874c22327d7bfe1de62e3c76afdea0197d"
N_TASKS = 104
TASK_FILES = (
    "start.md",
    "test_case_count.txt",
    "test_commands.json",
    "test_files.json",
)

# OpenHands 0.56 (app image) under enroot; the per-task grade images.
OPENHANDS_IMAGE = "docker.all-hands.dev/all-hands-ai/openhands:0.56"
IMAGE_PREFIX = "ghcr.io/multimodal-art-projection/nl2repobench"
IMAGE_TAG = "1.0"

CACHE_ENV = "GRANITE_EVALS_NL2REPO_DIR"
"""An existing test_files tree to use instead of the fetched pin (trusted)."""

# Upstream's exact -t string (openhands_app.py at COMMIT): part of the ported
# harness logic, kept verbatim so scores stay comparable with upstream runs.
TASK_PROMPT = (
    "According to the start.md in the workspace, implement the entire project as per the "
    "requirements specified in the document, ensuring that the final product can be directly "
    "run in the current directory. The running requirements should comply with the "
    "<API Usage Guide> section of the document. Please complete this task step by step."
)

# post_processor.py's remove_package_files: deleted recursively anywhere in
# the workspace before grading, so what the agent ships cannot reshape how
# the task's own install/test commands behave.
PACKAGE_FILES = (
    "setup.py",
    "pyproject.toml",
    "setup.cfg",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "tox.ini",
    "pytest.ini",
    "poetry.lock",
    "Pipfile",
    "Pipfile.lock",
    "environment.yml",
    "conda-env.yaml",
    "manifest.in",
    "MANIFEST.in",
)

RUN_DIR = "/tmp/nl2repo-run"
"""Where one generation sandbox keeps its config, workspace and trajectories."""
WS_B64 = "/tmp/nl2repo-workspace.tar.gz.b64"
STAGE_TIMEOUT_S = 300
AGENT_TIMEOUT_S = 14_400
EVAL_TIMEOUT_S = 1800
WS_MAX_MB = 512

DEVIATIONS = (
    "enroot + OpenHands 0.56 local (process) runtime instead of upstream's Docker runtime; "
    "the enroot container is the isolation boundary",
    "task commands run through bash -c (upstream shlex.split-then-exec breaks &&, & and "
    "VAR=... commands; upstream PR #14)",
    "command output is captured on non-zero exit too, so pytest summaries are parsed from "
    "the real output",
    "wall-clock bounds upstream has none: eval_timeout per grade command, agent_timeout per "
    "agent run; a timed-out grade keeps the score parsed from the partial output",
    "agent=gold grades the task images with an untouched workspace (a model-free pipeline "
    "check upstream does not have)",
    "an infrastructure failure is retried on resume (failure_policy), not recorded as a "
    "failed task",
)


def tree_digest(root: Path) -> str:
    """sha256 over every file under ``root``: sorted posix relpath, NUL, the
    file's sha256 digest. The pinned shape of the task data, independent of
    how it was transported (tarball bytes, git clone, ...)."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(b"\0")
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


class Task:
    """One upstream task: ``test_files/<proName>/`` with its four files.

    Upstream's test_data_service.py picks the first ``.txt``, the first json
    with ``commands`` in its name, the first with ``files`` and the first
    ``.md``; at the pinned commit every task dir holds exactly one of each
    under these canonical names, which is what this port reads.
    """

    def __init__(self, root: Path):
        self.root = root
        self.name = root.name
        missing = [n for n in TASK_FILES if not (root / n).is_file()]
        if missing:
            raise RuntimeError(f"{root}: missing {', '.join(missing)}")

    @functools.cached_property
    def start_md(self) -> str:
        return (self.root / "start.md").read_text(encoding="utf-8")

    @functools.cached_property
    def test_case_count(self) -> int:
        return int((self.root / "test_case_count.txt").read_text().strip())

    @functools.cached_property
    def commands(self) -> list[str]:
        commands = json.loads((self.root / "test_commands.json").read_text())
        if not isinstance(commands, list) or not all(
            isinstance(c, str) for c in commands
        ):
            raise RuntimeError(
                f"{self.root}/test_commands.json: expected a list of strings"
            )
        return commands

    @functools.cached_property
    def test_files(self) -> list[str]:
        files = json.loads((self.root / "test_files.json").read_text())
        if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
            raise RuntimeError(
                f"{self.root}/test_files.json: expected a list of strings"
            )
        return files

    def image(self, prefix: str, tag: str) -> str:
        return f"{prefix}/{self.name}:{tag}"


class TaskSet:
    """``test_files/`` of :data:`REPO` at :data:`COMMIT`, fetched at run start.

    The pinned tarball is downloaded once into ``root/<commit>/`` and only
    ``test_files/`` is extracted from it (regular files only, paths checked
    against traversal). The extracted tree must digest to :data:`TASKS_SHA256`
    and hold :data:`N_TASKS` task dirs, else the run stops before any task is
    processed. A failed or tampered fetch never replaces a good cache.
    """

    def __init__(self, root: Path | None = None, *, fetch=None):
        self.root = root or Path.home() / ".cache" / "granite" / "nl2repobench"
        self._fetch = fetch or self._http_get

    def _http_get(self) -> bytes:
        import httpx

        for attempt in range(5):
            try:
                r = httpx.get(TARBALL_URL, timeout=120, follow_redirects=True)
                r.raise_for_status()
                return r.content
            except httpx.HTTPError:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def _extract(self, tarball: bytes, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as tar:
            for member in tar:
                parts = member.name.split("/")
                # <archive-root>/test_files/<task>/<file>: regular files only,
                # wherever the archive root sits (GitHub uses <repo>-<commit>/).
                if not member.isfile() or "test_files" not in parts:
                    continue
                rest = parts[parts.index("test_files") + 1 :]
                if len(rest) < 2 or any(p in ("", "..", ".") for p in rest):
                    continue
                out = dest.joinpath(*rest)
                out.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as f, open(out, "wb") as g:
                    shutil.copyfileobj(f, g)

    def _verified(self, test_files: Path) -> bool:
        if not test_files.is_dir():
            return False
        tasks = [p for p in test_files.iterdir() if p.is_dir()]
        if len(tasks) != N_TASKS:
            log.error("%s: %d task dirs, expected %d", test_files, len(tasks), N_TASKS)
            return False
        digest = tree_digest(test_files)
        if digest != TASKS_SHA256:
            log.error(
                "%s: tree sha256 %s, expected %s", test_files, digest, TASKS_SHA256
            )
            return False
        return True

    def ensure(self) -> Path:
        """The verified ``test_files`` tree, fetching it if not cached."""
        test_files = self.root / COMMIT / "test_files"
        if self._verified(test_files):
            return test_files
        log.info(
            "nl2repo: fetching %s@%s (test_files only; nothing is baked or vendored)",
            REPO,
            COMMIT,
        )
        partial = self.root / COMMIT / "test_files.partial"
        shutil.rmtree(partial, ignore_errors=True)
        self._extract(self._fetch(), partial)
        if not self._verified(partial):
            shutil.rmtree(partial, ignore_errors=True)
            raise RuntimeError(
                f"{REPO}@{COMMIT}: fetched test_files does not match the pinned tree "
                f"(sha256 {TASKS_SHA256}, {N_TASKS} tasks); refusing to run"
            )
        shutil.rmtree(test_files, ignore_errors=True)
        partial.rename(test_files)
        return test_files


def strip_workspace(ws: Path, test_files: list[str]) -> dict[str, list[str]]:
    """Upstream's workspace strip before grading: delete every packaging file
    (:data:`PACKAGE_FILES`, recursively anywhere) and every gold test path
    (file or directory) the task lists, so the agent's copies cannot shadow
    the copies the task image ships. Returns what was removed."""
    root = ws.resolve()
    packages, tests, missing = [], [], []
    for p in sorted(ws.rglob("*")):
        if p.is_file() and p.name in PACKAGE_FILES:
            p.unlink()
            packages.append(str(p.relative_to(ws)))
    for rel in test_files:
        target = (ws / rel).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"test file {rel!r} escapes the workspace")
        if target.is_dir():
            shutil.rmtree(target)
            tests.append(rel)
        elif target.exists():
            target.unlink()
            tests.append(rel)
        else:
            missing.append(rel)
    return {
        "package_files": packages,
        "test_files": tests,
        "test_files_missing": missing,
    }


def parse_pytest_results(
    command_results: list[dict[str, Any]], total: int
) -> dict[str, Any]:
    """Upstream's analyze_pytest_results, ported: over the outputs of commands
    containing ``pytest`` (case-insensitive), sum the first ``N passed`` /
    ``N failed`` / ``N error`` per line; the rate is ``passed / total`` capped
    at 1 (``total`` is the task's own test case count)."""
    out = {"passed": 0, "failed": 0, "errors": 0, "total": total, "success_rate": 0.0}
    for result in command_results:
        if "pytest" not in result["command"].lower():
            continue
        for line in result["output"].split("\n"):
            for key, pattern in (
                ("passed", r"(\d+) passed"),
                ("failed", r"(\d+) failed"),
                ("errors", r"(\d+) error"),
            ):
                if m := re.search(pattern, line):
                    out[key] += int(m.group(1))
    if out["total"] > 0:
        out["success_rate"] = min(out["passed"] / out["total"], 1)
    return out


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)  # a JSON string is a valid TOML basic string
    return str(value)


def render_config(
    *,
    workspace: str,
    llm_key: str,
    model: str,
    base_url: str,
    max_iterations: int,
    enable_browsing: bool,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """The OpenHands config.toml for one task.

    Written by us (nothing of upstream's is copied) to the *effective*
    settings of its ``template/config.template.toml`` at :data:`COMMIT`: the
    uncommented values there (agent tools, noop condenser, 500 max
    iterations, trajectories under ``./trajectories``), with the runtime
    switched to 0.56's ``local`` mode for enroot and the ``[llm.<key>]``
    block upstream renders in place of its ``{{MODULE_CONFIG}}`` placeholder.
    """
    llm = f"""
[llm.{llm_key}]
model = "openai/{model}"
api_key = "EMPTY"
base_url = {_toml(base_url)}
"""
    if temperature is not None:
        llm += f"temperature = {_toml(temperature)}\n"
    if top_p is not None:
        llm += f"top_p = {_toml(top_p)}\n"
    if max_tokens is not None:
        llm += f"max_output_tokens = {_toml(max_tokens)}\n"
    return f"""# Generated by granite-evals (nl2repo) for one NL2RepoBench task.
[core]
runtime = "local"
max_iterations = {_toml(max_iterations)}
save_trajectory_path = "./trajectories"
{llm}
[agent]
enable_browsing = {_toml(enable_browsing)}
enable_llm_editor = false
enable_editor = true
enable_jupyter = true
enable_cmd = true
enable_think = true
enable_finish = true
enable_history_truncation = true
enable_condensation_request = false
llm_config = {_toml(llm_key)}

[condenser]
type = "noop"

[sandbox]
volumes = {_toml(f"{workspace}:/workspace:rw")}
"""


def llm_key(served_model_name: str) -> str:
    """A TOML-safe ``[llm.<key>]`` name for the served model."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", served_model_name)


def _unb64(text: str) -> bytes:
    return base64.b64decode("".join(text.split()), validate=True)


def _dir_tar_b64(ws: Path) -> str:
    """The workspace tree as a base64 gzip tar with paths relative to ``ws``."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(ws.rglob("*")):
            rel = p.relative_to(ws).as_posix()
            if p.is_dir():
                info = tarfile.TarInfo(rel + "/")
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            elif p.is_file():
                tar.add(p, arcname=rel)
    return base64.b64encode(buf.getvalue()).decode()


def _empty_workspace_tar() -> bytes:
    """An empty ``workspace/`` tar: what an agent that deleted its workspace
    grades as (upstream's zip of a missing directory is empty too)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("workspace/")
        info.type = tarfile.DIRTYPE
        tar.addfile(info)
    return buf.getvalue()


def _extract_tar(tarball: Path, dest: Path) -> None:
    """Extract our own workspace tars (regular-file and dir members only,
    relative paths only -- a member name cannot escape ``dest``)."""
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar:
            parts = [p for p in member.name.split("/") if p not in ("", ".")]
            if not parts or ".." in parts:
                continue
            out = dest.joinpath(*parts)
            if member.isdir():
                out.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                out.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as f, open(out, "wb") as g:
                    shutil.copyfileobj(f, g)


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


@register
class NL2RepoBench(Benchmark):
    """The port above; see the module docstring for pins and deviations."""

    id = "nl2repo"
    metric = "test pass rate"
    splittable = True
    dataset = REPO
    dataset_revision = COMMIT

    # -- options -----------------------------------------------------------

    def opt(self, key: str, default: Any) -> Any:
        """A ``--option key=value`` (always a string on the CLI), cast like
        ``default``; bools parse ``false``/``0``/``no``/``off`` as false."""
        value = self.config.options.get(key)
        if value is None:
            return default
        if isinstance(default, bool):
            return str(value).strip().lower() not in ("", "0", "false", "no", "off")
        return type(default)(value)

    @property
    def gold(self) -> bool:
        """``--option agent=gold``: no model, no OpenHands. Each task's
        workspace is its ``start.md`` alone, so the grade exercises the
        fetch, the task image, the strip, the staging and the parser; how
        much passes is whatever the image itself ships."""
        return self.opt("agent", "model") == "gold"

    def needs_server(self) -> bool:
        return not self.gold

    # -- entry point -------------------------------------------------------

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        # The launch-time install: the pinned task set is fetched and verified
        # before any task is processed, never lazily mid-benchmark.
        source, revision = self.dataset_source()
        root, pinned = self._tasks_root()
        tasks = self._select(root)
        log.info(
            "%s: %d tasks x %d repeats (phase %s)",
            self.id,
            len(tasks),
            self.repeats,
            self.phase,
        )

        per_repeat, all_reports = [], []
        for k in range(self.repeats):
            repeat_dir = self.config.output_dir / f"repeat-{k}"
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.workers
            ) as pool:
                reports = list(
                    pool.map(
                        lambda t: self._task(
                            t, repeat_dir, base_url, served_model_name, k
                        ),
                        tasks,
                    )
                )
            all_reports.append(reports)
            if not self.scoring:
                failed = [
                    r["task"] for r in reports if r["status"].startswith("error:")
                ]
                per_repeat.append(
                    {
                        "repeat": k,
                        "generated": len(reports) - len(failed),
                        "failed": failed,
                    }
                )
                log.info(
                    "%s repeat %d: %d/%d generated",
                    self.id,
                    k,
                    len(reports) - len(failed),
                    len(reports),
                )
                continue
            rates = [r["rate"] for r in reports]
            per_repeat.append(
                {
                    "repeat": k,
                    "n": len(reports),
                    "test_pass_rate": sum(rates) / len(rates) if rates else 0.0,
                    "statuses": _count(r["status"] for r in reports),
                    "passed": sum(r["passed"] for r in reports),
                    "test_cases": sum(r["total"] for r in reports),
                }
            )
            log.info(
                "%s repeat %d: test pass rate %.4f",
                self.id,
                k,
                per_repeat[-1]["test_pass_rate"],
            )

        out: dict[str, Any] = {
            "dataset": source,
            "dataset_revision": revision,
            "per_repeat": per_repeat,
            "tasks": [t.name for t in tasks],
            "task_set": {
                "root": str(root),
                "tasks": len(tasks),
                "tasks_sha256": TASKS_SHA256 if pinned else None,
            },
            "images": {
                "openhands": self.opt("openhands_image", OPENHANDS_IMAGE),
                "grade_prefix": self.opt("image_prefix", IMAGE_PREFIX),
                "grade_tag": self.opt("image_tag", IMAGE_TAG),
            },
            "deviations": list(DEVIATIONS),
        }
        if not self.scoring:
            # Generated: a workspace in every repeat (failed ones are retried by rerunning generate).
            ok = [
                all(not rs[i]["status"].startswith("error:") for rs in all_reports)
                for i in range(len(tasks))
            ]
            return {"n": sum(ok), **out}
        rates = [
            [float(rs[i]["rate"]) for rs in all_reports] for i in range(len(tasks))
        ]
        failed = sum(
            any(rs[i]["status"].startswith("error:") for rs in all_reports)
            for i in range(len(tasks))
        )
        return {
            "value": sum(r["test_pass_rate"] for r in per_repeat) / len(per_repeat),
            "n": len(tasks),
            "pass_at_k": pass_at_k(
                rates, self.repeats, "test pass rate of the best of the k runs"
            ),
            **failure_policy(
                self.id, "task", failed, len(tasks), self.opt("max_failed_frac", 0.05)
            ),
            **out,
        }

    def _tasks_root(self) -> tuple[Path, bool]:
        """``(root, pinned)``: a local ``--dataset`` dir or
        ``GRANITE_EVALS_NL2REPO_DIR`` is trusted as-is (``pinned=False``);
        otherwise the fetched tree, digest-verified against the pin
        (``pinned=True``)."""
        if self.config.dataset:
            root = Path(self.config.dataset)
            if not root.is_dir():
                raise SystemExit(f"{self.id}: --dataset {root}: not a directory")
            return root, False
        if env_dir := os.environ.get(CACHE_ENV, "").strip():
            root = Path(env_dir)
            if not root.is_dir():
                raise SystemExit(f"{self.id}: {CACHE_ENV}={root}: not a directory")
            return root, False
        return TaskSet().ensure(), True

    def _select(self, root: Path) -> list[Task]:
        tasks = [Task(p) for p in sorted(root.iterdir()) if p.is_dir()]
        if pattern := self.opt("tasks", ""):
            tasks = [t for t in tasks if re.search(pattern, t.name)]
        return tasks[: self.config.limit] if self.config.limit else tasks

    # -- one task ----------------------------------------------------------

    def _task(
        self, task: Task, repeat_dir: Path, base_url: str, served: str, k: int
    ) -> dict[str, Any]:
        idir = repeat_dir / task.name
        idir.mkdir(parents=True, exist_ok=True)
        report_path = idir / "report.json"
        if report_path.exists():
            return json.loads(report_path.read_text())

        try:
            ws_tar = idir / "workspace.tar.gz"
            if not ws_tar.exists():
                if not self.generating:
                    # Unresolved under --phase all; never generated here.
                    log.warning("%s", self.not_generated(f"{task.name} repeat {k}"))
                    return {
                        "task": task.name,
                        "rate": 0.0,
                        "passed": 0,
                        "total": task.test_case_count,
                        "status": "not_generated",
                    }
                if self.gold:
                    self._gold_workspace(task, ws_tar)
                else:
                    self._generate(task, idir, base_url, served, k)
            if not self.scoring:
                return {"task": task.name, "status": "generated"}
            report = self._grade(task, idir)
        except Exception as e:  # one broken task must not sink the run
            log.exception("%s: %s failed", self.id, task.name)
            # Not persisted: an infrastructure error should be retried on resume.
            return {
                "task": task.name,
                "rate": 0.0,
                "passed": 0,
                "total": task.test_case_count,
                "status": f"error:{type(e).__name__}",
            }
        report_path.write_text(json.dumps(report, indent=2))
        log.info(
            "%s repeat %d: %s %s rate=%.4f",
            self.id,
            k,
            task.name,
            report["status"],
            report["rate"],
        )
        return report

    # -- generate ----------------------------------------------------------

    def _gold_workspace(self, task: Task, ws_tar: Path) -> None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo("workspace/")
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
            data = task.start_md.encode("utf-8")
            info = tarfile.TarInfo("workspace/start.md")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        ws_tar.write_bytes(buf.getvalue())
        (ws_tar.parent / "meta.json").write_text(json.dumps({"gold": True}, indent=2))

    def _generate(
        self, task: Task, idir: Path, base_url: str, served: str, k: int
    ) -> None:
        if not base_url:
            raise RuntimeError(
                f"{task.name}: no model endpoint (agent={self.opt('agent', 'model')!r} needs one)"
            )
        image = self.opt("openhands_image", OPENHANDS_IMAGE)
        key = llm_key(served)
        config = render_config(
            workspace=f"{RUN_DIR}/workspace",
            llm_key=key,
            model=served,
            base_url=base_url,
            max_iterations=self.opt("max_iterations", 500),
            enable_browsing=self.opt("enable_browsing", True),
            temperature=(
                None
                if "temperature" not in self.config.options
                else float(self.config.options["temperature"])
            ),
            top_p=(
                None
                if "top_p" not in self.config.options
                else float(self.config.options["top_p"])
            ),
            max_tokens=(
                None
                if "max_tokens" not in self.config.options
                else int(self.config.options["max_tokens"])
            ),
        )
        # The env upstream sets for its headless run, plus what the local
        # runtime needs. SANDBOX_RUNTIME_CONTAINER_IMAGE is not set: it only
        # names the Docker-runtime sandbox image, inert (and confusing) here.
        env = {
            "RUNTIME": "local",
            "SKIP_DEPENDENCY_CHECK": "1",
            "LOG_ALL_EVENTS": "true",
            "PYTHONUNBUFFERED": "1",
            "CONFIG_FILE": f"{RUN_DIR}/config.toml",
            "AGENT_LLM_CONFIG": key,
            "OH_TASK": TASK_PROMPT,
        }
        started = time.time()
        with make_sandbox(image, backend=self.opt("sandbox", ""), env=env) as sb:
            sb.execute(f"mkdir -p {RUN_DIR}/workspace")
            sb.write_file(f"{RUN_DIR}/workspace/start.md", task.start_md)
            sb.write_file(f"{RUN_DIR}/config.toml", config)
            result = sb.execute(
                f'cd {RUN_DIR} && python -m openhands.core.main --config-file={RUN_DIR}/config.toml -t "$OH_TASK"',
                cwd=RUN_DIR,
                timeout=self.opt("agent_timeout", AGENT_TIMEOUT_S) or None,
            )
            (idir / "openhands.log").write_text(result.output)
            meta = {
                "image": image,
                "model": f"openai/{served}",
                "exit_code": result.returncode,
                "timed_out": result.timed_out,
                "duration_s": round(time.time() - started, 1),
            }
            # The workspace and trajectories live in the sandbox's mounted
            # /tmp; carry them out as base64 tars over stdout.
            ws = sb.execute(
                f"set -o pipefail && tar -czf - -C {RUN_DIR} workspace 2>/dev/null | base64",
                timeout=STAGE_TIMEOUT_S,
            )
            if ws.returncode == 0 and ws.output.strip():
                ws_tar_data = _unb64(ws.output)
            else:
                log.warning(
                    "%s: staging the workspace out failed (exit %d); grading an empty workspace",
                    task.name,
                    ws.returncode,
                )
                ws_tar_data = _empty_workspace_tar()
            trajectory = sb.execute(
                f"set -o pipefail; if [ -d {RUN_DIR}/trajectories ]; then "
                f"tar -czf - -C {RUN_DIR} trajectories 2>/dev/null | base64; fi",
                timeout=STAGE_TIMEOUT_S,
            )
            if trajectory.returncode == 0 and trajectory.output.strip():
                (idir / "trajectory.tar.gz").write_bytes(_unb64(trajectory.output))
        (idir / "workspace.tar.gz").write_bytes(ws_tar_data)
        (idir / "meta.json").write_text(json.dumps(meta, indent=2))

    # -- grade -------------------------------------------------------------

    def _grade(self, task: Task, idir: Path) -> dict[str, Any]:
        ws_tar = idir / "workspace.tar.gz"
        base = {
            "task": task.name,
            "rate": 0.0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "total": task.test_case_count,
            "status": "graded",
        }
        if (meta_path := idir / "meta.json").exists():
            base["agent"] = json.loads(meta_path.read_text())
        if ws_tar.stat().st_size > self.opt("ws_max_mb", WS_MAX_MB) * 1_000_000:
            return {**base, "status": "workspace_too_large"}

        with tempfile.TemporaryDirectory(prefix=f"nl2repo-{task.name}-") as tmp:
            _extract_tar(ws_tar, Path(tmp))
            ws = Path(tmp) / "workspace"
            ws.mkdir(
                parents=True, exist_ok=True
            )  # an empty workspace tar has only the dir member
            stripped = strip_workspace(ws, task.test_files)
            image = task.image(
                self.opt("image_prefix", IMAGE_PREFIX), self.opt("image_tag", IMAGE_TAG)
            )
            timeout = self.opt("eval_timeout", EVAL_TIMEOUT_S)
            with make_sandbox(
                image, backend=self.opt("sandbox", ""), env={"PYTHONPATH": "/workspace"}
            ) as sb:
                # Docker's COPY workspace /workspace: extract the stripped tree
                # over the image's /workspace, modes preserved; files the agent
                # does not have (the gold tests among them) stay as the image
                # ships them.
                sb.write_file(WS_B64, _dir_tar_b64(ws))
                extract = sb.execute(
                    f"mkdir -p /workspace && base64 -d {WS_B64} | tar -xzpf - --no-same-owner -C /workspace",
                    timeout=STAGE_TIMEOUT_S,
                )
                if extract.returncode != 0:
                    raise RuntimeError(
                        f"{task.name}: staging the workspace in failed: {extract.output[-500:]}"
                    )
                command_results = []
                for command in task.commands:
                    r = sb.execute(command, cwd="/workspace", timeout=timeout)
                    command_results.append(
                        {
                            "command": command,
                            "exit_code": r.returncode,
                            "timed_out": r.timed_out,
                            "output": r.output,
                        }
                    )

        (idir / "test_output.txt").write_text(
            "\n".join(
                f"$ {r['command']}  (exit {r['exit_code']})\n{r['output']}"
                for r in command_results
            )
        )
        pytest = parse_pytest_results(command_results, task.test_case_count)
        timed_out = any(r["timed_out"] for r in command_results)
        return {
            **base,
            "rate": pytest["success_rate"],
            "passed": pytest["passed"],
            "failed": pytest["failed"],
            "errors": pytest["errors"],
            "status": "eval_timeout" if timed_out else "graded",
            "image": image,
            "stripped": stripped,
            "commands": [
                {
                    "command": r["command"],
                    "exit_code": r["exit_code"],
                    "timed_out": r["timed_out"],
                }
                for r in command_results
            ],
        }
