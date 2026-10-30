"""SWE-bench family (Granite: SWE Bench Verified / Pro / Multilingual).

Metric: pass@1 resolve rate, the resolve rate of one agent run per instance
(``repeats`` 1). With ``--repeats k`` the value is pass@1[avg-of-k], the resolve
rate of each of the k independent runs, averaged, and ``details.pass_at_k``
adds pass@k: the fraction of instances resolved in at least one of the k.

Verified and Multilingual share one pipeline: their HF datasets carry each
instance's image, eval_script and log_parser, from which swebench's harness
builds the test spec and the grade. SWE-bench Pro (V2) ships its own images
and verifier; :class:`SWEBenchPro` swaps both phases for Scale's protocol.

Two phases per instance and repeat, both inside a per-instance sandbox built
from the instance's SWE-bench image:

1. generate: mini-swe-agent (upstream loop, prompts and swebench.yaml config)
   drives the served model until it submits a patch.
2. grade: a *fresh* sandbox applies the patch, runs the dataset's eval_script
   and scores the log with swebench's own ``get_eval_report``.

Everything is written under ``<output_dir>/repeat-<k>/<instance_id>/`` and a
finished instance is skipped on restart, so granite.build retries resume.

``--phase generate`` runs only step 1 and stops at ``patch.diff``; ``--phase
score`` runs only step 2 on those patches (no model; an instance without one is
``not_generated``, unresolved). ``report.json`` marks a graded instance.
"""

from __future__ import annotations

import concurrent.futures
import functools
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, ClassVar

from granite_evals import data
from granite_evals.registry import Benchmark, pass_at_k, register
from granite_evals.sandbox import make_sandbox
from granite_evals.sandbox.nodelock import NodeLock, node_locks  # noqa: F401

log = logging.getLogger(__name__)

PATCH_FILE = "/tmp/patch.diff"
EVAL_TIMEOUT_S = 1800

MAVEN_MIRROR_ENV = "GRANITE_EVALS_MAVEN_MIRROR"
"""URL of a Maven Central mirror for the instance sandboxes (unset: none).
The Java images resolve a few artifacts at test time, e.g. surefire's junit
provider (Maven: druid, gson, javaparser) or dependencies the Gradle cache
lacks (lucene, rxjava); where repo.maven.apache.org refuses the node's egress
IP (HTTP 429 on every request), those tests fail on the environment, not the
patch. Maven gets a settings.xml mirror, Gradle an init script."""
MAVEN_SETTINGS_STAGED = "/tmp/granite-maven-settings.xml"
# cp -n: a settings.xml the image ships wins.
MAVEN_SETTINGS_INSTALL = (
    f'mkdir -p "$HOME/.m2" && cp -n {MAVEN_SETTINGS_STAGED} "$HOME/.m2/settings.xml"; rm -f {MAVEN_SETTINGS_STAGED}'
)


def maven_settings(mirror: str) -> str:
    """A user settings.xml that sends Maven Central (repository id ``central``)
    to ``mirror``; other repositories a project declares are left alone."""
    from xml.sax.saxutils import escape

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<settings xmlns="http://maven.apache.org/SETTINGS/1.2.0">\n'
        "  <mirrors>\n"
        "    <mirror>\n"
        "      <id>granite-central-mirror</id>\n"
        "      <mirrorOf>central</mirrorOf>\n"
        f"      <url>{escape(mirror)}</url>\n"
        "    </mirror>\n"
        "  </mirrors>\n"
        "</settings>\n"
    )


GRADLE_INIT_STAGED = "/tmp/granite-gradle-mirror.gradle"
# Gradle runs every script in init.d; cp -n: one the image ships under that name wins.
GRADLE_INIT_INSTALL = (
    'd="${GRADLE_USER_HOME:-$HOME/.gradle}/init.d" && mkdir -p "$d" && '
    f'cp -n {GRADLE_INIT_STAGED} "$d/granite-maven-mirror.gradle"; rm -f {GRADLE_INIT_STAGED}'
)
MAVEN_CENTRAL_HOSTS = ("repo.maven.apache.org", "repo1.maven.org")


def gradle_init_script(mirror: str) -> str:
    """A Gradle init script that points every Maven Central repository (by host:
    ``mavenCentral()`` or an explicit URL) at ``mirror``, in project, buildscript
    and settings repositories; other repositories are left alone. Groovy DSL,
    ``all`` rather than ``configureEach`` so old Gradle versions run it too."""
    quoted = "'" + mirror.replace("\\", "\\\\").replace("'", "\\'") + "'"
    hosts = ", ".join(f"'{h}'" for h in MAVEN_CENTRAL_HOSTS)
    return f"""// granite-evals: Maven Central -> {MAVEN_MIRROR_ENV}
def graniteMirror = {quoted}
def graniteRedirect = {{ repos ->
    repos.all {{ r ->
        if (r instanceof MavenArtifactRepository && r.url != null && [{hosts}].contains(r.url.host)) {{
            r.url = graniteMirror
        }}
    }}
}}
allprojects {{
    graniteRedirect(buildscript.repositories)
    graniteRedirect(repositories)
}}
if (gradle.gradleVersion.tokenize('.')[0].toInteger() >= 6) {{
    settingsEvaluated {{ s ->
        graniteRedirect(s.pluginManagement.repositories)
        if (s.hasProperty('dependencyResolutionManagement')) {{
            graniteRedirect(s.dependencyResolutionManagement.repositories)
        }}
    }}
}}
"""


def maven_mirror() -> str:
    return os.environ.get(MAVEN_MIRROR_ENV, "").strip()


def _install_maven_mirror(sb) -> None:
    if mirror := maven_mirror():
        sb.write_file(MAVEN_SETTINGS_STAGED, maven_settings(mirror))
        sb.execute(MAVEN_SETTINGS_INSTALL, timeout=60)
        sb.write_file(GRADLE_INIT_STAGED, gradle_init_script(mirror))
        sb.execute(GRADLE_INIT_INSTALL, timeout=60)


class SWEBench(Benchmark):
    metric = "pass@1 resolve rate"
    splittable = True
    extra = "swebench"
    harness_packages = ("mini-swe-agent", "swebench")
    split: ClassVar[str] = "test"

    # -- options -----------------------------------------------------------

    def opt(self, key: str, default: Any) -> Any:
        """A ``--option key=value`` (always a string on the CLI), cast like ``default``."""
        value = self.config.options.get(key)
        return default if value is None else type(default)(value)

    @property
    def gold(self) -> bool:
        """``--option patch=gold`` grades the reference patch instead of the
        model's: checks images, sandbox and grading on a node without a model
        (every instance should resolve)."""
        return self.opt("patch", "model") == "gold"

    @property
    def check(self) -> bool:
        """``--option check=data`` runs no agent and no tests: it checks that
        every selected instance can be graded (test spec, log parser, reference
        patch, task files) and that its image exists in its registry. It has
        nothing to generate: ``--phase generate`` only counts the instances and
        the check runs in the score (or all) phase."""
        return self.opt("check", "") == "data"

    def needs_server(self) -> bool:
        return not (self.gold or self.check)

    def dataset_config(self) -> str | None:
        """HF config name of the dataset (None = its default)."""
        return None

    def _gold_patch(self, instance: dict) -> str:
        return instance["patch"]

    def _image(self, instance: dict) -> str:
        return instance["image"]

    # -- entry point -------------------------------------------------------

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        source, revision = self.dataset_source()
        rows = data.load_split(source, revision=revision, split=self.split, name=self.dataset_config())
        if pattern := self.opt("instances", ""):
            rows = [r for r in rows if re.search(pattern, r["instance_id"])]
        instances = data.take(rows, self.config.limit, key="instance_id")
        if self.check:
            if not self.scoring:
                return {"n": len(instances), "dataset": source, "dataset_revision": revision, "mode": "check=data"}
            return {"dataset": source, "dataset_revision": revision, **self._check_data(instances)}
        log.info("%s: %d instances x %d repeats (phase %s)", self.id, len(instances), self.repeats, self.phase)

        per_repeat, all_reports = [], []
        for k in range(self.repeats):
            repeat_dir = self.config.output_dir / f"repeat-{k}"
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.workers) as pool:
                reports = list(
                    pool.map(
                        lambda inst: self._instance(inst, repeat_dir, base_url, served_model_name, k),
                        instances,
                    )
                )
            all_reports.append(reports)
            if not self.scoring:
                failed = [r["instance_id"] for r in reports if r["status"].startswith("error:")]
                per_repeat.append({"repeat": k, "generated": len(reports) - len(failed), "failed": failed})
                log.info("%s repeat %d: %d/%d generated", self.id, k, len(reports) - len(failed), len(reports))
                continue
            resolved = sum(r["resolved"] for r in reports)
            per_repeat.append(
                {
                    "repeat": k,
                    "resolved": resolved,
                    "n": len(reports),
                    "resolve_rate": resolved / len(reports) if reports else 0.0,
                    "statuses": _count(r["status"] for r in reports),
                    **self._repeat_details(reports),
                }
            )
            log.info("%s repeat %d: %d/%d resolved", self.id, k, resolved, len(reports))

        out = {
            "dataset": source,
            "dataset_revision": revision,
            "per_repeat": per_repeat,
            "instances": [i["instance_id"] for i in instances],
            "maven_mirror": maven_mirror() or None,
        }
        if not self.scoring:
            # Generated: a patch in every repeat (failed ones are retried by rerunning generate).
            ok = [all(not rs[i]["status"].startswith("error:") for rs in all_reports) for i in range(len(instances))]
            return {"n": sum(ok), **out}
        resolved = [[bool(rs[i]["resolved"]) for rs in all_reports] for i in range(len(instances))]
        return {
            "value": sum(r["resolve_rate"] for r in per_repeat) / len(per_repeat),
            "n": len(instances),
            "pass_at_k": pass_at_k(resolved, self.repeats, "resolved in any of the k runs"),
            **out,
        }

    def _repeat_details(self, reports: list[dict]) -> dict[str, Any]:
        """Extra per-repeat counts for results.json."""
        return {}

    # -- one instance ------------------------------------------------------

    def _instance(self, instance: dict, repeat_dir: Path, base_url: str, served: str, k: int) -> dict:
        iid = instance["instance_id"]
        idir = repeat_dir / iid
        idir.mkdir(parents=True, exist_ok=True)
        report_path = idir / "report.json"
        if report_path.exists():
            return json.loads(report_path.read_text())

        try:
            patch_path = idir / "patch.diff"
            if not patch_path.exists():
                if not self.generating:
                    # Unresolved, as a failed generation is under --phase all; never generated here.
                    log.warning("%s", self.not_generated(f"{iid} repeat {k}"))
                    return {"instance_id": iid, "resolved": False, "status": "not_generated"}
                if self.gold:
                    patch_path.write_text(self._gold_patch(instance))
                else:
                    patch_path.write_text(self._generate(instance, idir, base_url, served, k))
            if not self.scoring:
                return {"instance_id": iid, "resolved": False, "status": "generated"}
            report = self._grade(instance, patch_path.read_text(), idir)
        except Exception as e:  # one broken instance must not sink the run
            log.exception("%s: %s failed", self.id, iid)
            # Not persisted: an infrastructure error should be retried on resume.
            return {"instance_id": iid, "resolved": False, "status": f"error:{type(e).__name__}"}
        report_path.write_text(json.dumps(report, indent=2))
        log.info("%s repeat %d: %s %s resolved=%s", self.id, k, iid, report["status"], report["resolved"])
        return report

    def _agent_config(self, base_url: str, served: str, k: int) -> dict:
        from minisweagent.config import builtin_config_dir, get_config_from_spec
        from minisweagent.utils.serialize import recursive_merge

        base = get_config_from_spec(builtin_config_dir / "benchmarks" / "swebench.yaml")
        model_kwargs = {"api_base": base_url, "api_key": "EMPTY", "seed": self.config.seed + k}
        # Unset sampling options fall back to the checkpoint's generation_config,
        # which vLLM applies by default.
        for key, cast in (("temperature", float), ("top_p", float), ("max_tokens", int)):
            if key in self.config.options:
                model_kwargs[key] = cast(self.config.options[key])
        return recursive_merge(
            base,
            {
                "agent": {
                    "step_limit": self.opt("step_limit", 250),
                    "cost_limit": 0,  # local model: no cost, bound by step_limit
                },
                "model": {
                    "model_name": f"hosted_vllm/{served}",
                    "cost_tracking": "ignore_errors",
                    "model_kwargs": model_kwargs,
                },
            },
        )

    def _generate(self, instance: dict, idir: Path, base_url: str, served: str, k: int) -> str:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.models import get_model

        from granite_evals.sandbox.minisweagent_env import SandboxEnvironment

        config = self._agent_config(base_url, served, k)
        env_config = {key: v for key, v in config["environment"].items() if key != "environment_class"}
        marker = f"{instance['instance_id']}-r{k}-gen-{uuid.uuid4().hex[:8]}"
        env_config["env"] = {**env_config.get("env", {}), PRO_MARKER: marker}
        env = SandboxEnvironment(image=instance["image"], backend=self.opt("sandbox", ""), **env_config)
        try:
            _install_maven_mirror(env.sandbox)
            agent = DefaultAgent(get_model(config=config["model"]), env, **config["agent"])
            exit_status, submission = "unknown", ""
            try:
                info = agent.run(instance["problem_statement"])
                exit_status, submission = info.get("exit_status"), info.get("submission") or ""
            except _agent_end_exceptions() as e:
                # mini-swe-agent's swebench runner records these as the run's
                # exit status with an empty submission (unresolved, not retried).
                log.info("%s: agent ended by %s", instance["instance_id"], type(e).__name__)
                exit_status = type(e).__name__
            finally:
                agent.save(idir / "traj.json", {"info": {"exit_status": exit_status}, "instance_id": instance["instance_id"]})
            return submission
        finally:
            env.cleanup()
            _kill_marked(marker)

    def _grade(self, instance: dict, patch: str, idir: Path) -> dict:
        from swebench.harness.utils import make_test_spec

        iid = instance["instance_id"]
        base = {"instance_id": iid, "resolved": False}
        if not patch.strip():
            return {**base, "status": "empty_patch"}
        if instance.get("image_assets"):
            raise NotImplementedError(f"{iid}: image_assets staging is not supported yet")

        spec = make_test_spec(instance)
        marker = f"{iid}-grade-{uuid.uuid4().hex[:8]}"
        try:
            return self._grade_in(instance, patch, idir, spec, marker)
        finally:
            # A timed-out eval_script's processes (mvn, gradle daemons) outlive
            # the enroot sandbox otherwise, and keep the LSF job running.
            _kill_marked(marker)

    def _grade_in(self, instance: dict, patch: str, idir: Path, spec, marker: str) -> dict:
        from swebench.harness.grading import get_eval_report
        from swebench.harness.run_evaluation import GIT_APPLY_CMDS

        iid = instance["instance_id"]
        base = {"instance_id": iid, "resolved": False}
        with make_sandbox(instance["image"], backend=self.opt("sandbox", ""), env={PRO_MARKER: marker}) as sb:
            sb.write_file(PATCH_FILE, patch)
            attempts = []
            for i, cmd in enumerate(GIT_APPLY_CMDS):
                if i:
                    sb.execute("git checkout -- . ; git clean -fd", cwd="/testbed")
                r = sb.execute(f"{cmd} {PATCH_FILE}", cwd="/testbed")
                attempts.append(f"$ {cmd} {PATCH_FILE}  (exit {r.returncode})\n{r.output}")
                if r.returncode == 0:
                    break
            (idir / "patch_apply.txt").write_text("\n".join(attempts))
            if r.returncode != 0:
                return {**base, "status": "patch_failed"}

            _install_maven_mirror(sb)
            sb.write_file("/eval.sh", spec.eval_script)
            result = sb.execute("/bin/bash /eval.sh", cwd="/testbed", timeout=self.opt("eval_timeout", EVAL_TIMEOUT_S))
            log_path = idir / "test_output.txt"
            log_path.write_text(result.output)
            if result.timed_out:
                return {**base, "status": "eval_timeout"}

        report = get_eval_report(
            test_spec=spec,
            prediction={"instance_id": iid, "model_patch": patch, "model_name_or_path": self.config.model},
            test_log_path=str(log_path),
            include_tests_status=True,
        )[iid]
        return {**base, "resolved": bool(report["resolved"]), "status": "graded", "report": report}

    # -- check=data --------------------------------------------------------

    def _check_instance(self, instance: dict) -> list[str]:
        """What would stop this instance from being graded (empty = nothing)."""
        from swebench.harness.log_parsers import PARSER_REGISTRY
        from swebench.harness.utils import make_test_spec

        problems = []
        if instance.get("image_assets"):
            problems.append("image_assets staging is not supported")
        spec = make_test_spec(instance)
        if spec.log_parser not in PARSER_REGISTRY:
            problems.append(f"unknown log_parser {spec.log_parser!r}")
        if not self._gold_patch(instance).strip():
            problems.append("empty reference patch")
        return problems

    def _check_data(self, instances: list[dict]) -> dict[str, Any]:
        def one(instance: dict) -> dict:
            out: dict[str, Any] = {"instance_id": instance["instance_id"], "image": self._image(instance)}
            try:
                problems = self._check_instance(instance)
            except Exception as e:
                problems = [f"{type(e).__name__}: {e}"]
            try:
                out["manifest"] = probe_image(out["image"])
                if out["manifest"]["status"] != 200:
                    problems.append(f"image manifest: HTTP {out['manifest']['status']}")
            except Exception as e:
                problems.append(f"image manifest: {type(e).__name__}: {e}")
            return {**out, "ok": not problems, "problems": problems}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.workers) as pool:
            results = list(pool.map(one, instances))
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "check.json").write_text(json.dumps(results, indent=2))
        ok = sum(r["ok"] for r in results)
        for r in results:
            if not r["ok"]:
                log.warning("%s check: %s %s", self.id, r["instance_id"], "; ".join(r["problems"]))
        log.info("%s check: %d/%d instances ok", self.id, ok, len(results))
        return {
            # Not a score: the fraction of instances whose data and image check out.
            "value": ok / len(results) if results else 0.0,
            "n": len(results),
            "mode": "check=data",
            "ok": ok,
            "failures": [r for r in results if not r["ok"]],
            "instances": [r["instance_id"] for r in results],
        }


MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


def parse_image_ref(ref: str) -> tuple[str, str, str]:
    """``(registry host, repository, tag)`` of a docker image reference."""
    first, _, rest = ref.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        registry, path = first, rest
    else:
        registry, path = "docker.io", ref if "/" in ref else f"library/{ref}"
    if registry == "docker.io":
        registry = "registry-1.docker.io"
    tag = "latest"
    if ":" in path.rsplit("/", 1)[-1]:
        path, _, tag = path.rpartition(":")
    return registry, path, tag


def probe_image(ref: str, *, client=None) -> dict[str, Any]:
    """HEAD the image's manifest with an anonymous pull token, as ``enroot
    import`` would fetch it. Docker Hub does not count HEADs against its pull
    rate limit."""
    import httpx

    registry, repo, tag = parse_image_ref(ref)
    url = f"https://{registry}/v2/{repo}/manifests/{tag}"
    headers = {"Accept": MANIFEST_ACCEPT}
    own = client is None
    client = client or httpx.Client(timeout=30, follow_redirects=True)
    try:
        for attempt in range(4):
            r = client.head(url, headers=headers)
            if r.status_code == 401 and "Authorization" not in headers:
                challenge = dict(re.findall(r'(\w+)="([^"]*)"', r.headers.get("www-authenticate", "")))
                params = {"service": challenge.get("service", registry), "scope": f"repository:{repo}:pull"}
                t = client.get(challenge["realm"], params=params)
                t.raise_for_status()
                headers["Authorization"] = "Bearer " + (t.json().get("token") or t.json()["access_token"])
                r = client.head(url, headers=headers)
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == 3:
                break
            time.sleep(5 * 2**attempt)
        return {"status": r.status_code, "digest": r.headers.get("docker-content-digest")}
    finally:
        if own:
            client.close()


def _agent_end_exceptions() -> tuple[type[Exception], ...]:
    """Model errors that end an agent run as the model's own outcome (the
    trajectory outgrew the context window), not an infrastructure failure to
    retry. Other errors (server down, sandbox failure) still propagate."""
    from litellm.exceptions import ContextWindowExceededError

    return (ContextWindowExceededError,)


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


@register
class SWEBenchVerified(SWEBench):
    id = "swebench-verified"
    dataset = "SWE-bench/SWE-bench_Verified"
    dataset_revision = "78f471bf655a3137b2e8a75af1501690ec009ec3"


@register
class SWEBenchMultilingual(SWEBench):
    """300 instances in 9 languages (C/C++, Go, Java, JS/TS, PHP, Ruby, Rust).
    Same agent config as Verified (mini-swe-agent runs its ``multilingual``
    subset with swebench.yaml); swebench's per-language log parsers grade it."""

    id = "swebench-multilingual"
    dataset = "SWE-bench/SWE-bench_Multilingual"
    dataset_revision = "846e647b9f33c0b51b739d005d13d85493c9af09"


# -- SWE-bench Pro -------------------------------------------------------------

PRO_REPO = "scaleapi/SWE-bench_Pro-os"
PRO_COMMIT = "66f92766bba642462d4bbe5479e83f91f9211862"  # tag v2.0.0
PRO_SHA256SUMS_SHA256 = "9d84f8507c89241d42d8b3ef911600a1ec75dbbb32687ce9b45b93318502c0bd"
"""sha256 of ``v2/SHA256SUMS`` at PRO_COMMIT; that file hashes every file under v2/."""
PRO_AGENT_CONFIG = "tooling/configs/mini_toolcall.yaml"
PRO_VERIFIER_TIMEOUT_S = 3000  # task.toml [verifier] timeout_sec
PRO_AGENT_BUDGET_S = 3000 - 60  # task.toml [agent] timeout_sec, less locked_mini_swe's margin
# locked_mini_swe.py's _CAPTURE: the agent's work is its staged `git diff`,
# whatever it submitted and however it stopped.
PRO_CAPTURE = (
    "repo=$(git -C /app rev-parse --show-toplevel 2>/dev/null "
    "|| git -C /testbed rev-parse --show-toplevel 2>/dev/null || echo /app); "
    'cd "$repo" && git add -A 2>/dev/null; git diff --cached > /tmp/model.patch 2>/dev/null; '
    "git reset -q 2>/dev/null; true"
)
# patch_replay.py's apply chain. It grades whatever applied, even if every step failed.
PRO_APPLY = (
    "git apply --verbose /tmp/replay.patch || git apply --3way /tmp/replay.patch "
    "|| patch --fuzz=3 -p1 -i /tmp/replay.patch"
)
# The repo is at /app (a few tasks use /testbed), as every V2 script assumes.
PRO_WORKDIR = "if [ -d /app ]; then echo /app; else echo /testbed; fi"
# Under Harbor the verifier's services (NodeBB's redis-server and test server,
# qutebrowser's Xvfb) die with its container. enroot has no PID namespace, so
# test.sh runs in one of its own when the probe says the sandbox can make one;
# otherwise they outlive the sandbox (the marker kill still gets them) and the
# report says ``verifier_pid_ns: false``.
PRO_PIDNS = "unshare --pid --fork --mount-proc --kill-child"
PRO_PIDNS_PROBE = f"{PRO_PIDNS} true"
PRO_VERIFY_LOG = "> /logs/verifier/test-stdout.txt 2>&1"
PRO_VERIFY_PIDNS_CMD = f"{PRO_PIDNS} /tests/test.sh {PRO_VERIFY_LOG}"
PRO_VERIFY_PLAIN_CMD = f"/tests/test.sh {PRO_VERIFY_LOG}"
# LISTEN sockets in the sandbox's network namespace (the host's, under enroot),
# from /proc so no tool is needed: shows what a fixed-port service would hit.
PRO_LISTENING = "awk 'NR > 1 && $4 == \"0A\" {print $2}' /proc/net/tcp /proc/net/tcp6 2>/dev/null | sort -u"
# The verifier's raw logs that test.sh copies to /logs/verifier, capped.
PRO_VERIFIER_LOGS = ("run-script-stdout.txt", "run-script-stderr.txt")
PRO_LOG_CAP = 4_000_000


class ProTasks:
    """The V2 Harbor task directories (``v2/tasks/<instance_id>/``) of the
    SWE-bench_Pro-os repo at PRO_COMMIT: instruction, reference patch and the
    verifier (test.sh, run_script.sh, parser.py, config.json, test_patch.patch).
    Files are fetched one by one, checked against the pinned SHA256SUMS and
    cached under ``root``."""

    def __init__(self, root: Path, *, commit: str = PRO_COMMIT, fetch=None):
        self.root, self.commit = root / commit, commit
        self._fetch = fetch or self._http_get
        self._lock = threading.Lock()
        self._sums: dict[str, str] | None = None

    def _http_get(self, path: str) -> bytes:
        import httpx

        url = f"https://raw.githubusercontent.com/{PRO_REPO}/{self.commit}/v2/{path}"
        for attempt in range(5):
            try:
                r = httpx.get(url, timeout=60, follow_redirects=True)
                r.raise_for_status()
                return r.content
            except httpx.HTTPError:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def _get(self, path: str, sha256: str | None) -> bytes:
        cached = self.root / path
        if cached.exists():
            content = cached.read_bytes()
            if sha256 is None or hashlib.sha256(content).hexdigest() == sha256:
                return content
        content = self._fetch(path)
        digest = hashlib.sha256(content).hexdigest()
        if sha256 is not None and digest != sha256:
            raise RuntimeError(f"{PRO_REPO}@{self.commit} v2/{path}: sha256 {digest}, expected {sha256}")
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_name(f"{cached.name}.{threading.get_ident()}.partial")
        tmp.write_bytes(content)
        tmp.replace(cached)
        return content

    def sums(self) -> dict[str, str]:
        with self._lock:
            if self._sums is None:
                expected = PRO_SHA256SUMS_SHA256 if self.commit == PRO_COMMIT else None
                sums = {}
                for line in self._get("SHA256SUMS", expected).decode().splitlines():
                    digest, _, path = line.strip().partition("  ")
                    if path:
                        sums[path] = digest
                self._sums = sums
            return self._sums

    def file(self, path: str) -> str:
        """``v2/<path>``, verified."""
        sums = self.sums()
        if path not in sums:
            raise KeyError(f"{path} is not in {PRO_REPO}@{self.commit} v2/SHA256SUMS")
        return self._get(path, sums[path]).decode("utf-8")

    def task_file(self, iid: str, rel: str) -> str:
        return self.file(f"tasks/{iid}/{rel}")

    def tests(self, iid: str) -> dict[str, str]:
        """The verifier files, keyed by their path under /tests."""
        prefix = f"tasks/{iid}/tests/"
        names = sorted(p[len(prefix) :] for p in self.sums() if p.startswith(prefix))
        if "test.sh" not in names:
            raise KeyError(f"no verifier for {iid} in {PRO_REPO}@{self.commit}")
        return {name: self.file(prefix + name) for name in names}


PRO_MARKER = "GRANITE_EVALS_SANDBOX_ID"


def _kill_marked(marker: str) -> int:
    """SIGKILL every process whose environment carries ``PRO_MARKER=marker``.

    enroot gives a sandbox no PID namespace, so what its commands leave running
    (an agent's ``redis-server --daemonize``, a command killed by its timeout)
    outlives the sandbox, holds its fixed ports and answers the next task's
    tests. Every command of a Pro sandbox carries the marker, so this finds them
    (as sandbox/harbor_env.py does for its fallback)."""
    import signal

    needle = f"{PRO_MARKER}={marker}".encode()
    killed = 0
    if not os.path.isdir("/proc"):  # not Linux: podman/docker sandboxes clean up themselves
        return 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            if needle in (Path("/proc") / entry / "environ").read_bytes().split(b"\0"):
                os.kill(int(entry), signal.SIGKILL)
                killed += 1
        except (OSError, ProcessLookupError):
            continue
    if killed:
        log.info("killed %d process(es) left by sandbox %s", killed, marker)
    return killed


# -- fixed resources shared by every sandbox on a node ------------------------
#
# enroot sandboxes also share the host's network namespace, so two tasks whose
# scripts start a service on a fixed port (NodeBB: redis-server on 6379 and the
# forum on the port its run_script writes into config.json; qutebrowser: Xvfb
# on display :99) clash when they run at once, in this job or in another job on
# the node. Such tasks take a node-wide lock per resource around the whole agent
# run and the whole grade. The keys come from the scripts that run (run_script.sh
# and test.sh), not from test_patch, whose port literals are test data.

PRO_RESOURCE_SCRIPTS = ("run_script.sh", "test.sh")
_REDIS = re.compile(r"\bredis-server\b(?P<args>[^\n;&|]*)")
_REDIS_PORT = re.compile(r"--port[= ](\d+)")
_DISPLAY = re.compile(r"\b(?:Xvfb|Xvnc|Xephyr)\s+:(\d+)|\bDISPLAY=[\"']?:(\d+)")
_PORTS = (
    re.compile(r"\b(?:localhost|127\.0\.0\.1|0\.0\.0\.0):(\d{2,5})\b"),
    re.compile(r"--port[= ](\d{2,5})\b"),
    re.compile(r"\"port\"\s*:\s*\"?(\d{2,5})\b"),
)


def pro_fixed_resources(tests: dict[str, str]) -> list[str]:
    """Node-wide lock keys (``port-6379``, ``x99``) for the fixed ports and X
    displays the task's scripts use."""
    keys: set[str] = set()
    for name in PRO_RESOURCE_SCRIPTS:
        text = tests.get(name, "")
        for m in _REDIS.finditer(text):
            port = _REDIS_PORT.search(m["args"])
            keys.add(f"port-{port[1] if port else 6379}")
        for m in _DISPLAY.finditer(text):
            keys.add(f"x{m[1] or m[2]}")
        for pattern in _PORTS:
            keys.update(f"port-{p}" for p in pattern.findall(text))
    keys.discard("port-0")  # redis-server --port 0: no TCP port
    return sorted(keys)


def _last_line(output: str) -> str:
    lines = output.strip().splitlines()
    return lines[-1].strip() if lines else ""


@register
class SWEBenchPro(SWEBench):
    """SWE-bench Pro V2 (the public set, 642 tasks), graded by Scale's verifier.

    Follows the locked protocol of ``v2/README.md`` at PRO_COMMIT, as its
    ``locked_mini_swe`` and ``patch_replay`` agents run it under Harbor:

    - generate: mini-swe-agent with the builtin ``mini`` config overlaid by
      ``v2/tooling/configs/mini_toolcall.yaml`` (tool calling) solves the
      task's ``instruction.md`` in the task's image (``docker_image``) within
      a 50-minute budget. Its work is its staged ``git diff``, captured
      whatever it submitted and however it stopped.
    - grade: a *fresh* sandbox from the same image applies that diff with
      patch_replay's apply chain, gets the task's ``tests/`` at /tests and runs
      ``/tests/test.sh``, which writes ``/logs/verifier/reward.txt``. Reward 1
      means resolved.

    ``--option subset=hard`` runs the HARD-51 subset (HF config ``hard``).
    The phases split as in :class:`SWEBench`: generate ends at the captured diff.
    """

    id = "swebench-pro"
    dataset = "ScaleAI/SWE-bench_Pro"
    dataset_revision = "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50"
    harness_packages = ("mini-swe-agent",)

    def dataset_config(self) -> str | None:
        subset = self.opt("subset", "default")
        if subset not in ("default", "hard"):
            raise SystemExit(
                f"swebench-pro: subset={subset!r} is not supported (default = V2, hard = HARD-51); "
                "v1 needs Scale's retired v1 pipeline"
            )
        return subset

    @functools.cached_property
    def tasks(self) -> ProTasks:
        return ProTasks(self.config.output_dir / "swebench-pro-tasks")

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        out = super().run(base_url, served_model_name)
        out.pop("maven_mirror", None)  # Pro's own sandboxes do not install it
        out["subset"] = self.dataset_config()
        out["verifier"] = {"repo": PRO_REPO, "commit": PRO_COMMIT, "sha256sums": PRO_SHA256SUMS_SHA256}
        if self.scoring and "per_repeat" in out:
            total: dict[str, int] = {}
            for rep in out["per_repeat"]:
                for key, n in rep.get("verifier_pid_ns", {}).items():
                    total[key] = total.get(key, 0) + n
            out["verifier_pid_ns"] = total
        return out

    def _repeat_details(self, reports: list[dict]) -> dict[str, Any]:
        # Graded instances only: an empty patch or an error never ran the verifier.
        return {"verifier_pid_ns": _count(str(r["verifier_pid_ns"]).lower() for r in reports if "verifier_pid_ns" in r)}

    def _gold_patch(self, instance: dict) -> str:
        # What Harbor's oracle agent applies.
        return self.tasks.task_file(instance["instance_id"], "solution/gold_patch.diff")

    def _image(self, instance: dict) -> str:
        return instance["docker_image"]

    def _check_instance(self, instance: dict) -> list[str]:
        # Every file the run reads, fetched and checked against SHA256SUMS.
        iid = instance["instance_id"]
        self.tasks.file(PRO_AGENT_CONFIG)
        problems = []
        if not self.tasks.task_file(iid, "instruction.md").strip():
            problems.append("empty instruction.md")
        if not self._gold_patch(instance).strip():
            problems.append("empty solution/gold_patch.diff")
        tests = self.tasks.tests(iid)
        problems += [f"no tests/{name}" for name in ("run_script.sh", "parser.py") if name not in tests]
        return problems

    def _agent_config(self, base_url: str, served: str, k: int) -> dict:
        import yaml
        from minisweagent.config import builtin_config_dir, get_config_from_spec
        from minisweagent.utils.serialize import recursive_merge

        # Harbor's MiniSweAgent passes `-c mini -c <config_file>`: the builtin
        # mini.yaml with the protocol's file layered on top.
        base = recursive_merge(
            get_config_from_spec(builtin_config_dir / "mini.yaml"),
            yaml.safe_load(self.tasks.file(PRO_AGENT_CONFIG)),
        )
        model_kwargs = {"api_base": base_url, "api_key": "EMPTY", "seed": self.config.seed + k}
        for key, cast in (("temperature", float), ("top_p", float), ("max_tokens", int)):
            if key in self.config.options:
                model_kwargs[key] = cast(self.config.options[key])
        return recursive_merge(
            base,
            {
                "agent": {
                    "step_limit": self.opt("step_limit", int(base["agent"].get("step_limit", 0))),
                    "cost_limit": 0,  # local model: no cost, bound by the time budget
                    "wall_time_limit_seconds": self.opt("agent_timeout", PRO_AGENT_BUDGET_S),
                },
                "model": {
                    "model_name": f"hosted_vllm/{served}",
                    "cost_tracking": "ignore_errors",
                    "model_kwargs": model_kwargs,
                },
            },
        )

    def _generate(self, instance: dict, idir: Path, base_url: str, served: str, k: int) -> str:
        iid = instance["instance_id"]
        instruction = self.tasks.task_file(iid, "instruction.md")
        config = self._agent_config(base_url, served, k)
        env_config = {key: v for key, v in config.get("environment", {}).items() if key != "environment_class"}
        # Harbor runs mini-swe-agent's LocalEnvironment inside the container:
        # its default 30 s command timeout, in the image's repo directory.
        env_config.setdefault("timeout", 30)
        marker = f"{iid}-r{k}-gen-{uuid.uuid4().hex[:8]}"  # unique across jobs on one node
        env_config["env"] = {**env_config.get("env", {}), PRO_MARKER: marker}
        # The agent may start the task's services itself (it can read run_script.sh).
        with node_locks(pro_fixed_resources(self.tasks.tests(iid)), what=f"{iid} generate"):
            return self._generate_in(instance, idir, config, env_config, instruction, marker)

    def _generate_in(self, instance: dict, idir: Path, config: dict, env_config: dict, instruction: str, marker: str) -> str:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.models import get_model

        from granite_evals.sandbox.minisweagent_env import SandboxEnvironment

        iid = instance["instance_id"]
        env = SandboxEnvironment(image=self._image(instance), backend=self.opt("sandbox", ""), **env_config)
        try:
            env.config.cwd = _last_line(env.sandbox.execute(PRO_WORKDIR).output) or "/app"
            # mini.yaml puts the platform into the prompt; LocalEnvironment in
            # the container reports the container's uname, so ask the sandbox.
            uname = env.sandbox.execute("uname -s; uname -r; uname -v; uname -m").output.strip().splitlines()
            platform_vars = dict(zip(("system", "release", "version", "machine"), uname[-4:]))
            agent = DefaultAgent(get_model(config=config["model"]), env, **config["agent"])
            exit_status = "unknown"
            try:
                info = agent.run(instruction, **platform_vars)
                exit_status = info.get("exit_status")
            except _agent_end_exceptions() as e:
                log.info("%s: agent ended by %s", iid, type(e).__name__)
                exit_status = type(e).__name__  # its work so far is still captured and graded
            finally:
                agent.save(idir / "traj.json", {"info": {"exit_status": exit_status}, "instance_id": iid})
                capture = env.sandbox.execute(PRO_CAPTURE, cwd=env.config.cwd, timeout=300)
                (idir / "capture.txt").write_text(f"exit {capture.returncode}\n{capture.output}")
            patch = env.sandbox.execute("cat /tmp/model.patch", timeout=300)
            if patch.returncode != 0:
                raise RuntimeError(f"{iid}: reading the captured patch failed: {patch.output[-500:]}")
            return patch.output
        finally:
            env.cleanup()
            _kill_marked(marker)

    def _grade(self, instance: dict, patch: str, idir: Path) -> dict:
        iid = instance["instance_id"]
        base = {"instance_id": iid, "resolved": False}
        if not patch.strip():
            # patch_replay would apply nothing and run the verifier; V2's release
            # gate fails every task on an empty patch, so the run is skipped.
            return {**base, "status": "empty_patch"}
        tests = self.tasks.tests(iid)
        marker = f"{iid}-grade-{uuid.uuid4().hex[:8]}"
        keys = pro_fixed_resources(tests)
        # Held until the marker kill is done, so the next holder finds the port free.
        with node_locks(keys, what=f"{iid} grade") as waited:
            try:
                report = self._grade_in(instance, patch, idir, tests, marker)
            finally:
                _kill_marked(marker)
        return {**report, "node_locks": keys, "lock_wait_s": round(waited, 1)} if keys else report

    def _grade_in(self, instance: dict, patch: str, idir: Path, tests: dict[str, str], marker: str) -> dict:
        base = {"instance_id": instance["instance_id"], "resolved": False}
        with make_sandbox(self._image(instance), backend=self.opt("sandbox", ""), env={PRO_MARKER: marker}) as sb:
            workdir = _last_line(sb.execute(PRO_WORKDIR).output) or "/app"
            sb.write_file("/tmp/replay.patch", patch)
            applied = sb.execute(PRO_APPLY, cwd=workdir, timeout=300)
            (idir / "patch_apply.txt").write_text(f"$ {PRO_APPLY}  (exit {applied.returncode})\n{applied.output}")
            for name, content in tests.items():
                sb.write_file(f"/tests/{name}", content)
            sb.execute("mkdir -p /logs/verifier && chmod +x /tests/test.sh")
            (idir / "listening.txt").write_text(sb.execute(PRO_LISTENING).output)
            probe = sb.execute(PRO_PIDNS_PROBE, timeout=60)
            pid_ns = probe.returncode == 0
            if not pid_ns:
                log.warning(
                    "%s: no PID namespace for the verifier (%s exit %d: %s); its services outlive it until the marker kill",
                    instance["instance_id"], PRO_PIDNS_PROBE, probe.returncode, _last_line(probe.output),
                )
            result = sb.execute(
                PRO_VERIFY_PIDNS_CMD if pid_ns else PRO_VERIFY_PLAIN_CMD,
                cwd=workdir,
                timeout=self.opt("eval_timeout", PRO_VERIFIER_TIMEOUT_S),
            )
            (idir / "test_output.txt").write_text(sb.execute("cat /logs/verifier/test-stdout.txt").output)
            output_json = sb.execute("cat /logs/verifier/output.json")
            if output_json.returncode == 0:
                (idir / "output.json").write_text(output_json.output)
            for name in PRO_VERIFIER_LOGS:
                log_file = sb.execute(f"tail -c {PRO_LOG_CAP} /logs/verifier/{name}")
                if log_file.returncode == 0:
                    (idir / name).write_text(log_file.output)
            reward = sb.execute("cat /logs/verifier/reward.txt")

        info = {"apply_rc": applied.returncode, "verifier_rc": result.returncode, "verifier_pid_ns": pid_ns}
        if result.timed_out:
            return {**base, **info, "status": "eval_timeout"}
        try:
            value = float(_last_line(reward.output))
        except ValueError:
            return {**base, **info, "status": "no_reward"}
        return {**base, **info, "resolved": value == 1.0, "status": "graded", "reward": value}
