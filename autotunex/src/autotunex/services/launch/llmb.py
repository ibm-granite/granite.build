# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The llmb-backed TuningLauncher.

Mirrors ``HuggingFaceStorageBackend``: build the spec, write it to a per-job
spec file (``<spec_dir>/<job_id>/build.yaml``), authenticate the CLI, run
``llmb build start --quiet --format json -f <file>`` in a thread (keeping the
event loop free), and parse the handle from its JSON stdout. Unlike the HF
backend's scratch temp file, the spec is kept after submission so a launch can
be inspected or replayed. ``gb`` is an alias for ``llmb`` — the 2025
``gb build start`` contract is what this parses.

Authentication ports ``gb_service.login_gb``: before ``build start`` the launcher
runs ``llmb auth login --token <token>``, reading the token *value* from the env
var named by ``gb_token_env`` (``GB_TOKEN`` by default). The value is never loaded
into ``Settings`` — the same contract as the HuggingFace token — and is never
logged. When the var is unset, authentication is skipped and the CLI's ambient
credentials are used.

OPEN ITEM (design §15): the exact ``llmb build start`` subcommand/flags are taken
from the known 2025 ``gb`` contract; confirm against the installed binary. Only
``_run_llmb`` / ``_parse_output`` change if it differs.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from autotunex.core.logging import get_logger
from autotunex.services.launch.protocols import LaunchContext, LaunchHandle

logger = get_logger(__name__)


def _authenticate(llmb_command: str, token_env: str) -> None:
    """Run ``llmb auth login --token <token>`` when the token env var is set.

    Ports ``gb_service.login_gb`` and is shared by the launcher and the canceller.
    The token value is read from the environment (its var name is ``token_env``),
    never loaded into ``Settings`` or logged; a failed login raises ``RuntimeError``
    carrying stderr but never the token (deliberately not ``check=True``, whose
    ``CalledProcessError`` repr would embed the full argv, including the token).
    When the var is unset, authentication is skipped and the CLI's ambient
    credentials are used.
    """
    token = os.environ.get(token_env)
    if not token:
        logger.info(
            "%s not set; skipping `llmb auth login` (using the CLI's ambient credentials).",
            token_env,
        )
        return
    logger.info("Authenticating the llmb CLI via `%s auth login`.", llmb_command)
    result = subprocess.run(
        [llmb_command, "auth", "login", "--token", token],
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise RuntimeError(
            f"`{llmb_command} auth login` failed (exit {result.returncode}): {detail}"
        )


class LlmbTuningLauncher:
    """Submits a build via the ``llmb`` CLI. Satisfies :class:`TuningLauncher`."""

    def __init__(
        self,
        *,
        llmb_command: str,
        spec_dir: Path,
        token_env: str,
        tags: str,
        spec_builder: Callable[[LaunchContext], str],
    ) -> None:
        """Store the CLI command, spec directory, token env var, tags, and spec builder.

        Args:
            llmb_command: the ``llmb`` (or ``gb``) CLI command to invoke.
            spec_dir: root directory the per-job ``build.yaml`` is written under.
            token_env: name of the env var holding the CLI auth token value.
            tags: value passed verbatim as ``build start``'s ``--tag`` argument
                (a single tag or a comma-separated list; ``llmb`` splits it). An
                empty or whitespace-only value omits the flag entirely.
            spec_builder: the pure builder turning a :class:`LaunchContext` into
                the ``build.yaml`` text. The registry binds this to the
                custom_code builder (``spec.build_spec``), the local-bash
                builder (``bash_spec.build_bash_spec``), or the LSF/SkyPilot
                builder (``lsf_spec.build_lsf_spec``) with its extra arguments
                already applied, so the launcher is agnostic to which spec shape
                it submits.
        """
        self._llmb_command = llmb_command
        self._spec_dir = spec_dir
        self._token_env = token_env
        self._tags = tags
        self._spec_builder = spec_builder

    async def launch(self, ctx: LaunchContext) -> LaunchHandle:
        """Assemble the spec, submit it via ``llmb``, and parse the handle."""
        spec = self._spec_builder(ctx)
        completed = await asyncio.to_thread(self._run_llmb, spec, ctx.job_id)
        return self._parse_output(completed.stdout)

    def _run_llmb(self, spec: str, job_id: UUID) -> subprocess.CompletedProcess[str]:
        """Write ``<spec_dir>/<job_id>/build.yaml`` and run ``llmb build start`` (blocking).

        The spec file is deliberately *kept* after the run — not a scratch temp
        file — so a failed or surprising launch can be inspected or replayed by
        hand; the resolved path is logged at INFO for exactly that purpose. Each
        job gets its own ``<job_id>/`` directory, so concurrent launches never
        collide and a re-submit overwrites only its own spec.

        The full parent environment is forwarded so ``llmb`` reads its own
        cluster credentials from it — never loaded into ``Settings``, exactly as
        the HuggingFace storage backend does.

        A non-zero exit raises ``RuntimeError`` carrying llmb's ``stderr`` (or
        ``stdout`` when stderr is empty). Deliberately **not** ``check=True``:
        ``CalledProcessError`` keeps the CLI's output only in attributes, not in
        its string form, so ``InProcessJobRunner.process``'s ``logger.exception``
        would record the exit code without the cluster's actual error — leaving a
        failed launch undiagnosable. Surfacing the output turns "exit status 1"
        into the real reason (a rejected spec, an unknown step, an auth failure).
        Unlike ``_authenticate``'s argv, the ``build start`` argv carries no token,
        so echoing it is safe — but the message quotes only the output, not argv.
        """
        self._authenticate()
        spec_path = self._spec_dir / str(job_id) / "build.yaml"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(spec)
        logger.info("Wrote build spec for job %s to %s", job_id, spec_path)
        args = [
            self._llmb_command,
            "build",
            "start",
            "--quiet",
            "--format",
            "json",
            "-f",
            str(spec_path),
        ]
        # An empty/whitespace-only setting means "no tags": omit --tag rather than
        # pass an empty value the CLI would reject.
        if self._tags.strip():
            args += ["--tag", self._tags]
        completed = subprocess.run(args, capture_output=True, text=True, env=dict(os.environ))
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
            raise RuntimeError(
                f"`{self._llmb_command} build start` failed (exit {completed.returncode}): {detail}"
            )
        return completed

    def _authenticate(self) -> None:
        """Authenticate the llmb CLI (delegates to the shared module helper)."""
        _authenticate(self._llmb_command, self._token_env)

    def _parse_output(self, stdout: str) -> LaunchHandle:
        """Parse ``llmb build start --format json`` output into a handle.

        With ``--quiet --format json`` (verified in
        ``gbcli/commands/command_build.py:763-765``) stdout is exactly
        ``{"build_url": <web UI URL>, "uuid": <build_id>}``. ``build_id`` is the
        sole status key the reconcile loop uses, so a missing or unparseable
        ``uuid`` is a launch failure: raise, and ``InProcessJobRunner.process``
        marks the job ``error`` — a loud failure at submit time beats a zombie
        ``pending`` job no later mechanism can rescue.

        Raises:
            RuntimeError: stdout was not JSON, or carried no parseable build id.
        """
        try:
            payload = json.loads(stdout)
            build_id = UUID(str(payload["uuid"]))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"`llmb build start` returned no parseable build id: {stdout!r}"
            ) from exc
        pr_url = payload.get("build_url")
        return LaunchHandle(
            build_id=build_id,
            pr_url=pr_url if isinstance(pr_url, str) else None,
        )


class LlmbBuildCanceller:
    """Cancels a build via ``llmb build cancel``. Satisfies :class:`BuildCanceller`.

    Mirrors :class:`LlmbTuningLauncher`: authenticate, then run the CLI in a thread
    so the event loop stays free. A non-zero exit raises ``RuntimeError`` carrying
    the CLI's stderr/stdout (deliberately not ``check=True`` — see the launcher).
    """

    def __init__(self, *, llmb_command: str, token_env: str) -> None:
        self._llmb_command = llmb_command
        self._token_env = token_env

    async def cancel(self, build_id: UUID) -> None:
        """Request cancellation of ``build_id`` via the CLI, off the event loop."""
        await asyncio.to_thread(self._run_cancel, build_id)

    def _run_cancel(self, build_id: UUID) -> None:
        """Authenticate and run ``llmb build cancel <build_id>`` (blocking)."""
        _authenticate(self._llmb_command, self._token_env)
        # --skip-version-check: some granite.build CLI builds abort NONZERO on their
        # own "you're outdated" check before doing any work (e.g. a shallow-cloned
        # CLI stamped 0.0.1.dev1 by setuptools_scm), which turns every cancel into a
        # spurious BuildCancelUpstreamError. The check is advisory noise for an
        # automated caller, so opt out of it here. Deliberately scoped to cancel:
        # build start is left to rely on a correctly-versioned CLI (see Dockerfile.aio).
        args = [
            self._llmb_command,
            "build",
            "cancel",
            "--quiet",
            "--format",
            "json",
            "--skip-version-check",
            str(build_id),
        ]
        completed = subprocess.run(args, capture_output=True, text=True, env=dict(os.environ))
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
            raise RuntimeError(
                f"`{self._llmb_command} build cancel` failed "
                f"(exit {completed.returncode}): {detail}"
            )
