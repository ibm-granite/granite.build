#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Target reuse of a checkpoint's consumers across an in-place build retry (#664).

A distillation ``train`` target emits per-checkpoint artifacts while it runs, and
``export-<N>`` targets bind them. If ``train`` dies after some exports succeeded,
the build retries in place (same build id) with ``target_reuse_enabled``. This
pins what the already-succeeded consumers do on attempt 2.

Model, run for real on the in-process Bash environment:

* ``t`` (the trainer) emits ``checkpoint_1`` (an ``env://`` path that is the
  same on every attempt), then fails on attempt 1. On attempt 2 it either
  (a) re-emits ``checkpoint_1`` at the same path and then ``checkpoint_2``, or
  (b) emits only ``checkpoint_2``; then it succeeds.
* ``e`` (the export) binds ``t.checkpoint_1`` and emits ``exported``.
* ``f`` (the eval) binds ``e.exported``.

Reuse is decided in ``BuildRun.__dispatch_target``, which only runs once every
input binding of the target is available. So a consumer is reused only if its
producer re-delivers the binding on the retry:

* (a) re-emit: ``e``'s binding arrives with the same URI, so its target hash
  matches and it is reused (not re-run). Its prior outputs are re-propagated,
  so ``f`` is dispatched and reused too. The same-build re-registration of the
  checkpoint URI is accepted (the existing registration is reused).
* (b) no re-emit: ``e`` is never dispatched on attempt 2, so neither is ``f``.
  The build still ends SUCCESS (no hang): the build loop exits once every
  dispatched task is done. Nothing is re-run, but nothing is reused either.

The last two tests make ``e`` crash on attempt 1. With (a) it is re-run and ``f``
then runs; with (b) it is silently stranded FAILED while the build reports
SUCCESS (xfail, strict).
"""

from pathlib import Path
from typing import Self
from unittest.mock import patch

import pytest
import yaml
from libgbtest.buildrunner.buildtest import AbstractBuildTest, BuildTestSpecification
from libgbtest.buildrunner.utils import ExceptionRaisingThread
from libgbtest.constants import GBTEST_USER_NAME

from gbserver.build.buildrun import BuildRun
from gbserver.buildrunner.buildrunner import BuildRunner
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status

# The Bash environment refuses to start outside STANDALONE.
pytestmark = [
    pytest.mark.standalone,
    pytest.mark.usefixtures("allow_bash_environment"),
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPACE_URI = f"file://{_REPO_ROOT / 'configurations' / 'spaces' / 'local'}"

# Paths are keyed off GB_BUILD_ID: unique per build, stable across its in-place
# retry -- what the stable per-build checkpoint dir of #664 gives the trainer.
# env:// push/pull never touch the paths; the files only count executions.
_DIR = "/tmp/gb664_$GB_BUILD_ID"
_CKPT1 = f"GB_ARTIFACT_ID:checkpoint_1 GB_ARTIFACT_PATH:{_DIR}/ckpt1"
_CKPT2 = f"GB_ARTIFACT_ID:checkpoint_2 GB_ARTIFACT_PATH:{_DIR}/ckpt2"


def _trainer_command(reemit: bool) -> str:
    """Attempt 1: emit checkpoint_1, wait for the consumers, fail.

    Attempt 2: re-emit checkpoint_1 at the same path if ``reemit``, then emit
    checkpoint_2 and succeed.
    """
    retry = (f'echo "{_CKPT1}"; ' if reemit else "") + f'echo "{_CKPT2}"; exit 0'
    return (
        f"mkdir -p {_DIR}; m={_DIR}/t.attempt1; "
        f"if [ -f $m ]; then {retry}; fi; "
        f'touch $m; echo "{_CKPT1}"; '
        # Let e and f finish on attempt 1 before the "cluster fault".
        f"i=0; while [ ! -f {_DIR}/f.done ] && [ ! -f {_DIR}/e.failed ] "
        f"&& [ $i -lt 60 ]; do sleep 1; i=$((i+1)); done; exit 1"
    )


def _consumer(name: str, binding: str, output: str, fail_first: bool) -> dict:
    """Target ``name`` binding ``binding`` and emitting env:// ``output``.

    Each execution appends to ``<name>.runs``. With ``fail_first`` the first
    execution fails (an export that crashed on attempt 1).
    """
    fail = (
        f"if [ ! -f {_DIR}/{name}.failed ]; then touch {_DIR}/{name}.failed; exit 1; fi; "
        if fail_first
        else ""
    )
    return {
        "allow_unknown": True,
        "environment_uri": "space://environments/bash",
        "inputs": {"src": {"binding": binding}},
        "outputs": {output: {"uri": "env://{{ binding.path }}", "type": "model"}},
        "steps": [
            {
                "step_uri": "space://steps/command",
                "config": {
                    "command_config": {
                        "command": (
                            f"mkdir -p {_DIR}; echo x >> {_DIR}/{name}.runs; {fail}"
                            f'echo "GB_ARTIFACT_ID:{output} '
                            f'GB_ARTIFACT_PATH:{_DIR}/{output}"; '
                            f"touch {_DIR}/{name}.done"
                        )
                    },
                    "compute_config": {"num_nodes": 1},
                },
            }
        ],
    }


def _build_yaml(reemit: bool, export_fails_first: bool) -> dict:
    ckpt = {"uri": "env://{{ binding.path }}", "type": "model"}
    return {
        "granite.build": {
            "name": "gb664-checkpoint-consumer-reuse",
            "retries": {"max_retries": 1, "target_reuse_enabled": True},
            "targets": {
                "t": {
                    "allow_unknown": True,
                    "environment_uri": "space://environments/bash",
                    "outputs": {"checkpoint_1": ckpt, "checkpoint_2": dict(ckpt)},
                    "steps": [
                        {
                            "step_uri": "space://steps/command",
                            "config": {
                                "command_config": {"command": _trainer_command(reemit)},
                                "compute_config": {"num_nodes": 1},
                            },
                        }
                    ],
                },
                "e": _consumer("e", "t.checkpoint_1", "exported", export_fails_first),
                "f": _consumer("f", "e.exported", "evaluated", False),
            },
        }
    }


class _Outcome:
    """What happened to each target over the whole (two-attempt) build."""

    def __init__(self, runs: dict[str, list[Status]], execs: dict[str, int], skipped):
        self.runs = runs  # stored target-run statuses, by target name
        self.execs = execs  # how many times each target's command really ran
        self.skipped = skipped  # targets dispatched-and-reused on attempt 2

    def __repr__(self) -> str:
        return f"runs={self.runs} execs={self.execs} reused={self.skipped}"


@pytest.mark.xdist_group(name="buildrunner_bash_gb664_reuse")
class TestCheckpointConsumerReuseOnRetry(AbstractBuildTest):
    """What a checkpoint's consumers do when its producer is retried in place."""

    def setup_method(self, method):
        self.run_locally = True
        super().setup_method(method)

    def _run(self: Self, tmp_path: Path, reemit: bool, export_fails_first=False):
        build_yaml = tmp_path / "build.yaml"
        build_yaml.write_text(
            yaml.safe_dump(_build_yaml(reemit, export_fails_first), sort_keys=False)
        )
        spec = BuildTestSpecification(
            build_yaml=str(build_yaml),
            space_uri=_SPACE_URI,
            target_expectations=[],
            simulate_step_failure=False,
        )
        space = self._check_and_setup_space(spec)
        stored_build = StoredBuild.create(
            name="test-gb664-reuse",
            space_name=space.name,
            source_uri="",
            username=GBTEST_USER_NAME,
            build_yaml_path=spec.build_yaml,
            status=Status.PENDING,
        )
        build_id = stored_build.uuid
        self.storage.build_storage.add(stored_build)

        # Spy on the reuse path: which targets were dispatched and found reusable.
        skipped: list[str] = []
        original = BuildRun._BuildRun__handle_skipped_target

        async def spy(self_run, target, resolved_outputs, tg):
            skipped.append(target.name)
            return await original(self_run, target, resolved_outputs, tg)

        runner = BuildRunner(stored_build, space_uri=spec.space_uri, create_pr=False)
        with patch.object(BuildRun, "_BuildRun__handle_skipped_target", spy):
            thread = ExceptionRaisingThread(
                name="gb664 retry build", target=runner.start_and_wait, args=()
            )
            thread.start()
            try:
                # FAILED is transient: attempt 1 fails, the retry runs in place.
                self._wait_for_build_status(
                    build_id, [Status.SUCCESS], 300, transient_statuses=[Status.FAILED]
                )
            finally:
                thread.join(timeout=60)

        build = self.storage.build_storage.get_by_uuid(build_id)
        assert isinstance(build, StoredBuild)
        assert build.retry_count == 1, "expected exactly one in-place retry"
        runs: dict[str, list[Status]] = {"t": [], "e": [], "f": []}
        for r in self.storage.target_storage.get_by_where({"build_id": build_id}):
            assert isinstance(r, StoredTargetRun)
            runs[r.name].append(r.status)
        work = Path(f"/tmp/gb664_{build_id}")
        execs = {}
        for name in ("e", "f"):
            counter = work / f"{name}.runs"
            execs[name] = (
                len(counter.read_text().splitlines()) if counter.exists() else 0
            )
        return _Outcome(runs, execs, skipped)

    def test_a_reemit_reuses_export_and_its_eval(self, tmp_path):
        """(a) Same-path re-emit: e is dispatched, its hash matches, it is reused.

        Its stored outputs are re-propagated, so f is dispatched and reused too.
        The trainer's re-registration of the same checkpoint URI in the same
        build is accepted (the existing registration is reused), not refused.
        """
        out = self._run(tmp_path, reemit=True)
        assert sorted(out.runs["t"]) == sorted([Status.FAILED, Status.SUCCESS]), out
        assert sorted(out.skipped) == ["e", "f"], out
        assert out.execs == {"e": 1, "f": 1}, out
        assert out.runs["e"] == [Status.SUCCESS], out
        assert out.runs["f"] == [Status.SUCCESS], out

    def test_b_no_reemit_never_dispatches_export(self, tmp_path):
        """(b) No re-emit: e and f are never dispatched on attempt 2.

        They are neither re-run nor reused (the reuse check never runs for them),
        yet the build ends SUCCESS rather than hanging: the build loop exits once
        every dispatched task is done, never waiting on undelivered bindings.
        Harmless here only because e and f already succeeded on attempt 1.
        """
        out = self._run(tmp_path, reemit=False)
        assert sorted(out.runs["t"]) == sorted([Status.FAILED, Status.SUCCESS]), out
        assert out.skipped == [], out
        assert out.execs == {"e": 1, "f": 1}, out
        assert out.runs["e"] == [Status.SUCCESS], out
        assert out.runs["f"] == [Status.SUCCESS], out

    def test_a_reemit_reruns_export_that_failed_on_attempt_1(self, tmp_path):
        """(a) with an export that crashed on attempt 1: it is re-run and succeeds."""
        out = self._run(tmp_path, reemit=True, export_fails_first=True)
        assert sorted(out.runs["e"]) == sorted([Status.FAILED, Status.SUCCESS]), out
        assert out.runs["f"] == [Status.SUCCESS], out
        assert out.execs == {"e": 2, "f": 1}, out
        assert out.skipped == [], out

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "#664: without a re-emit, a consumer whose producer is re-run is never "
            "dispatched on the retry, so an export that failed on attempt 1 is "
            "silently left FAILED (and its eval never runs) while the build "
            "reports SUCCESS"
        ),
    )
    def test_b_no_reemit_strands_export_that_failed_on_attempt_1(self, tmp_path):
        """(b) with an export that crashed on attempt 1: it should be re-run."""
        out = self._run(tmp_path, reemit=False, export_fails_first=True)
        assert Status.SUCCESS in out.runs["e"], out
        assert out.runs["f"] == [Status.SUCCESS], out
