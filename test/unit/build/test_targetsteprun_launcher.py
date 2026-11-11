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

"""Tests that TargetStepRun respects the launcher override set by TargetStep."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gbserver.build.targetsteprun import TargetStepRun
from gbserver.types.buildevent import EntityRunMetadata
from gbserver.types.status import Status
from gbserver.types.stepconfig import (
    StepEnvironmentTypeConfig,
    StepLauncherConfig,
    StepMonitorConfig,
)

_GPU_IMAGE = "gbserver-test-trl-unitxt:gpu"
_CPU_IMAGE = "gbserver-test-trl-unitxt:cpu"

_FAKE_METADATA = EntityRunMetadata(
    build_id="test-build-id",
    username="test-user",
    type="TargetStep",
    target_name="finetune",
    targetrun_id="test-targetrun-id",
    targetsteprun_id="test-id",
    targetstep_uri="space://steps/trl-finetune",
    target_step_index=0,
)


def _make_env_config() -> StepEnvironmentTypeConfig:
    """Step environment with two Docker launchers: gpu (default by sort) and cpu (override)."""
    return StepEnvironmentTypeConfig(
        default_launcher=None,
        launchers={
            "trl-finetune": StepLauncherConfig(
                type="docker",
                monitors=["docker_log"],
                config={"image": _GPU_IMAGE, "command": "python3 run.py"},
            ),
            "trl-finetune-cpu": StepLauncherConfig(
                type="docker",
                monitors=["docker_log"],
                config={"image": _CPU_IMAGE, "command": "python3 run.py"},
            ),
        },
        monitors={
            "docker_log": StepMonitorConfig(type="docker_log", config={}),
        },
    )


def _make_targetstep_mock(launcher_name: str, tmp_path: Path) -> MagicMock:
    """Fake TargetStep with launcher_name already resolved (as targetstep.py would set it).

    Only the keys touched by the launcher-selection path in TargetStepRun.__init__ are
    set explicitly; everything else is left as MagicMock auto-attributes.
    """
    ts = MagicMock()
    ts.build_id = "test-build-id"
    ts.build_workspace_dir = tmp_path
    ts.full_config = {"environment_config": {}, "step": {"name": "trl-finetune"}}
    ts.is_step_file_exists = False
    ts.step_environment_config = _make_env_config()
    ts.env_type = "Docker"
    ts.launcher_name = launcher_name
    ts.merged_step_dir = tmp_path
    return ts


class TestTargetStepRunLauncherOverride:
    def test_build_yaml_launcher_override_is_respected(self, tmp_path):
        """When build YAML specifies launcher: trl-finetune-cpu, the cpu image must be used.

        Regression test for: targetsteprun.py re-derived launcher from sorted(launchers)[0]
        (picking 'trl-finetune' / :gpu), ignoring targetstep.launcher_name set from build YAML.
        """
        targetstep = _make_targetstep_mock("trl-finetune-cpu", tmp_path)

        with patch.object(
            TargetStepRun, "get_runmetadata", return_value=_FAKE_METADATA
        ):
            TargetStepRun(
                target=MagicMock(),
                targetstep=targetstep,
                targetrun_id="test-targetrun-id",
                event_q=asyncio.Queue(),
            )

        assert targetstep.launcher.config["image"] == _CPU_IMAGE, (
            f"Expected cpu image '{_CPU_IMAGE}' but got '{targetstep.launcher.config['image']}'. "
            "targetsteprun.py is ignoring the launcher_name set by targetstep.py."
        )

    def test_default_launcher_used_when_no_override(self, tmp_path):
        """When no launcher override is specified, the first sorted launcher is used."""
        targetstep = _make_targetstep_mock("", tmp_path)

        with patch.object(
            TargetStepRun, "get_runmetadata", return_value=_FAKE_METADATA
        ):
            TargetStepRun(
                target=MagicMock(),
                targetstep=targetstep,
                targetrun_id="test-targetrun-id",
                event_q=asyncio.Queue(),
            )

        # 'trl-finetune' sorts before 'trl-finetune-cpu', so gpu image is the fallback
        assert targetstep.launcher.config["image"] == _GPU_IMAGE


class TestTargetStepRunValidatesLaunchConfig:
    """Step-declared launch constraints (single_node) are checked against the
    filled configs at step creation, so a bad request fails "on creation"
    before anything is launched."""

    @staticmethod
    def _create(targetstep, event_q=None):
        with patch.object(
            TargetStepRun, "get_runmetadata", return_value=_FAKE_METADATA
        ):
            return TargetStepRun(
                target=MagicMock(),
                targetstep=targetstep,
                targetrun_id="test-targetrun-id",
                event_q=event_q or asyncio.Queue(),
            )

    def test_hook_gets_the_filled_launcher_config_and_config(self, tmp_path):
        targetstep = _make_targetstep_mock("trl-finetune-cpu", tmp_path)
        self._create(targetstep)

        hook = targetstep.environment.validate_launch_config
        hook.assert_called_once()
        step_env_config, launcher_config, config = hook.call_args.args
        assert step_env_config is targetstep.step_environment_config
        assert launcher_config == {"image": _CPU_IMAGE, "command": "python3 run.py"}
        assert isinstance(config, dict)

    def test_hook_error_fails_step_creation(self, tmp_path):
        targetstep = _make_targetstep_mock("trl-finetune-cpu", tmp_path)
        targetstep.environment.validate_launch_config.side_effect = ValueError(
            "num_nodes resolved to 2"
        )
        event_q = asyncio.Queue()

        with pytest.raises(ValueError, match="failed on creation") as exc:
            self._create(targetstep, event_q)

        assert "num_nodes resolved to 2" in str(exc.value.__cause__)
        events = []
        while not event_q.empty():
            events.append(event_q.get_nowait())
        assert any(
            getattr(e.payload, "status", None) == Status.FAILED for e in events
        ), events


class TestTargetStepRunMonitorConfigTemplating:
    """The monitor config passed to environment.monitor must have its Jinja
    templates rendered. Regression for: _run passed the raw (unrendered)
    step_environment_config.monitors[...] config, so values like
    log_retrieval.mode reached the monitor as literal '{{ config.* }}' strings
    (which made the skypilot monitor fall back to on_completion and never emit
    the rm-server URL binding)."""

    @pytest.mark.asyncio
    async def test_monitor_config_is_rendered_before_dispatch(self):
        from gbserver.build.targetsteprun import TargetStepRun

        # Build a TargetStepRun without running __init__'s config pipeline.
        tsr = TargetStepRun.__new__(TargetStepRun)
        tsr.full_config = {"config": {}}  # no overrides -> Jinja defaults apply
        tsr.build_id = "b1"
        tsr.event_q = asyncio.Queue()
        tsr.dir = "/tmp/does-not-matter"

        # target.setup_ids.keys() is iterated by _run
        target = MagicMock()
        target.setup_ids = {}
        tsr.target = target

        # Capture kwargs handed to environment.monitor.
        captured = {}

        def _monitor(**kwargs):
            captured.update(kwargs)
            fut = asyncio.get_event_loop().create_future()
            fut.set_result(None)
            return fut

        launch_task = asyncio.get_event_loop().create_future()
        launch_task.launch_id = "launch-123"
        launch_task.set_result(None)

        environment = MagicMock()
        environment.launch = MagicMock(return_value=launch_task)
        environment.monitor = MagicMock(side_effect=_monitor)

        # Templated monitor config (as stored, unrendered).
        env_config = StepEnvironmentTypeConfig(
            default_launcher="svc",
            launchers={
                "svc": StepLauncherConfig(
                    type="skypilot", monitors=["skypilot_monitor"], config={}
                )
            },
            monitors={
                "skypilot_monitor": StepMonitorConfig(
                    type="skypilot_monitor",
                    config={
                        "log_retrieval": {
                            "mode": "{{ config.log_retrieval_mode "
                            "| default('startup_window') }}",
                            "interval_seconds": "{{ config.log_retrieval_interval_seconds "
                            "| default(15) }}",
                        }
                    },
                )
            },
        )

        entity = MagicMock(
            spec_set=["config", "environment", "launcher", "step_environment_config"]
        )
        entity.config = None
        entity.environment = environment
        entity.launcher = StepLauncherConfig(
            type="skypilot", monitors=["skypilot_monitor"], config={}
        )
        entity.step_environment_config = env_config
        tsr.entity = entity

        with (
            patch.object(TargetStepRun, "get_runmetadata", return_value=_FAKE_METADATA),
            patch("gbserver.build.targetsteprun.TargetStep", MagicMock),
        ):
            await tsr._run()

        assert (
            captured.get("log_retrieval", {}).get("mode") == "startup_window"
        ), f"monitor received unrendered config: {captured.get('log_retrieval')}"
        assert captured["log_retrieval"]["interval_seconds"] == "15"
