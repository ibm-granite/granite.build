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
from gbserver.types.stepconfig import (
    StepEnvironmentTypeConfig,
    StepLauncherConfig,
    StepMonitorConfig,
)

pytestmark = [pytest.mark.g4os, pytest.mark.unit]

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
