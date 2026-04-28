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

"""Tests for types related to the step.yaml"""

from pathlib import Path
from typing import List, Self

import pytest

from gbserver.types.stepconfig import (
    StepConfig,
    StepEnvironmentTypeConfig,
    StepLauncherConfig,
    StepMonitorConfig,
)


@pytest.fixture
def test_data_dir() -> Path:
    src_file_dir = Path(__file__).resolve().parent
    assert src_file_dir.is_dir()
    # print("src_file_dir.parts", src_file_dir.parts)
    path_paths: List[str] = []
    test_done = False
    # start from the end and replace
    for x in src_file_dir.parts[::-1]:
        if not test_done and x == "test":
            test_done = True
            path_paths.append("test-data")
            continue
        path_paths.append(x)
    test_data_dir = Path(*path_paths[::-1])
    assert test_data_dir.is_dir()
    return test_data_dir


def get_expected_step() -> StepConfig:
    step_env_type_config = StepEnvironmentTypeConfig(
        default_launcher=None,
        setups={},
        launchers={
            "tuning": StepLauncherConfig(
                type="helm",
                setups=[],
                monitors=["log_monitor"],
                validators=[],
                config={"chart": "helm-charts/data_preprocessor"},
            ),
        },
        validators={},
        monitors={
            "log_monitor": StepMonitorConfig(
                type="sidecar_monitor",
                config={
                    "event_configs": [
                        {
                            "event_type": "newartifact_in_environment_event",
                            "line_regex": "Processed\\sData:\\s.*",
                            "is_json": False,
                            "event_fields": [
                                {
                                    "field_name": "binding_id",
                                    "field_value_template": "processed_dataset",
                                },
                                {
                                    "field_name": "path",
                                    "field_regex": "/.*",
                                    "is_data": True,
                                },
                                {
                                    "field_name": "binding",
                                    "field_value_template": '{ "path": "{{ fields.data.path }}" }',
                                    "is_json": True,
                                },
                            ],
                        },
                    ],
                },
            ),
        },
    )
    step_config = StepConfig(
        name="data_preprocessor",
        type="data_preprocessor",
        config={},
        environment_configs={"K8s": step_env_type_config},
    )
    return step_config


class TestStepConfig:
    """Test the StepConfig class."""

    def test_step_yaml_config_dict(self: Self, test_data_dir: Path) -> None:
        """A step.yaml with config: {}"""
        step_config_path = test_data_dir / "step-config-dict.yaml"
        assert step_config_path.is_file()
        step_config = StepConfig.from_yaml(step_config_path)
        expected = get_expected_step()
        assert step_config == expected

    def test_step_yaml_config_none(self: Self, test_data_dir: Path) -> None:
        """A step.yaml with config: None"""
        step_config_path = test_data_dir / "step-config-none.yaml"
        assert step_config_path.is_file()
        step_config = StepConfig.from_yaml(step_config_path)
        expected = get_expected_step()
        expected.config = None
        assert step_config == expected
