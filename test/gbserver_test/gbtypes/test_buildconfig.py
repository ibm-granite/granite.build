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

"""Tests for types related to the build.yaml"""

from pathlib import Path
from typing import List, Self

import pytest

from gbserver.types.buildconfig import BuildConfig, BuildTargetConfig


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


def get_expected_buildconfig(matched_base_key: str = "llm.build") -> BuildConfig:
    build_config = BuildConfig(
        matched_base_key=matched_base_key,
        targets={
            "foo": BuildTargetConfig(
                environment_uri="space://environments/vela1-gb",
                steps=[],
            ),
            "bar": BuildTargetConfig(
                environment_uri="space://environments/vela1-gb",
                steps=[],
            ),
        },
    )
    return build_config


class TestBuildConfig:
    """Test the BuildConfig class."""

    def test_build_yaml_with_new_basekey(self: Self, test_data_dir: Path) -> None:
        """A build.yaml with the new base key llm.build"""
        build_config_path = test_data_dir / "build-with-llm-build-base-key.yaml"
        assert build_config_path.is_file()
        build_config = BuildConfig.from_yaml(build_config_path)
        expected = get_expected_buildconfig(matched_base_key="llm.build")
        assert build_config == expected

    def test_build_yaml_with_old_basekey(self: Self, test_data_dir: Path) -> None:
        """A build.yaml with the old base key granite.build"""
        build_config_path = test_data_dir / "build-with-granite-build-base-key.yaml"
        assert build_config_path.is_file()
        build_config = BuildConfig.from_yaml(build_config_path)
        expected = get_expected_buildconfig(matched_base_key="granite.build")
        assert build_config == expected
