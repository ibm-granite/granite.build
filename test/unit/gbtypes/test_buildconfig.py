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


class TestOutputPublicField:
    """The generic `public` field on an output.

    `buildconfig` stays store-agnostic: it holds `public` as a plain optional
    bool and does not interpret it. The three-form fold, the public→private flip,
    and the HF-only guard all live with the HF push path
    (`gbserver.spaces.hf_push_config`) and are tested there.
    """

    def test_public_is_stored_verbatim(self):
        from gbserver.types.buildconfig import BuildTargetOutputConfig

        o = BuildTargetOutputConfig(uri="hf:///org/repo", public=True)
        assert o.public is True
        # No parse-time mutation into store_push: buildconfig doesn't interpret it.
        assert o.store_push is None

    def test_public_defaults_to_none(self):
        from gbserver.types.buildconfig import BuildTargetOutputConfig

        assert BuildTargetOutputConfig(uri="hf:///org/repo").public is None


class TestOutputPushHfOnlyGuard:
    """`public` / `store_push.config.hf.*` are HuggingFace-only push options.

    End-to-end through `BuildConfig.my_validate`, which delegates to the HF push
    path's validator. Fold/flip/conflict semantics are tested in
    `test/unit/spaces/test_hf_push_config.py`.
    """

    @staticmethod
    def _build(uri, output_extra):
        return BuildConfig.model_validate(
            {
                "targets": {
                    "t1": {
                        "environment_uri": "space://env/default",
                        "outputs": {"out": {"uri": uri, **output_extra}},
                        "steps": [{"step_uri": ""}],
                    }
                }
            }
        )

    def _push_errors(self, uri, output_extra):
        errs = self._build(uri, output_extra).my_validate()
        return [e for e in errs.errors if "HuggingFace push option" in e.error]

    @pytest.mark.parametrize("uri", ["file:///tmp/x", "lh://a/b", "env:///abs/p"])
    def test_public_on_non_hf_output_errors(self, uri):
        assert len(self._push_errors(uri, {"public": True})) == 1

    @pytest.mark.parametrize("uri", ["lh://a/b", "cos://bucket/key"])
    def test_hf_block_on_non_hf_output_errors(self, uri):
        errs = self._push_errors(
            uri, {"store_push": {"config": {"hf": {"resource_group_name": "x"}}}}
        )
        assert len(errs) == 1

    def test_public_on_templated_hf_output_is_allowed(self):
        # The uri is a Jinja template at load time — the guard checks the prefix.
        assert (
            self._push_errors("hf:///org/repo-{{ binding.path }}", {"public": True})
            == []
        )

    def test_public_on_plain_hf_output_is_allowed(self):
        assert (
            self._push_errors("hf://huggingface.co/datasets/o/r", {"public": True})
            == []
        )
