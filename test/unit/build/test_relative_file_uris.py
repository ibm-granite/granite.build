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

"""Tests for resolving relative ``file:`` ``environment_uri``/``step_uri`` in a build.

A relative path is looked up from the current directory first (as it always
was), then in the build's own directory (the copy of the build.yaml's directory),
so a build can ship its own environment or step next to its build.yaml.  These
pin that order, that a path found in neither place is left alone, and which URIs
count as relative.
"""

import os
from pathlib import Path
from typing import Optional

import pytest

from gbserver.build.build import _anchor_relative_file_uris, _relative_file_path
from gbserver.types.buildconfig import BuildConfig
from gbserver.types.constants import CODE_GBSERVER_BUILTINS_STEPS_GBSTEP_URI


def _write_env(env_dir: Path, name: str) -> None:
    """Write a minimal Bash ``environment.yaml`` named ``name`` into ``env_dir``."""
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "environment.yaml").write_text(f"name: {name}\ntype: Bash\n")


def _config(
    build_dir: Path, environment_uri: str, step_uri: Optional[str] = None
) -> BuildConfig:
    """Write and load a one-target build.yaml in ``build_dir``.

    Args:
        build_dir: Where to write build.yaml.
        environment_uri: The target's ``environment_uri``.
        step_uri: The one step's ``step_uri``; ``None`` writes no steps.
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    steps = f"\n        - step_uri: {step_uri}" if step_uri is not None else " []"
    (build_dir / "build.yaml").write_text(
        "granite.build:\n"
        "  name: probe\n"
        "  targets:\n"
        "    t:\n"
        f"      environment_uri: {environment_uri}\n"
        f"      steps:{steps}\n"
    )
    return BuildConfig.from_yaml(build_dir / "build.yaml")


def _write_step(step_dir: Path, name: str) -> None:
    """Write a minimal ``step.yaml`` named ``name`` into ``step_dir``."""
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "step.yaml").write_text(f"name: {name}\n")


@pytest.fixture
def in_dir(monkeypatch):
    """Return a function that changes the current directory for the test."""
    return monkeypatch.chdir


class TestRelativeFilePath:
    @pytest.mark.parametrize(
        "uri, expected",
        [
            ("file:environment", "environment"),
            ("environment", "environment"),
            ("file:sub/env", "sub/env"),
        ],
    )
    def test_relative_paths(self, uri, expected):
        assert _relative_file_path(uri) == expected

    @pytest.mark.parametrize(
        "uri",
        [
            "file:///abs/env",
            "/abs/env",
            "file://host/env",
            "space://environments/skypilot/slurm",
            "git+ssh://github.com/org/repo.git",
            "file:{{ space.variables.ENV }}",
            "",
            None,
        ],
    )
    def test_not_relative(self, uri):
        assert _relative_file_path(uri) is None


class TestAnchorRelativeEnvUris:
    def test_found_only_next_to_build_yaml_is_anchored(self, tmp_path, in_dir):
        """A path missing from the current dir resolves in the build dir."""
        build_dir = tmp_path / "build"
        _write_env(build_dir / "myenv", "from-build-dir")
        config = _config(build_dir, "file:myenv")
        in_dir(tmp_path)  # no ./myenv here

        _anchor_relative_file_uris(config, build_dir)

        expected = f"file://{(build_dir / 'myenv').resolve()}"
        assert config.targets["t"].environment_uri == expected

    def test_current_dir_wins_when_both_exist(self, tmp_path, in_dir):
        """The current dir is checked first, so existing builds keep their meaning."""
        build_dir, cwd = tmp_path / "build", tmp_path / "cwd"
        _write_env(build_dir / "myenv", "from-build-dir")
        _write_env(cwd / "myenv", "from-cwd")
        config = _config(build_dir, "file:myenv")
        in_dir(cwd)

        _anchor_relative_file_uris(config, build_dir)

        assert config.targets["t"].environment_uri == "file:myenv"

    def test_found_in_neither_is_unchanged(self, tmp_path, in_dir):
        """A missing path is left as written, so it fails the usual validation."""
        build_dir = tmp_path / "build"
        config = _config(build_dir, "file:missing")
        in_dir(tmp_path)

        _anchor_relative_file_uris(config, build_dir)

        assert config.targets["t"].environment_uri == "file:missing"

    def test_non_relative_uris_are_unchanged(self, tmp_path, in_dir):
        """``space://`` and absolute URIs are never rewritten."""
        build_dir = tmp_path / "build"
        _write_env(build_dir / "environments" / "x", "x")
        config = _config(build_dir, "space://environments/x")
        in_dir(tmp_path)

        _anchor_relative_file_uris(config, build_dir)

        assert config.targets["t"].environment_uri == "space://environments/x"

    def test_does_not_leave_the_build_dir(self, tmp_path, in_dir):
        """A symlink pointing outside the build dir is not followed out of it."""
        build_dir, outside = tmp_path / "build", tmp_path / "outside"
        _write_env(outside, "outside")
        config = _config(build_dir, "file:escape")
        os.symlink(outside, build_dir / "escape")
        in_dir(tmp_path)

        _anchor_relative_file_uris(config, build_dir)

        assert config.targets["t"].environment_uri == "file:escape"


class TestBuildDirDot:
    @pytest.mark.parametrize("uri", ["file:.", ".", "file:./"])
    def test_dot_always_means_the_build_dir(self, tmp_path, in_dir, uri):
        """``file:.`` names the build dir even though "." exists from any cwd."""
        build_dir = tmp_path / "build"
        _write_env(build_dir, "beside-build-yaml")
        config = _config(build_dir, uri)
        in_dir(tmp_path)  # "." exists here too, but is not the build

        _anchor_relative_file_uris(config, build_dir)

        assert config.targets["t"].environment_uri == f"file://{build_dir.resolve()}"

    def test_dot_slash_path_keeps_current_dir_first(self, tmp_path, in_dir):
        """Only a bare "." is special: ``file:./x`` follows the usual order."""
        build_dir, cwd = tmp_path / "build", tmp_path / "cwd"
        _write_env(build_dir / "x", "from-build-dir")
        _write_env(cwd / "x", "from-cwd")
        config = _config(build_dir, "file:./x")
        in_dir(cwd)

        _anchor_relative_file_uris(config, build_dir)

        assert config.targets["t"].environment_uri == "file:./x"


class TestAnchorRelativeStepUris:
    def test_found_only_next_to_build_yaml_is_anchored(self, tmp_path, in_dir):
        """A relative step path missing from the current dir resolves in the build dir."""
        build_dir = tmp_path / "build"
        _write_step(build_dir / "steps" / "hello", "hello")
        config = _config(build_dir, "space://environments/x", "file:steps/hello")
        in_dir(tmp_path)

        _anchor_relative_file_uris(config, build_dir)

        expected = f"file://{(build_dir / 'steps' / 'hello').resolve()}"
        assert config.targets["t"].steps[0].step_uri == expected

    def test_current_dir_wins_when_both_exist(self, tmp_path, in_dir):
        """The current dir is checked first, as for environment_uri."""
        build_dir, cwd = tmp_path / "build", tmp_path / "cwd"
        _write_step(build_dir / "steps" / "hello", "from-build-dir")
        _write_step(cwd / "steps" / "hello", "from-cwd")
        config = _config(build_dir, "space://environments/x", "file:steps/hello")
        in_dir(cwd)

        _anchor_relative_file_uris(config, build_dir)

        assert config.targets["t"].steps[0].step_uri == "file:steps/hello"

    def test_found_in_neither_is_unchanged(self, tmp_path, in_dir):
        """A missing step path is left as written."""
        build_dir = tmp_path / "build"
        config = _config(build_dir, "space://environments/x", "file:steps/missing")
        in_dir(tmp_path)

        _anchor_relative_file_uris(config, build_dir)

        assert config.targets["t"].steps[0].step_uri == "file:steps/missing"

    def test_space_and_default_step_uris_are_unchanged(self, tmp_path, in_dir):
        """``space://`` and the builtin gbstep default are never rewritten."""
        build_dir = tmp_path / "build"
        _write_step(build_dir / "steps" / "misc" / "byoc", "byoc")
        config = _config(build_dir, "space://environments/x", "space://steps/misc/byoc")
        empty = _config(tmp_path / "empty", "space://environments/x", '""')
        in_dir(tmp_path)

        _anchor_relative_file_uris(config, build_dir)
        _anchor_relative_file_uris(empty, tmp_path / "empty")

        assert config.targets["t"].steps[0].step_uri == "space://steps/misc/byoc"
        assert (
            empty.targets["t"].steps[0].step_uri
            == CODE_GBSERVER_BUILTINS_STEPS_GBSTEP_URI
        )
