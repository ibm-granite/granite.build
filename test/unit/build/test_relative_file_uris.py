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

import logging
import os
from pathlib import Path
from typing import Optional

import pytest

from gbserver.build.build import (
    Build,
    _anchor_relative_file_uris,
    _relative_file_path,
    unanchor_file_uri,
)
from gbserver.build.buildrun import BuildRun
from gbserver.build.space import Space
from gbserver.types.buildconfig import BuildConfig
from gbserver.types.constants import CODE_GBSERVER_BUILTINS_STEPS_GBSTEP_URI
from gbserver.utils.archive import create_archive_bytes, extract_archive


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

    def test_warns_when_the_build_dir_copy_is_shadowed(self, tmp_path, in_dir, caplog):
        """Both exist: say which one was used, so an ignored shipped env is visible."""
        build_dir, cwd = tmp_path / "build", tmp_path / "cwd"
        _write_env(build_dir / "myenv", "from-build-dir")
        _write_env(cwd / "myenv", "from-cwd")
        config = _config(build_dir, "file:myenv")
        in_dir(cwd)

        with caplog.at_level(logging.WARNING):
            _anchor_relative_file_uris(config, build_dir)

        assert "exists both in the current directory" in caplog.text
        assert str((build_dir / "myenv").resolve()) in caplog.text

    def test_no_warning_when_only_the_current_dir_has_it(
        self, tmp_path, in_dir, caplog
    ):
        build_dir, cwd = tmp_path / "build", tmp_path / "cwd"
        _write_env(cwd / "myenv", "from-cwd")
        config = _config(build_dir, "file:myenv")
        in_dir(cwd)

        with caplog.at_level(logging.WARNING):
            _anchor_relative_file_uris(config, build_dir)

        assert "exists both" not in caplog.text

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


# The repo root (this file is test/unit/build/<file>).
_REPO = Path(__file__).resolve().parents[3]


def _archive_and_extract(fixture: Path, dest: Path) -> Path:
    """Round-trip ``fixture`` through a build archive, as a stored build does.

    A stored build zips its build.yaml's directory and the runner extracts it
    elsewhere, so sibling files reach the build only via that copy.

    Args:
        fixture: The folder holding build.yaml and its sibling files.
        dest: Where to extract.

    Returns:
        The extracted build directory.
    """
    data = create_archive_bytes(dir=fixture, format="zip")
    dest.mkdir(parents=True)
    assert extract_archive(data, dest)
    return dest


class TestUnanchorFileUri:
    @pytest.mark.parametrize(
        "rel, expected",
        [(".", "file:."), ("environments/e", "file:environments/e")],
    )
    def test_inside_the_build_dir_is_made_relative(self, tmp_path, rel, expected):
        anchored = f"file://{(tmp_path / rel).resolve()}"
        assert unanchor_file_uri(anchored, tmp_path) == expected

    @pytest.mark.parametrize(
        "uri",
        ["file:///elsewhere/env", "space://environments/e", "file:env", None, ""],
    )
    def test_others_are_unchanged(self, tmp_path, uri):
        assert unanchor_file_uri(uri, tmp_path) == uri

    def test_no_build_dir_is_unchanged(self, tmp_path):
        uri = f"file://{tmp_path}"
        assert unanchor_file_uri(uri, None) == uri

    def test_round_trips_an_anchored_step(self, tmp_path, in_dir):
        build_dir = tmp_path / "build"
        _write_step(build_dir / "steps" / "s", "s")
        config = _config(build_dir, "space://environments/e", "file:steps/s")
        in_dir(tmp_path)
        _anchor_relative_file_uris(config, build_dir)
        step_uri = config.targets["t"].steps[0].step_uri
        assert unanchor_file_uri(step_uri, build_dir) == "file:steps/s"


def _write_env_beside_build_yaml(fixture: Path) -> None:
    """Write a build.yaml using ``environment_uri: file:.`` and its environment.yaml.

    Args:
        fixture: The folder to write both files into.
    """
    fixture.mkdir()
    (fixture / "environment.yaml").write_text(
        "name: shipped-with-the-build\ntype: Bash\nconfig: {}\n"
    )
    (fixture / "build.yaml").write_text(
        "granite.build:\n"
        "  name: env-beside-build-yaml\n"
        "  targets:\n"
        "    t:\n"
        "      environment_uri: file:.\n"
        "      steps:\n"
        "        - step_uri: space://steps/command\n"
        "          config:\n"
        "            command_config:\n"
        "              command: echo hi\n"
    )


class TestBuildWithEnvBesideBuildYaml:
    def test_target_hash_is_the_same_across_builds(self, tmp_path, in_dir, monkeypatch):
        """Each build has its own workspace, so the anchored env path differs.

        The target definition hash decides whether a target already ran, so it
        must not change with that path, or the target re-runs every time.
        """
        monkeypatch.setenv("GB_ENVIRONMENT", "STANDALONE")
        fixture = tmp_path / "fixture"
        _write_env_beside_build_yaml(fixture)
        in_dir(tmp_path)
        space = Space(f"file://{_REPO / 'configurations' / 'spaces' / 'local'}")

        hashes, env_uris = [], []
        for name in ("w1", "w2"):
            build = Build(build_dir=fixture, space=space, workspace_dir=tmp_path / name)
            env_uris.append(build.config.targets["t"].environment_uri)
            run = BuildRun(build)
            hashes.append(run._BuildRun__compute_target_def_hash(build.targets["t"]))

        assert env_uris[0] != env_uris[1]
        assert hashes[0] == hashes[1]

    def test_file_dot_loads_the_env_shipped_with_the_build(
        self, tmp_path, in_dir, monkeypatch
    ):
        """A real Build whose environment.yaml sits next to its build.yaml.

        Exercises ``environment_uri: file:.`` end to end: archive, extract,
        Build creation, validation and environment loading, run from an
        unrelated directory, so a regression in the build-dir lookup fails here
        rather than only on a live cluster.
        """
        monkeypatch.setenv("GB_ENVIRONMENT", "STANDALONE")
        fixture = tmp_path / "fixture"
        _write_env_beside_build_yaml(fixture)
        build_dir = _archive_and_extract(fixture, tmp_path / "extracted" / "b1")
        in_dir(tmp_path)  # neither the fixture nor the extracted build
        space = Space(f"file://{_REPO / 'configurations' / 'spaces' / 'local'}")

        build = Build(build_dir=build_dir, space=space)

        target = build.config.targets["t"]
        assert target.environment_uri.startswith("file://")
        assert not build.validate().errors
        assert build.targets["t"].environment.config.name == "shipped-with-the-build"
