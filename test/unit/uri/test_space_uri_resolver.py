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

"""Unit tests for the ``space://`` URI resolver in ``gbcommon.uri.space``.

These exercise the resolution *behavior* the build pipeline hinges on:

* the 3-tier ordering in ``SpaceURI.__new__`` (env-co-located → env-class
  match → env-agnostic fallback → ValueError);
* ``_try_env_class_match`` specificity ordering and lexicographic tie-break;
* ``with_current_env`` / ``with_current_env_class_name`` thread-local
  save/restore semantics;
* relative-base-uri resolution in ``gbserver.build.space._resolve_base_uris``
  /``_resolve_one_base_uri``, including the non-local ``ValueError`` path.

Importing ``gbcommon.uri.space`` triggers ``gbcommon.uri``'s package init,
which registers the ``space``/``file`` URI handlers used here.
"""

from pathlib import Path
from typing import Iterable, Optional

import pytest
import yaml

from gbcommon.uri.space import SpaceURI
from gbcommon.uri.uri import URI
from gbserver.build.space import (
    _resolve_base_uris,
    _resolve_one_base_uri,
    _space_dir_from_uri,
)

# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

_THREAD_LOCAL_ATTRS = (
    "base_uris",
    "space_secrets",
    "current_env_dir_uri",
    "current_env_class_name",
)


@pytest.fixture(autouse=True)
def _isolate_thread_local():
    """Snapshot and restore ``SpaceURI._thread_local`` around each test.

    The resolver keeps base_uris/secrets and the active-env context on a
    thread-local; without isolation, state would leak between tests (and
    between these tests and the rest of the suite running in-thread).
    """
    tl = SpaceURI._thread_local
    saved = {a: getattr(tl, a, None) for a in _THREAD_LOCAL_ATTRS}
    try:
        yield
    finally:
        for attr, value in saved.items():
            if value is None:
                if hasattr(tl, attr):
                    delattr(tl, attr)
            else:
                setattr(tl, attr, value)


def _set_bases(*dirs: Path) -> None:
    """Point the resolver's base_uris at the given local directories."""
    SpaceURI.set_baseuris([f"file://{d}" for d in dirs], {})


def _write_step(step_dir: Path, env_classes: Optional[Iterable[str]] = None) -> Path:
    """Create ``<step_dir>/step.yaml`` (and parents) and return ``step_dir``.

    Args:
        step_dir: Directory that will hold the ``step.yaml`` (its basename is
            the step name).
        env_classes: When provided, written as ``environment_configs`` keys so
            the env-class-match tier can select the file; ``None`` omits the
            key entirely.
    """
    step_dir.mkdir(parents=True, exist_ok=True)
    data: dict = {"name": step_dir.name, "version": "v1", "type": "custom"}
    if env_classes is not None:
        data["environment_configs"] = {cls: {} for cls in env_classes}
    (step_dir / "step.yaml").write_text(yaml.safe_dump(data))
    return step_dir


def _make_env(class_name: str, env_dir: Optional[Path] = None):
    """Build a stand-in environment whose class name and ``environment_dir_uri``
    drive ``with_current_env``."""
    env_dir_uri = f"file://{env_dir}" if env_dir is not None else None
    return type(class_name, (), {"environment_dir_uri": env_dir_uri})()


def _resolve(uri: str) -> URI:
    """Resolve a ``space://`` URI through the real ``SpaceURI`` resolver."""
    return URI.get_uri(uri, default_scheme="file", secrets={})


def _resolved_dir(uri: URI) -> Path:
    """Filesystem path the resolver landed on."""
    return Path(uri.uri.path)  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Tier 1 — env-co-located step lookup
# --------------------------------------------------------------------------- #


class TestTier1EnvColocated:
    def test_env_colocated_hit_wins_over_base(self, tmp_path):
        """A step inside the active env's own dir wins over a same-named step
        reachable via base_uris."""
        base = _write_step(tmp_path / "base" / "steps" / "hello").parent.parent
        env_dir = tmp_path / "envs" / "bash"
        colocated = _write_step(env_dir / "steps" / "hello")
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("Bash", env_dir)):
            resolved = _resolve("space://steps/hello")

        assert _resolved_dir(resolved).samefile(colocated)

    def test_tier1_miss_falls_through_to_class_match(self, tmp_path):
        """When the env dir lacks the step, resolution falls through to the
        env-class-match tier (proving ordering, not just tier-1)."""
        base = tmp_path / "base"
        class_match = _write_step(base / "k8s" / "digit", env_classes=["K8s"])
        env_dir = tmp_path / "envs" / "k8s"  # exists, but has no steps/digit
        env_dir.mkdir(parents=True)
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("K8s", env_dir)):
            resolved = _resolve("space://steps/digit")

        assert _resolved_dir(resolved).samefile(class_match)


# --------------------------------------------------------------------------- #
# Tier 1.5 — env-class match (specificity + tie-break)
# --------------------------------------------------------------------------- #


class TestTier15EnvClassMatch:
    def test_single_env_file_beats_multi_env_catchall(self, tmp_path):
        """A single-env split file (fewer environment_configs keys) beats a
        multi-env catch-all that also lists the active class."""
        base = tmp_path / "base"
        _write_step(base / "s3push", env_classes=["K8s", "Lsf", "Skypilot"])
        specific = _write_step(base / "k8s" / "s3push", env_classes=["K8s"])
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("K8s"):
            resolved = _resolve("space://steps/s3push")

        assert _resolved_dir(resolved).samefile(specific)

    def test_equal_specificity_lexicographic_tiebreak(self, tmp_path):
        """Among equally-specific matches, the lexicographically smaller path
        wins (deterministic tie-break)."""
        base = tmp_path / "base"
        first = _write_step(base / "aaa" / "dup", env_classes=["K8s"])
        _write_step(base / "bbb" / "dup", env_classes=["K8s"])
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("K8s"):
            resolved = _resolve("space://steps/dup")

        assert _resolved_dir(resolved).samefile(first)

    def test_no_class_match_when_class_absent(self, tmp_path):
        """A candidate that does not list the active env class is ignored;
        with no other tier matching, resolution raises."""
        base = tmp_path / "base"
        _write_step(base / "skypilot" / "only", env_classes=["Skypilot"])
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("K8s"):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/only")

    def test_subasset_uri_appends_rest_to_matched_dir(self, tmp_path):
        """`space://steps/<name>/<rest>` resolves against the matched step dir
        plus the `<rest>` suffix."""
        base = tmp_path / "base"
        step_dir = _write_step(base / "k8s" / "digit", env_classes=["K8s"])
        sub = step_dir / "helm-charts"
        sub.mkdir()
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("K8s"):
            resolved = _resolve("space://steps/digit/helm-charts")

        assert _resolved_dir(resolved).samefile(sub)


# --------------------------------------------------------------------------- #
# Tier 2 — env-agnostic fallback + unresolvable
# --------------------------------------------------------------------------- #


class TestTier2Fallback:
    def test_fallback_resolves_against_base(self, tmp_path):
        """With no active env, a `space://steps/<name>` resolves via the
        plain base_uris fallback."""
        base = tmp_path / "base"
        step_dir = _write_step(base / "steps" / "hello")
        _set_bases(base)

        resolved = _resolve("space://steps/hello")

        assert _resolved_dir(resolved).samefile(step_dir)

    def test_fallback_scans_base_uris_in_order(self, tmp_path):
        """The first base_uri lacking the path is skipped; a later one that
        has it resolves."""
        first = tmp_path / "first"
        first.mkdir()
        second = tmp_path / "second"
        target = _write_step(second / "steps" / "hello")
        _set_bases(first, second)

        resolved = _resolve("space://steps/hello")

        assert _resolved_dir(resolved).samefile(target)

    def test_non_step_uri_uses_fallback_only(self, tmp_path):
        """Tiers 1/1.5 apply only to `steps/`; an environments URI resolves
        purely via the base_uris fallback."""
        base = tmp_path / "base"
        env_dir = base / "environments" / "bash"
        env_dir.mkdir(parents=True)
        _set_bases(base)

        resolved = _resolve("space://environments/bash")

        assert _resolved_dir(resolved).samefile(env_dir)

    def test_unresolvable_raises(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        _set_bases(base)

        with pytest.raises(ValueError, match="Unresolvable space uri"):
            _resolve("space://steps/missing")


# --------------------------------------------------------------------------- #
# Thread-local save/restore
# --------------------------------------------------------------------------- #


class TestThreadLocalScoping:
    def test_class_name_set_and_cleared(self):
        tl = SpaceURI._thread_local
        assert getattr(tl, "current_env_class_name", None) is None
        with SpaceURI.with_current_env_class_name("K8s"):
            assert tl.current_env_class_name == "K8s"
        assert getattr(tl, "current_env_class_name", None) is None

    def test_class_name_nested_restore(self):
        tl = SpaceURI._thread_local
        with SpaceURI.with_current_env_class_name("K8s"):
            with SpaceURI.with_current_env_class_name("Lsf"):
                assert tl.current_env_class_name == "Lsf"
            # Inner exit restores the outer value, not "no value".
            assert tl.current_env_class_name == "K8s"
        assert getattr(tl, "current_env_class_name", None) is None

    def test_empty_class_name_is_none(self):
        tl = SpaceURI._thread_local
        with SpaceURI.with_current_env_class_name(""):
            assert getattr(tl, "current_env_class_name", None) is None

    def test_with_current_env_sets_and_restores_both_fields(self, tmp_path):
        tl = SpaceURI._thread_local
        env = _make_env("Docker", tmp_path / "envs" / "docker")
        with SpaceURI.with_current_env(env):
            assert tl.current_env_class_name == "Docker"
            assert tl.current_env_dir_uri == f"file://{tmp_path / 'envs' / 'docker'}"
        assert getattr(tl, "current_env_dir_uri", None) is None
        assert getattr(tl, "current_env_class_name", None) is None

    def test_with_current_env_nested_restore(self, tmp_path):
        tl = SpaceURI._thread_local
        outer = _make_env("Bash", tmp_path / "bash")
        inner = _make_env("K8s", tmp_path / "k8s")
        with SpaceURI.with_current_env(outer):
            with SpaceURI.with_current_env(inner):
                assert tl.current_env_class_name == "K8s"
            assert tl.current_env_class_name == "Bash"
            assert tl.current_env_dir_uri == f"file://{tmp_path / 'bash'}"
        assert getattr(tl, "current_env_class_name", None) is None


# --------------------------------------------------------------------------- #
# Relative base_uri resolution (_resolve_base_uris / _resolve_one_base_uri)
# --------------------------------------------------------------------------- #


class TestResolveBaseUris:
    def test_relative_bare_path_against_file_space(self, tmp_path):
        space_uri = f"file://{tmp_path}"
        result = _resolve_base_uris(["../assets"], space_uri)
        expected = (Path(str(tmp_path)) / "../assets").resolve()
        assert result == [f"file://{expected}"]

    def test_relative_file_uri_against_file_space(self, tmp_path):
        space_uri = f"file://{tmp_path}"
        # urlparse("file://sub/dir") -> netloc="sub", path="/dir" -> "sub/dir"
        resolved = _resolve_one_base_uri("file://sub/dir", tmp_path, space_uri)
        expected = (Path(str(tmp_path)) / "sub/dir").resolve()
        assert resolved == f"file://{expected}"

    def test_absolute_file_uri_passes_through(self, tmp_path):
        out = _resolve_one_base_uri(
            "file:///abs/assets", tmp_path, f"file://{tmp_path}"
        )
        assert out == "file:///abs/assets"

    def test_non_file_scheme_passes_through(self):
        git_uri = "git://github.ibm.com/org/repo"
        assert _resolve_one_base_uri(git_uri, None, git_uri) == git_uri

    def test_space_dir_none_for_non_local_uri(self):
        assert _space_dir_from_uri("git://github.ibm.com/org/repo") is None

    def test_relative_base_with_nonlocal_space_raises(self):
        """The non-local ValueError path: a relative base_uri has no anchor
        when the space URI is not a local file:// URI."""
        space_uri = "git://github.ibm.com/org/repo"
        with pytest.raises(ValueError, match="Cannot resolve relative base_uri"):
            _resolve_base_uris(["../assets"], space_uri)

    def test_relative_file_uri_with_nonlocal_space_raises(self):
        space_uri = "git://github.ibm.com/org/repo"
        with pytest.raises(ValueError) as exc:
            _resolve_one_base_uri("file://rel/path", None, space_uri)
        # Error names both the offending entry and the space URI.
        assert "file://rel/path" in str(exc.value)
        assert space_uri in str(exc.value)
