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

* base_uris order between env-matching copies (``TestBaseOrder``); within one
  base the space root, then the env-class match (deepest owning folder, then
  specificity, then path); then the env-agnostic fallback → ValueError;
* env-folder ownership: which ``environments/<...>/steps/`` folders serve an env;
* ``with_current_env`` / ``with_current_env_class_name`` thread-local
  save/restore semantics;
* relative-base-uri resolution in ``gbserver.build.space._resolve_base_uris``
  /``_resolve_one_base_uri``, including the non-local ``ValueError`` path.

Importing ``gbcommon.uri.space`` triggers ``gbcommon.uri``'s package init,
which registers the ``space``/``file`` URI handlers used here.
"""

import shutil
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
    "current_env_subtype",
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


def _write_step(
    step_dir: Path,
    env_classes: Optional[Iterable[str]] = None,
    subtypes: Optional[Iterable[str]] = None,
) -> Path:
    """Create ``<step_dir>/step.yaml`` (and parents) and return ``step_dir``.

    Args:
        step_dir: Directory that will hold the ``step.yaml`` (its basename is
            the step name).
        env_classes: When provided, written as ``environment_configs`` keys so
            the env-class match can select the file; ``None`` omits the
            key entirely.
        subtypes: When provided (with ``env_classes``), written as the
            ``subtypes`` list under each class entry so the sub-type filter can
            be exercised; ``None`` leaves the entry with no sub-type restriction.
    """
    step_dir.mkdir(parents=True, exist_ok=True)
    data: dict = {"name": step_dir.name, "version": "v1", "type": "custom"}
    if env_classes is not None:
        entry = {"subtypes": list(subtypes)} if subtypes is not None else {}
        data["environment_configs"] = {cls: dict(entry) for cls in env_classes}
    (step_dir / "step.yaml").write_text(yaml.safe_dump(data))
    return step_dir


def _make_env(
    class_name: str,
    env_dir: Optional[Path] = None,
    subtype: Optional[str] = None,
):
    """Build a stand-in environment driving ``with_current_env``.

    Args:
        class_name: Becomes the env's class name (``current_env_class_name``).
        env_dir: Directory holding the env; exposed as ``environment_dir_uri``.
        subtype: When provided, attached as ``config.subtype`` so the sub-type
            filter (``current_env_subtype``) is scoped.
    """
    env_dir_uri = f"file://{env_dir}" if env_dir is not None else None
    attrs: dict = {"environment_dir_uri": env_dir_uri}
    if subtype is not None:
        attrs["config"] = type("Cfg", (), {"subtype": subtype})()
    return type(class_name, (), attrs)()


def _resolve(uri: str) -> URI:
    """Resolve a ``space://`` URI through the real ``SpaceURI`` resolver."""
    return URI.get_uri(uri, default_scheme="file", secrets={})


def _resolved_dir(uri: URI) -> Path:
    """Filesystem path the resolver landed on."""
    return Path(uri.uri.path)  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Base order — base_uris order decides between env-matching copies
# --------------------------------------------------------------------------- #


class TestBaseOrder:
    """A step copy that admits the active env comes from the earliest base that
    has one: the space's own dir first, then base_uris in the order listed.
    Owning folder depth and specificity rank copies within one base
    only."""

    def test_earlier_base_wins_over_more_specific_later_base(self, tmp_path):
        """A later base's single-env copy no longer beats an earlier base's
        multi-env copy: specificity ranks within a base, not across bases."""
        first = tmp_path / "first"
        early = _write_step(first / "steps" / "s3push", env_classes=["K8s", "Lsf"])
        second = tmp_path / "second"
        _write_step(second / "k8s" / "steps" / "s3push", env_classes=["K8s"])
        _set_bases(first, second)

        with SpaceURI.with_current_env_class_name("K8s"):
            resolved = _resolve("space://steps/s3push")

        assert _resolved_dir(resolved).samefile(early)

    def test_bases_listed_order_not_path_order(self, tmp_path):
        """Two equally specific copies in different bases: the base listed first
        wins, even when its path sorts after the other's."""
        zzz = tmp_path / "zzz"
        listed_first = _write_step(zzz / "steps" / "dup", env_classes=["K8s"])
        aaa = tmp_path / "aaa"
        _write_step(aaa / "steps" / "dup", env_classes=["K8s"])
        _set_bases(zzz, aaa)

        with SpaceURI.with_current_env_class_name("K8s"):
            resolved = _resolve("space://steps/dup")

        assert _resolved_dir(resolved).samefile(listed_first)

    def test_earlier_base_wins_over_env_colocated_in_later_base(self, tmp_path):
        """A copy in the env's own folder ranks only within its own base, so a
        base listed before that one wins — e.g. a generated step space listed
        ahead of the assets tree whose env dir carries a published copy."""
        space_root = tmp_path / "space"
        space_root.mkdir()
        generated = tmp_path / "generated"
        local = _write_step(generated / "steps" / "hello", env_classes=["Skypilot"])
        assets = tmp_path / "assets"
        env_dir = assets / "skypilot" / "slurm"
        _write_step(env_dir / "steps" / "hello", env_classes=["Skypilot"])
        _set_bases(space_root, generated, assets)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/hello")

        assert _resolved_dir(resolved).samefile(local)

    def test_env_colocated_wins_within_its_base(self, tmp_path):
        """Within one base, the copy in the active env's own folder still wins over a
        sibling env's copy that sorts first by path."""
        assets = tmp_path / "assets"
        _write_step(assets / "skypilot" / "aws" / "steps" / "bfcl", ["Skypilot"])
        env_dir = assets / "skypilot" / "lsf"
        near = _write_step(env_dir / "steps" / "bfcl", env_classes=["Skypilot"])
        _set_bases(tmp_path / "space", assets)
        (tmp_path / "space").mkdir()

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/bfcl")

        assert _resolved_dir(resolved).samefile(near)

    def test_sibling_env_copy_in_earlier_base_is_not_a_candidate(self, tmp_path):
        """A copy in another env's ``steps/`` dir doesn't serve the active env,
        even from an earlier base: the BlueVela env gets its own copy, not the
        aws env's copy from a base listed first."""
        space_root = tmp_path / "space"
        space_root.mkdir()
        gbtest = tmp_path / "gb-test"
        aws = gbtest / "environments" / "skypilot" / "aws"
        _write_step(aws / "steps" / "bfcl-eval", env_classes=["Skypilot"])
        assets = tmp_path / "assets"
        env_dir = assets / "environments" / "skypilot" / "lsf" / "ibm-bluevela"
        own = _write_step(env_dir / "steps" / "bfcl-eval", env_classes=["Skypilot"])
        _set_bases(space_root, gbtest, assets)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/bfcl-eval")

        assert _resolved_dir(resolved).samefile(own)

    def test_unknown_env_dir_prefers_top_level_steps(self, tmp_path):
        """With the env class known but its dir not (the validator couldn't load
        it, or no ``environment_asset``), the base's top-level catalog copy wins
        over any env folder's copy."""
        space_root = tmp_path / "space"
        space_root.mkdir()
        assets = tmp_path / "assets"
        catalog = _write_step(assets / "steps" / "hello", env_classes=["Skypilot"])
        envs = assets / "environments" / "skypilot"
        _write_step(envs / "aws" / "steps" / "hello", env_classes=["Skypilot"])
        _write_step(envs / "lsf" / "steps" / "hello", env_classes=["Skypilot"])
        _set_bases(space_root, assets)

        with SpaceURI.with_current_env_class_name("Skypilot"):
            resolved = _resolve("space://steps/hello")

        assert _resolved_dir(resolved).samefile(catalog)

    def test_nested_bases_measure_env_from_each_base(self, tmp_path):
        """With nested bases (``<repo>`` and ``<repo>/configurations/assets``),
        the outer base sees the env at its own relative path, so the family's
        step under the assets tree serves the env and outranks the outer base's
        top-level catalog copy."""
        space_root = tmp_path / "space"
        space_root.mkdir()
        repo = tmp_path / "repo"
        assets = repo / "configurations" / "assets"
        family = _write_step(
            assets / "environments" / "skypilot" / "steps" / "digit", ["Skypilot"]
        )
        _write_step(repo / "steps" / "digit", env_classes=["Skypilot"])
        env_dir = assets / "environments" / "skypilot" / "slurm"
        env_dir.mkdir(parents=True)
        _set_bases(space_root, repo, assets)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/digit")

        assert _resolved_dir(resolved).samefile(family)

    def test_sibling_env_nested_step_does_not_pin_the_name(self, tmp_path):
        """A nested ``distill/foo`` that exists only in a sibling env's folder
        doesn't fix the step name: the env's own family ``distill`` step serves
        ``space://steps/distill/foo`` with ``foo`` as its sub-file."""
        base = tmp_path / "assets"
        family = base / "environments" / "skypilot"
        outer = _write_step(family / "steps" / "distill", env_classes=["Skypilot"])
        (outer / "foo").write_text("sub-file\n")
        _write_step(family / "aws" / "steps" / "distill" / "foo", ["Skypilot"])
        env_dir = family / "lsf"
        env_dir.mkdir(parents=True)
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/distill/foo")

        assert _resolved_dir(resolved).samefile(outer / "foo")

    def test_unknown_env_dir_still_finds_family_only_steps(self, tmp_path):
        """With the env dir unknown (e.g. a validator clone failure), a step that
        lives only in a family folder still resolves, so validation doesn't
        reject a build that would resolve at runtime."""
        space_root = tmp_path / "space"
        space_root.mkdir()
        assets = tmp_path / "assets"
        family = _write_step(
            assets / "environments" / "skypilot" / "steps" / "digit", ["Skypilot"]
        )
        _set_bases(space_root, assets)

        with SpaceURI.with_current_env_class_name("Skypilot"):
            resolved = _resolve("space://steps/digit")

        assert _resolved_dir(resolved).samefile(family)

    def test_unknown_env_dir_prefers_shared_family_over_one_env(self, tmp_path):
        """With the env dir unknown, a shared family copy outranks one env's own
        copy (shallowest owning folder first)."""
        space_root = tmp_path / "space"
        space_root.mkdir()
        envs = tmp_path / "assets" / "environments" / "skypilot"
        family = _write_step(envs / "steps" / "digit", env_classes=["Skypilot"])
        _write_step(envs / "aws" / "steps" / "digit", env_classes=["Skypilot"])
        _set_bases(space_root, tmp_path / "assets")

        with SpaceURI.with_current_env_class_name("Skypilot"):
            resolved = _resolve("space://steps/digit")

        assert _resolved_dir(resolved).samefile(family)

    def test_space_root_outer_step_yields_to_longer_name(self, tmp_path):
        """A space that ships ``steps/distill`` doesn't shadow a later base's
        ``steps/distill/foo``: the longer name wins, so the URI isn't demoted to
        the space's ``distill`` plus a missing ``foo``."""
        space_root = tmp_path / "space"
        _write_step(space_root / "steps" / "distill", env_classes=["Skypilot"])
        assets = tmp_path / "assets"
        nested = _write_step(assets / "steps" / "distill" / "foo", ["Skypilot"])
        env_dir = assets / "environments" / "skypilot" / "slurm"
        env_dir.mkdir(parents=True)
        _set_bases(space_root, assets)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/distill/foo")

        assert _resolved_dir(resolved).samefile(nested)

    def test_space_root_outer_step_serves_its_own_subfile(self, tmp_path):
        """The space's ``distill`` still serves ``space://steps/distill/foo``
        when ``foo`` is its own sub-file and no longer step name exists."""
        space_root = tmp_path / "space"
        outer = _write_step(space_root / "steps" / "distill", ["Skypilot"])
        (outer / "foo").write_text("sub-file\n")
        env_dir = tmp_path / "assets" / "environments" / "skypilot" / "slurm"
        env_dir.mkdir(parents=True)
        _set_bases(space_root, tmp_path / "assets")

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/distill/foo")

        assert _resolved_dir(resolved).samefile(outer / "foo")

    def test_step_index_refreshes_after_reclone(self, tmp_path):
        """The per-base step index notices a base re-cloned into the same path:
        a removed step is gone and a new one is found."""
        base = tmp_path / "assets"
        _write_step(base / "steps" / "skypilot" / "gone", ["Skypilot"])
        assert "gone" in SpaceURI._step_index(base)
        shutil.rmtree(base)
        _write_step(base / "steps" / "skypilot" / "other", ["Skypilot"])

        index = SpaceURI._step_index(base)

        assert "gone" not in index and "other" in index

    def test_family_steps_match_across_repo_layouts(self, tmp_path):
        """A family's steps under ``configurations/assets/environments/...`` in
        one repo serve that family's env loaded from a repo laid out as
        ``environments/...`` (compared from the ``environments`` segment)."""
        space_root = tmp_path / "space"
        space_root.mkdir()
        repo = tmp_path / "repo"
        family = _write_step(
            repo
            / "configurations"
            / "assets"
            / "environments"
            / "skypilot"
            / "steps"
            / "digit",
            ["Skypilot"],
        )
        gbtest = tmp_path / "gb-test"
        env_dir = gbtest / "environments" / "skypilot" / "slurm" / "bluevela"
        env_dir.mkdir(parents=True)
        _set_bases(space_root, gbtest, repo)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/digit")

        assert _resolved_dir(resolved).samefile(family)

    def test_family_steps_in_another_base_serve_the_env(self, tmp_path):
        """A family's shared ``environments/skypilot/steps/`` serves a skypilot
        env loaded from a different base: ownership is compared by path inside
        each base, so cross-repo family steps still resolve."""
        space_root = tmp_path / "space"
        space_root.mkdir()
        shared = tmp_path / "assets"
        family = _write_step(
            shared / "environments" / "skypilot" / "steps" / "digit", ["Skypilot"]
        )
        gbtest = tmp_path / "gb-test"
        env_dir = gbtest / "environments" / "skypilot" / "slurm" / "bluevela"
        env_dir.mkdir(parents=True)
        _set_bases(space_root, gbtest, shared)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/digit")

        assert _resolved_dir(resolved).samefile(family)

    def test_listed_base_overrides_a_builtin(self, tmp_path):
        """With the builtins last (as ``Space`` orders them), a base listed in
        space.yaml that ships the env's own copy of a builtin-named step wins."""
        space_root = tmp_path / "space"
        space_root.mkdir()
        assets = tmp_path / "assets"
        env_dir = assets / "environments" / "skypilot" / "slurm"
        override = _write_step(env_dir / "steps" / "hfpull", env_classes=["Skypilot"])
        builtins = tmp_path / "builtins"
        _write_step(builtins / "steps" / "skypilot" / "hfpull", ["Skypilot"])
        _set_bases(space_root, assets, builtins)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/hfpull")

        assert _resolved_dir(resolved).samefile(override)

    def test_subasset_comes_only_from_the_chosen_copy(self, tmp_path):
        """``space://steps/foo/<rest>`` resolves within the copy that
        ``space://steps/foo`` picks; when that copy lacks ``<rest>`` it's a miss,
        not a file taken from a different copy of the step."""
        first = tmp_path / "first"
        chosen = _write_step(first / "steps" / "foo", env_classes=["K8s"])
        second = tmp_path / "second"
        other = _write_step(second / "steps" / "foo", env_classes=["K8s"])
        (other / "scripts").mkdir()
        (other / "scripts" / "run.sh").write_text("echo other\n")
        _set_bases(first, second)

        with SpaceURI.with_current_env_class_name("K8s"):
            assert _resolved_dir(_resolve("space://steps/foo")).samefile(chosen)
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/foo/scripts/run.sh")

    def test_env_excluded_earlier_copy_is_skipped(self, tmp_path):
        """Base order applies only among copies that admit the env: an earlier
        base's copy for another env class is skipped."""
        first = tmp_path / "first"
        _write_step(first / "steps" / "hello", env_classes=["Lsf"])
        second = tmp_path / "second"
        admitted = _write_step(second / "steps" / "hello", env_classes=["K8s"])
        _set_bases(first, second)

        with SpaceURI.with_current_env_class_name("K8s"):
            resolved = _resolve("space://steps/hello")

        assert _resolved_dir(resolved).samefile(admitted)


# --------------------------------------------------------------------------- #
# Space root and the env's own folder
# --------------------------------------------------------------------------- #


class TestSpaceRootAndEnvFolder:
    def test_env_colocated_hit_wins_within_its_base(self, tmp_path):
        """In the base holding the env, the env's own copy wins over the base's
        top-level catalog copy (deeper owning dir first)."""
        space_root = tmp_path / "space"  # base_uris[0], ships no steps/hello
        space_root.mkdir()
        assets = tmp_path / "assets"
        _write_step(assets / "steps" / "hello", env_classes=["Bash"])
        env_dir = assets / "envs" / "bash"
        colocated = _write_step(env_dir / "steps" / "hello", env_classes=["Bash"])
        _set_bases(space_root, assets)

        with SpaceURI.with_current_env(_make_env("Bash", env_dir)):
            resolved = _resolve("space://steps/hello")

        assert _resolved_dir(resolved).samefile(colocated)

    def test_space_root_step_wins_over_env_colocated(self, tmp_path):
        """The space's own root step (base_uris[0]/steps/<name>) overrides an
        same-named step in the env's own folder — so a step developed in its own
        space is exercised before it is published into an inherited tree."""
        space_root = tmp_path / "space"
        space_step = _write_step(space_root / "steps" / "hello", ["Skypilot"])
        env_dir = tmp_path / "assets" / "skypilot" / "slurm"
        _write_step(env_dir / "steps" / "hello", ["Skypilot"])  # must lose
        _set_bases(space_root, tmp_path / "assets")

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/hello")

        assert _resolved_dir(resolved).samefile(space_step)

    def test_space_root_step_skipped_when_subtypes_exclude_env(self, tmp_path):
        """A space-root step whose ``subtypes`` exclude the active env is skipped,
        so resolution falls through to the env's own copy (the space-root
        priority never bypasses the sub-type gate)."""
        space_root = tmp_path / "space"
        _write_step(
            space_root / "steps" / "hello",
            env_classes=["Skypilot"],
            subtypes=["kubernetes"],  # excludes aws
        )
        env_dir = tmp_path / "assets" / "skypilot" / "aws"
        colocated = _write_step(env_dir / "steps" / "hello", env_classes=["Skypilot"])
        _set_bases(space_root, tmp_path / "assets")

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir, subtype="aws")):
            resolved = _resolve("space://steps/hello")

        assert _resolved_dir(resolved).samefile(colocated)

    def test_env_folder_miss_falls_through_to_class_match(self, tmp_path):
        """When the env dir lacks the step, resolution falls through to the
        env-class match in the base's catalog (proving ordering). The match sits
        in the base's top-level ``steps/`` (an env-agnostic catalog, like the
        builtins' ``steps/<env-class>/<name>``), which serves any env."""
        base = tmp_path / "base"
        class_match = _write_step(base / "steps" / "k8s" / "digit", env_classes=["K8s"])
        env_dir = tmp_path / "envs" / "k8s"  # exists, but has no steps/digit
        env_dir.mkdir(parents=True)
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("K8s", env_dir)):
            resolved = _resolve("space://steps/digit")

        assert _resolved_dir(resolved).samefile(class_match)


# --------------------------------------------------------------------------- #
# Env-folder ownership — family steps folders and deepest-owner ranking
# --------------------------------------------------------------------------- #


class TestEnvFolderOwnership:
    def test_parent_steps_shared_by_siblings(self, tmp_path):
        """A step at the family level is found by sibling envs below it."""
        base = tmp_path / "assets"
        shared = _write_step(base / "skypilot" / "steps" / "digit", ["Skypilot"])
        for leaf in ("kubernetes", "slurm"):
            env_dir = base / "skypilot" / leaf
            env_dir.mkdir(parents=True)
            _set_bases(base)
            with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
                resolved = _resolve("space://steps/digit")
            assert _resolved_dir(resolved).samefile(shared)

    def test_env_own_dir_overrides_parent(self, tmp_path):
        """A step in the env's own dir wins over a same-named ancestor step: the
        deeper owning dir ranks first within a base."""
        base = tmp_path / "assets"
        _write_step(base / "skypilot" / "steps" / "digit", env_classes=["Skypilot"])
        env_dir = base / "skypilot" / "kubernetes"
        own = _write_step(env_dir / "steps" / "digit", env_classes=["Skypilot"])
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/digit")

        assert _resolved_dir(resolved).samefile(own)

    def test_grandparent_folder_and_deeper_wins(self, tmp_path):
        """Across several family levels, the deepest owning folder wins."""
        base = tmp_path / "assets"
        _write_step(base / "skypilot" / "steps" / "x", ["Skypilot"])  # grandparent
        nearer = _write_step(base / "skypilot" / "lsf" / "steps" / "x", ["Skypilot"])
        env_dir = base / "skypilot" / "lsf" / "ibm-bluevela"
        env_dir.mkdir(parents=True)
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/x")

        assert _resolved_dir(resolved).samefile(nearer)

    def test_step_above_base_is_not_reachable(self, tmp_path):
        """A step above the enclosing base_uri is outside every base, so it isn't found."""
        base = tmp_path / "assets"
        _write_step(tmp_path / "steps" / "digit")  # ABOVE the base
        env_dir = base / "skypilot" / "kubernetes"
        env_dir.mkdir(parents=True)
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/digit")

    def test_env_dir_not_under_any_base(self, tmp_path):
        """An env dir under no base (e.g. an absolute ``file://`` env) is searched
        as its own base: its own steps resolve, but its parents' do not."""
        base = tmp_path / "other"
        base.mkdir()
        _write_step(tmp_path / "elsewhere" / "steps" / "parent_only", ["Bash"])
        env_dir = tmp_path / "elsewhere" / "env"
        own = _write_step(env_dir / "steps" / "hello", env_classes=["Bash"])
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("Bash", env_dir)):
            assert _resolved_dir(_resolve("space://steps/hello")).samefile(own)
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/parent_only")

    def test_outside_env_dir_outranks_listed_bases(self, tmp_path):
        """An outside env dir ranks right after the space dir: its own copy beats
        a listed base's copy, while the space's own copy still beats it."""
        space_root = tmp_path / "space"
        space_root.mkdir()
        listed = tmp_path / "assets"
        _write_step(listed / "steps" / "hello", env_classes=["Bash"])
        env_dir = tmp_path / "elsewhere" / "env"
        own = _write_step(env_dir / "steps" / "hello", env_classes=["Bash"])
        _set_bases(space_root, listed)

        with SpaceURI.with_current_env(_make_env("Bash", env_dir)):
            assert _resolved_dir(_resolve("space://steps/hello")).samefile(own)
            space_copy = _write_step(space_root / "steps" / "hello", ["Bash"])
            assert _resolved_dir(_resolve("space://steps/hello")).samefile(space_copy)

    def test_subasset_rest_from_parent(self, tmp_path):
        """`space://steps/<name>/<rest>` resolves against an ancestor step dir."""
        base = tmp_path / "assets"
        step_dir = _write_step(base / "skypilot" / "steps" / "sage", ["Skypilot"])
        sub = step_dir / "Dockerfile"
        sub.write_text("FROM scratch\n")
        env_dir = base / "skypilot" / "kubernetes"
        env_dir.mkdir(parents=True)
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/sage/Dockerfile")

        assert _resolved_dir(resolved).samefile(sub)

    def test_rest_traversal_outside_step_dir_rejected(self, tmp_path):
        """A `<rest>` that escapes the step dir (``../../secret``) is rejected —
        resolution must not resolve outside the matched step dir even
        though the target file exists."""
        base = tmp_path / "assets"
        _write_step(base / "skypilot" / "steps" / "sage")
        (base / "secret").write_text("password\n")  # real file, outside the step dir
        env_dir = base / "skypilot" / "kubernetes"
        env_dir.mkdir(parents=True)
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/sage/../../../secret")


# --------------------------------------------------------------------------- #
# Sub-type matching — gates every lookup by env sub-type
# --------------------------------------------------------------------------- #


class TestSubtypeMatching:
    def test_ancestor_step_matches_listed_subtype(self, tmp_path):
        """A shared ancestor step restricted to [kubernetes, slurm] resolves for
        both of those sub-types via the family steps folder."""
        base = tmp_path / "assets"
        shared = _write_step(
            base / "skypilot" / "steps" / "digit",
            env_classes=["Skypilot"],
            subtypes=["kubernetes", "slurm"],
        )
        for sub in ("kubernetes", "slurm"):
            env_dir = base / "skypilot" / sub
            env_dir.mkdir(parents=True, exist_ok=True)
            _set_bases(base)
            with SpaceURI.with_current_env(_make_env("Skypilot", env_dir, subtype=sub)):
                resolved = _resolve("space://steps/digit")
            assert _resolved_dir(resolved).samefile(shared)

    def test_ancestor_step_excludes_unlisted_subtype(self, tmp_path):
        """The same restricted step is unresolvable for a sub-type not listed
        (aws) — it is skipped and no other copy matches."""
        base = tmp_path / "assets"
        _write_step(
            base / "skypilot" / "steps" / "digit",
            env_classes=["Skypilot"],
            subtypes=["kubernetes", "slurm"],
        )
        env_dir = base / "skypilot" / "aws"
        env_dir.mkdir(parents=True)
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir, subtype="aws")):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/digit")

    def test_unset_subtype_excluded_from_restricted_step(self, tmp_path):
        """An env with no sub-type does not match a step that lists sub-types."""
        base = tmp_path / "assets"
        _write_step(
            base / "skypilot" / "steps" / "digit",
            env_classes=["Skypilot"],
            subtypes=["kubernetes", "slurm"],
        )
        env_dir = base / "skypilot" / "kubernetes"
        env_dir.mkdir(parents=True)
        _set_bases(base)

        # subtype not passed -> env has no sub-type
        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/digit")

    def test_empty_subtypes_is_universal(self, tmp_path):
        """A step with no sub-types matches any env, even a subtyped one — so
        builtins/general steps keep resolving for subtyped endpoints."""
        base = tmp_path / "assets"
        shared = _write_step(
            base / "skypilot" / "steps" / "s3push", env_classes=["Skypilot"]
        )  # no subtypes
        env_dir = base / "skypilot" / "aws"
        env_dir.mkdir(parents=True)
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir, subtype="aws")):
            resolved = _resolve("space://steps/s3push")

        assert _resolved_dir(resolved).samefile(shared)

    def test_scalar_subtypes_uses_exact_match_not_substring(self, tmp_path):
        """A scalar `subtypes: kubernetes` (not a list) must match the sub-type
        exactly, not as a substring — a prefix like ``"k"`` must NOT resolve,
        while the full ``"kubernetes"`` does."""
        base = tmp_path / "base"
        step_dir = base / "steps" / "digit"
        step_dir.mkdir(parents=True)
        (step_dir / "step.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "digit",
                    "version": "v1",
                    "type": "custom",
                    # scalar (not a list) — the bug turned membership into a
                    # substring test.
                    "environment_configs": {"Skypilot": {"subtypes": "kubernetes"}},
                }
            )
        )
        _set_bases(base)

        # substring "k" of "kubernetes" must not sneak past the exact-match gate
        with SpaceURI.with_current_env_class_name("Skypilot", env_subtype="k"):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/digit")
        # the exact sub-type still resolves
        with SpaceURI.with_current_env_class_name("Skypilot", env_subtype="kubernetes"):
            resolved = _resolve("space://steps/digit")
        assert _resolved_dir(resolved).samefile(step_dir)

    def test_own_dir_step_skipped_when_subtypes_exclude_env(self, tmp_path):
        """The sub-type filter applies even to a step in the env's OWN dir: if
        its ``subtypes`` exclude the active env it is skipped despite being the
        deepest, and the family folder's admitting copy is used."""
        base = tmp_path / "assets"
        ancestor = _write_step(
            base / "skypilot" / "steps" / "digit",
            env_classes=["Skypilot"],
            subtypes=["kubernetes", "aws"],  # admits aws
        )
        env_dir = base / "skypilot" / "aws"
        _write_step(
            env_dir / "steps" / "digit",
            env_classes=["Skypilot"],
            subtypes=["kubernetes"],  # own-dir copy EXCLUDES aws
        )
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir, subtype="aws")):
            resolved = _resolve("space://steps/digit")

        # own-dir copy is skipped by the filter; the family copy admits aws
        assert _resolved_dir(resolved).samefile(ancestor)

    def test_own_dir_step_excluded_unresolvable_without_fallback(self, tmp_path):
        """An own-dir step whose ``subtypes`` exclude the env, with no admitting
        ancestor, resolves to nothing (the own-dir hit is not a free pass)."""
        base = tmp_path / "assets"
        env_dir = base / "skypilot" / "aws"
        _write_step(
            env_dir / "steps" / "digit",
            env_classes=["Skypilot"],
            subtypes=["kubernetes", "slurm"],  # excludes aws
        )
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir, subtype="aws")):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/digit")

    def test_class_match_honors_subtype(self, tmp_path):
        """The env-class match applies the same sub-type filter: a
        restricted candidate is excluded for an unlisted sub-type."""
        base = tmp_path / "base"
        _write_step(
            base / "steps" / "skypilot" / "digit",
            env_classes=["Skypilot"],
            subtypes=["kubernetes", "slurm"],
        )
        _set_bases(base)

        # aws is not in the list -> class-match must not select the candidate
        with SpaceURI.with_current_env_class_name("Skypilot", env_subtype="aws"):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/digit")
        # kubernetes is listed -> class-match selects it
        with SpaceURI.with_current_env_class_name("Skypilot", env_subtype="kubernetes"):
            resolved = _resolve("space://steps/digit")
        assert _resolved_dir(resolved).name == "digit"

    def test_validator_scope_drives_resolution(self, tmp_path):
        """The validator's ``with_current_env_class_name`` scope (env_dir_uri +
        env_subtype) drives env-folder ownership and the sub-type filter without a full
        env: a listed sub-type resolves, an unlisted one does not."""
        base = tmp_path / "assets"
        family = _write_step(
            base / "skypilot" / "steps" / "digit",
            env_classes=["Skypilot"],
            subtypes=["kubernetes", "slurm"],
        )
        env_dir = base / "skypilot" / "kubernetes"
        env_dir.mkdir(parents=True)
        aws_dir = base / "skypilot" / "aws"
        aws_dir.mkdir(parents=True)
        _set_bases(base)

        with SpaceURI.with_current_env_class_name(
            "Skypilot", env_dir_uri=f"file://{env_dir}", env_subtype="kubernetes"
        ):
            assert _resolved_dir(_resolve("space://steps/digit")).samefile(family)
        with SpaceURI.with_current_env_class_name(
            "Skypilot", env_dir_uri=f"file://{aws_dir}", env_subtype="aws"
        ):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/digit")


# --------------------------------------------------------------------------- #
# Env-class presence — every lookup rejects steps scoped to other classes
# --------------------------------------------------------------------------- #


class TestEnvClassPresence:
    """The directory-based lookups (space root, env folders, fallback)
    must not admit a step whose ``environment_configs`` declares only *other*
    env classes — such a step cannot run under the active environment."""

    def test_space_root_mismatch_does_not_shadow_colocated(self, tmp_path):
        """A space-root step declaring only another class (``Bash``) must NOT
        shadow a valid env-folder step for the active class (``Skypilot``);
        resolution falls through the space-root priority to the env's own copy."""
        space_root = tmp_path / "space"
        _write_step(space_root / "steps" / "foo", env_classes=["Bash"])
        env_dir = tmp_path / "assets" / "skypilot" / "slurm"
        colocated = _write_step(env_dir / "steps" / "foo", env_classes=["Skypilot"])
        _set_bases(space_root, tmp_path / "assets")

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/foo")

        assert _resolved_dir(resolved).samefile(colocated)

    def test_env_folder_skips_mismatched_class_step(self, tmp_path):
        """A deepest (own-dir) step keyed for a different class is skipped by the
        class gate and the family folder's admitting copy is used (mirrors the
        sub-type own-dir-skip test, but for the class dimension)."""
        base = tmp_path / "assets"
        ancestor = _write_step(
            base / "skypilot" / "steps" / "digit", env_classes=["Skypilot"]
        )
        env_dir = base / "skypilot" / "aws"
        _write_step(env_dir / "steps" / "digit", env_classes=["Bash"])  # wrong class
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir, subtype="aws")):
            resolved = _resolve("space://steps/digit")

        assert _resolved_dir(resolved).samefile(ancestor)

    def test_fallback_rejects_mismatched_class_step(self, tmp_path):
        """An env-agnostic fallback hit whose ``environment_configs`` lists
        only another class is rejected, so resolution raises rather than
        resolving to a step that cannot run under the active env."""
        space_root = tmp_path / "space"  # base_uris[0], ships no steps/foo
        space_root.mkdir()
        _write_step(tmp_path / "assets" / "steps" / "foo", env_classes=["Bash"])
        _set_bases(space_root, tmp_path / "assets")

        # No env dir -> no space-root lookup; class match skips (class absent); fallback
        # finds assets/steps/foo but the class gate rejects it.
        with SpaceURI.with_current_env_class_name("Skypilot"):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/foo")

    def test_step_without_environment_configs_is_not_a_match(self, tmp_path):
        """A step that declares no ``environment_configs`` has no launcher for
        any env, so with an env class active it is never selected: resolving it
        fails up front instead of later in the build."""
        base = tmp_path / "assets"
        bare = _write_step(base / "steps" / "foo")  # no environment_configs
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("Skypilot"):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/foo")
        # With no env class active (e.g. tooling) the path still resolves.
        assert _resolved_dir(_resolve("space://steps/foo")).samefile(bare)

    def test_present_but_null_class_entry_resolves(self, tmp_path):
        """A step keyed for the active class with a **null** value
        (``environment_configs: {Skypilot:}``) declares the class and must
        resolve — presence is by key, not value. A value-based ``is None`` gate
        would wrongly skip this step, which is scoped to exactly the active env."""
        base = tmp_path / "assets"
        step_dir = base / "steps" / "foo"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "step.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "foo",
                    "version": "v1",
                    "type": "custom",
                    "environment_configs": {"Skypilot": None},
                }
            )
        )
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("Skypilot"):
            resolved = _resolve("space://steps/foo")

        assert _resolved_dir(resolved).samefile(step_dir)


# --------------------------------------------------------------------------- #
# Env-class match (specificity + tie-break)
# --------------------------------------------------------------------------- #


class TestEnvClassMatch:
    def test_single_env_file_beats_multi_env_catchall(self, tmp_path):
        """A single-env split file (fewer environment_configs keys) beats a
        multi-env catch-all that also lists the active class."""
        base = tmp_path / "base"
        _write_step(base / "steps" / "s3push", env_classes=["K8s", "Lsf", "Skypilot"])
        specific = _write_step(base / "steps" / "k8s" / "s3push", env_classes=["K8s"])
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("K8s"):
            resolved = _resolve("space://steps/s3push")

        assert _resolved_dir(resolved).samefile(specific)

    def test_present_but_null_class_entry_wins_specificity(self, tmp_path):
        """A present-but-null class entry (``{K8s:}``) is admitted by the class match and
        ranks by specificity like any declared class.

        Pins the fix for the second env-class gate: a value-based
        ``_env_config_entry(...) is not None`` check would silently drop the
        null-valued, single-env file, letting the less-specific multi-env catch-all
        win here — disagreeing with the sibling ``_env_ok`` gate. With presence
        decided by key, the null-valued split file (1 key) beats the catch-all.
        The class match must decide (no env dir, so no space-root lookup, and the
        fallback never runs).
        """
        base = tmp_path / "base"
        _write_step(
            base / "steps" / "foo", env_classes=["K8s", "Skypilot"]
        )  # catch-all, 2 keys
        specific = base / "steps" / "k8s" / "foo"  # null-valued, single-env
        specific.mkdir(parents=True, exist_ok=True)
        (specific / "step.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "foo",
                    "version": "v1",
                    "type": "custom",
                    "environment_configs": {"K8s": None},
                }
            )
        )
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("K8s"):
            resolved = _resolve("space://steps/foo")

        assert _resolved_dir(resolved).samefile(specific)

    def test_equal_specificity_lexicographic_tiebreak(self, tmp_path):
        """Among equally-specific matches, the lexicographically smaller path
        wins (deterministic tie-break)."""
        base = tmp_path / "base"
        first = _write_step(base / "steps" / "aaa" / "dup", env_classes=["K8s"])
        _write_step(base / "steps" / "bbb" / "dup", env_classes=["K8s"])
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("K8s"):
            resolved = _resolve("space://steps/dup")

        assert _resolved_dir(resolved).samefile(first)

    def test_no_class_match_when_class_absent(self, tmp_path):
        """A candidate that does not list the active env class is ignored;
        with nothing else matching, resolution raises."""
        base = tmp_path / "base"
        _write_step(base / "skypilot" / "steps" / "only", env_classes=["Skypilot"])
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("K8s"):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/only")

    def test_class_match_is_case_insensitive(self, tmp_path):
        """Env class `K8s` matches an `environment_configs` key `k8s` — the
        class-name comparison is case-insensitive.  Placed under `k8s/steps/`
        (not the space-root `steps/`) so only the env-class match finds it."""
        base = tmp_path / "base"
        match = _write_step(base / "steps" / "k8s" / "digit", env_classes=["k8s"])
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("K8s"):
            resolved = _resolve("space://steps/digit")

        assert _resolved_dir(resolved).samefile(match)

    def test_builtin_env_class_dir_layout_resolves(self, tmp_path):
        """A builtin step laid out ``steps/<env-class>/<name>/step.yaml`` (the
        env-partition dir sits *between* ``steps/`` and the step name, as under
        ``src/gbserver/builtins/steps/skypilot/command/``) resolves for its class.

        Regression pin: the class match's name pin and glob must tolerate the
        env-partition segment between ``steps/`` and ``<name>`` — anchoring the
        glob to ``steps/<name>`` directly (name immediately under ``steps/``) made
        every builtin ``space://steps/command`` unresolvable (issue: PR #437)."""
        base = tmp_path / "builtins"
        match = _write_step(
            base / "steps" / "skypilot" / "command", env_classes=["Skypilot"]
        )
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("Skypilot"):
            resolved = _resolve("space://steps/command")

        assert _resolved_dir(resolved).samefile(match)

    def test_builtin_layout_picks_active_class_variant(self, tmp_path):
        """With sibling env-class dirs under ``steps/`` (``steps/skypilot/command``
        and ``steps/bash/command``, each scoped to its own class), the active
        class selects its own variant — the ``**`` anchor matches both, the
        env-class gate disambiguates (mirrors the real multi-env builtin tree)."""
        base = tmp_path / "builtins"
        sky = _write_step(
            base / "steps" / "skypilot" / "command", env_classes=["Skypilot"]
        )
        bash = _write_step(base / "steps" / "bash" / "command", env_classes=["Bash"])
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("Skypilot"):
            assert _resolved_dir(_resolve("space://steps/command")).samefile(sky)
        with SpaceURI.with_current_env_class_name("Bash"):
            assert _resolved_dir(_resolve("space://steps/command")).samefile(bash)

    def test_subasset_uri_appends_rest_to_matched_dir(self, tmp_path):
        """`space://steps/<name>/<rest>` resolves against the matched step dir
        plus the `<rest>` suffix."""
        base = tmp_path / "base"
        step_dir = _write_step(base / "steps" / "k8s" / "digit", env_classes=["K8s"])
        sub = step_dir / "helm-charts"
        sub.mkdir()
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("K8s"):
            resolved = _resolve("space://steps/digit/helm-charts")

        assert _resolved_dir(resolved).samefile(sub)

    def test_subasset_rest_traversal_rejected(self, tmp_path):
        """A `<rest>` that escapes the matched step dir is rejected by the
        env-class match too (shares the containment guard)."""
        base = tmp_path / "base"
        _write_step(base / "k8s" / "steps" / "digit", env_classes=["K8s"])
        (base / "secret").write_text("password\n")  # real file, outside step dir
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("K8s"):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/digit/../../secret")


# --------------------------------------------------------------------------- #
# Env-agnostic fallback + unresolvable
# --------------------------------------------------------------------------- #


class TestFallback:
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

    def test_fallback_honors_subtype_restriction(self, tmp_path):
        """A restricted step at the env-agnostic `<base>/steps/<name>` path is not
        resolvable by the fallback for an excluded sub-type — the restriction is
        never bypassed by falling through to the fallback."""
        base = tmp_path / "base"
        _write_step(
            base / "steps" / "digit",
            env_classes=["Skypilot"],
            subtypes=["kubernetes"],
        )
        _set_bases(base)

        # aws excluded: the class match drops it AND the fallback must not rescue it by path.
        with SpaceURI.with_current_env_class_name("Skypilot", env_subtype="aws"):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/digit")
        # kubernetes is listed -> still resolvable.
        with SpaceURI.with_current_env_class_name("Skypilot", env_subtype="kubernetes"):
            resolved = _resolve("space://steps/digit")
        assert _resolved_dir(resolved).name == "digit"

    def test_fallback_universal_step_resolves_for_subtyped_env(self, tmp_path):
        """An env-agnostic step with no `subtypes` still resolves via the fallback
        for a subtyped env (the fallback filter only excludes restricted steps)."""
        base = tmp_path / "base"
        step_dir = _write_step(base / "steps" / "hello", ["Skypilot"])  # no subtypes
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("Skypilot", env_subtype="aws"):
            resolved = _resolve("space://steps/hello")

        assert _resolved_dir(resolved).samefile(step_dir)

    def test_non_step_uri_uses_fallback_only(self, tmp_path):
        """The step lookups apply only to `steps/`; an environments URI resolves
        purely via the base_uris fallback."""
        base = tmp_path / "base"
        env_dir = base / "environments" / "bash"
        env_dir.mkdir(parents=True)
        _set_bases(base)

        resolved = _resolve("space://environments/bash")

        assert _resolved_dir(resolved).samefile(env_dir)

    def test_fallback_rest_traversal_rejected(self, tmp_path):
        """A `<rest>` that escapes the step dir (``../../secret``) is rejected on
        the fallback too — the containment guard the step lookups have applies
        here, so the resolver won't land outside the step dir even though the
        target file exists."""
        base = tmp_path / "base"
        _write_step(base / "steps" / "hello")
        (base / "secret").write_text("password\n")  # real file, outside the step dir
        _set_bases(base)

        with pytest.raises(ValueError, match="Unresolvable space uri"):
            _resolve("space://steps/hello/../../secret")

    def test_fallback_dotdot_rest_cannot_bypass_subtype(self, tmp_path):
        """A `..` in `<rest>` cannot hop from a restricted step to a sibling.

        `space://steps/digit/../x` normalizes to `<base>/steps/x`, but the
        `subtypes` gate reads `digit`'s own `step.yaml` and the containment guard
        rejects the escape, so the restriction is not bypassed — the URI is
        Unresolvable even though `x` alone would resolve for this env."""
        base = tmp_path / "base"
        _write_step(
            base / "steps" / "digit",
            env_classes=["Skypilot"],
            subtypes=["kubernetes"],
        )
        _write_step(base / "steps" / "x")  # universal sibling that `x` alone resolves
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("Skypilot", env_subtype="aws"):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/digit/../x")

    def test_fallback_no_step_yaml_at_any_prefix(self, tmp_path):
        """A bare dir under ``steps/`` with no ``step.yaml`` at any prefix is
        admitted, as before multi-segment names existed.

        This pins the fallback's compat branch (``_fallback_steps_ok``): when
        ``_longest_step_in_root`` finds no ``step.yaml`` prefixing ``after``, the
        first segment is treated as the step name and the remainder as the
        sub-asset ``rest``, which the env gate admits on the missing
        ``step.yaml``. Every other fallback test writes a ``step.yaml`` and so
        exercises only the ``found is not None`` branch."""
        base = tmp_path / "base"
        plain = base / "steps" / "plain"  # dir under steps/, no step.yaml
        plain.mkdir(parents=True)
        (plain / "asset.txt").write_text("x\n")
        _set_bases(base)

        resolved = _resolve("space://steps/plain/asset.txt")

        assert _resolved_dir(resolved).samefile(plain / "asset.txt")

    def test_fallback_no_step_yaml_rest_traversal_rejected(self, tmp_path):
        """The containment guard still applies on the no-``step.yaml`` compat
        branch: a ``<rest>`` escaping the (step.yaml-less) ``steps/<name>`` dir is
        rejected even though the target file exists outside it."""
        base = tmp_path / "base"
        plain = base / "steps" / "plain"  # dir under steps/, no step.yaml
        plain.mkdir(parents=True)
        (base / "secret").write_text("password\n")  # real file, outside the step dir
        _set_bases(base)

        with pytest.raises(ValueError, match="Unresolvable space uri"):
            _resolve("space://steps/plain/../../secret")

    def test_unresolvable_raises(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        _set_bases(base)

        with pytest.raises(ValueError, match="Unresolvable space uri"):
            _resolve("space://steps/missing")


# --------------------------------------------------------------------------- #
# Nested (multi-segment) step identity — steps/<a>/<b> addressed as
# space://steps/<a>/<b> across every lookup
# --------------------------------------------------------------------------- #


class TestNestedStepIdentity:
    """A step whose dir is nested more than one level under a ``steps/`` ancestor
    (e.g. ``steps/distill/foo``) is addressable as ``space://steps/distill/foo``.

    The step name is the longest ``steps/``-relative prefix of the URI suffix
    whose directory holds a ``step.yaml``; the remainder is a contained
    sub-asset.  Name selection is by ``step.yaml`` existence and is orthogonal
    to the env gate, so identity is deterministic.
    """

    def test_space_root_nested_step(self, tmp_path):
        """Space root: a nested step the space itself ships resolves off base_uris[0]."""
        space_root = tmp_path / "space"
        nested = _write_step(space_root / "steps" / "distill" / "foo", ["Skypilot"])
        env_dir = tmp_path / "assets" / "skypilot" / "slurm"
        env_dir.mkdir(parents=True)
        _set_bases(space_root, tmp_path / "assets")

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/distill/foo")

        assert _resolved_dir(resolved).samefile(nested)

    def test_family_folder_nested_step(self, tmp_path):
        """Family folder: a nested step in the env family's steps folder resolves
        for an env below it."""
        base = tmp_path / "assets"
        nested = _write_step(
            base / "skypilot" / "steps" / "distill" / "foo", env_classes=["Skypilot"]
        )
        env_dir = base / "skypilot" / "kubernetes"
        env_dir.mkdir(parents=True)
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/distill/foo")

        assert _resolved_dir(resolved).samefile(nested)

    def test_class_match_nested_step(self, tmp_path):
        """Class match: a nested step under a ``steps/`` folder is matched by its
        ``steps/``-relative path (``distill/foo``), not a bare leaf dir name."""
        base = tmp_path / "base"
        nested = _write_step(
            base / "steps" / "k8s" / "distill" / "foo", env_classes=["K8s"]
        )
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("K8s"):
            resolved = _resolve("space://steps/distill/foo")

        assert _resolved_dir(resolved).samefile(nested)

    def test_fallback_nested_step(self, tmp_path):
        """Fallback: with no active env, a nested step resolves via the plain
        base_uris fallback."""
        base = tmp_path / "base"
        nested = _write_step(base / "steps" / "distill" / "foo")
        _set_bases(base)

        resolved = _resolve("space://steps/distill/foo")

        assert _resolved_dir(resolved).samefile(nested)

    def test_both_exist_longest_prefix_wins(self, tmp_path):
        """Both ``steps/distill/step.yaml`` and ``steps/distill/foo/step.yaml``
        exist: ``space://steps/distill/foo`` binds to the NESTED step, not
        'step distill + sub-asset foo'.

        The two steps are scoped to different env classes so the choice is
        observable through the env gate: the URI resolves only because the
        *nested* step's ``step.yaml`` (Skypilot) is the one consulted.
        """
        base = tmp_path / "base"
        _write_step(base / "steps" / "distill", env_classes=["Bash"])
        nested = _write_step(
            base / "steps" / "distill" / "foo", env_classes=["Skypilot"]
        )
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("Skypilot"):
            resolved = _resolve("space://steps/distill/foo")

        assert _resolved_dir(resolved).samefile(nested)

    def test_nested_step_subasset(self, tmp_path):
        """A sub-asset of a nested step resolves against the nested step dir:
        ``space://steps/distill/foo/helm-charts`` → step ``distill/foo`` + rest."""
        base = tmp_path / "base"
        step_dir = _write_step(base / "steps" / "distill" / "foo")
        sub = step_dir / "helm-charts"
        sub.mkdir()
        _set_bases(base)

        resolved = _resolve("space://steps/distill/foo/helm-charts")

        assert _resolved_dir(resolved).samefile(sub)

    def test_nested_step_rest_traversal_rejected(self, tmp_path):
        """A ``<rest>`` escaping a nested step dir (``../../../secret``) is still
        rejected by the containment guard."""
        base = tmp_path / "base"
        _write_step(base / "steps" / "distill" / "foo")
        (base / "secret").write_text("password\n")  # real file, outside step dir
        _set_bases(base)

        with pytest.raises(ValueError, match="Unresolvable space uri"):
            _resolve("space://steps/distill/foo/../../../secret")

    def test_subasset_not_treated_as_nested_step(self, tmp_path):
        """When no ``step.yaml`` exists at the longer prefix, resolution is
        unchanged: ``space://steps/digit/helm-charts`` → step ``digit`` + the
        ``helm-charts`` sub-asset (regression pin for single-segment steps)."""
        base = tmp_path / "base"
        step_dir = _write_step(base / "steps" / "digit")
        sub = step_dir / "helm-charts"  # plain sub-asset dir, NO step.yaml
        sub.mkdir()
        _set_bases(base)

        resolved = _resolve("space://steps/digit/helm-charts")

        assert _resolved_dir(resolved).samefile(sub)

    def test_env_excluded_nested_not_demoted_to_outer(self, tmp_path):
        """A nested step present but env-excluded must NOT silently demote to
        'outer step + sub-asset' — even when the outer step declares the active
        env class.

        This pins the class match's demotion path: the outer ``distill`` explicitly
        lists **Skypilot** (the active class) while the nested ``distill/foo``
        lists only **Bash**.  Choosing the name by env-class match (the old
        behavior) made ``distill/foo`` miss (Bash-only) and fall back to
        ``distill`` (Skypilot), returning ``distill`` + sub-asset ``foo`` and
        resolving to the nested dir as a mere asset.  Name selection is now
        pinned by ``step.yaml`` existence first (``distill/foo``), then gated; the
        gate misses and ``space://steps/distill/foo`` is Unresolvable for
        Skypilot rather than demoted.

        A *universal* outer (no ``environment_configs``) would NOT exercise this:
        the env-class match rejects a universal step by key presence, so the
        demotion never fires and the test would pass via the fallback's rejection
        instead — see :meth:`test_universal_outer_nested_excluded_unresolvable`
        for that distinct (fallback) path.  No env dir is set here, so the space-root
        lookup is skipped and this reaches the class match specifically.
        """
        base = tmp_path / "base"
        _write_step(
            base / "steps" / "distill", env_classes=["Skypilot"]
        )  # outer DECLARES the active class (the realistic demotion trap)
        _write_step(
            base / "steps" / "distill" / "foo", env_classes=["Bash"]
        )  # nested, excludes Skypilot
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("Skypilot"):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/distill/foo")

    def test_class_match_name_pin_ignores_stray_step_yaml_outside_steps(self, tmp_path):
        """Class-match regression: a stray ``step.yaml`` *outside* any ``steps/`` dir
        must not hijack an existing sub-asset URI.

        The multi-segment name pin globs for ``steps/<name>/step.yaml`` anchored
        to the enclosing ``steps/`` (the canonical-name invariant).  An
        unanchored ``<name>/step.yaml`` glob would match a ``step.yaml`` anywhere
        under a base — e.g. a test fixture at
        ``test-data/fixtures/digit/helm-charts/step.yaml`` — and, because the
        name can now be multi-segment, bind ``space://steps/digit/helm-charts``
        to that stray instead of to the real step ``digit``'s ``helm-charts``
        sub-asset.

        Layout: the real step ``.../steps/digit`` (declares the active class so
        the class match decides — it lives below ``base_uris[0]`` but not at
        ``base_uris[0]/steps``, so the space-root lookup misses) with a plain ``helm-charts``
        sub-asset dir, plus an unrelated stray step under ``test-data/fixtures``.
        Resolution must land on the real sub-asset, never the stray.
        """
        base = tmp_path / "base"
        real = _write_step(
            base / "steps" / "assets" / "digit", env_classes=["Skypilot"]
        )
        sub = real / "helm-charts"  # plain sub-asset dir, NO step.yaml
        sub.mkdir()
        _write_step(  # stray fixture outside any steps/ dir
            base / "test-data" / "fixtures" / "digit" / "helm-charts",
            env_classes=["Skypilot"],
        )
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("Skypilot"):
            resolved = _resolve("space://steps/digit/helm-charts")

        assert _resolved_dir(resolved).samefile(sub)

    def test_universal_outer_nested_excluded_unresolvable(self, tmp_path):
        """Companion to :meth:`test_env_excluded_nested_not_demoted_to_outer`
        covering the *universal* outer step (fallback rejection path).

        With a universal outer ``distill`` (no ``environment_configs``) and a
        Bash-only nested ``distill/foo``, the class match cannot match the outer (presence
        is by declared key), so the name stays pinned to the excluded
        ``distill/foo`` and resolution falls through to the fallback, which pins the
        same nested step by existence and rejects it on the env gate.  Either way
        ``space://steps/distill/foo`` is Unresolvable for Skypilot — the universal
        outer never captures ``foo`` as a sub-asset.
        """
        base = tmp_path / "base"
        _write_step(base / "steps" / "distill")  # universal outer step
        _write_step(
            base / "steps" / "distill" / "foo", env_classes=["Bash"]
        )  # nested, excludes Skypilot
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("Skypilot"):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/distill/foo")


# --------------------------------------------------------------------------- #
# Git base_uris — resolution off the reused local clone (no re-clone)
# --------------------------------------------------------------------------- #

_GIT_BASE = "git+ssh://example.com/org/repo.git@main"


def _fake_git_clone(monkeypatch, clone: Path) -> None:
    """Make any GitURI resolve to ``clone`` without touching the network.

    Both ``GitURI.exists()`` and ``SpaceURI._uri_to_local_path`` go through
    ``get_path_in_repo_from_cache``; patching it to return a ready local dir
    mimics a repo that was already cloned into the thread-local cache.
    """
    monkeypatch.setattr(
        "gbcommon.uri.git.GitURI.get_path_in_repo_from_cache",
        lambda self, force=False: clone,
    )


class TestGitBaseUri:
    def test_uri_to_local_path_reuses_git_clone(self, tmp_path, monkeypatch):
        """A git base resolves to its existing clone dir (no re-clone), so the
        resolver's local operations have a real directory to work against."""
        clone = tmp_path / "clone"
        clone.mkdir()
        _fake_git_clone(monkeypatch, clone)

        assert SpaceURI._uri_to_local_path(_GIT_BASE) == clone

    def test_fallback_admits_non_local_base(self, monkeypatch):
        """When a base can't be materialized locally, the fallback steps guard
        admits it (path-existence-only, pre-`subtypes` behavior) rather than
        excluding it — the regression that broke git-backed spaces."""
        monkeypatch.setattr(
            SpaceURI, "_uri_to_local_path", staticmethod(lambda _: None)
        )
        assert SpaceURI._fallback_steps_ok(_GIT_BASE, "digit")

    def test_git_fallback_resolves_step(self, tmp_path, monkeypatch):
        """`space://steps/digit` resolves against a git base via the reused
        clone and returns the (git) resolved URI — the end-to-end regression."""
        clone = tmp_path / "clone"
        _write_step(clone / "steps" / "digit")  # digit at repo-root steps/
        _fake_git_clone(monkeypatch, clone)
        SpaceURI.set_baseuris([_GIT_BASE], {})

        resolved = _resolve("space://steps/digit")

        assert resolved.uri.scheme.startswith("git")

    def test_git_fallback_honors_subtype(self, tmp_path, monkeypatch):
        """The subtype filter runs against the git clone: an excluded sub-type
        is unresolvable, an admitted one resolves."""
        clone = tmp_path / "clone"
        _write_step(
            clone / "steps" / "digit",
            env_classes=["Skypilot"],
            subtypes=["kubernetes"],
        )
        _fake_git_clone(monkeypatch, clone)
        SpaceURI.set_baseuris([_GIT_BASE], {})

        with SpaceURI.with_current_env_class_name("Skypilot", env_subtype="aws"):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/digit")
        # kubernetes is admitted; the env-class match resolves it off the
        # clone (returning a local file URI into the clone).
        with SpaceURI.with_current_env_class_name("Skypilot", env_subtype="kubernetes"):
            resolved = _resolve("space://steps/digit")
        assert _resolved_dir(resolved).samefile(clone / "steps" / "digit")

    def test_git_class_match(self, tmp_path, monkeypatch):
        """The env-class match globs the git clone: a class-keyed step under
        a nested `steps/` (not the clone-root `steps/`) still resolves for the
        matching env class."""
        clone = tmp_path / "clone"
        _write_step(
            clone / "steps" / "k8s" / "digit", env_classes=["k8s"]
        )  # class-match only, under a nested steps/
        _fake_git_clone(monkeypatch, clone)
        SpaceURI.set_baseuris([_GIT_BASE], {})

        with SpaceURI.with_current_env_class_name("K8s"):
            resolved = _resolve("space://steps/digit")

        assert _resolved_dir(resolved).samefile(clone / "steps" / "k8s" / "digit")


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
