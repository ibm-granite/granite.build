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
            the env-class-match tier can select the file; ``None`` omits the
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
# Tier 1 — env-co-located step lookup
# --------------------------------------------------------------------------- #


class TestTier1EnvColocated:
    def test_env_colocated_hit_wins_over_inherited_base(self, tmp_path):
        """A step in the active env's own dir wins over a same-named step in an
        *inherited* base_uri (base_uris[1:]).  Only the space's own root
        (base_uris[0]) outranks an env-co-located step (see the space-root tests
        below); an inherited base does not."""
        space_root = tmp_path / "space"  # base_uris[0], ships no steps/hello
        space_root.mkdir()
        inherited = _write_step(tmp_path / "assets" / "steps" / "hello").parent.parent
        env_dir = tmp_path / "envs" / "bash"
        colocated = _write_step(env_dir / "steps" / "hello")
        _set_bases(space_root, inherited)

        with SpaceURI.with_current_env(_make_env("Bash", env_dir)):
            resolved = _resolve("space://steps/hello")

        assert _resolved_dir(resolved).samefile(colocated)

    def test_space_root_step_wins_over_env_colocated(self, tmp_path):
        """The space's own root step (base_uris[0]/steps/<name>) overrides an
        env-co-located step of the same name — so a step developed in its own
        space is exercised before it is published into an inherited tree."""
        space_root = tmp_path / "space"
        space_step = _write_step(space_root / "steps" / "hello")
        env_dir = tmp_path / "assets" / "skypilot" / "slurm"
        _write_step(env_dir / "steps" / "hello")  # env-co-located, must lose
        _set_bases(space_root, tmp_path / "assets")

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/hello")

        assert _resolved_dir(resolved).samefile(space_step)

    def test_space_root_step_skipped_when_subtypes_exclude_env(self, tmp_path):
        """A space-root step whose ``subtypes`` exclude the active env is skipped,
        so resolution falls through to the env-co-located step (the space-root
        priority never bypasses the sub-type gate)."""
        space_root = tmp_path / "space"
        _write_step(
            space_root / "steps" / "hello",
            env_classes=["Skypilot"],
            subtypes=["kubernetes"],  # excludes aws
        )
        env_dir = tmp_path / "assets" / "skypilot" / "aws"
        colocated = _write_step(env_dir / "steps" / "hello")
        _set_bases(space_root, tmp_path / "assets")

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir, subtype="aws")):
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
# Tier 1 — ancestor-walk (nearest-wins, base_uri-bounded)
# --------------------------------------------------------------------------- #


class TestTier1AncestorWalk:
    def test_parent_steps_shared_by_siblings(self, tmp_path):
        """A step at the family level is found by sibling envs below it."""
        base = tmp_path / "assets"
        shared = _write_step(base / "skypilot" / "steps" / "digit")
        for leaf in ("kubernetes", "slurm"):
            env_dir = base / "skypilot" / leaf
            env_dir.mkdir(parents=True)
            _set_bases(base)
            with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
                resolved = _resolve("space://steps/digit")
            assert _resolved_dir(resolved).samefile(shared)

    def test_env_own_dir_overrides_parent(self, tmp_path):
        """A step in the env's own dir wins over a same-named ancestor step."""
        base = tmp_path / "assets"
        _write_step(base / "skypilot" / "steps" / "digit")
        env_dir = base / "skypilot" / "kubernetes"
        own = _write_step(env_dir / "steps" / "digit")
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/digit")

        assert _resolved_dir(resolved).samefile(own)

    def test_grandparent_walk_and_nearer_wins(self, tmp_path):
        """The nearest ancestor with the step wins across multiple levels."""
        base = tmp_path / "assets"
        _write_step(base / "skypilot" / "steps" / "x")  # grandparent
        nearer = _write_step(base / "skypilot" / "lsf" / "steps" / "x")  # parent
        env_dir = base / "skypilot" / "lsf" / "ibm-bluevela"
        env_dir.mkdir(parents=True)
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/x")

        assert _resolved_dir(resolved).samefile(nearer)

    def test_walk_stops_at_base_boundary(self, tmp_path):
        """A step above the enclosing base_uri is not reachable via the walk."""
        base = tmp_path / "assets"
        _write_step(tmp_path / "steps" / "digit")  # ABOVE the base
        env_dir = base / "skypilot" / "kubernetes"
        env_dir.mkdir(parents=True)
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/digit")

    def test_env_dir_not_under_any_base(self, tmp_path):
        """When the env dir is under no base, the walk degenerates to that dir:
        its own step resolves but a parent's step is not walked into."""
        base = tmp_path / "other"
        base.mkdir()
        _write_step(tmp_path / "elsewhere" / "steps" / "parent_only")
        env_dir = tmp_path / "elsewhere" / "env"
        own = _write_step(env_dir / "steps" / "hello")
        _set_bases(base)

        with SpaceURI.with_current_env(_make_env("Bash", env_dir)):
            resolved = _resolve("space://steps/hello")
            assert _resolved_dir(resolved).samefile(own)
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/parent_only")

    def test_subasset_rest_from_parent(self, tmp_path):
        """`space://steps/<name>/<rest>` resolves against an ancestor step dir."""
        base = tmp_path / "assets"
        step_dir = _write_step(base / "skypilot" / "steps" / "sage")
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
        the ancestor-walk must not resolve outside the matched step dir even
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
# Sub-type matching — gates ancestor-walk and env-class-match by env sub-type
# --------------------------------------------------------------------------- #


class TestSubtypeMatching:
    def test_ancestor_step_matches_listed_subtype(self, tmp_path):
        """A shared ancestor step restricted to [kubernetes, slurm] resolves for
        both of those sub-types via the ancestor-walk."""
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
        (aws) — the walk skips it and no other tier matches."""
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
        """The sub-type filter applies even to a step in the env's OWN dir (walk
        level 0): if its ``subtypes`` exclude the active env it is skipped despite
        being nearest, and the walk continues to an admitting ancestor."""
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

        # nearest (own-dir) copy is skipped by the filter; ancestor admits aws
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

    def test_class_match_tier_honors_subtype(self, tmp_path):
        """The env-class-match tier applies the same sub-type filter: a
        restricted candidate is excluded for an unlisted sub-type."""
        base = tmp_path / "base"
        _write_step(
            base / "skypilot" / "digit",
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

    def test_validator_scope_drives_both_tiers(self, tmp_path):
        """The validator's ``with_current_env_class_name`` scope (env_dir_uri +
        env_subtype) drives the ancestor-walk and sub-type filter without a full
        env: a listed sub-type resolves, an unlisted one does not."""
        base = tmp_path / "assets"
        walked = _write_step(
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
            assert _resolved_dir(_resolve("space://steps/digit")).samefile(walked)
        with SpaceURI.with_current_env_class_name(
            "Skypilot", env_dir_uri=f"file://{aws_dir}", env_subtype="aws"
        ):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/digit")


# --------------------------------------------------------------------------- #
# Env-class presence — directory tiers must reject steps scoped to other classes
# --------------------------------------------------------------------------- #


class TestEnvClassPresence:
    """The directory-based tiers (space-root, ancestor-walk, Tier-3 fallback)
    must not admit a step whose ``environment_configs`` declares only *other*
    env classes — such a step cannot run under the active environment."""

    def test_space_root_mismatch_does_not_shadow_colocated(self, tmp_path):
        """A space-root step declaring only another class (``Bash``) must NOT
        shadow a valid co-located step for the active class (``Skypilot``);
        resolution falls through the space-root priority to the co-located one."""
        space_root = tmp_path / "space"
        _write_step(space_root / "steps" / "foo", env_classes=["Bash"])
        env_dir = tmp_path / "assets" / "skypilot" / "slurm"
        colocated = _write_step(env_dir / "steps" / "foo", env_classes=["Skypilot"])
        _set_bases(space_root, tmp_path / "assets")

        with SpaceURI.with_current_env(_make_env("Skypilot", env_dir)):
            resolved = _resolve("space://steps/foo")

        assert _resolved_dir(resolved).samefile(colocated)

    def test_ancestor_walk_skips_mismatched_class_step(self, tmp_path):
        """A nearest (own-dir) step keyed for a different class is skipped by the
        class gate and the walk continues to an admitting ancestor (mirrors the
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
        """A Tier-3 env-agnostic fallback hit whose ``environment_configs`` lists
        only another class is rejected, so resolution raises rather than
        resolving to a step that cannot run under the active env."""
        space_root = tmp_path / "space"  # base_uris[0], ships no steps/foo
        space_root.mkdir()
        _write_step(tmp_path / "assets" / "steps" / "foo", env_classes=["Bash"])
        _set_bases(space_root, tmp_path / "assets")

        # No env dir -> Tier 1 walk is inert; Tier 2 skips (class absent); Tier 3
        # finds assets/steps/foo but the class gate rejects it.
        with SpaceURI.with_current_env_class_name("Skypilot"):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/foo")

    def test_step_without_environment_configs_stays_universal(self, tmp_path):
        """A step that declares no ``environment_configs`` at all is env-agnostic
        and still resolves for any active class — the class gate only applies to
        a step that *declares* the block (preserves directory-placed steps)."""
        base = tmp_path / "assets"
        universal = _write_step(base / "steps" / "foo")  # no environment_configs
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("Skypilot"):
            resolved = _resolve("space://steps/foo")

        assert _resolved_dir(resolved).samefile(universal)

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
# Tier 2 — env-class match (specificity + tie-break)
# --------------------------------------------------------------------------- #


class TestTier2EnvClassMatch:
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

    def test_present_but_null_class_entry_wins_specificity(self, tmp_path):
        """A present-but-null class entry (``{K8s:}``) is admitted by this tier and
        ranks by specificity like any declared class.

        Pins the fix for the second env-class gate: a value-based
        ``_env_config_entry(...) is not None`` check would silently drop the
        null-valued, single-env file, letting the less-specific multi-env catch-all
        win here — disagreeing with the sibling ``_env_ok`` tier. With presence
        decided by key, the null-valued split file (1 key) beats the catch-all.
        This tier must decide (no env dir, so Tier 1 is inert and Tier 3 never runs).
        """
        base = tmp_path / "base"
        _write_step(base / "foo", env_classes=["K8s", "Skypilot"])  # catch-all, 2 keys
        specific = base / "k8s" / "foo"  # null-valued, single-env
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

    def test_class_match_is_case_insensitive(self, tmp_path):
        """Env class `K8s` matches an `environment_configs` key `k8s` — the
        class-name comparison is case-insensitive.  Placed under `k8s/` (not the
        root `steps/`) so only the env-class-match tier can find it."""
        base = tmp_path / "base"
        match = _write_step(base / "k8s" / "digit", env_classes=["k8s"])
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("K8s"):
            resolved = _resolve("space://steps/digit")

        assert _resolved_dir(resolved).samefile(match)

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

    def test_subasset_rest_traversal_rejected(self, tmp_path):
        """A `<rest>` that escapes the matched step dir is rejected by the
        env-class-match tier too (shares the containment guard)."""
        base = tmp_path / "base"
        _write_step(base / "k8s" / "digit", env_classes=["K8s"])
        (base / "secret").write_text("password\n")  # real file, outside step dir
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("K8s"):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/digit/../../secret")


# --------------------------------------------------------------------------- #
# Tier 3 — env-agnostic fallback + unresolvable
# --------------------------------------------------------------------------- #


class TestTier3Fallback:
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
        never bypassed by falling through to Tier 3."""
        base = tmp_path / "base"
        _write_step(
            base / "steps" / "digit",
            env_classes=["Skypilot"],
            subtypes=["kubernetes"],
        )
        _set_bases(base)

        # aws excluded: Tier 2 drops it AND Tier 3 must not rescue it by path.
        with SpaceURI.with_current_env_class_name("Skypilot", env_subtype="aws"):
            with pytest.raises(ValueError, match="Unresolvable space uri"):
                _resolve("space://steps/digit")
        # kubernetes is listed -> still resolvable.
        with SpaceURI.with_current_env_class_name("Skypilot", env_subtype="kubernetes"):
            resolved = _resolve("space://steps/digit")
        assert _resolved_dir(resolved).name == "digit"

    def test_fallback_universal_step_resolves_for_subtyped_env(self, tmp_path):
        """An env-agnostic step with no `subtypes` still resolves via the fallback
        for a subtyped env (the Tier 3 filter only excludes restricted steps)."""
        base = tmp_path / "base"
        step_dir = _write_step(base / "steps" / "hello")  # no subtypes
        _set_bases(base)

        with SpaceURI.with_current_env_class_name("Skypilot", env_subtype="aws"):
            resolved = _resolve("space://steps/hello")

        assert _resolved_dir(resolved).samefile(step_dir)

    def test_non_step_uri_uses_fallback_only(self, tmp_path):
        """Tiers 1/2 apply only to `steps/`; an environments URI resolves
        purely via the base_uris fallback."""
        base = tmp_path / "base"
        env_dir = base / "environments" / "bash"
        env_dir.mkdir(parents=True)
        _set_bases(base)

        resolved = _resolve("space://environments/bash")

        assert _resolved_dir(resolved).samefile(env_dir)

    def test_fallback_rest_traversal_rejected(self, tmp_path):
        """A `<rest>` that escapes the step dir (``../../secret``) is rejected on
        the Tier 3 fallback too — the containment guard Tiers 1 & 2 have applies
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

    def test_unresolvable_raises(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        _set_bases(base)

        with pytest.raises(ValueError, match="Unresolvable space uri"):
            _resolve("space://steps/missing")


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
        """When a base can't be materialized locally, the Tier 3 steps guard
        admits it (path-existence-only, pre-`subtypes` behavior) rather than
        excluding it — the regression that broke git-backed spaces."""
        monkeypatch.setattr(
            SpaceURI, "_uri_to_local_path", staticmethod(lambda _: None)
        )
        assert SpaceURI._fallback_steps_ok(_GIT_BASE, ("digit", ""))

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
        # kubernetes is admitted; env-class-match (Tier 2) resolves it off the
        # clone (returning a local file URI into the clone).
        with SpaceURI.with_current_env_class_name("Skypilot", env_subtype="kubernetes"):
            resolved = _resolve("space://steps/digit")
        assert _resolved_dir(resolved).samefile(clone / "steps" / "digit")

    def test_git_tier2_class_match(self, tmp_path, monkeypatch):
        """Env-class-match (Tier 2) globs the git clone: a class-keyed step not
        at the root `steps/` still resolves for the matching env class."""
        clone = tmp_path / "clone"
        _write_step(clone / "k8s" / "digit", env_classes=["k8s"])  # class-match only
        _fake_git_clone(monkeypatch, clone)
        SpaceURI.set_baseuris([_GIT_BASE], {})

        with SpaceURI.with_current_env_class_name("K8s"):
            resolved = _resolve("space://steps/digit")

        assert _resolved_dir(resolved).samefile(clone / "k8s" / "digit")


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
