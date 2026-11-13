# Copyright Granite.Build Authors
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

"""Every published step build-test fixture's ``step_uri`` resolves in its space.

``make publish-step`` copies each framework step's build tests to
``test-data/steps/<category>/<name>/<env>/<cluster>/``, with a ``build.yaml``
and a ``buildtest.yaml`` whose ``space_uri`` names the space the test runs in.
Those builds only run in the extended / live suites, so a ``step_uri`` that no
longer resolves (e.g. after a change to how ``space://steps/...`` URIs resolve)
would otherwise surface only in an integration run.  This resolves each
target's environment and each of its steps the way a build does, against that
space, and fails at unit-test time instead.

A ``build.yaml`` with no ``buildtest.yaml`` beside it is checked against the
shared local space, which is what every published fixture uses today.
"""

import pathlib
from typing import Iterator, List, Optional, Tuple

import pytest
import yaml

from gbcommon.uri.space import SpaceURI
from gbcommon.uri.uri import URI
from gbserver.build.space import Space

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "test-data" / "steps"
DEFAULT_SPACE = REPO_ROOT / "configurations" / "spaces" / "local"
_BUILD_KEYS = ("granite.build", "llm.build")
_TL_ATTRS = (
    "base_uris",
    "space_secrets",
    "current_env_dir_uri",
    "current_env_class_name",
    "current_env_subtype",
)


@pytest.fixture(autouse=True)
def _isolate_thread_local() -> Iterator[None]:
    """Save and restore the resolver's thread-local around each test."""
    tl = SpaceURI._thread_local
    saved = {a: getattr(tl, a, None) for a in _TL_ATTRS}
    try:
        yield
    finally:
        for attr, value in saved.items():
            if value is None:
                if hasattr(tl, attr):
                    delattr(tl, attr)
            else:
                setattr(tl, attr, value)


def _space_dir(build_yaml: pathlib.Path) -> pathlib.Path:
    """Return the space a fixture's build runs in, from its ``buildtest.yaml``.

    Args:
        build_yaml: The fixture's ``build.yaml``.

    Returns:
        The space directory: ``space_uri`` resolved against the fixture dir, or
        the shared local space when there is no ``buildtest.yaml``.
    """
    buildtest = build_yaml.with_name("buildtest.yaml")
    if not buildtest.is_file():
        return DEFAULT_SPACE
    space_uri = (yaml.safe_load(buildtest.read_text()) or {}).get("space_uri")
    assert space_uri, f"{buildtest} has no space_uri"
    rel = space_uri.removeprefix("file://")
    return (build_yaml.parent / rel).resolve()


def _targets(build_yaml: pathlib.Path) -> List[Tuple[str, str, List[str]]]:
    """Return ``(target, environment_uri, step_uris)`` for each target.

    Args:
        build_yaml: The fixture's ``build.yaml``.

    Returns:
        One entry per target that names an environment and steps.
    """
    data = yaml.safe_load(build_yaml.read_text()) or {}
    build = next((data[k] for k in _BUILD_KEYS if k in data), {}) or {}
    found = []
    for name, target in (build.get("targets") or {}).items():
        target = target or {}
        env_uri = target.get("environment_uri")
        steps = [s["step_uri"] for s in target.get("steps") or [] if s.get("step_uri")]
        if env_uri and steps:
            found.append((name, env_uri, steps))
    return found


def _cases() -> List[Tuple[str, pathlib.Path, str, str, List[str]]]:
    """Return one case per fixture target, for parametrization."""
    cases = []
    for build_yaml in sorted(FIXTURES.glob("**/build.yaml")):
        for target, env_uri, steps in _targets(build_yaml):
            fixture = build_yaml.parent.relative_to(FIXTURES)
            cases.append((f"{fixture}:{target}", build_yaml, target, env_uri, steps))
    return cases


def _env_dir(env_uri: str, build_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """Return the environment's directory, as resolved for the build.

    Args:
        env_uri: The target's ``environment_uri``.
        build_dir: The fixture dir, which a relative ``file:`` URI is under.

    Returns:
        The directory holding the environment's ``environment.yaml``, or
        ``None`` when the URI doesn't resolve to a local directory.
    """
    if env_uri.startswith("file:") and not env_uri.startswith("file://"):
        rel = env_uri.removeprefix("file:")
        return (build_dir / rel).resolve()
    resolved = URI.get_uri(env_uri, default_scheme="file", secrets={})
    if resolved.uri is None:
        return None
    return pathlib.Path(resolved.uri.path).resolve()


def _env_types(env_dir: pathlib.Path) -> Tuple[Optional[str], Optional[str]]:
    """Return the environment's ``type`` and ``subtype``.

    Args:
        env_dir: The environment's directory.

    Returns:
        ``(type, subtype)`` from its ``environment.yaml``.
    """
    data = yaml.safe_load((env_dir / "environment.yaml").read_text()) or {}
    return data.get("type"), data.get("subtype")


CASES = _cases()


def test_fixtures_were_found():
    """Guard against the glob silently matching nothing."""
    assert len(CASES) >= 20, f"found only {len(CASES)} fixture targets"


@pytest.mark.parametrize(
    "build_yaml, target, env_uri, step_uris",
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_fixture_step_uris_resolve(build_yaml, target, env_uri, step_uris):
    """Each step of each target resolves for that target's environment."""
    Space(_space_dir(build_yaml).as_uri())
    env_dir = _env_dir(env_uri, build_yaml.parent)
    assert (
        env_dir is not None and (env_dir / "environment.yaml").is_file()
    ), f"{target}: environment {env_uri} doesn't resolve to an environment"
    env_class, env_subtype = _env_types(env_dir)

    with SpaceURI.with_current_env_class_name(
        env_class, env_dir_uri=env_dir.as_uri(), env_subtype=env_subtype
    ):
        for step_uri in step_uris:
            resolved = URI.get_uri(step_uri, default_scheme="file", secrets={})
            assert resolved.exists(), f"{target}: {step_uri} resolved but is missing"
