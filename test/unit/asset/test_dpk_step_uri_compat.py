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

"""The deprecated ``space://steps/dpk`` URI still resolves to the moved step.

dpk moved from ``steps/dpk`` to the ``data`` category (``steps/data/dpk``), so its
canonical URI is now ``space://steps/data/dpk``. The old URI is still used by
recipes and builds outside this repo, and by the BlueVela fixtures, whose
assets repo keeps the step at ``steps/skypilot/dpk``.

Nothing special-cases the old name. It keeps resolving because the resolver's
env-class-match tier searches ``steps/**/<name>/step.yaml``, and the ``**``
also matches through a category directory. That only holds while exactly one
published step is named ``dpk``. A second one (another category, or a stale
``steps/dpk`` left behind by an incomplete move) would make the old URI
ambiguous or shadow the moved step, so these tests pin both conditions against
the real ``configurations/assets`` tree.
"""

from pathlib import Path

import pytest

from gbcommon.uri.space import SpaceURI
from gbcommon.uri.uri import URI

REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "configurations" / "assets"
SKYPILOT_DIR = ASSETS_DIR / "environments" / "skypilot"
MOVED_STEP_DIR = SKYPILOT_DIR / "steps" / "data" / "dpk"

# dpk declares no `subtypes:`, so it must resolve on every SkyPilot endpoint.
SKYPILOT_ENVS = [
    ("slurm", "slurm"),
    ("aws", "aws"),
    ("kubernetes", "kubernetes"),
    ("lsf/ibm-bluevela", "lsf"),
]

_THREAD_LOCAL_ATTRS = (
    "base_uris",
    "space_secrets",
    "current_env_dir_uri",
    "current_env_class_name",
    "current_env_subtype",
)


@pytest.fixture(autouse=True)
def _isolate_thread_local():
    """Snapshot and restore ``SpaceURI._thread_local`` around each test."""
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


def _skypilot_env(env_dir: Path, subtype: str):
    """Stand-in for a ``Skypilot`` environment, as ``with_current_env`` reads it."""
    cfg = type("Cfg", (), {"subtype": subtype})()
    return type(
        "Skypilot", (), {"environment_dir_uri": f"file://{env_dir}", "config": cfg}
    )()


def _resolve_dir(uri: str, env_rel: str, subtype: str) -> Path:
    SpaceURI.set_baseuris([f"file://{ASSETS_DIR}"], {})
    env = _skypilot_env(SKYPILOT_DIR / env_rel, subtype)
    with SpaceURI.with_current_env(env):
        resolved = URI.get_uri(uri, default_scheme="file", secrets={})
    return Path(resolved.uri.path).resolve()  # type: ignore[union-attr]


@pytest.mark.parametrize("env_rel,subtype", SKYPILOT_ENVS)
@pytest.mark.parametrize("uri", ["space://steps/data/dpk", "space://steps/dpk"])
def test_both_uris_resolve_to_the_moved_step(uri, env_rel, subtype):
    assert (MOVED_STEP_DIR / "step.yaml").is_file(), "dpk is not published"
    assert _resolve_dir(uri, env_rel, subtype) == MOVED_STEP_DIR.resolve()


def test_exactly_one_published_step_is_named_dpk():
    """The old URI is a basename lookup; a second ``dpk`` would break it."""
    found = sorted(
        p.parent.relative_to(ASSETS_DIR).as_posix()
        for p in ASSETS_DIR.glob("environments/skypilot/steps/**/dpk/step.yaml")
    )
    assert found == ["environments/skypilot/steps/data/dpk"], (
        "space://steps/dpk relies on dpk being the only published step with that "
        f"name; found {found}"
    )
