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

"""Which copy of the distill recipe's eval, export and teardown steps each
SkyPilot environment resolves (#664).

``recipes/distill/distill/build.yaml`` runs on SkyPilot/SLURM, but these steps
lived only under ``skypilot/lsf/ibm-bluevela/steps/`` (plus AWS variants of the
two evals under ``skypilot/aws/steps/``). Neither is an ancestor of a SLURM env,
so env-class-match chose between them and its lexicographic tie-break picked the
AWS evals (``infra: aws/us-east-2``) for SLURM.

These tests resolve each step through the real ``SpaceURI`` resolver against the
checked-in ``configurations/assets`` tree, with the active env scoped the way
``gbserver.build.targetstep`` scopes it, for every SkyPilot sub-type.
"""

from pathlib import Path
from typing import Optional

import pytest
import yaml

from gbcommon.uri.space import SpaceURI
from gbcommon.uri.uri import URI

REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS = REPO_ROOT / "configurations/assets"
SKY = ASSETS / "environments/skypilot"

STEP_NAMES = (
    "skypilot-teardown",
    "sage-eval",
    "bfcl-eval",
    "sage-export",
    "bfcl-export",
    "combined-export",
)
EVALS = ("sage-eval", "bfcl-eval")

SHARED = "environments/skypilot/steps"
AWS = "environments/skypilot/aws/steps"

# Expected resolution, as a directory relative to configurations/assets, or None
# for unresolvable.
_SHARED_ALL = {name: f"{SHARED}/{name}" for name in STEP_NAMES}
EXPECTED = {
    # Every SkyPilot env shares one universal copy of each step, found by the
    # ancestor-walk from the env dir up to skypilot/steps/.
    "slurm": _SHARED_ALL,
    "local": _SHARED_ALL,
    "lsf": _SHARED_ALL,
    # Kubernetes: teardown and the exports are the same files as before #664, now
    # at the shared location. The evals deliberately change: before, class-match
    # gave kubernetes the AWS variants (infra: aws/us-east-2), which cannot run
    # there; now it gets the shared copies.
    "kubernetes": _SHARED_ALL,
    # AWS keeps its own evals (restricted to subtypes: [aws]); the other four
    # resolve to the shared copies, the same files it resolved before.
    "aws": {
        name: (f"{AWS}/{name}" if name in EVALS else f"{SHARED}/{name}")
        for name in STEP_NAMES
    },
}

_THREAD_LOCAL_ATTRS = (
    "base_uris",
    "space_secrets",
    "current_env_dir_uri",
    "current_env_class_name",
    "current_env_subtype",
)


@pytest.fixture(autouse=True)
def _isolate_thread_local():
    """Restore ``SpaceURI._thread_local`` after each test (see
    ``test_space_uri_resolver.py``)."""
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


def _env(env_dir: Path):
    """A stand-in for the env loaded from ``env_dir/environment.yaml``: its class
    is the file's ``type`` and its ``config.subtype`` the file's ``subtype``, the
    two fields ``SpaceURI.with_current_env`` reads besides the env dir."""
    data = yaml.safe_load((env_dir / "environment.yaml").read_text(encoding="utf-8"))
    cfg = type("Cfg", (), {"subtype": data.get("subtype")})()
    attrs = {"environment_dir_uri": f"file://{env_dir}", "config": cfg}
    return type(data["type"], (), attrs)()


@pytest.fixture
def local_overlay(tmp_path: Path) -> Path:
    """The standalone layout ``scripts/setup-local-space.sh`` writes: a SLURM env
    under ``<space>/overlay/environments/skypilot-local/<name>``, with the overlay
    ahead of ``configurations/assets`` in base_uris."""
    env_dir = tmp_path / "overlay/environments/skypilot-local/test-slurm"
    env_dir.mkdir(parents=True)
    src = SKY / "slurm/bluevela/environment.yaml"
    (env_dir / "environment.yaml").write_text(src.read_text(encoding="utf-8"))
    return tmp_path / "overlay"


def _setup(env_key: str, overlay: Optional[Path]) -> Path:
    """Point base_uris for ``env_key`` and return its env dir."""
    if env_key == "local":
        assert overlay is not None
        SpaceURI.set_baseuris([f"file://{overlay}", f"file://{ASSETS}"], {})
        return overlay / "environments/skypilot-local/test-slurm"
    SpaceURI.set_baseuris([f"file://{ASSETS}"], {})
    return {
        "slurm": SKY / "slurm/bluevela",
        "lsf": SKY / "lsf/ibm-bluevela",
        "aws": SKY / "aws",
        "kubernetes": SKY / "kubernetes",
    }[env_key]


def _resolve(env_dir: Path, name: str) -> Optional[str]:
    """The resolved step dir relative to configurations/assets, or None."""
    with SpaceURI.with_current_env(_env(env_dir)):
        try:
            uri = URI.get_uri(f"space://steps/{name}", default_scheme="file", secrets={})
        except ValueError:
            return None
    path = Path(uri.uri.path)  # type: ignore[union-attr]
    return str(path.resolve().relative_to(ASSETS.resolve()))


@pytest.mark.parametrize("name", STEP_NAMES)
@pytest.mark.parametrize("env_key", sorted(EXPECTED))
def test_resolution(env_key: str, name: str, local_overlay: Path) -> None:
    env_dir = _setup(env_key, local_overlay)
    assert _resolve(env_dir, name) == EXPECTED[env_key][name]


@pytest.mark.parametrize("name", STEP_NAMES)
@pytest.mark.parametrize("env_key", ["slurm", "local"])
def test_slurm_never_gets_an_aws_variant(env_key: str, name: str, local_overlay):
    """The #664 bug, stated directly: no SLURM env resolves an AWS step."""
    env_dir = _setup(env_key, local_overlay)
    resolved = _resolve(env_dir, name)
    assert resolved is not None
    step = yaml.safe_load((ASSETS / resolved / "step.yaml").read_text(encoding="utf-8"))
    assert "aws/" not in resolved
    assert "aws" not in yaml.safe_dump(step["environment_configs"]).lower()
