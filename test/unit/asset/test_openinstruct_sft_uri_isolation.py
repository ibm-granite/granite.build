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

"""The legacy `space://steps/openinstruct-sft` URI never resolves to the new step.

Two SFT steps share a leaf name: the hand-authored legacy assets at
skypilot/{lsf/ibm-bluevela,aws}/steps/openinstruct-sft (fork-based, used by the
granite4-350m recipes) and the framework step skypilot/steps/train/openinstruct-sft
(#229). Resolution pins the longest steps/-relative name whose step.yaml exists,
so the legacy URI must keep binding to a legacy asset and the new URI to the new
step, on every SkyPilot environment. Runs the real resolver against the real
configurations/assets tree.
"""

import pathlib

import pytest

from gbcommon.uri.space import SpaceURI
from gbcommon.uri.uri import URI

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ASSETS = REPO_ROOT / "configurations" / "assets"
SKY = ASSETS / "environments" / "skypilot"
NEW_STEP = SKY / "steps" / "train" / "openinstruct-sft"
LEGACY = {
    SKY / "lsf" / "ibm-bluevela" / "steps" / "openinstruct-sft",
    SKY / "aws" / "steps" / "openinstruct-sft",
}
ENVS = {
    "lsf-bluevela": (SKY / "lsf" / "ibm-bluevela", "lsf"),
    "aws": (SKY / "aws", None),
    "slurm-bluevela": (SKY / "slurm" / "ibm-bluevela", "slurm"),
    "kubernetes-local": (SKY / "kubernetes" / "local", "kubernetes"),
}
_TL_ATTRS = (
    "base_uris",
    "space_secrets",
    "current_env_dir_uri",
    "current_env_class_name",
    "current_env_subtype",
)


@pytest.fixture(autouse=True)
def _isolate_thread_local():
    tl = SpaceURI._thread_local
    saved = {a: getattr(tl, a, None) for a in _TL_ATTRS}
    SpaceURI.set_baseuris([f"file://{ASSETS}"], {})
    try:
        yield
    finally:
        for attr, value in saved.items():
            if value is None:
                if hasattr(tl, attr):
                    delattr(tl, attr)
            else:
                setattr(tl, attr, value)


def _resolve(uri: str, env: str) -> pathlib.Path:
    env_dir, subtype = ENVS[env]
    with SpaceURI.with_current_env_class_name(
        "Skypilot", env_dir_uri=f"file://{env_dir}", env_subtype=subtype
    ):
        resolved = URI.get_uri(uri, default_scheme="file", secrets={})
    assert resolved.uri is not None, f"{uri} did not resolve to a path"
    return pathlib.Path(resolved.uri.path).resolve()


def test_new_step_is_published():
    assert (NEW_STEP / "step.yaml").is_file()


@pytest.mark.parametrize("env", sorted(ENVS))
def test_new_uri_resolves_to_the_new_step(env):
    assert _resolve("space://steps/train/openinstruct-sft", env) == NEW_STEP.resolve()


@pytest.mark.parametrize("env", sorted(ENVS))
def test_legacy_uri_never_resolves_to_the_new_step(env):
    assert _resolve("space://steps/openinstruct-sft", env) in {
        p.resolve() for p in LEGACY
    }


@pytest.mark.parametrize("env", ["lsf-bluevela", "aws"])
def test_legacy_uri_keeps_its_env_colocated_asset(env):
    expected = ENVS[env][0] / "steps" / "openinstruct-sft"
    assert _resolve("space://steps/openinstruct-sft", env) == expected.resolve()
