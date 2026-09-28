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

"""Unit tests for Target.push_assets (#390).

Push destinations now resolve at push time on the dispatched path, so
``push_assets`` does no setup-time work: it is a deprecated no-op returning
``{}`` for every output shape. The former setup-time resolution rejected two
output shapes (a ``{{ binding.* }}``-dependent destination URI and a glob
output key); those restrictions are gone.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from gbserver.asset.hfstore import Hfstore
from gbserver.build.target import Target
from gbserver.types.buildconfig import BuildTargetConfig, BuildTargetOutputConfig
from gbserver.types.buildevent import EntityRunMetadata


def _out(uri):
    return BuildTargetOutputConfig(uri=uri)


def _make_target(outputs):
    """Build a bare Target wired with the given outputs and a mocked env.

    ``push_assets`` is now a no-op, so it never touches the environment; the
    mock is attached only so attribute access does not blow up.
    """
    target = object.__new__(Target)
    target.name = "t"
    target.config = BuildTargetConfig(
        environment_uri="env:///skypilot",
        outputs=outputs,
        steps=[],
    )
    target.environment = MagicMock()
    return target


def make_target(outputs):
    return _make_target(outputs)


def test_binding_dependent_push_uri_is_no_longer_rejected():
    # A destination URI that references the produced artifact used to raise at
    # setup; resolution is deferred to push time, so it must NOT raise now.
    t = make_target({"model": _out(uri="hf://o/{{ binding.path | short_hash }}")})
    assert t.push_assets() == {}


def test_glob_output_key_is_no_longer_rejected():
    # A glob output key used to raise at setup; it must NOT raise now.
    t = make_target({"model-*": _out(uri="hf://o/r")})
    assert t.push_assets() == {}


def test_push_assets_is_noop_for_plain_output():
    t = make_target({"model_out": _out(uri="hf:///ibm-granite/granite-4.0-h-350m")})
    assert t.push_assets() == {}


def test_push_assets_empty_when_no_outputs():
    target = object.__new__(Target)
    target.config = BuildTargetConfig(environment_uri="env:///skypilot", steps=[])
    target.environment = MagicMock()
    assert target.push_assets() == {}


# ---------------------------------------------------------------------------
# Multi-output regression lock (#390): N declared outputs must each become
# their own dispatched push step, one per output, each carrying its own
# ``binding_id``. This reproduces @toraponibm's two-output case where the old
# inline-push seam collapsed multiple outputs into a single push and dropped
# artifacts. It is a regression lock only -- no production code should change.
# ---------------------------------------------------------------------------


def _run_metadata() -> EntityRunMetadata:
    return EntityRunMetadata(build_id="build-multi-output")


def _hfstore_mock(token: str = "tok-abc") -> MagicMock:
    """An Hfstore mock (Enterprise org so resource-group resolution runs)."""
    store = MagicMock(spec=Hfstore)
    store.type = "Hfstore"
    store.resolve_token.return_value = token
    store.get_secrets.return_value = {}
    store.get_enterprise_organizations.return_value = ["myorg"]
    return store


@pytest.fixture
def hf_env():
    """A real Skypilot env whose hf:// store resolution is stubbed.

    ``Environment.pushasset`` resolves the URI to a declared assetstore via
    ``_get_storeconfig``; a bare env has none, so it is stubbed to an Hfstore
    mock. ``_shared_fs_provider_cache`` is primed truthy so the Task 2 preflight
    guard (hfstore push requires a shared_filesystem on SkyPilot/AWS) is
    satisfied and ``pushasset_hfstore`` queues a real step instead of raising.
    The event queue and per-thread asset-event map are the env's real objects.
    """
    from gbserver.environment.environment import Environment
    from gbserver.environment.skypilot import Skypilot
    from gbserver.types.environmentconfig import EnvironmentConfig

    env = Skypilot(
        event_q=asyncio.Queue(),
        environment_config=EnvironmentConfig(
            name="test-skypilot",
            type="Skypilot",
            config={"default_cloud": "k8s", "idle_minutes_to_autostop": 0},
        ),
    )
    env._shared_fs_provider_cache = MagicMock(name="shared_filesystem_provider")
    if not hasattr(Environment._thread_local, "asset_events"):
        Environment._thread_local.asset_events = {}

    assetstore = _hfstore_mock()
    assetstoreenv_config = MagicMock()
    assetstoreenv_config.push = None
    env._get_storeconfig = MagicMock(  # type: ignore[method-assign]
        return_value=(assetstore, assetstoreenv_config)
    )
    return env


@pytest.fixture
def two_output_bindings():
    """Two declared outputs, keyed by binding_id, each with its own dest URI.

    Mirrors a build.yaml declaring two outputs (a base model and an adapter):
    each maps to a distinct hf:// destination, exactly the two-output shape
    that previously dropped one artifact.
    """
    return {
        "model": ({"path": "/workspace/output/model"}, "hf:///myorg/model-repo"),
        "adapter": ({"path": "/workspace/output/adapter"}, "hf:///myorg/adapter-repo"),
    }


@pytest.mark.asyncio
async def test_two_outputs_produce_two_push_steps(hf_env, two_output_bindings):
    """Two declared outputs -> two queued push steps, each with its own binding_id.

    Regression lock: enqueue a ``pushasset`` per output onto one shared
    ``additional_targetsteps_queue``, let the async push tasks run, then assert
    exactly two ``BuildTargetStepConfig``s land -- one per output -- and that
    their ``hfpush_config.binding_id`` values are the two distinct output keys.
    A fan-out regression (outputs collapsed into a single push) would yield
    fewer than two configs or a duplicated binding_id.
    """
    q: asyncio.Queue = asyncio.Queue()
    tasks = []
    # The Enterprise resource-group resolver's innermost space/HF lookup is
    # patched so the test never touches storage or the HF API.
    with patch(
        "gbserver.spaces.hf_push_config.resolve_space_resource_group_id",
        return_value=None,
    ):
        for binding_id, (binding, uri) in two_output_bindings.items():
            tasks.append(
                hf_env.pushasset(
                    task_group=None,
                    binding=binding,
                    uristr=uri,
                    binding_id=binding_id,
                    additional_targetsteps_queue=q,
                    run_metadata=_run_metadata(),
                )
            )
        await asyncio.gather(*tasks)  # let the pushasset_as_uri tasks run

    configs = [q.get_nowait() for _ in range(q.qsize())]
    assert len(configs) == 2
    assert {c.config["hfpush_config"]["binding_id"] for c in configs} == {
        "model",
        "adapter",
    }
