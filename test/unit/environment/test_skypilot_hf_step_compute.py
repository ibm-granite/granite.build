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

"""The kube envs size the builtin hf steps through ``config.steps`` (#495).

The builtin skypilot ``hfpull``/``hfpush`` step.yamls request
``total_memory_per_node: 10Gi``. ``config.steps.<type>`` is the lowest layer of
the step-config merge, so it cannot lower that ``compute_config`` value; but the
step.yamls set no ``launcher_config``, and at launch
``launcher_config.resources`` overrides the ``compute_config`` floor. These tests
follow both layers with the real step.yamls and environment.yamls.
"""

import asyncio
from pathlib import Path

import pytest
import yaml

from gbserver.build.targetstep import _seed_step_config_with_env_defaults
from gbserver.utils.filesystem import merge_dicts

_REPO = Path(__file__).resolve().parents[3]
_KUBE_ENVS = ["kubernetes", "kubernetes/local"]
_SMALL = {"cpus": "1+", "memory": "2+"}


def _env_config(env):
    path = _REPO / f"configurations/assets/environments/skypilot/{env}/environment.yaml"
    return yaml.safe_load(path.read_text())["config"]


def _step_config(step):
    path = _REPO / f"src/gbserver/builtins/steps/skypilot/{step}/step.yaml"
    return yaml.safe_load(path.read_text())["config"]


def _merged(env, step):
    """The step's merged config: env config.steps <- step.yaml <- queued config.

    Mirrors merge_handle_configs (targetstep.py), which needs a full TargetStep
    to drive; the queued layer is what pull/pushasset_hfstore pass.
    """
    seeded = _seed_step_config_with_env_defaults(
        _env_config(env), f"space://steps/{step}", {}
    )
    merged = merge_dicts(seeded, _step_config(step))
    queued = {"launcher_config": {"envs": {"HF_TOKEN": "tok"}}}
    return merge_dicts(merged, queued)


@pytest.mark.parametrize("env", _KUBE_ENVS)
@pytest.mark.parametrize("step", ["hfpull", "hfpush"])
def test_kube_env_sizes_the_hf_step(env, step):
    merged = _merged(env, step)
    assert merged["launcher_config"]["resources"] == _SMALL
    # The queued HF_TOKEN still reaches the step alongside it.
    assert merged["launcher_config"]["envs"] == {"HF_TOKEN": "tok"}
    # compute_config is untouched: the 10Gi floor stays, and is overridden below.
    assert merged["compute_config"]["total_memory_per_node"] == "10Gi"


@pytest.mark.parametrize("step", ["hfpull", "hfpush"])
def test_pod_request_is_the_override_not_the_floor(step):
    """At launch, launcher_config.resources wins over the compute_config floor."""
    from gbserver.environment.skypilot import Skypilot
    from gbserver.types.environmentconfig import EnvironmentConfig

    env = Skypilot(
        event_q=asyncio.Queue(),
        environment_config=EnvironmentConfig(
            name="k", type="Skypilot", config={"default_cloud": "kubernetes"}
        ),
    )
    merged = _merged("kubernetes/local", step)
    floor = env._resources_from_compute_config(
        merged["compute_config"], cloud="kubernetes"
    )
    assert floor["memory"] == "10+"
    # launch_skypilot builds sky.Resources from {**floor, **override_res}.
    resources = {**floor, **merged["launcher_config"]["resources"]}
    assert resources["memory"] == "2+" and resources["cpus"] == "1+"


def test_builtin_steps_set_no_launcher_resources():
    """If they did, it would outrank config.steps and the kube sizing would be lost."""
    for step in ("hfpull", "hfpush"):
        assert "launcher_config" not in _step_config(step), step
