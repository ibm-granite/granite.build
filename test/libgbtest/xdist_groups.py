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

"""One xdist group for every test that drives a real SkyPilot environment.

They share one local SkyPilot API server and ``~/.sky/config.yaml``, which
gbserver rewrites per launch with the launching env's ``cloud_config`` (last
writer wins). Concurrently, a kube launch drops the slurm env's ``workdir``
and the slurm build's log download finds nothing. The server ignores client
overrides of ``slurm.cluster_configs``, so per-test client configs don't help.
"""

import pytest

SKYPILOT_XDIST_GROUP_NAME = "skypilot"
SKYPILOT_XDIST_GROUP = pytest.mark.xdist_group(name=SKYPILOT_XDIST_GROUP_NAME)


def group_skypilot_integration_tests(
    items: list[pytest.Item], strict: bool = False
) -> None:
    """Put every ``skypilot_integration`` item in :data:`SKYPILOT_XDIST_GROUP`.

    :param items: the collected items (modified in place).
    :param strict: raise on an item already in another group, which xdist would
        join into a separate, still-concurrent group. Only meaningful under
        ``--dist=loadgroup``; elsewhere groups are ignored.
    :raises pytest.UsageError: ``strict`` and an item is in another group.
    """
    for item in items:
        if item.get_closest_marker("skypilot_integration") is None:
            continue
        for mark in item.iter_markers("xdist_group"):
            name = mark.args[0] if mark.args else mark.kwargs.get("name", "default")
            if strict and name != SKYPILOT_XDIST_GROUP_NAME:
                raise pytest.UsageError(
                    f"{item.nodeid}: skypilot_integration tests must use xdist "
                    f"group '{SKYPILOT_XDIST_GROUP_NAME}', not '{name}'"
                )
        item.add_marker(SKYPILOT_XDIST_GROUP)
