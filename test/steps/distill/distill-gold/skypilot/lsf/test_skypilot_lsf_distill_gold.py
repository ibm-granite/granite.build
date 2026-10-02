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

"""Integration test: distill-gold runs multi-node end to end on Skypilot/LSF.

A **step-level** test, living beside the distill-gold step's Makefile in a
per-cluster subdir (``steps/distill/distill-gold/skypilot/test/lsf/``) with its fixtures
in the matching ``test-data/lsf/``. Run it from the step directory::

    make -C steps/distill/distill-gold/skypilot test

``make test`` depends on ``make space``, so the git-ignored ``space/`` directory
the fixture's ``space_uri`` points at is always rendered first; this test does no
rendering of its own.

What it proves that the unit tests cannot. The renderer and template tests cover
the config contract offline; only a real run shows that the two nodes each got
their own rank, that the config rendered *inside* the container against the /proj
checkout, and that rank 0 alone registered the checkpoint. The fixture asks for
``num_nodes: 2`` with a small model pair so it costs minutes rather than hours.

Requirements beyond the usual: a SkyPilot install with LSF multi-node support
(``sky.skylet.executor.lsf``). Without it gbserver refuses the launch outright
rather than running single-node and reporting success, so a missing dependency
surfaces as a launch error naming the pin, not as a silently wrong run.
"""

import os
import subprocess
from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

pytestmark = pytest.mark.ibm

# Environment defaults, mirroring
# configurations/assets/environments/skypilot/lsf/ibm-bluevela/environment.yaml.
_BV_HOST = os.environ.get("BV_SSH_HOST", "login4.bluevela.rmf.ibm.com")
_BV_USER = os.environ.get("BV_SSH_USER", "granitebuild")
_BV_KEY = os.environ.get("BV_SSH_KEY", "~/.ssh/ibm-bluevela.key")

# Opt-in switch for the cluster run. `make test` in the step directory invokes
# pytest with no marker filter, so without this a routine `make test` would
# submit a real two-node LSF job to a shared cluster — slow, costly to other
# users, and easy to abandon half-scheduled. The offline tests
# (test_render_gold_config.py, test_step_template.py) cover the step's contracts
# and always run; this one is deliberate:
#
#     GB_STEP_BLUEVELA_BUILD=1 make -C steps/distill/distill-gold/skypilot test
#
# The repository suites gate it by marker instead (`-m "not ibm"`), so this does
# not change how CI selects it.
_OPT_IN = "GB_STEP_BLUEVELA_BUILD"


def _opted_in() -> bool:
    """Whether the operator asked for the cluster run."""
    return os.environ.get(_OPT_IN, "").lower() in ("1", "true", "yes")


def _bluevela_reachable() -> bool:
    """Whether the BlueVela login node answers over SSH.

    Probed only when opted in, so the common case costs no network round trip at
    collection time.

    This gate is not optional politeness: without it a plain ``make test`` in
    this step directory submits a real two-node LSF job, and an interrupted run
    leaves it PEND-ing on a shared cluster holding an allocation nobody awaits.

    ``IdentitiesOnly=yes`` matters — an agent holding several keys otherwise
    exhausts the server's auth attempts before offering the right one, and the
    resulting "Too many authentication failures" reads as an outage rather than a
    client-side problem.
    """
    if not _opted_in():
        return False
    key = os.path.expanduser(_BV_KEY)
    if not os.path.exists(key):
        return False
    try:
        result = subprocess.run(
            [
                "ssh",
                "-i",
                key,
                "-o",
                "BatchMode=yes",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=10",
                f"{_BV_USER}@{_BV_HOST}",
                "command -v bsub",
            ],
            capture_output=True,
            timeout=25,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


# Real-infra multi-node build on a shared cluster: extended suite only.
@extended_testing_only
# Serialize with the other BlueVela build tests. Beyond the usual SSH-auth
# concurrency, this one holds a 2-node allocation, so overlapping runs would
# queue behind each other and risk the timeout.
@pytest.mark.xdist_group(name="buildtest_bv")
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="Needs environments/skypilot/lsf/bluevela/environment.yaml with a "
    "BV_SSH_PRIVATE_KEY reference in the space repos; locally this uses the "
    "local space and ~/.ssh/ibm-bluevela.key",
)
# Never submit a cluster job as a side effect of running the step's tests.
@pytest.mark.skipif(
    not _opted_in(),
    reason=f"Set {_OPT_IN}=1 to run the real two-node BlueVela build "
    "(the offline tests cover the step's contracts without a cluster)",
)
# And do not try from a machine that cannot reach the cluster.
@pytest.mark.skipif(
    not _bluevela_reachable(),
    reason="BlueVela login node not reachable over SSH (needs "
    f"{_BV_KEY} and network access to {_BV_HOST})",
)
class TestSkypilotLsfGoldDistill(AbstractYamlBuildRunnerTest):
    """GOLD distillation across two BlueVela nodes, via enroot."""

    def _get_yaml_spec_dir(self) -> Path:
        # Fixtures resolve through the repo's test/ <-> test-data/ helper, which
        # keys off the first `test/` segment so this works from both homes of the
        # file (see steps/README.md, "Two test modes"):
        #   Mode 1 (authoring)  steps/distill/distill-gold/skypilot/test/lsf/
        #       -> steps/distill/distill-gold/skypilot/test-data/lsf/   (co-located)
        #   Mode 2 (published)  test/steps/distill/distill-gold/skypilot/lsf/
        #       -> test-data/steps/distill/distill-gold/skypilot/lsf/   (parallel tree)
        return get_test_data_dir_for(__file__)
