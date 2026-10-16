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

"""Integration test: bfcl-eval runs end to end on Skypilot/LSF.

A **step-level** test, beside the step's Makefile in a per-cluster subdir with its
fixtures in the matching ``test-data/lsf/``. Run it from the step directory::

    GB_STEP_BLUEVELA_BUILD=1 make -C steps/bfcl-eval/skypilot test

The fixture is a single target: serve a pre-existing 3B granite checkpoint from /proj with
vLLM, generate the BFCL ``simple`` category, score it, register the score file. Unlike the
sibling distill-* fixtures it chains nothing, because this step's only input is a model
directory and one already on the cluster exercises everything the step owns.

**This test has never been executed.** It needs a published image — none has been built
from this step's Dockerfile (see the step's README.md) — and a GPU reservation. The
offline suite covers the config contract, the flag surface, the artifact contract and
``bash -n`` of the ``run`` block. Treat the first run as unproven work.

What it proves that the offline tests cannot: that the image starts under SkyPilot's
docker runtime (which runs ``apt-get`` *inside* the container during setup), that the vLLM
server became ready inside ``run-bfcl.sh``'s window and served the right checkpoint, that
``model_id`` resolved to a real handler rather than scoring with the wrong parser, that the
harness left exactly one score file where the step looks for it, and that the
``GB_ARTIFACT_ID`` line registered the dataset.

One H100 and one BFCL category, so the allocation costs minutes rather than hours.
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

# Opt-in switch for the cluster run. `make test` in the step directory invokes pytest with
# no marker filter, so without this a routine `make test` would submit a real LSF job to a
# shared cluster — costly to other users and easy to abandon half-scheduled. For this step
# `make test` also builds the image first (common.mk: `test: space image`), which makes the
# accidental-submission path longer but not narrower.
#
# The offline tests (test_step_template.py, test_run_bfcl.py, and the tool-call repair and
# id-selection tests) cover the step's contracts and always run; this one is deliberate:
#
#     GB_STEP_BLUEVELA_BUILD=1 make -C steps/bfcl-eval/skypilot test
#
# The repository suites gate it by marker instead (`-m "not ibm"`), so this does not change
# how CI selects it.
_OPT_IN = "GB_STEP_BLUEVELA_BUILD"


def _opted_in() -> bool:
    """Whether the operator asked for the cluster run."""
    return os.environ.get(_OPT_IN, "").lower() in ("1", "true", "yes")


def _bluevela_reachable() -> bool:
    """Whether the BlueVela login node answers over SSH.

    Probed only when opted in, so the common case costs no network round trip at
    collection time.

    This gate is not optional politeness: without it a plain ``make test`` in this step
    directory submits a real LSF job, and an interrupted run leaves it PEND-ing on a shared
    cluster holding a GPU allocation nobody awaits.

    ``IdentitiesOnly=yes`` matters — an agent holding several keys otherwise exhausts the
    server's auth attempts before offering the right one, and the resulting "Too many
    authentication failures" reads as an outage rather than a client-side problem.
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


# Real-infra build on a shared cluster: extended suite only.
@extended_testing_only
# Serialize with the other BlueVela build tests: they share one SSH auth path to the
# login node, and a shared account's concurrent submissions are the thing most likely
# to make a run look like an outage.
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
    reason=f"Set {_OPT_IN}=1 to run the real BlueVela build "
    "(the offline tests cover the step's contracts without a cluster)",
)
# And do not try from a machine that cannot reach the cluster.
@pytest.mark.skipif(
    not _bluevela_reachable(),
    reason="BlueVela login node not reachable over SSH (needs "
    f"{_BV_KEY} and network access to {_BV_HOST})",
)
class TestSkypilotLsfBfclEval(AbstractYamlBuildRunnerTest):
    """BFCL tool-calling evaluation on one BlueVela GPU, in this step's own image."""

    def _get_yaml_spec_dir(self) -> Path:
        # Fixtures resolve through the repo's test/ <-> test-data/ helper, which
        # keys off the first `test/` segment so this works from both homes of the
        # file (see steps/README.md, "Two test modes"):
        #   Mode 1 (authoring)  steps/bfcl-eval/skypilot/test/lsf/
        #       -> steps/bfcl-eval/skypilot/test-data/lsf/   (co-located)
        #   Mode 2 (published)  test/steps/bfcl-eval/skypilot/lsf/
        #       -> test-data/steps/bfcl-eval/skypilot/lsf/   (parallel tree)
        return get_test_data_dir_for(__file__)
