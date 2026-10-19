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

"""Integration test: distill-hf-export runs end to end on Skypilot/LSF.

A **step-level** test, beside the step's Makefile in a per-cluster subdir with its
fixtures in the matching ``test-data/lsf/``. Run it from the step directory::

    GB_STEP_BLUEVELA_BUILD=1 make -C steps/distill/hf-export/skypilot test

What it proves that the offline tests cannot. test_step_template.py covers the config
contract and the flag surface, and the 31 ported unit tests cover classification and the
manifest against synthetic directories — neither has a real DeepSpeed checkpoint to
classify. Only a real run shows that a named checkpoint was selected out of nine, that
the ``global_step*``/optimizer/RNG state was left behind, that ``padding_side`` was
normalised in the published copy, and that the result loads with AutoConfig/AutoTokenizer
before being registered.

It reads a trainer output dir that already exists on /proj, so it costs no training run.
The step only copies OUT of that directory, so a shared-account checkpoint is safe input.
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
# submit a real LSF job to a shared cluster — costly to other users and easy to
# abandon half-scheduled. The offline tests (test_step_template.py, plus the
# ported test_export_hf_model.py when a checkout is present) cover the step's
# contracts and always run; this one is deliberate:
#
#     GB_STEP_BLUEVELA_BUILD=1 make -C steps/distill/hf-export/skypilot test
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
    this step directory submits a real LSF job, and an interrupted run
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
class TestSkypilotLsfDistillHfExport(AbstractYamlBuildRunnerTest):
    """GOLD distillation across two BlueVela nodes, via enroot."""

    def _get_yaml_spec_dir(self) -> Path:
        # Fixtures resolve through the repo's test/ <-> test-data/ helper, which
        # keys off the first `test/` segment so this works from both homes of the
        # file (see steps/README.md, "Two test modes"):
        #   Mode 1 (authoring)  steps/distill/hf-export/skypilot/test/lsf/
        #       -> steps/distill/hf-export/skypilot/test-data/lsf/   (co-located)
        #   Mode 2 (published)  test/steps/distill/hf-export/skypilot/lsf/
        #       -> test-data/steps/distill/hf-export/skypilot/lsf/   (parallel tree)
        return get_test_data_dir_for(__file__)
