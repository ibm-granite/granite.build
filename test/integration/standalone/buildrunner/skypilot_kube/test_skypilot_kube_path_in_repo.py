# SPDX-License-Identifier: Apache-2.0

"""Two hf:// inputs naming different files of one repo, on SkyPilot/kubernetes.

Runs on ``space://environments/skypilot/kubernetes/local``: each input is pulled
by its own hfpull step (a separate pod) into the same dir on the shared volume,
downloading only its file, and the command step (another pod) checks that each
input is bound at its own file.

Requires the local kind cluster and its shared volume (``make kube-setup``);
auto-skips otherwise.

The fixture's build.yaml and buildtest.yaml live in the directory returned by
_get_yaml_spec_dir below.
"""

from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only
from libgbtest.kube import (
    KUBE_XDIST_GROUP,
    kind_cluster_reachable,
    kind_skip_reason,
)

pytestmark = pytest.mark.skypilot_integration


# Real-infra build test (launches pods via SkyPilot, and a REAL HuggingFace
# pull) — only run in the extended suite (make extended-tests).
@KUBE_XDIST_GROUP
@extended_testing_only
@pytest.mark.skipif(not kind_cluster_reachable(), reason=kind_skip_reason())
class TestSkypilotKubePathInRepo(AbstractYamlBuildRunnerTest):
    """Two files of one hf:// repo, each bound at its own file in a pod."""

    def _get_yaml_spec_dir(self) -> Path:
        """The fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "path-in-repo"
