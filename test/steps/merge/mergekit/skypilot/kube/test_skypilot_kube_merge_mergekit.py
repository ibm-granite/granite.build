# SPDX-License-Identifier: Apache-2.0

"""mergekit on kube: one merge in a pod on the local kind cluster reaches SUCCESS.

Reaching SUCCESS proves the uv install of MergeKit in a pod and the merge itself,
with the merged model on the shared /shared volume. Gated on a reachable kind
cluster (``make test-setup-kube`` brings it up). Extended suite only. Fixtures live
in the test-data/ dir mirroring this file (get_test_data_dir_for), so the pairing
holds in both test modes.
"""

import os
from pathlib import Path

import pytest
import yaml
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only
from libgbtest.kube import (
    KUBE_XDIST_GROUP,
    kind_cluster_reachable,
    kind_skip_reason,
    remove_from_shared,
)

pytestmark = pytest.mark.skypilot_integration


@KUBE_XDIST_GROUP
@extended_testing_only
@pytest.mark.skipif(not kind_cluster_reachable(), reason=kind_skip_reason())
class TestSkypilotKubeMergekit(AbstractYamlBuildRunnerTest):
    """mergekit step, end to end on kube."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)

    @pytest.fixture(autouse=True)
    def _fresh_output(self):
        """Remove what an earlier run left, so the step doesn't refuse the output.

        Removes the output's parent, this test's own ``/shared/mergekit-test/<test>``
        directory, which holds the model, its retry record and any staging dir. The
        path comes from build.yaml, so the two can't drift.
        """
        build = yaml.safe_load(
            (get_test_data_dir_for(__file__) / "build.yaml").read_text()
        )
        steps = build["granite.build"]["targets"]["merge"]["steps"]
        output = steps[0]["config"]["mergekit_config"]["output_path"]
        remove_from_shared(os.path.dirname(output))
