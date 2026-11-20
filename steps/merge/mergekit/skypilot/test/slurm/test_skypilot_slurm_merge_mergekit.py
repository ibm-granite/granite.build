# SPDX-License-Identifier: Apache-2.0

"""mergekit on slurm: one merge on the local Docker SLURM cluster reaches SUCCESS.

Reaching SUCCESS proves the bare-node uv install of MergeKit and the merge itself,
with the merged model on the shared /shared volume. Gated on a reachable Docker
SLURM cluster, so it auto-skips in CI and on machines without one (``make
test-setup`` brings it up). Extended suite only. Fixtures live in the test-data/ dir mirroring this file
(get_test_data_dir_for), so the pairing holds in both test modes.
"""

from pathlib import Path

import pytest
from integration.environment.test_skypilot_slurm_e2e import _slurm_cluster_reachable
from integration.ibm.buildrunner.skypilot.slurm_local.cluster import remove_from_shared
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only
from libgbtest.shared_paths import step_output_dir

pytestmark = pytest.mark.skypilot_integration


#: This test's fixture (also read for the output path to clear).
_BUILD_YAML = get_test_data_dir_for(__file__) / "build.yaml"


@extended_testing_only
@pytest.mark.skipif(
    not _slurm_cluster_reachable(),
    reason="Docker SLURM cluster not reachable (run: make test-setup)",
)
class TestSkypilotSlurmMergekit(AbstractYamlBuildRunnerTest):
    """mergekit step, end to end on slurm."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)

    @pytest.fixture(autouse=True)
    def _fresh_output(self):
        """Clear this test's ``/shared/mergekit-test/<test>`` dir before it runs.

        It holds the model, its retry record and any staging dir; a re-run would
        otherwise be refused. The path comes from build.yaml.
        """
        remove_from_shared(
            step_output_dir(_BUILD_YAML, "merge", 0, "mergekit_config", "output_path")
        )
