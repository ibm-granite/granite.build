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

"""data-designer against RITS on the local Docker SLURM cluster, output KEPT.

The ``test_data_designer_rits.py`` generation with an explicit ``output_path`` on
``/shared``, outside the per-run workdir that teardown removes. After the build reaches
SUCCESS the test reads the dataset back over the cluster's SSH and checks it: the jsonl
export has one non-empty ``question`` per row, the parquet batches are the 3 row groups
that ``run_config.buffer_size: 2`` produces, and the RITS key appears in none of the
files. The buildtest.yaml also requires the step's lineage metadata:
``records_generated: 6`` and ``resumed_records: 0`` (a fresh run, nothing reused).

The rows stay on the cluster afterwards, at ``/shared/data-designer-test/slurm-rits-keep``,
for a person to look at. The path is emptied before each run, so a pass is never
satisfied by a previous run's data.

**IBM-only, two gates**, as for ``test_data_designer_rits.py``: the Docker SLURM
cluster must be reachable (``make slurm-setup``) and ``RITS_API_KEY`` exported (also
needed as the space secret ``RITS_API_KEY``; see the step's README.md, "IBM-only build
tests").
"""

import json
import os
from pathlib import Path

import pytest
from integration.environment.test_skypilot_slurm_e2e import _slurm_cluster_reachable
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

from .cluster import files_containing, on_cluster

pytestmark = [pytest.mark.ibm, pytest.mark.skypilot_integration]

# Must match output_path in the fixture's build.yaml.
_OUTPUT = "/shared/data-designer-test/slurm-rits-keep"
_NUM_RECORDS = 6
_BUFFER_SIZE = 2


@extended_testing_only
@pytest.mark.skipif(
    not _slurm_cluster_reachable(),
    reason="Docker SLURM cluster not reachable (run: make slurm-setup)",
)
@pytest.mark.skipif(
    not os.environ.get("RITS_API_KEY"),
    reason="RITS_API_KEY not exported (the build also needs it as a space secret)",
)
class TestSkypilotSlurmDataDesignerRitsKeep(AbstractYamlBuildRunnerTest):
    """data-designer step: a live RITS generation whose rows outlive the build."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "data-designer-rits-keep"

    def test_runner(self):
        # Empty the fixed path first: the fixture uses resume: never, which refuses a
        # non-empty output_path, and a stale dataset must never satisfy the checks.
        on_cluster(f"rm -rf {_OUTPUT}")
        super().test_runner()

        files = on_cluster(f"cd {_OUTPUT} && find . -type f | sort").split()
        batches = [f for f in files if f.startswith("./parquet-files/")]
        assert len(batches) == _NUM_RECORDS // _BUFFER_SIZE, files
        assert {"./metadata.json", "./builder_config.json"} <= set(files), files

        jsonl = on_cluster(f"cat {_OUTPUT}/slurm-rits-keep.jsonl")
        rows = [json.loads(line) for line in jsonl.splitlines()]
        assert len(rows) == _NUM_RECORDS
        for row in rows:
            assert row["topic"] in {"astronomy", "cooking", "geology"}, row
            assert row["question"].strip(), row
        print(f"\n{_OUTPUT}/slurm-rits-keep.jsonl:\n{jsonl}")

        key = os.environ["RITS_API_KEY"]
        leaked = files_containing(key, _OUTPUT)
        assert not leaked, f"RITS key found in: {leaked}"
