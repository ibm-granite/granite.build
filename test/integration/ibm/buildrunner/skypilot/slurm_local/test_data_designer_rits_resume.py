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

"""data-designer resume across builds, against RITS on the local Docker SLURM cluster.

One test, two builds of the same build.yaml against one fixed ``output_path`` on
``/shared``:

1. **Interrupted.** The build runs until at least two row groups are checkpointed on
   the node, then is cancelled through the harness's own cancellation path AND its
   SLURM job is ``scancel``ed — cancelling the build alone leaves the job running (see
   ``_wait_for_running_then_cancel_one_build``). The test
   then requires the dataset to be INCOMPLETE and to stop growing — a run that
   finished before the cancel reached it would prove nothing, so that fails loudly as
   inconclusive rather than passing.
2. **Resumed.** The unchanged build is submitted again. It must reach SUCCESS with all
   200 rows, record ``resumed_records`` > 0 (the step's count of checkpointed rows it
   kept), and leave every batch written by the first build byte-identical — so those
   rows were reused, not regenerated.

The harness cancels a build as soon as it is RUNNING, which is before the install and
long before any checkpoint, so this test overrides that one hook
(``_wait_for_running_then_cancel_one_build``) to wait for checkpoints on the node first.

**IBM-only, two gates**, as for ``test_data_designer_rits.py``: the Docker SLURM
cluster must be reachable (``make slurm-setup``) and ``RITS_API_KEY`` exported (also
needed as the space secret ``RITS_API_KEY``; see the step's README.md, "IBM-only build
tests").
"""

import os
import time
from pathlib import Path

import pytest
from integration.environment.test_skypilot_slurm_e2e import _slurm_cluster_reachable
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    ClassTestedEnum,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

from .cluster import files_containing, on_cluster

pytestmark = [pytest.mark.ibm, pytest.mark.skypilot_integration]

# Must match the fixture's build.yaml.
_OUTPUT = "/shared/data-designer-test/slurm-rits-resume"
_NUM_RECORDS = 200
_BUFFER_SIZE = 2
_TOTAL_BATCHES = _NUM_RECORDS // _BUFFER_SIZE
# Cancel once this many row groups are checkpointed.
_CANCEL_AFTER_BATCHES = 2


def _batches() -> dict[str, str]:
    """Checkpointed batch file -> sha256, as they stand on the node now."""
    out = on_cluster(
        f"cd {_OUTPUT}/parquet-files 2>/dev/null && sha256sum batch_*.parquet || true"
    )
    return {
        name: digest for digest, name in (line.split() for line in out.splitlines())
    }


@extended_testing_only
@pytest.mark.skipif(
    not _slurm_cluster_reachable(),
    reason="Docker SLURM cluster not reachable (run: make slurm-setup)",
)
@pytest.mark.skipif(
    not os.environ.get("RITS_API_KEY"),
    reason="RITS_API_KEY not exported (the build also needs it as a space secret)",
)
class TestSkypilotSlurmDataDesignerRitsResume(AbstractYamlBuildRunnerTest):
    """data-designer step: a cancelled build's checkpoints are reused by the next."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__) / "data-designer-rits-resume"

    def _wait_for_running_then_cancel_one_build(self, build_id: str, timeout: float):
        """Interrupt only once checkpoints exist — the harness's own hook cancels at RUNNING.

        Cancelling the BUILD alone does not stop the work on the node (measured
        2026-10-20, twice): gbserver marks it CANCELLED and runs ``sky.down``, yet the
        SLURM job ran to completion — every row group was written after "Torn down", and
        sacct recorded the job COMPLETED, not CANCELLED. The cause is in the pinned
        SkyPilot fork, which lacks an upstream SkyPilot fix (see
        https://github.com/ibm-granite/granite.build/issues/533). So after requesting
        the build cancellation the test also ``scancel``s the step's SLURM job, which is
        what a preemption or node failure does to a real run anyway. Drop the scancel
        once that issue is fixed.

        ``GBTEST_DD_BUILD_CANCEL_ONLY=true`` skips the scancel, so the build's own
        cancellation must stop the job — the check for a SkyPilot carrying the fix.
        """
        deadline = time.monotonic() + timeout
        while len(_batches()) < _CANCEL_AFTER_BATCHES:
            assert time.monotonic() < deadline, (
                f"no {_CANCEL_AFTER_BATCHES} checkpointed row groups in {_OUTPUT} "
                f"within {timeout:.0f}s; the generation never started"
            )
            time.sleep(1)
        self._request_build_cancellation(build_id)
        if os.environ.get("GBTEST_DD_BUILD_CANCEL_ONLY", "").lower() == "true":
            print("\nbuild cancellation only; no scancel")
            return
        # The step's job is the only running `gb-skypilot-slurm-data-generate-*` one;
        # `-td-` jobs are teardown helpers.
        jobs = on_cluster(
            "squeue -h -t RUNNING -o '%i %j' "
            "| awk '$2 ~ /^gb-skypilot-slurm-data-generate-/ && $2 !~ /-td-/ {print $1}'"
        ).split()
        assert jobs, "no running step job to interrupt; the generation already ended"
        on_cluster(f"scancel {' '.join(jobs)}")
        print(f"\nscancel'd step job(s): {jobs}")

    @pytest.mark.timeout(4200)  # two builds, each up to buildtest.yaml's 30 minutes
    def test_runner(self):
        spec = self._get_test_specification()
        on_cluster(f"rm -rf {_OUTPUT}")

        # 1. Interrupted.
        self._run_build_test(
            tested_class=ClassTestedEnum.TEST_BUILDRUNNER,
            test_spec=spec,
            test_cancel=True,
        )
        first = _batches()
        time.sleep(10)
        assert _batches() == first, (
            "checkpoints still changing after the build was CANCELLED — the "
            "generation kept running on the node"
        )
        assert _CANCEL_AFTER_BATCHES <= len(first) < _TOTAL_BATCHES, (
            f"inconclusive: {len(first)} of {_TOTAL_BATCHES} row groups were written "
            f"before the cancel took effect, so there is nothing left to resume. Make "
            f"the generation slower (more num_records) rather than accepting this."
        )
        print(
            f"\ncancelled with {len(first)} of {_TOTAL_BATCHES} row groups checkpointed"
        )

        # 2. Resumed. The spec's expectations (records_generated 200, resumed_records
        # > 0) are checked against this run by the harness.
        self._run_build_test(
            tested_class=ClassTestedEnum.TEST_BUILDRUNNER,
            test_spec=spec,
            test_cancel=False,
        )
        final = _batches()
        assert len(final) == _TOTAL_BATCHES, sorted(final)
        changed = [name for name, digest in first.items() if final.get(name) != digest]
        assert not changed, f"resume regenerated checkpointed row groups: {changed}"

        rows = on_cluster(f"wc -l < {_OUTPUT}/slurm-rits-resume.jsonl").strip()
        assert rows == str(_NUM_RECORDS), rows
        key = os.environ["RITS_API_KEY"]
        leaked = files_containing(key, _OUTPUT)
        assert not leaked, f"RITS key found in: {leaked}"
        print(
            f"resumed: kept {len(first)} row groups byte-identical, generated the other "
            f"{_TOTAL_BATCHES - len(first)}; {rows} rows in {_OUTPUT}"
        )
