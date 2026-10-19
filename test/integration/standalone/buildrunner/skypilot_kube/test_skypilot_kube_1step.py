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

"""Integration test for the command step on SkyPilot/kubernetes with an HF URI input.

The kube sibling of ``skypilot_slurm/test_skypilot_1step.py``:

  HF URI input  ->  command step in a pod on SkyPilot/kubernetes  ->  hf:// + env:// outputs

Runs on ``space://environments/skypilot/kubernetes/local``, the slurm model on
kube: a separate hfpull step (its own pod) downloads the ``hf://`` input into the
shared volume every pod mounts at /shared, and the command step — a DIFFERENT pod —
fails unless it can see it. So it proves the shared volume carries data between
pods, and ``step_count: 5`` pins the separate pull and the three pushes
(one per artifact marker).

The two hf:// outputs are pushed by hfpush steps — again separate pods, reading
the artifacts from the shared volume. :class:`libgbtest.hfpush.HfPushCleanupMixin`
names the datasets per run in ``GBTEST_HF_PUSH_ORG``, checks they landed on the
Hub, and deletes them, so it needs an ``HF_TOKEN`` that can write that org.

Requires the local kind cluster and its shared volume (``make kube-setup``);
auto-skips otherwise. The env pins SkyPilot to the kind context, so pods never
land on whatever cluster happens to be current.

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
from libgbtest.hfpush import HfPushCleanupMixin
from libgbtest.kube import (
    KUBE_XDIST_GROUP,
    kind_cluster_reachable,
    kind_skip_reason,
)

pytestmark = pytest.mark.skypilot_integration


# Real-infra build test (launches a pod via SkyPilot) — only run in the extended
# suite (make extended-tests), not the fast quick-tests suite.
@KUBE_XDIST_GROUP
@extended_testing_only
@pytest.mark.skipif(not kind_cluster_reachable(), reason=kind_skip_reason())
class TestSkypilotKube1Step(HfPushCleanupMixin, AbstractYamlBuildRunnerTest):
    """HF input -> command step in a pod via SkyPilot/kubernetes -> hf:// outputs."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "1step"
