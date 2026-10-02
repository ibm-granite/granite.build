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

  HF URI input  ->  command step in a pod on SkyPilot/kubernetes  ->  env:// output

What it proves that the slurm test cannot: the shipped
``space://environments/skypilot/kubernetes`` env stages an ``hf://`` input into
the SAME pod the step runs in. Each step is its own pod and there is no shared
filesystem, so the env must download inline (``inline: true``) in the step's own
setup; a separate hfpull step would land the bytes in a pod that is torn down.
The command fails on an empty input dir, and ``step_count: 1`` (no queued
hfpull) pins the inline path.

Outputs are env:// only, unlike the slurm test's hf:// pushes, so a run never
writes to a HuggingFace org. The input dataset is public: no HF_TOKEN needed.

Requires a local kind cluster as the CURRENT kube context (``make kube-setup``).
Auto-skips otherwise — including when the current context is any other cluster,
so pods are never scheduled somewhere shared by accident.

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
from libgbtest.kube import kind_cluster_reachable, kind_skip_reason

pytestmark = pytest.mark.skypilot_integration


# Real-infra build test (launches a pod via SkyPilot) — only run in the extended
# suite (make extended-tests), not the fast quick-tests suite.
@extended_testing_only
@pytest.mark.skipif(not kind_cluster_reachable(), reason=kind_skip_reason())
class TestSkypilotKube1Step(AbstractYamlBuildRunnerTest):
    """HF input -> command step in a pod via SkyPilot/kubernetes -> env:// output."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "1step"
