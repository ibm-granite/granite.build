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

"""Integration test for cross-target mem:// binding on SkyPilot/kubernetes.

The kube sibling of ``skypilot_slurm/test_skypilot_slurm_2target.py``. Two
command-step targets, each launched as its own pod: the first emits a
``GB_ARTIFACT_ID:… GB_ARTIFACT_STATE:…`` marker (a mem:// output), the second
binds it and fails unless the value arrived verbatim — proving the skypilot
monitor's STATE marker and mem:// bindings work on kube, and that a dependent
target launches only after its producer.

This is the only multi-target kube test. It does NOT cover a file handoff between
targets: on the shipped skypilot/kubernetes env each pod has its own filesystem and
there is no s3 assetstore, so that needs an env that adds one (see
docs/environments/skypilot-kubernetes.md).

Requires a local kind cluster as the CURRENT kube context (``make kube-setup``).
Auto-skips otherwise, so pods are never scheduled on another cluster by accident.

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


# One kube build test at a time: each pod requests 2 CPUs, and a CI runner's kind
# node has ~3 left after the control plane, so concurrent xdist workers would leave
# pods unschedulable. loadgroup (the Makefile's PYTEST_DIST_MODE) runs a group on
# one worker, serially.
@pytest.mark.xdist_group(name="skypilot_kube")
@extended_testing_only
@pytest.mark.skipif(not kind_cluster_reachable(), reason=kind_skip_reason())
class TestSkypilotKube2Target(AbstractYamlBuildRunnerTest):
    """mem:// output of one kube target bound as the input of a second target."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "2target"
