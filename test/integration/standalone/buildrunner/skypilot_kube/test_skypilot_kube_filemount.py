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

"""Integration test: SkyPilot/kubernetes file_mounts copies a step-relative directory.

The kube sibling of ``skypilot_slurm/test_skypilot_filemount.py``: the target runs
a custom step (defined in a co-located test space) whose ``file_mounts`` copies the
``payload/`` directory shipped next to its ``step.yaml`` into the pod; the step's
``run`` fails the build unless it arrived. This is the mechanism step bundles use
to reach the node (e.g. the dpk step's ``file_mounts: {src: src}``), so on kube it
also covers rsync into a pod over ``kubectl port-forward``.

Requires a local kind cluster as the CURRENT kube context (``make kube-setup``).
Auto-skips otherwise, so pods are never scheduled on another cluster by accident.

The fixture's build.yaml, buildtest.yaml, and test space live in the directory
returned by _get_yaml_spec_dir below.
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


@KUBE_XDIST_GROUP
@extended_testing_only
@pytest.mark.skipif(not kind_cluster_reachable(), reason=kind_skip_reason())
class TestSkypilotKubeFileMount(AbstractYamlBuildRunnerTest):
    """Custom step copies a step-relative dir via file_mounts into a kube pod."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "filemount"
