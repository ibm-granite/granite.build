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

"""Integration test for the command step WITH an image on SkyPilot/kubernetes.

The kube sibling of ``skypilot_slurm/test_skypilot_1step_image.py``:

  HF URI input  ->  command step in a container image on SkyPilot/kubernetes  ->  hf:// + env:// outputs

Unlike SLURM, which needs the Pyxis SPANK plugin, Kubernetes runs images natively,
so this always runs on the local kind cluster. The hf input is downloaded by a
separate hfpull step into the shared volume, and the step's image (plain Ubuntu)
reads it from /shared — proving the env's volume is mounted into image pods too.
SkyPilot's kube pod bootstrap needs a Debian/Ubuntu image (it installs packages
with apt-get), which is why this is not the slurm test's fedora-minimal. The bare
image-less path is ``test_skypilot_kube_1step.py``.

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


@KUBE_XDIST_GROUP
@extended_testing_only
@pytest.mark.skipif(not kind_cluster_reachable(), reason=kind_skip_reason())
class TestSkypilotKube1StepImage(HfPushCleanupMixin, AbstractYamlBuildRunnerTest):
    """HF input -> command step in a container on kube via SkyPilot -> hf:// outputs."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "1step-image"
