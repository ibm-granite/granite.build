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

  HF URI input  ->  command step in a container image on SkyPilot/kubernetes  ->  env:// output

Unlike SLURM, which needs the Pyxis SPANK plugin, Kubernetes runs images natively,
so this always runs on the local kind cluster. Because kube pulls hf inputs INLINE,
the download runs inside the step's image; the image here (plain Ubuntu) has no
pip, so the test proves the inline pull bootstraps its own hf client. The bare
image-less path is ``test_skypilot_kube_1step.py``.

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


@extended_testing_only
@pytest.mark.skipif(not kind_cluster_reachable(), reason=kind_skip_reason())
class TestSkypilotKube1StepImage(AbstractYamlBuildRunnerTest):
    """HF input -> command step in a container on kube via SkyPilot -> env:// output."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "1step-image"
