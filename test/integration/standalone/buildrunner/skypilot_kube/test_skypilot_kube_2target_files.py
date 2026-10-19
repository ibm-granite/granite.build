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

"""Integration test for a cross-target FILE handoff on SkyPilot/kubernetes.

Two command-step targets, each its own pod, on
``space://environments/skypilot/kubernetes/local``: ``produce`` writes a per-run
token into a file under /shared and registers it as an env:// output (plus the
token as a mem:// value); ``consume`` binds both and fails unless the file it reads
holds that token. env:// moves no bytes, so this passes only because the env's
shared volume is mounted in both pods — the capability the generic
skypilot/kubernetes env lacks (each pod there has only its own filesystem).

The mem:// value handoff alone is ``test_skypilot_kube_2target.py``.

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
from libgbtest.kube import (
    KUBE_XDIST_GROUP,
    kind_cluster_reachable,
    kind_skip_reason,
)

pytestmark = pytest.mark.skypilot_integration


@KUBE_XDIST_GROUP
@extended_testing_only
@pytest.mark.skipif(not kind_cluster_reachable(), reason=kind_skip_reason())
class TestSkypilotKube2TargetFiles(AbstractYamlBuildRunnerTest):
    """A file written to /shared by one kube target's pod is read by the next's."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "2target-files"
