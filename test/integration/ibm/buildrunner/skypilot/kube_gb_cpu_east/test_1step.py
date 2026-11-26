#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Command step on the gb CPU cluster (OpenShift) via SkyPilot/kubernetes, HF URI input.

The CPU-cluster copy of
``test/integration/standalone/buildrunner/skypilot_kube/test_skypilot_kube_1step.py``
and the permissive stepping stone to running SkyPilot on Vela:

  HF URI input  ->  command step in a pod on the CPU cluster  ->  hf:// + env:// outputs

Runs on ``space://environments/skypilot/kubernetes/gb-cpu-east``. Unlike Vela, this
cluster allows plain pods, exec and port-forward, so it checks SkyPilot on
OpenShift itself: pods running as root through the ``gdr`` service account, a
separate hfpull pod writing the ``hf://`` input to the shared
claim gb jobs there already use (mounted at /shared through a subPath), and the
command pod reading it. ``step_count: 5`` pins the pull and the three pushes.

The hf:// outputs are pushed by hfpush pods and, through
:class:`libgbtest.hfpush.HfPushCleanupMixin`, named per run in
``GBTEST_HF_PUSH_ORG`` (default ``ibm-research``), checked on the Hub, and
deleted, so it needs an ``HF_TOKEN`` that can write that org.

Requires the ``gb-cpu-east`` kube context (one-time setup in the env file's header),
a current login, ``GBTEST_SKY_CPU_NAMESPACE`` / ``GBTEST_SKY_CPU_CLAIM`` naming
the cluster's namespace and claim, and the internal gb-test space, which holds
the env (both are site-specific and kept out of this repo); auto-skips otherwise,
and
always in CI/CD.

The fixture's build.yaml and buildtest.yaml live in the directory returned by
_get_yaml_spec_dir below.
"""

import os
from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only
from libgbtest.hfpush import HfPushCleanupMixin
from libgbtest.kube import KUBE_XDIST_GROUP, cpu_cluster_reachable, cpu_skip_reason

pytestmark = pytest.mark.ibm


# Same xdist group as the kind tests: every SkyPilot env deep-merges its
# cloud_config into the one global ~/.sky/config.yaml.
@KUBE_XDIST_GROUP
@extended_testing_only
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="Skip in SPS CI/CD: no CPU-cluster context there",
)
@pytest.mark.skipif(not cpu_cluster_reachable(), reason=cpu_skip_reason())
class TestSkypilotKubeCpu1Step(HfPushCleanupMixin, AbstractYamlBuildRunnerTest):
    """HF input -> command step in a pod on the CPU cluster -> hf:// outputs."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "1step"
