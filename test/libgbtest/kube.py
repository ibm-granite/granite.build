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

"""Skip gate for build tests that launch into a local kind cluster via SkyPilot.

The Kubernetes counterpart of the slurm gate (``_slurm_cluster_reachable``) and
of ``aws_credentials_present``: a ``skypilot/kubernetes`` build test runs only
when ``make kube-setup`` (``scripts/kube/setup-kind.sh``) has brought the
cluster up.

Reachability alone is not enough. SkyPilot launches into the CURRENT kube
context, so a developer whose current context is a shared or production cluster
would otherwise have test pods scheduled there just because it answered. The
gate therefore also requires the current context to be the expected kind one —
the same "never by accident" rule the aws tests enforce by skipping unless
credentials are explicitly exported.
"""

import functools
import os
import shutil
import subprocess

import pytest

#: Context ``sky local up`` creates with its default cluster name.
DEFAULT_KIND_CONTEXT = "kind-skypilot"

#: Every kube build test class carries this, so they run one at a time. Each pod
#: requests 2 CPUs and a CI runner's kind node has ~3 left after the control
#: plane, so concurrent xdist workers would leave pods unschedulable; with the
#: Makefile's ``--dist=loadgroup`` a group runs serially on one worker.
KUBE_XDIST_GROUP = pytest.mark.xdist_group(name="skypilot_kube")


def expected_kind_context() -> str:
    """The kube context the tests may launch into (``GBTEST_SKY_KUBE_CONTEXT``)."""
    return os.environ.get("GBTEST_SKY_KUBE_CONTEXT") or DEFAULT_KIND_CONTEXT


def kind_cluster_reachable() -> bool:
    """True when the current kube context is the expected one and it answers.

    Evaluated by ``skipif`` at collection time in every kube test module, on
    every xdist worker, so the probe is cached per (kubectl, context) for the
    life of the process — at most one pair of ``kubectl`` calls each.

    :returns: False when kubectl is missing, the current context is anything
        other than :func:`expected_kind_context`, or its API server does not
        report ready within a few seconds.
    """
    kubectl = shutil.which("kubectl")
    if kubectl is None:
        return False
    return _probe(kubectl, expected_kind_context())


@functools.lru_cache(maxsize=None)
def _probe(kubectl: str, expected: str) -> bool:
    """Uncached check behind :func:`kind_cluster_reachable`."""
    try:
        current = subprocess.run(
            [kubectl, "config", "current-context"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if current.returncode != 0 or current.stdout.strip() != expected:
            return False
        ready = subprocess.run(
            [
                kubectl,
                "--context",
                expected,
                "get",
                "--raw",
                "/readyz",
                "--request-timeout=3s",
            ],
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return ready.returncode == 0


def kind_skip_reason() -> str:
    """Skip message naming the context the gate wanted."""
    return (
        f"kind cluster not reachable as the current kube context "
        f"'{expected_kind_context()}' (run: make kube-setup)"
    )
