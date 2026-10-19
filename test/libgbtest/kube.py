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
of ``aws_credentials_present``: a kube build test runs only when ``make
kube-setup`` (``scripts/kube/setup-kind.sh``) has brought the cluster and its
shared volume up.

The tests use ``space://environments/skypilot/kubernetes/local``, which pins
SkyPilot to the kind context through ``kubernetes.allowed_contexts`` in its
``cloud_config``. So the gate does not require kind to be the CURRENT kube
context — the env, not the ambient context, decides where pods go. It checks
that the pinned context exists and answers, and that the env's shared-volume
claim exists, so a missing ``kube-setup`` skips cleanly instead of leaving pods
``Pending`` on an absent claim.
"""

import functools
import os
import shutil
import subprocess

from libgbtest.xdist_groups import SKYPILOT_XDIST_GROUP

#: Context ``sky local up`` creates with its default cluster name; the one
#: skypilot/kubernetes/local pins in ``allowed_contexts``.
DEFAULT_KIND_CONTEXT = "kind-skypilot"

#: PersistentVolumeClaim skypilot/kubernetes/local mounts at /shared; created by
#: ``make kube-setup`` in the ``default`` namespace.
SHARED_CLAIM = "gb-shared"

#: Every kube build test class carries this, so they run one at a time. Each pod
#: requests 2 CPUs and a CI runner's kind node has ~3 left after the control
#: plane, so concurrent xdist workers would leave pods unschedulable; with the
#: Makefile's ``--dist=loadgroup`` a group runs serially on one worker. Shared with
#: every skypilot_integration test, so kube never overlaps slurm either.
KUBE_XDIST_GROUP = SKYPILOT_XDIST_GROUP


def expected_kind_context() -> str:
    """The kube context the tests expect (``GBTEST_SKY_KUBE_CONTEXT``).

    Overriding it only changes what the gate checks; skypilot/kubernetes/local
    itself pins ``kind-skypilot``.
    """
    return os.environ.get("GBTEST_SKY_KUBE_CONTEXT") or DEFAULT_KIND_CONTEXT


def kind_cluster_reachable() -> bool:
    """True when the kind context exists, answers, and has the shared claim.

    Evaluated by ``skipif`` at collection time in every kube test module, on
    every xdist worker, so the probe is cached per (kubectl, context) for the
    life of the process.

    :returns: False when kubectl is missing, the context is not in the
        kubeconfig, its API server does not report ready within a few seconds,
        or the ``gb-shared`` claim does not exist.
    """
    kubectl = shutil.which("kubectl")
    if kubectl is None:
        return False
    return _probe(kubectl, expected_kind_context())


@functools.lru_cache(maxsize=None)
def _probe(kubectl: str, context: str) -> bool:
    """Uncached check behind :func:`kind_cluster_reachable`."""

    def ok(*args: str) -> bool:
        proc = subprocess.run(
            [kubectl, *args], capture_output=True, text=True, timeout=10, check=False
        )
        return proc.returncode == 0

    try:
        names = subprocess.run(
            [kubectl, "config", "get-contexts", "-o", "name"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if names.returncode != 0 or context not in names.stdout.split():
            return False
        return ok(
            "--context", context, "get", "--raw", "/readyz", "--request-timeout=3s"
        ) and ok(
            "--context",
            context,
            "-n",
            "default",
            "get",
            "pvc",
            SHARED_CLAIM,
            "--request-timeout=3s",
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def kind_skip_reason() -> str:
    """Skip message naming what the gate wanted."""
    return (
        f"kind cluster context '{expected_kind_context()}' not reachable, or its "
        f"'{SHARED_CLAIM}' volume claim is missing (run: make kube-setup)"
    )
