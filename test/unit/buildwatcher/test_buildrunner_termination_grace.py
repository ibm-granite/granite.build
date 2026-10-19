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

"""The build-runner pod gets the configured grace period between SIGTERM and SIGKILL.

`gb build cancel` deletes the build-runner pod. The runner records CANCELLED right
after SIGTERM, but it then needs time to tear down its workload (e.g. helm uninstall);
too short a grace period orphans that workload. Both places that set the grace must
use BUILDRUNNERJOB_TERMINATION_GRACE_SECONDS: the pod spec in the Job template, and
the explicit pod delete in BuildRunnerJob. A shorter grace on that delete would
override the pod spec.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from libgbtest.constants import HAS_K8S, requires_k8s

from gbserver.types.constants import BUILDRUNNERJOB_TERMINATION_GRACE_SECONDS
from gbserver.utils.template import fill_template

if HAS_K8S:  # buildrunnerjob imports kubernetes_asyncio at module level
    from gbserver.buildrunnerjob import buildrunnerjob as brj_mod
    from gbserver.buildrunnerjob.buildrunnerjob import BuildRunnerJob

_TEMPLATE = Path(__file__).resolve().parents[3] / "k8s" / "dep-build-runner.yaml"

# Every value the template requires (see its "Required:" header), so strict
# rendering fails if the template grows a variable this test doesn't supply.
_TEMPLATE_DATA = {
    "build_runner_name": "gb-build-runner-test",
    "build_runner_namespace": "test-ns",
    "build_runner_image": "test-image",
    "build_runner_command": ["gbserver"],
    "main_secret_name": "test-secret",
    "build_workspace_pvc_name": "test-pvc",
    "configmap_name": "test-configmap",
    "dev_staging_prod_env": "DEV",
    "build_runner_labels": {},
    "build_runner_annotations": {},
    "build_runner_extra_env_vars": {},
    "is_metrics_enabled": False,
    "is_metrics_auth_enabled": False,
    "termination_grace_period_seconds": BUILDRUNNERJOB_TERMINATION_GRACE_SECONDS,
}


def test_template_sets_termination_grace_period():
    """The rendered Job's pod spec carries the configured grace period as an int."""
    rendered = fill_template(
        templ=_TEMPLATE.read_text(encoding="utf-8"), data=_TEMPLATE_DATA, strict=True
    )
    pod_spec = yaml.safe_load(rendered)["spec"]["template"]["spec"]
    assert (
        pod_spec["terminationGracePeriodSeconds"]
        == BUILDRUNNERJOB_TERMINATION_GRACE_SECONDS
    )


class _FakeApiClient:
    """Async context manager standing in for the kubernetes ApiClient."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeCoreV1Api:
    """Records the pod deletions BuildRunnerJob requests.

    Args:
        pod_names: names of the pods the label selector should match.
    """

    def __init__(self, pod_names: list[str]):
        self.pod_names = pod_names
        self.deletes: list[tuple[str, object]] = []

    async def list_namespaced_pod(self, namespace, label_selector):
        return SimpleNamespace(
            items=[
                SimpleNamespace(metadata=SimpleNamespace(name=n))
                for n in self.pod_names
            ]
        )

    async def delete_namespaced_pod(self, name, namespace, body):
        self.deletes.append((name, body))


@requires_k8s
def test_pod_delete_uses_configured_grace_period(monkeypatch):
    """Each runner pod is deleted with the configured grace, not a shorter one.

    A shorter grace on this explicit delete (like the former hardcoded 5s) would cut
    the pod spec's grace short and SIGKILL the runner during workload teardown.
    """
    core = _FakeCoreV1Api(["gb-build-runner-x-abcde", "gb-build-runner-x-fghij"])

    async def _create_api_client(**_kwargs):
        return _FakeApiClient()

    monkeypatch.setattr(
        brj_mod.AtomicApiClient, "create_api_client", _create_api_client
    )
    monkeypatch.setattr(brj_mod.client, "CoreV1Api", lambda api: core)
    runner = BuildRunnerJob.__new__(BuildRunnerJob)  # skip the config-reading __init__
    runner.namespace = "test-ns"

    asyncio.run(
        runner._BuildRunnerJob__delete_pods_by_label_with_retry(
            "granite-dot-build/build-id=x"
        )
    )

    assert [name for name, _ in core.deletes] == core.pod_names
    for _, body in core.deletes:
        assert body.grace_period_seconds == BUILDRUNNERJOB_TERMINATION_GRACE_SECONDS
