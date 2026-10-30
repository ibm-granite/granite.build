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

"""Unit tests for libgbtest.kube.kind_cluster_reachable, the skypilot/kubernetes skip gate.

Runs the gate against a stub ``kubectl`` on PATH, so no cluster is needed. The gate
admits a run only when the pinned kind context exists, answers, and has the
shared-volume claim; it no longer cares which context is CURRENT, because
skypilot/kubernetes/local pins SkyPilot to the kind context itself.
"""

import pathlib
import shutil
import stat

import pytest
from libgbtest import kube

_BASH = shutil.which("bash") or "/bin/bash"


def _stub_kubectl(
    tmp_path: pathlib.Path,
    contexts: tuple[str, ...] = ("prod-cluster", "kind-skypilot"),
    ready_rc: int = 0,
    pvc_rc: int = 0,
    calls: pathlib.Path | None = None,
) -> None:
    """A kubectl that lists ``contexts``, answers /readyz and `get pvc` with the rcs."""
    script = tmp_path / "kubectl"
    log = f'echo "$*" >> {calls}\n' if calls else ""
    listing = "\\n".join(contexts)
    # Absolute interpreter: PATH is narrowed to tmp_path, so `env bash` would fail.
    script.write_text(
        f"#!{_BASH}\n"
        f"{log}"
        'case "$*" in\n'
        f'  "config get-contexts -o name") printf "{listing}\\n" ;;\n'
        f"  *readyz*) exit {ready_rc} ;;\n"
        f'  *" get pvc "*) exit {pvc_rc} ;;\n'
        "  *) exit 2 ;;\n"
        "esac\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def stub_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("GBTEST_SKY_KUBE_CONTEXT", raising=False)
    return tmp_path


class TestKindClusterGate:
    def test_ready_context_with_claim_passes(self, stub_path):
        _stub_kubectl(stub_path)
        assert kube.kind_cluster_reachable()

    def test_another_current_context_does_not_matter(self, stub_path):
        # The stub's contexts include prod-cluster; which one is CURRENT is never
        # consulted: the env pins kind-skypilot via allowed_contexts.
        _stub_kubectl(stub_path, contexts=("prod-cluster", "kind-skypilot"))
        assert kube.kind_cluster_reachable()

    def test_missing_kind_context_is_refused(self, stub_path):
        _stub_kubectl(stub_path, contexts=("prod-cluster",))
        assert not kube.kind_cluster_reachable()

    def test_unready_api_server_is_refused(self, stub_path):
        _stub_kubectl(stub_path, ready_rc=1)
        assert not kube.kind_cluster_reachable()

    def test_missing_shared_claim_is_refused(self, stub_path):
        # Without the claim the env's pods would sit Pending; skip instead.
        _stub_kubectl(stub_path, pvc_rc=1)
        assert not kube.kind_cluster_reachable()
        assert "gb-shared" in kube.kind_skip_reason()

    def test_missing_kubectl_is_refused(self, stub_path):
        assert not kube.kind_cluster_reachable()

    def test_context_override(self, stub_path, monkeypatch):
        monkeypatch.setenv("GBTEST_SKY_KUBE_CONTEXT", "kind-dpk")
        _stub_kubectl(stub_path, contexts=("kind-dpk",))
        assert kube.kind_cluster_reachable()
        assert "kind-dpk" in kube.kind_skip_reason()

    def test_default_context_is_the_sky_local_up_one(self, stub_path):
        _stub_kubectl(stub_path, contexts=("kind-other",))
        assert kube.expected_kind_context() == "kind-skypilot"
        assert not kube.kind_cluster_reachable()


class TestProbeIsCached:
    """The gate runs from ``skipif`` in every kube module on every xdist worker."""

    def test_repeated_checks_call_kubectl_once(self, stub_path):
        calls = stub_path / "calls"
        _stub_kubectl(stub_path, calls=calls)
        assert all(kube.kind_cluster_reachable() for _ in range(5))
        # get-contexts + /readyz + get pvc once, not five times each.
        assert len(calls.read_text().splitlines()) == 3

    def test_shared_marker_is_the_skypilot_group(self):
        """Same group as every skypilot_integration test, so kube never overlaps slurm."""
        assert kube.KUBE_XDIST_GROUP.name == "xdist_group"
        assert kube.KUBE_XDIST_GROUP.kwargs == {"name": "skypilot"}
