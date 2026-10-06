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

Runs the gate against a stub ``kubectl`` on PATH, so no cluster is needed. The
case that matters most is the refusal: a reachable cluster that is NOT the
current kind context must never let a build test launch pods into it.
"""

import pathlib
import shutil
import stat

import pytest
from libgbtest import kube


def _stub_kubectl(tmp_path: pathlib.Path, current: str, ready_rc: int = 0) -> None:
    """Write a kubectl that reports ``current`` and answers /readyz with ``ready_rc``."""
    script = tmp_path / "kubectl"
    # Absolute interpreter: PATH is narrowed to tmp_path, so `env bash` would fail.
    script.write_text(
        f"#!{_BASH}\n"
        'if [ "$1 $2" = "config current-context" ]; then\n'
        f'  [ -n "{current}" ] && echo "{current}" && exit 0\n'
        "  exit 1\n"
        "fi\n"
        f"exit {ready_rc}\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


_BASH = shutil.which("bash") or "/bin/bash"


@pytest.fixture
def stub_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("GBTEST_SKY_KUBE_CONTEXT", raising=False)
    return tmp_path


class TestKindClusterGate:
    def test_ready_kind_context_passes(self, stub_path):
        _stub_kubectl(stub_path, "kind-skypilot")
        assert kube.kind_cluster_reachable()

    def test_another_current_context_is_refused_even_if_ready(self, stub_path):
        _stub_kubectl(stub_path, "prod-cluster")
        assert not kube.kind_cluster_reachable()

    def test_no_current_context_is_refused(self, stub_path):
        _stub_kubectl(stub_path, "")
        assert not kube.kind_cluster_reachable()

    def test_unready_api_server_is_refused(self, stub_path):
        _stub_kubectl(stub_path, "kind-skypilot", ready_rc=1)
        assert not kube.kind_cluster_reachable()

    def test_missing_kubectl_is_refused(self, stub_path):
        assert not kube.kind_cluster_reachable()

    def test_context_override(self, stub_path, monkeypatch):
        monkeypatch.setenv("GBTEST_SKY_KUBE_CONTEXT", "kind-dpk")
        _stub_kubectl(stub_path, "kind-dpk")
        assert kube.kind_cluster_reachable()
        assert "kind-dpk" in kube.kind_skip_reason()

    def test_default_context_is_the_sky_local_up_one(self, stub_path):
        _stub_kubectl(stub_path, "kind-other")
        assert kube.expected_kind_context() == "kind-skypilot"
        assert not kube.kind_cluster_reachable()


class TestProbeIsCached:
    """The gate runs from ``skipif`` in every kube module on every xdist worker."""

    def test_repeated_checks_call_kubectl_once(self, stub_path):
        calls = stub_path / "calls"
        script = stub_path / "kubectl"
        script.write_text(
            f"#!{_BASH}\n"
            f'echo "$*" >> {calls}\n'
            'if [ "$1 $2" = "config current-context" ]; then echo kind-skypilot; fi\n'
            "exit 0\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        assert all(kube.kind_cluster_reachable() for _ in range(5))
        # One current-context lookup + one /readyz probe, not five of each.
        assert len(calls.read_text().splitlines()) == 2

    def test_shared_marker_is_the_skypilot_kube_group(self):
        assert kube.KUBE_XDIST_GROUP.name == "xdist_group"
        assert kube.KUBE_XDIST_GROUP.kwargs == {"name": "skypilot_kube"}
