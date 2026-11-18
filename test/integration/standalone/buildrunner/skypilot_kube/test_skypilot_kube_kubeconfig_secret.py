#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""A build reaches a cluster whose credentials come only from a space secret (#505, #571).

Reuses the local kind cluster rather than a second one:

* the kind context's kubeconfig (``kubectl config view --minify --flatten``) is
  written as the space secret ``KIND-KUBECONFIG-SECRET``;
* the build runs with ``KUBECONFIG`` pointing at a copy of the developer's
  kubeconfig *without* the kind context, so SkyPilot cannot reach the cluster
  from the host's kubeconfig at all;
* the test space's ``skypilot/kubernetes/secret`` env supplies the secret through
  ``kube_configs:``; gbserver writes it to the environment instance's own
  kubeconfig, which only that instance's SkyPilot requests carry.

A second test checks the isolation itself: two environment instances define the
same context name, one with the kind cluster's credentials and one with a decoy
pointing at a dead port, and their concurrent requests each resolve their own.

The developer's ``~/.kube/config`` is never modified, and the process
``KUBECONFIG`` is never changed. Requires the kind cluster (``make kube-setup``)
and a SkyPilot API server started by this checkout's gbserver (its plugin
reports per-request ``KUBECONFIG`` forwarding); auto-skips without kind.
"""

import asyncio
import base64
import os
import shutil
import subprocess
import urllib.parse
from pathlib import Path
from typing import Optional

import pytest
import yaml
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only
from libgbtest.kube import (
    KUBE_XDIST_GROUP,
    expected_kind_context,
    kind_cluster_reachable,
    kind_skip_reason,
)

pytestmark = pytest.mark.skypilot_integration


def _kubectl_view(*args: str) -> dict:
    out = subprocess.run(
        [shutil.which("kubectl") or "kubectl", "config", "view", "--raw", *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    return yaml.safe_load(out) or {}


_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _endpoint(server: Optional[str]) -> tuple:
    """A cluster ``server`` URL as ``(host, port)``, with loopback names unified."""
    url = urllib.parse.urlsplit(server or "")
    host = (url.hostname or "").lower()
    return ("loopback" if host in _LOOPBACK_HOSTS else host, url.port or 443)


def _context_unresolvable(kubeconfig: Path, context: str) -> bool:
    """Whether ``kubeconfig`` neither names ``context`` nor has a cluster at its server.

    kubectl must fail to find the context, and no cluster entry may point at
    the kind API server's host and port (``localhost`` and ``127.0.0.1`` count
    as the same), so a copy under another context name is caught too. It
    checks the kubeconfig the build sees, not the network. The API server,
    started before ``KUBECONFIG`` changes, still has the kind context in its
    own environment; that it uses each request's ``KUBECONFIG`` instead is
    what ``check_server_forwards_kubeconfig`` guarantees.
    """
    kind_server = next(
        c["cluster"]["server"]
        for c in _kubectl_view("--minify", "--context", context)["clusters"]
    )
    named = subprocess.run(
        [shutil.which("kubectl") or "kubectl", "config", "get-contexts", context],
        env={**os.environ, "KUBECONFIG": str(kubeconfig)},
        capture_output=True,
        timeout=30,
        check=False,
    )
    doc = yaml.safe_load(kubeconfig.read_text()) or {}
    endpoints = {
        _endpoint((c.get("cluster") or {}).get("server"))
        for c in doc.get("clusters") or []
    }
    return named.returncode != 0 and _endpoint(kind_server) not in endpoints


def _without_context(doc: dict, context: str) -> dict:
    """``doc`` minus ``context`` and the cluster and user only it refers to."""
    ctx = next(c for c in doc.get("contexts") or [] if c["name"] == context)
    refs = ctx.get("context") or {}
    others = [c for c in doc.get("contexts") or [] if c["name"] != context]
    used = {(c.get("context") or {}).get("cluster") for c in others}
    users = {(c.get("context") or {}).get("user") for c in others}
    out = dict(doc)
    out["contexts"] = others
    out["clusters"] = [
        c
        for c in doc.get("clusters") or []
        if c["name"] != refs.get("cluster") or c["name"] in used
    ]
    out["users"] = [
        u
        for u in doc.get("users") or []
        if u["name"] != refs.get("user") or u["name"] in users
    ]
    if out.get("current-context") == context:
        out.pop("current-context")
    return out


@KUBE_XDIST_GROUP
@extended_testing_only
@pytest.mark.skipif(not kind_cluster_reachable(), reason=kind_skip_reason())
class TestSkypilotKubeKubeconfigSecret(AbstractYamlBuildRunnerTest):
    """A pod launches using only kube_configs-supplied credentials."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "kubeconfig-secret"

    @pytest.fixture(autouse=True)
    def _credentials_only_from_the_secret(self, tmp_path, monkeypatch):
        _api_server_started_before_kubeconfig_changes()
        context = expected_kind_context()
        secrets_dir = tmp_path / "secrets"
        secrets_dir.mkdir()
        kind_only = _kubectl_view("--minify", "--flatten", "--context", context)
        # The local secret store's flat format holds base64 values.
        encoded = base64.b64encode(yaml.safe_dump(kind_only).encode()).decode()
        (secrets_dir / "kube.yaml").write_text(
            yaml.safe_dump({"KIND-KUBECONFIG-SECRET": encoded})
        )
        host = tmp_path / "host-kubeconfig"
        host.write_text(yaml.safe_dump(_without_context(_kubectl_view(), context)))
        os.chmod(host, 0o600)
        # Negative control: the build's own KUBECONFIG must not reach the kind
        # cluster, or a passing build wouldn't prove the secret worked.
        assert _context_unresolvable(host, context)
        monkeypatch.setenv("GBTEST_KUBE_SECRETS_DIR", str(secrets_dir))
        monkeypatch.setenv("KUBECONFIG", str(host))
        yield
        _remove_this_process_kubeconfigs()


def _api_server_started_before_kubeconfig_changes() -> None:
    """Start the SkyPilot API server (if needed) before a test changes KUBECONFIG.

    A server started while KUBECONFIG points at a test's temporary file keeps it
    as its own; later requests that forward no KUBECONFIG (any build without
    kube_configs, with KUBECONFIG unset) would then miss the kind context.
    """
    from gbserver.environment.skypilot import _ensure_skypilot_api_running

    _ensure_skypilot_api_running()


def _remove_this_process_kubeconfigs() -> None:
    """Drop the instance kubeconfigs this test process wrote (they hold kind creds)."""
    from gbserver.environment.skypilot_kube import KUBE_DIR, _process_tag

    # Names are <host>-<pid>-<start>-<nonce>-<id>.yaml (instance_kubeconfig_path).
    for path in (Path.home() / ".sky" / KUBE_DIR).glob(f"{_process_tag()}-*.yaml"):
        path.unlink(missing_ok=True)


def _kube_env(name: str, secret_text: str):
    from gbserver.environment.skypilot import Skypilot
    from gbserver.types.environmentconfig import EnvironmentConfig

    context = expected_kind_context()
    return Skypilot(
        event_q=asyncio.Queue(),
        environment_config=EnvironmentConfig(
            name=name,
            type="Skypilot",
            config={
                "default_cloud": "kubernetes",
                "kube_configs": [{"context": context, "kubeconfig": "kc"}],
                "cloud_config": {"kubernetes": {"allowed_contexts": [context]}},
            },
        ),
        secrets={"kc": secret_text},
    )


@KUBE_XDIST_GROUP
@extended_testing_only
@pytest.mark.skipif(not kind_cluster_reachable(), reason=kind_skip_reason())
class TestSkypilotKubeInstancesStayApart:
    """Two instances, same context name, different secrets: no cross-talk."""

    def test_concurrent_requests_resolve_their_own_instance(
        self, tmp_path, monkeypatch
    ):
        from sky import get as sky_get
        from sky.client import sdk

        from gbserver.environment.skypilot import _kube_scoped

        context = expected_kind_context()
        real = _kubectl_view("--minify", "--flatten", "--context", context)
        decoy = yaml.safe_load(yaml.safe_dump(real))
        decoy["clusters"][0]["cluster"] = {
            "server": "https://127.0.0.1:1",
            "insecure-skip-tls-verify": True,
        }
        decoy["users"][0]["user"] = {"token": "decoy"}
        host = tmp_path / "host-kubeconfig"
        host.write_text(yaml.safe_dump(_without_context(_kubectl_view(), context)))
        os.chmod(host, 0o600)
        _api_server_started_before_kubeconfig_changes()
        monkeypatch.setenv("KUBECONFIG", str(host))

        async def node_names(self):
            info = await asyncio.to_thread(
                lambda: sky_get(sdk.kubernetes_node_info(context=context))
            )
            return sorted(getattr(info, "node_info_dict", {}) or {})

        probe = _kube_scoped(strict=True)(node_names)
        envs = {
            "real": _kube_env("kube-real", yaml.safe_dump(real)),
            "decoy": _kube_env("kube-decoy", yaml.safe_dump(decoy)),
        }

        async def run_all():
            return await asyncio.gather(
                *(probe(envs[k]) for k in ("real", "decoy", "real", "decoy")),
                return_exceptions=True,
            )

        try:
            results = asyncio.run(run_all())
        finally:
            _remove_this_process_kubeconfigs()
        real_results, decoy_results = results[0::2], results[1::2]
        assert all(isinstance(r, list) and r for r in real_results), real_results
        # The decoy's own dead endpoint, not any other failure (a race, say).
        assert all(
            type(r).__name__ == "KubeAPIUnreachableError" for r in decoy_results
        ), decoy_results
        assert os.environ["KUBECONFIG"] == str(host)  # the process value is untouched
