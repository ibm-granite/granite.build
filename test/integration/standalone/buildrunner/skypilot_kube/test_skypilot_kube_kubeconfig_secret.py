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

"""A build reaches a cluster whose credentials come only from a space secret (#505).

Reuses the local kind cluster rather than a second one:

* the kind context's kubeconfig (``kubectl config view --minify --flatten``) is
  written as the space secret ``kind-kubeconfig``;
* the build runs with ``KUBECONFIG`` pointing at a copy of the developer's
  kubeconfig *without* the kind context, so SkyPilot cannot reach the cluster
  from the host's kubeconfig at all;
* the test space's ``skypilot/kubernetes/secret`` env supplies the secret through
  ``kube_configs:``, which gbserver merges into its own kubeconfig and puts
  first on ``KUBECONFIG``.

The developer's ``~/.kube/config`` is never modified, and the entries merged
into gbserver's kubeconfig are released at teardown. Requires the kind cluster
(``make kube-setup``); auto-skips otherwise.
"""

import base64
import os
import shutil
import subprocess
from pathlib import Path

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
        context = expected_kind_context()
        secrets_dir = tmp_path / "secrets"
        secrets_dir.mkdir()
        kind_only = _kubectl_view("--minify", "--flatten", "--context", context)
        # The local secret store's flat format holds base64 values.
        encoded = base64.b64encode(yaml.safe_dump(kind_only).encode()).decode()
        (secrets_dir / "kube.yaml").write_text(
            yaml.safe_dump({"kind-kubeconfig": encoded})
        )
        host = tmp_path / "host-kubeconfig"
        host.write_text(yaml.safe_dump(_without_context(_kubectl_view(), context)))
        os.chmod(host, 0o600)
        monkeypatch.setenv("GBTEST_KUBE_SECRETS_DIR", str(secrets_dir))
        monkeypatch.setenv("KUBECONFIG", str(host))
        yield
        # Remove what the env merged into gbserver's kubeconfig (~/.sky), so it
        # does not outlive the test. KUBECONFIG itself is restored by monkeypatch.
        from gbserver.environment.skypilot_config import release_kube_configs

        release_kube_configs("sky-kube-secret")
