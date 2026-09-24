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

"""The build-runner pod inherits the SkyPilot SSH-probe setting.

The runner, not the watcher, calls ``launch_skypilot``, so a setting applied only to
the watcher would look set and do nothing. ``BuildRunnerJob`` forwards it via
``build_runner_extra_env_vars``, which ``k8s/dep-build-runner.yaml`` renders.

Asserted against the source, not a live ``BuildRunnerJob``: its constructor reads the
deployment YAML off disk and resolves an image tag, neither of which this needs.
"""

import ast
from pathlib import Path

import pytest

from gbserver.types.constants import (
    ENV_VAR_SKYPILOT_SSH_PROBE_TIMEOUT_S,
    GBSERVER_SKYPILOT_SSH_PROBE_TIMEOUT_S,
)

_BUILDRUNNERJOB = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "gbserver"
    / "buildrunnerjob"
    / "buildrunnerjob.py"
)
_TEMPLATE = Path(__file__).resolve().parents[3] / "k8s" / "dep-build-runner.yaml"


def test_env_var_name_is_the_one_skypilot_reads():
    """The forwarded key must be the name the probe guard actually consults."""
    assert ENV_VAR_SKYPILOT_SSH_PROBE_TIMEOUT_S == (
        "GBSERVER_SKYPILOT_SSH_PROBE_TIMEOUT_S"
    )
    # An int, so the rendered pod env value is a plain number (Jinja stringifies it).
    assert isinstance(GBSERVER_SKYPILOT_SSH_PROBE_TIMEOUT_S, int)


def test_probe_timeout_is_forwarded_to_the_build_runner():
    """The probe constant is a value in the extra-env dict, keyed by its env-var name.

    Parsed from the AST rather than matched as text so reformatting cannot break it
    and a commented-out line cannot satisfy it.
    """
    tree = ast.parse(_BUILDRUNNERJOB.read_text(encoding="utf-8"))

    dict_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "build_runner_extra_env_vars"
        and isinstance(node.value, ast.Dict)
    ]
    assert dict_nodes, "build_runner_extra_env_vars dict literal not found"

    pairs = {
        key.id: value.id
        for node in dict_nodes
        for key, value in zip(node.value.keys, node.value.values)
        if isinstance(key, ast.Name) and isinstance(value, ast.Name)
    }
    assert pairs.get("ENV_VAR_SKYPILOT_SSH_PROBE_TIMEOUT_S") == (
        "GBSERVER_SKYPILOT_SSH_PROBE_TIMEOUT_S"
    ), f"probe timeout not forwarded to the build runner; found {pairs}"


@pytest.mark.parametrize(
    "name",
    [
        "ENV_VAR_SKYPILOT_SSH_PROBE_TIMEOUT_S",
        # Both halves: dropping either NameErrors at import.
        "GBSERVER_SKYPILOT_SSH_PROBE_TIMEOUT_S",
    ],
)
def test_forwarded_names_are_imported(name):
    """A name used in the dict must be imported, or the module fails at import."""
    tree = ast.parse(_BUILDRUNNERJOB.read_text(encoding="utf-8"))
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert name in imported


def test_template_renders_the_extra_env_vars():
    """The forwarding is inert unless the template still loops over the dict."""
    template = _TEMPLATE.read_text(encoding="utf-8")
    assert "build_runner_extra_env_vars.items()" in template
    assert "- name: '{{ k }}'" in template
