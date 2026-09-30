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

"""The build-runner pod inherits gbserver's forwarded environment settings.

The runner, not the watcher, executes builds, so a setting applied only to the
watcher would look set and do nothing. ``BuildRunnerJob`` forwards a small set of
knobs via ``build_runner_extra_env_vars``, which ``k8s/dep-build-runner.yaml``
renders into the runner pod's env.

Asserted against the source, not a live ``BuildRunnerJob``: its constructor reads the
deployment YAML off disk and resolves an image tag, neither of which this needs.
"""

import ast
from pathlib import Path

import pytest

_BUILDRUNNERJOB = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "gbserver"
    / "buildrunnerjob"
    / "buildrunnerjob.py"
)
_TEMPLATE = Path(__file__).resolve().parents[3] / "k8s" / "dep-build-runner.yaml"

# The (env-var-name constant -> value constant) pairs BuildRunnerJob must forward.
_FORWARDED_PAIRS = {
    "ENV_VAR_GBSERVER_K8S_USE_ASPERA": "K8S_USE_ASPERA",
    "ENV_VAR_GBSERVER_LSF_USE_ASPERA": "LSF_USE_ASPERA",
}


def _extra_env_pairs() -> dict:
    """Extract the ``build_runner_extra_env_vars`` dict literal as name->name pairs.

    Parsed from the AST rather than matched as text so reformatting cannot break it
    and a commented-out line cannot satisfy it.

    :returns: Mapping of the dict's key ``Name`` ids to their value ``Name`` ids.
    :raises AssertionError: If the dict literal is not found in the source.
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
    return {
        key.id: value.id
        for node in dict_nodes
        for key, value in zip(node.value.keys, node.value.values)
        if isinstance(key, ast.Name) and isinstance(value, ast.Name)
    }


def test_forwarded_settings_are_in_the_extra_env_dict():
    """Each forwarded knob is a value in the extra-env dict, keyed by its name const."""
    pairs = _extra_env_pairs()
    for key_const, value_const in _FORWARDED_PAIRS.items():
        assert (
            pairs.get(key_const) == value_const
        ), f"{key_const} not forwarded to the build runner; found {pairs}"


@pytest.mark.parametrize(
    "name", sorted(set(_FORWARDED_PAIRS) | set(_FORWARDED_PAIRS.values()))
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
