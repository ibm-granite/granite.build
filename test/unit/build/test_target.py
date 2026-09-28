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

"""Unit tests for Target.push_assets (#390).

Push destinations now resolve at push time on the dispatched path, so
``push_assets`` does no setup-time work: it is a deprecated no-op returning
``{}`` for every output shape. The former setup-time resolution rejected two
output shapes (a ``{{ binding.* }}``-dependent destination URI and a glob
output key); those restrictions are gone.
"""

from unittest.mock import MagicMock

from gbserver.build.target import Target
from gbserver.types.buildconfig import BuildTargetConfig, BuildTargetOutputConfig


def _out(uri):
    return BuildTargetOutputConfig(uri=uri)


def _make_target(outputs):
    """Build a bare Target wired with the given outputs and a mocked env.

    ``push_assets`` is now a no-op, so it never touches the environment; the
    mock is attached only so attribute access does not blow up.
    """
    target = object.__new__(Target)
    target.name = "t"
    target.config = BuildTargetConfig(
        environment_uri="env:///skypilot",
        outputs=outputs,
        steps=[],
    )
    target.environment = MagicMock()
    return target


def make_target(outputs):
    return _make_target(outputs)


def test_binding_dependent_push_uri_is_no_longer_rejected():
    # A destination URI that references the produced artifact used to raise at
    # setup; resolution is deferred to push time, so it must NOT raise now.
    t = make_target({"model": _out(uri="hf://o/{{ binding.path | short_hash }}")})
    assert t.push_assets() == {}


def test_glob_output_key_is_no_longer_rejected():
    # A glob output key used to raise at setup; it must NOT raise now.
    t = make_target({"model-*": _out(uri="hf://o/r")})
    assert t.push_assets() == {}


def test_push_assets_is_noop_for_plain_output():
    t = make_target({"model_out": _out(uri="hf:///ibm-granite/granite-4.0-h-350m")})
    assert t.push_assets() == {}


def test_push_assets_empty_when_no_outputs():
    target = object.__new__(Target)
    target.config = BuildTargetConfig(environment_uri="env:///skypilot", steps=[])
    target.environment = MagicMock()
    assert target.push_assets() == {}
