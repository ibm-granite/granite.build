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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load-time rejection of ``file:`` URIs in a build.yaml on non-standalone servers.

A ``file:`` URI names a path on the build host itself, so on a shared server a
build.yaml could otherwise make the server sync its own files to compute
(``step_uri: file:///home/gbserver/.kube``) or read/write them via a ``file:``
input/output. Only a STANDALONE server, which runs on the user's own machine,
accepts them.
"""

from unittest.mock import patch

import pytest

from gbserver.types import buildconfig as buildconfig_module
from gbserver.types.buildconfig import (
    BuildConfig,
    BuildTargetConfig,
    BuildTargetInputConfig,
    BuildTargetOutputConfig,
    BuildTargetStepConfig,
)
from gbserver.types.constants import CODE_GBSERVER_BUILTINS_STEPS_GBSTEP_URI


def _config(
    environment_uri="space://environments/bash",
    step_uri="space://steps/cat",
    input_uri=None,
    output_uri=None,
) -> BuildConfig:
    inputs = {"in": BuildTargetInputConfig(uri=input_uri)} if input_uri else {}
    outputs = (
        {"out": BuildTargetOutputConfig(uri=output_uri, type="fileset")}
        if output_uri
        else {}
    )
    return BuildConfig(
        matched_base_key="granite.build",
        targets={
            "t": BuildTargetConfig(
                environment_uri=environment_uri,
                inputs=inputs,
                outputs=outputs,
                steps=[BuildTargetStepConfig(step_uri=step_uri)],
            ),
        },
    )


def _file_uri_errors(cfg: BuildConfig) -> list[str]:
    return [str(e) for e in cfg.my_validate() if "file:" in str(e)]


@pytest.fixture
def hosted():
    with patch.object(buildconfig_module, "is_standalone", return_value=False):
        yield


@pytest.mark.usefixtures("hosted")
class TestHostedRejectsFileUris:
    @pytest.mark.parametrize(
        "uri",
        [
            "file:///home/gbserver/.kube",
            "file:steps/mystep",
            "file://host/etc",
            "steps/mystep",  # no scheme: step_uri defaults to file
            "/home/gbserver/.kube",
            "{{ var }}/x",  # templated scheme falls back to file
        ],
    )
    def test_step_uri(self, uri):
        errors = _file_uri_errors(_config(step_uri=uri))
        assert len(errors) == 1 and "Step `0`" in errors[0]

    @pytest.mark.parametrize("uri", ["file:///home/gbserver/.sky", "environments/x"])
    def test_environment_uri(self, uri):
        errors = _file_uri_errors(_config(environment_uri=uri))
        assert len(errors) == 1 and "environment_uri" in errors[0]

    def test_input_uri(self):
        errors = _file_uri_errors(_config(input_uri="file:///home/gbserver/.ssh"))
        assert len(errors) == 1 and "Input `in`" in errors[0]

    def test_output_uri(self):
        errors = _file_uri_errors(_config(output_uri="file:outputs/run/"))
        assert len(errors) == 1 and "Output `out`" in errors[0]

    def test_default_gbstep_step_uri_is_allowed(self):
        # An empty/missing step_uri is rewritten to the server's own builtin
        # file:// gbstep before validation; that must not be rejected.
        for step in (BuildTargetStepConfig(step_uri=""), BuildTargetStepConfig()):
            assert step.step_uri == CODE_GBSERVER_BUILTINS_STEPS_GBSTEP_URI
        assert _file_uri_errors(_config(step_uri="")) == []

    @pytest.mark.parametrize(
        "field", ["step_uri", "environment_uri", "input_uri", "output_uri"]
    )
    def test_non_file_schemes_allowed(self, field):
        uri = {
            "step_uri": "git+https://github.com/org/repo#subdirectory=steps/x",
            "environment_uri": "space://environments/k8s",
            "input_uri": "hf://huggingface.co/models/org/repo",
            "output_uri": "env:///abs/out/",
        }[field]
        assert _file_uri_errors(_config(**{field: uri})) == []

    def test_bare_input_path_is_not_file(self):
        # Inputs/outputs parse bare strings with URI.get_uri's "git" default.
        assert _file_uri_errors(_config(input_uri="org/repo")) == []


def test_standalone_allows_file_uris():
    with patch.object(buildconfig_module, "is_standalone", return_value=True):
        cfg = _config(
            environment_uri="file:samples/environments/local",
            step_uri="file:///abs/steps/mystep",
            input_uri="file:samples/data/in.txt",
            output_uri="file:outputs/run/",
        )
        assert _file_uri_errors(cfg) == []


# ------------------------------------------------------------------ ".." segments
#
# A ".." path segment in step_uri/environment_uri could climb out of whatever
# directory the URI resolves against (a space's base_uris, a git checkout's
# #subdirectory=, a local path), so it is rejected in every mode, STANDALONE
# included.


def _parent_segment_errors(cfg: BuildConfig) -> list[str]:
    return [str(e) for e in cfg.my_validate() if "'..'" in str(e)]


@pytest.mark.parametrize("standalone", [True, False], ids=["standalone", "hosted"])
class TestParentSegmentsRejected:
    @pytest.mark.parametrize(
        "uri",
        [
            "space://steps/../../../home/gbserver/.kube",
            "space://environments/k8s/..",
            "git+https://github.com/org/repo#subdirectory=steps/../../x",
            "file:../outside",
            "steps/../x",
            "space://environments/{{ '..' }}/x",  # renders to a ".." segment
        ],
    )
    def test_step_and_environment_uri(self, standalone, uri):
        with patch.object(buildconfig_module, "is_standalone", return_value=standalone):
            step_errors = _parent_segment_errors(_config(step_uri=uri))
            env_errors = _parent_segment_errors(_config(environment_uri=uri))
        assert len(step_errors) == 1 and "Step `0`" in step_errors[0]
        assert len(env_errors) == 1 and "environment_uri" in env_errors[0]


@pytest.mark.parametrize(
    "uri", ["space://steps/a..b", "space://steps/..hidden", "file:///abs/x../y"]
)
def test_dots_inside_a_segment_are_allowed(uri):
    with patch.object(buildconfig_module, "is_standalone", return_value=True):
        assert _parent_segment_errors(_config(step_uri=uri)) == []


def test_parent_segments_not_checked_on_inputs_outputs():
    with patch.object(buildconfig_module, "is_standalone", return_value=True):
        cfg = _config(input_uri="env:///a/../b", output_uri="env:///c/../d/")
        assert _parent_segment_errors(cfg) == []


@pytest.mark.usefixtures("hosted")
@pytest.mark.parametrize(
    "uri",
    [
        "{# x #}file:///home/gbserver/.kube/config",  # Jinja comment hides the scheme
        "{% if true %}file:///home/gbserver/.ssh{% endif %}",  # block tag
        "fi{# #}le:///etc",
    ],
)
def test_jinja_comment_or_block_before_scheme_rejected(uri):
    # Inputs/outputs default to the git scheme, so without this a templated
    # prefix would hide a file: URI until it is rendered at run time.
    assert len(_file_uri_errors(_config(input_uri=uri))) == 1
    assert len(_file_uri_errors(_config(output_uri=uri))) == 1
