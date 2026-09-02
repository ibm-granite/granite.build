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

"""CLI-level tests for `gb artifact copy`, pinning the URI-class refactor.

`copy` used to detect an HF source with `uri.startswith("hf://")`; it now uses
`isinstance(URI.get_uri(uri), HfURI)`. These tests assert the HF source is still
refused with the same message and the LH source still proceeds past the guard.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from gbcli.commands.command_artifact import cli
from gbcli.utils.utils import DecodedURIResponse


def _artifact(uri):
    return {
        "uri": uri,
        "name": "my-model",
        "description": "",
        "checksum": "",
        "origin_uris": [],
        "tags": [],
        "status": "success",
        "certified_no_restrictions": True,
    }


@pytest.fixture
def copy_env():
    with (
        patch("gbcli.commands.common_options.is_standalone", return_value=False),
        patch(
            "gbcli.commands.command_artifact.check_current_and_latest_versions",
            return_value=None,
        ),
        patch("gbcli.commands.command_artifact.get_user_token", return_value="tok"),
        patch("gbcli.commands.command_artifact.GBClient") as gbclient,
    ):
        artifact_client = MagicMock()
        artifact_client.github_token = "tok"
        gbclient.Artifact.return_value = artifact_client
        gbclient.Auth.lakehouse_token_for_space.return_value = "lh-tok"
        yield artifact_client


def _invoke(artifact_id):
    runner = CliRunner()
    return runner.invoke(
        cli,
        ["copy", artifact_id, "--space-to", "dest", "--quiet"],
        catch_exceptions=False,
    )


def test_copy_hf_not_supported(copy_env):
    """An HF source is refused with the exact message; no copy attempted."""
    copy_env.fetch_artifact_uri.return_value = _artifact(
        "hf://huggingface.co/models/org/repo"
    )
    result = _invoke("hf://huggingface.co/models/org/repo")
    assert result.exit_code != 0
    assert "Copy is not supported for HuggingFace artifacts." in result.output
    copy_env.artifact_copy.assert_not_called()


def test_copy_lh_model_proceeds_past_guard(copy_env):
    """An lh:// model source is detected as non-HF and proceeds past the guard."""
    decoded = DecodedURIResponse(
        uri="lh://prod/ns/model_shared/my-model/v3",
        namespace="ns",
        table_name="model_shared",
        type="model",
        model_label="my-model",
        model_revision="v3",
    )
    # A valid lh model URI: lh://<env>/<ns>/models/<table>/<label>/<rev>.
    lh_uri = "lh://prod/ns/models/model_shared/my-model/v3"
    copy_env.fetch_artifact_uri.return_value = _artifact(lh_uri)
    with (
        patch("gbcli.commands.command_artifact.decode_uri", return_value=decoded),
        patch(
            "gbcli.commands.command_artifact.get_artifact_formatted_name",
            return_value="x.model_shared",
        ),
    ):
        result = _invoke(lh_uri)
    # Should NOT be blocked by the HF guard (it may proceed to the LH copy path).
    assert "Copy is not supported for HuggingFace artifacts." not in result.output
