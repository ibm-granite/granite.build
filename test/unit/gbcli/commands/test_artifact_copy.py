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

"""CLI-level tests for `gb artifact copy`.

`copy` detects the store from the URI scheme: an HF source is refused, and an LH
model source proceeds to the Lakehouse copy path.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from gbcli.commands.command_artifact import cli


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


def test_copy_lh_model_proceeds_to_copy_with_source_table(copy_env):
    """An lh:// model source proceeds past the HF guard and copies with the
    source table derived directly from the URI's table_name.

    A valid lh model URI: lh://<env>/<ns>/models/<table>/<label>/<rev>. Here the
    URI table is the shared model table, so source_table resolves to it.
    """
    lh_uri = "lh://prod/ns/models/model_shared/my-model/v3"
    copy_env.fetch_artifact_uri.return_value = _artifact(lh_uri)
    copy_response = MagicMock()
    copy_response.status = "SUCCESS"
    copy_env.artifact_copy.return_value = {
        "copy_response": copy_response,
        "target_table": "model",
    }
    copy_env.register_artifact.return_value = {"uuid": "uuid-1", "uri": lh_uri}
    result = _invoke(lh_uri)
    assert result.exit_code == 0, result.output
    assert "Copy is not supported for HuggingFace artifacts." not in result.output
    # artifact_copy(lh_token, namespace, source_table, space_to, label, revision, cb)
    args = copy_env.artifact_copy.call_args.args
    assert args[1] == "ns"  # namespace
    assert args[2] == "model_shared"  # source_table (from URI table_name)
    assert args[4] == "my-model"  # model_label
    assert args[5] == "v3"  # revision


def test_copy_name_identifier_clean_error(copy_env):
    """A name-format identifier (not a UUID/URI) errors cleanly, not with a
    traceback from an unbound is_hf_artifact."""
    result = _invoke("ns.tbl")
    assert result.exit_code != 0
    assert "Artifact identifier formatted incorrectly" in result.output
    copy_env.fetch_artifact.assert_not_called()
    copy_env.fetch_artifact_uri.assert_not_called()
    copy_env.artifact_copy.assert_not_called()
