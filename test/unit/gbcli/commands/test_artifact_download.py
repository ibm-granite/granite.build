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

"""CLI-level tests for `gb artifact download`.

`download` detects the store from the URI scheme. The HF path passes a
plain-string `artifact_type` and an `org/repo` repo_id to `download_hf_artifact`;
the LH path routes through the decoded namespace/table/label/revision.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from gbcli.commands.command_artifact import cli


@pytest.fixture
def download_env():
    """Patch out infra collaborators and yield the mocked Artifact client."""
    with (
        patch("gbcli.commands.common_options.is_standalone", return_value=False),
        patch(
            "gbcli.commands.command_artifact.check_current_and_latest_versions",
            return_value=None,
        ),
        patch("gbcli.commands.command_artifact.get_user_token", return_value="tok"),
        patch("gbcli.commands.command_artifact.hf_token", return_value="hf-tok"),
        patch("gbcli.commands.command_artifact.GBClient") as gbclient,
    ):
        artifact_client = MagicMock()
        artifact_client.github_token = "tok"
        artifact_client.download_hf_artifact.return_value = {
            "download_dir": "/tmp/x",
            "repo_id": "org/repo",
            "artifact_type": "model",
            "file_count": 1,
            "total_size": 1,
        }
        gbclient.Artifact.return_value = artifact_client
        gbclient.Auth.lakehouse_user_token.return_value = "lh-tok"
        yield artifact_client


def _invoke(tmp_path, artifact_id, extra=None):
    runner = CliRunner()
    return runner.invoke(
        cli,
        ["download", artifact_id, "-d", str(tmp_path), "--quiet", *(extra or [])],
        catch_exceptions=False,
    )


@pytest.mark.parametrize(
    "uri,expected_repo,expected_type",
    [
        ("hf:///models/org/repo", "org/repo", "model"),
        ("hf:///datasets/org/ds", "org/ds", "dataset"),
        ("hf:///buckets/org/b", "org/b", "bucket"),
        ("hf:///org/repo", "org/repo", "model"),  # implicit model
    ],
)
def test_download_hf_passes_string_type(
    download_env, tmp_path, uri, expected_repo, expected_type
):
    """HF download passes a plain-string artifact_type + org/repo repo_id."""
    download_env.fetch_artifact_uri.return_value = {"uri": uri}
    result = _invoke(tmp_path, uri)
    assert result.exit_code == 0, result.output
    args = download_env.download_hf_artifact.call_args.args
    # download_hf_artifact(hf_token, repo_id, artifact_type, dir, revision, cb)
    assert args[1] == expected_repo
    assert args[2] == expected_type
    assert isinstance(args[2], str)


def test_download_hf_missing_token(download_env, tmp_path):
    """A falsy HF token produces the exact 'token not found' error."""
    download_env.fetch_artifact_uri.return_value = {"uri": "hf:///models/org/repo"}
    with patch("gbcli.commands.command_artifact.hf_token", return_value=None):
        result = _invoke(tmp_path, "hf:///models/org/repo")
    assert result.exit_code != 0
    assert "HuggingFace token not found" in result.output
    download_env.download_hf_artifact.assert_not_called()


def test_download_lh_model_routes_with_decoded_fields(download_env, tmp_path):
    """An lh:// model URI still routes to download_model with decoded fields.

    A valid lh model URI: lh://<env>/<ns>/models/<table>/<label>/<rev>. The
    shared LhURI class parses it for real (no decode_uri patch needed).
    """
    lh_uri = "lh://prod/ns/models/model_shared/my-model/v3"
    download_env.fetch_artifact_uri.return_value = {"uri": lh_uri}
    result = _invoke(tmp_path, lh_uri)
    assert result.exit_code == 0, result.output
    download_env.download_hf_artifact.assert_not_called()
    args = download_env.download_model.call_args.args
    # download_model(lh_token, namespace, table, label, revision, dir, space, cb)
    assert args[1] == "ns"
    assert args[2] == "model_shared"
    assert args[3] == "my-model"
    assert args[4] == "v3"
