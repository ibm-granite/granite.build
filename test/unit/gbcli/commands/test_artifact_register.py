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

"""CLI-level tests for `gb artifact register`, focused on the HuggingFace path.

The command decodes a `--uri` differently per scheme (lh:// vs hf://), infers the
store from the scheme (so an explicit `--store` conflicts), and treats
`--label`/`--repo` as the model name (the HF repo), distinct from
`--artifact-name` (the registry name). These tests mock every collaborator that
would touch infrastructure so the command's argument handling can be exercised in
isolation.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from gbcli.commands.command_artifact import cli

# These tests exercise a @reject_standalone command; patch is_standalone -> False.


@pytest.fixture
def register_env():
    """Patch out infra collaborators and yield the mocked `register_artifact`.

    Returns the MagicMock standing in for `GBClient.Artifact.register_artifact`
    so tests can assert on the arguments the command derived.
    """
    with (
        patch("gbcli.commands.common_options.is_standalone", return_value=False),
        patch(
            "gbcli.commands.command_artifact.check_current_and_latest_versions",
            return_value=None,
        ),
        patch("gbcli.commands.command_artifact.get_user_token", return_value="tok"),
        patch(
            "gbcli.commands.command_artifact.validate_tags",
            return_value=["sys-official"],
        ),
        patch("gbcli.commands.command_artifact.GBClient") as gbclient,
    ):
        artifact_client = MagicMock()
        artifact_client.github_token = "tok"
        artifact_client.register_artifact.return_value = {
            "uuid": "uuid-1",
            "uri": "hf://huggingface.co/models/org/repo",
        }
        gbclient.Artifact.return_value = artifact_client
        yield artifact_client.register_artifact


def _invoke(args, stdin=""):
    runner = CliRunner()
    return runner.invoke(cli, ["register", *args], input=stdin, catch_exceptions=False)


def test_hf_uri_infers_store_type_org_and_label(register_env):
    """`--uri hf:///org/repo` (no --store, no -t) → store=hf, type=model, no prompt."""
    result = _invoke(
        [
            "--uri",
            "hf:///ibm-granite/granite-4.2-3b",
            "--artifact-name",
            "granite-4.2-3b",
            "--certify-no-restrictions",
            "--tag",
            "sys-official",
        ]
    )
    assert result.exit_code == 0, result.output
    kwargs = register_env.call_args.kwargs
    assert kwargs["store"] == "hf"
    assert kwargs["type"] == "model"
    assert kwargs["hf_organization"] == "ibm-granite"
    assert kwargs["label"] == "granite-4.2-3b"
    assert kwargs["artifact_name"] == "granite-4.2-3b"
    # The revision comes solely from the URI; when omitted it decodes to the
    # "main" default, which the CLI forwards as-is.
    assert kwargs["revision"] == "main"
    # No interactive prompt should have been shown.
    assert "Model table" not in result.output
    assert "Revision" not in result.output


def test_hf_uri_dataset(register_env):
    result = _invoke(
        [
            "--uri",
            "hf:///datasets/org/my-dataset",
            "--artifact-name",
            "my-dataset",
            "--certify-no-restrictions",
        ]
    )
    assert result.exit_code == 0, result.output
    kwargs = register_env.call_args.kwargs
    assert kwargs["store"] == "hf"
    assert kwargs["type"] == "dataset"
    assert kwargs["hf_organization"] == "org"
    assert kwargs["label"] == "my-dataset"
    assert "Dataset name" not in result.output
    assert "Table" not in result.output


def test_hf_uri_bucket(register_env):
    result = _invoke(
        [
            "--uri",
            "hf:///buckets/org/my-bucket",
            "--artifact-name",
            "my-bucket",
            "--certify-no-restrictions",
        ]
    )
    assert result.exit_code == 0, result.output
    kwargs = register_env.call_args.kwargs
    assert kwargs["store"] == "hf"
    assert kwargs["type"] == "bucket"
    assert kwargs["label"] == "my-bucket"
    # A bucket URI decodes revision to "", which the CLI normalizes to None so
    # the service defaults it to "main" (matching the --store hf -t bucket path)
    # rather than forwarding an empty string.
    assert kwargs["revision"] is None


def test_hf_uri_explicit_revision(register_env):
    result = _invoke(
        [
            "--uri",
            "hf:///org/repo/v1.2",
            "--artifact-name",
            "repo",
            "--certify-no-restrictions",
        ]
    )
    assert result.exit_code == 0, result.output
    assert register_env.call_args.kwargs["revision"] == "v1.2"


def test_hf_dataset_uri_preserves_revision(register_env):
    """A revision in a dataset URI must survive the dataset type-handling block."""
    result = _invoke(
        [
            "--uri",
            "hf:///datasets/org/ds/v2",
            "--artifact-name",
            "ds",
            "--certify-no-restrictions",
        ]
    )
    assert result.exit_code == 0, result.output
    kwargs = register_env.call_args.kwargs
    assert kwargs["type"] == "dataset"
    assert kwargs["revision"] == "v2"


# The type, org, repo and revision live in the hf:// URI; the flags that would
# otherwise supply them (--type, --hf-organization, --label/--repo, --revision)
# are rejected outright when a URI is given, whether or not they agree with it.
_URI_FLAG_REJECTED = "cannot be used with an hf:// URI"


@pytest.mark.parametrize(
    "extra_args",
    [
        pytest.param(["-t", "model"], id="type-mismatch"),
        pytest.param(["-t", "dataset"], id="type-matching"),
        pytest.param(["--hf-organization", "other-org"], id="org-mismatch"),
        pytest.param(["--hf-organization", "org"], id="org-matching"),
        pytest.param(["--repo", "other-repo"], id="repo-mismatch"),
        pytest.param(["--repo", "ds"], id="repo-matching"),
        pytest.param(["--revision", "v9"], id="revision-uri-has-none"),
    ],
)
def test_uri_flags_rejected(register_env, extra_args):
    """--type/--hf-organization/--repo/--revision are all rejected alongside a URI."""
    result = _invoke(
        [
            "--uri",
            "hf:///datasets/org/ds",
            "--artifact-name",
            "ds",
            "--certify-no-restrictions",
            *extra_args,
        ]
    )
    assert result.exit_code != 0
    assert _URI_FLAG_REJECTED in result.output
    register_env.assert_not_called()


def test_revision_flag_rejected_when_uri_has_revision(register_env):
    """--revision alongside a URI that also carries one is still a flag rejection."""
    result = _invoke(
        [
            "--uri",
            "hf:///org/repo/v1",
            "--revision",
            "v2",
            "--artifact-name",
            "repo",
            "--certify-no-restrictions",
        ]
    )
    assert result.exit_code != 0
    assert _URI_FLAG_REJECTED in result.output
    register_env.assert_not_called()


def test_revision_flag_rejected_on_bucket_uri(register_env):
    """A bucket URI plus --revision is rejected by the same flag guard."""
    result = _invoke(
        [
            "--uri",
            "hf:///buckets/org/b",
            "--revision",
            "v1",
            "--artifact-name",
            "b",
            "--certify-no-restrictions",
        ]
    )
    assert result.exit_code != 0
    assert _URI_FLAG_REJECTED in result.output
    register_env.assert_not_called()


def test_malformed_hf_uri_clean_error(register_env):
    """A malformed hf:// URI produces a clean CLI error, not a traceback."""
    result = _invoke(
        [
            "--uri",
            "hf:///onlyowner",
            "--artifact-name",
            "x",
            "--certify-no-restrictions",
        ]
    )
    assert result.exit_code != 0
    assert "invalid artifact URI" in result.output
    register_env.assert_not_called()


def test_hf_uri_undefined_template_var_clean_error(register_env):
    """An unresolved template var in the URI raises RuntimeError in the URI
    layer's strict templating; it must still surface as a clean CLI error, not
    a traceback (a malformed template string raises ValueError; both are
    caught).

    The strict-template failure logs at ERROR; silence logging so those
    handlers don't write to CliRunner's captured stream after it is torn down.
    """
    logging.disable(logging.CRITICAL)
    try:
        result = _invoke(
            [
                "--uri",
                "hf:///org/{{missing}}",
                "--artifact-name",
                "x",
                "--certify-no-restrictions",
            ]
        )
    finally:
        logging.disable(logging.NOTSET)
    assert result.exit_code != 0
    assert "invalid artifact URI" in result.output
    register_env.assert_not_called()


def test_explicit_store_conflicts_with_uri(register_env):
    result = _invoke(
        [
            "--store",
            "hf",
            "--uri",
            "hf:///org/repo",
            "--artifact-name",
            "repo",
            "--certify-no-restrictions",
        ]
    )
    assert result.exit_code != 0
    assert "--store cannot be combined with --uri" in result.output
    register_env.assert_not_called()


def test_hf_space_uri_rejected(register_env):
    result = _invoke(
        [
            "--uri",
            "hf:///spaces/org/some-space",
            "--artifact-name",
            "x",
            "--certify-no-restrictions",
        ]
    )
    assert result.exit_code != 0
    assert "'model', 'dataset', 'bucket'" in result.output
    register_env.assert_not_called()


def test_lh_bucket_rejected(register_env):
    """buckets are HF-only, so `--store lh -t bucket` is rejected for register too."""
    result = _invoke(
        [
            "--store",
            "lh",
            "-t",
            "bucket",
            "--artifact-name",
            "x",
            "--certify-no-restrictions",
        ]
    )
    assert result.exit_code != 0
    assert "store 'lh' is only allowed for artifact types" in result.output
    register_env.assert_not_called()


def test_repo_is_a_synonym_for_label(register_env):
    """`--repo` binds the same value as `--label` (a true click synonym)."""
    result = _invoke(
        [
            "--store",
            "hf",
            "-t",
            "model",
            "--hf-organization",
            "org",
            "--repo",
            "a",
            "--artifact-name",
            "x",
            "--certify-no-restrictions",
        ]
    )
    assert result.exit_code == 0, result.output
    assert register_env.call_args.kwargs["label"] == "a"


def test_hf_model_without_repo_is_ok(register_env):
    """`--store hf -t model` with no --repo succeeds; repo falls back in the service.

    The CLI leaves `label` unset (no prompt, no requirement for the HF store) and
    the service layer applies `repo_id = label or artifact_name`.
    """
    result = _invoke(
        [
            "--store",
            "hf",
            "-t",
            "model",
            "--hf-organization",
            "org",
            "--artifact-name",
            "x",
            "--certify-no-restrictions",
        ]
    )
    assert result.exit_code == 0, result.output
    # No interactive "Model label" prompt on the HF path.
    assert "Model label" not in result.output
    assert register_env.call_args.kwargs["store"] == "hf"


def test_hf_dataset_without_repo_is_ok(register_env):
    """HF dataset without --repo succeeds too (symmetric with model/bucket)."""
    result = _invoke(
        [
            "--store",
            "hf",
            "-t",
            "dataset",
            "--hf-organization",
            "org",
            "--artifact-name",
            "x",
            "--certify-no-restrictions",
        ]
    )
    assert result.exit_code == 0, result.output
    assert "Dataset name" not in result.output
    assert register_env.call_args.kwargs["store"] == "hf"


def test_hf_bucket_without_repo_is_ok(register_env):
    result = _invoke(
        [
            "--store",
            "hf",
            "-t",
            "bucket",
            "--hf-organization",
            "org",
            "--artifact-name",
            "x",
            "--certify-no-restrictions",
        ]
    )
    assert result.exit_code == 0, result.output
    assert register_env.call_args.kwargs["store"] == "hf"


def test_hf_store_rejects_lakehouse_table_flag(register_env):
    """`--store hf --table ...` errors — HF has no table concept (finding #1).

    Guards the parity between the two entry paths: the --uri path already
    rejected Lakehouse-only flags; the explicit --store hf path must too.
    """
    result = _invoke(
        [
            "--store",
            "hf",
            "-t",
            "dataset",
            "--hf-organization",
            "org",
            "--repo",
            "ds",
            "--table",
            "some_table",
            "--artifact-name",
            "x",
            "--certify-no-restrictions",
        ]
    )
    assert result.exit_code != 0
    assert "Lakehouse-only" in result.output
    register_env.assert_not_called()


def test_hf_uri_rejects_lakehouse_table_flag(register_env):
    """The --uri HF path rejects a Lakehouse-only flag via the same store check."""
    result = _invoke(
        [
            "--uri",
            "hf:///datasets/org/ds",
            "--table",
            "some_table",
            "--artifact-name",
            "x",
            "--certify-no-restrictions",
        ]
    )
    assert result.exit_code != 0
    assert "Lakehouse-only" in result.output
    register_env.assert_not_called()


# --- Lakehouse (lh://) regression coverage -------------------------------------
#
# The lh:// branch decodes the URI through the shared URI layer (URI.get_uri +
# get_metadata in _resolve_uri). These tests pass real, valid lh:// URIs so the
# URI class does the parsing, and patch `gb_environment_config` so the CLI env
# matches the URI's environment and the cross-env mismatch branch is skipped.
# Valid lh layout: lh://<env>/<ns>/<models|datasets|filesets>/<table>/...


def test_lh_uri_model_decodes_through_to_register(register_env):
    """`--uri lh://...` for a model forwards the decoded type/namespace/table/label/revision."""
    with patch(
        "gbcli.commands.command_artifact.gb_environment_config",
    ) as gb_env:
        gb_env.return_value.lakehouse_environment = "prod"
        result = _invoke(
            [
                "--uri",
                "lh://prod/ns/models/model_shared/my-model/v3",
                "--artifact-name",
                "my-model",
                "--certify-no-restrictions",
            ]
        )
    assert result.exit_code == 0, result.output
    kwargs = register_env.call_args.kwargs
    assert kwargs["store"] == "lh"
    assert kwargs["type"] == "model"
    assert kwargs["namespace"] == "ns"
    assert kwargs["table"] == "model_shared"
    assert kwargs["label"] == "my-model"
    assert kwargs["revision"] == "v3"
    # No interactive prompt should have been shown: every field came from the URI.
    assert "Model label" not in result.output
    assert "Revision" not in result.output
    assert "Model table" not in result.output


def test_lh_uri_dataset_decodes_through_to_register(register_env):
    """`--uri lh://...` for a dataset forwards the decoded dataset/table and nulls model fields."""
    with patch(
        "gbcli.commands.command_artifact.gb_environment_config",
    ) as gb_env:
        gb_env.return_value.lakehouse_environment = "prod"
        result = _invoke(
            [
                "--uri",
                "lh://prod/ns/datasets/tbl/my-dataset",
                "--artifact-name",
                "my-dataset",
                "--certify-no-restrictions",
            ]
        )
    assert result.exit_code == 0, result.output
    kwargs = register_env.call_args.kwargs
    assert kwargs["store"] == "lh"
    assert kwargs["type"] == "dataset"
    assert kwargs["namespace"] == "ns"
    assert kwargs["table"] == "tbl"
    assert kwargs["dataset_name"] == "my-dataset"
    # The dataset block nulls revision/label/version for the Lakehouse store.
    assert kwargs["revision"] is None
    assert kwargs["label"] is None
    assert kwargs["version"] is None
    assert "Dataset name" not in result.output
    assert "Table" not in result.output
