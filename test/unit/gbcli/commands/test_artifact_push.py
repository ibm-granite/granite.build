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

"""CLI-level tests for `gb artifact push`, focused on the `--uri` option.

`push` shares its `--uri` handling with `register` via `_resolve_uri`: the store
is inferred from the scheme (lh:// or hf://), an explicit `--store` conflicts, and
the identity flags (`--type`, `--hf-organization`, `--label`/`--repo`, `--table`)
are rejected alongside a URI because the URI already encodes them. `push` supports
both schemes. These tests mock every collaborator that would touch infrastructure
so the command's argument handling is exercised in isolation.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from gbcli.commands.command_artifact import cli

# These tests exercise a @reject_standalone command; patch is_standalone -> False.


@pytest.fixture
def push_env():
    """Patch out infra collaborators and yield the mocked Artifact client.

    Returns the MagicMock standing in for `GBClient.Artifact(...)` so tests can
    assert on the arguments the command derived (esp. `push.call_args`).
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
        patch("gbcli.commands.command_artifact.origins_from_local", return_value=[]),
        # A non-enterprise org short-circuits resource-group resolution.
        patch(
            "gbcli.commands.command_artifact.is_enterprise_hf_org", return_value=False
        ),
        patch("gbcli.commands.command_artifact.GBClient") as gbclient,
    ):
        artifact_client = MagicMock()
        artifact_client.github_token = "tok"
        artifact_client.validate_origins.return_value = ["origin-1"]
        artifact_client.existing_checksum_artifacts.return_value = None
        artifact_client.register_artifact.return_value = {"uuid": "uuid-1"}
        artifact_client.push.return_value = {"uuid": "uuid-1"}
        artifact_client.update_artifact.return_value = {
            "uuid": "uuid-1",
            "uri": "hf://huggingface.co/models/org/repo",
            "checksum": "",
            "status": "success",
        }
        gbclient.Artifact.return_value = artifact_client
        # Tokens must be truthy so the flow doesn't early-return.
        gbclient.Auth.return_value.hf_token.return_value = "hf-tok"
        gbclient.Auth.lakehouse_token_for_space.return_value = "lh-tok"
        yield artifact_client


def _invoke(tmp_path, args, stdin=""):
    runner = CliRunner()
    from_local = tmp_path / "artifact"
    from_local.mkdir(exist_ok=True)
    return runner.invoke(
        cli,
        [
            "push",
            "--from-local",
            str(from_local),
            "--certify-no-restrictions",
            "--yes",
            "--quiet",
            *args,
        ],
        input=stdin,
        catch_exceptions=False,
    )


# --- hf:// happy paths -----------------------------------------------------------


def test_hf_uri_infers_store_type_org_and_label(push_env, tmp_path):
    """`--uri hf:///org/repo` (no --store, no -t) → store=hf, type=model."""
    result = _invoke(
        tmp_path,
        [
            "--uri",
            "hf:///ibm-granite/granite-4.2-3b",
            "--artifact-name",
            "granite-4.2-3b",
        ],
    )
    assert result.exit_code == 0, result.output
    kwargs = push_env.push.call_args.kwargs
    args = push_env.push.call_args.args
    # push() takes type/label positionally: (lh_token, from_local, type, label,
    # artifact_name, size, variant, model_type, version, space, table, ...).
    assert kwargs["store"] == "hf"
    assert args[2] == "model"  # type
    assert args[3] == "granite-4.2-3b"  # label (repo)
    assert kwargs["hf_organization"] == "ibm-granite"


def test_hf_uri_dataset(push_env, tmp_path):
    result = _invoke(
        tmp_path,
        ["--uri", "hf:///datasets/org/my-dataset", "--artifact-name", "my-dataset"],
    )
    assert result.exit_code == 0, result.output
    args = push_env.push.call_args.args
    kwargs = push_env.push.call_args.kwargs
    assert kwargs["store"] == "hf"
    assert args[2] == "dataset"
    assert kwargs["hf_organization"] == "org"


def test_hf_uri_bucket(push_env, tmp_path):
    result = _invoke(
        tmp_path,
        ["--uri", "hf:///buckets/org/my-bucket", "--artifact-name", "my-bucket"],
    )
    assert result.exit_code == 0, result.output
    args = push_env.push.call_args.args
    assert args[2] == "bucket"


def test_hf_dataset_uri_does_not_trip_label_guard(push_env, tmp_path):
    """A dataset URI sets label from the repo; the `and not uri` guard must not fire.

    Without the guard, line "The --label option is not valid for dataset
    artifacts" would falsely reject a URI-derived label.
    """
    result = _invoke(
        tmp_path,
        ["--uri", "hf:///datasets/org/ds", "--artifact-name", "ds"],
    )
    assert result.exit_code == 0, result.output
    assert "not valid for dataset" not in result.output
    push_env.push.assert_called_once()


def test_repo_synonym(push_env, tmp_path):
    """`--repo` binds the same value as `--label` on push too."""
    result = _invoke(
        tmp_path,
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
        ],
    )
    assert result.exit_code == 0, result.output
    assert push_env.push.call_args.args[3] == "a"  # label


# --- hf:// rejections ------------------------------------------------------------


def test_explicit_store_conflicts_with_uri(push_env, tmp_path):
    result = _invoke(
        tmp_path,
        ["--store", "hf", "--uri", "hf:///org/repo", "--artifact-name", "repo"],
    )
    assert result.exit_code != 0
    assert "--store cannot be combined with --uri" in result.output
    push_env.push.assert_not_called()


_URI_FLAG_REJECTED = "cannot be used with an hf:// URI"


@pytest.mark.parametrize(
    "extra_args",
    [
        pytest.param(["-t", "model"], id="type"),
        pytest.param(["--hf-organization", "other-org"], id="org"),
        pytest.param(["--repo", "other-repo"], id="repo"),
    ],
)
def test_uri_flags_rejected(push_env, tmp_path, extra_args):
    """--type/--hf-organization/--repo are rejected alongside an hf:// URI.

    push has no --revision flag, so (unlike register) it is not parametrized here.
    """
    result = _invoke(
        tmp_path,
        ["--uri", "hf:///datasets/org/ds", "--artifact-name", "ds", *extra_args],
    )
    assert result.exit_code != 0
    assert _URI_FLAG_REJECTED in result.output
    push_env.push.assert_not_called()


def test_malformed_hf_uri_clean_error(push_env, tmp_path):
    result = _invoke(tmp_path, ["--uri", "hf:///onlyowner", "--artifact-name", "x"])
    assert result.exit_code != 0
    assert "invalid artifact URI" in result.output
    push_env.push.assert_not_called()


def test_hf_uri_undefined_template_var_clean_error(push_env, tmp_path):
    """An unresolved template var surfaces as a clean CLI error, not a traceback."""
    logging.disable(logging.CRITICAL)
    try:
        result = _invoke(
            tmp_path, ["--uri", "hf:///org/{{missing}}", "--artifact-name", "x"]
        )
    finally:
        logging.disable(logging.NOTSET)
    assert result.exit_code != 0
    assert "invalid artifact URI" in result.output
    push_env.push.assert_not_called()


def test_hf_space_uri_rejected(push_env, tmp_path):
    """`hf:///spaces/...` decodes to type=space, rejected by the type-compat guard."""
    result = _invoke(
        tmp_path, ["--uri", "hf:///spaces/org/some-space", "--artifact-name", "x"]
    )
    assert result.exit_code != 0
    assert "'model', 'dataset', 'bucket'" in result.output
    push_env.push.assert_not_called()


# --- lh:// paths --------------------------------------------------------------
#
# Pass real, valid lh:// URIs so the shared URI class does the parsing; patch
# `gb_environment_config` to control the CLI-side environment comparison.
# Valid lh layout: lh://<env>/<ns>/<models|datasets|filesets>/<table>/...


def test_lh_uri_model_forwarded_to_push(push_env, tmp_path):
    """`--uri lh://...` for a model forwards decoded type/label/table to push()."""
    with patch(
        "gbcli.commands.command_artifact.gb_environment_config",
    ) as gb_env:
        gb_env.return_value.lakehouse_environment = "prod"
        result = _invoke(
            tmp_path,
            [
                "--uri",
                "lh://prod/ns/models/model_shared/my-model/v3",
                "--artifact-name",
                "my-model",
            ],
        )
    assert result.exit_code == 0, result.output
    args = push_env.push.call_args.args
    kwargs = push_env.push.call_args.kwargs
    assert kwargs["store"] == "lh"
    assert args[2] == "model"  # type
    assert args[3] == "my-model"  # label
    assert args[10] == "model_shared"  # table


def test_lh_uri_fileset_forwarded_to_push(push_env, tmp_path):
    with patch(
        "gbcli.commands.command_artifact.gb_environment_config",
    ) as gb_env:
        gb_env.return_value.lakehouse_environment = "prod"
        result = _invoke(
            tmp_path,
            [
                "--uri",
                "lh://prod/ns/filesets/fileset_shared/my-fs/v1",
                "--artifact-name",
                "my-fs",
            ],
        )
    assert result.exit_code == 0, result.output
    args = push_env.push.call_args.args
    assert push_env.push.call_args.kwargs["store"] == "lh"
    assert args[2] == "fileset"
    assert args[3] == "my-fs"
    assert args[10] == "fileset_shared"


def test_lh_uri_dataset_rejected(push_env, tmp_path):
    """An lh:// dataset URI is rejected: push cannot push lh dataset content."""
    with patch(
        "gbcli.commands.command_artifact.gb_environment_config",
    ) as gb_env:
        gb_env.return_value.lakehouse_environment = "prod"
        result = _invoke(
            tmp_path,
            [
                "--uri",
                "lh://prod/ns/datasets/tbl/my-dataset",
                "--artifact-name",
                "my-dataset",
            ],
        )
    assert result.exit_code != 0
    assert "store 'lh' is only allowed for artifact types" in result.output
    push_env.push.assert_not_called()


def test_lh_uri_env_mismatch_rejected(push_env, tmp_path):
    """A non-prod cross-environment lh:// URI is rejected.

    The URI host is 'staging'; the CLI env is 'dev', so the environments differ
    and neither is the prod override case.
    """
    with patch(
        "gbcli.commands.command_artifact.gb_environment_config",
    ) as gb_env:
        gb_env.return_value.lakehouse_environment = "dev"
        result = _invoke(
            tmp_path,
            [
                "--uri",
                "lh://staging/ns/models/model_shared/m/v1",
                "--artifact-name",
                "m",
            ],
        )
    assert result.exit_code != 0
    assert "doesn't match the CLI environment" in result.output
    push_env.push.assert_not_called()


def test_lh_uri_conflict_flag_rejected(push_env, tmp_path):
    """`--uri lh://... --table t` is rejected by the lh conflict guard."""
    with patch(
        "gbcli.commands.command_artifact.gb_environment_config",
    ) as gb_env:
        gb_env.return_value.lakehouse_environment = "prod"
        result = _invoke(
            tmp_path,
            [
                "--uri",
                "lh://prod/ns/models/model_shared/m/v1",
                "--table",
                "t",
                "--artifact-name",
                "m",
            ],
        )
    assert result.exit_code != 0
    assert "cannot be used along with type, label or table" in result.output
    push_env.push.assert_not_called()


# --- backward compatibility ------------------------------------------------------


def test_plain_hf_model_push_unchanged(push_env, tmp_path):
    """No --uri, explicit --store hf -t model still pushes as before."""
    result = _invoke(
        tmp_path,
        [
            "--store",
            "hf",
            "-t",
            "model",
            "--hf-organization",
            "org",
            "--artifact-name",
            "x",
        ],
    )
    assert result.exit_code == 0, result.output
    assert push_env.push.call_args.kwargs["store"] == "hf"
    assert push_env.push.call_args.args[2] == "model"


def test_missing_type_without_uri_errors(push_env, tmp_path):
    """No --uri and no -t → the relaxed 'type or uri required' error."""
    result = _invoke(tmp_path, ["--artifact-name", "x"])
    assert result.exit_code != 0
    assert "Artifact type or artifact uri is required" in result.output
    push_env.push.assert_not_called()


def test_lh_bucket_rejected(push_env, tmp_path):
    """buckets are HF-only, so `--store lh -t bucket` is rejected."""
    result = _invoke(
        tmp_path, ["--store", "lh", "-t", "bucket", "--artifact-name", "x"]
    )
    assert result.exit_code != 0
    assert "store 'lh' is only allowed for artifact types" in result.output
    push_env.push.assert_not_called()


def test_checksum_collision_check_existence_label_keys_on_existing_type(
    push_env, tmp_path
):
    """On a checksum collision, check_existence's label is derived from the
    EXISTING artifact's type/URI (consistent with revision/version), not the
    pushed --type."""
    existing_uri = "lh://prod/ns/models/model_shared/found-model/v7"
    push_env.existing_checksum_artifacts.return_value = {
        "uuid": "existing-uuid",
        "type": "model",
        "uri": existing_uri,
        "status": "success",
    }
    push_env.check_existence.return_value = {"success": False}
    with patch(
        "gbcli.commands.command_artifact.calculate_checksum_",
        return_value="c" * 32,
    ):
        # Pushed type is model; --calculate-checksum triggers the collision path.
        # size/variant/model-type are supplied so no interactive prompt fires.
        _invoke(
            tmp_path,
            [
                "--store",
                "lh",
                "-t",
                "model",
                "--size",
                "8b",
                "--variant",
                "base",
                "--model-type",
                "granite",
                "--calculate-checksum",
                "--artifact-name",
                "x",
            ],
        )
    kwargs = push_env.check_existence.call_args.kwargs
    assert kwargs["type"] == "model"
    assert kwargs["namespace"] == "ns"
    assert kwargs["table"] == "model_shared"
    assert kwargs["label"] == "found-model"  # from the existing URI, not None
    assert kwargs["revision"] == "v7"
