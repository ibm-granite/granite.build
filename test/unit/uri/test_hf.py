#!/usr/bin/env python3

# Copyright Granite.Build Authors
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

"""Tests for HfURI, covering the pull(), push(), exists(), and delete() methods."""

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from gbcommon.uri.hf import DEFAULT_REVISION, HF_HOST, HfType, HfURI
from gbcommon.uri.uri import URI
from gbserver.types.artifact import ArtifactType

# ---------------------------------------------------------------------------
# Unit tests – snapshot_download is mocked, no network required
# ---------------------------------------------------------------------------


class TestHfURIPullUnit:
    """Verify pull() arguments without hitting the network."""

    def test_model_forwards_correct_args(self, tmp_path, monkeypatch):
        """pull() passes repo_id, repo_type, revision, local_dir, and token."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        uri = HfURI.from_parts(
            owner="ibm-granite", repo="granite-3.3-2b-instruct", hf_type=HfType.MODEL
        )
        with patch("gbcommon.uri.hf.snapshot_download") as mock_dl:
            result = uri.pull(tmp_path)

        assert result is True
        mock_dl.assert_called_once_with(
            repo_id="ibm-granite/granite-3.3-2b-instruct",
            repo_type="model",
            revision=DEFAULT_REVISION,
            local_dir=str(tmp_path),
            token=None,
            force_download=False,
            endpoint=None,
        )

    def test_dataset_sets_repo_type(self, tmp_path):
        """HfType.DATASET maps to repo_type='dataset'."""
        uri = HfURI.from_parts(
            owner="wikitext", repo="wikitext-103-v1", hf_type=HfType.DATASET
        )
        with patch("gbcommon.uri.hf.snapshot_download") as mock_dl:
            uri.pull(tmp_path)

        _, kwargs = mock_dl.call_args
        assert kwargs["repo_type"] == "dataset"

    def test_model_type_uses_model_repo_type(self, tmp_path):
        """HfType.MODEL maps to repo_type='model'."""
        uri = HfURI.from_parts(owner="owner", repo="repo", hf_type=HfType.MODEL)
        with patch("gbcommon.uri.hf.snapshot_download") as mock_dl:
            uri.pull(tmp_path)

        _, kwargs = mock_dl.call_args
        assert kwargs["repo_type"] == "model"

    def test_token_from_secrets(self, tmp_path):
        """Token is resolved from secrets dict when HF_TOKEN is present."""
        uri = HfURI.from_parts(owner="owner", repo="repo", hf_type=HfType.MODEL)
        uri.secrets = {"HF_TOKEN": "secret-token"}

        with patch("gbcommon.uri.hf.snapshot_download") as mock_dl:
            uri.pull(tmp_path)

        _, kwargs = mock_dl.call_args
        assert kwargs["token"] == "secret-token"

    def test_token_from_env(self, tmp_path, monkeypatch):
        """Token falls back to the HF_TOKEN environment variable."""
        monkeypatch.setenv("HF_TOKEN", "env-token")
        uri = HfURI.from_parts(owner="owner", repo="repo", hf_type=HfType.MODEL)

        with patch("gbcommon.uri.hf.snapshot_download") as mock_dl:
            uri.pull(tmp_path)

        _, kwargs = mock_dl.call_args
        assert kwargs["token"] == "env-token"

    def test_blank_token_treated_as_none(self, tmp_path):
        """A whitespace-only token in secrets is treated as no token."""
        uri = HfURI.from_parts(owner="owner", repo="repo", hf_type=HfType.MODEL)
        uri.secrets = {"HF_TOKEN": "   "}

        with patch("gbcommon.uri.hf.snapshot_download") as mock_dl:
            uri.pull(tmp_path)

        _, kwargs = mock_dl.call_args
        assert kwargs["token"] is None

    def test_force_flag_forwarded(self, tmp_path):
        """force=True is forwarded as force_download=True."""
        uri = HfURI.from_parts(owner="owner", repo="repo", hf_type=HfType.MODEL)

        with patch("gbcommon.uri.hf.snapshot_download") as mock_dl:
            uri.pull(tmp_path, force=True)

        _, kwargs = mock_dl.call_args
        assert kwargs["force_download"] is True

    def test_custom_host_sets_endpoint(self, tmp_path):
        """A non-default host is forwarded as an HTTPS endpoint URL."""
        uri = HfURI.from_parts(
            owner="owner",
            repo="repo",
            hf_type=HfType.MODEL,
            host="internal-hub.example.com",
        )
        with patch("gbcommon.uri.hf.snapshot_download") as mock_dl:
            uri.pull(tmp_path)

        _, kwargs = mock_dl.call_args
        assert kwargs["endpoint"] == "https://internal-hub.example.com"

    def test_default_host_sends_none_endpoint(self, tmp_path):
        """The default huggingface.co host results in endpoint=None."""
        uri = HfURI.from_parts(
            owner="owner", repo="repo", hf_type=HfType.MODEL, host=HF_HOST
        )

        with patch("gbcommon.uri.hf.snapshot_download") as mock_dl:
            uri.pull(tmp_path)

        _, kwargs = mock_dl.call_args
        assert kwargs["endpoint"] is None

    def test_returns_false_on_exception(self, tmp_path):
        """pull() catches any exception and returns False."""
        uri = HfURI.from_parts(owner="owner", repo="repo", hf_type=HfType.MODEL)

        with patch(
            "gbcommon.uri.hf.snapshot_download", side_effect=RuntimeError("boom")
        ):
            result = uri.pull(tmp_path)

        assert result is False


# ---------------------------------------------------------------------------
# Unit tests – HfApi is mocked, no network required
# ---------------------------------------------------------------------------


class TestHfURIPushUnit:
    """Verify push() behaviour without hitting the network."""

    def _make_api(self):
        return MagicMock()

    def test_push_file_calls_upload_file(self, tmp_path):
        """push() calls upload_file with correct args for a single file."""
        src = tmp_path / "weights.bin"
        src.write_bytes(b"data")
        uri = HfURI.from_parts(owner="org", repo="my-model", hf_type=HfType.MODEL)

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            uri.push(src)

        MockApi.return_value.upload_file.assert_called_once_with(
            path_or_fileobj=src,
            path_in_repo="weights.bin",  # defaults to filename
            repo_id="org/my-model",
            repo_type="model",
            revision=DEFAULT_REVISION,
            commit_message="Upload via gbserver",
        )

    def test_push_directory_calls_upload_folder(self, tmp_path):
        """push() calls upload_folder for a directory source."""
        src_dir = tmp_path / "checkpoint"
        src_dir.mkdir()
        uri = HfURI.from_parts(owner="org", repo="my-model", hf_type=HfType.MODEL)

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            uri.push(src_dir)

        MockApi.return_value.upload_folder.assert_called_once_with(
            folder_path=str(src_dir),
            path_in_repo="",
            repo_id="org/my-model",
            repo_type="model",
            revision=DEFAULT_REVISION,
            commit_message="Upload via gbserver",
        )

    def test_push_uri_path_in_repo_used_for_file(self, tmp_path):
        """path_in_repo encoded in the URI is used as the file destination."""
        src = tmp_path / "config.json"
        src.write_text("{}")
        uri = HfURI.from_parts(
            owner="org",
            repo="my-model",
            hf_type=HfType.MODEL,
            path_in_repo="configs/config.json",
        )

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            uri.push(src)

        _, kwargs = MockApi.return_value.upload_file.call_args
        assert kwargs["path_in_repo"] == "configs/config.json"

    def test_push_uri_path_in_repo_used_for_directory(self, tmp_path):
        """path_in_repo encoded in the URI is used as the folder prefix."""
        src_dir = tmp_path / "ckpt"
        src_dir.mkdir()
        uri = HfURI.from_parts(
            owner="org",
            repo="my-model",
            hf_type=HfType.MODEL,
            path_in_repo="checkpoints/v1",
        )

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            uri.push(src_dir)

        _, kwargs = MockApi.return_value.upload_folder.call_args
        assert kwargs["path_in_repo"] == "checkpoints/v1"

    def test_push_custom_commit_message(self, tmp_path):
        """commit_message is forwarded to the Hub API."""
        src = tmp_path / "model.bin"
        src.write_bytes(b"x")
        uri = HfURI.from_parts(owner="org", repo="my-model", hf_type=HfType.MODEL)

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            uri.push(src, commit_message="Add fine-tuned weights")

        _, kwargs = MockApi.return_value.upload_file.call_args
        assert kwargs["commit_message"] == "Add fine-tuned weights"

    def test_push_dataset_repo_type(self, tmp_path):
        """HfType.DATASET is passed as repo_type='dataset'."""
        src_dir = tmp_path / "data"
        src_dir.mkdir()
        uri = HfURI.from_parts(owner="org", repo="my-dataset", hf_type=HfType.DATASET)

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            uri.push(src_dir)

        _, kwargs = MockApi.return_value.upload_folder.call_args
        assert kwargs["repo_type"] == "dataset"

    def test_push_passes_token_to_api(self, tmp_path):
        """Token from secrets is forwarded to HfApi constructor."""
        src = tmp_path / "f.txt"
        src.write_text("hi")
        uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
        uri.secrets = {"HF_TOKEN": "push-token"}

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            uri.push(src)

        MockApi.assert_called_once_with(endpoint=None, token="push-token")

    def test_push_custom_host_sets_endpoint(self, tmp_path, monkeypatch):
        """A non-default host is forwarded as an HTTPS endpoint to HfApi."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        src = tmp_path / "f.txt"
        src.write_text("hi")
        uri = HfURI.from_parts(
            owner="org", repo="repo", hf_type=HfType.MODEL, host="hub.example.com"
        )

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            uri.push(src)

        MockApi.assert_called_once_with(endpoint="https://hub.example.com", token=None)

    def test_push_raises_on_exception(self, tmp_path):
        """push() propagates exceptions from the Hub API."""
        src = tmp_path / "f.bin"
        src.write_bytes(b"x")
        uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            MockApi.return_value.upload_file.side_effect = RuntimeError("network error")
            with pytest.raises(RuntimeError, match="network error"):
                uri.push(src)


# ---------------------------------------------------------------------------
# Unit tests – repo_exists / revision_exists are mocked, no network required
# ---------------------------------------------------------------------------


class ExistsExpection(BaseModel):
    host: str
    type: HfType
    owner: str
    repo_name: str
    revision: Optional[str] = DEFAULT_REVISION
    path_in_repo: Optional[str] = ""


class TestHfURIPartsUnit:
    """Verify exists() behaviour without hitting the network."""

    def test_hf_parts(self):
        self._helper(
            "hf://huggingface.co/datasets/owner/repo_name",
            ExistsExpection(
                host="huggingface.co",
                type=HfType.DATASET,
                owner="owner",
                repo_name="repo_name",
            ),
        )
        self._helper(
            "hf:///models/owner/repo_name",
            ExistsExpection(
                host="huggingface.co",
                type=HfType.MODEL,
                owner="owner",
                repo_name="repo_name",
            ),
        )
        self._helper(
            "hf:///owner/repo_name",  # Without 'models'
            ExistsExpection(
                host="huggingface.co",
                type=HfType.MODEL,
                owner="owner",
                repo_name="repo_name",
            ),
        )
        self._helper(
            "hf://huggingface.co/datasets/ibm-research/vira-intents-live",
            ExistsExpection(
                host="huggingface.co",
                type=HfType.DATASET,
                owner="ibm-research",
                repo_name="vira-intents-live",
            ),
        )
        self._helper(
            "hf:///datasets/ibm-research/test-output2_xyz",
            ExistsExpection(
                host="huggingface.co",
                type=HfType.DATASET,
                owner="ibm-research",
                repo_name="test-output2_xyz",
            ),
        )
        self._helper(
            "hf:///datasets/ibm-research/test-output2_xyz/revision/path/a/b",
            ExistsExpection(
                host="huggingface.co",
                type=HfType.DATASET,
                owner="ibm-research",
                repo_name="test-output2_xyz",
                revision="revision",
                path_in_repo="path/a/b",
            ),
        )

    def _helper(self, hfuri: str, expectations: ExistsExpection) -> None:
        uri = URI.get_uri(hfuri)
        assert isinstance(uri, HfURI)
        assert uri.get_host() == expectations.host
        assert uri.get_hf_type() == expectations.type
        assert uri.get_owner() == expectations.owner
        assert uri.get_repo() == expectations.repo_name
        assert uri.get_revision() == expectations.revision
        assert uri.get_path_in_repo() == expectations.path_in_repo


# ---------------------------------------------------------------------------
# Integration test – real network download of a tiny public HF model
# ---------------------------------------------------------------------------


def test_pull_downloads_tiny_public_model(tmp_path):
    """Download a tiny public model from huggingface.co and verify files land in dest.

    Uses hf-internal-testing/tiny-random-bert — a minimal fixture model
    maintained by HuggingFace specifically for CI/testing (< 1 MB).
    No token is required; the repo is public.
    Skipped automatically if the Hub is unreachable.
    """

    uri = HfURI.from_parts(
        owner="hf-internal-testing",
        repo="tiny-random-bert",
        hf_type=HfType.MODEL,
    )

    try:
        result = uri.pull(tmp_path)
    except Exception as exc:
        pytest.skip(f"HuggingFace Hub not reachable: {exc}")

    assert result is True, "pull() should return True on success"

    downloaded = [f for f in tmp_path.rglob("*") if f.is_file()]
    assert downloaded, f"Expected files in {tmp_path}, found none"


# ---------------------------------------------------------------------------
# Integration test – real network upload to HuggingFace
# ---------------------------------------------------------------------------


def test_push_uploads_file_to_huggingface(tmp_path):
    """Upload a small file to a temporary HF repo and verify it lands there.

    Requires HF_TOKEN to be set with write access to the authenticated user's
    namespace.  The test creates a throwaway repo, pushes one file, asserts
    it appears in the repo's file listing, then deletes the repo.
    Skipped automatically when HF_TOKEN is absent or the Hub is unreachable.
    """
    import os

    from huggingface_hub import HfApi

    token = os.getenv("HF_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN not set — skipping push integration test")

    api = HfApi(token=token)

    try:
        username = api.whoami()["name"]
    except Exception as exc:
        pytest.skip(f"HuggingFace Hub not reachable: {exc}")

    repo_name = "gbserver-push-integ-test"
    repo_id = f"{username}/{repo_name}"

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    except Exception as exc:
        pytest.skip(f"Could not create temporary test repo {repo_id}: {exc}")

    try:
        src = tmp_path / "hello.txt"
        src.write_text("gbserver push integration test")

        uri = HfURI.from_parts(owner=username, repo=repo_name, hf_type=HfType.MODEL)
        uri.secrets = {"HF_TOKEN": token}

        uri.push(src, commit_message="CI: push integration test")

        assert "hello.txt" in list(api.list_repo_files(repo_id=repo_id))
    finally:
        try:
            api.delete_repo(repo_id=repo_id, repo_type="model")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Bucket – artifact type mapping
# ---------------------------------------------------------------------------


class TestHfURIBucketArtifactType:
    def test_bucket_maps_to_bucket(self):
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)
        assert uri.get_artifact_type() == ArtifactType.BUCKET

    def test_model_still_maps_to_model(self):
        uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.MODEL)
        assert uri.get_artifact_type() == ArtifactType.MODEL

    def test_dataset_still_maps_to_dataset(self):
        uri = HfURI.from_parts(owner="org", repo="repo", hf_type=HfType.DATASET)
        assert uri.get_artifact_type() == ArtifactType.DATASET


# ---------------------------------------------------------------------------
# Bucket – URI parsing
# ---------------------------------------------------------------------------


class TestHfURIBucketParts:
    def test_bucket_uri_parts(self):
        uri = HfURI.parse("hf://huggingface.co/buckets/org/my-bucket")
        assert uri.get_hf_type() == HfType.BUCKET
        assert uri.get_owner() == "org"
        assert uri.get_repo() == "my-bucket"
        assert uri.get_host() == "huggingface.co"

    def test_bucket_uri_round_trips(self):
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)
        reparsed = HfURI.parse(str(uri))
        assert reparsed.get_hf_type() == HfType.BUCKET
        assert reparsed.get_owner() == "org"
        assert reparsed.get_repo() == "my-bucket"

    def test_bucket_uri_with_path_in_repo(self):
        uri = HfURI.parse("hf://huggingface.co/buckets/org/my-bucket/subdir/file.bin")
        assert uri.get_hf_type() == HfType.BUCKET
        assert uri.get_path_in_repo() == "subdir/file.bin"
        assert uri.get_revision() == ""


# ---------------------------------------------------------------------------
# Bucket – pull (mocked)
# ---------------------------------------------------------------------------


class TestHfURIBucketPullUnit:
    def test_calls_sync_bucket(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            result = uri.pull(tmp_path)

        assert result is True
        MockApi.return_value.sync_bucket.assert_called_once_with(
            source="hf://buckets/org/my-bucket",
            dest=str(tmp_path),
        )

    def test_with_path_in_repo(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        uri = HfURI.from_parts(
            owner="org",
            repo="my-bucket",
            hf_type=HfType.BUCKET,
            path_in_repo="data/train",
        )

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            uri.pull(tmp_path)

        _, kwargs = MockApi.return_value.sync_bucket.call_args
        assert kwargs["source"] == "hf://buckets/org/my-bucket/data/train"

    def test_does_not_call_snapshot_download(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)

        with (
            patch("gbcommon.uri.hf.snapshot_download") as mock_dl,
            patch("gbcommon.uri.hf.HfApi"),
        ):
            uri.pull(tmp_path)

        mock_dl.assert_not_called()

    def test_returns_false_on_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            MockApi.return_value.sync_bucket.side_effect = RuntimeError("fail")
            result = uri.pull(tmp_path)

        assert result is False

    def test_custom_host_sets_endpoint(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        uri = HfURI.from_parts(
            owner="org", repo="my-bucket", hf_type=HfType.BUCKET, host="hub.example.com"
        )

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            uri.pull(tmp_path)

        MockApi.assert_called_once_with(endpoint="https://hub.example.com", token=None)


# ---------------------------------------------------------------------------
# Bucket – push (mocked)
# ---------------------------------------------------------------------------


class TestHfURIBucketPushUnit:
    def test_file_calls_batch_bucket_files(self, tmp_path):
        src = tmp_path / "data.bin"
        src.write_bytes(b"content")
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            uri.push(src)

        MockApi.return_value.batch_bucket_files.assert_called_once_with(
            bucket_id="org/my-bucket",
            add=[(src, "data.bin")],
        )

    def test_file_with_path_in_repo(self, tmp_path):
        src = tmp_path / "data.bin"
        src.write_bytes(b"x")
        uri = HfURI.from_parts(
            owner="org",
            repo="my-bucket",
            hf_type=HfType.BUCKET,
            path_in_repo="subdir/data.bin",
        )

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            uri.push(src)

        _, kwargs = MockApi.return_value.batch_bucket_files.call_args
        assert kwargs["add"] == [(src, "subdir/data.bin")]

    def test_folder_calls_sync_bucket(self, tmp_path):
        src_dir = tmp_path / "output"
        src_dir.mkdir()
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            uri.push(src_dir)

        MockApi.return_value.sync_bucket.assert_called_once_with(
            source=str(src_dir),
            dest="hf://buckets/org/my-bucket",
        )

    def test_folder_with_path_in_repo(self, tmp_path):
        src_dir = tmp_path / "output"
        src_dir.mkdir()
        uri = HfURI.from_parts(
            owner="org",
            repo="my-bucket",
            hf_type=HfType.BUCKET,
            path_in_repo="prefix/v1",
        )

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            uri.push(src_dir)

        _, kwargs = MockApi.return_value.sync_bucket.call_args
        assert kwargs["dest"] == "hf://buckets/org/my-bucket/prefix/v1"

    def test_creates_bucket_first(self, tmp_path):
        src = tmp_path / "f.txt"
        src.write_text("x")
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            uri.push(src)

        MockApi.return_value.create_bucket.assert_called_once_with(
            bucket_id="org/my-bucket",
            private=None,
            resource_group_id=None,
            exist_ok=True,
        )

    def test_passes_private_and_resource_group(self, tmp_path):
        src = tmp_path / "f.txt"
        src.write_text("x")
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            uri.push(src, private=True, resource_group_id="rg-123")

        MockApi.return_value.create_bucket.assert_called_once_with(
            bucket_id="org/my-bucket",
            private=True,
            resource_group_id="rg-123",
            exist_ok=True,
        )

    def test_does_not_call_create_repo(self, tmp_path):
        src = tmp_path / "f.txt"
        src.write_text("x")
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            uri.push(src)

        MockApi.return_value.create_repo.assert_not_called()

    def test_raises_on_missing_src(self, tmp_path):
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)
        with pytest.raises(ValueError, match="does not exist"):
            uri.push(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Bucket – exists (mocked)
# ---------------------------------------------------------------------------


class TestHfURIBucketExistsUnit:
    def test_calls_bucket_info(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            result = uri.exists()

        assert result is True
        MockApi.return_value.bucket_info.assert_called_once_with(
            bucket_id="org/my-bucket"
        )

    def test_returns_false_when_missing(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        uri = HfURI.from_parts(owner="org", repo="gone", hf_type=HfType.BUCKET)

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            MockApi.return_value.bucket_info.side_effect = RuntimeError("not found")
            result = uri.exists()

        assert result is False

    def test_does_not_call_repo_exists(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)

        with (
            patch("gbcommon.uri.hf.repo_exists") as mock_re,
            patch("gbcommon.uri.hf.HfApi"),
        ):
            uri.exists()

        mock_re.assert_not_called()


# ---------------------------------------------------------------------------
# Bucket – delete (mocked)
# ---------------------------------------------------------------------------


class TestHfURIBucketDeleteUnit:
    def test_deletes_entire_bucket(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            result = uri.delete()

        assert result is True
        MockApi.return_value.delete_bucket.assert_called_once_with(
            bucket_id="org/my-bucket"
        )

    def test_deletes_file_from_bucket(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        uri = HfURI.from_parts(
            owner="org",
            repo="my-bucket",
            hf_type=HfType.BUCKET,
            path_in_repo="old/data.bin",
        )

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            result = uri.delete()

        assert result is True
        MockApi.return_value.batch_bucket_files.assert_called_once_with(
            bucket_id="org/my-bucket", delete=["old/data.bin"]
        )

    def test_does_not_call_delete_repo(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            uri.delete()

        MockApi.return_value.delete_repo.assert_not_called()

    def test_returns_false_on_error(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)

        with patch("gbcommon.uri.hf.HfApi") as MockApi:
            MockApi.return_value.delete_bucket.side_effect = RuntimeError("fail")
            result = uri.delete()

        assert result is False


# ---------------------------------------------------------------------------
# space_name_to_resource_group_name — environment suffix logic
# ---------------------------------------------------------------------------


class TestSpaceNameToResourceGroupName:
    def test_prod_no_suffix(self, monkeypatch):
        monkeypatch.setattr("gbcommon.uri.hf.GB_ENVIRONMENT", "PROD")
        assert HfURI.space_name_to_resource_group_name("public") == "gbspace-public"

    def test_empty_env_no_suffix(self, monkeypatch):
        monkeypatch.setattr("gbcommon.uri.hf.GB_ENVIRONMENT", "")
        assert HfURI.space_name_to_resource_group_name("public") == "gbspace-public"

    def test_staging_suffix(self, monkeypatch):
        monkeypatch.setattr("gbcommon.uri.hf.GB_ENVIRONMENT", "STAGING")
        assert (
            HfURI.space_name_to_resource_group_name("public")
            == "gbspace-public-staging"
        )

    def test_dev_suffix(self, monkeypatch):
        monkeypatch.setattr("gbcommon.uri.hf.GB_ENVIRONMENT", "DEV")
        assert HfURI.space_name_to_resource_group_name("public") == "gbspace-public-dev"

    def test_standalone_no_suffix(self, monkeypatch):
        monkeypatch.setattr("gbcommon.uri.hf.GB_ENVIRONMENT", "STANDALONE")
        assert HfURI.space_name_to_resource_group_name("public") == "gbspace-public"

    def test_empty_space_name_returns_empty(self, monkeypatch):
        monkeypatch.setattr("gbcommon.uri.hf.GB_ENVIRONMENT", "STAGING")
        assert HfURI.space_name_to_resource_group_name("") == ""

    def test_none_space_name_returns_empty(self, monkeypatch):
        monkeypatch.setattr("gbcommon.uri.hf.GB_ENVIRONMENT", "DEV")
        assert HfURI.space_name_to_resource_group_name(None) == ""

    def test_custom_space_name(self, monkeypatch):
        monkeypatch.setattr("gbcommon.uri.hf.GB_ENVIRONMENT", "STAGING")
        assert (
            HfURI.space_name_to_resource_group_name("my-team")
            == "gbspace-my-team-staging"
        )
        monkeypatch.setattr("gbcommon.uri.hf.GB_ENVIRONMENT", "PROD")
        assert HfURI.space_name_to_resource_group_name("my-team") == "gbspace-my-team"


# ---------------------------------------------------------------------------
# resolve_resource_group_id with environment-derived names (mocked)
# ---------------------------------------------------------------------------


class TestResolveResourceGroupIdWithEnvironment:
    """Test that resolve_resource_group_id uses the environment-aware
    resource group name when space_name is provided."""

    def _mock_hf_api_response(self, groups):
        """Patch the HF HTTP session to return the given resource groups list."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = groups
        mock_session.get.return_value = mock_response
        return mock_session

    def test_staging_env_resolves_suffixed_name(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr("gbcommon.uri.hf.GB_ENVIRONMENT", "STAGING")
        uri = HfURI.from_parts(owner="ibm-research", repo="dummy")
        groups = [{"name": "gbspace-public-staging", "id": "staging-id-123"}]

        with patch("gbcommon.uri.hf.HfApi"):
            with patch(
                "huggingface_hub.utils._http.get_session",
                return_value=self._mock_hf_api_response(groups),
            ):
                with patch("huggingface_hub.utils._http.hf_raise_for_status"):
                    result = uri.resolve_resource_group_id(
                        token="fake-token",
                        space_name="public",
                    )

        assert result == "staging-id-123"

    def test_prod_env_resolves_unsuffixed_name(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr("gbcommon.uri.hf.GB_ENVIRONMENT", "PROD")
        uri = HfURI.from_parts(owner="ibm-research", repo="dummy")
        groups = [{"name": "gbspace-public", "id": "prod-id-456"}]

        with patch("gbcommon.uri.hf.HfApi"):
            with patch(
                "huggingface_hub.utils._http.get_session",
                return_value=self._mock_hf_api_response(groups),
            ):
                with patch("huggingface_hub.utils._http.hf_raise_for_status"):
                    result = uri.resolve_resource_group_id(
                        token="fake-token",
                        space_name="public",
                    )

        assert result == "prod-id-456"

    def test_dev_env_resolves_suffixed_name(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr("gbcommon.uri.hf.GB_ENVIRONMENT", "DEV")
        uri = HfURI.from_parts(owner="ibm-research", repo="dummy")
        groups = [{"name": "gbspace-public-dev", "id": "dev-id-789"}]

        with patch("gbcommon.uri.hf.HfApi"):
            with patch(
                "huggingface_hub.utils._http.get_session",
                return_value=self._mock_hf_api_response(groups),
            ):
                with patch("huggingface_hub.utils._http.hf_raise_for_status"):
                    result = uri.resolve_resource_group_id(
                        token="fake-token",
                        space_name="public",
                    )

        assert result == "dev-id-789"

    def test_explicit_name_ignores_environment(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr("gbcommon.uri.hf.GB_ENVIRONMENT", "STAGING")
        uri = HfURI.from_parts(owner="ibm-research", repo="dummy")
        groups = [{"name": "my-custom-group", "id": "custom-id"}]

        with patch("gbcommon.uri.hf.HfApi"):
            with patch(
                "huggingface_hub.utils._http.get_session",
                return_value=self._mock_hf_api_response(groups),
            ):
                with patch("huggingface_hub.utils._http.hf_raise_for_status"):
                    result = uri.resolve_resource_group_id(
                        token="fake-token",
                        resource_group_name="my-custom-group",
                    )

        assert result == "custom-id"

    def test_raises_when_group_not_found(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr("gbcommon.uri.hf.GB_ENVIRONMENT", "STAGING")
        uri = HfURI.from_parts(owner="ibm-research", repo="dummy")
        groups = [{"name": "gbspace-other", "id": "other-id"}]

        with patch("gbcommon.uri.hf.HfApi"):
            with patch(
                "huggingface_hub.utils._http.get_session",
                return_value=self._mock_hf_api_response(groups),
            ):
                with patch("huggingface_hub.utils._http.hf_raise_for_status"):
                    with pytest.raises(ValueError, match="Could not resolve"):
                        uri.resolve_resource_group_id(
                            token="fake-token",
                            space_name="public",
                        )
