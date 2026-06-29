"""Tests for Hfstore, covering bucket-specific behaviour."""

from pathlib import Path

import yaml

from gbcommon.uri.hf import HfType, HfURI
from gbserver.asset.hfstore import Hfstore
from gbserver.types.artifact import ArtifactType


class TestHfstoreAssetType:
    def test_bucket_returns_bucket(self):
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)
        store = Hfstore(uri)
        assert store.get_asset_type(uri) == ArtifactType.BUCKET

    def test_model_returns_model(self):
        uri = HfURI.from_parts(owner="org", repo="my-model", hf_type=HfType.MODEL)
        store = Hfstore(uri)
        assert store.get_asset_type(uri) == ArtifactType.MODEL

    def test_dataset_returns_dataset(self):
        uri = HfURI.from_parts(owner="org", repo="my-dataset", hf_type=HfType.DATASET)
        store = Hfstore(uri)
        assert store.get_asset_type(uri) == ArtifactType.DATASET

    def test_no_type_defaults_to_model(self):
        uri = HfURI.from_parts(owner="org", repo="my-repo")
        store = Hfstore(uri)
        assert store.get_asset_type(uri) == ArtifactType.MODEL


class TestHfstoreRelpath:
    def test_bucket_omits_revision(self):
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)
        store = Hfstore(uri)
        assert store.get_relpath(uri) == "org/my-bucket"

    def test_model_includes_revision(self):
        uri = HfURI.from_parts(
            owner="org", repo="my-model", hf_type=HfType.MODEL, revision="v1.0"
        )
        store = Hfstore(uri)
        assert store.get_relpath(uri) == "org/my-model/v1.0"

    def test_dataset_includes_revision(self):
        uri = HfURI.from_parts(owner="org", repo="my-dataset", hf_type=HfType.DATASET)
        store = Hfstore(uri)
        assert store.get_relpath(uri) == "org/my-dataset/main"


class TestHfstoreStepConfigEndpoint:
    """The step config dicts include an `endpoint` key derived from the
    URI host so step.yaml jinja templates and bash exports can pick it
    up uniformly."""

    def test_hfpush_step_config_default_host(self):
        uri = HfURI.from_parts(owner="org", repo="my-model", hf_type=HfType.MODEL)
        cfg = Hfstore.build_hfpush_step_config(
            hfuri=uri,
            binding_path="/tmp/x",
            binding_id="b-1",
            hf_private=True,
        )
        assert cfg["endpoint"] == "https://huggingface.co"

    def test_hfpush_step_config_custom_host(self):
        uri = HfURI.from_parts(
            owner="org",
            repo="my-model",
            hf_type=HfType.MODEL,
            host="my-enterprise.example.com",
        )
        cfg = Hfstore.build_hfpush_step_config(
            hfuri=uri,
            binding_path="/tmp/x",
            binding_id="b-1",
            hf_private=True,
        )
        assert cfg["endpoint"] == "https://my-enterprise.example.com"

    def test_hfpull_step_config_default_host(self):
        uri = HfURI.from_parts(owner="org", repo="my-model", hf_type=HfType.MODEL)
        cfg = Hfstore.build_hfpull_step_config(hfuri=uri, binding_path="/tmp/x")
        assert cfg["endpoint"] == "https://huggingface.co"

    def test_hfpull_step_config_custom_host(self):
        uri = HfURI.from_parts(
            owner="org",
            repo="my-model",
            hf_type=HfType.MODEL,
            host="my-enterprise.example.com",
        )
        cfg = Hfstore.build_hfpull_step_config(hfuri=uri, binding_path="/tmp/x")
        assert cfg["endpoint"] == "https://my-enterprise.example.com"


class TestHfstoreStepConfigPathInRepo:
    """The push step config pre-resolves ``path_in_repo`` from the URI so the
    skypilot worker's inline push needs no URI parser. ``hf.type`` is carried
    too, so the worker can branch repo vs bucket without re-parsing."""

    def test_path_in_repo_empty_when_absent(self):
        uri = HfURI.from_parts(owner="org", repo="my-dataset", hf_type=HfType.DATASET)
        cfg = Hfstore.build_hfpush_step_config(
            hfuri=uri, binding_path="/tmp/x", binding_id="b-1", hf_private=True
        )
        assert cfg["path_in_repo"] == ""
        assert cfg["hf"]["type"] == "dataset"

    def test_path_in_repo_carried_through(self):
        uri = HfURI.from_parts(
            owner="org",
            repo="my-model",
            hf_type=HfType.MODEL,
            revision="main",
            path_in_repo="sub/dir/file.bin",
        )
        cfg = Hfstore.build_hfpush_step_config(
            hfuri=uri, binding_path="/tmp/x", binding_id="b-1", hf_private=True
        )
        assert cfg["path_in_repo"] == "sub/dir/file.bin"

    def test_bucket_type_carried_through(self):
        uri = HfURI.from_parts(owner="org", repo="my-bucket", hf_type=HfType.BUCKET)
        cfg = Hfstore.build_hfpush_step_config(
            hfuri=uri, binding_path="/tmp/x", binding_id="b-1", hf_private=True
        )
        assert cfg["hf"]["type"] == "bucket"
        assert cfg["path_in_repo"] == ""


class TestSkypilotHfpushStepParity:
    """Guard against drift between the skypilot hfpush step's inline python and
    HfURI.push() (src/gbcommon/uri/hf.py). The skypilot worker has no gbserver
    install, so push() is reimplemented inline in step.yaml; this test asserts
    that reimplementation still covers every branch/behaviour push() has."""

    STEP_YAML = (
        Path(__file__).resolve().parents[3]
        / "src/gbserver/builtins/steps/skypilot/hfpush/step.yaml"
    )

    def _run_script(self) -> str:
        doc = yaml.safe_load(self.STEP_YAML.read_text())
        return doc["environment_configs"]["Skypilot"]["launchers"]["hfpush"]["config"][
            "run"
        ]

    def test_covers_repo_and_bucket_apis(self):
        run = self._run_script()
        for api in (
            "create_repo",
            "upload_file",
            "upload_folder",
            "create_bucket",
            "batch_bucket_files",
            "sync_bucket",
        ):
            assert api in run, f"hfpush step missing HF API call: {api}"

    def test_covers_empty_source_validation(self):
        run = self._run_script()
        assert "refusing to push zero-length file" in run
        assert "refusing to push directory with no non-empty files" in run

    def test_covers_error_status_classification(self):
        run = self._run_script()
        # Mirrors _classify_hf_error severities: 429 / 5xx / 401,403 / 404.
        assert "429" in run
        assert "500" in run
        assert "401" in run and "403" in run
        assert "404" in run

    def test_preserves_success_line_for_monitor(self):
        run = self._run_script()
        # The skypilot_monitor greps this exact line; it must be emitted by bash.
        assert 'echo "Pushed HF URI: ${HF_URI} for binding ${BINDING_ID}"' in run

    def test_no_longer_uses_hf_upload_cli(self):
        doc = yaml.safe_load(self.STEP_YAML.read_text())
        cfg = doc["environment_configs"]["Skypilot"]["launchers"]["hfpush"]["config"]
        assert "hf upload" not in cfg["run"]
        # CLI extra dropped since the upload is done via HfApi now.
        assert "[cli]" not in cfg["setup"]
