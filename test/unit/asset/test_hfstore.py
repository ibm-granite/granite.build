"""Tests for Hfstore, covering bucket-specific behaviour."""

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
