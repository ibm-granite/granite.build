"""Tests for Hfstore, covering bucket-specific behaviour."""

import pytest

from gbcommon.uri.hf import HfType, HfURI
from gbserver.asset.hfstore import Hfstore


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
