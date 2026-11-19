# SPDX-License-Identifier: Apache-2.0

"""Unit tests for store-routed existence checks.

Covers ``select_assetstore`` / ``asset_uri_exists`` (the routing shared with
pull/push dispatch), ``Environment.load_declared_assetstores`` (an environment's
stores without constructing it), and ``Hfstore.uri_exists`` (the store's
``token_secretname`` reaches ``HfURI.exists``).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gbcommon.types.testing import ENV_VAR_GBTEST_MOCK_COS, ENV_VAR_GBTEST_MOCK_HF
from gbcommon.uri.hf import HfURI
from gbcommon.uri.uri import URI
from gbserver.asset.cosstore import Cosstore
from gbserver.asset.hfstore import Hfstore
from gbserver.environment.environment import (
    Environment,
    asset_uri_exists,
    select_assetstore,
)
from gbserver.types.assetstoreconfig import AssetStoreConfig

REPO = Path(__file__).resolve().parents[3]
S3_LOCAL_STORE = REPO / "configurations/assets/assetstores/s3-local"


@pytest.fixture(autouse=True)
def _unmocked(monkeypatch):
    """Run with COS and HF mocking off unless a test enables them."""
    monkeypatch.delenv(ENV_VAR_GBTEST_MOCK_COS, raising=False)
    monkeypatch.delenv(ENV_VAR_GBTEST_MOCK_HF, raising=False)


def _cos(base_uri: str) -> Cosstore:
    """A Cosstore for ``base_uri`` (no credentials needed by these tests)."""
    return Cosstore(AssetStoreConfig(base_uri=base_uri, config={}))


class TestSelectAssetstore:
    def test_longest_prefix_wins(self):
        """The most specific base_uri matching the URI is selected."""
        generic, local = _cos("s3://"), _cos("s3://local/")
        stores = {generic: "generic-entry", local: "local-entry"}
        assert select_assetstore(stores, URI.get_uri("s3://local/a")) == (
            local,
            "local-entry",
        )
        assert select_assetstore(stores, URI.get_uri("s3://other/a")) == (
            generic,
            "generic-entry",
        )

    def test_no_match(self):
        """No store handling the URI yields (None, None)."""
        assert select_assetstore({_cos("cos://"): "e"}, URI.get_uri("s3://b/k")) == (
            None,
            None,
        )


class TestAssetUriExists:
    def test_routes_to_the_selected_store(self):
        """The selected store's uri_exists answers."""
        store = MagicMock(spec=Cosstore)
        store.can_handle_len.return_value = 5
        store.uri_exists.return_value = True
        uri = URI.get_uri("s3://b/k")
        assert asset_uri_exists({store: "e"}, uri)
        store.uri_exists.assert_called_once_with(uri)

    def test_falls_back_to_uri_exists(self, tmp_path):
        """With no matching store, the URI checks itself."""
        existing = URI.get_uri(tmp_path.as_uri())
        assert asset_uri_exists({}, existing)
        assert not asset_uri_exists({}, URI.get_uri((tmp_path / "nope").as_uri()))


class TestLoadDeclaredAssetstores:
    def test_loads_stores_without_building_the_env(self, tmp_path):
        """The declared stores load from environment.yaml alone."""
        (tmp_path / "environment.yaml").write_text(
            "name: t\ntype: Skypilot\nconfig: {}\n"
            f"assetstores:\n  - store_uri: {S3_LOCAL_STORE.as_uri()}\n"
        )
        with patch("gbserver.environment.skypilot.Skypilot.__init__") as init:
            stores = Environment.load_declared_assetstores(tmp_path.as_uri())
        init.assert_not_called()
        (store,) = stores
        assert isinstance(store, Cosstore)
        assert store.config.base_uri == "s3://local/"
        assert stores[store].store_uri == S3_LOCAL_STORE.as_uri()


class TestHfstoreUriExists:
    def test_uses_the_store_token_secret_name(self):
        """A non-default token_secretname is resolved and passed to the check."""
        store = Hfstore(
            AssetStoreConfig(base_uri="hf:/", config={"token_secretname": "MY_HF"}),
            secrets={"MY_HF": "store-token", "HF_TOKEN": "default-token"},
        )
        uri = URI.get_uri("hf://huggingface.co/datasets/org/repo")
        with patch.object(HfURI, "exists", return_value=True) as exists:
            assert store.uri_exists(uri)
        assert exists.call_args.kwargs == {"token": "store-token"}

    def test_mocked_needs_no_token(self, monkeypatch):
        """With HF mocked, the check is True without any token or Hub call."""
        monkeypatch.setenv(ENV_VAR_GBTEST_MOCK_HF, "true")
        store = Hfstore(AssetStoreConfig(base_uri="hf:/", config={}), secrets={})
        assert store.uri_exists(URI.get_uri("hf://huggingface.co/datasets/org/repo"))
