"""Unit tests for Skypilot S3/COS push/pull and the shared Cosstore helpers.

Covers endpoint normalization, per-store settings, hf-style credential
resolution (store.yaml names the secrets; values come from the space secrets,
then the same-named env var), the env methods' step configs, multi-endpoint
routing, and CosURI.exists() (mocked, and a real HEAD/LIST against a stub S3
client).
"""

import asyncio
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gbcommon.types.testing import ENV_VAR_GBTEST_MOCK_COS
from gbcommon.uri import cos as cos_module
from gbcommon.uri.uri import URI
from gbserver.asset.cosstore import Cosstore, normalize_s3_endpoint
from gbserver.environment._skypilot_cos import (
    CosSettings,
    cos_launcher_envs,
    resolve_cos_store_settings,
)
from gbserver.types.assetstoreconfig import AssetStoreConfig
from gbserver.types.buildconfig import BuildTargetStepConfig

SECRETS = {
    "COS_ACCESS_KEY_ID": "local-key-id",
    "COS_SECRET_ACCESS_KEY": "local-secret-value",
    "IBMCOS_KEY_ID": "ibm-key-id",
    "IBMCOS_SECRET": "ibm-secret-value",
}


def _store(base_uri: str, secrets=None, **config) -> Cosstore:
    """Build a real Cosstore for ``base_uri`` with the given store config.

    :param base_uri: the store's base_uri (selects the Cosstore class).
    :param secrets: the store's secret bag (defaults to :data:`SECRETS`).
    :param config: store ``config`` keys (cos_endpoint, cos_*_secret_name, ...).
    :returns: the Cosstore.
    """
    return Cosstore(
        AssetStoreConfig(base_uri=base_uri, config=config),
        secrets=dict(SECRETS) if secrets is None else secrets,
    )


def _local_store(**overrides) -> Cosstore:
    """The s3-local store: ``s3://local/`` aliasing the ``gb-checkpoints`` bucket."""
    config = {
        "cos_bucket_name": "gb-checkpoints",
        "cos_endpoint": "http://gb-s3:9000",
        "cos_gbserver_endpoint": "http://localhost:9000",
        "cos_region": "us-east-1",
        "cos_access_key_id_secret_name": "COS_ACCESS_KEY_ID",
        "cos_secret_access_key_secret_name": "COS_SECRET_ACCESS_KEY",
    }
    config.update(overrides)
    return _store("s3://local/", **config)


def _ibm_store() -> Cosstore:
    """An IBM COS store serving ``cos://`` URIs with its own secret names."""
    return _store(
        "cos://",
        cos_endpoint="s3.us-east.cloud-object-storage.appdomain.cloud",
        cos_region="us-east",
        cos_access_key_id_secret_name="IBMCOS_KEY_ID",
        cos_secret_access_key_secret_name="IBMCOS_SECRET",
    )


@pytest.fixture
def skypilot_env():
    """A minimal Skypilot env on a cluster with a shared workdir."""
    from gbserver.environment.skypilot import Skypilot
    from gbserver.types.environmentconfig import EnvironmentConfig

    config = EnvironmentConfig(
        name="test-skypilot",
        type="Skypilot",
        config={"default_cloud": "slurm", "shared_workdir": "/shared/gbroot"},
    )
    return Skypilot(event_q=asyncio.Queue(), environment_config=config)


@pytest.fixture(autouse=True)
def _unmocked(monkeypatch):
    """Run every test with COS mocking off unless a test enables it."""
    monkeypatch.delenv(ENV_VAR_GBTEST_MOCK_COS, raising=False)


def _storeload(**config):
    """A StoreLoad/StorePush stand-in with mode 'default' and ``config``."""
    store_config = MagicMock()
    store_config.mode = "default"
    store_config.config = config
    return store_config


class TestNormalizeS3Endpoint:
    @pytest.mark.parametrize(
        "endpoint,expected",
        [
            ("", ""),
            (None, ""),
            ("s3.amazonaws.com", "https://s3.amazonaws.com"),
            ("https://s3.example.com", "https://s3.example.com"),
            ("http://gb-s3:9000", "http://gb-s3:9000"),
            ("http://localhost:9000", "http://localhost:9000"),
            ("http://127.0.0.1:9000", "http://127.0.0.1:9000"),
        ],
    )
    def test_accepted(self, endpoint, expected):
        """Bare hosts get https; https and private-host http pass through."""
        assert normalize_s3_endpoint(endpoint) == expected

    @pytest.mark.parametrize(
        "endpoint", ["http://s3.example.com", "http://10.0.0.5:9000", "ftp://x"]
    )
    def test_rejected(self, endpoint):
        """Plain http to a routable host, and other schemes, are refused."""
        with pytest.raises(ValueError, match="unsupported S3 endpoint"):
            normalize_s3_endpoint(endpoint)


class TestResolveCosStoreSettings:
    def test_settings_come_from_store(self):
        """The s3 URI, cluster endpoint and region are read from the given store."""
        uri = URI.get_uri("s3://local/run1/model")
        assert resolve_cos_store_settings(uri, _local_store()) == CosSettings(
            s3_uri="s3://gb-checkpoints/run1/model",
            endpoint_url="http://gb-s3:9000",
            region="us-east-1",
        )

    def test_cos_scheme_becomes_s3(self):
        """A cos:// URI is handed to the AWS CLI as s3://."""
        uri = URI.get_uri("cos://bucket/data/x.jsonl")
        assert resolve_cos_store_settings(uri, _ibm_store()).s3_uri == (
            "s3://bucket/data/x.jsonl"
        )

    def test_cos_bucket_name_override(self):
        """A store cos_bucket_name replaces the URI's bucket (as on K8s/LSF)."""
        store = _store("s3://alias/", cos_bucket_name="real-bucket")
        uri = URI.get_uri("s3://alias/p/q")
        assert resolve_cos_store_settings(uri, store).s3_uri == "s3://real-bucket/p/q"

    def test_bucket_only_uri(self):
        """A bucket-only URI maps to s3://bucket."""
        uri = URI.get_uri("s3://local")
        assert resolve_cos_store_settings(uri, _local_store()).s3_uri == (
            "s3://gb-checkpoints"
        )


class TestResolveCredentials:
    """Cosstore.resolve_credentials mirrors Hfstore.resolve_token."""

    def test_from_store_secrets_by_configured_names(self):
        """The store.yaml secret names are looked up in the space secrets."""
        uri = URI.get_uri("cos://b/k")
        assert _ibm_store().resolve_credentials(uri) == ("ibm-key-id", "ibm-secret-value")

    def test_env_var_fallback(self, monkeypatch):
        """A name absent from the secrets falls back to the same-named env var."""
        monkeypatch.setenv("COS_ACCESS_KEY_ID", "env-id")
        monkeypatch.setenv("COS_SECRET_ACCESS_KEY", "env-secret")
        store = _store("s3://", secrets={})
        assert store.resolve_credentials(URI.get_uri("s3://b/k")) == ("env-id", "env-secret")

    def test_blank_or_missing_is_none(self, monkeypatch):
        """Blank values count as unset; a missing credential yields None."""
        monkeypatch.delenv("COS_SECRET_ACCESS_KEY", raising=False)
        store = _store("s3://", secrets={"COS_ACCESS_KEY_ID": " "})
        assert store.resolve_credentials(URI.get_uri("s3://b/k")) is None


class TestCosLauncherEnvs:
    def test_passes_values_as_aws_envs(self):
        """Credentials reach the step as AWS_* launcher envs (like HF_TOKEN)."""
        envs = cos_launcher_envs(URI.get_uri("cos://b/k"), _ibm_store())
        assert envs == {
            "AWS_ACCESS_KEY_ID": "ibm-key-id",
            "AWS_SECRET_ACCESS_KEY": "ibm-secret-value",
        }

    def test_missing_fails_fast_without_values(self, monkeypatch):
        """Missing credentials raise without revealing any secret value."""
        monkeypatch.delenv("NOPE", raising=False)
        store = _local_store(cos_secret_access_key_secret_name="NOPE")
        with pytest.raises(ValueError) as exc:
            cos_launcher_envs(URI.get_uri("s3://local/k"), store)
        assert "local-key-id" not in str(exc.value)

    def test_mocked_needs_no_credentials(self, monkeypatch):
        """Under GBTEST_MOCK_COS no credentials are required or passed."""
        monkeypatch.setenv(ENV_VAR_GBTEST_MOCK_COS, "true")
        store = _store("s3://", secrets={})
        assert cos_launcher_envs(URI.get_uri("s3://b/k"), store) == {}


class TestPullassetCosstore:
    @pytest.mark.asyncio
    async def test_binding_and_step_config(self, skypilot_env):
        """Default cache is {shared_workdir}/cos_cache/<uri hash>; builtin s3pull."""
        uri = URI.get_uri("s3://local/in/data.jsonl")
        binding, step = await skypilot_env.pullasset_cosstore(
            uri=uri, assetstore=_local_store(), storeload_config=_storeload()
        )
        expected = str(Path("/shared/gbroot/cos_cache") / uri.hash())
        assert binding == {"binding": {"path": expected}}
        assert isinstance(step, BuildTargetStepConfig)
        assert step.step_uri == "space://steps/s3pull"
        assert step.config["s3pull_config"] == {
            "uri": "s3://local/in/data.jsonl",
            "s3_uri": "s3://gb-checkpoints/in/data.jsonl",
            "local_path": expected,
            "endpoint_url": "http://gb-s3:9000",
            "region": "us-east-1",
        }
        assert step.config["launcher_config"]["envs"]["AWS_ACCESS_KEY_ID"] == (
            "local-key-id"
        )

    @pytest.mark.asyncio
    async def test_cache_path_and_step_uri_override(self, skypilot_env):
        """storeload cache_path and step_uri override the defaults."""
        uri = URI.get_uri("s3://local/in")
        binding, step = await skypilot_env.pullasset_cosstore(
            uri=uri,
            assetstore=_local_store(),
            storeload_config=_storeload(
                cache_path="/data/c", step_uri="file:///custom/s3pull"
            ),
        )
        assert binding["binding"]["path"] == str(Path("/data/c") / uri.hash())
        assert step.step_uri == "file:///custom/s3pull"

    @pytest.mark.asyncio
    async def test_rejects_non_cosstore(self, skypilot_env):
        """A non-Cosstore assetstore is rejected."""
        with pytest.raises(AssertionError, match="Cosstore"):
            await skypilot_env.pullasset_cosstore(
                uri=URI.get_uri("s3://b/k"),
                assetstore=MagicMock(),
                storeload_config=_storeload(),
            )


class TestPushassetCosstore:
    @pytest.mark.asyncio
    async def test_step_config(self, skypilot_env):
        """Push queues the builtin s3push with the binding path as source."""
        step = await skypilot_env.pushasset_cosstore(
            binding={"path": "out/model"},
            binding_id="out-1",
            uri="s3://local/run1/model",
            assetstore=_local_store(),
            storepush_config=_storeload(),
        )
        assert step.step_uri == "space://steps/s3push"
        push_config = step.config["s3push_config"]
        assert push_config["local_path"] == "out/model"
        # The artifact URI and binding id are echoed back by the step so its
        # monitor can mark the artifact pushed; s3_uri is the real bucket.
        assert push_config["uri"] == "s3://local/run1/model"
        assert push_config["binding_id"] == "out-1"
        assert step.config["s3push_config"]["s3_uri"] == (
            "s3://gb-checkpoints/run1/model"
        )

    @pytest.mark.asyncio
    async def test_empty_uri_raises(self, skypilot_env):
        """An empty destination URI is an error."""
        with pytest.raises(ValueError, match="Empty uri"):
            await skypilot_env.pushasset_cosstore(
                binding={"path": "x"}, uri="", assetstore=_local_store()
            )


class TestMultipleEndpoints:
    @pytest.mark.asyncio
    async def test_each_uri_uses_its_own_store(self, skypilot_env):
        """With several Cosstores, dispatch's longest-prefix match picks the
        store, and each step carries that store's endpoint and credentials."""
        local, ibm = _local_store(), _ibm_store()
        aws = _store("s3://", cos_endpoint="s3.amazonaws.com")
        skypilot_env.supported_assetstores = {s: MagicMock() for s in (local, ibm, aws)}

        cases = {
            "s3://local/in": (local, "http://gb-s3:9000", "local-key-id"),
            "cos://ibm-bucket/out": (
                ibm,
                "https://s3.us-east.cloud-object-storage.appdomain.cloud",
                "ibm-key-id",
            ),
            "s3://other/x": (aws, "https://s3.amazonaws.com", "local-key-id"),
        }
        for uristr, (want_store, endpoint, key_id) in cases.items():
            uri = URI.get_uri(uristr)
            store, _ = skypilot_env._get_storeconfig(uri)
            assert store is want_store, uristr
            step = await skypilot_env.pushasset_cosstore(
                binding={"path": "out"}, uri=uri, assetstore=store
            )
            assert step.config["s3push_config"]["endpoint_url"] == endpoint
            assert step.config["launcher_config"]["envs"]["AWS_ACCESS_KEY_ID"] == key_id


class _ClientError(Exception):
    """Stand-in for botocore ClientError carrying an HTTP status."""

    def __init__(self, status: int):
        super().__init__(f"status {status}")
        self.response = {"ResponseMetadata": {"HTTPStatusCode": status}}


@pytest.fixture
def stub_s3():
    """Route CosURI.exists() to the local store and a stub S3 client."""
    store = _local_store()
    client = MagicMock()
    with patch.object(
        cos_module, "_check_resolver", lambda uri: store.get_check_target(uri)
    ), patch.object(cos_module, "_make_s3_client", return_value=client) as make, patch(
        "botocore.exceptions.ClientError", _ClientError
    ):
        yield client, make


class TestCosURIExists:
    def test_mocked_is_true(self, monkeypatch):
        """Like HfURI.exists: True whenever GBTEST_MOCK_COS is set."""
        monkeypatch.setenv(ENV_VAR_GBTEST_MOCK_COS, "1")
        with patch.object(cos_module, "_check_resolver", None):
            assert URI.get_uri("s3://b/k").exists()

    def test_no_resolver_is_false(self):
        """Without a store resolver the check cannot run: False."""
        with patch.object(cos_module, "_check_resolver", None):
            assert not URI.get_uri("s3://b/k").exists()

    def test_object_found_uses_gbserver_endpoint(self, stub_s3):
        """An object HEAD succeeds; the check uses cos_gbserver_endpoint."""
        client, make = stub_s3
        assert URI.get_uri("s3://local/in/data.txt").exists()
        client.head_object.assert_called_once_with(
            Bucket="gb-checkpoints", Key="in/data.txt"
        )
        assert make.call_args.args[0].endpoint_url == "http://localhost:9000"

    def test_prefix_found_after_object_miss(self, stub_s3):
        """A 404 object HEAD falls back to a prefix LIST."""
        client, _ = stub_s3
        client.head_object.side_effect = _ClientError(404)
        client.list_objects_v2.return_value = {"KeyCount": 1}
        assert URI.get_uri("s3://local/in").exists()
        client.list_objects_v2.assert_called_once_with(
            Bucket="gb-checkpoints", Prefix="in/", MaxKeys=1
        )

    def test_missing_is_false(self, stub_s3):
        """No object and an empty prefix: False."""
        client, _ = stub_s3
        client.head_object.side_effect = _ClientError(404)
        client.list_objects_v2.return_value = {"KeyCount": 0}
        assert not URI.get_uri("s3://local/none").exists()

    def test_error_is_false(self, stub_s3):
        """An auth/server error is a (logged) False, not an exception."""
        client, _ = stub_s3
        client.head_object.side_effect = _ClientError(500)
        assert not URI.get_uri("s3://local/k").exists()

    def test_secret_not_in_repr(self, stub_s3):
        """The check target never prints its secret."""
        target = _local_store().get_check_target(URI.get_uri("s3://local/k"))
        assert "local-secret-value" not in repr(target)


class TestGetCosCacheDir:
    def test_explicit_cache_path_wins(self):
        """storeload cache_path is used as-is."""
        from gbserver.environment.local_assets import get_cos_cache_dir

        assert get_cos_cache_dir(_storeload(cache_path="/c"), "/shared") == "/c"

    def test_shared_workdir_default(self):
        """Without cache_path, the cache lives under the shared workdir."""
        from gbserver.environment.local_assets import get_cos_cache_dir

        assert get_cos_cache_dir(_storeload(), "/shared") == "/shared/cos_cache"

    def test_home_fallback(self):
        """With neither, a process-local ~/.cache/gbserver/cos is used."""
        from gbserver.environment.local_assets import get_cos_cache_dir

        assert get_cos_cache_dir(None).endswith(str(Path(".cache/gbserver/cos")))


class TestCosMockHelpers:
    def test_enable_disable_restores_prior(self, monkeypatch):
        """enable/disable_cos_mocks toggle is_cos_mocked and restore the prior value."""
        from gbcommon.types.testing import (
            disable_cos_mocks,
            enable_cos_mocks,
            get_exported_gbtest_env_vars,
            is_cos_mocked,
        )

        monkeypatch.setenv(ENV_VAR_GBTEST_MOCK_COS, "false")
        enable_cos_mocks()
        assert is_cos_mocked()
        assert get_exported_gbtest_env_vars()[ENV_VAR_GBTEST_MOCK_COS] == "true"
        disable_cos_mocks()
        assert not is_cos_mocked()
        assert os.environ[ENV_VAR_GBTEST_MOCK_COS] == "false"
