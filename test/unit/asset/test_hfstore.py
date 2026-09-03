"""Tests for Hfstore, covering bucket-specific behaviour."""

import subprocess
from pathlib import Path

import pytest
import yaml

from gbcommon.uri.hf import HfType, HfURI
from gbserver.asset.hfstore import Hfstore
from gbserver.types.artifact import ArtifactType
from gbserver.types.assetstoreconfig import AssetStoreConfig
from gbserver.utils.template import fill_template


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

    def test_private_only_at_top_level_not_nested(self):
        """``private`` must live only at the top level, never inside ``hf``.

        Every step template reads ``hfpush_config.private``; none reads
        ``hf.private``. Duplicating it into the nested block is a trap: the
        k8s/skypilot overlay rewrites ``hf.*`` from the raw push config without
        re-resolving ``private``, so a nested copy would silently carry the
        unresolved value. Guard that it stays absent.
        """
        for private in (True, False):
            uri = HfURI.from_parts(owner="org", repo="my-model", hf_type=HfType.MODEL)
            cfg = Hfstore.build_hfpush_step_config(
                hfuri=uri,
                binding_path="/tmp/x",
                binding_id="b-1",
                hf_private=private,
            )
            assert cfg["private"] is private
            assert "private" not in cfg["hf"]


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


class TestHfstoreEnterpriseOrganizations:
    """``enterprise_organizations`` from store.yaml drives the Enterprise split."""

    @staticmethod
    def _store(store_config):
        return Hfstore(AssetStoreConfig(**store_config))

    def test_absent_key_returns_none(self):
        """None (not []) so callers keep the pre-split "all Enterprise" behavior."""
        store = self._store({"base_uri": "hf:/", "config": {"token_secretname": "T"}})
        assert store.get_enterprise_organizations() is None

    def test_absent_config_block_returns_none(self):
        store = self._store({"base_uri": "hf:/"})
        assert store.get_enterprise_organizations() is None

    def test_no_store_config_returns_none(self):
        """A store built without a config (e.g. ad-hoc) must not blow up."""
        assert Hfstore(None).get_enterprise_organizations() is None

    def test_returns_configured_list(self):
        store = self._store(
            {
                "base_uri": "hf:/",
                "config": {"enterprise_organizations": ["ibm-research", "ibm-granite"]},
            }
        )
        assert store.get_enterprise_organizations() == ["ibm-research", "ibm-granite"]

    def test_explicit_empty_list_is_preserved(self):
        """[] is a real opt-out and must not collapse to None."""
        store = self._store(
            {"base_uri": "hf:/", "config": {"enterprise_organizations": []}}
        )
        assert store.get_enterprise_organizations() == []

    def test_explicit_null_returns_none(self):
        store = self._store(
            {"base_uri": "hf:/", "config": {"enterprise_organizations": None}}
        )
        assert store.get_enterprise_organizations() is None

    def test_non_list_raises(self):
        store = self._store(
            {"base_uri": "hf:/", "config": {"enterprise_organizations": "ibm-research"}}
        )
        with pytest.raises(ValueError, match="must be a list"):
            store.get_enterprise_organizations()

    def test_shipped_store_yaml_declares_enterprise_orgs(self):
        """Guard the shipped standalone config against accidental removal."""
        path = (
            Path(__file__).resolve().parents[3]
            / "configurations"
            / "assets"
            / "assetstores"
            / "hf"
            / "store.yaml"
        )
        cfg = yaml.safe_load(path.read_text())
        orgs = cfg["config"]["enterprise_organizations"]
        assert "ibm-research" in orgs
        assert "ibm-granite" in orgs


class TestLsfHfpushNoResourceGroup:
    """The LSF hfpush script must omit resourceGroupId when there is no group.

    Jinja renders a Python ``None`` as the literal string ``"None"``, which is
    non-empty to bash — without normalization the worker would POST
    ``"resourceGroupId":"None"`` and the HF create call would fail. This is the
    common case now that non-Enterprise orgs resolve to no resource group.
    """

    @staticmethod
    def _command_sh() -> str:
        return (
            Path(__file__).resolve().parents[3]
            / "src"
            / "gbserver"
            / "builtins"
            / "steps"
            / "lsf"
            / "hfpush"
            / "lsf_scripts"
            / "hfpush"
            / "command.sh"
        ).read_text()

    def test_script_normalizes_literal_none(self):
        """The guard that maps the rendered "None" back to empty is present."""
        sh = self._command_sh()
        assert '"${HF_RESOURCE_GROUP_ID}" == "None"' in sh
        assert 'HF_RESOURCE_GROUP_ID=""' in sh

    def test_create_body_omits_resource_group_when_none(self):
        """Render with resource_group_id=None and run the resulting bash logic."""
        rendered = fill_template(
            self._command_sh(),
            {
                "config": {
                    "hfpush_config": {
                        "hf": {"resource_group_id": None, "type": "model"},
                        "endpoint": "https://huggingface.co",
                        "owner": "my-user",
                        "repo": "my-model",
                        "revision": "main",
                        "private": True,
                        "binding_id": "b",
                        "path": "/tmp/x",
                        "uri": "hf:///my-user/my-model",
                    }
                }
            },
        )
        # Sanity: the raw render really does produce the literal "None".
        assert "HF_RESOURCE_GROUP_ID='None'" in rendered

        assigns = [
            line
            for line in rendered.splitlines()
            if line.startswith(("HF_RESOURCE_GROUP_ID=", "HF_OWNER=", "HF_TYPE="))
            or line.startswith("HF_REPO_NAME=")
        ]
        snippet = "\n".join(assigns) + (
            '\nif [[ "${HF_RESOURCE_GROUP_ID}" == "None" ]]; then'
            ' HF_RESOURCE_GROUP_ID=""; fi\n'
            'if [[ -n "${HF_RESOURCE_GROUP_ID}" ]]; then echo WITH_RG;'
            " else echo WITHOUT_RG; fi"
        )
        result = subprocess.run(
            ["bash", "-c", snippet], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "WITHOUT_RG"


class TestEnterpriseOrgListParity:
    """The CLI and server Enterprise lists must agree.

    The list is maintained in two places by necessity: the server reads
    `config.enterprise_organizations` from the hf asset store's store.yaml (which
    lives in a space's git repo), and the CLI cannot read that file, so it carries
    the same list on GBEnvConfig. If they diverge, the CLI and server disagree on
    whether an org is Enterprise — the CLI would reject a --resource-group-id the
    server requires, or vice versa. Only the shipped standalone store.yaml can be
    checked here; a remote space's copy is outside the repo.
    """

    @staticmethod
    def _shipped_store_orgs() -> list:
        path = (
            Path(__file__).resolve().parents[3]
            / "configurations"
            / "assets"
            / "assetstores"
            / "hf"
            / "store.yaml"
        )
        return yaml.safe_load(path.read_text())["config"]["enterprise_organizations"]

    def test_shipped_store_yaml_matches_every_environment_config(self):
        from gbcommon.types.gbenvconfig import _GB_ENVIRONMENT_CONFIGS

        expected = self._shipped_store_orgs()
        for env, config in _GB_ENVIRONMENT_CONFIGS.items():
            assert config.hf_enterprise_organizations == expected, (
                f"{env}'s hf_enterprise_organizations "
                f"({config.hf_enterprise_organizations}) does not match "
                f"enterprise_organizations in the shipped hf store.yaml "
                f"({expected}) — update both together"
            )

    def test_cli_constant_matches_the_shipped_store_yaml(self):
        """The value gbcli actually reads, not just the config it comes from."""
        from gbcli.utils.gbconstants import HF_ENTERPRISE_ORGANIZATIONS

        assert HF_ENTERPRISE_ORGANIZATIONS == self._shipped_store_orgs()

    def test_both_sides_classify_the_same_orgs(self):
        """Parity where it matters: the classifier agrees for either source."""
        from gbcli.utils.gbconstants import HF_ENTERPRISE_ORGANIZATIONS
        from gbcommon.utils.hf_utils import is_enterprise_hf_org

        store_orgs = self._shipped_store_orgs()
        for org in (*store_orgs, "my-user", "some-community-org", ""):
            assert is_enterprise_hf_org(org, HF_ENTERPRISE_ORGANIZATIONS) == (
                is_enterprise_hf_org(org, store_orgs)
            ), f"CLI and server disagree on '{org}'"
