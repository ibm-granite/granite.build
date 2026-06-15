import importlib
import os
import re
from typing import Optional

import pytest
from pydantic import BaseModel

try:
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    from ibm_secrets_manager_sdk.secrets_manager_v2 import SecretsManagerV2

    _HAS_IBM_SDK = True
except ImportError:
    _HAS_IBM_SDK = False


def _can_import(*modules: str) -> bool:
    """Return True if all named modules are importable."""
    for mod in modules:
        if importlib.util.find_spec(mod) is None:
            return False
    return True


# Conditionally ignore test files whose dependencies are not installed.
# When the full venv (.[all,dev]) is active these are all importable and
# every test is collected. With a minimal / standalone venv the files are
# silently skipped instead of raising ImportError during collection.
collect_ignore: list[str] = []

if not _can_import("psutil"):
    collect_ignore += [
        "integration/ibm/buildrunner/k8s/test_buildrunner_1step_cpu.py",
        "integration/ibm/buildrunner/k8s/test_buildrunner_1step_gpu.py",
        "integration/ibm/buildrunner/k8s/test_buildrunner_retry.py",
        "integration/ibm/buildrunner/k8s/test_buildrunnerjob.py",
        "integration/ibm/buildrunner/k8s/test_builds.py",
        "integration/ibm/buildwatcher/test_buildwatcher.py",
        "integration/ibm/githubmanager/test_githubmanager_subselect_targets.py",
        "e2e/sidecar/test_multi_sidecar_cmdmon_delayed_pytest.py",
        "e2e/sidecar/test_multi_sidecar_cmdmon_pytest.py",
        "e2e/sidecar/test_multi_sidecar_pytest.py",
        "e2e/sidecar/test_sidecar_cmdmon_delayed_pytest.py",
        "e2e/sidecar/test_sidecar_cmdmon_pytest.py",
        "e2e/sidecar/test_sidecar_pytest.py",
        "e2e/sidecar/test_sidecar_tuning_pytest.py",
    ]

if not _can_import("kubernetes_asyncio"):
    collect_ignore.append("unit/resilience/test_k8s_retry.py")
    collect_ignore.append("unit/monitoring/test_appwrapper_monitor.py")
    collect_ignore.append("unit/environment/test_cleanup_retry.py")
    collect_ignore.append("integration/ibm/environment/test_k8s_raycluster_cleanup.py")

if not _can_import("asyncssh"):
    collect_ignore.append("integration/ibm/utils/test_ssh_tunnel.py")


import libgbtest
import libgbtest.constants
from libgbtest.constants import BUILD_ID_PATTERN

import gbserver.types.constants
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

SECRET_TYPE = "arbitrary"
SPS_SECRET_GROUP_NAME = "SPS-Secret-Group"

ENV_VAR_SPS_IBMCLOUD_API_KEY = "GBTEST_SPS_IBMCLOUD_API_KEY"

# By default, have the don't allow local env vars to override SPS secret values.
ENV_VAR_SPS_ENABLE_ENV_VAR_OVERRIDE = "GBTEST_SPS_ENABLE_ENV_VAR_OVERRIDE"
ENABLE_ENV_VAR_OVERRIDE_DEFAULT = "False"

SECRET_MANAGER_ENDPOINT = "https://c78b6ab2-edd0-407e-afac-5892d6017045.us-south.secrets-manager.appdomain.cloud"

TEST_REQUIRED_ENV_VARS = [
    # GB Server Configuration
    "GB_ENVIRONMENT",
    "GBSERVER_RAISE_BUILD_EXCEPTIONS",  # Have build tests fail on buildrunner exceptions
    # GB Test Configuration
    "GBTEST_HAS_COMPUTE_CLUSTER_ACCESS",
    "GBTEST_HAS_GB_CLUSTER_ACCESS",
    # GB Server Secrets
    "GBSERVER_GITHUB_TOKEN",
    "GBSERVER_SQL_PASSWD",
    "GBSERVER_SQL_SSLROOT_CERT_BASE64",
    "GITHUB_TOKEN",
    "IBM_CLOUD_API_KEY",
    "LAKEHOUSE_TOKEN",
    "HF_TOKEN",
    # GB Test Secrets
    "GBTEST_GB_CLUSTER_API_KEY",
    "GBTEST_ADMIN_GITHUB_TOKEN",
    "GBTEST_NON_ADMIN_GITHUB_TOKEN",
    "GBSERVER_WANDB_ENTITY",
    "GBSERVER_WANDB_API_KEY",
]

DEFAULT_NON_SECRET_ENV_VAR_VALUES = {}
DEFAULT_NON_SECRET_ENV_VAR_VALUES["GB_ENVIRONMENT"] = "DEV"
DEFAULT_NON_SECRET_ENV_VAR_VALUES["GBTEST_HAS_COMPUTE_CLUSTER_ACCESS"] = "True"
DEFAULT_NON_SECRET_ENV_VAR_VALUES["GBTEST_HAS_GB_CLUSTER_ACCESS"] = "True"
DEFAULT_NON_SECRET_ENV_VAR_VALUES["GBSERVER_RAISE_BUILD_EXCEPTIONS"] = "True"

# SPS id of the secret
TEST_ENV_VAR_SPS_NAMES = {}
# GHE token for Granite.Dot.Build.Test
TEST_ENV_VAR_SPS_NAMES["GBSERVER_GITHUB_TOKEN"] = "github-token"
TEST_ENV_VAR_SPS_NAMES["GITHUB_TOKEN"] = "github-token"
TEST_ENV_VAR_SPS_NAMES["GBTEST_NON_ADMIN_GITHUB_TOKEN"] = "github-token"
# GHE token for Granite.Dot.Build.Test.Admin
TEST_ENV_VAR_SPS_NAMES["GBTEST_ADMIN_GITHUB_TOKEN"] = "github-admin-token"
TEST_ENV_VAR_SPS_NAMES["GBSERVER_SQL_PASSWD"] = "gbserver-sql-passwd"
TEST_ENV_VAR_SPS_NAMES["GBSERVER_SQL_SSLROOT_CERT_BASE64"] = (
    "gbserver-sql-sslroot-cert-base64"
)
TEST_ENV_VAR_SPS_NAMES["LAKEHOUSE_TOKEN"] = "lakehouse-token"
# RIS3
TEST_ENV_VAR_SPS_NAMES["IBM_CLOUD_API_KEY"] = "ris3-api-key"

# Get the cluster API key secret name from environment variable
gb_api_key_secret_name = os.getenv("GBTEST_GB_API_KEY_SECRET_NAME", "vpc-api-key")
TEST_ENV_VAR_SPS_NAMES["GBTEST_GB_CLUSTER_API_KEY"] = gb_api_key_secret_name
TEST_ENV_VAR_SPS_NAMES["HF_TOKEN"] = "hf-token"
TEST_ENV_VAR_SPS_NAMES["GBSERVER_WANDB_ENTITY"] = "gbserver-wandb-entry"
TEST_ENV_VAR_SPS_NAMES["GBSERVER_WANDB_API_KEY"] = "gbserver-wandb-api-key"


# Set each required environment variable, either from value or secret manager
def set_test_env(sps_api_key: str, enable_env_var_override: bool):
    """Set up the test env vars using one of 1) local env vars, 2) default values or 3) sps secrets.
    if enable_env_var_override is True, then let local env vars supersede secret values.

    Args:
        sps_api_key (str): _description_
        enable_env_var_override (bool):  if True, then let local env vars supersede secret values.
    """
    if not _HAS_IBM_SDK:
        logger.info(
            "IBM SDK not installed — skipping SPS secret loading. Install ibm_cloud_sdk_core for full test env setup."
        )
        return
    authenticator = IAMAuthenticator(sps_api_key)
    secrets_manager_service = SecretsManagerV2(authenticator=authenticator)
    secrets_manager_service.set_service_url(SECRET_MANAGER_ENDPOINT)
    # breakpoint() # Debugging.
    for env_var in TEST_REQUIRED_ENV_VARS:
        env_var_value = os.getenv(env_var, None)
        if env_var in TEST_ENV_VAR_SPS_NAMES:
            # Handle secrets
            if env_var_value is not None and enable_env_var_override:
                value = env_var_value
                value_source = "Local Environment"
            else:
                secret_name = TEST_ENV_VAR_SPS_NAMES[env_var]
                response = secrets_manager_service.get_secret_by_name_type(
                    secret_type=SECRET_TYPE,
                    name=secret_name,
                    secret_group_name=SPS_SECRET_GROUP_NAME,
                )
                secret_payload = response.get_result()
                value = secret_payload["payload"]
                value_source = "Secret Manager"
        elif env_var in DEFAULT_NON_SECRET_ENV_VAR_VALUES:
            # Handle non-secrets with default values.
            if env_var_value is None:
                value = DEFAULT_NON_SECRET_ENV_VAR_VALUES[env_var]
                value_source = "Default Value"
            else:
                value = env_var_value
                value_source = "Local Environment"

        if value is None:
            logger.warning(f"Potential missing Environment Variable: {env_var}")
        elif env_var in TEST_ENV_VAR_SPS_NAMES:
            logger.info(
                f"Setting Environment Variable from {value_source}: {env_var}=<secret>"
            )
            # logger.info(f"Setting Environment Variable from {value_source}: {env_var}={value}")
            os.environ[env_var] = value
        else:
            logger.info(
                f"Setting Environment Variable from {value_source}: {env_var}={value}"
            )
            os.environ[env_var] = value


@pytest.fixture(autouse=True)
def _reset_space_access_manager():
    """Reset the global space access manager after each test.

    Tests that call _run_standalone() set a StandaloneSpaceAccessManager
    singleton that persists in the xdist worker process.  This fixture
    ensures subsequent tests get the default StorageSpaceAccessManager.
    """
    yield
    from gbserver.spaces.space_access_manager import set_space_access_manager

    set_space_access_manager(None)  # type: ignore[arg-type]


def log_gb_env_vars():
    """Log all environment variables whose name starts with 'GB', masking secret values.

    Variables whose names contain 'TOKEN', 'KEY', 'PASSWD', or 'SECRET' have
    their values replaced with '<secret>' in the output.
    """
    _SECRET_PATTERNS = re.compile(r"TOKEN|KEY|PASSWD|SECRET", re.IGNORECASE)
    gb_vars = {k: v for k, v in sorted(os.environ.items()) if k.startswith("GB")}
    if not gb_vars:
        logger.info("No GB* environment variables found.")
        return
    lines = []
    for k, v in gb_vars.items():
        display = "<secret>" if _SECRET_PATTERNS.search(k) else v
        lines.append(f"  {k}={display}")
    logger.info("GB* environment variables:\n" + "\n".join(lines))


@pytest.fixture(autouse=True)
def _reset_lineage_store():
    """Reset the lineage store singleton after each test.

    The lineage store backend is selected by a feature flag evaluated once at
    first use.  Without a reset, a test that patches the flag (or reloads
    constants with a different GB_ENVIRONMENT) can leave a stale backend that
    causes subsequent tests to write/read lineage from the wrong store.
    """
    yield
    from gbserver.lineage.jobstats import reset_lineage_store

    reset_lineage_store()


def pytest_addoption(parser):
    """Register custom pytest CLI options.

    --buildtest-yaml: path to a buildtest.yaml file consumed by
    TestYamlRunnerCli in test/libgbtest/buildrunner/gbtest_runner.py (driven by
    the ``gbtest`` console script).  Allows running a single YAML-driven
    build test from the command line without authoring a concrete test
    class.
    """
    parser.addoption(
        "--buildtest-yaml",
        action="store",
        default=None,
        metavar="PATH",
        help="Path to a buildtest.yaml file to run via the generic YAML runner test.",
    )


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    from libgbtest.mode import get_test_mode

    test_mode = get_test_mode()
    logger.info(f"GBTEST_MODE={test_mode}")

    if test_mode != "live":
        # Mock mode: apply placeholder env vars so modules can import safely
        from libgbtest.mock_env import MOCK_ENV_DEFAULTS, MOCK_ENV_FORCED

        for key, value in MOCK_ENV_FORCED.items():
            os.environ[key] = value
        for key, value in MOCK_ENV_DEFAULTS.items():
            os.environ.setdefault(key, value)
        logger.info(
            "Mock mode: applied placeholder env vars. "
            "Set GBTEST_MODE=live or per-service GBTEST_LIVE_<SERVICE>=true for real connections."
        )

    # In mock mode, skip SPS secret loading unless GBTEST_LIVE_SECRETS=true
    load_secrets = (
        test_mode == "live" or os.getenv("GBTEST_LIVE_SECRETS", "").lower() == "true"
    )

    # Set GBTEST_SPS_IBMCLOUD_API_KEY environment variable by generating an API Key in ibmcloud inside the ETE SPS Account
    sps_api_key = os.getenv(ENV_VAR_SPS_IBMCLOUD_API_KEY, "")
    if not load_secrets:
        logger.info("Mock mode: skipping SPS secret loading.")
    elif sps_api_key == "":
        logger.info(
            f"To load test environment variables from SPS, set {ENV_VAR_SPS_IBMCLOUD_API_KEY} environment variable to an IBM Cloud API Key from the SPS ETE Account."
        )
    else:
        value = os.getenv(
            ENV_VAR_SPS_ENABLE_ENV_VAR_OVERRIDE, ENABLE_ENV_VAR_OVERRIDE_DEFAULT
        ).lower()
        enable_env_var_override = value == "true"
        set_test_env(sps_api_key, enable_env_var_override)

        # If GBTEST_GB_CLUSTER_PROJECT is set, also set the backend namespace and buildrunnerjob namespace
        # so that runtime code uses the same namespace as the test cluster login
        gbtest_cluster_project = os.getenv("GBTEST_GB_CLUSTER_PROJECT")
        if gbtest_cluster_project:
            gb_env = os.getenv("GB_ENVIRONMENT", "STAGING").upper()
            if gb_env == "STAGING":
                os.environ["GBSERVER_BACKEND_SERVER_NAMESPACE_STAGING"] = (
                    gbtest_cluster_project
                )
                logger.info(
                    f"Setting GBSERVER_BACKEND_SERVER_NAMESPACE_STAGING={gbtest_cluster_project} based on GBTEST_GB_CLUSTER_PROJECT"
                )
            elif gb_env == "DEV":
                os.environ["GBSERVER_BACKEND_SERVER_NAMESPACE_DEV"] = (
                    gbtest_cluster_project
                )
                logger.info(
                    f"Setting GBSERVER_BACKEND_SERVER_NAMESPACE_DEV={gbtest_cluster_project} based on GBTEST_GB_CLUSTER_PROJECT"
                )
            elif gb_env == "PROD":
                os.environ["GBSERVER_BACKEND_SERVER_NAMESPACE_PROD"] = (
                    gbtest_cluster_project
                )
                logger.info(
                    f"Setting GBSERVER_BACKEND_SERVER_NAMESPACE_PROD={gbtest_cluster_project} based on GBTEST_GB_CLUSTER_PROJECT"
                )
            # Also set BUILDRUNNERJOB_NAMESPACE directly
            os.environ["GBSERVER_BUILDRUNNERJOB_NAMESPACE"] = gbtest_cluster_project
            logger.info(
                f"Setting GBSERVER_BUILDRUNNERJOB_NAMESPACE={gbtest_cluster_project} based on GBTEST_GB_CLUSTER_PROJECT"
            )

        importlib.reload(gbserver.types.constants)
        importlib.reload(libgbtest.constants)

        from gbserver.lineage.jobstats import reset_lineage_store

        reset_lineage_store()
        # importlib.reload(gbserver_test.test_utils)
        # import gbserver_test.buildwatcher.utils
        # importlib.reload(gbserver_test.buildwatcher.utils)
        log_gb_env_vars()


class BuildAggregation(BaseModel):
    build: Optional[StoredBuild]
    targets: list[StoredTargetRun] = []
    steps: list[StoredStepRun] = []
    artifacts: list[ArtifactRegistration] = []
    assert_message: str = ""

    @staticmethod
    def create(build_id: str, assert_message: str) -> "BuildAggregation":
        from gbserver.storage.singleton_storage import (
            get_admin_storage,  # So conftest env var setting works on this
        )

        storage = get_admin_storage()
        build = storage.build_storage.get_by_uuid(build_id)
        if build is not None:
            assert isinstance(build, StoredBuild)
            targets = storage.target_storage.get_by_where({"build_id": build_id})
            steps = storage.step_storage.get_by_where({"build_id": build_id})
            artifacts = storage.artifact_registry.get_by_where(
                {"created_by_build_id": build_id}
            )
            ba = BuildAggregation(
                build=build,
                targets=targets,
                steps=steps,
                artifacts=artifacts,
                assert_message=assert_message,
            )
        else:
            ba = BuildAggregation(
                build=None, assert_message=f"Build with id {build_id} not found?!"
            )
        return ba


FAILURE_MARKER = "Failed Test Build: "


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture build information when buildtest assertions fail."""
    outcome = yield
    report = outcome.get_result()

    # Only process failures during the test call phase
    if report.when == "call" and report.failed:
        # Try to parse build ID from the assertion message format: [Build: <id>]
        if report.longrepr:
            longrepr_str = str(report.longrepr)
            assert_msg = str(call.excinfo)
            match = re.search(BUILD_ID_PATTERN, longrepr_str)
            if match:
                failed_build_id = match.group(1)
                build_aggregation = BuildAggregation.create(
                    failed_build_id, assert_message=assert_msg
                )
                build_json = build_aggregation.model_dump_json()
                info = f"id={failed_build_id} build={build_json}\n"
                logger.info(info)
                # breakpoint()    # Debugging
                extra_info = f"\n\n{FAILURE_MARKER}{info}\n"
                report.longrepr = str(report.longrepr) + extra_info


# ---------------------------------------------------------------------------
# Mock fixtures (formerly test/gbserver_test/conftest.py)
# Applied to all test tiers: unit/, integration/, e2e/
# ---------------------------------------------------------------------------

from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

from libgbtest.fixture_loader import load_fixture  # noqa: E402
from libgbtest.mode import should_use_live  # noqa: E402

# ---------------------------------------------------------------------------
# 0. Test-environment guard — never run against PROD; `ibm` tests need cloud creds
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _check_test_env(request):
    """Guard the test environment for every test.

    - Always: refuse to run with GB_ENVIRONMENT=PROD.
    - For tests marked ``ibm``: assert the IBM/cloud secret bundle is present
      (no-op in mock mode).  This replaces the old per-class
      ``_is_cloud_config_required`` gate — the requirement now tracks the
      ``ibm`` marker, which already designates "needs IBM infrastructure".
    """
    from gbserver.types.constants import GB_ENVIRONMENT

    assert GB_ENVIRONMENT != "PROD", "Refusing to run tests with GB_ENVIRONMENT=PROD."
    if request.node.get_closest_marker("ibm"):
        from libgbtest.utils import check_cloud_config

        check_cloud_config()
    yield


# ---------------------------------------------------------------------------
# a. Storage — mock = SQLite, live = whatever the test class chooses
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="class")
def _configure_storage_for_mock(request):
    """In mock mode, override storage factory to use SQLite."""
    if should_use_live(request, "storage"):
        yield
        return

    cls = request.cls
    if cls is None or not hasattr(cls, "_get_storage_factory"):
        yield
        return

    from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory

    original_factory = cls.__dict__.get("_get_storage_factory")

    cls._get_storage_factory = classmethod(lambda c: SqliteStorageFactory())

    yield

    if original_factory is not None:
        cls._get_storage_factory = original_factory
    else:
        if "_get_storage_factory" in cls.__dict__:
            delattr(cls, "_get_storage_factory")


# ---------------------------------------------------------------------------
# b. GitHub Auth — mock = synthetic user via apikey localhost bypass
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_github_auth(request):
    """In mock mode, patch get_gh_user to return a synthetic user."""
    if should_use_live(request, "github"):
        yield
        return

    from gbserver.types.auth import User

    user_data = load_fixture("github", "user.json")
    fake_user = User(**user_data)
    with (
        patch("gbserver.api.auth.get_gh_user", return_value=(fake_user, "")),
        patch(
            "libgbtest.storage.build_storage.get_gh_user",
            return_value=(fake_user, ""),
        ),
        patch(
            "libgbtest.api.utils.get_gh_user",
            return_value=(fake_user, ""),
        ),
        patch(
            "integration.ibm.api.test_spaces.get_gh_user",
            return_value=(fake_user, ""),
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# c. GitHub API — mock = stub MyGHApi
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_github_api(request):
    """In mock mode, patch MyGHApi to return canned responses."""
    if should_use_live(request, "github"):
        yield
        return

    branch_data = load_fixture("github", "branch_exists.json")
    file_data = load_fixture("github", "file_content.json")
    pr_data = load_fixture("github", "pull_request.json")

    instance = MagicMock()
    instance.branch_exists.return_value = branch_data["exists"]
    instance.get_file_content.return_value = file_data.get("content", "{}")
    instance.is_repo_present.return_value = True
    instance.get_pull_request.return_value = pr_data
    instance.get_open_prs.return_value = []
    instance.create_pr.return_value = pr_data
    instance.merge_pr.return_value = True

    mock_cls = MagicMock(return_value=instance)
    with (
        patch("gbserver.github.myghapi.MyGHApi", mock_cls),
        patch("gbserver.buildrunner.build_setup.MyGHApi", mock_cls),
        patch("gbserver.buildrunner.buildrunner.MyGHApi", mock_cls),
        patch("gbserver.buildrunner.buildlogger.MyGHApi", mock_cls),
    ):
        yield instance


# ---------------------------------------------------------------------------
# c1b. GitURI branch check — mock = simulate branch existence per repo
# ---------------------------------------------------------------------------

_REPOS_WITH_CONFIG_BRANCH = {"gbspace-public", "gb-test"}


def _fake_get_config_branch(token, uri, config_branch_name):
    """Return config_branch_name only for repos that 'have' the branch."""
    for repo in _REPOS_WITH_CONFIG_BRANCH:
        if repo in uri:
            return config_branch_name
    return None


@pytest.fixture(autouse=True)
def _mock_git_uri_branch_check(request):
    """In mock mode, patch GitURI.__get_config_branch to avoid GitHub calls.

    Also injects a non-empty token default so the empty-token guard in
    get_gb_space_config_uri does not short-circuit the branch check.
    """
    if should_use_live(request, "github"):
        yield
        return

    from gbcommon.uri.git import GitURI

    func = GitURI.get_gb_space_config_uri
    original_defaults = func.__defaults__
    func.__defaults__ = ("mock-token", original_defaults[1])

    with patch(
        "gbcommon.uri.git.GitURI._GitURI__get_config_branch",
        side_effect=_fake_get_config_branch,
    ):
        yield

    func.__defaults__ = original_defaults


# ---------------------------------------------------------------------------
# c2. Space access — mock = always allow writes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_space_access(request):
    """In mock mode, bypass space write-access checks."""
    if should_use_live(request, "github"):
        yield
        return

    with (
        patch(
            "gbserver.api.utils.has_space_write_access",
            return_value=(True, "standalone"),
        ),
        patch(
            "gbserver.api.utils.is_super_admin",
            return_value=True,
        ),
        patch(
            "gbserver.api.artifacts.confirm_space_write_access",
            return_value=None,
        ),
        patch(
            "gbserver.api.artifacts.is_super_admin",
            return_value=True,
        ),
        patch(
            "gbserver.api.builds.confirm_space_write_access",
            return_value=None,
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# d. Kubernetes — mock = AsyncMock
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_kubernetes(request):
    """In mock mode, patch K8s client creation."""
    if should_use_live(request, "kubernetes"):
        yield
        return

    try:
        import kubernetes_asyncio  # noqa: F401
    except ImportError:
        yield
        return

    mock_client = AsyncMock()
    with (
        patch(
            "gbserver.environment.k8s.kubernetes_asyncio.config.load_incluster_config",
            side_effect=lambda: None,
        ),
        patch(
            "gbserver.environment.k8s.kubernetes_asyncio.client.BatchV1Api",
            return_value=mock_client,
        ),
        patch(
            "gbserver.environment.k8s.kubernetes_asyncio.client.CoreV1Api",
            return_value=mock_client,
        ),
    ):
        yield mock_client


# ---------------------------------------------------------------------------
# e. HuggingFace — mock = stub
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_huggingface(request):
    """In mock mode, patch HuggingFace Hub downloads."""
    if should_use_live(request, "hf"):
        yield
        return

    with patch("huggingface_hub.snapshot_download", return_value="/tmp/fake-model"):
        yield


# ---------------------------------------------------------------------------
# f. Lineage store — mock = stub that returns empty/True
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_lineage(request):
    """In mock mode, patch get_lineage_store to return a no-op stub."""
    if should_use_live(request, "lakehouse"):
        yield
        return

    mock_store = MagicMock()
    mock_store.does_release_id_exist.return_value = True
    mock_store.count_release_ids.return_value = 1
    mock_store.add_jobstats_for_build.return_value = None
    mock_store.add_jobstats_for_build_target.return_value = None
    mock_store.add_jobstats_for_original_artifact.return_value = None
    mock_store.create_jobstats_for_target.return_value = ([], {})
    mock_store.create_jobstats_for_original_artifact.return_value = None

    with (
        patch("gbserver.lineage.jobstats.get_lineage_store", return_value=mock_store),
        patch("gbserver.api.artifacts.get_lineage_store", return_value=mock_store),
        patch(
            "gbserver.buildrunner.buildrunner.get_lineage_store",
            return_value=mock_store,
        ),
        patch(
            "integration.ibm.api.test_artifacts.get_lineage_store",
            return_value=mock_store,
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# g. Time/Sleep — mock = instant (sync only)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_time(request):
    """In mock mode, replace time.sleep with a short real sleep."""
    if should_use_live(request, "time"):
        yield
        return

    import time as _time

    _real_sleep = _time.sleep

    def _short_sleep(seconds):
        _real_sleep(seconds if seconds >= 1 else 0.01)

    mock_sleep = MagicMock(side_effect=_short_sleep)
    with patch("time.sleep", mock_sleep):
        request.node._mock_sync_sleep = mock_sleep
        yield
