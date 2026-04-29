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
        "gbserver_test/buildwatcher/test_buildrunner_1step.py",
        "gbserver_test/buildwatcher/test_buildrunner_retry.py",
        "gbserver_test/buildwatcher/test_buildrunnerjob.py",
        "gbserver_test/buildwatcher/test_builds.py",
        "gbserver_test/buildwatcher/test_buildwatcher.py",
        "gbserver_test/githubmanager/test_githubmanager_subselect_targets.py",
        "sidecar_test/test_multi_sidecar_cmdmon_delayed_pytest.py",
        "sidecar_test/test_multi_sidecar_cmdmon_pytest.py",
        "sidecar_test/test_multi_sidecar_pytest.py",
        "sidecar_test/test_sidecar_cmdmon_delayed_pytest.py",
        "sidecar_test/test_sidecar_cmdmon_pytest.py",
        "sidecar_test/test_sidecar_pytest.py",
        "sidecar_test/test_sidecar_tuning_pytest.py",
    ]

if not _can_import("kubernetes_asyncio"):
    collect_ignore.append("gbserver_test/resilience/test_k8s_retry.py")

if not _can_import("asyncssh"):
    collect_ignore.append("gbserver_test/utils/test_ssh_tunnel.py")

if not _can_import("lakehouse"):
    collect_ignore += [
        "gbserver_test/lineage/test_jobstats.py",
        "gbserver_test/storage/test_lh_loader.py",
    ]

import gbserver_test
import gbserver_test.constants
from gbserver_test.constants import BUILD_ID_PATTERN

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
    "GBTEST_SKIP_LAKEHOUSE_ADMIN_TESTS",
    "GBTEST_SKIP_SHADOW_ADMIN_TESTS",
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
]

DEFAULT_NON_SECRET_ENV_VAR_VALUES = {}
DEFAULT_NON_SECRET_ENV_VAR_VALUES["GB_ENVIRONMENT"] = "DEV"
DEFAULT_NON_SECRET_ENV_VAR_VALUES["GBTEST_HAS_COMPUTE_CLUSTER_ACCESS"] = "True"
DEFAULT_NON_SECRET_ENV_VAR_VALUES["GBTEST_HAS_GB_CLUSTER_ACCESS"] = "True"
DEFAULT_NON_SECRET_ENV_VAR_VALUES["GBTEST_SKIP_LAKEHOUSE_ADMIN_TESTS"] = "True"
DEFAULT_NON_SECRET_ENV_VAR_VALUES["GBTEST_SKIP_SHADOW_ADMIN_TESTS"] = "True"
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
    ensures subsequent tests get the default LakehouseSpaceAccessManager.
    """
    yield
    from gbserver.spaces.space_access_manager import set_space_access_manager

    set_space_access_manager(None)  # type: ignore[arg-type]


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    # Set GBTEST_SPS_IBMCLOUD_API_KEY environment variable by generating an API Key in ibmcloud inside the ETE SPS Account
    sps_api_key = os.getenv(ENV_VAR_SPS_IBMCLOUD_API_KEY, "")
    if sps_api_key == "":
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
        importlib.reload(gbserver_test.constants)
        # importlib.reload(gbserver_test.test_utils)
        # import gbserver_test.buildwatcher.utils
        # importlib.reload(gbserver_test.buildwatcher.utils)


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
