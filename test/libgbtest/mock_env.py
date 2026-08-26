"""Mock environment variable defaults for running tests without external dependencies.

Applied by conftest.py:pytest_sessionstart when GBTEST_MODE=mock.

MOCK_ENV_FORCED: Always set in mock mode (override any existing env vars).
    These are the variables that control whether external calls happen — they
    must be forced to prevent a developer's shell env from defeating mock mode.

MOCK_ENV_DEFAULTS: Only set if not already present (setdefault semantics).
    These provide safe placeholders for variables read at import time.
"""

# Force-set in mock mode — these prevent external connections regardless
# of what's in the developer's shell environment.
#
# GBTEST_MOCK_HF makes mock mode never touch real HuggingFace (push/pull/
# exists/delete), so CI can't flake on HF rate limits or spend token quota.
# It is forwarded to dispatched worker jobs/pods (see get_exported_gbtest_env_vars)
# so they mock too. A single test can still exercise real HF via
# @pytest.mark.live("hf"), which the _hf_mock fixture honors by lifting the var.
MOCK_ENV_FORCED = {
    "GBSERVER_AUTH_MODE": "apikey",
    "GBTEST_HAS_COMPUTE_CLUSTER_ACCESS": "False",
    "GBTEST_HAS_GB_CLUSTER_ACCESS": "False",
    "GBTEST_MOCK_HF": "true",
    "WANDB_MODE": "disabled",
}

# Set only if not already present — safe placeholder values.
MOCK_ENV_DEFAULTS = {
    # Server config
    "GB_ENVIRONMENT": "DEV",
    "GBSERVER_RAISE_BUILD_EXCEPTIONS": "True",
    # Tokens — non-empty placeholders so import-time reads in constants.py work
    "GBSERVER_GITHUB_TOKEN": "mock-github-token",
    "GITHUB_TOKEN": "mock-github-token",
    "GBTEST_ADMIN_GITHUB_TOKEN": "mock-admin-token",
    "GBTEST_NON_ADMIN_GITHUB_TOKEN": "mock-nonadmin-token",
    "HF_TOKEN": "mock-hf-token",
    # Credentials — empty is fine because mock mode won't connect
    "GBSERVER_SQL_PASSWD": "",
    "GBSERVER_SQL_SSLROOT_CERT_BASE64": "",
    "IBM_CLOUD_API_KEY": "",
    "LAKEHOUSE_TOKEN": "",
    "GBTEST_GB_CLUSTER_API_KEY": "",
}
