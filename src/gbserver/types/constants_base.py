#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Environment variable prefix and some basic stuff"""

import os

from gbcommon.types.gbenvconfig import DEFAULT_GB_ENVIRONMENT  # noqa: F401
from gbcommon.types.gbenvconfig import getenv_boolean  # noqa: F401

ENV_VAR_PREFIX = "GBSERVER"

ENV_VAR_GBSERVER_BACKEND_SERVER_NAMESPACE_PROD = (
    ENV_VAR_PREFIX + "_BACKEND_SERVER_NAMESPACE_PROD"
)
ENV_VAR_GBSERVER_BACKEND_SERVER_NAMESPACE_STAGING = (
    ENV_VAR_PREFIX + "_BACKEND_SERVER_NAMESPACE_STAGING"
)
ENV_VAR_GBSERVER_BACKEND_SERVER_NAMESPACE_DEV = (
    ENV_VAR_PREFIX + "_BACKEND_SERVER_NAMESPACE_DEV"
)

BACKEND_SERVER_NAMESPACE_PROD = os.getenv(
    ENV_VAR_GBSERVER_BACKEND_SERVER_NAMESPACE_PROD, "llm-build-prod"
)
BACKEND_SERVER_NAMESPACE_STAGING = os.getenv(
    ENV_VAR_GBSERVER_BACKEND_SERVER_NAMESPACE_STAGING, "llm-build-staging"
)
BACKEND_SERVER_NAMESPACE_DEV = os.getenv(
    ENV_VAR_GBSERVER_BACKEND_SERVER_NAMESPACE_DEV, "llm-build-dev"
)

# SSO (OIDC) authentication. Read these only through oidc_setting().
_OIDC_PREFIX = ENV_VAR_PREFIX + "_OIDC_"
ENV_VAR_OIDC_ISSUER = _OIDC_PREFIX + "ISSUER"
ENV_VAR_OIDC_JWKS_URI = _OIDC_PREFIX + "JWKS_URI"
ENV_VAR_OIDC_CLIENT_ID = _OIDC_PREFIX + "CLIENT_ID"
ENV_VAR_OIDC_CLIENT_SECRET = _OIDC_PREFIX + "CLIENT_SECRET"
ENV_VAR_OIDC_AUTHORIZE_URL = _OIDC_PREFIX + "AUTHORIZE_URL"
ENV_VAR_OIDC_TOKEN_URL = _OIDC_PREFIX + "TOKEN_URL"
ENV_VAR_OIDC_USERINFO_URL = _OIDC_PREFIX + "USERINFO_URL"
ENV_VAR_OIDC_CALLBACK_URL = _OIDC_PREFIX + "CALLBACK_URL"
ENV_VAR_OIDC_REMOTE_CALLBACK_URL = _OIDC_PREFIX + "REMOTE_CALLBACK_URL"

# Deployments still set the legacy GBSERVER_IBMID_* names; accepted here only.
_LEGACY_OIDC_PREFIX = ENV_VAR_PREFIX + "_IBMID_"

_OIDC_DEFAULTS = {
    ENV_VAR_OIDC_ISSUER: "https://login.ibm.com/oidc/endpoint/default",
    ENV_VAR_OIDC_JWKS_URI: "https://login.ibm.com/oidc/endpoint/default/jwks",
    ENV_VAR_OIDC_AUTHORIZE_URL: "https://login.ibm.com/v1.0/endpoint/default/authorize",
    ENV_VAR_OIDC_TOKEN_URL: "https://login.ibm.com/v1.0/endpoint/default/token",
    ENV_VAR_OIDC_USERINFO_URL: "https://login.ibm.com/v1.0/endpoint/default/userinfo",
}


def oidc_setting(env_var: str) -> str:
    """An ``ENV_VAR_OIDC_*`` setting, read now: the variable, else its legacy
    ``GBSERVER_IBMID_*`` name, else the default ("" if none)."""
    legacy = _LEGACY_OIDC_PREFIX + env_var[len(_OIDC_PREFIX) :]
    return os.getenv(env_var) or os.getenv(legacy) or _OIDC_DEFAULTS.get(env_var, "")
