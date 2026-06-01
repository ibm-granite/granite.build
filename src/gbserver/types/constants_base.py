#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

# IBMid OIDC authentication
ENV_VAR_IBMID_ISSUER = ENV_VAR_PREFIX + "_IBMID_ISSUER"
ENV_VAR_IBMID_JWKS_URI = ENV_VAR_PREFIX + "_IBMID_JWKS_URI"
ENV_VAR_IBMID_CLIENT_ID = ENV_VAR_PREFIX + "_IBMID_CLIENT_ID"
ENV_VAR_IBMID_CLIENT_SECRET = ENV_VAR_PREFIX + "_IBMID_CLIENT_SECRET"
ENV_VAR_IBMID_AUTHORIZE_URL = ENV_VAR_PREFIX + "_IBMID_AUTHORIZE_URL"
ENV_VAR_IBMID_TOKEN_URL = ENV_VAR_PREFIX + "_IBMID_TOKEN_URL"
ENV_VAR_IBMID_USERINFO_URL = ENV_VAR_PREFIX + "_IBMID_USERINFO_URL"
ENV_VAR_IBMID_CALLBACK_URL = ENV_VAR_PREFIX + "_IBMID_CALLBACK_URL"
