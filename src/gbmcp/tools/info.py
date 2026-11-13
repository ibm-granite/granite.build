# Copyright Granite.Build Authors
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

import json
from importlib.metadata import version

from fastmcp.tools import tool
from fastmcp.utilities.logging import get_logger

logger = get_logger(__name__)


@tool(description="Health check.")
def info_health() -> str:
    """Check the health of the service."""
    return "OK"


@tool(description="Return the gbmcp MCP server version.")
def info_version() -> str:
    """Return the gbmcp server version as JSON.

    Returns:
        JSON object with 'version' set to the installed gbmcp package version.
    """
    try:
        # gbmcp ships inside the granite.build distribution (pip name: granite-build).
        _v = version("granite-build")
    except Exception:
        _v = "unknown"
    result = json.dumps({"version": _v}, indent=4)
    logger.debug(f"info_version result: {result}")
    return result
