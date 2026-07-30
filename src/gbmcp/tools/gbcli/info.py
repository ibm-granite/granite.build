import json

from fastmcp.tools import tool
from fastmcp.utilities.logging import get_logger

from gbcli.client.client import GBClient
from gbcli.utils.versionutil import get_current_version

logger = get_logger(__name__)


@tool(description="Return Granite.build's gbcli and gbserver version.")
def info_gb_version() -> str:
    """Return gbcli version as JSON.

    Returns:
        JSON object with 'clientVersion' (major, minor, patch) and optionally
        'serverVersion' (gitCommit) if the gbserver is reachable.
    """
    # The gbcli client ships inside the "granite-build" distribution; the
    # importable "gbcli" module has no separate version, so both refer to the
    # same number. Query the distribution name (not "granite.build", which is
    # the product/brand name and is not a pip-installable distribution).
    client_version = get_current_version("granite-build")
    parts = client_version.split(".")
    if len(parts) >= 3:
        client_version_info = {"major": parts[0], "minor": parts[1], "patch": parts[2]}
    else:
        client_version_info = {"version": client_version}
    version_obj = {"package": "granite-build", "clientVersion": client_version_info}

    gbserver_version = GBClient.Version(None).get_gbserver_version(quiet=True)
    if gbserver_version:
        version_obj["serverVersion"] = {"gitCommit": gbserver_version}

    result = json.dumps(version_obj, indent=4)
    logger.debug(f"info_gb_version result: {result}")
    return result
