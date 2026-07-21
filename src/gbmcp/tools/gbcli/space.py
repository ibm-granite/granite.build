import json

from fastmcp.tools import tool
from fastmcp.utilities.logging import get_logger

from gbcli.client.client import GBClient
from gbcommon.types.gbenvconfig import is_standalone

from gbmcp.utils.auth import get_github_token
from gbmcp.utils.gbserver_errors import actionable_gbserver_errors
from gbmcp.utils.output_filter import apply_output_filters

logger = get_logger(__name__)


@tool(
    description="Return list of spaces. Supports output filtering: grep, wc, head, tail."
)
@actionable_gbserver_errors
def space_list(
    grep: str | None = None,
    wc: bool | None = None,
    head: int | None = None,
    tail: int | None = None,
) -> str:
    """Return gbcli space list output as JSON.

    Args:
        grep: Filter output lines by regex. Supports -Cn, -An, -Bn, -i, -v, -F, -w, -x, -c, -n, -o, -mN flags.
        wc: If True, return only line and character count instead of full output.
        head: Return only the first N lines. Mutually exclusive with tail.
        tail: Return only the last N lines. Mutually exclusive with head.

    Returns:
        JSON array of available spaces.
    """
    token = get_github_token()
    # refresh=True triggers gbcli's save_new_spaces() config-write path, which
    # sys.exit()s on a malformed ~/.gbcli/config and crashes the server. In
    # standalone, get_spaces() fetches live from gbserver with no config access,
    # so refresh=False returns fresh data and avoids the write entirely.
    refresh = not is_standalone()
    result = GBClient.Space(token).list_spaces(all=True, refresh=refresh)
    logger.debug(f"space_list result: {result}")
    output = json.dumps(result, indent=4)
    return apply_output_filters(
        output, tool_name="space_list", grep=grep, wc=wc, head=head, tail=tail
    )
