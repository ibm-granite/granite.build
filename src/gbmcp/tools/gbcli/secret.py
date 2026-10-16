import json
import os
import shlex

from fastmcp.tools import tool
from fastmcp.utilities.logging import get_logger

from gbcli.client.client import GBClient
from gbmcp.utils.gbserver_errors import actionable_gbserver_errors
from gbmcp.utils.output_filter import apply_output_filters

logger = get_logger(__name__)


@tool(
    description=(
        "Return list of secret names. Omit space for the authenticated user's personal secrets; "
        "provide space for a space's secrets (requires space admin). "
        "Returns names only — never values. Supports output filtering: grep, wc, head, tail."
    )
)
@actionable_gbserver_errors
def secret_list(
    space: str | None = None,
    grep: str | None = None,
    wc: bool | None = None,
    head: int | None = None,
    tail: int | None = None,
) -> str:
    """Return secret names as JSON.

    Lists secret names only — secret values are never returned through the MCP server.

    Args:
        space: Space name. If omitted, lists the authenticated user's personal secrets.
        grep: Filter output lines by regex. Supports -Cn, -An, -Bn, -i, -v, -F, -w, -x, -c, -n, -o, -mN flags.
        wc: If True, return only line and character count instead of full output.
        head: Return only the first N lines. Mutually exclusive with tail.
        tail: Return only the last N lines. Mutually exclusive with head.

    Returns:
        JSON object with a "secrets" array of secret names.
    """
    result = GBClient.Secret(None).list_secrets(personal=(space is None), space=space)
    logger.debug(f"secret_list result: {result}")
    if result is None:
        if space:
            return f"Error: Could not list secrets for space '{space}'. Check that the space exists and you are a space admin."
        return "Error: Could not list user secrets. Check that the server is reachable and your token is valid."
    output = json.dumps(result, indent=4)
    return apply_output_filters(
        output, tool_name="secret_list", grep=grep, wc=wc, head=head, tail=tail
    )


@tool(
    description=(
        "Return the gbcli command to retrieve a secret value. The value is revealed only in the "
        "user's terminal, never returned to the agent — secret values never flow through the MCP server. "
        "Omit space for a personal secret; provide space for a space secret (requires space admin). "
        "Do not ask the user to paste the output back into the conversation."
    )
)
def secret_get(secret_name: str, space: str | None = None) -> str:
    """Return the gbcli command to retrieve a secret value.

    Retrieval happens out-of-band: the user runs the returned command in their own terminal,
    where the value is displayed. The value never reaches the agent or the MCP server.

    Args:
        secret_name: Name of the secret to retrieve.
        space: Space name. If omitted, retrieves a personal secret.

    Returns:
        Shell command string the user should run to view the secret value.
    """
    gb_env = os.environ.get("GB_ENVIRONMENT", "PROD")
    cmd = (
        f"export GB_ENVIRONMENT={gb_env} && gbcli secret get {shlex.quote(secret_name)}"
    )
    if space:
        cmd += f" --space {shlex.quote(space)}"
    else:
        cmd += " --personal"
    cmd += " --format json"
    return cmd


@tool(
    description=(
        "Return the gbcli command to create a new secret. The command contains a literal "
        "<secret-value> placeholder — the user replaces it with the real value in their own terminal, "
        "so the value never flows through the agent or the MCP server. Do not fill in the placeholder "
        "or ask the user for the value in conversation. "
        "Omit space for a personal secret; provide space for a space secret (requires space admin)."
    )
)
def secret_create(secret_name: str, space: str | None = None) -> str:
    """Return the gbcli command to create a new secret.

    The returned command carries a `<secret-value>` placeholder. The user substitutes the real
    value when running the command in their terminal — the value is never a tool argument and
    never transits the MCP server.

    Args:
        secret_name: Name for the new secret.
        space: Space name. If omitted, creates a personal secret.

    Returns:
        Shell command string (with a <secret-value> placeholder) for the user to run.
    """
    gb_env = os.environ.get("GB_ENVIRONMENT", "PROD")
    cmd = f"export GB_ENVIRONMENT={gb_env} && gbcli secret create {shlex.quote(secret_name)}"
    if space:
        cmd += f" --space {shlex.quote(space)}"
    else:
        cmd += " --personal"
    cmd += " --value <secret-value>"
    return cmd


@tool(
    description=(
        "Return the gbcli command to update an existing secret. The command contains a literal "
        "<secret-value> placeholder — the user replaces it with the real value in their own terminal, "
        "so the value never flows through the agent or the MCP server. Do not fill in the placeholder "
        "or ask the user for the value in conversation. "
        "Omit space for a personal secret; provide space for a space secret (requires space admin)."
    )
)
def secret_update(secret_name: str, space: str | None = None) -> str:
    """Return the gbcli command to update an existing secret.

    The returned command carries a `<secret-value>` placeholder. The user substitutes the real
    value when running the command in their terminal — the value is never a tool argument and
    never transits the MCP server.

    Args:
        secret_name: Name of the secret to update.
        space: Space name. If omitted, updates a personal secret.

    Returns:
        Shell command string (with a <secret-value> placeholder) for the user to run.
    """
    gb_env = os.environ.get("GB_ENVIRONMENT", "PROD")
    cmd = f"export GB_ENVIRONMENT={gb_env} && gbcli secret update {shlex.quote(secret_name)}"
    if space:
        cmd += f" --space {shlex.quote(space)}"
    else:
        cmd += " --personal"
    cmd += " --value <secret-value>"
    return cmd


@tool(
    description="Delete a secret. Omit space for a personal secret; provide space for a space secret (requires space admin)."
)
@actionable_gbserver_errors
def secret_delete(secret_name: str, space: str | None = None) -> str:
    """Delete a secret.

    Args:
        secret_name: Name of the secret to delete.
        space: Space name. If omitted, deletes a personal secret.

    Returns:
        Success or error message.
    """
    result = GBClient.Secret(None).delete_secret(
        secret_name, personal=(space is None), space=space
    )
    # delete_secret returns (secret, space_name, username); a None tuple OR a None
    # first element (the gbserver response) both mean the delete did not happen —
    # e.g. the secret does not exist, or space resolution / admin checks failed.
    # Mirror the gbcli CLI, which gates success on the first element being truthy.
    secret = result[0] if result else None
    if not secret:
        if space:
            return f"Error: Failed to delete secret '{secret_name}' from space '{space}'. Check that the secret exists and you are a space admin."
        return f"Error: Failed to delete secret '{secret_name}'. Check that the secret exists."
    if space:
        return f"Secret '{secret_name}' deleted successfully from space '{space}'."
    return f"Secret '{secret_name}' deleted successfully."
