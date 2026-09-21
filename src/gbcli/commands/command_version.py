import json
import sys
from typing import Dict

import click

from gbcli.client.client import GBClient
from gbcli.commands.command_auth import str_exc_chain
from gbcli.utils import versionutil
from gbcli.utils.gbconstants import gb_environment
from gbcli.utils.gbcredentials import get_user_token
from gbcli.utils.versionutil import get_current_version


@click.command()
@click.pass_context
@click.option(
    "--check-updates",
    is_flag=True,
    default=False,
    help="Perform version check.",
)
@click.option(
    "--client",
    is_flag=True,
    default=False,
    help="Display only the client version.",
)
@click.option(
    "--format",
    default="simple",
    type=click.Choice(["simple", "json"], case_sensitive=True),
    help="Response format: simple (default), json",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Enables quiet mode.",
)
def cli(ctx, check_updates, client, format, quiet):
    """Show client version"""

    if format == "json":
        quiet = True

    erase_sequence = "\r\033[K"

    if check_updates:
        result = versionutil.evaluate_version_status()
        if result.status is versionutil.VersionStatus.BELOW_FLOOR:
            # Below the min-supported floor: mandatory upgrade, non-zero exit.
            click.echo(result.message, err=True)
            ctx.exit(1)  # Exit with a non-zero status
        elif result.status is versionutil.VersionStatus.OUTDATED_WARN:
            # A newer version exists but we're still supported: notify, exit 0.
            click.echo(result.message, err=True)
        elif result.status is versionutil.VersionStatus.UNKNOWN:
            # The check couldn't complete (offline, rate-limited, unparseable version).
            # An explicit --check-updates shouldn't claim "up to date" when it verified
            # nothing; say so on stderr and still exit 0 (best-effort, not a failure).
            click.echo(
                f"Could not verify whether {get_current_version('granite.build')} is the "
                "latest version (version check unavailable).",
                err=True,
            )
        else:
            click.echo(
                f"The current client version ({get_current_version('granite.build')}) is up to date."
            )
    else:
        client_version = get_current_version("granite.build")
        gbserver_version = None

        if not client:

            def echo_callback(callback_event: str, callback_args: Dict):
                if callback_event == "error":
                    reason = callback_args.get("reason", "")
                    if "The token is invalid" in reason:
                        error_message = "GitHub token is invalid. Please reauthenticate by obtaining a new token with auth login."
                    else:
                        error_message = "Unable to connect to the server. Make sure you are connected to the VPN and try again."
                    click.echo(
                        f"{erase_sequence}{error_message}",
                        err=True,
                    )
                    sys.exit(1)  # Exit with a non-zero status
                elif callback_event == "fetching_server_version":
                    click.echo("Fetching server version...", nl=False)
                else:
                    pass  # Ignore unknown events

            try:
                version_client = GBClient.Version(get_user_token())

                gbserver_version = version_client.get_gbserver_version(
                    quiet, callback=echo_callback
                )
            except Exception as e:
                click.echo(str_exc_chain(e), err=True)
                ctx.exit(1)  # Exit with a non-zero status

        if format == "simple":
            click.echo(f"{erase_sequence}Client Version: {client_version}")
            if gbserver_version:
                click.echo(f"Server Version: {gb_environment()} @{gbserver_version}")
        else:
            client_version_scheme = client_version.split(".")
            version_obj = {
                "clientVersion": {
                    "major": client_version_scheme[0],
                    "minor": client_version_scheme[1],
                    "patch": client_version_scheme[2],
                }
            }
            if gbserver_version:
                version_obj["serverVersion"] = {"gitCommit": gbserver_version}

            click.echo(json.dumps(version_obj, indent=4))
