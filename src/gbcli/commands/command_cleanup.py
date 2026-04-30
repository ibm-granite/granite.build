"""Command cleanup module."""

import sys
import webbrowser
from typing import Dict

import click

from gbcli.client.client import GBClient
from gbcli.commands.command_auth import str_exc_chain
from gbcli.commands.common_options import common_options
from gbcli.utils.gbconstants import CLIPBOARD_CHAR
from gbcli.utils.gbcredentials import get_user_token
from gbcli.utils.utils import check_runnable_browser
from gbcli.utils.versionutil import check_current_and_latest_versions


@click.command()
@click.pass_context
@click.option(
    "--all",
    is_flag=True,
    help="runs all cleanup options except for --space-repo-fork",
)
@click.option(
    "--config",
    is_flag=True,
    help="Cleanup config. Set by env var GB_CONFIG. If not set, defaults to: (~/.gbcli/config).",
)
@click.option(
    "--credentials",
    is_flag=True,
    help="Cleanup credentials. Set by env var GB_CONFIG. If not set, defaults to: (~/.gbcli/credentials).",
)
@click.option(
    "--local-cache",
    is_flag=True,
    help="Cleanup local cache (~/.gbcli/workdir or a location specified by GB_CACHE).",
)
@click.option(
    "--space-repo-fork",
    is_flag=True,
    help="Show instructions to delete user fork repo for space",
)
@common_options
def cli(
    ctx,
    all,
    config,
    credentials,
    local_cache,
    space_repo_fork,
    skip_version_check,
    quiet,
):
    """Perform cleanup"""

    def echo_callback(callback_event: str, callback_args: Dict):
        if callback_event == "error":
            reason = callback_args.get("reason", "")
            click.echo(
                f"❌ Cleanup can't be executed at this moment... Reason: {reason}",
                err=True,
            )
            sys.exit(1)  # Exit with a non-zero status
        else:
            pass  # Ignore unknown events

    try:
        cleanup_client = GBClient.Cleanup(get_user_token())

        if not all and not config and not credentials and not local_cache and not space_repo_fork:
            click.echo("❌ Please select an option.", err=True)
            click.echo(ctx.get_help(), err=True)
            return

        if config or all:
            click.echo(f"{CLIPBOARD_CHAR}Cleanup config")
            config_output = cleanup_client.remove_config()
            if "Error" not in config_output:
                click.echo(f"✅ Config at '{config_output}' has been successfully removed")
            else:
                click.echo(config_output)

        if credentials or all:
            click.echo(f"{CLIPBOARD_CHAR}Cleanup credentials")
            credentials_output = cleanup_client.remove_credentials()
            if "Error" not in credentials_output:
                click.echo(
                    f"✅ Credentials at '{credentials_output}' has been successfully removed"
                )
            else:
                click.echo(credentials_output)

        if local_cache or all:
            click.echo(f"{CLIPBOARD_CHAR}Cleanup local cache")
            local_cache_output = cleanup_client.remove_local_cache()
            if "Error" not in local_cache_output:
                click.echo(
                    f"✅ Local cache at '{local_cache_output}' has been successfully removed"
                )
            else:
                click.echo(local_cache_output)

        if space_repo_fork:
            if not skip_version_check:
                try:
                    outdated_version = check_current_and_latest_versions()
                except Exception as e:
                    click.echo(f"❌ {str(e)}.", err=True)
                    ctx.exit(1)  # Exit with a non-zero status
                if outdated_version:
                    click.echo(outdated_version, err=True)
                    ctx.exit(1)  # Exit with a non-zero status

            click.echo(f"{CLIPBOARD_CHAR}Find user's forked repository from default space")
            remove_fork_output = cleanup_client.remove_default_fork(callback=echo_callback)
            if "Error" not in remove_fork_output:
                settings_url = f"https://github.ibm.com/{remove_fork_output}/settings#danger-zone"
                click.echo(
                    f"Navigate to {settings_url}, scroll to the bottom 'Danger Zone' section, and select 'Delete this repository'"
                )
                if check_runnable_browser():
                    answer = click.confirm("Open the browser?", True)

                    if answer:
                        # https://docs.python.org/3/library/webbrowser.html
                        webbrowser.open(settings_url)

            else:
                click.echo(remove_fork_output)

    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status
