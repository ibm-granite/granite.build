import json
import sys
from typing import Dict

import click
from tabulate import tabulate
from tqdm import tqdm

from gbcli.client.client import GBClient
from gbcli.commands.command_auth import str_exc_chain
from gbcli.commands.common_options import common_options
from gbcli.utils.gbconstants import CLIPBOARD_CHAR, PROJECT_NAME, SPACE_LIST_HEADERS
from gbcli.utils.gbcredentials import get_user_token
from gbcli.utils.spaceutil import get_spaces
from gbcli.utils.versionutil import check_current_and_latest_versions
from gbcommon.types.gbenvconfig import is_standalone


@click.group("space")
@click.pass_context
def cli(ctx):
    """Work with spaces"""
    pass


@cli.command()
@click.pass_context
@click.option(
    "--format",
    default="plain",
    type=click.Choice(["plain", "json"], case_sensitive=True),
    help=f"Output format: plain (default), json",
)
@click.option("--all", is_flag=True, help=f"All spaces user has access to")
@click.option(
    "--refresh",
    is_flag=True,
    help=f"Refreshes local cache of spaces user has access to",
)
@common_options
def list(
    ctx, format: str, all: bool, refresh: bool, skip_version_check: bool, quiet: bool
):
    """List the spaces set to the build or available for the current user"""
    if not skip_version_check:
        try:
            outdated_version = check_current_and_latest_versions()
        except Exception as e:
            click.echo(f"❌ {str(e)}.", err=True)
            ctx.exit(1)  # Exit with a non-zero status
        if outdated_version:
            click.echo(outdated_version, err=True)
            ctx.exit(1)  # Exit with a non-zero status

    if format == "json":
        quiet = True

    if refresh and not all:
        click.echo(
            f"❌ Try running again with 'llmb space list --all --refresh", err=True
        )
        ctx.exit(1)  # Exit with a non-zero status

    if refresh and is_standalone():
        # In standalone mode spaces are always fetched fresh from the local gbserver,
        # and the local cache/profile that --refresh repopulates is never used (it would
        # also corrupt ~/.gbcli/config because the standalone config has no spaces
        # section). Block the flag with a clear message instead.
        click.echo(
            "⚠️  Warning: '--refresh' is not supported in standalone mode "
            "(spaces are always fetched fresh from the local gbserver).",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status

    if not quiet:
        if all:
            click.echo(f"{CLIPBOARD_CHAR}List {PROJECT_NAME} spaces available for you.")
        else:
            click.echo(f"{CLIPBOARD_CHAR}List {PROJECT_NAME} spaces set to this build.")

    def format_user_role(is_admin: str | bool):
        if is_admin == "<unknown>":
            return is_admin
        else:
            return "admin" if is_admin == True else "user"

    try:
        space_client = GBClient.Space(get_user_token())

        if quiet:
            spaces = space_client.list_spaces(all, refresh, None)
        else:
            with tqdm(
                total=100,
                miniters=1,
                desc="Listing available spaces",
                bar_format="{desc} [{bar}] {percentage:3.0f}% {postfix}",
                ascii="-#",
                leave=False,
            ) as progress_bar:

                def update_bar(callback_event: str, callback_args: Dict):
                    steps = callback_args.get("steps", 0)
                    progress_bar.update(n=steps)
                    match callback_event:
                        case "fetching_spaces":
                            progress_bar.reset(total=100)
                            progress_bar.set_description(
                                f"Fetching spaces from GBSERVER"
                            )
                            progress_bar.update(n=steps)
                        case "done_fetching_spaces":
                            progress_bar.set_description("📝 Listing available spaces")
                            progress_bar.update(n=steps)
                        case "complete":
                            progress_bar.reset(total=100)
                            progress_bar.update(n=steps)
                            progress_bar.write(f"📝 Listing available spaces:")
                        case "error":
                            reason = callback_args.get("reason", "")
                            progress_bar.clear()
                            click.echo(
                                f"\n❌ Spaces could not be listed at this moment... Reason: {reason}",
                                err=True,
                            )
                            sys.exit(1)  # Exit with a non-zero status

                    return

                spaces = space_client.list_spaces(all, refresh, update_bar)
                progress_bar.close()

        if spaces and len(spaces) > 0:
            if format == "plain":
                spaces_table = [
                    [
                        s["name"],
                        s["git_repo_uri"],
                        s["lakehouse_namespace"],
                        format_user_role(s["is_admin"]),
                    ]
                    for s in spaces
                ]
                spaces_output = tabulate(
                    spaces_table, SPACE_LIST_HEADERS, tablefmt="plain"
                )
            else:
                spaces_output = json.dumps(spaces)
            click.echo(spaces_output)

    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        click.echo(f"❌ Space list failed!", err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.argument("space_name", required=True)
@click.option(
    "--default",
    is_flag=True,
    help="WARNING Command line option '--default' has been deprecated and will be removed from a future update. Just run 'llmb space set' without this option to set the default space",
)
@click.option(
    "--name",
    help="Specify a custom space name. It only affects your local environment.",
)
@click.option(
    "--format",
    "format",
    default="simple",
    type=click.Choice(["simple", "json"], case_sensitive=True),
    help="Output format: simple (default), json",
)
@common_options
def set(ctx, space_name, default, name, format, skip_version_check, quiet):
    """Set an available space as target"""
    if format == "json":
        quiet = True
    if not skip_version_check:
        try:
            outdated_version = check_current_and_latest_versions()
        except Exception as e:
            click.echo(f"❌ {str(e)}.", err=True)
            ctx.exit(1)  # Exit with a non-zero status
        if outdated_version:
            click.echo(outdated_version, err=True)
            ctx.exit(1)  # Exit with a non-zero status

    reserved_space_keys = ["domain", "local"]
    try:
        if name and default:
            raise Exception(
                f"Error: Please run --default and --name options separately"
            )

        if default:
            click.echo(
                "Warning: Command line option '--default' has been deprecated and will be removed from a future update. Just run 'llmb space set' without this option to set the default space"
            )

        # always set space to be default
        if not name:
            default = True
        else:
            default = False

        if space_name in reserved_space_keys:
            raise Exception(f"Error: {space_name} is a reserved word")

        setting_space_message = (
            f"Setting the space '{space_name}' as default space to this build"
            if default
            else f"Setting the space '{space_name}'"
        )

        if not quiet:
            click.echo(setting_space_message)

        def echo_callback(callback_event: str, callback_args: Dict):
            match callback_event:
                case "error":
                    reason = callback_args.get("reason", "")
                    click.echo(
                        f"\n❌ Space could not be set at this moment... Reason: {reason}",
                        err=True,
                    )
                    sys.exit(1)  # Exit with a non-zero status
                case _:
                    pass

        space_client = GBClient.Space(get_user_token())
        user_spaces = get_spaces(
            space_client.github_token, echo_callback
        )  # TODO fix callback behavior

        if not user_spaces and not quiet:
            with tqdm(
                total=100,
                miniters=1,
                desc="Setting space",
                bar_format="{desc} [{bar}] {percentage:3.0f}% {postfix}",
                ascii="-#",
                leave=False,
            ) as progress_bar:

                def update_bar(callback_event: str, callback_args: Dict):
                    steps = callback_args.get("steps", 0)
                    progress_bar.update(n=steps)
                    match callback_event:
                        case "fetching_spaces":
                            progress_bar.reset(total=100)
                            progress_bar.set_description(
                                f"fetching user spaces from GBSERVER"
                            )
                            progress_bar.update(n=steps)
                        case "done_fetching_spaces":
                            progress_bar.set_description("📝 Listing available spaces")
                            progress_bar.update(n=steps)
                        case "error":
                            reason = callback_args.get("reason", "")
                            click.echo(
                                f"\n❌ Space could not be set at this moment... Reason: {reason}",
                                err=True,
                            )
                            sys.exit(1)  # Exit with a non-zero status
                    return

                space_client.set_space(space_name, default, update_bar, name)
                progress_bar.close()
                completion_post_text = " as default" if default else "."
                click.echo(
                    f"✅ Space '{space_name}' has been succesfully set{completion_post_text}"
                )
                if format == "json":
                    click.echo(json.dumps({"space": space_name, "default": default}))
        else:
            space_client.set_space(space_name, default, None, name)
            completion_post_text = (
                " as default" if default else f" as name '{name}'" if name else "."
            )

            if not quiet:
                click.echo(
                    f"✅ Space '{space_name}' has been succesfully set{completion_post_text}"
                )
            if format == "json":
                click.echo(json.dumps({"space": space_name, "default": default}))

    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        click.echo(f"❌ Space set failed!", err=True)
        ctx.exit(1)  # Exit with a non-zero status
