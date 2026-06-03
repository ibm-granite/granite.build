import json
import re
import sys
from typing import Dict

import click
from tabulate import tabulate

from gbcli.client.client import GBClient
from gbcli.commands.command_auth import str_exc_chain
from gbcli.commands.common_options import common_options, unsupported_in_standalone
from gbcli.utils.gbconstants import CLIPBOARD_CHAR, PROJECT_NAME
from gbcli.utils.gbcredentials import get_user_token
from gbcli.utils.versionutil import check_current_and_latest_versions


@click.group("secret")
@unsupported_in_standalone("secret")
def cli():
    """Work with secrets"""


@cli.command()
@click.pass_context
@click.option("--space", help="Space name.")
@click.option(
    "--personal",
    is_flag=True,
    default=False,
    help="per-user mode",
)
@click.option(
    "--format",
    default="plain",
    type=click.Choice(["plain", "json"], case_sensitive=True),
    help=f"Output format: plain (default), json",
)
@common_options
def list(ctx, space, personal, format, skip_version_check, quiet):
    """List secrets"""
    if not skip_version_check:
        try:
            outdated_version = check_current_and_latest_versions()
        except Exception as e:
            click.echo(f"❌ {str(e)}.", err=True)
            ctx.exit(1)  # Exit with a non-zero status
        if outdated_version:
            click.echo(outdated_version, err=True)
            ctx.exit(1)  # Exit with a non-zero status

    if personal and space:
        click.echo(
            f"❌ Error: --personal and --space were provided. Only one can be provided.",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status

    if format == "json":
        quiet = True

    if not quiet:
        click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} list")

    secrets_client = GBClient.Secret(get_user_token())

    erase_sequence = "\r\033[K"

    def echo_callback(callback_event: str, callback_args: Dict):
        steps = callback_args.get("steps", 0)
        match callback_event:
            case "listing_secrets_spinner":
                spinner = callback_args.get("spinner", "")
                if personal:
                    user = callback_args.get("user", "")
                    callback_message = (
                        f"\r📝 Obtaining secrets for user '{user}'... {spinner}"
                    )
                else:
                    space = callback_args.get("space", "")
                    space_name = callback_args.get("space_name", "")
                    callback_message = f"\r📝 Obtaining secrets for space '{space}' ({space_name})... {spinner}"
                click.echo(callback_message, nl=False)
            case "listed_secrets":
                if personal:
                    user = callback_args.get("user", "")
                    callback_message = f"{erase_sequence}📝 List of secrets obtained for user '{user}':"
                else:
                    space = callback_args.get("space", "")
                    space_name = callback_args.get("space_name", "")
                    callback_message = f"{erase_sequence}📝 List of secrets obtained for space '{space}' ({space_name}):"
                click.echo(callback_message)
            case "error":
                click.echo(
                    f"\n❌ Secrets can't be retrieved at this moment.. Reason: {callback_args.get('reason', '')}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case _:
                pass

    try:
        secrets = secrets_client.list_secrets(
            personal, space, callback=(None if quiet else echo_callback)
        )
        if secrets:
            if format == "plain":
                secrets_table = [[s] for s in secrets["secrets"]]
                secrets_output = tabulate(
                    secrets_table, ["SECRET_NAME"], tablefmt="plain"
                )
            else:
                secrets_output = json.dumps(secrets)

            click.echo(secrets_output)
    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.argument("secret_name", required=True)
@click.option("--space", help="Space name.")
@click.option(
    "--personal",
    is_flag=True,
    default=False,
    help="per-user mode",
)
@click.option(
    "--format",
    default="plain",
    type=click.Choice(["plain", "json"], case_sensitive=True),
    help=f"Output format: plain (default), json",
)
@common_options
def get(ctx, secret_name, space, personal, format, skip_version_check, quiet):
    """Get secret"""
    if not skip_version_check:
        try:
            outdated_version = check_current_and_latest_versions()
        except Exception as e:
            click.echo(f"❌ {str(e)}.", err=True)
            ctx.exit(1)  # Exit with a non-zero status
        if outdated_version:
            click.echo(outdated_version, err=True)
            ctx.exit(1)  # Exit with a non-zero status

    if personal and space:
        click.echo(
            f"❌ Error: --personal and --space were provided. Only one can be provided.",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status

    if format == "json":
        quiet = True

    if not quiet:
        click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} get")

    secrets_client = GBClient.Secret(get_user_token())

    erase_sequence = "\r\033[K"

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "obtaining_secret_spinner":
                spinner = callback_args.get("spinner", "")
                if personal:
                    user = callback_args.get("user", "")
                    callback_message = f"\r📝 Obtaining secret {secret_name} from user '{user}'... {spinner}"
                else:
                    space = callback_args.get("space", "")
                    space_name = callback_args.get("space_name", "")
                    callback_message = f"\r📝 Obtaining secret {secret_name} from space '{space}' ({space_name})... {spinner}"
                click.echo(callback_message, nl=False)
            case "obtained_secret":
                if personal:
                    user = callback_args.get("user", "")
                    callback_message = f"{erase_sequence}📝 Secret {secret_name} obtained from user '{user}':"
                else:
                    space = callback_args.get("space", "")
                    space_name = callback_args.get("space_name", "")
                    callback_message = f"{erase_sequence}📝 Secret {secret_name} obtained from space '{space}' ({space_name}):"
                click.echo(callback_message)
            case "error":
                click.echo(
                    f"\n❌ Secret can't be retrieved at this moment.. Reason: {callback_args.get('reason', '')}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case _:
                pass

    try:
        secret = secrets_client.get_secret(
            secret_name, personal, space, callback=(None if quiet else echo_callback)
        )
        if secret:
            if format == "plain":
                secrets_output = (
                    f"Secret name: {secret['secret_name']}\n"
                    + "Secret value: 🔒\n"
                    + f"Encoding: {secret['encoding']}\n"
                )

                if not personal:
                    secrets_output = (
                        f"Space name: {secret['space_name']}\n" + secrets_output
                    )
            else:
                secrets_output = json.dumps(secret)

            click.echo(secrets_output)
    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.argument("secret_name", required=True)
@click.option("--value", help="Secret value.")
@click.option("--space", help="Space name.")
@click.option(
    "--personal",
    is_flag=True,
    default=False,
    help="per-user mode",
)
@click.option(
    "--from-file",
    help=f"Path to the local .txt file containing the secret value.",
)
@click.option(
    "--format",
    default="plain",
    type=click.Choice(["plain", "json"], case_sensitive=True),
    help=f"Output format: plain (default), json",
)
@common_options
def create(
    ctx,
    secret_name,
    value,
    space,
    personal,
    from_file,
    format,
    skip_version_check,
    quiet,
):
    """Create secret"""
    if not skip_version_check:
        try:
            outdated_version = check_current_and_latest_versions()
        except Exception as e:
            click.echo(f"❌ {str(e)}.", err=True)
            ctx.exit(1)  # Exit with a non-zero status
        if outdated_version:
            click.echo(outdated_version, err=True)
            ctx.exit(1)  # Exit with a non-zero status

    if personal and space:
        click.echo(
            f"❌ Error: --personal and --space were provided. Only one can be provided.",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status

    if format == "json":
        quiet = True

    if not quiet:
        click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} create")

    secrets_client = GBClient.Secret(get_user_token())

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "encoding_secret":
                click.echo(f"📝 (1/2) Encoding secret...")
            case "creating_secret_spinner":
                spinner = callback_args.get("spinner", "")
                if personal:
                    user = callback_args.get("user", "")
                    callback_message = f"\r📝 (2/2) Creating new secret {secret_name} for user '{user}'... {spinner}"
                else:
                    space = callback_args.get("space", "")
                    space_name = callback_args.get("space_name", "")
                    callback_message = f"\r📝 (2/2) Creating new secret {secret_name} in space '{space}' ({space_name})... {spinner}"
                click.echo(callback_message, nl=False)
            case "error":
                click.echo(
                    f"\n❌ Secret can't be created at this moment.. Reason: {callback_args.get('reason', '')}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case _:
                pass

    def validate_secret_name(secret_name: str) -> bool:
        if len(secret_name) < 2 or len(secret_name) > 256:
            return False

        pattern = re.compile("^[a-zA-Z0-9_.-]*$")
        if not pattern.match(secret_name):
            return False

        return True

    if not validate_secret_name(secret_name):
        click.echo(
            "\n❌ Error: Invalid secret name. Reason: Possible values: 2 ≤ length ≤ 256, consisting of only alphanumeric letters and some symbols ( . - _ ).",
            err=True,
        )
        sys.exit(1)  # Exit with a non-zero status

    if not from_file and not value:
        value = click.prompt(
            "Please enter the secret value",
            type=str,
            hide_input=True,
        ).strip()

    try:
        secret, space_name, user = secrets_client.create_secret(
            secret_name,
            personal,
            value,
            space,
            from_file,
            callback=(None if quiet else echo_callback),
        )
        if secret:
            if format == "plain":
                if personal:
                    secrets_output = f"\n✅ New secret {secret_name} successfully created for user '{user}'."
                else:
                    secrets_output = f"\n✅ New secret {secret_name} successfully created in space '{space_name}'."
            else:
                secrets_output = json.dumps(secret)

            click.echo(secrets_output)
    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.argument("secret_name", required=True)
@click.option("--value", help="Secret value.")
@click.option("--space", help="Space name.")
@click.option(
    "--personal",
    is_flag=True,
    default=False,
    help="per-user mode",
)
@click.option(
    "--from-file",
    help=f"Path to the local .txt file containing the secret value.",
)
@click.option(
    "--format",
    default="plain",
    type=click.Choice(["plain", "json"], case_sensitive=True),
    help=f"Output format: plain (default), json",
)
@common_options
def update(
    ctx,
    secret_name,
    value,
    space,
    personal,
    from_file,
    format,
    skip_version_check,
    quiet,
):
    """Update secret"""
    if not skip_version_check:
        try:
            outdated_version = check_current_and_latest_versions()
        except Exception as e:
            click.echo(f"❌ {str(e)}.", err=True)
            ctx.exit(1)  # Exit with a non-zero status
        if outdated_version:
            click.echo(outdated_version, err=True)
            ctx.exit(1)  # Exit with a non-zero status

    if personal and space:
        click.echo(
            f"❌ Error: --personal and --space were provided. Only one can be provided.",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status

    if format == "json":
        quiet = True

    if not quiet:
        click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} update")

    secrets_client = GBClient.Secret(get_user_token())

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "encoding_update_secret":
                click.echo(f"📝 (1/2) Encoding secret...")
            case "updating_secret_spinner":
                spinner = callback_args.get("spinner", "")
                if personal:
                    user = callback_args.get("user", "")
                    callback_message = f"\r📝 (2/2) Updating secret {secret_name} for user '{user}'... {spinner}"
                else:
                    space = callback_args.get("space", "")
                    space_name = callback_args.get("space_name", "")
                    callback_message = f"\r📝 (2/2) Updating secret {secret_name} in space '{space}' ({space_name})... {spinner}"
                click.echo(callback_message, nl=False)
            case "error":
                click.echo(
                    f"\n❌ Secret can't be updated at this moment.. Reason: {callback_args.get('reason', '')}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case _:
                pass

    if not from_file and not value:
        value = click.prompt(
            "Please enter the new secret value",
            type=str,
            hide_input=True,
        ).strip()

    try:
        secret, space_name, user = secrets_client.update_secret(
            secret_name,
            personal,
            value,
            space,
            from_file,
            callback=(None if quiet else echo_callback),
        )
        if secret:
            if format == "plain":
                if personal:
                    secrets_output = f"\n✅ Secret {secret_name} successfully updated for user '{user}'."
                else:
                    secrets_output = f"\n✅ Secret {secret_name} successfully updated in space '{space_name}'."
            else:
                secrets_output = json.dumps(secret)

            click.echo(secrets_output)
    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.argument("secret_name", required=True)
@click.option("--space", help="Space name.")
@click.option(
    "--personal",
    is_flag=True,
    default=False,
    help="per-user mode",
)
@click.option(
    "--format",
    default="plain",
    type=click.Choice(["plain", "json"], case_sensitive=True),
    help=f"Output format: plain (default), json",
)
@common_options
def delete(ctx, secret_name, space, personal, format, skip_version_check, quiet):
    """Delete secret"""
    if not skip_version_check:
        try:
            outdated_version = check_current_and_latest_versions()
        except Exception as e:
            click.echo(f"❌ {str(e)}.", err=True)
            ctx.exit(1)  # Exit with a non-zero status
        if outdated_version:
            click.echo(outdated_version, err=True)
            ctx.exit(1)  # Exit with a non-zero status

    if personal and space:
        click.echo(
            f"❌ Error: --personal and --space were provided. Only one can be provided.",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status

    if format == "json":
        quiet = True

    if not quiet:
        click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} delete")

    secrets_client = GBClient.Secret(get_user_token())

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "deleting_secret_spinner":
                spinner = callback_args.get("spinner", "")
                if personal:
                    user = callback_args.get("user", "")
                    callback_message = f"\r📝 Deleting secret {secret_name} for user '{user}'... {spinner}"
                else:
                    space = callback_args.get("space", "")
                    space_name = callback_args.get("space_name", "")
                    callback_message = f"\r📝 Deleting secret {secret_name} in space '{space}' ({space_name})... {spinner}"
                click.echo(callback_message, nl=False)
            case "error":
                click.echo(
                    f"\n❌ Secret can't be deleted at this moment.. Reason: {callback_args.get('reason', '')}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case _:
                pass

    try:
        secret, space_name, user = secrets_client.delete_secret(
            secret_name,
            personal,
            space,
            callback=(None if quiet else echo_callback),
        )
        if secret:
            if format == "plain":
                if personal:
                    secrets_output = f"\n✅ Secret {secret_name} successfully deleted for user '{user}'."
                else:
                    secrets_output = f"\n✅ Secret {secret_name} successfully deleted in space '{space_name}'."
            else:
                secrets_output = json.dumps(secret)

            click.echo(secrets_output)
    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status
