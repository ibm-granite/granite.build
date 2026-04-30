"""Command tag module."""

import json

import click
from tabulate import tabulate

from gbcli.client.client import GBClient
from gbcli.commands.command_auth import str_exc_chain
from gbcli.commands.common_options import common_options
from gbcli.utils.gbconstants import CLIPBOARD_CHAR, PROJECT_NAME
from gbcli.utils.gbcredentials import get_user_token
from gbcli.utils.versionutil import check_current_and_latest_versions

logger_name = __name__


@click.group("tag")
@click.pass_context
def cli(ctx):
    """Work with tags"""
    ctx.ensure_object(dict)
    pass


@cli.command()
@click.pass_context
@click.option(
    "--builds",
    is_flag=True,
    default=False,
    help="List tags for builds.",
)
@click.option(
    "--artifacts",
    is_flag=True,
    default=False,
    help="List tags for artifacts.",
)
@click.option("--space", help="Space name.")
@click.option(
    "-u",
    "--username",
    help="Filter by username.",
)
@click.option(
    "--format",
    default="plain",
    type=click.Choice(["plain", "json"]),
    help="Output format.",
)
@common_options
def list(
    ctx,
    builds,
    artifacts,
    space,
    username,
    format,
    skip_version_check,
    quiet,
):
    """List tags from builds or artifacts"""

    # Validate that exactly one resource type is specified
    if builds and artifacts:
        click.echo(
            "❌ Error: Cannot specify both --builds and --artifacts",
            err=True,
        )
        ctx.exit(1)

    if not builds and not artifacts:
        click.echo(
            "❌ Error: You must specify either --builds or --artifacts",
            err=True,
        )
        ctx.exit(1)

    # Determine resource type
    resource_type = "builds" if builds else "artifacts"

    if not skip_version_check:
        try:
            outdated_version = check_current_and_latest_versions()
        except Exception as e:
            click.echo(f"❌ {str(e)}.", err=True)
            ctx.exit(1)
        if outdated_version:
            click.echo(outdated_version, err=True)
            ctx.exit(1)

    if format == "json":
        quiet = True

    tag_client = GBClient.Tag(get_user_token())

    if not quiet:
        click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} tag list")

    try:
        if resource_type == "builds":
            tags = tag_client.build_tag_list(username=username, space=space)
        else:  # artifacts
            tags = tag_client.artifact_tag_list(username=username, space=space)

        # Format and display output
        if format == "json":
            click.echo(json.dumps(tags))
        else:  # plain
            if not tags:
                click.echo("No tags found.")
            else:
                table_data = [[tag] for tag in tags]
                click.echo(tabulate(table_data, headers=["TAG"], tablefmt="plain"))

    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)
