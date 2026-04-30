"""Command template module."""

import json
import sys
from typing import Dict

import click
from tabulate import tabulate
from tqdm import tqdm

from gbcli.client.client import GBClient
from gbcli.commands.command_auth import str_exc_chain
from gbcli.commands.common_options import common_options
from gbcli.utils.gbconstants import (
    BUILD_DESCRIBE_ARTIFACTS_HEADERS,
    BUILD_DESCRIBE_STEPS_HEADERS,
    CLIPBOARD_CHAR,
    PROJECT_NAME,
    TEMPLATE_LIST_HEADERS,
)
from gbcli.utils.gbcredentials import get_user_token
from gbcli.utils.versionutil import check_current_and_latest_versions


@click.group("template")
@click.pass_context
def cli(ctx):
    """Work with templates"""
    pass


@cli.command()
@click.pass_context
@click.option("--template-repo", help="Template GitHub repository URL.")
@click.option("--space", help="Space name.")
@click.option(
    "--format",
    default="plain",
    type=click.Choice(["plain", "json"], case_sensitive=True),
    help=f"Output format: plain (default), json",
)
@common_options
def list(
    ctx,
    template_repo: str,
    space: str,
    format: str,
    skip_version_check: bool,
    quiet: bool,
):
    """List all available build definition templates"""
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

    if template_repo and space:
        click.echo(
            f"❌ Error: --template-repo and --space were provided. Only one can be provided.",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status

    if not quiet:
        click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} list templates")

    template_client = GBClient.Template(get_user_token())

    try:
        templates = []

        if quiet:
            templates = template_client.list_templates(space, template_repo, callback=None)
        else:
            with tqdm(
                total=100,
                miniters=1,
                desc=f"📝 Listing templates",
                bar_format="{desc} [{bar}] {percentage:3.0f}% {postfix}",
                ascii="-#",
                leave=False,
            ) as progress_bar:

                def update_bar(callback_event: str, callback_args: Dict):
                    steps = callback_args.get("steps", 0)
                    match callback_event:
                        case "listing_templates":
                            progress_bar.set_description(
                                f"📝 Listing templates from {'/'.join(callback_args.get('assets_repo', '').split('/')[3:])} repository"
                            )
                        case "listed_templates":
                            progress_bar.write(
                                f"📝 Listing templates from {'/'.join(callback_args.get('assets_repo', '').split('/')[3:])} repository:"
                            )
                        case _:
                            pass

                    progress_bar.update(n=steps)

                    return

                templates = template_client.list_templates(
                    space, template_repo, callback=update_bar
                )

        if len(templates) > 0:
            if format == "plain":
                templates_table = [[p["template_name"], p["description"]] for p in templates]
                templates_output = tabulate(
                    templates_table, TEMPLATE_LIST_HEADERS, tablefmt="plain"
                )
            else:
                templates_output = json.dumps(templates)

            click.echo(templates_output)
        else:
            click.echo("No templates found in supplied repository.")

    except Exception as e:
        error_message = str_exc_chain(e)
        if "not enough values to unpack" in error_message:
            click.echo(
                f"❌ Template repository could not be reached",
                err=True,
            )

        else:
            click.echo(f"❌ {error_message}", err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.argument("template_name", required=True)
@click.option("--template-repo", help="Template GitHub repository URL.")
@click.option("--space", help="Space name.")
@click.option(
    "--format",
    default="plain",
    type=click.Choice(["plain", "json"], case_sensitive=True),
    help=f"Output format: plain (default), json, full",
)
@common_options
def describe(
    ctx,
    template_name: str,
    template_repo: str,
    space: str,
    format: str,
    skip_version_check: bool,
    quiet: bool,
):
    """Show template contents"""
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

    if template_repo and space:
        click.echo(
            f"❌ Error: --template-repo and --space were provided. Only one can be provided.",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status

    if not quiet:
        click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} template describe")

    template_client = GBClient.Template(get_user_token())

    def parse_config_output(config: dict) -> str:
        config_str = ""
        for key in config.keys():
            if isinstance(config[key], dict):
                for inner_key in config[key].keys():
                    config_str = config_str + f"{key}.{inner_key}: {config[key][inner_key]}\n"
            else:
                config_str = config_str + f"{key}: {config[key]}\n"
            if "echo" in config_str:
                config_str = config_str.replace(";", ";\n")
        return config_str.replace("\\", "")

    try:
        targets = None

        if quiet:
            targets = template_client.describe_template(
                template_name, format, space, template_repo, callback=None
            )
        else:
            with tqdm(
                total=100,
                miniters=1,
                desc=f"📝 Describing template",
                bar_format="{desc} [{bar}] {percentage:3.0f}% {postfix}",
                ascii="-#",
                leave=False,
            ) as progress_bar:

                def update_bar(callback_event: str, callback_args: Dict):
                    steps = callback_args.get("steps", 0)
                    match callback_event:
                        case "processing_template_content":
                            progress_bar.set_description(
                                f"📝 Processing templates in {callback_args.get('assets_repo', '')}"
                            )
                        case "processed_template_content":
                            progress_bar.write(
                                f"📝 Listing {template_name} contents from {callback_args.get('assets_repo', '')}:"
                            )
                        case "error":
                            reason = callback_args.get("reason", "")
                            click.echo(
                                f"\n❌ {template_name} contents can't be retrieved at this moment... Reason: {reason}",
                                err=True,
                            )
                            sys.exit(1)  # Exit with a non-zero status
                        case _:
                            pass

                    progress_bar.update(n=steps)

                    return

                targets = template_client.describe_template(
                    template_name,
                    format,
                    space,
                    template_repo,
                    callback=update_bar,
                )

        if len(targets) > 0:
            if format != "json":
                steps_header = BUILD_DESCRIBE_STEPS_HEADERS
                if format == "simple":
                    steps_header.remove("CONFIG")
                for target in targets:
                    description_output = (
                        f"{target['target_name']}\n" + f"{target['environment_uri']}\n"
                    )

                    input_artifacts_table = [[i["name"], i["uri"]] for i in target["inputs"]]

                    output_artifacts_table = [[o["name"], o["uri"]] for o in target["outputs"]]

                    output_steps_table = [
                        [s["uri"]]
                        + ([parse_config_output(s["config"])] if format == "full" else [])
                        for s in target["steps"]
                    ]

                    input_artifacts_output = tabulate(
                        input_artifacts_table,
                        BUILD_DESCRIBE_ARTIFACTS_HEADERS,
                        tablefmt="plain",
                    )

                    output_artifacts_output = tabulate(
                        output_artifacts_table,
                        BUILD_DESCRIBE_ARTIFACTS_HEADERS,
                        tablefmt="plain",
                    )

                    output_steps_output = tabulate(
                        output_steps_table,
                        steps_header,
                        tablefmt="plain",
                    )

                    description = (
                        "------------------------------\n"
                        + description_output
                        + (
                            ("\n*️⃣  Input artifacts\n" + input_artifacts_output + "\n")
                            if len(target["inputs"]) > 0
                            else ""
                        )
                        + (
                            ("\n*️⃣  Output artifacts\n" + output_artifacts_output + "\n")
                            if len(target["outputs"]) > 0
                            else ""
                        )
                        + (
                            ("\n*️⃣  Steps\n" + output_steps_output)
                            if len(target["steps"]) > 0
                            else ""
                        )
                    )

                    click.echo(description)
            else:
                click.echo(targets)
    except Exception as e:
        error_message = str_exc_chain(e)
        if "not enough values to unpack" in error_message:
            click.echo(
                f"❌ Template repository could not be reached",
                err=True,
            )

        else:
            click.echo(f"❌ {error_message}", err=True)
        ctx.exit(1)  # Exit with a non-zero status
