"""Command step module."""

import json
import sys
import time
from typing import Dict, List

import click
from tabulate import tabulate
from tqdm import tqdm

from gbcli.client.client import GBClient
from gbcli.commands.command_auth import str_exc_chain
from gbcli.commands.common_options import common_options
from gbcli.utils.gbconstants import (
    CLIPBOARD_CHAR,
    PROJECT_NAME,
    STEP_DESCRIBE_HEADERS,
    STEP_LIST_HEADERS,
)
from gbcli.utils.gbcredentials import get_user_token
from gbcli.utils.utils import step_uri_notation
from gbcli.utils.versionutil import check_current_and_latest_versions


@click.group("step")
@click.pass_context
def cli(ctx):
    """Work with steps"""
    pass


@cli.command()
@click.pass_context
@click.option("--space", help="Space name.")
@click.option("--step-repo", help="Step GitHub repository URL.")
@click.option(
    "--format",
    default="plain",
    type=click.Choice(["plain", "json"], case_sensitive=True),
    help=f"Output format: plain (default), json",
)
@common_options
def list(ctx, space: str, step_repo: str, format: str, skip_version_check: bool, quiet: bool):
    """List all available build definition steps"""
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

    if step_repo and space:
        click.echo(
            f"❌ Error: --step-repo and --space were provided. Only one can be provided.",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status

    if not quiet:
        click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} list steps")

    step_client = GBClient.Step(get_user_token())

    try:
        steps = []

        if quiet:
            steps = step_client.list_steps(step_repo, space, callback=None)
        else:
            with tqdm(
                total=100,
                miniters=1,
                desc=f"📝 Listing steps",
                bar_format="{desc} [{bar}] {percentage:3.0f}% {postfix}",
                ascii="-#",
                leave=False,
            ) as progress_bar:

                def update_bar(callback_event: str, callback_args: Dict):
                    steps = callback_args.get("steps", 0)
                    match callback_event:
                        case "fetching_spaces":
                            progress_bar.set_description(f"📝 Fetching available spaces...")
                            progress_bar.update(n=steps)
                        case "done_fetching_spaces":
                            progress_bar.update(n=steps)
                            time.sleep(0.1)
                            progress_bar.write(f"📝 Obtained available spaces.")
                        case "listing_steps":
                            progress_bar.reset(total=100)
                            callback_steps_repo = callback_args.get("steps_repo", "")
                            listing_steps_description = (
                                f"📝 Listing steps for space {callback_args.get('space', '')} ({'/'.join(callback_steps_repo.split('/')[-2:])})"
                                if not step_repo
                                else f"📝 Listing steps from {'/'.join(callback_steps_repo.split('/')[-2:])} repository"
                            )
                            progress_bar.set_description(listing_steps_description)
                            progress_bar.update(n=steps)
                        case "listed_steps":
                            progress_bar.update(n=steps)
                            callback_steps_repo = callback_args.get("steps_repo", "")
                            listed_steps_output = (
                                f"📝 Listing steps for space {callback_args.get('space', '')} ({'/'.join(callback_steps_repo.split('/')[-2:])}):"
                                if not step_repo
                                else f"📝 Listing steps from {'/'.join(callback_steps_repo.split('/')[-2:])} repository:"
                            )
                            progress_bar.write(listed_steps_output)
                        case "error":
                            reason = callback_args.get("reason", "")
                            click.echo(
                                f"\n❌ Steps can't be retrieved at this moment... Reason: {reason}",
                                err=True,
                            )
                            sys.exit(1)  # Exit with a non-zero status
                        case _:
                            pass

                steps = step_client.list_steps(step_repo, space, callback=update_bar)

        if len(steps) > 0:
            if format == "plain":
                steps_table = [
                    [
                        s["step_name"],
                        s["description"],
                        step_uri_notation(s["step_name"]),
                    ]
                    for s in steps
                ]
                steps_output = tabulate(steps_table, STEP_LIST_HEADERS, tablefmt="plain")
            else:
                steps_output = json.dumps(steps)

            click.echo(steps_output)

    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.argument("step_name", required=True)
@click.option("--space", help="Space name.")
@click.option("--step-repo", help="Step GitHub repository URL.")
@click.option(
    "--format",
    default="plain",
    type=click.Choice(["plain", "json"], case_sensitive=True),
    help=f"Output format: plain (default), json",
)
@common_options
def describe(
    ctx,
    step_name: str,
    space: str,
    step_repo: str,
    format: str,
    skip_version_check: bool,
    quiet: bool,
):
    """Show step contents"""
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

    if step_repo and space:
        click.echo(
            f"❌ Error: --step-repo and --space were provided. Only one can be provided.",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status

    if not quiet:
        click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} step describe")

    step_client = GBClient.Step(get_user_token())

    def parse_step_config_output(config: dict) -> str:
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

    def parse_event_config_output(event_config: dict) -> str:
        config_str = ""
        for key in event_config.keys():
            if isinstance(event_config[key], List):
                config_str = config_str + "\n"
                for inner_property in event_config[key]:
                    config_str = config_str + f"{key}.{parse_step_config_output(inner_property)}\n"
            else:
                config_str = config_str + parse_step_config_output({key: event_config[key]})
        return config_str

    try:
        if quiet:
            step_content = step_client.describe_step(step_name, step_repo, space, callback=None)
        else:
            with tqdm(
                total=100,
                miniters=1,
                desc=f"📝 Describing step",
                bar_format="{desc} [{bar}] {percentage:3.0f}% {postfix}",
                ascii="-#",
                leave=False,
            ) as progress_bar:

                def update_bar(callback_event: str, callback_args: Dict):
                    steps = callback_args.get("steps", 0)
                    match callback_event:
                        case "processing_step_content":
                            progress_bar.set_description(
                                f"📝 Fetching step from {callback_args.get('steps_repo', '')}"
                            )
                        case "processed_step_content":
                            progress_bar.write(
                                f"📝 Listing {step_name} contents from {callback_args.get('steps_repo', '')}:"
                            )
                        case "error":
                            reason = callback_args.get("reason", "")
                            click.echo(
                                f"\n❌ {step_name} contents can't be retrieved at this moment... Reason: {reason}",
                                err=True,
                            )
                            sys.exit(1)  # Exit with a non-zero status
                        case _:
                            pass

                    progress_bar.update(n=steps)

                    return

                step_content = step_client.describe_step(
                    step_name, step_repo, space, callback=update_bar
                )

        if step_content:
            if format == "plain":
                parsed_configs = []
                for config in step_content["config"]:
                    for c in config:
                        if isinstance(config[c], dict):
                            parsed_configs.append([c, parse_step_config_output(config[c])])
                        else:
                            parsed_configs.append([c, config[c]])

                config_output = tabulate(
                    parsed_configs,
                    STEP_DESCRIBE_HEADERS,
                    tablefmt="plain",
                )

                parsed_env_configs = []
                for env_config in step_content["environment_configs"]:
                    env_config_name = next(iter(env_config))
                    launchers = env_config[env_config_name].get("launchers", [])
                    monitors = env_config[env_config_name].get("monitors", [])
                    for index, launcher in enumerate(launchers):
                        for l in launcher:
                            parsed_env_configs.append(
                                [
                                    f"{env_config_name}.launchers.{l}",
                                    parse_step_config_output(launchers[index][l]),
                                ]
                            )
                    for index, monitor in enumerate(monitors):
                        for m in monitor:
                            parsed_env_configs.append(
                                [
                                    f"{env_config_name}.monitors.{m}.type",
                                    monitors[index][m].get("type"),
                                ]
                            )

                            monitor_config = monitors[index][m].get("config")
                            if monitor_config:
                                event_configs = monitor_config.get("event_configs", [])
                                parsed_event_configs = [
                                    parse_event_config_output(ec) for ec in event_configs
                                ]
                                parsed_env_configs.append(
                                    [
                                        f"{env_config_name}.monitors.{m}.config.event_configs",
                                        "\n".join(parsed_event_configs),
                                    ]
                                )

                env_config_output = tabulate(
                    parsed_env_configs,
                    STEP_DESCRIBE_HEADERS,
                    tablefmt="plain",
                )

                describe_output = (
                    f"Name: {step_content['name']}\n"
                    + f"Type: {step_content['type']}\n"
                    + f"URI: {step_uri_notation(step_content['name'])}\n"
                    + (
                        ("\n*️⃣  Configs\n" + config_output + "\n")
                        if len(step_content["config"]) > 0
                        else ""
                    )
                    + (
                        ("\n*️⃣  Environment Configs\n" + env_config_output + "\n")
                        if len(step_content["environment_configs"]) > 0
                        else ""
                    )
                    + (
                        ("\n*️⃣  Documentation\n" + step_content["readme"] + "\n")
                        if len(step_content["readme"]) > 0
                        else ""
                    )
                )
            else:
                describe_output = json.dumps(step_content)

            click.echo(describe_output)

    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status


# @cli.command()
# @click.pass_context
# def add(ctx):
#     """Create a step from existing script. Sets up a new launcher to use. Adds to the step registry for this space."""
#     click.echo("This command is not yet available.")
#     ctx.exit(1)
