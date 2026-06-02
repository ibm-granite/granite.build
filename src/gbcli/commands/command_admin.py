import json
import sys
from typing import Dict

import click
import dateparser
from tabulate import tabulate

from gbcli.client.client import GBClient
from gbcli.commands.command_auth import str_exc_chain
from gbcli.commands.common_options import common_options, exit_if_standalone
from gbcli.utils.gbconstants import (
    BUILD_LOG_DEFAULT_QUERY_RANGE,
    BUILD_LOG_MAX_LOG_LIFESPAN,
    BUILD_LOG_MAX_QUERY_RANGE,
    BUILD_LOG_SECONDS_IN_A_DAY,
    CLIPBOARD_CHAR,
    PROJECT_NAME,
)
from gbcli.utils.gbcredentials import get_user_token
from gbcli.utils.utils import (
    change_timestamp_by_days,
    epoch_to_iso_date,
    get_current_epoch,
    parse_build_identifier,
)
from gbcli.utils.versionutil import check_current_and_latest_versions


@click.group("admin")
@click.pass_context
def cli(ctx):
    """Functions for admin users"""
    if ctx.invoked_subcommand is not None:
        exit_if_standalone("admin")


@cli.command()
@click.pass_context
@click.argument("module", required=True)
@click.option("--build-id", help=f"Filtered by build id")
@click.option("--build-step-id", help=f"Filtered by step id")
@click.option("--build-step-name", help=f"Filtered by step name")
@click.option(
    "--start-date",
    type=click.STRING,
    help=f"Start date: '2025/05/31', '3 days ago', 'now', 'May 31 23:00:00', unix epoch time (in seconds)",
)
@click.option(
    "--end-date",
    type=click.STRING,
    help=f"End date: '2025/05/31', '3 days ago', 'now', 'May 31 23:00:00', unix epoch time (in seconds)",
)
@click.option("--page-size", "-n", type=int, default=50, help=f"Show max n logs")
@click.option("--page-index", type=int, default=0, help=f"Page index")
@click.option(
    "--stream",
    type=click.Choice(["stdout", "stderr"], case_sensitive=True),
    help=f"Stream option: stdout, stderr",
)
@click.option("--text", help=f"Search text (Lucene)")
@click.option(
    "--sort",
    default="desc",
    type=click.Choice(["desc", "asc"], case_sensitive=True),
    help=f"Sort direction: desc (default), asc",
)
@click.option("--head", type=int, help=f"Show the first n logs")
@click.option("--tail", type=int, help=f"Show the last n logs")
@click.option(
    "--format",
    default="plain",
    type=click.Choice(["plain", "json", "full"], case_sensitive=True),
    help=f"Output format: plain (default), json, full",
)
@click.option(
    "--follow",
    "-f",
    is_flag=True,
    default=False,
    help="Get recent logs.",
)
@click.option(
    "--all",
    is_flag=True,
    default=False,
    help="Retrieve all build logs.",
)
@common_options
def log(
    ctx,
    module,
    build_id,
    build_step_id,
    build_step_name,
    start_date,
    end_date,
    page_size,
    page_index,
    stream,
    text,
    sort,
    head,
    tail,
    format,
    follow,
    all,
    skip_version_check,
    quiet,
):
    """
    Get server module logs

    Provide module: gbserver-rest-server, gbserver-pr-watch, gbserver-build-watch, or gbserver-build-runner
    """
    if not skip_version_check:
        try:
            outdated_version = check_current_and_latest_versions()
        except Exception as e:
            click.echo(f"❌ {str(e)}.", err=True)
            ctx.exit(1)  # Exit with a non-zero status
        if outdated_version:
            click.echo(outdated_version, err=True)
            ctx.exit(1)  # Exit with a non-zero status

    if follow and (end_date or sort == "asc" or head or tail or format == "json"):
        click.echo(
            f"❌ Error: -f/--follow was provided. It can't be used with --end-date, --head, --tail, '--sort asc' or '--format json' options.",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status
    if all and (head or tail or format == "json"):
        click.echo(
            f"❌ Error: --all was provided. It can't be used with --head, --tail or '--format json' options.",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status

    id_format = None
    if build_id:
        id_format = parse_build_identifier(build_id)
        if id_format not in ["uuid", "url"]:
            click.echo(
                f"❌ Build identifier formatted incorrectly. Please try again with valid build ID or URL.",
                err=True,
            )
            sys.exit(1)

        if id_format == "uuid" and len(build_id) < 36:
            click.echo(
                f"❌ Build ID formatted incorrectly. Please try again with a valid build ID.",
                err=True,
            )
            sys.exit(1)

    # start_date and end_date
    current_epoch = get_current_epoch()
    if start_date:
        try:
            start_epoch = int(start_date)  # unix epoch time
        except ValueError:
            start_epoch = round(dateparser.parse(start_date).timestamp())
    else:
        start_epoch = change_timestamp_by_days(
            current_epoch, BUILD_LOG_DEFAULT_QUERY_RANGE
        )

    if end_date:
        try:
            end_epoch = int(end_date)  # unix epoch time
        except ValueError:
            end_epoch = round(dateparser.parse(end_date).timestamp())
    else:
        if all:
            end_epoch = None
        else:
            end_epoch = current_epoch

    if not start_date and end_date:
        start_epoch = change_timestamp_by_days(end_epoch, BUILD_LOG_MAX_QUERY_RANGE)
        click.echo(
            f"⚠️  Warning: the maximum log time range is {BUILD_LOG_MAX_QUERY_RANGE} days. Automatically setting the start date.",
            err=True,
        )
    if start_date and not end_date and not follow and not all:
        end_epoch = change_timestamp_by_days(start_epoch, BUILD_LOG_MAX_QUERY_RANGE)
        click.echo(
            f"⚠️  Warning: the maximum log time range is {BUILD_LOG_MAX_QUERY_RANGE} days. Automatically setting the end date.",
            err=True,
        )

    if (
        round((current_epoch - start_epoch) / BUILD_LOG_SECONDS_IN_A_DAY)
        > BUILD_LOG_MAX_LOG_LIFESPAN
    ):
        click.echo(
            f"⚠️  Warning: the log service only keeps the logs up to {BUILD_LOG_MAX_LOG_LIFESPAN} days. start_date is beyond that.",
            err=True,
        )
    if (
        end_epoch != None
        and round((current_epoch - end_epoch) / BUILD_LOG_SECONDS_IN_A_DAY)
        > BUILD_LOG_MAX_LOG_LIFESPAN
    ):
        click.echo(
            f"⚠️  Warning: the log service only keeps the logs up to {BUILD_LOG_MAX_LOG_LIFESPAN} days. end_date is beyond that.",
            err=True,
        )

    if end_epoch and start_epoch > end_epoch:
        click.echo(
            f"❌ start_date is later than end_date. Please enter a valid start and end date.",
            err=True,
        )
        sys.exit(1)
    if (
        not all
        and round((end_epoch - start_epoch) / BUILD_LOG_SECONDS_IN_A_DAY)
        > BUILD_LOG_MAX_QUERY_RANGE
    ):
        click.echo(
            f"❌ The maximum log time range is {BUILD_LOG_MAX_QUERY_RANGE} days. Please enter a valid start and end date.",
            err=True,
        )
        sys.exit(1)

    click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} build log")
    admin_client = GBClient.Admin(get_user_token())

    if head != None:
        sort = "asc"
        page_size = head

    if tail != None:
        sort = "desc"
        page_size = tail

    erase_sequence = "\r\033[K"

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "fetching_build_id":
                click.echo(
                    f"Obtaining build ID for build URL {callback_args.get('source_uri', '')}.",
                    nl=False,
                )
            case "fetched_build_id":
                click.echo(
                    f"{erase_sequence}Obtained build ID {callback_args.get('build_id', '')} for build URL {callback_args.get('source_uri', '')}."
                )
            case "querying_log":
                click.echo(
                    f"Querying the logs between {epoch_to_iso_date(callback_args.get('start_epoch', ''))} and {epoch_to_iso_date(callback_args.get('end_epoch', ''))}\nQuerying log server..\n"
                )
            case "display_logs":
                logs = callback_args.get("logs", [])
                output_format_plain(logs)
            case "error":
                click.echo(
                    f"\n❌ Logs can't be retrieved at this moment.. Reason: {callback_args.get('reason', '')}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case _:
                pass

    def output_format_plain(logs):
        for log in logs:
            log_json = json.loads(log["text"])
            if log_json.get("log") != None:
                click.echo(f"{log_json['log']}")
            else:
                click.echo(f"<null>")

    def output_format_json(logs):
        click.echo(json.dumps(logs))

    def output_format_full(logs):
        for _index, log in enumerate(logs):
            index = _index + 1

            text_json = json.loads(log["text"])
            kubernetes_json = text_json.pop("kubernetes")
            labels_json = kubernetes_json.pop("labels")

            HEADER = ["KEY", "VALUE"]
            table1 = tabulate(
                sorted(text_json.items()),
                HEADER,
                tablefmt="plain",
                maxcolwidths=120,
            )
            table2 = tabulate(
                sorted(kubernetes_json.items()),
                HEADER,
                tablefmt="plain",
                maxcolwidths=120,
            )
            table3 = tabulate(
                sorted(labels_json.items()),
                HEADER,
                tablefmt="plain",
                maxcolwidths=120,
            )

            click.echo(f"\n[{index}]\n")
            click.echo(f"{table1}\n")
            click.echo(f"For 'kubernetes':\n{table2}\n")
            click.echo(f"For 'label' in 'kubernetes':\n{table3}\n")

    try:
        logs = admin_client.server_log(
            module,
            id_format,
            start_epoch,
            end_epoch,
            page_size,
            page_index,
            stream,
            text,
            sort,
            build_id,
            build_step_id,
            build_step_name,
            follow,
            all,
            echo_callback,
        )

        if logs != None:
            # query is successful
            if len(logs) > 0:
                if not all:
                    if sort == "desc":
                        logs.reverse()

                    match format:
                        case "plain":
                            output_format_plain(logs)
                        case "json":
                            output_format_json(logs)
                        case "full":
                            output_format_full(logs)
                        case _:
                            pass
            else:
                click.echo(
                    f"❗ Query is successful, but it can't find the logs based on input parameters",
                    err=True,
                )

            click.echo(f"\n✅ Total number of logs returned: {len(logs)}")

    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command("space-membership")
@click.pass_context
@click.option("--add", "add_user", default=None, help="Username to add to the space.")
@click.option(
    "--delete", "delete_user", default=None, help="Username to remove from the space."
)
@click.option(
    "--update",
    "update_user",
    default=None,
    help="Username whose role to update.",
)
@click.option(
    "--role",
    type=click.Choice(["admin", "member"], case_sensitive=True),
    help="Role for --add or --update.",
)
@click.option(
    "--space", default=None, help="Space name. Uses default space if not specified."
)
@click.option(
    "--format",
    default="plain",
    type=click.Choice(["plain", "json"], case_sensitive=True),
    help="Output format: plain (default), json",
)
@common_options
def space_membership(
    ctx,
    add_user,
    delete_user,
    update_user,
    role,
    space,
    format,
    skip_version_check,
    quiet,
):
    """Manage space members

    By default, lists all members in the space. Use --add, --delete, or --update to modify membership.
    """
    if not skip_version_check:
        try:
            outdated_version = check_current_and_latest_versions()
        except Exception as e:
            click.echo(f"❌ {str(e)}.", err=True)
            ctx.exit(1)
        if outdated_version:
            click.echo(outdated_version, err=True)
            ctx.exit(1)

    # Validate mutually exclusive action options
    actions = [add_user, delete_user, update_user]
    action_count = sum(1 for a in actions if a is not None)
    if action_count > 1:
        click.echo(
            "❌ Error: only one of --add, --delete, or --update can be provided.",
            err=True,
        )
        ctx.exit(1)

    # --role is required for --add and --update, and invalid otherwise
    if (add_user or update_user) and role is None:
        click.echo(
            "❌ Error: --role is required when using --add or --update.",
            err=True,
        )
        ctx.exit(1)
    if role is not None and not add_user and not update_user:
        click.echo(
            "❌ Error: --role can only be used with --add or --update.",
            err=True,
        )
        ctx.exit(1)

    if format == "json":
        quiet = True

    admin_client = GBClient.Admin(get_user_token())

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "error":
                click.echo(
                    f"\n❌ {callback_args.get('reason', '')}",
                    err=True,
                )
                sys.exit(1)
            case _:
                pass

    callback = None if quiet else echo_callback

    try:
        if add_user:
            if not quiet:
                click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} adding member to space")
            result = admin_client.add_space_member(space, add_user, role, callback)
            if result:
                if format == "json":
                    click.echo(json.dumps(result))
                else:
                    member = result.get("member", {})
                    click.echo(
                        f"✅ User '{member.get('username')}' added to space '{member.get('space_name')}' with role '{member.get('role')}'."
                    )
        elif delete_user:
            if not quiet:
                click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} removing member from space")
            result = admin_client.delete_space_member(space, delete_user, callback)
            if result:
                if format == "json":
                    click.echo(json.dumps(result))
                else:
                    click.echo(f"✅ User '{delete_user}' removed from space.")
        elif update_user:
            if not quiet:
                click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} updating member role")
            result = admin_client.update_space_member(
                space, update_user, role, callback
            )
            if result:
                if format == "json":
                    click.echo(json.dumps(result))
                else:
                    member = result.get("member", {})
                    click.echo(
                        f"✅ User '{member.get('username')}' role updated to '{member.get('role')}' in space '{member.get('space_name')}'."
                    )
        else:
            # Default: list members
            if not quiet:
                click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} space membership")
            result = admin_client.list_space_members(space, callback)
            if result:
                members = result.get("members", [])
                if format == "json":
                    click.echo(json.dumps(result))
                else:
                    if members:
                        members_table = [
                            [m.get("username"), m.get("role")] for m in members
                        ]
                        click.echo(
                            tabulate(
                                members_table,
                                ["USERNAME", "ROLE"],
                                tablefmt="plain",
                            )
                        )
                    else:
                        click.echo("No members found in this space.")
    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)
