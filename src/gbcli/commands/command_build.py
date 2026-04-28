import click
import dateparser
import json
import logging
import os
import sys
import webbrowser
from numpy import ceil
from tabulate import tabulate
from tqdm import tqdm
from typing import Any, Dict, List

from gbcli.client.client import GBClient
from gbcli.commands.command_auth import execute_with_spinner, str_exc_chain
from gbcli.utils.gbcredentials import get_user_token
from gbcli.commands.common_options import common_options
from gbcli.utils.gbconstants import (
    ASSETS_REPO_NAME,
    BUILD_DESCRIBE_ARTIFACTS_HEADERS,
    BUILD_DESCRIBE_STEPS_HEADERS,
    BUILD_LINEAGE_DEFAULT_HEADERS,
    BUILD_LINEAGE_FULL_HEADERS,
    BUILD_LIST_HEADERS,
    BUILD_LOG_DEFAULT_QUERY_RANGE,
    BUILD_LOG_SECONDS_IN_A_DAY,
    BUILD_LOG_MAX_LOG_LIFESPAN,
    BUILD_LOG_MAX_QUERY_RANGE,
    BUILD_STATUS_ARTIFACTS_HEADERS,
    BUILD_STATUS_HISTORY_HEADERS,
    BUILD_STATUS_STEPS_HEADERS,
    CLIPBOARD_CHAR,
    DMF_URL,
    PROJECT_NAME,
    is_standalone,
)
from gbcli.utils.lh_auth import AuthException
from gbcli.utils.utils import (
    change_timestamp_by_days,
    check_runnable_browser,
    combine_tags,
    custom_parse_markdown_str,
    datetime_to_string,
    epoch_to_iso_date,
    get_build_lineage_url,
    get_current_epoch,
    humanize_iso_date,
    pagination_range,
    parse_build_identifier,
    parse_markdown_str,
    validate_tags,
)
from gbcli.utils.versionutil import check_current_and_latest_versions
from gbcli.utils.click_utils import validation_formatting

logger = logging.getLogger(__name__)

ALLOWED_VAL_TYPES = ["static", "dynamic"]


def get_status_emoji(status: str) -> str:
    match status.lower():
        case "submitted":
            return "◌"
        case "success":
            return "✅"
        case "pending":
            return "🔵"
        case "running":
            return "⚡"
        case "failed":
            return "❌"
        case "invalid":
            return "❌"
        case "cancelled":
            return "⚠️"
        case "cancel_requested":
            return "⚠️"
        case _:
            return ""


def execution_status_plain_output(
    details: Any,
    targets: List[Any],
    history: Any,
    show_events: bool,
):

    targets_overview = [
        f"\n\tTarget #{index + 1} {target}: {get_status_emoji(targets[target]['status'])} {str(targets[target]['status']).upper()}\n"
        for index, target in enumerate(targets)
    ]
    source_pr = f"<{details['source_pr']}>" if details["source_pr"] else "-"
    details_output = f"""
# Build {details['build_id']}

- **Build name**: {details['name']}
- **Build description**: {details.get('description', '')}
- **Status**: {get_status_emoji(details['status'])} {str(details['status']).upper()}
- **Started**: {datetime_to_string(details['started_at'])}
- **Updated**: {datetime_to_string(details['updated_at'])}
- **Status page**: <{DMF_URL}/gb/builds/{details['build_id']}>
- **Build PR**: {source_pr}
- **Targets**
{"".join(targets_overview)}
    """

    target_outputs = []
    for index, target in enumerate(targets):
        input_artifacts_table = [
            [i["artifact_id"], i["uri"]] for i in targets[target]["input_artifacts"]
        ]
        output_artifacts_table = [
            [o["artifact_id"], o["uri"]] for o in targets[target]["output_artifacts"]
        ]

        steps_table = [
            [
                s["step_id"],
                s["uri"].split("/")[-1],
                f"{get_status_emoji(s['status'])} {str(s['status']).upper()}",
                s["uri"],
            ]
            for s in targets[target]["steps"]
        ]

        input_artifacts_output = tabulate(
            input_artifacts_table,
            BUILD_STATUS_ARTIFACTS_HEADERS,
            tablefmt="github",
        )

        output_artifacts_output = tabulate(
            output_artifacts_table,
            BUILD_STATUS_ARTIFACTS_HEADERS,
            tablefmt="github",
        )

        steps_output = tabulate(
            steps_table,
            BUILD_STATUS_STEPS_HEADERS,
            tablefmt="github",
        )

        target_output = f"""
---

## Target #{index + 1} {target}

{get_status_emoji(targets[target]['status'])} **Status**: {str(targets[target]['status']).upper()}

### ⚙️  Steps

{steps_output if len(targets[target]["steps"]) > 0 else ""}

### 📦 Input artifacts

{input_artifacts_output if len(targets[target]["input_artifacts"]) > 0 else ""}

### 📦 Output artifacts

{output_artifacts_output if len(targets[target]["output_artifacts"]) > 0 else ""}
        """

        target_outputs.append(target_output)
    target_outputs = "".join(target_outputs)

    history_table = [[humanize_iso_date(h["time"]), h["description"]] for h in history]
    history_output = tabulate(
        history_table,
        BUILD_STATUS_HISTORY_HEADERS,
        tablefmt="plain",
    )

    markdown_output = f"""
{details_output}
{target_outputs}


---

## Build history

    """
    status = f"""
{custom_parse_markdown_str(markdown_output)}
{'Use "--show-events" to show all build events.' if not show_events else (history_output if len(history) > 0 else "")}
    """

    return status


@click.group("build")
@click.pass_context
def cli(ctx):
    """Work with builds"""
    pass


@cli.command()
@click.pass_context
@click.argument("build_name", required=False)
@click.option(
    "--filename",
    "-f",
    help="Build file name instead of build.yaml. A build folder won't be created if this option is used.",
)
@click.option("--space", help="Space name.")
@click.option(
    "--from-build",
    help=f"Initialize build from a previous build. Provide build ID or URL.",
)
@click.option(
    "--from-template", help=f"Create build definition from an existing template"
)
@click.option("--template-repo", help="Template GitHub repository URL")
@click.option(
    "--format",
    "format",
    default="simple",
    type=click.Choice(["simple", "json"], case_sensitive=True),
    help="Output format: simple (default), json",
)
@common_options
def init(
    ctx,
    filename,
    space,
    build_name,
    from_build,
    from_template,
    template_repo,
    format,
    skip_version_check,
    quiet,
):
    """Create a build definition with the folder structure"""
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

    if filename is None:
        if build_name is None:
            click.echo("Error: Missing argument 'BUILD_NAME'.", err=True)
            ctx.exit(1)  # Exit with a non-zero status
    else:
        if build_name is not None:
            click.echo(
                f"Warning: BUILD_NAME '{build_name}' is ignored when filename is specified."
            )
        if os.path.exists(filename):
            click.echo(f"Error: File {filename} already exists.", err=True)
            ctx.exit(1)  # Exit with a non-zero status

    if not from_template and not from_build:
        click.echo(
            f"❌ Error: Please specify a template with --from-template option or a build with --from-build.",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status

    if template_repo and space:
        click.echo(
            f"❌ Error: --template-repo and --space were provided. Only one can be provided.",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status

    id_format = None
    if from_build:
        id_format = parse_build_identifier(from_build)
        if id_format not in ["uuid", "url"]:
            click.echo(
                f"❌ Build identifier formatted incorrectly. Please try again with valid build ID or URL.",
                err=True,
            )
            sys.exit(1)

        if id_format == "uuid" and len(from_build) < 36:
            click.echo(
                f"❌ Build ID formatted incorrectly. Please try again with a valid build ID.",
                err=True,
            )
            sys.exit(1)

    if not quiet:
        click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} build init")

    build_client = GBClient.Build(get_user_token())

    try:
        init = None

        if from_build:
            progress_bar_description = f"Cloning build from existing build {from_build}"
        else:
            progress_bar_description = (
                f"Cloning template from {template_repo.split('/')[-1] if template_repo else ASSETS_REPO_NAME} repository"
                if not space
                else f"Cloning template from the space '{space}' repository"
            )

        if quiet:
            init, source = build_client.build_init(
                build_name,
                filename,
                space,
                from_build,
                from_template,
                template_repo,
                id_format,
                None,
            )
        else:
            with tqdm(
                total=100,
                miniters=1,
                desc=progress_bar_description,
                bar_format="{desc} [{bar}] {percentage:3.0f}% {postfix}",
                ascii="-#",
                leave=False,
            ) as progress_bar:

                def update_bar(callback_event: str, callback_args: Dict):
                    steps = callback_args.get("steps", 0)
                    match callback_event:
                        case "fetching_spaces":
                            progress_bar.reset(total=100)
                            progress_bar.set_description(
                                "Fetching spaces from GBSERVER"
                            )
                            progress_bar.update(n=steps)
                        case "done_fetching_spaces":
                            progress_bar.update(n=steps)
                        case "preparing_contents":
                            if steps == 1:
                                progress_bar.reset(total=400)
                                if progress_bar.desc != progress_bar_description:
                                    progress_bar.set_description(
                                        progress_bar_description
                                    )
                            progress_bar.update(n=steps)
                        case "warning":
                            progress_bar.clear()
                            reason = callback_args.get("reason", "")
                            if not quiet:
                                click.echo(f"\n⚠️  Warning: {reason}\n", err=True)
                        case "error":
                            reason = callback_args.get("reason", "")
                            progress_bar.clear()
                            click.echo(
                                f"\n❌ Build folder can't be created at this moment... Reason: {reason}",
                                err=True,
                            )
                            sys.exit(1)  # Exit with a non-zero status
                    return

                init, source = build_client.build_init(
                    build_name,
                    filename,
                    space,
                    from_build,
                    from_template,
                    template_repo,
                    id_format,
                    update_bar,
                )

        if init and source:
            if not quiet:
                if from_build:
                    click.echo(
                        f"✅ Build {build_name if build_name else filename} was successfully created from build {source}."
                    )
                else:
                    click.echo(
                        f"✅ Build {build_name if build_name else filename} was successfully created from template {source}."
                    )
            if format == "json":
                click.echo(
                    json.dumps(
                        {
                            "init": init,
                            "source": source,
                            "build_name": build_name if build_name else filename,
                        }
                    )
                )
    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.option(
    "--filename",
    "-f",
    type=click.Path(exists=True, readable=True, dir_okay=False),
    help="file name (and path) for build yaml",
)
@click.option("--space", help="Space name.")
@click.option(
    "--param",
    multiple=True,
    help="Build run parameter ('param_name=param_value'). Use this option again to provide multiple parameters.",
)
@click.option(
    "--description",  # Enable --description is an alias of -m/--message.
    "-m",
    "--message",
    type=str,
    default="",
    help="Add a custom message in the PR comments. Useful for adding some info about the build",
)
@click.option(
    "--tag",
    multiple=True,
    help="Single tag name or comma-separated list of tag names. Tag names can only contain alphanumeric characters, underscores, and hyphens.",
)
@click.option(
    "--tags", help="Comma-separated list of tag names. For example: 'tagA, tagB'."
)
@click.option("--skip-validation", is_flag=True, help="Skip build contents validation.")
@click.option(
    "--parameters-path",
    type=click.Path(exists=True, readable=True, dir_okay=False),
    help="Path to parameters.yaml file.",
)
@click.option(
    "--verbose-validation",
    is_flag=True,
    default=False,
    help="Show greater detail for validation errors and warnings",
)
@click.option(
    "--format",
    "format",
    default="simple",
    type=click.Choice(["simple", "json"], case_sensitive=True),
    help="Output format: simple (default), json",
)
@click.argument("targets", nargs=-1)
@click.option(
    "--validation-type",
    type=click.Choice(ALLOWED_VAL_TYPES),
    default="static",
    help=f"The type of validation to perform, one of {ALLOWED_VAL_TYPES} . Default is 'static'",
)
@common_options
def start(
    ctx,
    filename,
    space,
    param,
    description,
    tag: tuple,
    tags: str,
    skip_validation,
    parameters_path,
    verbose_validation,
    format,
    targets: tuple[str, ...],
    skip_version_check,
    quiet: bool,
    validation_type: str,
):
    """Start a build. Specify targets to run as arguments. If no target is specified, all targets in build.yaml are executed."""
    if format == "json":
        quiet = True

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "error":
                reason = callback_args.get("reason", "")
                click.echo(
                    f"\n❌ Build can't be submitted at this moment... Reason: {reason}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case "validation":
                has_errors = callback_args.get("has_errors")
                validation_formatting(
                    callback_args,
                    verbose_validation,
                    quiet,
                    format,
                    json_to_stderr=True,
                )
                if has_errors:
                    sys.exit(1)  # Exit with a non-zero status
            case _:
                pass

    if not skip_version_check and not is_standalone():
        try:
            outdated_version = check_current_and_latest_versions()
        except Exception as e:
            click.echo(f"❌ {str(e)}.", err=True)
            ctx.exit(1)  # Exit with a non-zero status
        if outdated_version:
            click.echo(outdated_version, err=True)
            ctx.exit(1)  # Exit with a non-zero status

    if not quiet:
        click.echo(f"🏁 {PROJECT_NAME} build start")

    build_client = GBClient.Build(get_user_token())

    try:
        requested_build_url = None
        normalized_tags = []
        if bool(tag) or bool(tags):
            normalized_tags = validate_tags(
                build_client.github_token,
                tags_as_tuple=tag,
                tags_str=tags,
                callback=echo_callback,
            )
            if normalized_tags == None or len(normalized_tags) == 0:
                click.echo(f"❌ Given tag/tags are not correct", err=True)
                sys.exit(1)  # Exit with a non-zero status

        if quiet:

            requested_build_url = build_client.build_start(
                quiet,
                filename,
                space,
                param,
                skip_validation,
                parameters_path,
                targets,
                description,
                normalized_tags,
                callback=echo_callback,
                validation_type=validation_type,
            )

        else:
            with tqdm(
                total=100,
                miniters=1,
                bar_format="{desc} [{bar}] {percentage:3.0f}%",
                ascii="-#",
                leave=False,
            ) as progress_bar:

                def update_bar(callback_event: str, callback_args: Dict):
                    verbose_callback_steps = 4 if len(targets) > 0 else 3
                    if skip_validation:
                        verbose_callback_steps -= 1
                    callback_steps = verbose_callback_steps if not quiet else 2
                    steps = callback_args.get("steps", 0)
                    match callback_event:
                        case "preparing_contents":
                            if steps == 1:
                                progress_bar.reset(total=400)
                                progress_bar.set_description(
                                    f"(1/{callback_steps}) Preparing build contents."
                                )
                            progress_bar.update(n=steps)
                        case "prepared_contents":
                            progress_bar.update(n=400)
                            progress_bar.write(
                                f"(1/{callback_steps}) Prepared build contents."
                            )
                        case "skip__pr_validation":
                            progress_bar.write(
                                f"(2/{verbose_callback_steps}) Skipping build contents validation."
                            )
                        case "validating_pr":
                            if steps == 1:
                                progress_bar.reset(total=100)
                            progress_bar.set_description(
                                f"(2/{verbose_callback_steps}) Validating build contents."
                            )
                            progress_bar.update(n=steps)
                        case "validated_pr":
                            progress_bar.update(n=steps)
                            progress_bar.write(
                                f"(2/{verbose_callback_steps}) Validated build contents."
                            )
                        case "validating_targets":
                            if steps == 1:
                                progress_bar.reset(total=100)
                            progress_bar.set_description(
                                f"(3/4) Validating build targets."
                            )
                            progress_bar.update(n=steps)
                        case "validated_targets":
                            progress_bar.update(n=steps)
                            progress_bar.write(f"(3/4) Validated build targets.")
                        case "submitting_pr":
                            if steps == 1:
                                progress_bar.reset(total=100)
                            progress_bar.set_description(
                                f"({callback_steps}/{callback_steps}) Submitting build request."
                            )
                            progress_bar.update(n=steps)
                        case "submitted_pr":
                            space_org = callback_args.get("space_org", "")
                            space_name = callback_args.get("space_name", "")
                            if space_org:
                                description = (
                                    f"Submitted build to {space_org}/{space_name}."
                                )
                            else:
                                description = "Submitted build request."
                            progress_bar.update(n=steps)
                            progress_bar.write(
                                f"({callback_steps}/{callback_steps}) {description}"
                            )
                        case "clear":
                            progress_bar.clear()
                        case "warning":
                            progress_bar.clear()
                            reason = callback_args.get("reason", "")
                            if not quiet:
                                click.echo(f"\n⚠️ Warning: {reason}\n", err=True)
                        case "error":
                            reason = callback_args.get("reason", "")
                            click.echo(
                                f"\n❌ Build can't be submitted at this moment... Reason: {reason}",
                                err=True,
                            )
                            sys.exit(1)  # Exit with a non-zero status
                        case "validation":
                            has_errors = callback_args.get("has_errors")
                            if has_errors:
                                progress_bar.clear()
                            validation_formatting(
                                callback_args,
                                verbose_validation,
                                quiet,
                                format,
                                json_to_stderr=True,
                            )
                            if has_errors:
                                sys.exit(1)  # Exit with a non-zero status

                        case _:
                            pass

                requested_build_url = build_client.build_start(
                    quiet,
                    filename,
                    space,
                    param,
                    skip_validation,
                    parameters_path,
                    targets,
                    description,
                    normalized_tags,
                    callback=update_bar,
                    validation_type=validation_type,
                )

        if requested_build_url:
            show_build_id = (
                requested_build_url
                if parse_build_identifier(requested_build_url) == "uuid"
                else "BUILD_ID"
            )
            details_page = f"{DMF_URL}/gb/builds/{requested_build_url}"
            if not quiet:
                click.echo(
                    f"✅ Requested build: {details_page if show_build_id != 'BUILD_ID' else requested_build_url}"
                )
            markdown_str = f"""
{'Once the build is running, you can find the build ID of the above issue:' if show_build_id == 'BUILD_ID' else ''}
```
llmb build list | grep {requested_build_url}
llmb build list --show-all # See all your builds, including old ones.
```
To get the build status:
```
llmb build status {show_build_id}
```
To get the last 10k lines of the logs:
```
llmb build log --all {show_build_id}
```
By default this gives you the logs of the last step in the build.
To get the logs of a particular step you can use:
```
llmb build log --all {show_build_id} --build-step-id <step id>
```
        """
            if not quiet:
                click.echo(f"\n{parse_markdown_str(markdown_str)}")
            if format == "json":
                click.echo(
                    json.dumps({"build_url": details_page, "uuid": requested_build_url})
                )

    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.option("--space", help="Space name.")
@click.option(
    "--filename",
    "-f",
    type=click.Path(exists=True, readable=True, dir_okay=False),
    help="file name (and path) for build yaml",
)
@click.option(
    "--param",
    multiple=True,
    help="Build run parameter ('param_name=param_value'). Use this option again to provide multiple parameters.",
)
@click.option(
    "--parameters-path",
    type=click.Path(exists=True, readable=True, dir_okay=False),
    help="Path to parameters.yaml file.",
)
@click.option(
    "--verbose-validation",
    is_flag=True,
    default=False,
    help="Show greater detail for validation errors and warnings",
)
@click.option(
    "--format",
    "format",
    default="simple",
    type=click.Choice(["simple", "json"], case_sensitive=True),
    help="Output format: simple (default), json",
)
@click.option(
    "--validation-type",
    type=click.Choice(ALLOWED_VAL_TYPES),
    default="static",
    help=f"The type of validation to perform, one of {ALLOWED_VAL_TYPES} . Default is 'static'",
)
@click.argument("targets", nargs=-1)
@common_options
def validate(
    ctx,
    space,
    filename,
    param,
    parameters_path,
    verbose_validation,
    format,
    targets: tuple[str, ...],
    skip_version_check: bool,
    quiet: bool,
    validation_type: str,
):
    """Validate a build. Specify targets to run as arguments. If no target is specified, all targets in build.yaml are executed."""
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

    if not quiet:
        click.echo(f"🏁 {PROJECT_NAME} build validate")

    build_client = GBClient.Build(get_user_token())

    try:
        requested_build_url = None
        validation_emitted = False

        if quiet:

            def echo_callback(callback_event: str, callback_args: Dict):
                nonlocal validation_emitted
                match callback_event:
                    case "error":
                        reason = callback_args.get("reason", "")
                        click.echo(
                            f"\n❌ Build can't be validated at this moment... Reason: {reason}",
                            err=True,
                        )
                        sys.exit(1)  # Exit with a non-zero status
                    case "validation":
                        has_errors = callback_args.get("has_errors")
                        validation_formatting(
                            callback_args, verbose_validation, quiet, format
                        )
                        validation_emitted = True
                        if has_errors:
                            sys.exit(1)  # Exit with a non-zero status
                    case _:
                        pass

            result = build_client.build_validate(
                quiet,
                filename,
                space,
                param,
                parameters_path,
                targets,
                echo_callback,
                validation_type=validation_type,
            )
            requested_build_url, validate_response = result if result else (None, None)

        else:

            with tqdm(
                total=100,
                miniters=1,
                bar_format="{desc} [{bar}] {percentage:3.0f}%",
                ascii="-#",
                leave=False,
            ) as progress_bar:

                def update_bar(callback_event: str, callback_args: Dict):
                    nonlocal validation_emitted
                    verbose_callback_steps = 2 if len(targets) > 0 else 1
                    steps = callback_args.get("steps", 0)
                    match callback_event:

                        case "validating_pr":
                            if steps == 1:
                                progress_bar.reset(total=100)
                            progress_bar.set_description(
                                f"(1/{verbose_callback_steps}) Validating build contents."
                            )
                            progress_bar.update(n=steps)
                        case "validated_pr":
                            progress_bar.update(n=steps)
                            progress_bar.write(
                                f"(1/{verbose_callback_steps}) Validated build contents."
                            )
                        case "validating_targets":
                            if steps == 1:
                                progress_bar.reset(total=100)
                            progress_bar.set_description(
                                f"(2/2) Validating build targets."
                            )
                            progress_bar.update(n=steps)
                        case "validated_targets":
                            progress_bar.update(n=steps)
                            progress_bar.write(f"(1/2) Validated build targets.")

                        case "warning":
                            progress_bar.clear()
                            reason = callback_args.get("reason", "")
                            if not quiet:
                                click.echo(f"\n⚠️ Warning: {reason}\n", err=True)
                        case "error":
                            reason = callback_args.get("reason", "")
                            click.echo(
                                f"\n❌ Build can't be validated at this moment... Reason: {reason}",
                                err=True,
                            )
                            sys.exit(1)  # Exit with a non-zero status
                        case "validation":
                            has_errors = callback_args.get("has_errors")
                            if has_errors:
                                progress_bar.clear()
                            validation_formatting(
                                callback_args, verbose_validation, quiet, format
                            )
                            validation_emitted = True
                            if has_errors:
                                sys.exit(1)  # Exit with a non-zero status

                        case _:
                            pass

                result = build_client.build_validate(
                    quiet,
                    filename,
                    space,
                    param,
                    parameters_path,
                    targets,
                    update_bar,
                    validation_type=validation_type,
                )
                requested_build_url, validate_response = (
                    result if result else (None, None)
                )

        if requested_build_url:
            if not quiet:
                click.echo(f"✅ Build contents have been succesfully validated")
            if format == "json" and not validation_emitted:
                click.echo(
                    json.dumps({"validated": True, "errors": [], "warnings": []})
                )

    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.option("--space", help="Space name.")
@click.argument("build_id", required=True)
@click.option(
    "--format",
    "format",
    default="simple",
    type=click.Choice(["simple", "json"], case_sensitive=True),
    help="Output format: simple (default), json",
)
@common_options
def cancel(ctx, space, build_id, format, skip_version_check, quiet):
    """
    Cancel a build

    Provide build ID or URL
    """
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

    if not quiet:
        click.echo(f"❕ {PROJECT_NAME} build cancel")

    build_client = GBClient.Build(get_user_token())

    try:
        build = None

        if quiet:
            build = build_client.build_cancel(build_id, id_format, space, None)
        else:
            with tqdm(
                total=100,
                miniters=1,
                bar_format="{desc} [{bar}] {percentage:3.0f}%",
                ascii="-#",
                leave=False,
            ) as progress_bar:

                def update_bar(callback_event: str, callback_args: Dict):
                    total_steps = 3 if id_format == "url" else 2
                    steps = callback_args.get("steps", 0)
                    match callback_event:
                        case "fetching_build_id":
                            if steps == 1:
                                progress_bar.reset(total=100)
                            source_uri = callback_args.get("source_uri", "")
                            progress_bar.set_description(
                                f"(1/{total_steps}) Obtaining build ID for build URL {source_uri}."
                            )
                            progress_bar.update(n=steps)
                        case "fetched_build_id":
                            source_uri = callback_args.get("source_uri", "")
                            build_id = callback_args.get("build_id", "")
                            progress_bar.update(n=steps)
                            progress_bar.write(
                                f"(1/{total_steps}) Obtained build ID {build_id} for build URL {source_uri}."
                            )
                        case "obtaining_build":
                            if steps == 1:
                                progress_bar.reset(total=100)
                            build_id = callback_args.get("build_id", "")
                            progress_bar.set_description(
                                f"({total_steps - 1}/{total_steps}) Obtaining build {build_id}."
                            )
                            progress_bar.update(n=steps)
                        case "obtained_build":
                            progress_bar.update(n=steps)
                            build_id = callback_args.get("build_id", "")
                            progress_bar.write(
                                f"({total_steps - 1}/{total_steps}) Obtained build {build_id}."
                            )
                        case "cancelling_build":
                            if steps == 1:
                                progress_bar.reset(total=100)
                            build_id = callback_args.get("build_id", "")
                            progress_bar.set_description(
                                f"({total_steps}/{total_steps}) Cancelling build {build_id}."
                            )
                            progress_bar.update(n=steps)
                        case "canceled_build":
                            build_id = callback_args.get("build_id", "")
                            progress_bar.update(n=steps)
                            progress_bar.write(
                                f"({total_steps}/{total_steps}) Requested cancellation of build {build_id}."
                            )
                        case "warning":
                            progress_bar.clear()
                            reason = callback_args.get("reason", "")
                            if not quiet:
                                click.echo(f"\n⚠️ Warning: {reason}\n", err=True)
                        case "error":
                            progress_bar.clear()
                            reason = callback_args.get("reason", "")
                            click.echo(
                                f"\n❌ Build can't be canceled at this moment... Reason: {reason}",
                                err=True,
                            )
                            sys.exit(1)  # Exit with a non-zero status
                        case _:
                            pass

                build = build_client.build_cancel(
                    build_id, id_format, space, update_bar
                )

        if build:
            if not quiet:
                click.echo(
                    f"✅ Build {build['canceled']['uuid']} requested to be canceled."
                )
            if format == "json":
                click.echo(json.dumps({"uuid": build["canceled"]["uuid"]}))

    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.argument("build_id", required=True)
@click.option(
    "--format",
    default="simple",
    type=click.Choice(["simple", "full", "json"], case_sensitive=True),
    help="Lineage contents format: simple (default), full, json",
)
@click.option(
    "--lakehouse",
    is_flag=True,
    default=False,
    help="Fetch lineage from Lakehouse.",
)
@common_options
def lineage(ctx, build_id, format, lakehouse, skip_version_check, quiet):
    """
    Access build lineage

    Provide build ID or URL
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

    if format == "json":
        quiet = True

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

    total_steps = 2 if lakehouse else 1
    if id_format == "url":
        total_steps += 1

    erase_sequence = "\r\033[K"

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "fetching_build_id" if not quiet:
                current_step = 2 if lakehouse else 1
                source_uri = callback_args.get("source_uri", "")
                click.echo(
                    f"({current_step}/{total_steps}) Obtaining build ID for build URL {source_uri}.",
                    nl=False,
                )
            case "fetched_build_id" if not quiet:
                current_step = 2 if lakehouse else 1
                source_uri = callback_args.get("source_uri", "")
                build_id = callback_args.get("build_id", "")
                click.echo(
                    f"{erase_sequence}({current_step}/{total_steps}) Obtained build ID {build_id} for build URL {source_uri}."
                )
            case "fetching_build_lineage_lh" if not quiet:
                current_step = 3 if id_format == "url" else 2
                click.echo(
                    f"({current_step}/{total_steps}) Fetching build lineage from Lakehouse. This may take a while, please wait..."
                )
            case "fetching_build_lineage_gbserver" if not quiet:
                click.echo(
                    f"{'(2/2) ' if id_format == 'url' else ''}Fetching build lineage. This may take a while, please wait..."
                )
            case "build_lineage_spinner" if not quiet:
                spinner = callback_args.get("spinner", "")
                click.echo(
                    f"\rProcessing... {spinner}",
                    nl=False,
                )
            case "warning" if not quiet:
                reason = callback_args.get("reason", "")
                click.echo(f"\n⚠️ Warning: {reason}\n", err=True)
            case "error":
                reason = callback_args.get("reason", "")
                click.echo(
                    f"\n❌ Build lineage can't be retrieved at this moment... Reason: {reason}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case _:
                pass  # Ignore unknown events

    try:
        build_client = GBClient.Build(get_user_token())

        if lakehouse:
            if not quiet:
                click.echo(f"(1/{total_steps}) Obtaining Lakehouse token.")

            token = GBClient.Auth.lakehouse_user_token(
                build_client.github_token, callback=echo_callback
            )

            if not token:
                return
            if not quiet:
                click.echo(f"Lakehouse token obtained successfully!.")

            status, lineage_dict = build_client.build_lineage_lh(
                token, build_id, id_format, echo_callback
            )
        else:
            status, lineage_dict = build_client.build_lineage(
                build_id, id_format, echo_callback
            )

        if status in ["pending", "running", "submitted"]:
            click.echo(
                "\nNote: This build is still in progress. The lineage information may not be complete until it finishes."
            )

        if len(lineage_dict) == 0:
            if not quiet:
                click.echo("\nNo lineage found.")
            return

        if format == "json":
            click.echo("\n" + json.dumps(lineage_dict, indent=2, default=str))

        else:
            if format == "full":
                lineage = [
                    [
                        l["release_id"],
                        l["category"],
                        l["job_name"],
                        l["job_id"],
                        l["job_type"],
                        l["job_started_at"],
                        l["job_completed_at"],
                        l["job_status"],
                        l["owner"],
                        l["source"],
                        l["source_type"],
                        l["source_object"],
                        l["target"],
                        l["target_type"],
                        l["target_object"],
                        l["source_code_details"],
                        l["job_input_params"],
                        l["execution_stats"],
                        l["job_output_stats"],
                    ]
                    for l in lineage_dict
                ]

                lineage_table = tabulate(
                    lineage,
                    BUILD_LINEAGE_FULL_HEADERS,
                    tablefmt="plain",
                )
            elif format == "simple":
                lineage = [
                    [
                        l["release_id"],
                        l["job_name"],
                        l["job_id"],
                        l["job_status"],
                        l["source_type"].capitalize() + ": " + l["source"],
                        l["target_type"].capitalize() + ": " + l["target"],
                    ]
                    for l in lineage_dict
                ]

                lineage_table = tabulate(
                    lineage,
                    BUILD_LINEAGE_DEFAULT_HEADERS,
                    tablefmt="plain",
                )

            click.echo("\n\n" + lineage_table)

        build_lineage_url = (
            get_build_lineage_url(build_id)
            if id_format == "uuid"
            else get_build_lineage_url(lineage_dict[0]["release_id"])
        )

        if not quiet:
            click.echo(f"\nView the lineage visually at: {build_lineage_url}")
            if check_runnable_browser():
                answer = click.confirm("Open the browser?", False)

                if answer:
                    webbrowser.open(build_lineage_url)

    except AuthException as e:
        click.echo(f"❌ {str(e)}", err=True)
        ctx.exit(1)  # Exit with a non-zero status
    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.option("--space", help="Space name.")
@click.option(
    "--all-spaces", is_flag=True, default=False, help="Builds from all spaces."
)
@click.option(
    "--show-done",
    flag_value=True,
    default=False,
    help=f"List builds including the ones that finished.",
)
@click.option(
    "--show-all",
    flag_value=True,
    default=False,
    help=f"Equivalent to --show-done.",
)
@click.option(
    "--username",
    "-u",
    is_flag=False,
    flag_value="default",
    help="Filter builds by users. Provide a username. If value is omitted, your username will be used.",
)
@click.option(
    "--tag",
    multiple=True,
    help="Single tag name or comma-separated list of tag names. Tag names can only contain alphanumeric characters, underscores, and hyphens.",
)
@click.option(
    "--tags", help="Comma-separated list of tag names. For example: 'tagA, tagB'."
)
@click.option(
    "--page-size",
    "-n",
    type=int,
    default=10,
    help="Number of builds to display per page (default: 10)",
)
@click.option(
    "--page-index",
    type=int,
    default=0,
    help="Page number to display, starts at 0 (default: 0)",
)
@click.option(
    "--all", "-a", flag_value=True, default=False, help=f"List build for all users."
)
@click.option(
    "--format",
    default="plain",
    type=click.Choice(["plain", "json"], case_sensitive=True),
    help=f"Output format: plain (default), json",
)
@click.option(
    "--wide",
    "-w",
    is_flag=True,
    default=False,
    help="Display brief descriptions of artifacts. Run 'artifact describe' for the full details.",
)
@common_options
def list(
    ctx,
    space,
    all_spaces,
    all,
    show_done,
    show_all,
    username,
    tag: tuple | None,
    tags: str | None,
    page_index,
    page_size,
    format,
    wide,
    skip_version_check,
    quiet,
):
    """List builds"""
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

    if not quiet:
        click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} build list")

    if all and username:
        click.echo(
            f"❌ Error: --all was provided. It can't be used with the --username option."
        )
        ctx.exit(1)

    if all_spaces and space:
        click.echo(
            f"❌ Error: --space and --all-spaces cannot be used together",
            err=True,
        )
        sys.exit(1)

    normalized_tags = []
    if bool(tag) or bool(tags):
        normalized_tags = combine_tags(tags_str=tags, tags_tuple=tag)
        if normalized_tags == None or len(normalized_tags) == 0:
            click.echo(f"❌ Given tag/tags are not correct", err=True)
            sys.exit(1)  # Exit with a non-zero status

    if username == "default":
        username = None

    build_client = GBClient.Build(get_user_token())

    try:
        builds = None
        count = 0

        if quiet:
            builds_data = build_client.build_list(
                all,
                show_done,
                show_all,
                all_spaces,
                space,
                username,
                tags=normalized_tags,
                page_index=page_index,
                page_size=page_size,
                callback=None,
            )
            if builds_data:
                builds = builds_data["items"]
                count = builds_data["count"]
        else:
            with tqdm(
                total=100,
                miniters=1,
                desc=f"📝 Listing builds",
                bar_format="{desc} [{bar}] {percentage:3.0f}%",
                ascii="-#",
                leave=False,
            ) as progress_bar:

                def update_bar(callback_event: str, callback_args: Dict):
                    steps = callback_args.get("steps", 0)
                    match callback_event:
                        case "listing_builds":
                            if steps == 1:
                                progress_bar.reset(total=100)
                            used_all_spaces = callback_args.get(
                                "used_all_spaces", False
                            )
                            space = callback_args.get("space", "")
                            space_name = callback_args.get("space_name", "")
                            if used_all_spaces:
                                listing_builds_description = (
                                    "📝 (1/2) Obtaining list of builds for all spaces."
                                )
                            else:
                                listing_builds_description = f'📝 (1/2) Obtaining list of builds for space "{space}" ({space_name}).'
                            progress_bar.set_description(listing_builds_description)
                            progress_bar.update(n=steps)
                        case "listed_builds":
                            progress_bar.update(n=steps)
                            used_all_spaces = callback_args.get(
                                "used_all_spaces", False
                            )
                            space = callback_args.get("space", "")
                            space_name = callback_args.get("space_name", "")
                            if used_all_spaces:
                                output_listed_builds = (
                                    "📝 (1/2) List of builds obtained for all spaces."
                                )
                            else:
                                output_listed_builds = f'📝 (1/2) List of builds obtained for space "{space}" ({space_name}).'
                            progress_bar.write(output_listed_builds)
                        case "processing_builds":
                            if steps == 1:
                                progress_bar.reset(total=100)
                            used_all_spaces = callback_args.get(
                                "used_all_spaces", False
                            )
                            space = callback_args.get("space", "")
                            space_name = callback_args.get("space_name", "")
                            if used_all_spaces:
                                output_processing_builds = (
                                    "📝 (2/2) Processing builds from all spaces."
                                )
                            else:
                                output_processing_builds = f'📝 (2/2) Processing builds from space "{space}" ({space_name}).'
                            progress_bar.set_description(output_processing_builds)
                            progress_bar.update(n=steps)
                        case "processed_builds":
                            used_all_spaces = callback_args.get(
                                "used_all_spaces", False
                            )
                            space = callback_args.get("space", "")
                            space_name = callback_args.get("space_name", "")
                            progress_bar.update(n=steps)
                            if used_all_spaces:
                                output_processed_builds = (
                                    "📝 (2/2) Processed builds from all spaces."
                                )
                            else:
                                output_processed_builds = f'📝 (2/2) Processed builds from space "{space}" ({space_name}).'
                            progress_bar.write(output_processed_builds)
                        case "listing_prs":
                            if steps == 1:
                                progress_bar.reset(total=100)
                            used_all_spaces = callback_args.get(
                                "used_all_spaces", False
                            )
                            space_name = callback_args.get("space_name", "")
                            progress_bar.set_description(
                                f"📝 (2/2) Obtaining additional information from {'all spaces' if used_all_spaces else space_name}."
                            )
                            progress_bar.update(n=steps)
                        case "listed_prs":
                            used_all_spaces = callback_args.get(
                                "used_all_spaces", False
                            )
                            space_name = callback_args.get("space_name", "")
                            progress_bar.update(n=steps)
                            progress_bar.write(
                                f"📝 (2/2) Additional information from {'all spaces' if used_all_spaces else space_name} obtained:"
                            )
                        case "warning":
                            progress_bar.clear()
                            reason = callback_args.get("reason", "")
                            if not quiet:
                                click.echo(f"\n⚠️{' '}Warning: {reason}\n", err=True)
                            sys.exit(1)  # Exit with a non-zero status
                        case "all_spaces_warning":
                            progress_bar.clear()
                            click.echo(
                                f"\n⚠️{' '}Warning: The --all-spaces option can only be used with the new version of build start. The default space will be used.\n"
                            )
                        case "error":
                            reason = callback_args.get("reason", "")
                            click.echo(
                                f"\n❌ Builds can't be retrieved at this moment... Reason: {reason}",
                                err=True,
                            )
                            sys.exit(1)  # Exit with a non-zero status
                        case _:
                            pass  # Ignore unknown events

                builds_data = build_client.build_list(
                    all,
                    show_done,
                    show_all,
                    all_spaces,
                    space,
                    username,
                    tags=normalized_tags,
                    page_index=page_index,
                    page_size=page_size,
                    callback=update_bar,
                )
                if builds_data:
                    builds = builds_data["items"]
                    count = builds_data["count"]

        if builds:
            if format == "plain":
                headers = BUILD_LIST_HEADERS

                if not (all or username):
                    headers.remove("USER")

                if all_spaces:
                    headers.insert(2, "SPACE_NAME")

                    # need to filter by user spaces (will be in gbserver eventually)
                    spaces = [
                        s["name"]
                        for s in GBClient.Space(get_user_token()).list_spaces(
                            all, False, None
                        )
                    ]
                    builds = [b for b in builds if b["space_name"] in spaces]
                if wide:
                    name_index = headers.index("NAME")
                    headers.insert(name_index + 1, "DESCRIPTION")

                prs_table = [
                    [p["build_id"], p["name"]]
                    + ([p["description"]] if (wide) else [])
                    + ([p["user"]] if (all or username) else [])
                    + ([p["space_name"]] if (all_spaces) else [])
                    + ([p["tags"]])
                    + [
                        p["status"],
                        humanize_iso_date(p["start_time"]),
                    ]
                    for p in builds
                ]
                builds_output = tabulate(prs_table, headers, tablefmt="plain")
            else:
                builds_output = json.dumps(builds)

            click.echo(builds_output)
            if format != "json" and page_index >= 0 and page_size > 0:
                start, end = pagination_range(
                    total_items=count,
                    page_index=page_index,
                    page_size=page_size,
                )
                last_page_index = max((count - 1) // page_size, 0)
                pagination = f"\nShowing {start}-{end} of {count} results | Page {page_index}/{last_page_index}"
                click.echo(pagination)

    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.argument("build_id", required=True)
@click.option(
    "--format",
    default="plain",
    type=click.Choice(["plain", "json"], case_sensitive=True),
    help=f"Output format: plain (default), json",
)
@click.option(
    "--show-events",
    is_flag=True,
    default=False,
    help="Show all build events.",
)
@click.option(
    "--fetch-pr",
    is_flag=True,
    default=False,
    help="Fetch build PR events.",
)
@common_options
def status(ctx, build_id, format, show_events, fetch_pr, skip_version_check, quiet):
    """
    Get status of a build execution

    Provide build ID or URL
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

    if format == "json":
        quiet = True

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

    if not quiet:
        click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} build status")

    build_client = GBClient.Build(get_user_token())

    def echo_callback_error(callback_event: str, callback_args: Dict):
        match callback_event:
            case "error":
                reason = callback_args.get("reason", "")
                click.echo(
                    f"\n❌ Execution status can't be retrieved at this moment... Reason: {reason}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case _:
                pass  # Ignore unknown events

    try:
        details = None

        if quiet:
            details, targets, history, error = build_client.build_status(
                build_id,
                quiet,
                id_format,
                show_events,
                fetch_pr,
                format,
                callback=echo_callback_error,
            )
        else:
            with tqdm(
                total=100,
                miniters=1,
                desc=f"📝 Fetching status",
                bar_format="{desc} [{bar}] {percentage:3.0f}%",
                ascii="-#",
                leave=False,
            ) as progress_bar:

                def update_bar(callback_event: str, callback_args: Dict):
                    total_steps = 3 if not quiet else 2
                    callback_steps = (
                        (total_steps + 1) if id_format == "url" else total_steps
                    )
                    steps = callback_args.get("steps", 0)
                    match callback_event:
                        case "fetching_build_id":
                            if steps == 1:
                                progress_bar.reset(total=100)
                            source_uri = callback_args.get("source_uri", "")
                            progress_bar.set_description(
                                f"📝 (1/{callback_steps}) Obtaining build ID for build URL {source_uri}."
                            )
                            progress_bar.update(n=steps)
                        case "fetched_build_id":
                            source_uri = callback_args.get("source_uri", "")
                            build_id = callback_args.get("build_id", "")
                            progress_bar.update(n=steps)
                            progress_bar.write(
                                f"📝 (1/{callback_steps}) Obtained build ID {build_id} for build URL {source_uri}."
                            )
                        case "fetching_build_status":
                            current_step = 2 if id_format == "url" else 1
                            progress_bar.clear()
                            spinner = callback_args.get("spinner", "")
                            click.echo(
                                f"📝 ({current_step}/{callback_steps}) Fetching execution status... {spinner}",
                                nl=False,
                            )
                        case "fetched_build_status":
                            current_step = 2 if id_format == "url" else 1
                            searched_build = callback_args.get("build_id", "")
                            click.echo(
                                f"\r📝 ({current_step}/{callback_steps}) Status obtained for build {searched_build}."
                            )
                        case "processing_status_artifacts":
                            current_step = 3 if id_format == "url" else 2
                            if steps == 1:
                                progress_bar.reset(total=100)
                            progress_bar.set_description(
                                f"📝 ({current_step}/{callback_steps}) Processing status artifacts..."
                            )
                            progress_bar.update(n=steps)
                        case "processed_status_artifacts":
                            current_step = 3 if id_format == "url" else 2
                            progress_bar.update(n=steps)
                            progress_bar.write(
                                f"📝 ({current_step}/{callback_steps}) Processed status artifacts."
                            )
                        case "fetching_additional_info":
                            if steps == 1:
                                progress_bar.reset(total=100)
                            progress_bar.set_description(
                                f"📝 ({callback_steps}/{callback_steps}) Fetching additional information..."
                            )
                            progress_bar.update(n=steps)
                        case "fetched_additional_info":
                            progress_bar.update(n=steps)
                            progress_bar.write(
                                f"📝 ({callback_steps}/{callback_steps}) Obtained additional information."
                            )
                        case "skip_additional_info":
                            progress_bar.write(
                                f"📝 ({callback_steps}/{callback_steps}) Skipping additional information..."
                            )
                        case "warning":
                            progress_bar.clear()
                            reason = callback_args.get("reason", "")
                            if not quiet:
                                click.echo(f"\n⚠️ Warning: {reason}\n", err=True)
                        case "error":
                            reason = callback_args.get("reason", "")
                            click.echo(
                                f"\n❌ Execution status can't be retrieved at this moment... Reason: {reason}",
                                err=True,
                            )
                            sys.exit(1)  # Exit with a non-zero status
                        case _:
                            pass  # Ignore unknown events

                details, targets, history, error = build_client.build_status(
                    build_id,
                    quiet,
                    id_format,
                    show_events,
                    fetch_pr,
                    format,
                    callback=update_bar,
                )

        if details:
            if format == "plain":
                status = execution_status_plain_output(
                    details, targets, history, show_events
                )
            else:
                if error:
                    builds_output = {"error": error}
                else:
                    builds_output = {
                        "details": details,
                        "targets": targets,
                        "build_history": history,
                    }
                status = json.dumps(builds_output)

            click.echo(status)
        elif error:
            click.echo(json.dumps({"error": error}))
    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.argument("build_id", required=True)
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
@click.option(
    "--runner",
    is_flag=True,
    default=False,
    help=f"Get logs from {PROJECT_NAME} server build runner",
)
@click.option(
    "--skip-id-check",
    is_flag=True,
    default=False,
    help="Run build query without build ID existence checks. We may remove this option in the future.",
)
@common_options
def log(
    ctx,
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
    runner,
    skip_id_check,
    skip_version_check,
    quiet,
):
    """
    Get logs of a build execution

    Provide build ID or URL
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
    if all and (end_date or head or tail or format == "json"):
        click.echo(
            f"❌ Error: --all was provided. It can't be used with --head, --tail or '--format json' options.",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status

    if format == "json":
        quiet = True

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
            f"⚠️  Warning: the maximum log time range is {BUILD_LOG_MAX_QUERY_RANGE} days. Automatically setting the start date."
        )
    if start_date and not end_date and not follow and not all:
        end_epoch = change_timestamp_by_days(
            start_epoch, BUILD_LOG_MAX_QUERY_RANGE, True
        )
        click.echo(
            f"⚠️  Warning: the maximum log time range is {BUILD_LOG_MAX_QUERY_RANGE} days. Automatically setting the end date."
        )

    if (
        round((current_epoch - start_epoch) / BUILD_LOG_SECONDS_IN_A_DAY)
        > BUILD_LOG_MAX_LOG_LIFESPAN
    ):
        click.echo(
            f"⚠️  Warning: the log service only keeps the logs up to {BUILD_LOG_MAX_LOG_LIFESPAN} days. start_date is beyond that."
        )
    if (
        end_epoch != None
        and round((current_epoch - end_epoch) / BUILD_LOG_SECONDS_IN_A_DAY)
        > BUILD_LOG_MAX_LOG_LIFESPAN
    ):
        click.echo(
            f"⚠️  Warning: the log service only keeps the logs up to {BUILD_LOG_MAX_LOG_LIFESPAN} days. end_date is beyond that."
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

    if not quiet:
        click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} build log")

    build_client = GBClient.Build(get_user_token())

    if head != None:
        sort = "asc"
        page_size = head

    if tail != None:
        sort = "desc"
        page_size = tail

    erase_sequence = "\r\033[K"

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "fetching_build_id" if not quiet:
                click.echo(
                    f"Obtaining build ID for build URL {callback_args.get('source_uri', '')}.",
                    nl=False,
                )
            case "fetched_build_id" if not quiet:
                click.echo(
                    f"{erase_sequence}Obtained build ID {callback_args.get('build_id', '')} for build URL {callback_args.get('source_uri', '')}."
                )
            case "querying_log" if not quiet:
                click.echo(
                    f"Querying the logs between {epoch_to_iso_date(callback_args.get('start_epoch', ''))} and {epoch_to_iso_date(callback_args.get('end_epoch', ''))}\nQuerying log server..\n"
                )
            case "display_logs":
                logs = callback_args.get("logs", [])
                output_format_plain(logs)
            case "warning" if not quiet:
                reason = callback_args.get("reason", "")
                click.echo(f"\n⚠️  Warning: {reason}\n", err=True)
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
            logger.debug(f"Log ID: {log['logId']}.")
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
        logs = build_client.build_log(
            id_format,
            start_epoch,
            end_epoch if end_epoch else current_epoch,
            page_size,
            page_index,
            stream,
            text,
            sort,
            build_id,
            build_step_id,
            build_step_name,
            runner,
            follow,
            all,
            skip_id_check,
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

            if not quiet:
                click.echo(f"\n✅ Total number of logs returned: {len(logs)}")

    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status
    except KeyboardInterrupt:
        ctx.exit(0)


# @cli.command()
# @click.pass_context
# @click.argument("build_id", required=True)
# def kill(ctx, build_id):
#     """Send stop command to the build to back-end"""
#     click.echo("This command is not yet available.")
#     ctx.exit(1)


# @cli.command()
# @click.pass_context
# @click.argument("build_id", required=True)
# def freeze(ctx, build_id):
#     """Freeze artifacts used for build"""
#     click.echo("This command is not yet available.")
#     ctx.exit(1)


@cli.command()
@click.pass_context
@click.argument("build_id", required=False)
@click.option(
    "--filename",
    "-f",
    type=click.Path(exists=True, readable=True, dir_okay=False),
    help="file name (and path) for build yaml",
)
@click.option(
    "--format",
    default="simple",
    type=click.Choice(["simple", "full", "json"], case_sensitive=True),
    help=f"Output format: simple (default), full, json",
)
@click.option("--space", help="Space name.")
@click.option(
    "--raw",
    is_flag=True,
    default=False,
    help="Output build file contents",
)
@common_options
def describe(ctx, build_id, filename, format, space, raw, skip_version_check, quiet):
    """Describe targets steps, description of a build"""
    if not skip_version_check:
        try:
            outdated_version = check_current_and_latest_versions()
        except Exception as e:
            click.echo(f"❌ {str(e)}.", err=True)
            ctx.exit(1)  # Exit with a non-zero status
        if outdated_version:
            click.echo(outdated_version, err=True)
            ctx.exit(1)  # Exit with a non-zero status

    quiet = True if format == "json" else quiet

    if not quiet:
        click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} build describe")

    if space is not None and build_id is None:
        if not quiet:
            click.echo(
                f"Warning: Space '{space}' is ignored when a local build file is specified.",
                err=True,
            )

    if filename and build_id:
        click.echo(
            f"❌ build describe can not be run with both a build_id and a filename. Please select one of the two options",
            err=True,
        )
        sys.exit(1)  # Exit with a non-zero status

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

    build_client = GBClient.Build(get_user_token())

    def echo_callback(callback_event: str, callback_args: Dict):
        if callback_event == "error":
            reason = callback_args.get("reason", "")
            click.echo(
                f"\n❌ Build description can't be retrieved at this moment... Reason: {reason}",
                err=True,
            )
            sys.exit(1)  # Exit with a non-zero status
        elif callback_event == "warning":
            reason = callback_args.get("reason", "")
            click.echo(f"\n⚠️  Warning: {reason}\n")
        else:
            pass  # Ignore unknown events

    def parse_config_output(config: dict) -> str:
        config_str = ""
        for key in config.keys():
            if isinstance(config[key], dict):
                for inner_key in config[key].keys():
                    config_str = (
                        config_str + f"{key}.{inner_key}: {config[key][inner_key]}\n"
                    )
            else:
                config_str = config_str + f"{key}: {config[key]}\n"
            if "echo" in config_str:
                config_str = config_str.replace(";", ";\n")
        return config_str.replace("\\", "")

    try:
        if not raw:
            targets, build = build_client.build_describe(
                filename,
                format,
                raw,
                build_id,
                id_format,
                space,
                callback=echo_callback,
            )
            if build and build["build"]:
                details = build["build"]
                markdown_output = f"""
# Build {details['uuid']}
- **Build name**: {details['name']}
- **Build description**: {details['description']}
- **Tags**: {"" if not details.get('tags', '') else details.get('tags', '')}
---
"""

                if not quiet:
                    click.echo(custom_parse_markdown_str(markdown_output))

            if len(targets) > 0:
                if format != "json":
                    steps_header = BUILD_DESCRIBE_STEPS_HEADERS
                    if format == "simple":
                        steps_header.remove("CONFIG")
                    for target in targets:
                        description_output = (
                            f"{target['target_name']}\n"
                            + f"{target['environment_uri']}\n"
                        )

                        input_artifacts_table = [
                            [i["name"], i["uri"]] for i in target["inputs"]
                        ]

                        output_artifacts_table = [
                            [o["name"], o["uri"]] for o in target["outputs"]
                        ]

                        output_steps_table = [
                            [s["uri"]]
                            + (
                                [parse_config_output(s["config"])]
                                if format == "full"
                                else []
                            )
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
                            description_output
                            + (
                                (
                                    "\n*️⃣  Input artifacts\n"
                                    + input_artifacts_output
                                    + "\n"
                                )
                                if len(target["inputs"]) > 0
                                else ""
                            )
                            + (
                                (
                                    "\n*️⃣  Output artifacts\n"
                                    + output_artifacts_output
                                    + "\n"
                                )
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
                    click.echo(json.dumps(targets))

        else:
            yaml_str = build_client.build_describe(
                filename,
                format,
                raw,
                build_id,
                id_format,
                space,
                callback=echo_callback,
            )
            click.echo(yaml_str)
    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.argument("build_id_1", required=True)
@click.argument("build_id_2", required=False)
@click.option("--space", help="Space name.")
@click.option(
    "--format",
    default="simple",
    type=click.Choice(["simple", "json"], case_sensitive=True),
    help=f"Output format: simple (default), json",
)
@click.pass_context
@common_options
def diff(ctx, build_id_1, build_id_2, space, format, skip_version_check, quiet):
    """
    Show changes between builds

    Provide build ID or URL
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

    if format == "json":
        quiet = True

    id_format_1 = parse_build_identifier(build_id_1)
    id_format_2 = parse_build_identifier(build_id_2)
    if id_format_1 not in ["uuid", "url", "filename"] or (
        build_id_2 and id_format_2 not in ["uuid", "url", "filename"]
    ):
        click.echo(
            f"❌ Build identifier formatted incorrectly. Please try again with valid build ID, URL or file path.",
            err=True,
        )
        sys.exit(1)

    if (id_format_1 == "uuid" and len(build_id_1) < 36) or (
        id_format_2 == "uuid" and len(build_id_2) < 36
    ):
        click.echo(
            f"❌ Build ID formatted incorrectly. Please try again with a valid build ID.",
            err=True,
        )
        sys.exit(1)

    if not quiet:
        click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} build diff")

    build_client = GBClient.Build(get_user_token())

    erase_sequence = "\r\033[K"

    url_steps = (
        2
        if (id_format_1 == "url" and id_format_2 == "url")
        else 1 if (id_format_1 == "url" or id_format_2 == "url") else 0
    )
    callback_steps = (
        (2 + url_steps)
        if (build_id_2 and id_format_2 != "filename" and id_format_1 != "filename")
        else (1 + url_steps)
    )

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "fetching_build_id" if not quiet:
                source_uri = callback_args.get("source_uri", "")
                if (source_uri == build_id_2) or (
                    source_uri == build_id_1 and id_format_2 == "filename"
                ):
                    current_step = f"(1/{callback_steps})"
                else:
                    current_step = (
                        f"(1/{callback_steps})"
                        if not build_id_2
                        else (
                            f"(3/{callback_steps})"
                            if id_format_2 == "url"
                            else f"(2/{callback_steps})"
                        )
                    )
                click.echo(
                    f"📝 {current_step} Obtaining build ID for build {source_uri}...",
                    nl=False,
                )
            case "fetched_build_id" if not quiet:
                source_uri = callback_args.get("source_uri", "")
                if (source_uri == build_id_2) or (
                    source_uri == build_id_1 and id_format_2 == "filename"
                ):
                    current_step = f"(1/{callback_steps})"
                else:
                    current_step = (
                        f"(1/{callback_steps})"
                        if not build_id_2
                        else (
                            f"(3/{callback_steps})"
                            if id_format_2 == "url"
                            else f"(2/{callback_steps})"
                        )
                    )
                click.echo(
                    f"{erase_sequence}📝 {current_step} Obtained build ID for build {source_uri}."
                )
            case "break_line" if not quiet:
                click.echo("")
            case "fetching_build_file" if not quiet:
                spinner = callback_args.get("spinner", "")
                build_id = callback_args.get("build_id", "")
                if build_id == build_id_2:
                    current_step = (
                        f"(2/{callback_steps})"
                        if id_format_2 == "url"
                        else f"(1/{callback_steps})"
                    )
                else:
                    current_step = f"({callback_steps}/{callback_steps})"
                click.echo(
                    f"\r📝 {current_step} Fetching build file for {build_id}... {spinner}",
                    nl=False,
                )
            case "warning" if not quiet:
                reason = callback_args.get("reason", "")
                if not quiet:
                    click.echo(f"\n⚠️ Warning: {reason}\n", err=True)
            case "error":
                reason = callback_args.get("reason", "")
                click.echo(
                    f"\n❌ Build diff can't be produced at this moment... Reason: {reason}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case _:
                pass

    try:
        build_filename_1, build_filename_2, diff = build_client.build_diff(
            build_id_1,
            id_format_1,
            build_id_2,
            id_format_2,
            space,
            callback=echo_callback,
        )
        if format == "simple":
            if len(diff) > 0:
                no_callback = (
                    True
                    if (
                        quiet
                        or (not build_id_2 and id_format_1 == "filename")
                        or (id_format_1 == "filename" and id_format_2 == "filename")
                    )
                    else False
                )
                click.echo(f"{diff[0]}" if no_callback else f"\n{diff[0]}")
                click.echo(f"{diff[1]}")
                for line in diff[2:-1]:
                    if "\n" in line:
                        click.echo(line, nl=False)
                    else:
                        click.echo(line)
                        if not "@@" in line:
                            click.echo("\\ No newline at end of file")
                click.echo(diff[-1])
                if "\n" not in diff[-1]:
                    click.echo("\\ No newline at end of file")
            else:
                click.echo("\nNo changes found between builds.")
        else:
            if len(diff) > 0:
                added = [line[1:] for line in diff[2:] if line[0] == "+"]
                removed = [line[1:] for line in diff[2:] if line[0] == "-"]
                output_json = {
                    "build_filename_1": build_filename_1,
                    "build_filename_2": build_filename_2,
                    "added": added,
                    "removed": removed,
                }
            else:
                output_json = {
                    "build_filename_1": build_filename_1,
                    "build_filename_2": build_filename_2,
                    "added": [],
                    "removed": [],
                }
            click.echo(output_json)
    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.argument("build_id", required=True)
@click.option(
    "--show-events",
    is_flag=True,
    default=False,
    help="Show all build events.",
)
@click.option(
    "--fetch-pr",
    is_flag=True,
    default=False,
    help="Fetch build PR events.",
)
@common_options
def monitor(ctx, build_id, show_events, fetch_pr, skip_version_check, quiet):
    """
    Monitor a build execution

    Provide build ID or URL
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

    click.echo(f"🏁 {PROJECT_NAME} build monitor")

    build_client = GBClient.Build(get_user_token())

    erase_sequence = "\r\033[K"

    def output_format_plain(logs, previous_logs=None):
        for log in logs:
            if not previous_logs or (
                previous_logs and log["logId"] not in previous_logs
            ):
                log_json = json.loads(log["text"])
                if log_json.get("log") != None:
                    click.echo(f"{log_json['log']}\n")
                else:
                    click.echo(f"<null>")

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
            case "fetching_build_status":
                spinner = callback_args.get("spinner", "")
                click.echo(
                    f"\rFetching execution status... {spinner}",
                    nl=False,
                )
            case "fetched_build_status":
                build_status = callback_args.get("build_status", "")
                click.echo(f"\n*️⃣  Execution status: {build_status}.")
            case "fetching_additional_info":
                spinner = callback_args.get("spinner", "")
                click.echo(
                    f"\rFetching additional information... {spinner}",
                    nl=False,
                )
            case "querying_log_range":
                click.echo(
                    f"\nQuerying the logs between {epoch_to_iso_date(callback_args.get('start_epoch', ''))} and {epoch_to_iso_date(callback_args.get('end_epoch', ''))}"
                )
            case "querying_log_server":
                new_line = callback_args.get("new_line", "")
                if new_line:
                    click.echo("Querying log server..\n")
                else:
                    click.echo(f"{erase_sequence}Querying log server..", nl=False)
            case "display_logs":
                logs = callback_args.get("logs", [])
                previous_logs = callback_args.get("previous_logs", [])
                click.echo(erase_sequence, nl=False)
                output_format_plain(logs, previous_logs)
            case "warning":
                reason = callback_args.get("reason", "")
                click.echo(f"\n⚠️ Warning: {reason}\n")
            case "error":
                click.echo(
                    f"\n❌ Logs can't be retrieved at this moment.. Reason: {callback_args.get('reason', '')}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case _:
                pass

    try:
        monitor_output, targets, history = build_client.build_monitor(
            build_id, show_events, fetch_pr, id_format, echo_callback
        )
        if monitor_output:
            monitor_obj = execution_status_plain_output(
                monitor_output,
                targets,
                history,
                show_events,
            )

            click.echo(f"{erase_sequence}{monitor_obj}")
    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status
    except KeyboardInterrupt:
        ctx.exit(0)


@cli.command()
@click.pass_context
@click.argument("status", nargs=-1)
@click.option("--space", help="Space name.")
@click.option(
    "--format",
    "format",
    default="simple",
    type=click.Choice(["simple", "json"], case_sensitive=True),
    help="Output format: simple (default), json",
)
@common_options
def notification(ctx, status, space, format, skip_version_check, quiet):
    """
    Turn on/off the notification per space repository.

    Provide notification status to be set e.g. on, off.
    """
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

    if not quiet:
        click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} build notification")

    if status:
        status = status[0]
        if status not in ["on", "off"]:
            click.echo(
                f"❌ Error: notification status can only be set to 'on' or 'off'.",
                err=True,
            )
            ctx.exit(1)  # Exit with a non-zero status

    build_client = GBClient.Build(get_user_token())

    def echo_callback(callback_event: str, callback_args: Dict):
        if callback_event == "error":
            error_type = callback_args.get("error_type", "")
            reason = callback_args.get("reason", "")
            if error_type == "set":
                click.echo(
                    f"\n❌ Build notification can't be set at this moment... Reason: {reason}",
                    err=True,
                )
            else:
                click.echo(
                    f"\n❌ Build notification can't be retrieved at this moment... Reason: {reason}",
                    err=True,
                )
            sys.exit(1)  # Exit with a non-zero status
        else:
            pass  # Ignore unknown events

    try:
        notification_ignore, notification_space = build_client.build_notification(
            status, space, callback=(echo_callback if not quiet else None)
        )
        if not quiet:
            if status:
                click.echo(
                    f"✅ Notification status set for space {notification_space}: {'on' if notification_ignore == True else 'off'}."
                )
            else:
                click.echo(
                    f"✅ Notification status retrieved for space {notification_space}: {'on' if notification_ignore == True else 'off'}."
                )
        if format == "json":
            import re

            _m = re.match(r'^"([^"]+)"\s+\(([^)]+)\)$', notification_space or "")
            click.echo(
                json.dumps(
                    {
                        "space": _m.group(1) if _m else notification_space,
                        "namespace": _m.group(2) if _m else None,
                        "status": "on" if notification_ignore else "off",
                    }
                )
            )
    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status
    except KeyboardInterrupt:
        ctx.exit(0)


@cli.command()
@click.pass_context
@click.argument(
    "build-id",
    required=True,
)
@click.option(
    "--description",
    help="Build description. Set a description for the build. This can be used to provide additional context or details. If the text contains spaces, enclose it in quotes.",
)
@click.option(
    "--tag",
    multiple=True,
    help="Single tag name or comma-separated list of tag names. Tag names can only contain alphanumeric characters, underscores, and hyphens.",
)
@click.option(
    "--tags", help="Comma-separated list of tag names. For example: 'tagA, tagB'."
)
@click.option(
    "--append",
    is_flag=True,
    default=False,
    help="Adds the tags without replacing existing ones.",
)
@click.option(
    "--format",
    "format",
    default="simple",
    type=click.Choice(["simple", "json"], case_sensitive=True),
    help="Output format: simple (default), json",
)
@common_options
def update(
    ctx,
    build_id: str,
    description: str,
    tag: tuple,
    tags: str,
    append: bool,
    format: str,
    skip_version_check: bool,
    quiet: bool,
):
    """
    Update a detailed description of the specified build or tags.
    Provide an build UUID or URI.
    """
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

    build_client = GBClient.Build(get_user_token())

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "error":
                reason = callback_args.get("reason", "")
                click.echo(
                    f"\n❌ Build can't be updated at this moment... Reason: {reason}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case _:
                pass  # Ignore unknown events

    try:
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

        build_client = GBClient.Build(get_user_token())

        if not quiet:
            click.echo(f"(1/2) Fetching build with build_id {build_id}")

        build = (
            build_client.fetch_build(
                build_id,
                id_format,
                echo_callback,
            )
            if quiet
            else execute_with_spinner(
                build_client.fetch_build,
                build_id,
                id_format,
                echo_callback,
            )
        )

        if build:
            if not tag and tags is None and description is None:
                click.echo(f"\n No values provided for update.", err=True)
                sys.exit(1)  # Exit with a non-zero status

            normalized_tags = None
            # Check if tags were explicitly provided (including empty string)
            tags_explicitly_provided = bool(tag) or tags is not None

            if tags_explicitly_provided:
                # Special case: User explicitly provided --tags "" (empty string)
                if tags == "" and not tag:
                    # User explicitly cleared tags
                    normalized_tags = []
                else:
                    # Validate provided tags (handles --tag and --tags with values)
                    normalized_tags = validate_tags(
                        build_client.github_token,
                        tags_as_tuple=tag,
                        tags_str=tags,
                        callback=echo_callback,
                    )

                    if normalized_tags == None or len(normalized_tags) == 0:
                        # Only error if user provided tags but they're invalid
                        # (not if user provided --tags "" which should result in empty list)
                        if not (tags == "" and not tag):
                            click.echo(f"❌ Given tag/tags are not correct", err=True)
                            sys.exit(1)  # Exit with a non-zero status

            # Validate that --append is not used with empty tags
            if append and tags == "" and not tag:
                click.echo(f'\n❌ --append cannot be used with --tags ""', err=True)
                sys.exit(1)  # Exit with a non-zero status
            if not quiet:
                click.echo(f"\n(2/2) Updating build with build_id {build_id}")
            build = (
                build_client.update_build(
                    build_id=build_id,
                    description=description,
                    tags=normalized_tags,
                    append=append,
                    callback=echo_callback,
                )
                if quiet
                else execute_with_spinner(
                    build_client.update_build,
                    build_id=build_id,
                    description=description,
                    tags=normalized_tags,
                    append=append,
                    callback=echo_callback,
                )
            )

            if not quiet and build:
                click.echo(f"\n✅ Build was updated sucessfully!")
                name = build["name"]
                description = build["description"] or "No description provided"
                tags = build["tags"] or ""
                click.echo("-" * 60)
                click.echo(f"NAME       : {name}")
                click.echo(f"DESCRIPTION: {description}")
                click.echo(f"TAGS: {tags}")
                click.echo("-" * 60)
            if format == "json" and build:
                click.echo(
                    json.dumps(
                        {
                            "name": build["name"],
                            "description": build["description"],
                            "tags": build["tags"],
                        }
                    )
                )

        else:
            click.echo(f"\n❌ Build wat not found. Please try again.", err=True)
            sys.exit(1)  # Exit with a non-zero status

    except AuthException as e:
        click.echo(f"❌ {str(e)}", err=True)
        ctx.exit(1)  # Exit with a non-zero status
    except Exception as e:
        click.echo(f"\n{str_exc_chain(e)}", err=True)
        click.echo(f"❌ Build failed!", err=True)
        ctx.exit(1)  # Exit with a non-zero status
