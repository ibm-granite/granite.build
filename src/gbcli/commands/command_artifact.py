import json
import logging
import sys
import webbrowser
from pathlib import Path
from typing import Dict

import click
from rich.console import Console
from rich.table import Table
from tabulate import tabulate
from tqdm import tqdm

from gbcli.client.client import GBClient
from gbcli.commands.command_auth import execute_with_spinner, str_exc_chain
from gbcli.commands.common_options import (
    common_options,
    pass_context_and_reject_standalone,
)
from gbcli.services.service_artifact import (
    ArtifactURIError,
    lookup_hf_resource_group_id,
)
from gbcli.utils.checksum import calculate_checksum_
from gbcli.utils.gbconstants import (
    ARTIFACT_LINEAGE_DEFAULT_HEADERS,
    ARTIFACT_LINEAGE_FULL_HEADERS,
    ARTIFACT_LIST_HEADERS,
    CLIPBOARD_CHAR,
    DEFAULT_CHECKSUM_CONCURRENCY,
    HF_ORGANIZATION_DEFAULT,
    LAKEHOUSE_FILESET_SHARED_TABLE_NAME,
    LAKEHOUSE_FILESET_TABLE_NAME,
    LAKEHOUSE_MODEL_SHARED_TABLE,
    LAKEHOUSE_MODEL_TABLE,
    ORIGIN_CERTIFY_MESSAGE,
    PROJECT_NAME,
    hf_token,
)
from gbcli.utils.gbcredentials import get_user_token
from gbcli.utils.lh_auth import AuthException
from gbcli.utils.lh_fileset import get_fileset_subforlder
from gbcli.utils.lh_model import get_model_subforlder
from gbcli.utils.spaceutil import resolve_space
from gbcli.utils.utils import (
    check_runnable_browser,
    combine_tags,
    compare_env_uri,
    decode_uri,
    get_artifact_formatted_name,
    get_artifact_lineage_url,
    get_artifact_uuid,
    humanize_iso_date,
    is_valid_checksum,
    is_valid_name,
    origins_from_local,
    parse_artifact_identifier,
    parse_build_identifier,
    validate_tags,
)
from gbcli.utils.versionutil import check_current_and_latest_versions
from gbcommon.utils.hf_utils import (
    convert_hf_uri_to_url,
    parse_hf_uri,
)

logger = logging.getLogger(__name__)


@click.group("artifact")
@pass_context_and_reject_standalone
def cli(ctx):
    """Work with artifacts"""
    ctx.ensure_object(dict)  # Ensures `ctx.obj` is a dictionary


@cli.command()
@click.pass_context
@click.option(
    "--from-local",
    required=True,
    help="Path to the local file (for table or dataset) or directory (for model, fileset, dataset, or bucket).",
)
@click.option(
    "--format",
    default="simple",
    type=click.Choice(["simple", "json"], case_sensitive=True),
    help="Response format: simple (default), json",
)
@click.option("--artifact-name", required=True, help="Artifact name to be registered.")
@click.option(
    "--label",
    help="Label for model or fileset artifacts. If not provided, the artifact name will be used as the label.",
)
@click.option(
    "-t",
    "--type",
    type=click.Choice(
        ["model", "table", "fileset", "dataset", "bucket"], case_sensitive=True
    ),
    help="Artifact type to be saved. Options: model, fileset, table, dataset, bucket",
    required=True,
)
@click.option(
    "-s",
    "--size",
    help="Specifies the size of the model. For example: 1b, 3b, etc. If --type = model, size is required. If it not given or it is not part of the name, for example: 'model_name-size-variant' to be infered, it will be requested during the execution. ",
)
@click.option(
    "--variant",
    help="Specifies the variant of the model. If --type = model, variant is required. If it not given or it is not part of the name, for example: 'model_name-size-variant' to be infered, it will be requested during the execution. ",
)
@click.option(
    "--model-type",
    help="Specifies the type of the model. Default: granite",
)
@click.option(
    "--table",
    help="Table name for table artifacts. If not provided, the artifact name will be used as the table name.",
)
@click.option(
    "--description",
    default="",
    help="Artifact description. Set a description for the artifact. This can be used to provide additional context or details. If the text contains spaces, enclose it in quotes.",
)
@click.option(
    "--calculate-checksum",
    is_flag=True,
    default=False,
    help="Calculate a checksum value for the artifact. Default: false.",
)
@click.option(
    "--tag",
    multiple=True,
    help="Single tag name or comma-separated list of tag names. Tag names can only contain alphanumeric characters, underscores, and hyphens.",
)
@click.option(
    "--tags", help="Comma-separated list of tag names. For example: 'tagA, tagB'."
)
@click.option("--space", help="Space name.")
@click.option(
    "--origin",
    multiple=True,
    help="""Origin is an optional, multiple options which can either take artifact uuid or artifact uri. This is for the new requirement to meet the obligations to track the usage of artifacts originated from a model with restricted use. You can also use keep the 'origin' file created from 'artifact download' command. See 'llmb artifact download --help' for more details.""",
)
@click.option(
    "--origin-list",
    help="Take a text file (one line per entry) that lists artifact uuid/uri, instead of --origin options.",
)
@click.option(
    "--certify-no-restrictions",
    is_flag=True,
    default=False,
    help="If --origin is not provided, use this option to certify that the artifact to be uploaded was not created using a model with restricted use. Your certification information is recorded",
)
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="If --yes is provided, origin file will be used if is present in the local path.",
)
@click.option(
    "--hf-organization",
    default=None,
    help="HuggingFace organization. Overrides the environment default.",
)
@click.option(
    "--resource-group-id",
    default=None,
    help="Resource group ID. If omitted, resolved from the GB space name via the HF Enterprise API.",
)
@click.option(
    "--store",
    default="lh",
    type=click.Choice(["lh", "hf"], case_sensitive=True),
    help="Target artifact store: lh (Lakehouse, default) or hf (HuggingFace).",
)
@click.option(
    "--public",
    is_flag=True,
    default=False,
    help="Make the HuggingFace repository public (default is private, only for --store hf).",
)
@common_options
def push(
    ctx,
    from_local: str,
    format: str,
    artifact_name: str,
    label: str,
    type: str,
    size: str,
    variant: str,
    model_type: str,
    table: str,
    description: str,
    calculate_checksum: bool,
    tag: tuple,
    tags: str,
    space: str,
    origin: tuple,
    origin_list: str,
    certify_no_restrictions: bool,
    yes: bool,
    hf_organization: str,
    resource_group_id: str,
    store: str,
    public: bool,
    skip_version_check: bool,
    quiet: bool,
):
    """A file (for a table) or a directory (for a model, fileset, dataset, or bucket) is uploaded and registered. Additional options will be required depending on the type of artifact."""

    # Validate that --public is only used with HuggingFace
    if public and store != "hf":
        click.echo(
            f"❌ Error: The --public flag is only valid for HuggingFace storage (--store hf). Got store '{store}'.",
            err=True,
        )
        ctx.exit(1)

    # For HuggingFace, default is private unless --public flag is used
    private = not public if store == "hf" else False

    # Validate store type compatibility with artifact type
    if store == "hf" and type not in ["model", "dataset", "bucket"]:
        click.echo(
            f"❌ Error: Push and register with store 'hf' is only allowed for artifact types 'model', 'dataset', and 'bucket'. Got type '{type}'.",
            err=True,
        )
        ctx.exit(1)

    if store == "lh" and type == "dataset":
        click.echo(
            f"❌ Error: Push and register with store 'lh' does not support artifact type 'dataset'. Use store 'hf' instead.",
            err=True,
        )
        ctx.exit(1)

    if not skip_version_check:
        try:
            outdated_version = check_current_and_latest_versions()
        except Exception as e:
            click.echo(f"❌ {str(e)}.", err=True)
            ctx.exit(1)  # Exit with a non-zero status
        if outdated_version:
            click.echo(outdated_version, err=True)
            ctx.exit(1)  # Exit with a non-zero status

    if not from_local or not artifact_name:
        click.echo(
            "Error: Both --from-local and --artifact-name parameters are required.",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status

    if (
        type == "model" or type == "fileset" or type == "dataset" or type == "bucket"
    ) and table:
        click.echo(
            f"❌ The --table option is not valid for {type} artifacts. Please try again without the --table option.",
            err=True,
        )
        ctx.exit(1)

    if (type == "table" or type == "dataset" or type == "bucket") and label:
        click.echo(
            f"❌ The --label option is not valid for {type} artifacts. Please try again without the --label option.",
            err=True,
        )
        ctx.exit(1)

    # Check artifact name formatting
    name_valid, name_invalid_chars = is_valid_name(artifact_name, "artifact_name")
    if not name_valid:
        click.echo(
            f"❌ Artifact name contains invalid characters: {name_invalid_chars} Please try again.",
            err=True,
        )
        ctx.exit(1)

    if format == "json":
        quiet = True

    showModelTypePrompt = not model_type or model_type.strip() == ""
    showSizePrompt = not size or size.strip() == ""
    showVariantPrompt = not variant or variant.strip() == ""

    if type == "table":
        table = artifact_name if table is None else table

    if table:
        table_valid, table_invalid_chars = is_valid_name(table, "table_name")
        if not table_valid:
            click.echo(
                f"❌ Table name contains invalid characters: {table_invalid_chars} Please try again.",
                err=True,
            )
            ctx.exit(1)

    if label:
        label_valid, label_invalid_chars = is_valid_name(label, "label_name")
        if not label_valid:
            click.echo(
                f"❌ Label contains invalid characters: {label_invalid_chars} Please try again.",
                err=True,
            )
            ctx.exit(1)

    label = artifact_name if label is None else label

    artifact_client = GBClient.Artifact(get_user_token())

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "error":
                reason = callback_args.get("reason", "")
                click.echo(
                    f"\n❌ Artifact push can't be executed at this moment... Reason: {reason}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case "warning":
                reason = callback_args.get("reason", "")
                if not quiet:
                    click.echo(f"\n⚠️ Warning: {reason}", err=True)
                sys.exit(1)  # Exit with a non-zero status
            case "origin-file" | "remove-empty-values":
                reason = callback_args.get("reason", "")
                click.echo(f"\n{reason}")
                if yes == False:
                    if (
                        click.confirm(
                            "Do you want to proceed",
                            default=False,
                        )
                        == False
                    ):
                        sys.exit(1)  # Exit with a non-zero status
            case _:
                pass  # Ignore unknown events

    def update_artifact_status(artifact_id, status):
        artifact = (
            artifact_client.update_artifact(
                artifact_id=artifact_id,
                status=status,
                isUpdate=False,
                callback=echo_callback,
            )
            if quiet
            else execute_with_spinner(
                artifact_client.update_artifact,
                artifact_id=artifact_id,
                status=status,
                isUpdate=False,
                callback=echo_callback,
            )
        )
        return artifact

    artifact_id = None

    def handle_lh_push_exception(artifact_id):
        if artifact_id:
            if not quiet:
                click.echo(f"Updating artifact status")
            status = "failed"
            update_artifact_status(artifact_id, status)

    try:
        from lakehouse.core import UnauthorizedException

        total_steps = 6 if calculate_checksum else 5

        normalized_tags = []
        if bool(tag) or bool(tags):

            normalized_tags = validate_tags(
                artifact_client.github_token,
                space=space,
                tags_as_tuple=tag,
                tags_str=tags,
                callback=echo_callback,
            )

            if normalized_tags == None or len(normalized_tags) == 0:
                click.echo(f"❌ Given tag/tags are not correct", err=True)
                sys.exit(1)  # Exit with a non-zero status

        normalized_origins = []
        local_origins = origins_from_local(from_local)
        if not quiet:
            click.echo(f"(1/{total_steps}) Validating origins.")

        if bool(local_origins) or bool(origin) or bool(origin_list):
            normalized_origins = artifact_client.validate_origins(
                artifact_name=artifact_name,
                local_origins=local_origins,
                origin=origin,
                origin_list=origin_list,
                certify_no_restrictions=certify_no_restrictions,
                callback=echo_callback,
            )

            if normalized_origins == None or len(normalized_origins) == 0:
                click.echo(f"❌ Given origin/origin-list are not correct", err=True)
                sys.exit(1)  # Exit with a non-zero status

        elif not certify_no_restrictions:
            click.echo(ORIGIN_CERTIFY_MESSAGE)
            sys.exit(1)  # Exit with a non-zero status

        # Initialize tokens
        hf_token = None
        lh_token = None
        revision = None
        version = None

        # === Lakehouse-specific setup ===
        if store == "lh":
            # Prompt for model properties if needed
            if type == "model" and (
                showSizePrompt or showVariantPrompt or showModelTypePrompt
            ):
                try:
                    # try to infer it
                    parts = artifact_name.split("-")
                    size = parts[-2]  # get the size segment (8b), for this format
                    variant = parts[-1]  # get the  (base)

                except Exception:
                    pass

                # Prompts the user for model properties with default values.
                model_type = (
                    click.prompt("Type", default="granite", show_default=True).strip()
                    if showModelTypePrompt
                    else model_type
                )
                size = (
                    click.prompt("Size", default=size, show_default=True).strip()
                    if showSizePrompt
                    else size
                )
                variant = (
                    click.prompt("Variant", default=variant, show_default=True).strip()
                    if showVariantPrompt
                    else variant
                )
                if size == "" or variant == "" or model_type == "":
                    click.echo(
                        f"\n❌ Error: Either provide --size and --variant and --model-type, and/or inferable model label, for example 'model-x.x-size-variant'",
                        err=True,
                    )
                    ctx.exit(1)  # Exit with a non-zero status

            if not quiet:
                click.echo(f"(2/{total_steps}) Obtaining Lakehouse token.")
            lh_token = GBClient.Auth.lakehouse_token_for_space(
                artifact_client.github_token, space=space, callback=echo_callback
            )
            if not lh_token:
                return
            if not quiet:
                click.echo(f"Lakehouse token obtained successfully!")

        # === HuggingFace-specific setup ===
        if store == "hf":
            if not quiet:
                click.echo(f"(2/{total_steps}) Obtaining HuggingFace token.")
            hf_token = GBClient.Auth().hf_token()
            if not hf_token:
                return
            if not quiet:
                click.echo(f"HuggingFace token obtained successfully!")

            # Resolve resource group id from the GB space name when not given.
            if not resource_group_id:
                org = hf_organization or HF_ORGANIZATION_DEFAULT
                resolved_space_name = space
                if not resolved_space_name:
                    global_space = resolve_space(
                        artifact_client.github_token, space, callback=echo_callback
                    )
                    if global_space is not None:
                        resolved_space_name = global_space.get("name")
                if not resolved_space_name:
                    click.echo(
                        "❌ Could not determine GB space name to resolve the "
                        "HuggingFace resource group id. Pass --space, set a "
                        "default space, or pass --resource-group-id explicitly.",
                        err=True,
                    )
                    sys.exit(1)
                resource_group_id = lookup_hf_resource_group_id(
                    github_token=artifact_client.github_token,
                    space_name=resolved_space_name,
                    organization=org,
                )
                if not resource_group_id:
                    click.echo(
                        f"❌ Could not resolve HuggingFace resource group id for "
                        f"space '{resolved_space_name}' in organization '{org}'. "
                        f"Pass --resource-group-id explicitly, or ensure the "
                        f"resource group exists.",
                        err=True,
                    )
                    sys.exit(1)

        checksum = None
        existing_registration = None

        # === Lakehouse-specific checksum and existence check ===
        if store == "lh" and calculate_checksum:
            click.echo(f"(3/{total_steps}) Calculating artifact checksum value.")
            checksum = calculate_checksum_(from_local, DEFAULT_CHECKSUM_CONCURRENCY)
            click.echo(f"Calculated artifact checksum: {checksum}")

            # Check for existing registration if checksums are being used
            existing_registration = artifact_client.existing_checksum_artifacts(
                space, checksum
            )

            # if existing artifact registration
            if existing_registration:
                artifact_id = existing_registration.get("uuid")
                revision = (
                    existing_registration.get("uri").split("/")[7]
                    if existing_registration.get("type") == "model"
                    else None
                )
                version = (
                    existing_registration.get("uri").split("/")[7]
                    if existing_registration.get("type") == "fileset"
                    else None
                )

                # check if artifact content exists in LH
                artifact_existence = artifact_client.check_existence(
                    lh_token=lh_token,
                    type=existing_registration.get("type"),
                    space=space,
                    namespace=existing_registration.get("uri").split("/")[3],
                    table=existing_registration.get("uri").split("/")[5],
                    dataset=None,
                    label=(
                        existing_registration.get("uri").split("/")[6]
                        if type in ("fileset", "model")
                        else None
                    ),
                    revision=revision,
                    version=version,
                )

                match existing_registration.get("status", None):
                    case "failed":
                        # If artifact exists
                        if artifact_existence.get("success"):
                            click.echo(
                                f"❌ An artifact with the same checksum value already exists in Lakehouse, but the status is 'failed' (uuid: {existing_registration.get('uuid')}). Please verify contents and contact a space admin to update the status or delete the artifact.",
                                err=True,
                            )
                            sys.exit(1)
                        # If artifact does not exist
                        else:
                            # If user is the owner
                            if existing_registration.get("user_is_owner"):
                                click.echo(
                                    f"⚠️{'  '}An artifact with the same checksum value is already registered, but the status is 'failed' (uuid: {existing_registration.get('uuid')}).\n({total_steps-2}/{total_steps}) Skipping the registration step and retrying artifact push."
                                )
                            else:
                                click.echo(
                                    f"❌ An artifact with the same checksum value is already registered, but the status is 'failed' (uuid: {existing_registration.get('uuid')}). Please contact a space admin to update the status or delete the artifact.",
                                    err=True,
                                )
                                sys.exit(1)

                    case "pending":
                        # If artifact exists
                        if artifact_existence.get("success"):
                            click.echo(
                                f"❌ An artifact with the same checksum value already exists in Lakehouse, but the status is 'pending' (uuid: {existing_registration.get('uuid')}). Please verify contents and contact a space admin to update the status or delete the artifact.",
                                err=True,
                            )
                            sys.exit(1)
                        # If artifact does not exist
                        else:
                            # If user is the owner
                            if existing_registration.get("user_is_owner"):
                                click.echo(
                                    f"⚠️{'  '}An artifact with the same checksum value is already registered, but the status is 'pending' (uuid: {existing_registration.get('uuid')}).\n({total_steps-2}/{total_steps}) Skipping the registration step and proceeding with artifact push."
                                )
                            else:
                                click.echo(
                                    f"❌ An artifact with the same checksum value is already registered, but the status is 'pending' (uuid: {existing_registration.get('uuid')}). Please contact a space admin to push the artifact."
                                )
                                sys.exit(1)

                    case "success":
                        click.echo(
                            f"❌ An artifact with the same checksum value already exists. The uuid is {existing_registration.get('uuid')}.",
                            err=True,
                        )
                        sys.exit(1)  # Exit with a non-zero status

                    case _:
                        click.echo(
                            f"❌ An artifact with the same checksum value already exists.",
                            err=True,
                        )
                        sys.exit(1)  # Exit with a non-zero status

        # Only register if there was not an existing registration
        if not existing_registration:
            if not quiet:
                click.echo(
                    f"({total_steps-2}/{total_steps}) Registering the artifact with status 'pending'."
                )

            artifact_id = None
            status = "pending"

            # === Lakehouse-specific metadata ===
            if store == "lh":
                revision = "granite-dot-build"
                version = "granite-dot-build"

            response_register = (
                artifact_client.register_artifact(
                    artifact_name=artifact_name,
                    description=description,
                    checksum=checksum,
                    type=type,
                    label=label,
                    table=table,
                    revision=revision,
                    version=version,
                    tags=normalized_tags,
                    status=status,
                    space=space,
                    origin_uris=normalized_origins,
                    hf_organization=hf_organization,
                    resource_group_id=resource_group_id,
                    store=store,
                    certified_no_restrictions=certify_no_restrictions,
                    callback=echo_callback,
                )
                if quiet
                else execute_with_spinner(
                    artifact_client.register_artifact,
                    artifact_name=artifact_name,
                    description=description,
                    checksum=checksum,
                    tags=normalized_tags,
                    status=status,
                    type=type,
                    label=label,
                    table=table,
                    revision=revision,
                    version=version,
                    space=space,
                    origin_uris=normalized_origins,
                    store=store,
                    hf_organization=hf_organization,
                    resource_group_id=resource_group_id,
                    certified_no_restrictions=certify_no_restrictions,
                    callback=echo_callback,
                )
            )

            artifact_id = response_register["uuid"]
            if not quiet:
                click.echo(
                    f"\nArtifact registered successfully with uuid {artifact_id}"
                )

        store_name = "HuggingFace" if store == "hf" else "Lakehouse"
        if not quiet:
            click.echo(
                f"({total_steps-1}/{total_steps}) Pushing file/files to {store_name}. This may take a while, please wait..."
            )

        response_push = (
            artifact_client.push(
                lh_token,
                from_local,
                type,
                label,
                artifact_name,
                size,
                variant,
                model_type,
                version,
                space,
                table,
                namespace=None,
                store=store,
                hf_token=hf_token,
                hf_organization=hf_organization,
                resource_group_id=resource_group_id,
                private=private,
                callback=echo_callback,
            )
            if quiet
            else execute_with_spinner(
                artifact_client.push,
                lh_token,
                from_local,
                type,
                label,
                artifact_name,
                size,
                variant,
                model_type,
                version,
                space,
                table,
                namespace=None,
                store=store,
                hf_token=hf_token,
                hf_organization=hf_organization,
                resource_group_id=resource_group_id,
                private=private,
                callback=echo_callback,
            )
        )

        if not quiet:
            click.echo(f"({total_steps}/{total_steps}) Updating artifact status")

        status = "success"
        artifact = update_artifact_status(artifact_id, status)

        if not quiet:
            click.echo(
                f"\n✅ Artifact/{type.title()} '{artifact_name}' was successfully pushed with uuid {artifact['uuid']} uri {artifact['uri']}"
            )

            # Show HuggingFace URL if this was an HF push
            if store == "hf":
                try:
                    uri = artifact["uri"]
                    hf_url = convert_hf_uri_to_url(uri)
                    click.echo(f"🔗 View on HuggingFace: {hf_url}")
                except Exception as e:
                    logger.debug(f"Could not generate HuggingFace URL: {e}")

        if format == "json":
            dict_format = json.dumps(
                {
                    "uuid": artifact["uuid"],
                    "uri": artifact["uri"],
                    "checksum": artifact["checksum"],
                    "status": artifact["status"],
                }
            )
            click.echo(dict_format)
            return dict_format

    except ValueError as e:
        click.echo(f"\n❌ {e}", err=True)
        handle_lh_push_exception(artifact_id)
        ctx.exit(1)

    # === Lakehouse-specific exception handlers ===
    except AuthException as e:
        click.echo(f"❌ {str(e)}", err=True)
        handle_lh_push_exception(artifact_id)
        ctx.exit(1)  # Exit with a non-zero status
    except UnauthorizedException as e:
        click.echo(f"❌ {str(e)}", err=True)
        handle_lh_push_exception(artifact_id)
        ctx.exit(1)  # Exit with a non-zero status

    except Exception as e:
        click.echo(f"\n{str_exc_chain(e)}", err=True)
        store_name = "HuggingFace" if store == "hf" else "Lakehouse"
        click.echo(f"❌ Artifact/{type.title()} push to {store_name} failed!", err=True)
        handle_lh_push_exception(artifact_id)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.option(
    "--artifact-name",
    required=True,
    help="Artifact name to be registered.",
)
@click.option(
    "--uri",
    help="Lakehouse URI of the artifact.",
)
@click.option(
    "-t",
    "--type",
    type=click.Choice(
        ["dataset", "model", "table", "fileset", "bucket"], case_sensitive=True
    ),
    help="Artifact type to be saved. Options: dataset, model, table (default), fileset, bucket",
)
@click.option(
    "--namespace",
    help="Namespace where artifact is located.",
)
@click.option(
    "--table",
    help=f"Table where artifact is located. Model artifact default: {LAKEHOUSE_MODEL_SHARED_TABLE}. Fileset artifact default: {LAKEHOUSE_FILESET_SHARED_TABLE_NAME}",
)
@click.option(
    "--label",
    help="Artifact label. If model, model label. If fileset, fileset label.",
)
@click.option(
    "--revision",
    help="Model revision.",
)
@click.option(
    "--version",
    help="Fileset version.",
)
@click.option(
    "--dataset",
    help="Dataset name. Corresponds to the 'dataset' column in Lakehouse dataset objects.",
)
@click.option(
    "--description",
    default="",
    help="Artifact description. Set a description for the artifact. This can be used to provide additional context or details. If the text contains spaces, enclose it in quotes.",
)
@click.option(
    "--checksum",
    default="",
    help="Set a checksum for the artifact.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Register artifact without verification. Default: False",
)
@click.option("--space", help="Space name.")
@click.option(
    "--origin",
    multiple=True,
    help="""Origin is an optional, multiple options which can either take artifact uuid or artifact uri. This is for the new requirement to meet the obligations to track the usage of artifacts originated from a model of a restricted license and restricted use. You can also use keep the 'origin' file generated from 'artifact download' command. See 'llmb artifact download --help' for more details.""",
)
@click.option(
    "--origin-list",
    help="Take a text file (one line per entry) that lists artifact uuid/uri, instead of --origin options.",
)
@click.option(
    "--certify-no-restrictions",
    is_flag=True,
    default=False,
    help="If --origin is not provided, use this option to certify that the artifact to be uploaded was not created using a model of a restricted license and restricted use. The artifact isn't from a restricted origin.",
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
    "--status",
    type=click.Choice(["pending", "success", "failed"], case_sensitive=False),
    help="Status of the artifact: pending, success, failed",
    default="success",
)
@click.option(
    "--hf-organization",
    help="HuggingFace organization name for artifact registration.",
)
@click.option(
    "--resource-group-id",
    help="Resource group ID for artifact registration.",
)
@click.option(
    "--store",
    default="lh",
    type=click.Choice(["lh", "hf"], case_sensitive=True),
    help="Target artifact store: lh (Lakehouse, default) or hf (HuggingFace).",
)
@click.option(
    "--format",
    "format",
    default="simple",
    type=click.Choice(["simple", "json"], case_sensitive=True),
    help="Output format: simple (default), json",
)
@common_options
def register(
    ctx,
    artifact_name: str,
    uri: str,
    type: str,
    namespace: str,
    table: str,
    label: str,
    revision: str,
    version: str,
    dataset: str,
    description: str,
    checksum: str,
    tag: tuple,
    tags: str,
    status: str,
    force: bool,
    store: str,
    space: str,
    origin: tuple,
    origin_list: str,
    certify_no_restrictions: bool,
    hf_organization: str,
    resource_group_id: str,
    format: str,
    skip_version_check: bool,
    quiet: bool,
):
    """
    Register a new artifact.
    """
    if format == "json":
        quiet = True

    # Validate store type compatibility with artifact type
    # Note: type might be inferred from URI, so we validate after URI decoding below
    if not skip_version_check:
        try:
            outdated_version = check_current_and_latest_versions()
        except Exception as e:
            click.echo(f"❌ {str(e)}.", err=True)
            ctx.exit(1)  # Exit with a non-zero status
        if outdated_version:
            click.echo(outdated_version, err=True)
            ctx.exit(1)  # Exit with a non-zero status

    # Check artifact name formatting
    name_valid, name_invalid_chars = is_valid_name(artifact_name, "artifact_name")
    if not name_valid:
        click.echo(
            f"❌ Artifact name contains invalid characters: {name_invalid_chars} Please try again.",
            err=True,
        )
        ctx.exit(1)

    # Check checksum pattern
    if checksum:
        expected_checksum_128bit_length = 32
        checksum_valid, checksum_invalid_chars, checksum_length = is_valid_checksum(
            checksum, expected_checksum_128bit_length
        )
        if not checksum_valid:
            if len(checksum_invalid_chars) > 0:
                click.echo(
                    f"❌ Artifact checksum contains invalid characters: {checksum_invalid_chars} Please try again.",
                    err=True,
                )
                ctx.exit(1)
            if checksum_length != expected_checksum_128bit_length:
                click.echo(
                    f"❌ Artifact checksum length is invalid: {checksum_length}. Please try again.",
                    err=True,
                )
                ctx.exit(1)

    # === Lakehouse-specific URI decoding ===
    uri_env = None
    if uri:
        decoded_artifact = decode_uri(uri)

        uri_env, cli_env = compare_env_uri(uri)

        if uri_env == "prod" and cli_env != "prod":
            click.echo(
                f"⚠️{' '} Warning: You are registering a '{uri_env}' artifact in the '{cli_env}' environment."
            )
            force = True
        elif uri_env != cli_env:
            click.echo(
                f"❌ The environment '{uri_env}' doesn't match the CLI environment '{cli_env}'. CLI doesn't support artifact register across environments except for 'prod' artifacts."
            )
            ctx.exit(1)

        if type or table or label or revision or version or dataset or namespace:
            click.echo(
                f"❌ Error: artifact URI cannot be used along with type, namespace, table, label, revision, version or dataset arguments."
            )
            ctx.exit(1)

        type = decoded_artifact.type
        namespace = decoded_artifact.namespace
        table = decoded_artifact.table_name

        if type == "model":
            label = decoded_artifact.model_label
            revision = decoded_artifact.model_revision

        if type == "fileset":
            label = decoded_artifact.fileset_label
            version = decoded_artifact.fileset_version

        if type == "dataset":
            dataset = decoded_artifact.dataset_name

    if not uri and not type:
        click.echo(
            f"❌ Artifact type or artifact uri is required. Please try again.",
            err=True,
        )
        ctx.exit(1)

    # Validate store type compatibility with artifact type
    if store == "hf" and type not in ["model", "dataset", "bucket"]:
        click.echo(
            f"❌ Error: Registration with store 'hf' is only allowed for artifact types 'model', 'dataset' and 'bucket'. Got type '{type}'.",
            err=True,
        )
        ctx.exit(1)

    # === Lakehouse-specific type handling ===
    if type == "model":
        dataset = None
        version = None
        showRevisionPrompt = not revision or revision.strip() == ""
        showLabelPrompt = not label or label.strip() == ""

        label = (
            click.prompt("Model label", show_default=True).strip()
            if showLabelPrompt
            else label
        )

        revision = (
            click.prompt(
                "Revision", default=revision or "", show_default=False, type=str
            ).strip()
            if showRevisionPrompt
            else revision
        )
        if not table or table.strip() == "":
            table = click.prompt(
                "Model table",
                default=LAKEHOUSE_MODEL_SHARED_TABLE,
                show_default=True,
                type=click.Choice(
                    [LAKEHOUSE_MODEL_SHARED_TABLE, LAKEHOUSE_MODEL_TABLE],
                    case_sensitive=True,
                ),
            ).strip()
        elif table not in [LAKEHOUSE_MODEL_SHARED_TABLE, LAKEHOUSE_MODEL_TABLE]:
            click.echo(
                f"❌ '{table}' is not a valid table for models. Please try again with a model from the '{LAKEHOUSE_MODEL_SHARED_TABLE}' or '{LAKEHOUSE_MODEL_TABLE}' tables.",
                err=True,
            )
            ctx.exit(1)

        if label == "":
            click.echo(f"\n❌ Error: Please provide model label", err=True)
            ctx.exit(1)  # Exit with a non-zero status

    if type == "dataset":
        label = None
        revision = None
        version = None

        if not dataset or dataset.strip() == "":
            dataset = click.prompt("Dataset name").strip()

        if not table or table.strip() == "":
            table = click.prompt("Table").strip()

        if dataset == "" or table == "":
            click.echo(
                f"\n❌ Error: Please provide both table and dataset name", err=True
            )
            ctx.exit(1)  # Exit with a non-zero status

    if type == "fileset":
        revision = None
        dataset = None
        showVersionPrompt = not version or version.strip() == ""
        showLabelPrompt = not label or label.strip() == ""

        label = (
            click.prompt("Fileset label", default=label, show_default=True).strip()
            if showLabelPrompt
            else label
        )
        version = (
            click.prompt(
                "Fileset version", default=version, show_default=False, type=str
            ).strip()
            if showVersionPrompt
            else version
        )

        if not table or table.strip() == "":
            table = click.prompt(
                "Fileset table",
                default=LAKEHOUSE_FILESET_SHARED_TABLE_NAME,
                show_default=True,
                type=click.Choice(
                    [LAKEHOUSE_FILESET_SHARED_TABLE_NAME, LAKEHOUSE_FILESET_TABLE_NAME],
                    case_sensitive=True,
                ),
            ).strip()
        elif table not in [
            LAKEHOUSE_FILESET_SHARED_TABLE_NAME,
            LAKEHOUSE_FILESET_TABLE_NAME,
        ]:
            click.echo(
                f"❌ '{table}' is not a valid table for filesets. Please try again with a fileset from the '{LAKEHOUSE_FILESET_SHARED_TABLE_NAME}'  or '{LAKEHOUSE_FILESET_TABLE_NAME}' tables.",
                err=True,
            )
            ctx.exit(1)

        if label == "":
            click.echo(f"\n❌ Error: Please provide fileset label", err=True)
            ctx.exit(1)  # Exit with a non-zero status

    if type == "table":
        if not table or table.strip() == "":
            table = click.prompt("Table").strip()

        label = None
        revision = None
        dataset = None
        version = None

    if table:
        table_valid, table_invalid_chars = is_valid_name(table, "table_name")
        if not table_valid:
            click.echo(
                f"❌ Table name contains invalid characters: {table_invalid_chars} Please try again.",
                err=True,
            )
            ctx.exit(1)

    if label:
        label_valid, label_invalid_chars = is_valid_name(label, "label_name")
        if not label_valid:
            click.echo(
                f"❌ Label contains invalid characters: {label_invalid_chars} Please try again.",
                err=True,
            )
            ctx.exit(1)

    artifact_client = GBClient.Artifact(get_user_token())

    def echo_callback(callback_event: str, callback_args: Dict):
        if callback_event == "error":
            reason = callback_args.get("reason", "")
            click.echo(
                f"\n❌ Artifact registration can't be executed at this moment... Reason: {reason}",
                err=True,
            )
            sys.exit(1)  # Exit with a non-zero status
        if callback_event == "warning":
            reason = callback_args.get("reason", "")
            click.echo(f"\n⚠️ Warning: {reason}")
            sys.exit(1)  # Exit with a non-zero status
        if callback_event == "origin-file":
            reason = callback_args.get("reason", "")
            click.echo(f"\n{reason}")
            if (
                click.confirm(
                    "Do you want to proceed",
                    default=False,
                )
                == False
            ):
                sys.exit(1)  # Exit with a non-zero status

        else:
            pass  # Ignore unknown events

    class TableVerificationException(Exception):
        pass

    class DatasetVerificationException(Exception):
        pass

    try:
        normalized_tags = []
        if bool(tag) or bool(tags):

            normalized_tags = validate_tags(
                artifact_client.github_token,
                space=space,
                tags_as_tuple=tag,
                tags_str=tags,
                callback=echo_callback,
            )

            if normalized_tags == None or len(normalized_tags) == 0:
                click.echo(f"❌ Given tag/tags are not correct", err=True)
                sys.exit(1)  # Exit with a non-zero status

        normalized_origins = []
        totalSteps = "3" if force else "4"

        if not quiet:
            click.echo(f"(1/{totalSteps}) Validating origins.")

        if bool(origin) or bool(origin_list):
            normalized_origins = artifact_client.validate_origins(
                artifact_name=artifact_name,
                local_origins=None,
                origin=origin,
                origin_list=origin_list,
                certify_no_restrictions=certify_no_restrictions,
                callback=echo_callback,
            )
            if normalized_origins == None or len(normalized_origins) == 0:
                click.echo(f"❌ Given origin/origin-list are not correct", err=True)
                sys.exit(1)  # Exit with a non-zero status

        elif not certify_no_restrictions:
            click.echo(ORIGIN_CERTIFY_MESSAGE)
            sys.exit(1)  # Exit with a non-zero status

        # === Lakehouse-specific token and verification ===
        if store == "lh":
            if not quiet:
                click.echo(f"(2/{totalSteps}) Obtaining Lakehouse token.")

            lh_token = GBClient.Auth.lakehouse_user_token(
                artifact_client.github_token, callback=echo_callback
            )
            if not lh_token:
                return
            if not quiet:
                click.echo(f"Lakehouse token obtained successfully!")

            if not force:
                if not quiet:
                    click.echo(f"(3/{totalSteps}) Verifying artifact.")

                # check if artifact exists
                artifact_existence = artifact_client.check_existence(
                    lh_token,
                    type,
                    space,
                    namespace,
                    table,
                    dataset,
                    label,
                    revision,
                    version,
                )

                if not artifact_existence["success"]:
                    click.echo(artifact_existence["error"], err=True)
                    sys.exit(1)

                if not quiet:
                    click.echo(f"\nArtifact verified sucessfully!")
                    click.echo(f"(4/{totalSteps}) Registering the artifact/{type}")

        if force:
            if not quiet:
                click.echo(f"(3/{totalSteps}) Registering the artifact/{type}")

        # === Lakehouse-specific lh_env parameter ===
        lh_env = uri_env if uri and store == "lh" else None

        if quiet:
            response = artifact_client.register_artifact(
                artifact_name=artifact_name,
                description=description,
                checksum=checksum,
                type=type,
                label=label,
                revision=revision,
                version=version,
                space=space,
                namespace=namespace,
                table=table,
                dataset_name=dataset,
                tags=normalized_tags,
                status=status,
                lh_env=lh_env,
                origin_uris=normalized_origins,
                certified_no_restrictions=certify_no_restrictions,
                hf_organization=hf_organization,
                resource_group_id=resource_group_id,
                store=store,
                callback=echo_callback,
            )
        else:
            response = execute_with_spinner(
                artifact_client.register_artifact,
                artifact_name=artifact_name,
                description=description,
                checksum=checksum,
                type=type,
                label=label,
                revision=revision,
                version=version,
                space=space,
                namespace=namespace,
                table=table,
                dataset_name=dataset,
                tags=normalized_tags,
                status=status,
                lh_env=lh_env,
                origin_uris=normalized_origins,
                certified_no_restrictions=certify_no_restrictions,
                hf_organization=hf_organization,
                resource_group_id=resource_group_id,
                store=store,
                callback=echo_callback,
            )
            click.echo(
                f"\n✅ Artifact/{type.title()} '{artifact_name}' was successfully registered with uuid {response['uuid']} uri {response['uri']}"
            )
        if format == "json" and response:
            click.echo(json.dumps({"uuid": response["uuid"], "uri": response["uri"]}))

    # === Lakehouse-specific exception handlers ===
    except DatasetVerificationException as e:
        click.echo(
            f"\n❌ Dataset {dataset} does not exist in {namespace}.{table}! Please verify artifact name and location.",
            err=True,
        )
        ctx.exit(1)
    except TableVerificationException as e:
        click.echo(
            f"\n❌ Table {namespace}.{table} does not exist! Please verify artifact location.",
            err=True,
        )
        ctx.exit(1)
    except AuthException as e:
        click.echo(f"❌ {str(e)}", err=True)
        ctx.exit(1)  # Exit with a non-zero status

    except ValueError as e:
        click.echo(f"\n❌ {e}", err=True)
        ctx.exit(1)
    except Exception as e:
        click.echo(f"\n{str_exc_chain(e)}", err=True)
        store_name = "HuggingFace" if store == "hf" else "Lakehouse"
        click.echo(
            f"❌ Artifact/{type.title()} register to {store_name} failed!", err=True
        )
        ctx.exit(1)  # Exit with a non-zero status


def _lineage_lh(ctx, artifact_client, artifact, format, quiet, echo_callback):
    if not quiet:
        click.echo(f"(2/3) Obtaining Lakehouse token.")

    lh_token = GBClient.Auth.lakehouse_user_token(
        artifact_client.github_token, callback=echo_callback
    )
    if not lh_token:
        return
    if not quiet:
        click.echo(f"Lakehouse token obtained successfully!")

    decoded_artifact = decode_uri(artifact["uri"])
    artifact_name = get_artifact_formatted_name(decoded_artifact)

    if artifact_name is None:
        click.echo(
            f"\nArtifact name couldn't be parsed from object URI. Please try again."
        )
        return

    if not quiet:
        click.echo(
            f"(3/3) Fetching artifact lineage from Lakehouse. This may take a while, please wait..."
        )

    lineage_dict = (
        artifact_client.artifact_lineage(lh_token, artifact_name)
        if quiet
        else execute_with_spinner(
            artifact_client.artifact_lineage, lh_token, artifact_name
        )
    )

    if len(lineage_dict) == 0:
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
                    l["source_filter"],
                    l["source_type"],
                    l["source_object"],
                    l["target"],
                    l["target_filter"],
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
                ARTIFACT_LINEAGE_FULL_HEADERS,
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
                ARTIFACT_LINEAGE_DEFAULT_HEADERS,
                tablefmt="plain",
            )

        click.echo("\n\n" + lineage_table)

    artifact_lineage_url = get_artifact_lineage_url(decoded_artifact, artifact["uuid"])

    if artifact_lineage_url is not None and not quiet:
        click.echo(f"\nView the lineage visually at: {artifact_lineage_url}")
        if check_runnable_browser():
            answer = click.confirm("Open the browser?", False)

            if answer:
                webbrowser.open(artifact_lineage_url)


def _lineage_hf(ctx, artifact_client, artifact, format, quiet):
    artifact_uri = artifact["uri"]

    if not quiet:
        click.echo(f"(2/2) Fetching artifact lineage for {artifact_uri}...")

    response = (
        artifact_client.artifact_lineage_hf(artifact_uri)
        if quiet
        else execute_with_spinner(artifact_client.artifact_lineage_hf, artifact_uri)
    )

    runs = response.get("runs", [])
    if len(runs) == 0:
        click.echo("\nNo lineage found.")
        return

    if format == "json":
        click.echo("\n" + json.dumps(response, indent=2, default=str))
    else:
        table = Table(title="Artifact Lineage", padding=(0, 1))
        table.add_column("Job Name", overflow="fold")
        table.add_column("Job Type", overflow="fold")
        table.add_column("Run ID", overflow="fold")
        table.add_column("Status", overflow="fold")
        table.add_column("Created At", overflow="fold")
        table.add_column("Inputs", overflow="fold")
        table.add_column("Outputs", overflow="fold")

        def _format_ref(ref):
            uri = ref.get("uri", "")
            name = ref.get("name", "")
            if uri and uri.startswith("hf://"):
                parts = [p for p in uri.replace("hf://", "").split("/") if p]
                if parts[0] in ("datasets", "spaces", "buckets"):
                    artifact_type = parts[0].rstrip("s").capitalize()
                    repo_id = "/".join(parts[1:3])
                elif "." in parts[0]:
                    # host is present, check next segment
                    if len(parts) > 1 and parts[1] in ("datasets", "spaces", "buckets"):
                        artifact_type = parts[1].rstrip("s").capitalize()
                        repo_id = "/".join(parts[2:4])
                    else:
                        artifact_type = "Model"
                        repo_id = "/".join(parts[1:3])
                else:
                    artifact_type = "Model"
                    repo_id = "/".join(parts[0:2])
                return f"{artifact_type}:\n{repo_id}|{uri}"
            return f"{name}\n{uri}" if uri else name

        for run in runs:
            inputs_list = [
                _format_ref(ref)
                for ref in run.get("inputs", [])
                if ref.get("node_type") == "artifact"
            ]
            outputs_list = [
                _format_ref(ref)
                for ref in run.get("outputs", [])
                if ref.get("node_type") == "artifact"
            ]
            inputs_str = "\n".join(inputs_list)
            outputs_str = "\n".join(outputs_list)
            table.add_row(
                run.get("job_name", "-"),
                run.get("job_type", "-"),
                run.get("run_id", "-"),
                run.get("status", "-"),
                run.get("created_at", "-"),
                inputs_str or "-",
                outputs_str or "-",
            )

        console = Console()
        console.print(table)

        if response.get("truncated"):
            click.echo("(graph truncated at max depth)")


@cli.command()
@click.pass_context
@click.argument("artifact-id", required=True)
@click.option(
    "--format",
    default="simple",
    type=click.Choice(["simple", "full", "json"], case_sensitive=True),
    help="Lineage contents format: simple (default), full, json",
)
@common_options
def lineage(ctx, artifact_id: str, format: str, skip_version_check: bool, quiet: bool):
    """
    Access artifact lineage

    Provide artifact UUID or URI.
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

    # Name formatting reference: https://pages.github.ibm.com/arc/dmf-library/code_reference/pythonic_access/lakehouse_reference/lineage/#data-lineage-consumption
    if not artifact_id:
        click.echo(
            "Error: --artifact-id parameter is required.",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status
    artifact_client = GBClient.Artifact(get_user_token())

    def echo_callback(callback_event: str, callback_args: Dict):
        if callback_event == "error":
            reason = callback_args.get("reason", "")
            click.echo(
                f"\n❌ Artifact lineage can't be retrieved at this moment... Reason: {reason}",
                err=True,
            )
            sys.exit(1)  # Exit with a non-zero status
        else:
            pass  # Ignore unknown events

    try:
        id_format = parse_artifact_identifier(artifact_id)
        if id_format in ["uuid", "uri"]:
            if not quiet:
                click.echo(f"(1/2) Obtaining artifact..")

            artifact = (
                artifact_client.fetch_artifact(artifact_id, callback=echo_callback)
                if id_format == "uuid"
                else artifact_client.fetch_artifact_uri(
                    artifact_id, callback=echo_callback
                )
            )

            if not artifact:
                return

            artifact_uri = artifact["uri"]
            if artifact_uri.startswith("hf://"):
                _lineage_hf(ctx, artifact_client, artifact, format, quiet)
            elif artifact_uri.startswith("lh://"):
                _lineage_lh(
                    ctx, artifact_client, artifact, format, quiet, echo_callback
                )
            else:
                click.echo(
                    f"Artifact lineage is not supported for this URI scheme.",
                    err=True,
                )
                ctx.exit(1)
        else:
            click.echo(
                f"❌ Artifact identifier formatted incorrectly. Please try again with artifact UUID or URI.",
                err=True,
            )
            sys.exit(1)  # Exit with a non-zero status

    except AuthException as e:
        click.echo(f"❌ {str(e)}", err=True)
        ctx.exit(1)  # Exit with a non-zero status
    except ArtifactURIError as e:
        click.echo(f"❌ {str(e)}", err=True)
        ctx.exit(1)  # Exit with a non-zero status
    except Exception as e:
        click.echo(f"\n{str_exc_chain(e)}", err=True)
        click.echo(f"❌ Artifact lineage failed!", err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.option("--build-id", help="Filter artifacts by build. Provide Build ID or URL.")
@click.option("--space", help="Space name.")
@click.option(
    "--all-spaces", is_flag=True, default=False, help="Artifacts from all spaces."
)
@click.option(
    "--format",
    default="plain",
    type=click.Choice(["plain", "json"], case_sensitive=True),
    help=f"Output format: plain (default), json",
)
@click.option(
    "--username",
    "-u",
    is_flag=False,
    flag_value="default",
    help="Filter artifacts by users. Provide a username. If value is omitted, your username will be used.",
    type=str,
    required=False,
)
@click.option(
    "--checksum",
    help="Filter artifacts by checksum.",
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
    "--all", "-a", flag_value=True, default=False, help=f"List artifacts for all users."
)
@click.option(
    "--show-pending",
    is_flag=True,
    default=False,
    help="Flag to show pending artifacts.",
)
@click.option(
    "--show-archived",
    is_flag=True,
    default=False,
    help="Flag to show archived artifacts.",
)
@click.option(
    "--show-all",
    is_flag=True,
    default=False,
    help="Flag to show all artifacts, including pending and archived.",
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
    build_id: str,
    space: str,
    all_spaces: bool,
    format: str,
    username: str | None,
    checksum: str | None,
    tag: tuple | None,
    tags: str | None,
    all: bool,
    show_pending: bool,
    show_archived: bool,
    show_all: bool,
    wide: bool,
    skip_version_check: bool,
    quiet: bool,
):
    """List artifacts from a given space"""

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "error":
                reason = callback_args.get("reason", "")
                click.echo(
                    f"\n❌ Artifacts can't be listed at this moment... Reason: {reason}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case _:
                pass  # Ignore unknown events

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

    artifact_client = GBClient.Artifact(get_user_token())

    if not quiet:
        click.echo(f"{CLIPBOARD_CHAR}{PROJECT_NAME} artifact list")

    if show_all:
        show_archived = True
        show_pending = True

    id_format = parse_build_identifier(build_id)
    if build_id is not None and id_format not in ["uuid", "url"]:
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

    if all_spaces and space:
        click.echo(
            f"❌ Error: --space and --all-spaces cannot be used together",
            err=True,
        )
        sys.exit(1)

    if all and username:
        click.echo(
            f"❌ Error: --all was provided. It can't be used with the --username option."
        )
        ctx.exit(1)

    normalized_tags = []
    if bool(tag) or bool(tags):
        normalized_tags = combine_tags(tags_str=tags, tags_tuple=tag)
        if normalized_tags == None or len(normalized_tags) == 0:
            click.echo(f"❌ Given tag/tags are not correct", err=True)
            sys.exit(1)  # Exit with a non-zero status
    try:
        if quiet:
            artifacts = artifact_client.artifact_list(
                all,
                show_archived,
                show_pending,
                build_id,
                id_format,
                space,
                all_spaces,
                username,
                checksum,
                tags=normalized_tags,
                callback=None,
            )
        else:
            with tqdm(
                total=100,
                miniters=1,
                desc=f"📝 Listing artifacts",
                bar_format="{desc} [{bar}] {percentage:3.0f}%",
                ascii="-#",
                leave=False,
            ) as progress_bar:

                def update_bar(callback_event: str, callback_args: Dict):
                    steps = callback_args.get("steps", 0)
                    match callback_event:
                        case "fetching_build_id":
                            source_uri = callback_args.get("source_uri", "")
                            progress_bar.set_description(
                                f"📝 (1/2) Obtaining build ID for build URL {source_uri}."
                            )
                            progress_bar.update(n=steps)
                        case "fetched_build_id":
                            source_uri = callback_args.get("source_uri", "")
                            progress_bar.update(n=steps)
                            progress_bar.write(
                                f"📝 (1/2) Obtained build ID for build URL {source_uri}."
                            )
                        case "listing_artifacts":
                            space = callback_args.get("space", "")
                            space_name = callback_args.get("space_name", "")
                            fetched_id = callback_args.get("build_id", "")
                            progress_bar.reset(total=100)
                            progress_bar.update(steps)
                            if build_id:
                                output_listing_artifacts = f"📝 {'(2/2) ' if id_format == 'url' else ''}Listing artifacts for build id {fetched_id if fetched_id is not None else build_id}"
                            else:
                                if all_spaces:
                                    output_listing_artifacts = f"📝 {'(2/2) ' if id_format == 'url' else ''}Listing artifacts from all spaces."
                                else:
                                    output_listing_artifacts = f"📝 {'(2/2) ' if id_format == 'url' else ''}Listing artifacts from space \"{space}\" ({space_name})."
                            progress_bar.set_description(output_listing_artifacts)
                        case "listed_artifacts":
                            space = callback_args.get("space", "")
                            space_name = callback_args.get("space_name", "")
                            fetched_id = callback_args.get("build_id", "")
                            progress_bar.update(steps)
                            if build_id:
                                output_listed_artifacts = f"📝 {'(2/2) ' if id_format == 'url' else ''}Listing artifacts for build id {fetched_id if fetched_id is not None else build_id}:"
                            else:
                                if all_spaces:
                                    output_listed_artifacts = f"📝 {'(2/2) ' if id_format == 'url' else ''}Listing artifacts from all spaces."
                                else:
                                    output_listed_artifacts = f"📝 {'(2/2) ' if id_format == 'url' else ''}Listing artifacts from space \"{space}\" ({space_name})."
                            progress_bar.write(output_listed_artifacts)
                        case "error":
                            reason = callback_args.get("reason", "")
                            click.echo(
                                f"\n❌ Artifacts can't be retrieved at this moment... Reason: {reason}",
                                err=True,
                            )
                            sys.exit(1)  # Exit with a non-zero status
                        case _:
                            logger.debug(f"Callback event: {callback_event}")
                            pass

                artifacts = artifact_client.artifact_list(
                    all,
                    show_archived,
                    show_pending,
                    build_id,
                    id_format,
                    space,
                    all_spaces,
                    username,
                    checksum,
                    normalized_tags,
                    callback=update_bar,
                )

        if len(artifacts) > 0:
            if format == "plain":
                artifacts_formatted = []
                headers = ARTIFACT_LIST_HEADERS
                attribute_keys = [
                    "uuid",
                    "name",
                    "uri",
                    "type",
                    "tags",
                    "status",
                    "created_by_build_id",
                    "username",
                    "created_at",
                ]

                if all_spaces:
                    headers.insert(7, "SPACE_NAME")
                    attribute_keys.insert(7, "space_name")

                    # need to filter by user spaces (will be in gbserver eventually)
                    spaces = [
                        s["name"]
                        for s in GBClient.Space(get_user_token()).list_spaces(
                            all, False, None
                        )
                    ]
                    artifacts = [a for a in artifacts if a["space_name"] in spaces]

                if show_archived:
                    headers.append("ARCHIVED")
                    attribute_keys.append("is_archived")
                if wide:
                    name_index = headers.index("NAME")
                    headers.insert(name_index + 1, "DESCRIPTION")
                    attribute_keys.insert(name_index + 1, "description")
                    headers.insert(name_index + 2, "CHECKSUM")
                    attribute_keys.insert(name_index + 2, "checksum")
                for a in artifacts:
                    entry = []
                    for k in attribute_keys:
                        match k:
                            case "status":
                                entry.append(a.get("status", "success"))
                            case "created_at":
                                entry.append(humanize_iso_date(a["created_at"]))
                            case "description":
                                entry.append(a.get("description", ""))
                            case "checksum":
                                entry.append(a.get("checksum", ""))
                            case _:
                                entry.append(a[k])

                    artifacts_formatted.append(entry)

                table = Table(title="Artifacts", padding=(0, 1))
                for header in headers:
                    # Set width constraints for description column only
                    if header in ["DESCRIPTION", "URI"]:
                        table.add_column(header, width=25, overflow="fold")
                    else:
                        table.add_column(header, overflow="fold")

                for row in artifacts_formatted:
                    table.add_row(*[str(cell) for cell in row])

                console = Console()
                console.print(table)
            else:
                artifacts_output = json.dumps(artifacts)
                click.echo(artifacts_output)
    except Exception as e:
        click.echo(str_exc_chain(e), err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.argument(
    "artifact", required=False
)  # not required until deprecated option is removed
@click.option(
    "--artifact-id",
    help="""
    DEPRECATED - please provide artifact id as the first argument. This option will be removed in future versions. \n\n
    
    Artifact id to be download. Provide artifact UUID or URI
    """,
)
@click.option(
    "-d",
    "--directory",
    default=".",
    help="Output directory name. Default is current directory)",
)
@click.option(
    "--format",
    default="json",
    type=click.Choice(["json", "jsonl", "csv", "parquet"], case_sensitive=True),
    help="""Specifies the output file format. Options include:\n
    - 'jsonl': JSON Lines format, where each line is a separate JSON object.\n
    - 'json' (default): Standard JSON format with a single structured object or array.\n
    - 'csv': Comma-separated values, useful for tabular data.\n
    - 'parquet': Columnar storage format optimized for big data processing.
    """,
)
@click.option(
    "-t",
    "--type",
    type=click.Choice(["model", "dataset", "table", "fileset"], case_sensitive=True),
    help="""
    DEPRECATED - artifact type is retrieved from the artifact object. This option will be removed in future versions. \n\n

    Specifies the type of input to be processed. Options: dataset, model, table (default), fileset
    """,
)
@click.option(
    "--file-label",
    help="""
    DEPRECATED - fileset label is retrieved from the artifact object. This option will be removed in future versions. \n\n

    Fileset label.
    """,
)
@click.option(
    "--version",
    help="""
    DEPRECATED - fileset version is retrieved from the artifact object. This option will be removed in future versions. \n\n

    Fileset version.
    """,
)
@click.option("--space", help="Space name.")
@click.option(
    "-r",
    "--revision",
    default="main",
    help="Git revision/branch name for HuggingFace artifacts (default: main)",
)
@click.pass_context
@common_options
def download(
    ctx,
    artifact: str,
    artifact_id: str,
    directory: str,
    format: str,
    type: str,
    file_label: str,
    version: str,
    space: str = None,
    revision: str = "main",
    skip_version_check: bool = False,
    quiet: bool = False,
):
    """
    Access artifact download

    Provide artifact id as an argument as the UUID or URI.\n
    The --artifact-id option is deprecated.
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

    if artifact and artifact_id:
        click.echo(
            "Error: Please only provide the artifact identifier as an argument.",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status

    if artifact:
        artifact_id = artifact
    elif artifact_id:
        if not quiet:
            click.echo(
                "Deprecated: the --artifact-id option is deprecated. Please provide artifact ids as an argument.",
                err=True,
            )
    else:
        click.echo(
            "Error: --artifact-id parameter is required.",
            err=True,
        )
        ctx.exit(1)  # Exit with a non-zero status

    # Deprecated options
    if type:
        if not quiet:
            click.echo(
                "Deprecated: the --type option is deprecated. This value will be ignored.",
                err=True,
            )
        type = None

    if file_label:
        if not quiet:
            click.echo(
                "Deprecated: the --file_label option is deprecated. This value will be ignored.",
                err=True,
            )
        file_label = None

    if version:
        if not quiet:
            click.echo(
                "Deprecated: the --version option is deprecated. This value will be ignored.",
                err=True,
            )
        version = None

    artifact_client = GBClient.Artifact(get_user_token())

    def echo_callback(callback_event: str, callback_args: Dict):
        if callback_event == "error":
            reason = callback_args.get("reason", "")
            click.echo(
                f"❌ Artifact download can't be executed at this moment... Reason: {reason}",
                err=True,
            )
            sys.exit(1)  # Exit with a non-zero status
        else:
            pass  # Ignore unknown events

    try:
        total_steps = 3
        namespace = None
        directory_path = Path(directory).expanduser()
        if not directory_path.exists():
            directory_path.mkdir(parents=True, exist_ok=True)
        id_format = parse_artifact_identifier(artifact_id)

        if id_format in ["uuid", "uri"]:
            if not quiet:
                click.echo(f"(1/3) Obtaining artifact..")

            artifact = (
                artifact_client.fetch_artifact(artifact_id, callback=echo_callback)
                if id_format == "uuid"
                else artifact_client.fetch_artifact_uri(
                    artifact_id, callback=echo_callback
                )
            )

            if not artifact:
                click.echo(f"\n❌ Error: artifact does not exist.", err=True)
                ctx.exit(1)

            # Detect if artifact is in HuggingFace or Lakehouse
            artifact_uri = artifact["uri"]
            if artifact_uri.startswith("hf://"):
                # === HF download path ===
                hf_token_value = hf_token()
                if not hf_token_value:
                    click.echo(
                        "❌ HuggingFace token not found. Please set HF_TOKEN environment variable.",
                        err=True,
                    )
                    ctx.exit(1)

                org, name, artifact_type, domain = parse_hf_uri(artifact_uri)
                repo_id = f"{org}/{name}"

                if not quiet:
                    click.echo(f"Downloading {artifact_type} from HuggingFace...")
                    click.echo(f"  Repository: {repo_id}")
                    click.echo(f"  Revision: {revision}")
                    click.echo(f"  Directory: {directory_path}")

                def hf_echo_callback(callback_event: str, callback_args: Dict):
                    if callback_event == "error":
                        reason = callback_args.get("reason", "")
                        click.echo(
                            f"❌ HuggingFace artifact download failed... Reason: {reason}",
                            err=True,
                        )
                        sys.exit(1)

                result = (
                    artifact_client.download_hf_artifact(
                        hf_token_value,
                        repo_id,
                        artifact_type,
                        str(directory_path),
                        revision,
                        hf_echo_callback,
                    )
                    if quiet
                    else execute_with_spinner(
                        artifact_client.download_hf_artifact,
                        hf_token_value,
                        repo_id,
                        artifact_type,
                        str(directory_path),
                        revision,
                        hf_echo_callback,
                    )
                )

                snapshot_dir = result.get("download_dir")
                file_path = f"{snapshot_dir}/artifact.origin"
                artifact_client.save_origin(file_path, artifact)

                if not quiet:
                    click.echo(f"\n✅ Download completed successfully!")
                    click.echo(f"  Repository: {result.get('repo_id')}")
                    click.echo(f"  Type: {result.get('artifact_type')}")
                    click.echo(f"  Files downloaded: {result.get('file_count')}")
                    click.echo(
                        f"  Total size: {result.get('total_size', 0) / (1024 ** 2):.2f} MB"
                    )
                    click.echo(f"  Location: {result.get('download_dir')}")

                if format == "json":
                    click.echo(json.dumps(result, indent=2, default=str))

                return  # done, skip LH path below

            # === LH-specific logic (grouped for future disable) ===
            decoded_artifact = decode_uri(artifact_uri)
            namespace = decoded_artifact.namespace
            table_name = decoded_artifact.table_name
            type = decoded_artifact.type

            if type == "dataset":
                selected_type = "dataset"

            elif type == "model":
                selected_type = "model"
                model_label = decoded_artifact.model_label
                model_revision = decoded_artifact.model_revision

            elif type == "fileset":
                selected_type = "fileset"
                file_label = decoded_artifact.fileset_label
                version = decoded_artifact.fileset_version

            else:
                selected_type = "table"

        else:
            click.echo(
                f"❌ Artifact identifier formatted incorrectly. Please try again with artifact UUID or URI.",
                err=True,
            )
            sys.exit(1)

        if not quiet:
            click.echo(f"(2/3) Obtaining Lakehouse token.")

        lh_token = GBClient.Auth.lakehouse_user_token(artifact_client.github_token)
        if not lh_token:
            return
        if not quiet:
            click.echo(f"Lakehouse token obtained successfully!")

        if not quiet:
            click.echo(
                f"({total_steps}/{total_steps}) Download artifact to folder '{directory_path}'"
            )

        if table_name != None:
            if selected_type == "dataset" or selected_type == "table":
                (
                    artifact_client.download_table(
                        lh_token,
                        namespace,
                        table_name,
                        directory_path,
                        format,
                        space,
                        echo_callback,
                    )
                    if quiet
                    else execute_with_spinner(
                        artifact_client.download_table,
                        lh_token,
                        namespace,
                        table_name,
                        directory_path,
                        format,
                        space,
                        echo_callback,
                    )
                )

                file_name = f"{table_name}.{format}.origin"
                file_path = f"{directory_path}/{file_name}"
                artifact_client.save_origin(file_path, artifact)

                if not quiet:
                    click.echo(
                        f"\n\n✅ Artifact/{selected_type.title()} {id_format} '{artifact_id}' was successfully downloaded to '{directory_path}' directory."
                    )
            elif selected_type == "model":
                (
                    artifact_client.download_model(
                        lh_token,
                        namespace,
                        table_name,
                        model_label,
                        model_revision,
                        directory_path,
                        space,
                        echo_callback,
                    )
                    if quiet
                    else execute_with_spinner(
                        artifact_client.download_model,
                        lh_token,
                        namespace,
                        table_name,
                        model_label,
                        model_revision,
                        directory_path,
                        space,
                        echo_callback,
                    )
                )
                file_name = f"{get_model_subforlder(model_label, model_revision)}/artifact.origin"
                file_path = f"{directory_path}/{file_name}"
                artifact_client.save_origin(file_path, artifact)

                if not quiet:
                    click.echo(
                        f"\n\n✅ Artifact/{selected_type.title()} {id_format} '{artifact_id}' was successfully downloaded to '{directory_path}' directory."
                    )
            elif selected_type == "fileset":
                (
                    artifact_client.download_fileset(
                        lh_token,
                        namespace,
                        table_name,
                        file_label,
                        version,
                        directory_path,
                        space,
                        echo_callback,
                    )
                    if quiet
                    else execute_with_spinner(
                        artifact_client.download_fileset,
                        lh_token,
                        namespace,
                        table_name,
                        file_label,
                        version,
                        directory_path,
                        space,
                        echo_callback,
                    )
                )
                file_name = (
                    f"{get_fileset_subforlder( file_label, version)}/artifact.origin"
                )
                file_path = f"{directory_path}/{file_name}"
                artifact_client.save_origin(file_path, artifact)

                if not quiet:
                    click.echo(
                        f"\n\n✅ Artifact/{selected_type.title()} {id_format} '{artifact_id}' was successfully downloaded to '{directory_path}' directory."
                    )
            else:
                click.echo(
                    f"\n❌ Artifact type is unknown. Specify the artifact type with -t option.",
                    err=True,
                )
                ctx.exit(1)  # Exit with a non-zero status

        else:
            click.echo(
                f"❌ There is not content for {type} with id: {artifact_id}. Please provide --type argument.",
                err=True,
            )
            # === end LH-specific logic ===

    except AuthException as e:
        click.echo(f"❌ {str(e)}", err=True)
        ctx.exit(1)  # Exit with a non-zero status
    except click.exceptions.Exit as e:
        raise e
    except Exception as e:
        if "Error downloading file:" in str(e):
            click.echo(f"\n{str(e)}", err=True)

        else:
            click.echo(f"\n{str_exc_chain(e)}", err=True)
        click.echo(f"❌ Artifact download from Lakehouse failed!", err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.argument(
    "artifact-id",
    required=True,
)
@click.option(
    "--format",
    "format",
    default="simple",
    type=click.Choice(["simple", "json"], case_sensitive=True),
    help="Output format: simple (default), json",
)
@common_options
def archive(ctx, artifact_id: str, format: str, skip_version_check: bool, quiet: bool):
    """
    Archive an artifact.

    Provide an artifact UUID or URI.
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

    artifact_client = GBClient.Artifact(get_user_token())

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "error":
                reason = callback_args.get("reason", "")
                click.echo(
                    f"\n❌ Artifact can't be archived at this moment... Reason: {reason}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case _:
                pass  # Ignore unknown events

    try:
        id_format = parse_artifact_identifier(artifact_id)

        if id_format == "uri":
            if not quiet:
                click.echo("(1/2) Getting artifact uuid from the uri.")

            artifact_uuid = artifact_client.fetch_artifact_uri(artifact_id)["uuid"]

            if not quiet:
                click.echo(f"Artifact uuid obtained successfully!")
        elif id_format == "uuid":
            artifact_uuid = artifact_id
        else:
            click.echo(
                f"❌ Artifact identifier formatted incorrectly. Please try again.",
                err=True,
            )
            sys.exit(1)  # Exit with a non-zero status

        if not quiet:
            click.echo(
                f"{'(2/2) ' if id_format == 'uri' else ''}Archiving artifact with uuid {artifact_uuid}"
            )
        resp = (
            artifact_client.archive_artifact(artifact_uuid, True, echo_callback)
            if quiet
            else execute_with_spinner(
                artifact_client.archive_artifact, artifact_uuid, True, echo_callback
            )
        )

        if resp["artifact"]["is_archived"] == True:
            if not quiet:
                click.echo(f"\n✅ Artifact archive successful!")
            if format == "json":
                click.echo(
                    json.dumps({"artifact_id": artifact_uuid, "is_archived": True})
                )
        else:
            click.echo(f"\n❌ Artifact archive failed. Please try again.", err=True)
            sys.exit(1)  # Exit with a non-zero status

    except AuthException as e:
        click.echo(f"❌ {str(e)}", err=True)
        ctx.exit(1)  # Exit with a non-zero status
    except Exception as e:
        click.echo(f"\n{str_exc_chain(e)}", err=True)
        click.echo(f"❌ Artifact archive from Lakehouse failed!", err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.argument(
    "artifact-id",
    required=True,
)
@click.option(
    "--format",
    "format",
    default="simple",
    type=click.Choice(["simple", "json"], case_sensitive=True),
    help="Output format: simple (default), json",
)
@common_options
def unarchive(
    ctx, artifact_id: str, format: str, skip_version_check: bool, quiet: bool
):
    """
    Unarchive an artifact.

    Provide an artifact UUID or URI.
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

    artifact_client = GBClient.Artifact(get_user_token())

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "error":
                reason = callback_args.get("reason", "")
                click.echo(
                    f"\n❌ Artifact can't be unarchived at this moment... Reason: {reason}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case _:
                pass  # Ignore unknown events

    try:
        id_format = parse_artifact_identifier(artifact_id)

        if id_format == "uri":
            if not quiet:
                click.echo("(1/2) Getting artifact uuid from the uri.")
            artifact_uuid = get_artifact_uuid(artifact_client.github_token, artifact_id)

            if not quiet:
                click.echo(f"Artifact uuid obtained successfully!")
        elif id_format == "uuid":
            artifact_uuid = artifact_id
        else:
            click.echo(
                f"❌ Artifact identifier formatted incorrectly. Please try again.",
                err=True,
            )
            sys.exit(1)  # Exit with a non-zero status

        if not quiet:
            click.echo(
                f"{'(2/2) ' if id_format == 'uri' else ''}Unarchiving artifact with uuid {artifact_uuid}"
            )
        resp = (
            artifact_client.archive_artifact(artifact_uuid, False, echo_callback)
            if quiet
            else execute_with_spinner(
                artifact_client.archive_artifact, artifact_uuid, False, echo_callback
            )
        )

        if resp["artifact"]["is_archived"] == False:
            if not quiet:
                click.echo(f"\n✅ Artifact unarchive successful!")
            if format == "json":
                click.echo(
                    json.dumps({"artifact_id": artifact_uuid, "is_archived": False})
                )
        else:
            click.echo(f"\n❌ Artifact unarchive failed. Please try again.", err=True)
            sys.exit(1)  # Exit with a non-zero status

    except AuthException as e:
        click.echo(f"❌ {str(e)}", err=True)
        ctx.exit(1)  # Exit with a non-zero status
    except Exception as e:
        click.echo(f"\n{str_exc_chain(e)}", err=True)
        click.echo(f"❌ Artifact unarchive from Lakehouse failed!", err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.argument("artifact_id")
@click.option("--space-to", required=True, help="Destination space name.")
@click.option(
    "--artifact-name", required=False, help="New name for the copied artifact."
)
@common_options
def copy(
    ctx,
    artifact_id: str,
    space_to: str,
    artifact_name: str,
    skip_version_check: bool,
    quiet: bool,
):
    """
    Copy an artifact from one space to another. Currently, only model artifacts are supported.

    artifact_id: The source artifact ID or URI.
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

    artifact_client = GBClient.Artifact(get_user_token())

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "error":
                reason = callback_args.get("reason", "")
                click.echo(
                    f"\n❌ Artifact can't be copied at this moment... Reason: {reason}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case _:
                pass  # Ignore unknown events

    try:
        from lakehouse.core import UnauthorizedException

        id_format = parse_artifact_identifier(artifact_id)
        if id_format in ["uuid", "uri"]:
            if not quiet:
                click.echo(f"(1/4) Obtaining artifact..")
            artifact = (
                artifact_client.fetch_artifact(artifact_id, callback=echo_callback)
                if id_format == "uuid"
                else artifact_client.fetch_artifact_uri(
                    artifact_id, callback=echo_callback
                )
            )

            if not artifact:
                click.echo(
                    f"❌ Artifact identifier formatted incorrectly. Please try again.",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status

            # --artifact-name is optional, the default is the current artifact name before copy.
            artifact_name = artifact_name if artifact_name != None else artifact["name"]
            description = artifact["description"]
            checksum = artifact["checksum"]
            origin_uris = artifact["origin_uris"]
            tags = artifact["tags"]
            status = artifact["status"]
            certified_no_restrictions = artifact["certified_no_restrictions"]
            uri = artifact["uri"]
            table_name = artifact_name

            # Detect artifact source: HF or LH

            is_hf_artifact = uri.startswith("hf://")

        if is_hf_artifact:
            click.echo(
                f"❌ Copy is not supported for HuggingFace artifacts.",
                err=True,
            )
            sys.exit(1)  # Exit with a non-zero status

        # === Lakehouse-specific copy logic (can be disabled together) ===
        decoded_artifact = decode_uri(uri)
        type = decoded_artifact.type

        if type != "model":
            click.echo(f"\n❌ Only model artifacts are supported for copy.", err=True)
            sys.exit(1)  # Exit with a non-zero status

        if bool(origin_uris) or not certified_no_restrictions:
            certified_no_restrictions = click.confirm(
                "Origin not found. Do you certify this artifact wasn't created with a restricted-use model?",
                default=False,
            )

        if not certified_no_restrictions:
            click.echo(
                f"\n❌ Only artifacts created with models under non-restricted use can be copied.",
                err=True,
            )
            sys.exit(1)  # Exit with a non-zero status

        if not quiet:
            click.echo(f"(2/4) Obtaining Lakehouse token.")

        lh_token = GBClient.Auth.lakehouse_token_for_space(
            artifact_client.github_token, space=space_to, callback=echo_callback
        )
        if not lh_token:
            return
        if not quiet:
            click.echo(f"Lakehouse token obtained successfully!")

        if not quiet:
            click.echo(
                f"(3/4) Copying artifact to Lakehouse. This may take a while, please wait..."
            )

        revision = decoded_artifact.model_revision
        namespace = decoded_artifact.namespace
        model_label = decoded_artifact.model_label

        source_table_name = (
            get_artifact_formatted_name(decoded_artifact).split(".")[-1]
            if uri != None
            else LAKEHOUSE_MODEL_SHARED_TABLE
        )
        source_table = (
            source_table_name
            if source_table_name == LAKEHOUSE_MODEL_TABLE
            else LAKEHOUSE_MODEL_SHARED_TABLE
        )
        target_table = LAKEHOUSE_MODEL_TABLE

        resp = artifact_client.artifact_copy(
            lh_token,
            namespace,
            source_table,
            space_to,
            model_label,
            revision,
            echo_callback,
        )

        if (
            resp != None
            and resp.get("copy_response")
            and resp.get("copy_response").status == "SUCCESS"
        ):
            target_table = resp.get("target_table")
            if not quiet:
                click.echo(f"\n✅ Artifact copied successful!")
                click.echo(f"(4/4) Registering the artifact.")
        else:
            click.echo(f"\n❌ Artifact copy failed. Please try again.", err=True)
            sys.exit(1)  # Exit with a non-zero status
        try:

            response = (
                artifact_client.register_artifact(
                    artifact_name=artifact_name,
                    description=description,
                    checksum=checksum,
                    type=type,
                    label=model_label,
                    space=space_to,
                    table=target_table,
                    revision=revision,
                    tags=tags,
                    status=status,
                    origin_uris=origin_uris,
                    certified_no_restrictions=certified_no_restrictions,
                    callback=echo_callback,
                )
                if quiet
                else execute_with_spinner(
                    artifact_client.register_artifact,
                    artifact_name=artifact_name,
                    description=description,
                    type=type,
                    space=space_to,
                    label=model_label,
                    table=target_table,
                    revision=revision,
                    tags=tags,
                    status=status,
                    origin_uris=origin_uris,
                    certified_no_restrictions=certified_no_restrictions,
                    callback=echo_callback,
                )
            )
        except Exception as e:
            click.echo(f"\n❌ Artifact register failed! {str(e)}", err=True)
            sys.exit(1)

        if not quiet:
            click.echo(
                f"\n✅ Artifact/{type.title()} '{artifact_name}' was successfully copied with uuid {response['uuid']} uri {response['uri']}"
            )
        # === End Lakehouse-specific logic ===

    except AuthException as e:
        click.echo(f"\n❌ {str(e)}", err=True)
        ctx.exit(1)  # Exit with a non-zero status
    except ValueError as e:
        click.echo(f"\n❌ Artifact copy failed! {str(e)}", err=True)
        ctx.exit(1)
    except UnauthorizedException as e:
        click.echo(f"\n❌ {str(e)}", err=True)
        ctx.exit(1)  # Exit with a non-zero status
    except Exception as e:
        click.echo(f"\n{str_exc_chain(e)}", err=True)
        click.echo(f"\n❌ Artifact copy failed! ", err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.argument(
    "artifact-id",
    required=True,
)
@click.option(
    "--format",
    "format",
    default="simple",
    type=click.Choice(["simple", "json"], case_sensitive=True),
    help="Output format: simple (default), json",
)
@common_options
def describe(ctx, artifact_id: str, format: str, skip_version_check: bool, quiet: bool):
    """
    Show a detailed description and tags of the specified artifact.

    Provide an artifact UUID or URI.
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

    artifact_client = GBClient.Artifact(get_user_token())

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "error":
                reason = callback_args.get("reason", "")
                click.echo(
                    f"\n❌ Artifact can't be retrieved at this moment... Reason: {reason}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case _:
                pass  # Ignore unknown events

    try:
        id_format = parse_artifact_identifier(artifact_id)

        if id_format == "uri":
            if not quiet:
                click.echo("(1/2) Getting artifact uuid from the uri.")

            artifact_uuid = artifact_client.fetch_artifact_uri(artifact_id)["uuid"]

            if not quiet:
                click.echo(f"Artifact uuid obtained successfully!")
        elif id_format == "uuid":
            artifact_uuid = artifact_id
        else:
            click.echo(
                f"❌ Artifact identifier formatted incorrectly. Please try again.",
                err=True,
            )
            sys.exit(1)  # Exit with a non-zero status

        if not quiet:
            click.echo(
                f"{'(2/2) ' if id_format == 'uri' else ''}Fetching artifact with uuid {artifact_uuid}"
            )
        artifact = (
            artifact_client.fetch_artifact(artifact_uuid, echo_callback)
            if quiet
            else execute_with_spinner(
                artifact_client.fetch_artifact, artifact_uuid, echo_callback
            )
        )

        if artifact:
            if format == "json":
                click.echo(
                    json.dumps(
                        {
                            "name": artifact["name"],
                            "description": artifact["description"],
                            "checksum": artifact["checksum"],
                            "tags": artifact["tags"],
                            "status": artifact["status"],
                        }
                    )
                )
            else:
                name = artifact["name"]
                description = artifact["description"] or "No description provided"
                checksum = artifact["checksum"] or "No checksum provided"
                tags = artifact["tags"] or ""
                status = artifact["status"] or "No status provided"
                click.echo("\n")
                click.echo("-" * 60)
                click.echo(f"NAME       : {name}")
                click.echo(f"DESCRIPTION: {description}")
                click.echo(f"CHECKSUM: {checksum}")
                click.echo(f"TAGS: {tags}")
                click.echo(f"STATUS: {status}")
                click.echo("-" * 60)

        else:
            click.echo(f"\n❌ Artifact wat not found. Please try again.", err=True)
            sys.exit(1)  # Exit with a non-zero status

    except AuthException as e:
        click.echo(f"❌ {str(e)}", err=True)
        ctx.exit(1)  # Exit with a non-zero status
    except Exception as e:
        click.echo(f"\n{str_exc_chain(e)}", err=True)
        click.echo(f"❌ Artifact failed!", err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.argument(
    "artifact-id",
    required=True,
)
@click.option(
    "--format",
    "format",
    default="simple",
    type=click.Choice(["simple", "json"], case_sensitive=True),
    help="Output format: simple (default), json",
)
@common_options
def checksum(ctx, artifact_id: str, format: str, skip_version_check: bool, quiet: bool):
    """
    Show the checksum of the specified artifact.

    Provide an artifact UUID or URI.
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

    artifact_client = GBClient.Artifact(get_user_token())

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "error":
                reason = callback_args.get("reason", "")
                click.echo(
                    f"\n❌ Artifact can't be retrieved at this moment... Reason: {reason}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case _:
                pass  # Ignore unknown events

    try:
        id_format = parse_artifact_identifier(artifact_id)
        if id_format in ["uuid", "uri"]:
            if not quiet:
                click.echo(f"Obtaining artifact..")

            artifact = (
                artifact_client.fetch_artifact(artifact_id, callback=echo_callback)
                if id_format == "uuid"
                else artifact_client.fetch_artifact_uri(
                    artifact_id, callback=echo_callback
                )
            )

            if artifact:
                if format == "json":
                    click.echo(
                        json.dumps(
                            {
                                "artifact_id": artifact_id,
                                "checksum": artifact["checksum"],
                            }
                        )
                    )
                else:
                    checksum = artifact["checksum"] or "No checksum provided"
                    click.echo(f"CHECKSUM: {checksum}")

            else:
                click.echo(f"\n❌ Artifact wat not found. Please try again.", err=True)
                sys.exit(1)  # Exit with a non-zero status

        else:
            click.echo(
                f"❌ Artifact identifier formatted incorrectly. Please try again with artifact UUID or URI.",
                err=True,
            )
            sys.exit(1)  # Exit with a non-zero status

    except AuthException as e:
        click.echo(f"❌ {str(e)}", err=True)
        ctx.exit(1)  # Exit with a non-zero status
    except ArtifactURIError as e:
        click.echo(f"❌ {str(e)}", err=True)
        ctx.exit(1)  # Exit with a non-zero status
    except Exception as e:
        click.echo(f"\n{str_exc_chain(e)}", err=True)
        click.echo(f"❌ Artifact failed!", err=True)
        ctx.exit(1)  # Exit with a non-zero status


@cli.command()
@click.pass_context
@click.argument(
    "artifact-id",
    required=True,
)
@click.option(
    "--description",
    help="Artifact description. Set a description for the artifact. This can be used to provide additional context or details. If the text contains spaces, enclose it in quotes.",
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
    "--status",
    type=click.Choice(
        ["pending", "success", "failed", "cancelled"], case_sensitive=False
    ),
    help="Status of the artifact: pending, success, failed, cancelled",
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
    artifact_id: str,
    description: str,
    tag: tuple,
    tags: str,
    status: str,
    append: bool,
    format: str,
    skip_version_check: bool,
    quiet: bool,
):
    """
    Update a detailed description of the specified artifact or tags.
    Provide an artifact UUID or URI.
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

    artifact_client = GBClient.Artifact(get_user_token())

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "error":
                reason = callback_args.get("reason", "")
                click.echo(
                    f"\n❌ Artifact can't be updated at this moment... Reason: {reason}",
                    err=True,
                )
                sys.exit(1)  # Exit with a non-zero status
            case _:
                pass  # Ignore unknown events

    try:
        id_format = parse_artifact_identifier(artifact_id)

        if id_format == "uri":
            if not quiet:
                click.echo("(1/3) Getting artifact uuid from the uri.")

            artifact_uuid = artifact_client.fetch_artifact_uri(artifact_id)["uuid"]

            if not quiet:
                click.echo(f"Artifact uuid obtained successfully!")
        elif id_format == "uuid":
            artifact_uuid = artifact_id
        else:
            click.echo(
                f"❌ Artifact identifier formatted incorrectly. Please try again.",
                err=True,
            )
            sys.exit(1)  # Exit with a non-zero status

        if not quiet:
            click.echo(
                f"{'(2/3) ' if id_format == 'uri' else '(1/2) '} Fetching artifact with uuid {artifact_uuid}"
            )
        artifact = (
            artifact_client.fetch_artifact(artifact_uuid, echo_callback)
            if quiet
            else execute_with_spinner(
                artifact_client.fetch_artifact, artifact_uuid, echo_callback
            )
        )

        if artifact:
            if not tag and tags is None and description is None and not status:
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
                        artifact_client.github_token,
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
                click.echo(
                    f"{'(3/3) ' if id_format == 'uri' else '(2/2) '} Updating artifact with uuid {artifact_uuid}"
                )
            artifact = (
                artifact_client.update_artifact(
                    artifact_id=artifact_uuid,
                    description=description,
                    tags=normalized_tags,
                    status=status,
                    append=append,
                    isUpdate=True,
                    callback=echo_callback,
                )
                if quiet
                else execute_with_spinner(
                    artifact_client.update_artifact,
                    artifact_id=artifact_uuid,
                    description=description,
                    tags=normalized_tags,
                    status=status,
                    append=append,
                    isUpdate=True,
                    callback=echo_callback,
                )
            )

            if not quiet and artifact:
                click.echo(f"\n✅ Artifact was updated sucessfully!")
                name = artifact["name"]
                description = artifact["description"] or "No description provided"
                checksum = artifact["checksum"] or "No checksum provided"
                tags = artifact["tags"] or ""
                status = artifact["status"] or "No status provided"
                click.echo("-" * 60)
                click.echo(f"NAME       : {name}")
                click.echo(f"DESCRIPTION: {description}")
                click.echo(f"CHECKSUM: {checksum}")
                click.echo(f"TAGS: {tags}")
                click.echo(f"STATUS: {status}")
                click.echo("-" * 60)
            if format == "json" and artifact:
                click.echo(
                    json.dumps(
                        {
                            "name": artifact["name"],
                            "description": artifact["description"],
                            "checksum": artifact["checksum"],
                            "tags": artifact["tags"],
                            "status": artifact["status"],
                        }
                    )
                )

        else:
            click.echo(f"\n❌ Artifact wat not found. Please try again.", err=True)
            sys.exit(1)  # Exit with a non-zero status

    except AuthException as e:
        click.echo(f"❌ {str(e)}", err=True)
        ctx.exit(1)  # Exit with a non-zero status
    except Exception as e:
        click.echo(f"\n{str_exc_chain(e)}", err=True)
        click.echo(f"❌ Artifact failed!", err=True)
        ctx.exit(1)  # Exit with a non-zero status
