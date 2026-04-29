import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import click
from tqdm import tqdm

from gbcli.client.client import GBClient
from gbcli.utils.cli_config import get_local_gb_config
from gbcli.utils.gbconstants import DMF_URL, getenv_boolean
from gbcli.utils.utils import parse_markdown_str
from llmbsub.utils.llmbsub_constants import (
    BUILD_YAML_TO_STDOUT,
    FEATURE_FLAGS,
    REGISTRATION_FEATURE_FLAG,
    UPLOAD_FEATURE_FLAG,
)
from llmbsub.utils.scriptutil import (
    create_shell_script,
    create_symlink,
    generate_build_yaml,
    generate_metadata_log,
    get_inputs_table,
    get_outputs_table,
    parse_inputs,
    parse_inputs_yaml_file,
    parse_outputs,
    parse_outputs_yaml_file,
    parse_text_file,
)

logger = logging.getLogger(__name__)


def submit_upload_job(
    inputs_to_upload: list,
    space: str,
    log_dir: Path,
    run_identifier: str,
    parent_job_id: str,
    dry_run: bool = False,
) -> Optional[str]:
    """
    Submit a CPU-only bsub job to upload input datasets to Lakehouse.

    This enables input lineage tracking by uploading input filesets to DMF.

    Args:
        inputs_to_upload: List of inputs with 'path', 'target_uri', and 'checksum' keys
        space: LLM.build space name
        log_dir: Base log directory
        run_identifier: UUID for this run
        parent_job_id: Job ID of the main llmbsub job
        dry_run: If True, don't actually submit the job

    Returns:
        Job ID of the upload job, or None if submission failed
    """
    from gbcli.utils.cli_config import get_local_build_cache

    if not inputs_to_upload:
        logger.debug("No inputs to upload")
        return None

    # Create upload job directory
    upload_dir = Path(get_local_build_cache()) / "llmbsub" / "upload"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Write input paths to file
    inputs_file = upload_dir / f"inputs-{run_identifier}.jsonl"
    with open(inputs_file, "w") as f:
        for input in inputs_to_upload:
            json_line = json.dumps(input)  # Serialize the dictionary to a JSON string
            f.write(json_line + "\n")

    # Prepare log file path
    upload_log_dir = log_dir / f"{run_identifier}"
    upload_log_dir.mkdir(parents=True, exist_ok=True)
    upload_log_file = upload_log_dir / "upload.log"

    # Get wrapper script path
    wrapper_script = Path(__file__).parent / "utils" / "upload_wrapper.sh"

    # Build bsub command for CPU-only job
    bsub_cmd = [
        "bsub",
        "-G",
        "grp_granite_dot_build",
        "-q",
        "normal",  # Use normal queue (CPU only)
        "-n",
        "1",  # Single CPU
        "-R",
        "rusage[mem=4096]",  # 4GB memory
        "-J",
        f"upload_{parent_job_id}",
        "-o",
        str(upload_log_file),
        "-e",
        str(upload_log_file),
        str(wrapper_script),
        str(inputs_file),
        space,
        str(upload_log_file),
    ]

    if dry_run:
        logger.info(f"Dry-run: would submit upload job: {' '.join(bsub_cmd)}")
        return "dry-run-upload-job"

    try:
        result = subprocess.run(bsub_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Extract job ID from bsub output
            job_info = re.findall(r"\<(.*?)\>", result.stdout)
            if job_info:
                upload_job_id = job_info[0]
                logger.info(f"Upload job submitted: {upload_job_id}")
                return upload_job_id
            else:
                logger.warning(
                    f"Could not extract job ID from bsub output: {result.stdout}"
                )
        else:
            logger.warning(f"Upload job submission failed: {result.stderr}")
    except Exception as e:
        logger.warning(f"Failed to submit upload job: {e}")

    return None


def run_upload_script(
    inputs_to_upload: list,
    space: str,
    log_dir: Path,
    run_identifier: str,
    parent_job_id: str,
    dry_run: bool = False,
    verbose: bool = False,
) -> Optional[Path]:
    """
    Run artifact push script as a background subprocess.

    The upload runs in the background so llmbsub can complete immediately.
    Logs are written to a dedicated upload.log file in the run's log directory.

    Args:
        inputs_to_upload: List of inputs with 'path', 'target_uri', and 'checksum' keys
        space: LLM.build space name
        log_dir: Base log directory
        run_identifier: UUID for this run
        parent_job_id: Job ID of the main llmbsub job
        dry_run: If True, don't actually run the upload

    Returns:
        Path to the upload log file, or None if no inputs to upload
    """
    from gbcli.utils.cli_config import get_local_build_cache

    if not inputs_to_upload:
        logger.debug("No inputs to upload")
        return None

    # Create upload job directory
    upload_dir = Path(get_local_build_cache()) / "llmbsub" / "upload"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Write input paths to file
    inputs_file = upload_dir / f"inputs-{run_identifier}.jsonl"
    with open(inputs_file, "w") as f:
        for input_item in inputs_to_upload:
            json_line = json.dumps(input_item)
            f.write(json_line + "\n")

    # Prepare log file path in the run's log directory
    upload_log_dir = log_dir / f"{run_identifier}"
    upload_log_dir.mkdir(parents=True, exist_ok=True)
    upload_log_file = upload_log_dir / "upload.log"

    # Build command - calls input_upload_service.py directly as a module
    # Use sys.executable to ensure we use the same Python interpreter
    upload_service_module = (
        Path(__file__).parent / "services" / "input_upload_service.py"
    )
    command_str = [
        sys.executable,
        str(upload_service_module),
        str(inputs_file),
        space,
        str(upload_log_file),
    ]

    # Print the command and log file location
    click.echo(f"  Upload command: {' '.join(command_str)}")
    click.echo(f"  Upload log file: {upload_log_file}")

    if dry_run:
        click.echo("  (dry-run: upload not started)")
        return upload_log_file

    try:
        # Open log file for writing subprocess output
        # Note: Don't use context manager - the subprocess needs the file handle to remain open
        log_file = open(upload_log_file, "w")

        # Inherit the current environment (including PYTHONPATH, virtual env, etc.)
        env = os.environ.copy()

        # Start subprocess in background, redirecting stdout/stderr to log file
        subprocess.Popen(
            command_str,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,  # Pass current environment to subprocess
            start_new_session=True,  # Detach from parent process
            close_fds=False,  # Keep log file handle open for subprocess
        )
        # Note: log_file intentionally not closed - subprocess will inherit and use it

        click.echo("  Upload started in background")
        return upload_log_file

    except Exception as e:
        logger.warning(f"Failed to start upload job: {e}")
        click.echo(f"  ⚠️  Failed to start upload: {e}")
        return None


def submit_bsub(
    llmbin: Tuple,
    llmbout: Tuple,
    llmb_output_dir: str,
    llmbin_from_file: str,
    llmbin_from_yaml: str,
    llmb_default_artifact_type: str,
    bsub_args: Tuple,
    llmb_command: Tuple,
    dry_run: bool,
    llmbconfig: Optional[str] = None,
    llmb_project: Optional[str] = None,
    llmb_output_dir_from_yaml: Optional[str] = None,
    space: Optional[str] = None,
    verbose: bool = False,
    validation_type: str = "static",
):
    if dry_run:
        click.echo("Dry-run mode")
    else:
        if (
            not sys.platform.startswith("linux")
            or not os.path.exists("/proj")
            or not shutil.which("bsub")
        ):
            sys.exit(
                "Error: 'llmbsub' command works only in LLM.build enabed LSF environments such as BlueVela."
            )

    if space and space.lower() != "public":
        sys.exit(
            f"Error: invalid space name '{space}'. Currently, this option only supports space name 'public'."
        )

    if not llmb_output_dir and not llmb_output_dir_from_yaml:
        sys.exit(
            "Error: no output directory was provided. Please specify it using --llmb-output-dir or --llmb-output-dir-from-yaml."
        )
    elif llmb_output_dir and llmb_output_dir_from_yaml:
        sys.exit(
            "Error: --llmb-output-dir and --llmb-output-dir-from-yaml were provided. Please choose one to specify.",
        )

    if llmbconfig and not os.path.exists(llmbconfig):
        sys.exit(f"Error: config file {llmbconfig} not found.")

    if llmbin_from_yaml:
        if not llmbconfig:
            sys.exit(
                "Error: Please specify the YAML file path with --llmbconfig to enable --llmbin-from-yaml."
            )
        for input in llmbin_from_yaml:
            inputs_yaml_file = parse_inputs_yaml_file(input, llmbconfig)
            if len(inputs_yaml_file) == 0:
                sys.exit(
                    f"Error: no inputs found in {llmbconfig} using expression {input}."
                )

            for i in inputs_yaml_file:
                file_path = i.split(":")[-1]
                if not any(file_path in l for l in llmbin):
                    llmbin += (i,)
                else:
                    click.echo(
                        f"⚠️  Warning: skipping duplicate input {i} from YAML file."
                    )

    if llmbin_from_file:
        inputs_text_file = parse_text_file(llmbin_from_file)
        if len(inputs_text_file) == 0:
            sys.exit(f"Error: no inputs found in {llmbin_from_file}.")

        for i in inputs_text_file:
            file_path = i.split(":")[-1]
            if not any(file_path in l for l in llmbin):
                llmbin += (i,)
            else:
                click.echo(
                    f"⚠️  Warning: skipping duplicate input {i} from {llmbin_from_file}.\n"
                )

    if llmb_output_dir_from_yaml:
        if not llmbconfig:
            sys.exit(
                "Error: Please specify the YAML file path with --llmbconfig to enable --llmb-output-dir-from-yaml."
            )
        output_yaml_file = parse_outputs_yaml_file(
            llmb_output_dir_from_yaml, llmbconfig
        )
        if len(output_yaml_file) > 1:
            sys.exit(
                "Error: Only one output directory should be specified using --llmb-output-dir-from-yaml."
            )
        elif len(output_yaml_file) == 0:
            sys.exit(f"Error: no output directory found in {llmbconfig}.")
        else:
            llmb_output_dir = output_yaml_file[0]

    from gbcli.utils.cli_config import get_local_build_cache

    output_dir = Path(get_local_build_cache()).resolve() / "llmbsub"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_identifier = uuid.uuid4()

    if not FEATURE_FLAGS["use_project_log_folder"]:
        llmb_log_dir_base = Path("/proj/granite-build/llmb/logs")
    else:
        proj_root_directory = "proj"
        if not llmb_project:
            current_directory = os.getcwd().split("/")
            proj_current_directory = (
                current_directory[current_directory.index(proj_root_directory) + 1]
                if proj_root_directory in current_directory
                else None
            )

            llmb_output_dir_path = Path(llmb_output_dir).resolve()
            llmb_output_directory = str(llmb_output_dir_path).split("/")
            proj_llmb_output_directory = (
                llmb_output_directory[
                    llmb_output_directory.index(proj_root_directory) + 1
                ]
                if proj_root_directory in llmb_output_directory
                else None
            )

            if proj_current_directory:
                llmb_project = proj_current_directory
                click.echo(
                    f"Obtaining project name '{llmb_project}' from current directory..."
                )
            elif proj_llmb_output_directory:
                llmb_project = proj_llmb_output_directory
                click.echo(
                    f"Obtaining project name '{llmb_project}' from llmb-output-dir..."
                )
            else:
                click.echo(f"⚠️  Warning: Setting 'data-eng' as project name.")
                llmb_project = "data-eng"

        if not dry_run and not os.path.exists(
            f"/{proj_root_directory}/{llmb_project}/llmb-read-write"
        ):
            sys.exit(
                f"Error: You're running LLM.build Lite for the project {llmb_project}, but it doesn't appear to be configured yet."
            )

        llmb_log_dir_base = Path(
            f"/{proj_root_directory}/{llmb_project}/llmb-read-write/logs"
        )
    log_path = llmb_log_dir_base / f"{run_identifier}/job.log"

    output_is_subfolder = False
    if llmbout == ():
        output_is_subfolder = True
        llmbout = (os.path.split(llmb_output_dir)[-1],)
        llmb_output_dir_path = Path(llmb_output_dir)
        llmb_output_dir = llmb_output_dir_path.parent.absolute()

    if not space:
        llmb_space_name = f"bv-{llmb_project.lower()}"
        lakehouse_namespace = f"granite_dot_build.{llmb_space_name.replace('-', '_')}"
    else:
        llmb_space_name = space
        lakehouse_namespace = f"{{{{ space.variables.DEFAULT_LH_NAMESPACE }}}}"

    logger.debug(
        f"Using space name '{llmb_space_name}' and Lakehouse namespace '{lakehouse_namespace}'."
    )

    build_inputs = parse_inputs(llmbin)
    build_outputs = parse_outputs(llmbout, output_is_subfolder, lakehouse_namespace)
    bsub_flags = list(bsub_args)

    use_blaunch = llmb_command[0] == "blaunch"
    if use_blaunch:
        assert len(llmb_command) >= 2, f"Invalid llmb_command: {llmb_command}"
        llmb_command = llmb_command[1:]  # remove blaunch prefix
        last_is_blaunch = False
        if len(bsub_flags) > 0:
            t1 = bsub_flags[-1]
            last_is_blaunch = t1 == "blaunch"
        if not last_is_blaunch:
            bsub_flags.append("blaunch")

    click.echo("Generating a wrapper script...")
    blaunch_command = " ".join(f"'{x}'" if " " in x else x for x in llmb_command)

    wrapper_path, log_path_dir = create_shell_script(
        output_dir,
        blaunch_command,
        str(log_path),
        llmb_output_dir,
        run_identifier,
    )
    logger.debug(f"wrapper_path: {wrapper_path}")

    bsub_cmd = ["bsub", *bsub_flags, str(wrapper_path)]
    if dry_run:
        bsub_cmd = ["echo", "bsub", *bsub_flags, str(wrapper_path)]

    bsub_cmd_internal = ["bsub", *bsub_flags, blaunch_command]
    bsub_cmd_internal_str = " ".join(bsub_cmd_internal)
    click.echo("Executing user command:")
    click.echo(parse_markdown_str(f"```\n{bsub_cmd_internal_str}\n```"))

    job = subprocess.run(bsub_cmd, capture_output=True, text=True)
    job_id = None
    if job.returncode == 0:
        click.echo(f"⚡ Job submission successful!")
        job_info = re.findall(r"\<(.*?)\>", job.stdout)
        if len(job_info) > 0:
            job_id = job_info[0]
            click.echo(f"Job ID: {job_id}")
        elif dry_run:
            job_id = 1234
        else:
            click.echo(job.stderr, err=True)
            sys.exit(1)
    else:
        click.echo(f"❌ Command failed with return code: {job.returncode}")
        click.echo(job.stderr, err=True)
        sys.exit(1)

    symlink_bsub_log_folder = Path(f"{llmb_log_dir_base}/log-bsub/{job_id}")
    symlink_bsub_result = create_symlink(symlink_bsub_log_folder, log_path_dir)
    if not symlink_bsub_result:
        click.echo(
            f"Failed to create a symlink for the bsub job ID {symlink_bsub_log_folder}. Ignoring..."
        )

    # Register input artifacts and enrich build_inputs with metadata
    inputs_to_upload = []
    if REGISTRATION_FEATURE_FLAG and len(build_inputs) > 0:
        from llmbsub.services.artifact_registration_service import (
            ArtifactRegistrationService,
        )
        from llmbsub.services.input_upload_service import InputUploadService

        click.echo("Registering input artifacts...")
        if verbose:
            click.echo(f"  Space: {llmb_space_name}")
            click.echo(f"  Inputs to register: {list(build_inputs.keys())}")

        if any(obj.get("type") is None for obj in build_inputs.values()):
            non_typed_inputs = [
                f"   - {input_name}"
                for input_name, input_config in build_inputs.items()
                if input_config.get("type") is None
            ]

            if (
                click.confirm(
                    f"Some inputs do not have a specified type:\n{'\n'.join(non_typed_inputs)}\nUse the default {llmb_default_artifact_type} type?",
                    default=True,
                )
                == False
            ):
                llmb_default_artifact_type = click.prompt(
                    "New default type",
                    default=llmb_default_artifact_type,
                    show_default=True,
                    type=click.Choice(
                        ["models", "tables", "filesets"],
                        case_sensitive=True,
                    ),
                )

        registration_service = ArtifactRegistrationService(
            space=llmb_space_name,
            verbose=verbose,
        )

        for input_name, input_config in build_inputs.items():
            local_path = input_config.get("local_path")
            if not local_path:
                continue

            if verbose:
                click.echo(f"  Processing input: {input_name}")
                click.echo(f"    Local path: {local_path}")

            if input_config.get("type") == None:
                input_config["type"] = llmb_default_artifact_type

            # temporary
            if input_config.get("type") == "model":
                input_config["type"] = "fileset"

            try:
                result = registration_service.register_input(
                    local_path,
                    input_config.get("type"),
                    input_name,
                )

                # Enrich build_inputs with metadata (Plan B format)
                build_inputs[input_name]["metadata"] = {
                    "checksum": result.checksum,
                    "other_locations": [result.lh_uri],
                }

                if verbose:
                    click.echo(f"    Checksum: {result.checksum}")
                    click.echo(f"    Lakehouse URI: {result.lh_uri}")
                    click.echo(f"    Upload needed: {result.upload_needed}")
                    click.echo(f"    Status: {result.message}")

                logger.info(
                    f"Registered {input_name}: {result.lh_uri} "
                    f"(upload_needed={result.upload_needed})"
                )

                # Track which inputs need upload (only if UPLOAD_FEATURE_FLAG is also set)
                if result.upload_needed and UPLOAD_FEATURE_FLAG:
                    inputs_to_upload.append(
                        {
                            "path": local_path,
                            "target_uri": result.lh_uri,
                            "checksum": result.checksum,
                            "artifact_id": result.artifact_id,
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to register {input_name}: {e}")
                click.echo(f"⚠️  Warning: Failed to register {input_name}: {e}")

    # Remove local_path before generating build.yaml (not part of schema)
    for input_name in build_inputs:
        build_inputs[input_name].pop("local_path", None)

    # Submit input upload job if feature is enabled and there are inputs to upload
    if UPLOAD_FEATURE_FLAG and inputs_to_upload:
        # input_paths = [item["path"] for item in inputs_to_upload]
        # upload_job_id = submit_upload_job(
        #     inputs_to_upload=inputs_to_upload,
        #     space=llmb_space_name,
        #     log_dir=llmb_log_dir_base,
        #     run_identifier=str(run_identifier),
        #     parent_job_id=job_id,
        #     dry_run=dry_run,
        # )
        # if upload_job_id:
        #     click.echo(f"📤 Input upload job submitted: {upload_job_id}")

        # Running upload script
        click.echo(f"\nUploading {len(inputs_to_upload)} artifacts to Lakehouse...")
        run_upload_script(
            inputs_to_upload=inputs_to_upload,
            space=llmb_space_name,
            log_dir=llmb_log_dir_base,
            run_identifier=str(run_identifier),
            parent_job_id=job_id,
            dry_run=dry_run,
            verbose=verbose,
        )

    if "-J" in bsub_flags:
        bsub_build_name_index = bsub_flags.index("-J") + 1
        bsub_build_name = f"{bsub_flags[bsub_build_name_index]}_job_{job_id}"
        if len(bsub_build_name) > 256:
            logger.debug(
                f"Build name is too long: {bsub_build_name}. Taking the first 256 characters..."
            )
            bsub_build_name = bsub_build_name[:256]
    else:
        bsub_build_name = None

    click.echo("Generating build.yaml...")
    build_yaml_path = generate_build_yaml(
        output_dir,
        build_inputs,
        build_outputs,
        job_id,
        str(log_path),
        run_identifier,
        llmbconfig,
        bsub_build_name,
    )
    logger.debug(f"build_yaml_path: {build_yaml_path}")
    click.echo("Done.")

    # Output build.yaml to stdout if flag is set or in dry-run with registration
    if BUILD_YAML_TO_STDOUT or (dry_run and REGISTRATION_FEATURE_FLAG):
        click.echo("\n--- Generated build.yaml ---")
        with open(build_yaml_path, "r") as f:
            click.echo(f.read())
        click.echo("--- End build.yaml ---")

    click.echo("\nThe following artifacts from your job will be tracked by LLM.build.")

    click.echo("\n📦 Input artifacts")
    if len(build_inputs) > 0:
        click.echo(get_inputs_table(build_inputs))
    else:
        click.echo(
            "\nNo input artifacts will be tracked. Use the --llmbin option for each input artifact to track."
        )

    click.echo("\n📦 Output artifacts")
    if len(build_outputs) > 0:
        click.echo(f"{get_outputs_table(build_outputs, llmb_output_dir)}\n")
    else:
        click.echo(
            "\nNo output artifacts will be tracked. Use the --llmbout option for each output artifact to track.\n"
        )

    if dry_run:
        click.echo("Finishing without sending a build to LLM.build server.")
        return
    click.echo("Sending a build to LLM.build server...")

    with tqdm(
        total=100,
        miniters=1,
        bar_format="{desc} [{bar}] {percentage:3.0f}%",
        ascii="-#",
        leave=False,
    ) as progress_bar:

        def update_bar(callback_event: str, callback_args: Dict):
            steps = callback_args.get("steps", 0)
            match callback_event:
                case "preparing_contents":
                    if steps == 1:
                        progress_bar.reset(total=400)
                        progress_bar.set_description("(1/3) Preparing build contents.")
                    progress_bar.update(n=steps)
                case "prepared_contents":
                    progress_bar.update(n=400)
                    progress_bar.write("(1/3) Prepared build contents.")
                case "skip__pr_validation":
                    progress_bar.write("(2/3) Skipping build contents validation.")
                case "validating_pr":
                    if steps == 1:
                        progress_bar.reset(total=100)
                    progress_bar.set_description("(2/3) Validating build contents.")
                    progress_bar.update(n=steps)
                case "validated_pr":
                    progress_bar.update(n=steps)
                    progress_bar.write("(2/3) Validated build contents.")
                case "submitting_pr":
                    if steps == 1:
                        progress_bar.reset(total=100)
                    progress_bar.set_description("(3/3) Submitting build request.")
                    progress_bar.update(n=steps)
                case "submitted_pr":
                    space_org = callback_args.get("space_org", "")
                    space_name = callback_args.get("space_name", "")
                    if space_org:
                        description = f"Submitted build to {space_org}/{space_name}."
                    else:
                        description = "Submitted build request."
                    progress_bar.update(n=steps)
                    progress_bar.write(f"(3/3) {description}")
                    progress_bar.clear()
                case "warning":
                    progress_bar.clear()
                    reason = callback_args.get("reason", "")
                    print(f"\n⚠️ Warning: {reason}\n")
                case "error":
                    reason = callback_args.get("reason", "")
                    print(
                        f"\n❌ Build can't be submitted at this moment... Reason: {reason}"
                    )
                    sys.exit(1)  # Exit with a non-zero status
                case "validation_error":
                    progress_bar.clear()
                    number_errors = callback_args.get("number_errors", 0)
                    number_warnings = callback_args.get("number_warnings", 0)
                    build_path = callback_args.get("build_path", "")
                    error_text = callback_args.get("error_text", "")
                    option_text = "Use '--verbose-validation` to see more details."
                    print(
                        f"\n❌ Build validation failed with {number_errors} errors and {number_warnings} warnings for build definition '{build_path}'. {option_text}"
                    )
                    if error_text and len(error_text) > 0:
                        print(error_text)
                    sys.exit(1)  # Exit with a non-zero status
                case "validation_warning":
                    number_warnings = callback_args.get("number_warnings", 0)
                    build_path = callback_args.get("build_path", "")
                    warning_text = callback_args.get("number_warnings", "")
                    print(
                        f"\n⚠️ Build validation has {number_warnings} warnings for build definition '{build_path}'. Use '--verbose-validation` to see more details."
                    )

                    if warning_text and len(warning_text) > 0:
                        print(warning_text)
                case _:
                    pass

        if not dry_run and not getenv_boolean("SKIP_BUILD_START", False):
            from llmbsub.services.llmbsub_service_build import llmbsub_build_start

            requested_build_id = llmbsub_build_start(
                quiet=False,
                filename=str(build_yaml_path),
                skip_validation=True,
                callback=update_bar,
                space=llmb_space_name,
                skip_space_resolution=True,
                validation_type=validation_type,
            )

            symlink_llmb_log_folder = Path(
                f"{llmb_log_dir_base}/log-llmb/{requested_build_id}"
            )
            symlink_llmb_result = create_symlink(symlink_llmb_log_folder, log_path_dir)
            if not symlink_llmb_result:
                click.echo(
                    f"Failed to create a symlink for the build ID {symlink_llmb_log_folder}. Ignoring..."
                )

            log_metadata_path = llmb_log_dir_base / f"{run_identifier}/job.log.metadata"
            saved_metadata_path = generate_metadata_log(
                log_metadata_path, requested_build_id
            )

            logger.debug(f"log_metadata_path: {saved_metadata_path}")

            click.echo(f"✅ LLM.build ID: {requested_build_id}.")
            details_page = f"{DMF_URL}/gb/builds/{requested_build_id}"
            markdown_str = f"""
To get the LSF job status, run:
```
bjobs -l {job_id}
```

To get the LLM.build status, run:
```
llmb build status {requested_build_id}
```

or open the build status UI at [{details_page}]({details_page}).
            """
            click.echo(parse_markdown_str(markdown_str))


class LLMBSubCLI(click.Command):
    def __init__(
        self,
        **attrs: Any,
    ):
        super().__init__(**attrs)
        self._set_configs()

    def _set_configs(self):
        from gbcli.utils.cli_config import configureGBWorkingEnv

        configureGBWorkingEnv()

    def list_commands(self, ctx):
        return []


CONTEXT_SETTINGS = dict(
    auto_envvar_prefix="LLMBSUB", ignore_unknown_options=True, allow_extra_args=True
)


def split_args(args):
    index = args.index("--llmbcommand")
    bsub_args = args[:index]
    llmb_command = args[index + 1 :]
    return bsub_args, llmb_command


class MyHelpFormatter(click.HelpFormatter):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)
        self.indent_increment = 2

    def write_usage(self, prog: str, args: str = "", prefix: str | None = None) -> None:
        self.write(
            f"Usage: {prog} [BSUB OPTIONS] [OPTIONS] --llmbcommand [BSUB ARGS]...\n"
        )


click.Context.formatter_class = MyHelpFormatter


@click.command(cls=LLMBSubCLI, context_settings=CONTEXT_SETTINGS)
@click.option(
    "--llmbconfig",
    "--llmb-config",
    help="Path to a config file to track. The copy of the config file content is captured into LLM.build build definition file.",
)
@click.option(
    "--llmbin",
    "--llmb-in",
    multiple=True,
    help="Inputs to track, format is [artifact_type:]name:/path/to/input . artifact_type can be model(s), table(s), or fileset(s). If the artifact type is omitted, the default option will be used (--llmb-default-artifact-type).",
)
@click.option(
    "--llmbin-from-file",
    help="Read the list of inputs from a text file.",
)
@click.option(
    "--llmbin-from-yaml",
    multiple=True,
    help="Read the list of inputs from a YAML array. Format is [artifact_type:](expression). e.g. models:.input_array.",
)
@click.option(
    "--llmb-default-artifact-type",
    type=click.Choice(["model", "table", "fileset"], case_sensitive=True),
    default="fileset",
    help="Default artifact type to be used if type is not specified: model, table, fileset (default)",
)
@click.option(
    "--llmbout",
    "--llmb-out",
    multiple=True,
    help="Outputs to track, format is artifact_type:name[:/path/to/output] . artifact_type can be model(s), table(s), or fileset(s). The path is optional, and is ignored in the current version.",
)
@click.option(
    "--llmb-output-dir",
    "--workload-output-dir",
    help="Workload's output dir",
)
@click.option(
    "--llmb-output-dir-from-yaml",
    help="Read the workload's output dir from a YAML element or a YAML array. Format is (expression). e.g. .output_dir_array.",
)
@click.option(
    "--llmb-project",
    help="Project name.",
    hidden=(not FEATURE_FLAGS["use_project_log_folder"]),
)
@click.option(
    "--space",
    help="Space name. Currently, only 'public' is accepted.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Dry run mode. If specified, the command will just echo the bsub command instead of running it.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose mode. Enables detailed logging of REST API calls, checksum calculations, and internal operations.",
)
@click.option(
    "--loglevel",
    default="WARNING",
    help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def llmbsub(
    ctx,
    loglevel,
    llmbin,
    llmbout,
    llmbconfig,
    llmb_output_dir,
    llmb_output_dir_from_yaml,
    llmbin_from_file,
    llmbin_from_yaml,
    llmb_default_artifact_type,
    llmb_project,
    space,
    dry_run,
    verbose,
    args,
):
    """LLM.build command line interface for submitting a job to LSF.
    The command works as a "bsub" wrapper to submit a user job, with additional options to specify job inputs and outputs
    for artifact and lineage tracking.
    """
    # ctx.ensure_object(dict)
    # Verbose mode overrides loglevel to DEBUG
    effective_loglevel = "DEBUG" if verbose else loglevel.upper()
    logging.basicConfig(
        level=effective_loglevel,
        format=(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            if verbose
            else "%(levelname)s: %(message)s"
        ),
    )
    if verbose:
        click.echo("🔍 Verbose mode enabled - detailed logging active")
    #
    if not "--llmbcommand" in args:
        ctx.fail("Missing option '--llmbcommand'.")
    elif args.index("--llmbcommand") + 1 == len(args):
        ctx.fail("Option '--llmbcommand' requires an argument.")

    click.echo("📋 llmbsub")

    if not os.path.exists(
        os.path.abspath(os.path.join(get_local_gb_config(), "credentials"))
    ):
        click.echo(
            r"Error: User not logged in. Obtain a new token with 'source /u/granitebuild/llmb/setup && llmb auth login'.",
            err=True,
        )
        sys.exit(1)

    bsub_args, llmb_command = split_args(args)
    submit_bsub(
        llmbin,
        llmbout,
        llmb_output_dir,
        llmbin_from_file,
        llmbin_from_yaml,
        llmb_default_artifact_type,
        bsub_args,
        llmb_command,
        dry_run,
        llmbconfig,
        llmb_project,
        llmb_output_dir_from_yaml,
        space,
        verbose,
    )
