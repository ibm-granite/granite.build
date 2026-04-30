import logging
import math
import os
import pathlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import yaml
from tabulate import tabulate

from gbcli.utils.utils import remove_prefix
from llmbsub.utils.llmbsub_constants import (
    ARTIFACT_NAME_CHARACTERS,
    CONFIG_FILE_CONTENTS,
    LLMBSUB_OUTPUT_HEADER,
)

logger = logging.getLogger(__name__)


def get_absolute_path(file_path: str) -> Path:
    file_path_obj = pathlib.Path(file_path)
    if file_path_obj.is_absolute():
        return file_path_obj
    else:
        return file_path_obj.resolve()


def validate_artifact_name(artifact_name: str) -> bool:
    pattern = re.compile(f"^[{ARTIFACT_NAME_CHARACTERS}]*$")
    if not pattern.match(artifact_name):
        return False

    return True


def yaml_parser(yq_expression: str, yaml_path: str) -> list[str]:
    try:
        yaml_file_yq = subprocess.run(
            ["yq", "eval", f"{yq_expression}", yaml_path],
            capture_output=True,
            text=True,
        )
    except Exception as e:
        if "No such file or directory: 'yq'" in str(e):
            sys.exit(
                "Error: The 'yq' command is not found. Install 'yq' to use a 'from-yaml' option."
            )
        else:
            sys.exit(f"Error parsing YAML file {yaml_path}: {str(e)}.")

    if yaml_file_yq.returncode == 0:
        if yaml_file_yq.stdout.rstrip() == "null" or yaml_file_yq.stdout.rstrip() == "":
            return []
        array_contents = yaml_file_yq.stdout
        parsed_yaml_file = array_contents.replace('"', "").replace("\\n", "").split(" ")
    else:
        sys.exit(f"Error: {yaml_file_yq.stderr}")

    return parsed_yaml_file


def parse_outputs_yaml_file(output_syntax: str, yaml_path: str) -> list[str]:
    if not os.path.exists(yaml_path):
        sys.exit(f"Error: file {yaml_path} not found.")

    outputs_yaml_file = yaml_parser(output_syntax, yaml_path)

    return [i.rstrip() for i in outputs_yaml_file]


def parse_inputs_yaml_file(llmbin_from_yaml: str, yaml_path: str) -> list[str]:
    input_syntax = str(llmbin_from_yaml).split(":")
    if len(input_syntax) == 2:
        array_location = input_syntax[1]
    elif len(input_syntax) == 1:
        array_location = input_syntax[0]
    else:
        sys.exit(
            "Error: --llmbin-from-yaml must follow the format [artifact_type:](expression). e.g. models:.input_array."
        )

    if not os.path.exists(yaml_path):
        sys.exit(f"Error: file {yaml_path} not found.")

    inputs_yaml_file = yaml_parser(array_location, yaml_path)

    return [i.rstrip() for i in inputs_yaml_file]


def parse_text_file(llmbin_from_file: str) -> list[str]:
    input_syntax = str(llmbin_from_file).split(":")
    if len(input_syntax) == 2:
        file_path = input_syntax[1]
    else:
        file_path = input_syntax[0]
    if os.path.exists(file_path):
        inputs_text_file = []
        with open(file_path, "r", encoding="utf-8") as f:
            inputs_text_file = f.readlines()

        return [i.rstrip() for i in inputs_text_file]
    else:
        sys.exit(f"Error: file {file_path} not found.")


def parse_inputs(llmbin: tuple) -> dict:
    input_artifact_types = ["model", "models", "table", "tables", "fileset", "filesets"]
    inputs = {}

    for index, input in enumerate(llmbin):
        input_info = str(input).split(":")
        input_type = None
        if len(input_info) == 3:
            input_type = input_info[0]
            input_name = input_info[1]
            input_uri = input_info[2]
        elif len(input_info) == 2:
            if input_info[0] in input_artifact_types:
                # if formatted as [type:]path
                input_type = input_info[0]
                input_uri = input_info[1]
                input_name = re.sub(f"[^{ARTIFACT_NAME_CHARACTERS}]+", "_", input_info[1])
            else:
                # if formatted as name:path
                input_name = input_info[0]
                input_uri = input_info[1]
        else:
            file_name_split = os.path.splitext(os.path.basename(input_info[0]))
            file_name = file_name_split[0]
            if file_name == "":
                file_name = input_info[0].strip("/").split("/")[-1]
            input_name = re.sub(f"[^{ARTIFACT_NAME_CHARACTERS}]+", "_", file_name)
            input_uri = input_info[0]

        input_name = input_name.removeprefix("_")

        if not validate_artifact_name(input_name):
            sys.exit(
                f"Error: Invalid artifact name {input_name} for input {input}. Only alphanumeric characters and underscores are allowed."
            )

        if input_type is not None and input_type not in input_artifact_types:
            sys.exit(
                f"Error: Invalid artifact type {input_type} for input {input}. Only model(s), table(s), or fileset(s) are allowed."
            )

        if os.path.exists(input_uri):
            abs_path = get_absolute_path(input_uri)
            inputs[input_name] = {
                "uri": f"env://{abs_path}",
                "local_path": str(abs_path),  # For registration service
                "type": (input_type.removesuffix("s") if input_type is not None else None),
            }
        else:
            sys.exit(f"Error: Invalid URI {input_uri} for input {input_name}.")

    return inputs


def parse_outputs(llmbout: tuple, output_is_subfolder: bool, lakehouse_namespace: str) -> dict:
    outputs = {}
    base_path = f"lh://{{{{ space.variables.DEFAULT_LH_ENVIRONMENT }}}}/{lakehouse_namespace}"

    for output in llmbout:
        if output_is_subfolder:
            output_info = [output]
        else:
            output_info = str(output).split(":")
        if len(output_info) == 1:
            output_name = output_info[0]
            artifact_type = "filesets"
        else:
            output_name = output_info[1]
            artifact_type = f"{output_info[0]}"

        if artifact_type not in [
            "models",
            "model",
            "table",
            "tables",
            "fileset",
            "filesets",
        ]:
            sys.exit(f"Error: Unknown artifact type {artifact_type} for output {output}.")

        if not validate_artifact_name(output_name):
            sys.exit(
                f"Error: Invalid artifact name {output_name} for output {output}. Only alphanumeric characters and underscores are allowed."
            )

        artifact_path = (
            "tables"
            if (artifact_type == "tables" or artifact_type == "table")
            else (
                "models/{{ space.variables.DEFAULT_LH_MODEL_TABLE }}"
                if (artifact_type == "models" or artifact_type == "model")
                else "filesets/{{ space.variables.DEFAULT_LH_FILESET_TABLE }}"
            )
        )

        output_uri = f"{base_path}/{artifact_path}/{output_name.replace('*', '')}{{{{ run_metadata.targetsteprun_id | short_hash }}}}_{{{{ binding.path | path_basename }}}}"

        if artifact_type == "filesets" or artifact_type == "fileset":
            output_uri += "/1/"

        formatted_output_name = re.sub(r"_?\*", "", output_name)

        outputs[formatted_output_name] = {
            "uri": output_uri,
        }

        if formatted_output_name != output_name:
            outputs[formatted_output_name]["event_selectors"] = [
                {
                    "field_name": "binding.path",
                    "field_value_regex": re.escape(output_name)
                    .replace(r"\*", r".*")
                    .replace(r"\?", r"."),
                },
            ]
    return outputs


def generate_build_yaml(
    output_dir: str,
    build_inputs: dict,
    build_outputs: dict,
    job_id: str,
    log_path: str,
    run_identifier: str,
    llmbconfig: str,
    bsub_build_name=None,
):
    step = {
        "step_uri": "space://steps/env_exec",
        "config": {
            "lsf": {
                "bsub": {
                    "jobid": job_id,
                    "log_path": log_path,
                }
            },
        },
    }

    if llmbconfig:
        config_file_name = os.path.split(llmbconfig)[-1]
        with open(llmbconfig, "r", encoding="utf-8") as config_file:
            file_lines = config_file.readlines()

        indented_file_lines = [file_lines[0]] + [
            f"                {line}" for line in file_lines[1:]
        ]
        contents = "".join(indented_file_lines)

        step["config"]["llmb"] = {"additional_files": CONFIG_FILE_CONTENTS}

        embedded_file = f"""
              {config_file_name}: |
                {contents}
        """

    target = {
        "environment_uri": "space://environments/bluevela",
        "inputs": build_inputs,
        "outputs": build_outputs,
        "steps": [step],
    }

    build_config = {
        "granite.build": {
            "name": (f"bluevela_job_{job_id}" if not bsub_build_name else bsub_build_name),
            "targets": {"my_workload": target},
        }
    }

    build_yaml_path = output_dir / f"build-{run_identifier}.yaml"
    with open(build_yaml_path, "w", encoding="utf-8") as build_yaml_file:
        build_config_dump = yaml.dump(
            build_config,
            default_flow_style=False,
            sort_keys=False,
            width=math.inf,
        )
        if llmbconfig:
            build_config_dump = build_config_dump.replace(
                CONFIG_FILE_CONTENTS, str(embedded_file).rstrip()
            )
        build_yaml_file.write(build_config_dump)

    return build_yaml_path


def generate_metadata_log(metadata_path: str, build_id: str):
    metadata_path_as_path = Path(metadata_path)
    log_path_dir = metadata_path_as_path.parent.absolute()

    os.makedirs(log_path_dir, exist_ok=True)

    with open(metadata_path, "w") as m:
        m.write(f"""labels:
  kubernetes.labels.granite-dot-build/build-id: {build_id}
  kubernetes.labels.llmbuild/build-id: {build_id}
""")
    return metadata_path


def create_shell_script(
    output_dir: Path,
    llmbcommand: str,
    log_path: str,
    workload_output_dir: str,
    run_identifier: str,
) -> Tuple[Path, Path]:
    log_path_as_path = Path(log_path)
    log_path_dir = log_path_as_path.parent.absolute()
    wrapper_path = output_dir / f"wrapper-{run_identifier}.sh"
    with open(wrapper_path, "w") as wrapper:
        wrapper.write(rf"""#!/usr/bin/env bash

LLMB_LSF_OUTPUT_DIR="{get_absolute_path(workload_output_dir)}"
LLMB_LSF_LOG_FILE_COMBINED="{log_path}"
mkdir -p "{log_path_dir}"

# user provided
{llmbcommand} 2>&1 | tee -a "${{LLMB_LSF_LOG_FILE_COMBINED}}"

LLMB_LSF_JOB_EXIT_CODE="${{PIPESTATUS[0]}}"
echo `date '+[%Y-%m-%d %H:%M:%S%z]'` job ended | tee -a "${{LLMB_LSF_LOG_FILE_COMBINED}}"

if [[ "$LSF_PM_TASKID" == '1' ]]; then
# print log lines to create artifact events
echo `date '+[%Y-%m-%d %H:%M:%S%z]'` artifact registration requested | tee -a "${{LLMB_LSF_LOG_FILE_COMBINED}}"
find "${{LLMB_LSF_OUTPUT_DIR}}" -depth -mindepth 1 -maxdepth 1 -type d -exec bash -c 'echo "LLMB_ARTIFACT_ID:${{0##*/}} LLMB_ARTIFACT_PATH:{{}}"' {{}} \; | tee -a "${{LLMB_LSF_LOG_FILE_COMBINED}}"
fi

if [[ "${{LLMB_LSF_JOB_EXIT_CODE}}" != "0" ]]; then
    echo "${{LLMB_LSF_JOB_NAME}}: workload script failed, exit code: ${{LLMB_LSF_JOB_EXIT_CODE}}" | tee -a "${{LLMB_LSF_LOG_FILE_COMBINED}}"
    exit 1
fi

echo "${{LLMB_LSF_JOB_NAME}}: workload script finished successfully" | tee -a "${{LLMB_LSF_LOG_FILE_COMBINED}}"
echo `date '+[%Y-%m-%d %H:%M:%S%z]'` workload script finished | tee -a "${{LLMB_LSF_LOG_FILE_COMBINED}}"
""")

    wrapper_path.chmod(0o777)
    return wrapper_path, log_path_dir


def get_inputs_table(build_inputs: dict) -> str:
    input_artifacts_table = [
        [build_inputs[i]["type"], i, remove_prefix("env://", build_inputs[i]["uri"])]
        for i in build_inputs
    ]

    input_artifacts_output = tabulate(
        input_artifacts_table,
        LLMBSUB_OUTPUT_HEADER,
        tablefmt="plain",
    )

    return input_artifacts_output


def get_outputs_table(build_outputs: dict, llmb_output_dir: str) -> str:
    output_artifacts_table = []

    for o in build_outputs:
        artifact_type = (
            "model"
            if "models/" in build_outputs[o]["uri"]
            else "table" if "tables/" in build_outputs[o]["uri"] else "fileset"
        )
        output_artifacts_table.append(
            [artifact_type, o, f"{get_absolute_path(llmb_output_dir)}/{o}"]
        )

    output_artifacts_output = tabulate(
        output_artifacts_table,
        LLMBSUB_OUTPUT_HEADER,
        tablefmt="plain",
    )

    return output_artifacts_output


def create_symlink(symlink_log_folder: Path, log_path_dir: Path):
    try:
        symlink_log_folder.symlink_to(log_path_dir)
        logger.debug(f"Symbolic link created: {symlink_log_folder} -> {log_path_dir}.")
        return True
    except Exception as e:
        logger.debug(f"Symbolic link could not be created: {str(e)}.")
        return False
