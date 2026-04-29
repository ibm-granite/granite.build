import json
import logging
import os

import yaml
from jinja2 import StrictUndefined, Template
from jsonpatch import apply_patch

from gbcli.utils.gbconstants import BUILD_PARAMETERS_APPLIED_FILE

logger = logging.getLogger(__name__)


def apply_parameters(contents, params, params_from_file, build_folder_path):
    """Replace parameters given in the format etc. 'name=xyz, name0.name1=abc, name1.name3=def'"""
    template = Template(
        contents,
        undefined=StrictUndefined,
        variable_start_string="$${",
        variable_end_string="}",
        block_start_string="<%",
        block_end_string="%>",
    )
    params_dict = parse_params(params, params_from_file)
    try:
        params_replacement = template.render(params_dict)
        if len(params_dict) > 0:
            params_applied_path = os.path.join(
                build_folder_path, BUILD_PARAMETERS_APPLIED_FILE
            )
            with open(params_applied_path, "w", encoding="utf-8") as f:
                f.write(yaml.safe_dump(params_dict))
        return params_replacement
    except Exception as e:
        raise e


def parse_params(params, params_from_file):
    data = params_from_file
    for param in params:
        data = add_parameter(data, param)
    return data


def process_build_validation_response(validate_response):
    """
    validation response contains array of 'errors' and 'warnings'

    process those two arrays, return as a single array
    returns: single processed array
    """

    def adapt_response(validation, level):
        validation["level"] = level
        summary, separator, detail = validation.get(level).partition("\n")
        validation["summary"] = summary
        validation["detail"] = detail
        try:
            solution = validation.get("solution")
            solution_dict = json.loads(solution)
            if solution_dict and "json_patches" in solution_dict:
                validation["json_patch"] = solution_dict.get("json_patches")
            else:
                raise Exception
        except:
            validation["json_patch"] = None
        return validation

    errs = [adapt_response(x, "error") for x in validate_response.get("errors", [])]
    warnings = [
        adapt_response(x, "warning") for x in validate_response.get("warnings", [])
    ]
    return errs + warnings


def add_parameter(data, param):
    """Add a parameter noted as 'key=value', where key allows a dot notation."""
    key_value = param.split("=", 1)
    if len(key_value) != 2:
        raise Exception(
            f"Invalid parameter {param}. Parameters must be written in the format 'key=value'."
        )
    return add_key_value(data, key_value[0].strip(), key_value[1].strip())


def add_key_value(data, key, value):
    """Add a key/value pair to the data structure, where key allows a dot notation.
    For example, key='key.subkey.subsubkey' and value='value' will be added to the data structure as
    { "key": { "subkey": { "subsubkey": "value"}}}
    and a new data structure is returned
    """

    def add_branch(data_rec, prefix, key_vector, value):
        key = key_vector[0]
        if not isinstance(data_rec, dict):
            raise Exception(
                f"Error: param {prefix}.{key} cannot be used because the prefix {prefix} is already in use."
            )

        if len(key_vector) == 1:
            data_rec[key] = value
        else:
            data_rec[key] = add_branch(
                data_rec[key] if key in data_rec else {},
                f"{prefix}.{key}",
                key_vector[1:],
                value,
            )
        return data_rec

    return add_branch(data, "", key.split("."), value)


def build_updated_yaml(build_yaml_dict, json_patches):
    """
    accepts original yaml as a python dictionary, array of json_patches
    returns: yaml string with applied updates to original yaml
    """
    if (
        not build_yaml_dict
        or not isinstance(build_yaml_dict, dict)
        or not json_patches
        or not isinstance(json_patches, list)
        or len(json_patches) < 1
    ):
        return None

    try:
        return apply_patch(doc=build_yaml_dict, patch=json_patches)
    except Exception as e:
        return {"error": f"Could not apply patches to build file: {e}"}


def safely_load_yaml_file(yaml_path: str, callback):
    """
    safely loads yaml from provided file path
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as file:
            yaml_str = file.read()
        return yaml_str
    except FileNotFoundError:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={"reason": f"The file {yaml_path} could not be found"},
            )
            return None
    except Exception as e:
        if callback is not None:
            callback(
                callback_event="error",
                callback_args={
                    "reason": f"Error occurred while loading yaml file: {e}"
                },
            )
            return None


def get_yaml_diff(build_yaml_dict: dict, validation: dict):
    json_patch = validation.get("json_patch")
    if not json_patch:
        return None

    return build_updated_yaml(
        build_yaml_dict,
        json_patch,
    )


def get_yaml_patches_in_steps(original_build_yaml_dict: dict, validations: dict):
    """
    applies each jsonpatch to the the updated yaml after the prior patch

    updates validation["updated_yaml"] in place in validations
    if there is a patch that works
    """
    next_build_yaml_dict = original_build_yaml_dict
    for validation in validations:
        updated_yaml_dict = get_yaml_diff(next_build_yaml_dict, validation)

        if updated_yaml:
            updated_yaml = yaml.safe_dump(updated_yaml_dict)
            validation["updated_yaml"] = updated_yaml

        if updated_yaml_dict:
            next_build_yaml_dict = updated_yaml_dict
