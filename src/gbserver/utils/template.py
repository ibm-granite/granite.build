#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper methods to fill Jinja2 templates"""

import dataclasses
import json
import os
from base64 import b64encode
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

import yaml
from jinja2 import BaseLoader, StrictUndefined, Undefined
from jinja2.ext import debug
from jinja2.sandbox import SandboxedEnvironment
from pydantic import BaseModel

from gbserver.utils.logger import get_logger
from gbserver.utils.utils import short_alphanumeric_lower_hash

logger = get_logger(__name__)


class PreserveUndefined(Undefined):
    """Preserves unfillable parts of the template as is for later filling."""

    def __str__(self):
        return f"{{{{ {self._undefined_name} }}}}"

    def __getattr__(self, name):
        return PreserveUndefined(name=self._undefined_name + "." + name)

    def __getitem__(self, key):
        return PreserveUndefined(name=f"{self._undefined_name}[{repr(key)}]")


def json_to_yaml(value: Union[dict, str]):
    """Convert an object to yaml (for use in templates)"""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return f"Error: Invalid JSON input: {value}"
    try:
        return yaml.dump(value, default_flow_style=False)
    except yaml.YAMLError as e:
        return f"Error: YAML conversion failed: {e}"


def indent(s, width=4, indent_first_line=False):
    """Indent the string with the given amount of spaces (for use in templates)"""
    indentation = " " * width
    lines = s.splitlines()
    if not indent_first_line:
        return "\n".join([lines[0]] + [indentation + line for line in lines[1:]])
    return "\n".join([indentation + line for line in lines])


def get_file_extension(path: str) -> str:
    """
    Takes a path and returns the filename extension if any.
    Example: 'aa/bb/cc.dd.jsonl' -> '.jsonl'
    """
    return Path(path).suffix


def my_b64encode(xs: Union[str, bytes]) -> str:
    """Encode a str or bytes as base64"""
    if isinstance(xs, str):
        xs = xs.encode(encoding="utf-8")
    xs_b64 = b64encode(xs)
    xs_b64_str = xs_b64.decode(encoding="utf-8")
    return xs_b64_str


def raise_error(error: str):
    """Raise a ValueError with the give message (for use in Jinja2 templates)"""
    raise ValueError(error)


env = SandboxedEnvironment(loader=BaseLoader(), undefined=PreserveUndefined, extensions=[debug])
strict_env = SandboxedEnvironment(
    loader=BaseLoader(), undefined=StrictUndefined, extensions=[debug]
)
strict_env.filters["to_yaml"] = json_to_yaml
strict_env.filters["short_hash"] = short_alphanumeric_lower_hash
strict_env.filters["json_dumps"] = json.dumps
strict_env.filters["path_basename"] = lambda x: os.path.basename(os.path.normpath(x))
strict_env.filters["indent"] = indent
strict_env.filters["get_file_extension"] = get_file_extension
strict_env.filters["b64encode"] = my_b64encode

strict_env.globals["raise_error"] = raise_error

env.filters["to_yaml"] = json_to_yaml
env.filters["short_hash"] = short_alphanumeric_lower_hash
env.filters["json_dumps"] = json.dumps
env.filters["path_basename"] = lambda x: os.path.basename(os.path.normpath(x))
env.filters["indent"] = indent


def fill_template(templ: str, data: dict, strict: bool = False) -> str:
    """
    Fill the given template string with the given data.
    If strict is True, undefined/unfillable parts of the template will throw errors.
    """
    template = None
    curr_env = strict_env if strict else env
    try:
        template = curr_env.from_string(templ)
    except Exception as e:
        logger.error("curr_env.from_string error: %s", e)
        raise ValueError(f"the template is invalid:\n{templ}") from e
    try:
        return template.render(data)
    except Exception as e:
        logger.warning(
            "=== Please try adding .gbignore "
            "in the folder path "
            "which needs to be ignored "
            "by gb for filling and "
            "try again ==="
        )
        logger.error("template.render error: %s", e)
        logger.error("failed to use the data '%s' to fill the template:\n%s", data, templ)
        raise RuntimeError(f"failed to use the data to fill the template:\n{templ}") from e


def traverse_obj(
    obj: Any, f: Callable[[str], str], skip_keys: Optional[Iterable[str]] = None
) -> Any:
    """
    Traverse the fields of an object, applying the function to string
    Example: treat string fields as Jinja2 templates and fill them.
    """
    skip_keys = set(skip_keys or ())
    if isinstance(obj, str):
        filled_str = f(obj)
        return filled_str
    if isinstance(obj, dict):
        new_obj_dict = {}
        for k, v in obj.items():
            # skip filling if any skip keys is passed
            if skip_keys is not None and k in skip_keys:
                new_obj_dict[k] = v
            else:
                new_obj_dict[k] = traverse_obj(v, f, skip_keys)
        return new_obj_dict
    if isinstance(obj, list):
        new_obj_list = []
        for x in obj:
            new_obj_list.append(traverse_obj(x, f, skip_keys))
        return new_obj_list
    if isinstance(obj, BaseModel):
        obj_dict = obj.model_dump()
        new_obj_dict = traverse_obj(obj_dict, f, skip_keys)
        return new_obj_dict
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # https://docs.python.org/3/library/dataclasses.html#dataclasses.is_dataclass
        obj_dict = dataclasses.asdict(obj)
        new_obj_dict = traverse_obj(obj_dict, f, skip_keys)
        return new_obj_dict
    return obj


def fill_objtemplate(
    obj: Any,
    data: dict,
    strict: bool = False,
    skip_keys: Optional[Iterable[str]] = None,
) -> Any:
    """
    Traverses the object, finding fields of type string which
    it treats it as a template and tries to fill with the given data.
    Use skip_keys to specify a set of keys that should be skipped.
    """

    def fill_template_internal(s: str) -> str:
        return fill_template(templ=s, data=data, strict=strict)

    new_obj = traverse_obj(obj, fill_template_internal, skip_keys=skip_keys)
    return new_obj


def fill_template_in_file(filepath: Path, data: dict, strict: bool = False):
    """
    Reads a file, treats it as a template and fills it,
    then writes it back to the same location.
    """
    if data is None:
        data = {}
    with open(filepath, "r", encoding="utf-8") as file:
        string = file.read()
    try:
        filled = fill_template(string, data, strict)
        with open(filepath, "w", encoding="utf-8") as file:
            logger.info("filepath: %s", filepath)
            file.write(filled)
    except Exception as e:
        raise ValueError(f"failed to fill the template at path {filepath}") from e
