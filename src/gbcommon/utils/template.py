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

import dataclasses
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Union

import yaml
from jinja2 import BaseLoader, Undefined
from jinja2.sandbox import SandboxedEnvironment
from pydantic import BaseModel

from gbcommon.utils.utils import short_alphanumeric_lower_hash

logger = logging.getLogger(__name__)


class PreserveUndefined(Undefined):
    def __str__(self):
        return f"{{{{ {self._undefined_name} }}}}"

    def __getattr__(self, name):
        return PreserveUndefined(name=self._undefined_name + "." + name)

    def __getitem__(self, key):
        return PreserveUndefined(name=f"{self._undefined_name}[{repr(key)}]")


def json_to_yaml(value: Union[dict, str]):
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
    indentation = " " * width
    lines = s.splitlines()
    if not indent_first_line:
        return "\n".join([lines[0]] + [indentation + line for line in lines[1:]])
    return "\n".join([indentation + line for line in lines])


env = SandboxedEnvironment(loader=BaseLoader(), undefined=PreserveUndefined)
strict_env = SandboxedEnvironment(loader=BaseLoader())
strict_env.filters["to_yaml"] = json_to_yaml
strict_env.filters["short_hash"] = short_alphanumeric_lower_hash
strict_env.filters["json_dumps"] = json.dumps
strict_env.filters["path_basename"] = lambda x: os.path.basename(os.path.normpath(x))
strict_env.filters["indent"] = indent


def fill_template(templ: str, data: dict, strict: bool = False) -> str:
    template = None
    curr_env = strict_env if strict else env
    try:
        template = curr_env.from_string(templ)
    except Exception as e:
        raise ValueError(f"the template is invalid:\n{templ}") from e
    try:
        return template.render(data)
    except Exception as e:
        logger.error("failed to use the data %s to fill the template:\n%s", data, templ)
        raise RuntimeError(
            f"failed to use the data to fill the template:\n{templ}"
        ) from e


# def traverse_obj(obj: Any, f: Callable[[str], str]) -> Any:
#     """
#     Traverse the fields of an object, applying the function to string
#     Example: treat string fields as Jinja2 templates and fill them.
#     """

#     if isinstance(obj, str):
#         filled_str = f(obj)
#         return filled_str
#     if isinstance(obj, dict):
#         new_obj_dict = {}
#         for k, v in obj.items():
#             new_obj_dict[k] = traverse_obj(v, f)
#         return new_obj_dict
#     if isinstance(obj, list):
#         new_obj_list = []
#         for x in obj:
#             new_obj_list.append(traverse_obj(x, f))
#         return new_obj_list
#     if isinstance(obj, BaseModel):
#         obj_dict = obj.model_dump()
#         new_obj_dict = traverse_obj(obj_dict, f)
#         return new_obj_dict
#     if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
#         # https://docs.python.org/3/library/dataclasses.html#dataclasses.is_dataclass
#         obj_dict = dataclasses.asdict(obj)
#         new_obj_dict = traverse_obj(obj_dict, f)
#         return new_obj_dict
#     return obj


# def fill_objtemplate(obj: Any, data: dict, strict: bool = False):
#     def fill_template_internal(s: str) -> str:
#         return fill_template(templ=s, data=data, strict=strict)

#     new_obj = traverse_obj(obj, fill_template_internal)
#     return new_obj


# def fill_pydanticobjtemplate(obj: BaseModel, data: dict, strict: bool = False):
#     objstr = yaml.dump(obj.model_dump(), sort_keys=False)
#     objstr = fill_template(objstr, data, strict)
#     objasdict = yaml.safe_load(objstr)
#     return type(obj)(**objasdict)


# def fill_template_in_file(filepath: Path, data: dict, strict: bool = False):
#     if data is None:
#         data = {}
#     with open(filepath, "r", encoding="utf-8") as file:
#         string = file.read()
#     try:
#         with open(filepath, "w", encoding="utf-8") as file:
#             file.write(fill_template(string, data, strict))
#     except Exception as e:
#         raise ValueError(f"failed to fill the template at path {filepath}") from e
