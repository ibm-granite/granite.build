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

"""
Base class for all config files (usually YAML).
"""

from pathlib import Path
from typing import Dict, List, Optional, Self, Type

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    """A single config."""

    matched_base_key: str = ""

    @classmethod
    def from_yaml(
        cls: Type[Self],
        path: Path,
        basekey: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs: Dict,
    ) -> Self:
        """Construct a config from a yaml file."""
        if not path.is_absolute():
            if context is not None and context != "":
                path = context / path
            else:
                path = path.resolve()
        assert path.is_file(), f"expected path '{path}' to be a file"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        basekeys: List[str] = kwargs.get("basekeys", [])
        assert isinstance(basekeys, list), f"invalid basekeys: {basekeys}"
        if basekey is not None:
            basekeys = basekeys.copy()
            basekeys.append(basekey)
        matched_base_key = ""
        if len(basekeys) > 0:
            assert isinstance(data, dict), f"expected a dict, actual: {data}"
            done = False
            for bk in basekeys:
                if bk in data:
                    data = data[bk]
                    matched_base_key = bk
                    done = True
                    break
            if not done:
                raise ValueError(f"basekey(s) {basekeys} not found in data: {data}")
        assert isinstance(data, dict), f"expected a dict, actual: {data}"
        build_config = cls.model_validate(data)
        build_config.matched_base_key = matched_base_key
        return build_config
