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
The environment type.
"""

from typing import Dict, List, Optional, Union

from pydantic import Field, field_validator

from gbserver.types.config import Config

ENVIRONMENT_FILENAME = "environment.yaml"


class StoreLoad(Config):
    mode: Optional[str] = None
    config: Dict = Field(default_factory=dict)


class StorePush(Config):
    mode: Optional[str] = None
    config: Dict = Field(default_factory=dict)


class AssetStoreEnvironmentConfig(Config):
    store_uri: str = ""
    load: List[StoreLoad] = Field(default_factory=list)
    push: List[StorePush] = Field(default_factory=list)


class EnvironmentConfig(Config):
    """The environment.yaml file.

    Attributes:
        name: The user-facing name of the environment.
        type: The environment class identifier (e.g. ``Skypilot``, ``K8s``).
        step_type: Optional ordered fallback chain identifying which step
            implementations apply to this environment.  May be a single string
            (single-tier match) or a list of strings (most-preferred first).
            When omitted or empty, step resolution skips step_type-narrowing
            and falls straight through to the env-agnostic ``steps/<name>/``
            location.  See ``Environment.step_type_chain`` for the normalized
            list view used by the resolver.
        config: Free-form environment-class-specific config block.
        assetstores: Per-environment assetstore mappings.
    """

    name: str
    type: str
    step_type: Optional[Union[str, List[str]]] = None
    config: Dict = Field(default_factory=dict)
    assetstores: List[AssetStoreEnvironmentConfig] = Field(default_factory=list)

    @field_validator("step_type", mode="before")
    @classmethod
    def _coerce_empty_step_type(cls, v):
        """Treat the empty-string form ``step_type: ""`` as if the field were
        absent so callers can rely on a None/empty distinction."""
        if v == "" or v == []:
            return None
        return v
