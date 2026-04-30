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

from typing import Dict, List, Optional

from pydantic import Field

from gbserver.types.config import Config

ENVIRONMENT_FILENAME = "environment.yaml"


class StoreLoad(Config):
    """Store Load implementation."""

    mode: Optional[str] = None
    config: Dict = Field(default_factory=dict)


class StorePush(Config):
    """Store Push implementation."""

    mode: Optional[str] = None
    config: Dict = Field(default_factory=dict)


class AssetStoreEnvironmentConfig(Config):
    """Asset Store Environment Config implementation."""

    store_uri: str = ""
    load: List[StoreLoad] = Field(default_factory=list)
    push: List[StorePush] = Field(default_factory=list)


class EnvironmentConfig(Config):
    """The environment.yaml file."""

    name: str
    type: str
    config: Dict = Field(default_factory=dict)
    assetstores: List[AssetStoreEnvironmentConfig] = Field(default_factory=list)
