#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional

from pydantic import Field

from gbserver.types.config import Config


class SpaceSecretManagerConfig(Config):
    type: str
    config: dict


class SpaceConfig(Config):
    name: str = ""
    secret_manager: SpaceSecretManagerConfig
    base_uris: Optional[List[str]] = Field(default=None)
    variables: Optional[Dict[str, str]] = Field(default=None)
