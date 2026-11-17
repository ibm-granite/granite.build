#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import Field

from gbserver.types.config import Config


class AssetStoreConfig(Config):
    base_uri: str = Field(default=None)  # type: ignore[assignment]
    uri_regex: str = Field(default=None)  # type: ignore[assignment]
    config: Any = Field(default=None)
    asset_config: Any = Field(default=None)
