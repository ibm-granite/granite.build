#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""
For storing and accessing file/folders in a remote environment
"""

from typing import Union

from gbcommon.uri.env import EnvURI
from gbcommon.uri.uri import URI
from gbserver.asset.assetstore import Assetstore


class Envstore(Assetstore):
    """A class for storing and accessing file/folders in a remote environment"""

    def __init__(self, uri: Union[URI, str], **kwargs):
        super().__init__(uri, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def get_supported_uri_classes(self):
        return [EnvURI]
