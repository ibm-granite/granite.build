#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Access assets in the local filesystem."""

from typing import Self, Union

from gbcommon.uri.file import FileURI
from gbcommon.uri.uri import URI
from gbserver.asset.assetstore import Assetstore


class Filestore(Assetstore):
    """A class for storing and accessing file asset"""

    def __init__(self: Self, uri: Union[URI, str], **kwargs) -> None:
        super().__init__(uri, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def get_supported_uri_classes(self):
        return [FileURI]
