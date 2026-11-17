#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""
For storing and accessing small values in the build's shared in-memory state.

Used to pass a producer target's binding payload (e.g. a service URL) to a
consumer target verbatim, bypassing filesystem-URI normalisation that would
corrupt values like ``http://host:8000``.
"""

from typing import Union

from gbcommon.uri.mem import MemURI
from gbcommon.uri.uri import URI
from gbserver.asset.assetstore import Assetstore


class Memstore(Assetstore):
    """A class for storing and accessing values in the build's shared memory."""

    def __init__(self, uri: Union[URI, str], **kwargs):
        super().__init__(uri, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def get_supported_uri_classes(self):
        return [MemURI]
