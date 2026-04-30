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
An asset in an object stored in an asset store.
They can be accessed via URIs given to the appropriate asset stores.
"""

import tempfile
from pathlib import Path
from typing import Any, Optional, Self, Union

from gbcommon.uri.uri import URI
from gbserver.asset.assetstore import Assetstore
from gbserver.types.artifact import ArtifactType
from gbserver.utils.filesystem import merge_dicts
from gbserver.utils.logger import get_logger
from gbserver.utils.template import fill_objtemplate

logger = get_logger(__name__)


class Asset:
    """Base class for storing and accessing any asset"""

    # instance attributes
    uri: URI
    uristr: str

    def __init__(self: Self, uri: Union[URI, str], context: Optional[str] = None):
        if isinstance(uri, URI):
            self.uri = uri
        else:
            self.uri = URI.get_uri(uri=uri, default_scheme="file", context=context)
        # Set uristr after any templates get filled
        self.uristr = URI.get_uristr(self.uri)

    def urihash(self: Self) -> str:
        """Urihash."""
        return self.uri.hash()

    def sync(self: Self, dest: Optional[Path] = None, force: bool = False) -> Path:
        """Fetch the asset to the given path."""
        if dest is None:
            dest = Path(tempfile.mkdtemp())
        if self.uri.pull(dest=dest, force=force):
            self.local = dest
            return dest
        return None

    def get_metadata(self: Self) -> Any:
        """Get the metadata."""
        uri_metadata = self.uri.get_metadata()
        assetstore = self.get_assetstore(asset=self)
        uri_assetstore_metadata = assetstore.get_metadata(self.uri)
        config = fill_objtemplate(
            assetstore.config.asset_config, {"uri_metadata": uri_metadata}, strict=True
        )
        if config is None:
            config = {}
        return merge_dicts(uri_metadata, merge_dicts(uri_assetstore_metadata, config))

    @staticmethod
    def get_assetstore(
        asset: Optional["Asset"] = None, uri: Optional[URI] = None
    ) -> Optional[Assetstore]:
        """Returns the asset store that matches the longest prefix of the URI."""
        if asset is not None:
            uristr = asset.uristr
        elif isinstance(uri, URI):
            uristr = URI.get_uristr(uri)
        else:
            uristr = uri
        longest_match = 0
        t1 = None
        for assetstore in Assetstore._thread_local.assetstores.values():
            assert isinstance(assetstore, Assetstore)
            curr_len = assetstore.can_handle_len(uristr)
            if curr_len > longest_match:
                longest_match = curr_len
                t1 = assetstore
        return t1

    @staticmethod
    def get_assetstore_from_store_uri(
        store_uri: str, context: Optional[str] = None, force: bool = False
    ) -> Assetstore:
        """Get the assetstore from store uri."""
        uri = URI.get_uri(store_uri)
        store_asset = Asset(uri, context)
        store_asset_path = store_asset.sync(force=force)
        return Assetstore.load_asset_store(store_asset_path, context, secrets=uri.get_secrets())

    def get_asset_type(self) -> ArtifactType:
        """Get the asset type."""
        assetstore = self.get_assetstore(asset=self)
        atype = assetstore.get_asset_type(self.uri)
        if atype is None:
            return ArtifactType.UNDEFINED
        return atype
