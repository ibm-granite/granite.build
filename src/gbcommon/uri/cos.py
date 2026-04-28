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

"""S3/COS bucket URI"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Self
from urllib.parse import ParseResult

from gbcommon.uri.uri import URI

COS_SCHEME = "cos"
S3_SCHEME = "s3"


class CosURI(URI):
    """Defines support for a g.b url for entities stored in COS. The uri encodes the following
        bucket_name - the name of the COS bucket name
        format is as follows:
            <scheme_prefix>://<bucket_name>/<path>

            Example: s3://fm-openshift-training/granite-dot-build/gbxvsh3wc

    Args:
        URI (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(
        self: Self,
        uri: Optional[ParseResult] = None,
        context: Optional[str] = None,
        secrets: Optional[Dict] = None,
        config: Optional[Dict] = None,
        **kwargs: Dict,
    ) -> None:
        super().__init__(uri, context, secrets, **kwargs)
        try:
            bucket_name = self.get_uri_netloc()
            scheme = self.uri.scheme if self.uri else "<unknown>"
            if bucket_name == "":
                raise ValueError(
                    f"The bucket name cannot be empty in a '{scheme}://' URI"
                )
        except Exception as e:
            raise ValueError(f"failed to create from uri: {uri}") from e

    def get_uri_netloc(self) -> str:
        """Get the host/authority/netloc part of the URI."""
        assert self.uri is not None, "self.uri is None"
        return self.uri.netloc

    def get_metadata(self) -> Any:
        full_uri = self.get_uristr(self)
        bucket_name = self.get_uri_netloc()
        bucket_path = full_uri
        for scheme in self.get_supported_schemes():
            prefix = f"{scheme}://"
            if full_uri.startswith(prefix):
                bucket_path = full_uri[len(prefix) :]
                break
        md = {
            "uri": full_uri,
            "bucket_name": bucket_name,
            "bucket_path": bucket_path,
        }
        return md

    @staticmethod
    def get_supported_schemes() -> List[str]:
        """Return supported uri schemes as list"""
        return [COS_SCHEME, S3_SCHEME]

    def exists(self: Self, force: bool = False) -> bool:
        # TODO: Fix logic
        return False

    def is_accessible(self) -> bool:
        # TODO: Fix logic
        return False

    def pull(self: Self, dest: Path, force: bool = False) -> bool:
        # TODO: Fix logic
        raise NotImplementedError("CosURI pull is not implemented")

    def delete(self: Self) -> bool:
        # TODO: Fix logic
        raise NotImplementedError("CosURI delete is not implemented")
