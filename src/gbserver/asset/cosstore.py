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

"""Access assets in COS buckets."""

from typing import Dict, Self, Union

from gbcommon.uri.cos import CosURI
from gbcommon.uri.uri import URI
from gbserver.asset.assetstore import Assetstore
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class Cosstore(Assetstore):
    """A class for storing and accessing cos asset"""

    def __init__(self: Self, uri: Union[URI, str], **kwargs) -> None:
        super().__init__(uri, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def get_supported_uri_classes(cls):
        """Get the supported URI classes."""
        return [CosURI]

    def _get_config_key(self: Self, key: str, default: str) -> str:
        if self.config and isinstance(self.config.config, dict) and key in self.config.config:
            return self.config.config[key]
        return default

    def get_relpath(self: Self, uri: URI) -> str:
        """Get the relpath."""
        if not isinstance(uri, CosURI):
            raise ValueError(f"Expected a CosURI, got {type(uri)}")
        cos_md = self.get_metadata(uri)["config"]
        bucket_name = cos_md.get("cos_bucket_name") or uri.get_uri_netloc()
        full_uri = uri.get_metadata()["bucket_path"]
        if not full_uri.startswith(f"{bucket_name}/"):
            parts = full_uri.split("/", 1)
            rel_path = parts[1] if len(parts) > 1 else ""
        else:
            rel_path = full_uri[len(f"{bucket_name}/") :]
        logger.debug(f"Computed COS rel_path: {rel_path}")
        return rel_path

    def get_metadata(self, uri: URI) -> Dict:
        if not isinstance(uri, CosURI):
            raise ValueError(f"Expected a CosURI, got {type(uri)}: {uri}")
        self.cos_secret_access_key_secret_name = self._get_config_key(
            "cos_secret_access_key_secret_name", "COS_SECRET_ACCESS_KEY"
        )
        self.cos_access_key_id_secret_name = self._get_config_key(
            "cos_access_key_id_secret_name", "COS_ACCESS_KEY_ID"
        )
        self.cos_bucket_name = self._get_config_key("cos_bucket_name", "")
        self.cos_endpoint = self._get_config_key(
            "cos_endpoint", "s3.us-east.cloud-object-storage.appdomain.cloud"
        )
        self.cos_region = self._get_config_key("cos_region", "us-east")
        logger.info(
            "Using cos secret access key with config key %s from secrets",
            self.cos_secret_access_key_secret_name,
        )
        logger.info(
            "Using cos access key id with config key %s from secrets",
            self.cos_access_key_id_secret_name,
        )
        logger.info("Using cos bucket name %s", self.cos_bucket_name)
        logger.info("Using COS endpoint %s", self.cos_endpoint)
        logger.info("Using COS region %s", self.cos_region)
        return {
            "config": {
                "cos_secret_access_key_secret_name": self.cos_secret_access_key_secret_name,
                "cos_access_key_id_secret_name": self.cos_access_key_id_secret_name,
                "cos_bucket_name": self.cos_bucket_name,
                "cos_endpoint": self.cos_endpoint,
                "cos_region": self.cos_region,
            },
        }
