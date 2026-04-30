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

"""Access assets in HuffingFace Hub."""

import os
from pathlib import Path
from typing import Dict, Optional, Self, Union

from huggingface_hub import create_repo, upload_file, upload_folder

from gbcommon.uri.hf import HfURI
from gbcommon.uri.uri import URI
from gbserver.asset.assetstore import Assetstore
from gbserver.types.constants import GBSERVER_HF_TOKEN
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class Hfstore(Assetstore):
    """
    Hugging Face Assetstore.
    Supports authentication via token (default env/secret key: 'HF_TOKEN'.

    - Auth:
        Uses an HF token
    """

    DEFAULT_TOKEN_KEY = "HF_TOKEN"

    def __init__(self: Self, uri: Union[URI, str], **kwargs) -> None:
        super().__init__(uri, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def get_supported_uri_classes(self):
        return [HfURI]

    def get_relpath(self, uri: URI) -> str:
        """
        Return a relative path for container volume binding.
        Uses owner/repo/revision; buckets have empty revision so the
        trailing segment is naturally omitted.
        """
        hf_uri = uri if isinstance(uri, HfURI) else HfURI.parse(uri)  # type: ignore[arg-type]
        p = hf_uri._parts()
        rel_path = Path(p.owner) / p.repo / p.revision
        return str(rel_path)

    def get_metadata(self, uri: URI) -> Dict:
        """
        Report which secret key name we will look for to authenticate.
        We can override this via self.config.config['token_secretname'].
        """
        token_key = (
            self.config.config["token_secretname"]
            if self.config
            and isinstance(self.config.config, dict)
            and "token_secretname" in self.config.config
            else self.DEFAULT_TOKEN_KEY
        )
        return {"token_secretname": token_key}

    def _resolve_token(self, uri) -> Optional[str]:
        metadata = self.get_metadata(uri)
        token_key = metadata["token_secretname"]

        # explicit secrets passed to the store via store.yaml; fall back to environment.
        token = None
        if self.secrets and token_key in self.secrets:
            token = self.secrets[token_key] or None
        else:
            token = os.getenv(token_key, None)
            # Fall back to GBSERVER_HF_TOKEN if token_key is not set
            if token is None:
                token = GBSERVER_HF_TOKEN

        if token is not None and token.strip() == "":
            token = None
        return token
