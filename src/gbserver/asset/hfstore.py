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
from gbserver.types.artifact import ArtifactType
from gbserver.types.constants import get_hf_token
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

    def get_asset_type(self, uri: URI) -> ArtifactType:
        assert isinstance(uri, HfURI)
        return uri.get_artifact_type()

    def _resolve_token(self, uri) -> Optional[str]:
        metadata = self.get_metadata(uri)
        token_key = metadata["token_secretname"]

        # explicit secrets passed to the store via store.yaml; fall back to environment.
        token = None
        if self.secrets and token_key in self.secrets:
            token = self.secrets[token_key] or None
        else:
            token = os.getenv(token_key, None)
            # Fall back to get_hf_token() if token_key is not set
            if token is None:
                token = get_hf_token()

        if token is not None and token.strip() == "":
            token = None
        return token

    @staticmethod
    def build_hfpush_step_config(
        hfuri: HfURI,
        binding_path: str,
        binding_id: str,
        hf_private: bool,
        hf_resource_group_id: Optional[str] = None,
    ) -> dict:
        """Build the hfpush_config dict with all keys required by step templates.

        Both the LSF command.sh and Helm _helpers.tpl templates reference flat keys
        (owner, repo, revision, private, binding_id) plus nested ``hf.type`` and
        ``hf.resource_group_id``.  Caller is responsible for resolving any
        ``space_name`` / ``resource_group_name`` to the id passed here — see
        :meth:`HfURI.resolve_resource_group_id`.

        Args:
            hfuri: Parsed HuggingFace URI.
            binding_path: Local path to the artifact being pushed.
            binding_id: Output binding name for artifact tracking.
            hf_private: Whether the repo should be private.
            hf_resource_group_id: Pre-resolved HF Enterprise resource group id,
                or ``None`` when no resource group applies.

        Returns:
            Dict suitable for passing as config={"hfpush_config": ...} to
            BuildTargetStepConfig.
        """
        hf_type = hfuri.get_hf_type() or "model"
        return {
            "path": binding_path,
            "uri": str(hfuri),
            "owner": hfuri.get_owner(),
            "repo": hfuri.get_repo(),
            "revision": hfuri.get_revision(),
            "private": hf_private,
            "binding_id": binding_id,
            "hf": {
                "type": hf_type,
                "private": hf_private,
                "resource_group_id": hf_resource_group_id,
            },
        }

    @staticmethod
    def build_hfpull_step_config(
        hfuri: HfURI,
        binding_path: str,
    ) -> dict:
        """Build the hfpull_config dict with all keys required by step templates.

        The LSF command.sh template passes ``--repo-type`` to
        ``huggingface-cli download`` so it does not rely on CLI auto-detection,
        which matches the explicit ``repo_type`` passed by the Python path
        (``HfURI.pull``).

        Args:
            hfuri: Parsed HuggingFace URI.
            binding_path: Local path where the asset will be cached.

        Returns:
            Dict suitable for passing as config={"hfpull_config": ...} to
            BuildTargetStepConfig.
        """
        hf_type = hfuri.get_hf_type() or "model"
        return {
            "path": binding_path,
            "uri": str(hfuri),
            "owner": hfuri.get_owner(),
            "repo": hfuri.get_repo(),
            "revision": hfuri.get_revision(),
            "hf": {
                "type": hf_type,
            },
        }
