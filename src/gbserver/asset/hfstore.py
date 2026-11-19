#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Access assets in HuffingFace Hub."""

import os
from typing import Dict, List, Optional, Self, Union

from huggingface_hub import create_repo, upload_file, upload_folder

from gbcommon.uri.hf import HfType, HfURI
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

    def get_enterprise_organizations(self) -> Optional[List[str]]:
        """Return the Enterprise HF org names declared in store.yaml.

        Read from ``config.enterprise_organizations``. ``None`` (the key is
        absent) means *every* org is treated as Enterprise, preserving the
        behavior from before the enterprise/non-enterprise split; callers pass
        the result straight to :func:`is_enterprise_hf_org`, which encodes that
        rule.

        Returns:
            The configured org names, or ``None`` when the key is absent.

        Raises:
            ValueError: If the key is present but is not a list.
        """
        if (
            self.config
            and isinstance(self.config.config, dict)
            and "enterprise_organizations" in self.config.config
        ):
            orgs = self.config.config["enterprise_organizations"]
            if orgs is None:
                return None
            if not isinstance(orgs, list):
                raise ValueError(
                    "assetstore config 'enterprise_organizations' must be a "
                    f"list of org names, got {type(orgs).__name__}"
                )
            return [str(o) for o in orgs]
        return None

    def get_asset_type(self, uri: URI) -> ArtifactType:
        assert isinstance(uri, HfURI)
        return uri.get_artifact_type()

    def uri_exists(self, uri: URI) -> bool:
        """Check that ``uri`` exists on the Hub with this store's token.

        Resolves the token through the store (:meth:`resolve_token`, honoring
        ``token_secretname``) rather than the URI's fixed ``HF_TOKEN`` lookup, so
        existence checks use the same credential as pull/push.

        :param uri: an HfURI this store handles.
        :returns: whether it exists (True when HF is mocked).
        """
        assert isinstance(uri, HfURI), f"expected HfURI, got {type(uri)}"
        return uri.exists(token=self.resolve_token(uri))

    def resolve_token(self, uri) -> Optional[str]:
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
        ``hf.resource_group_id``.  The skypilot template additionally consumes
        ``path_in_repo`` (the LSF/Helm templates ignore it; k8s re-parses it from
        the URI inside :meth:`HfURI.hfpush_step`).  Caller is responsible for
        resolving any ``space_name`` / ``resource_group_name`` to the id passed
        here — see :meth:`HfURI.resolve_resource_group_id`.

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
            "endpoint": f"https://{hfuri.get_host()}",
            "owner": hfuri.get_owner(),
            "repo": hfuri.get_repo(),
            "revision": hfuri.get_revision(),
            # Sub-path within the repo ("" when none). Pre-resolved here — like
            # resource_group_id — so the skypilot worker's inline push needs no
            # URI parser to recover it. The k8s/lsf templates ignore this key
            # (k8s re-parses it from the URI inside HfURI.hfpush_step).
            "path_in_repo": hfuri.get_path_in_repo(),
            "private": hf_private,
            "binding_id": binding_id,
            # ``private`` lives only at the top level: every step template reads
            # ``hfpush_config.private`` (LSF command.sh, Helm _helpers.tpl,
            # skypilot step.yaml), never ``hf.private``. It is deliberately not
            # duplicated into this nested block — the k8s/skypilot overlay
            # (sanitize_hf_step_overlay + .update) rewrites ``hf.*`` from the raw
            # push config and does not re-resolve ``private``, so a nested copy
            # would be silently overwritten with the unresolved value.
            "hf": {
                "type": hf_type,
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

        The step downloads into *binding_path*. With a path in the repo, the
        input is bound at ``HfURI.local_path(binding_path)``, not at
        *binding_path* itself.

        Returns:
            Dict suitable for passing as config={"hfpull_config": ...} to
            BuildTargetStepConfig.
        """
        hf_type = hfuri.get_hf_type() or "model"
        return {
            "path": binding_path,
            "uri": str(hfuri),
            "endpoint": f"https://{hfuri.get_host()}",
            "owner": hfuri.get_owner(),
            "repo": hfuri.get_repo(),
            "revision": hfuri.get_revision(),
            # Sub-path within the repo ("" when none). The skypilot/LSF hfpull
            # scripts download only it (``--include``), into ``path`` with the
            # repo layout kept; the k8s template re-parses it from ``uri``.
            # A bucket is bound at ``path`` itself (HfURI.local_path), so it
            # gets no filter. (The scripts' ``hf download`` has no bucket repo
            # type, so they cannot pull a bucket either way.)
            "path_in_repo": (
                "" if hfuri.get_hf_type() == HfType.BUCKET else hfuri.get_path_in_repo()
            ),
            "hf": {
                "type": hf_type,
            },
        }
