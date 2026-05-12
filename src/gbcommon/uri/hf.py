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

"""URI referring to models/datasets/spaces/etc. in HuggingFace Hub"""

import os
import sys
import threading
import urllib.parse
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, List, Optional, Self

from huggingface_hub import (
    HfApi,
    repo_exists,
    revision_exists,
    scan_cache_dir,
    snapshot_download,
)

from gbcommon.types.testing import is_hf_mocked
from gbcommon.uri.uri import URI
from gbserver.types.artifact import ArtifactType
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


# Prefix applied to space names when defining the resource group name.
_GB_RG_SPACE_NAME_PREFIX = "gbspace-"
HF_HOST = "huggingface.co"
HF_URI_SCHEME = "hf"

URLSEGMENT_MODELS = "models"
URLSEGMENT_DATASETS = "datasets"
URLSEGMENT_SPACES = "spaces"
URLSEGMENT_BUCKETS = "buckets"

DEFAULT_REVISION = "main"


class HfType(StrEnum):
    """The different types of HfURI"""

    MODEL = auto()
    DATASET = auto()
    SPACE = auto()
    BUCKET = auto()


@dataclass(frozen=True)
class _HfParts:
    host: str
    owner: str
    repo: str
    revision: str
    hf_type: Optional[HfType] = None
    path_in_repo: str = ""  # path within the repo, e.g. "checkpoints/model.bin"


_HF_TYPE_TO_SEGMENT: dict[HfType, str] = {
    HfType.MODEL: URLSEGMENT_MODELS,
    HfType.DATASET: URLSEGMENT_DATASETS,
    HfType.SPACE: URLSEGMENT_SPACES,
    HfType.BUCKET: URLSEGMENT_BUCKETS,
}
_HF_SEGMENT_TO_TYPE: dict[str, HfType] = {v: k for k, v in _HF_TYPE_TO_SEGMENT.items()}

# Maps HfType to the repo_type string expected by huggingface_hub APIs.
# HfType.BUCKET has no huggingface_hub equivalent, so it is absent (callers default to "model").
_HF_TYPE_TO_REPO_TYPE: dict[HfType, str] = {
    HfType.MODEL: "model",
    HfType.DATASET: "dataset",
    HfType.SPACE: "space",
}


class HfURI(URI):
    """
    Hugging Face URI format:

    hf://[<host>/][<type>/]<owner>/<repo>[/<revision>[/<path_in_repo>]]

    Defaults:
      - host -> "huggingface.co"
      - type -> none (MODEL type can be omitted)
      - revision -> "main"
      - path_in_repo -> "" (repo root)

    When a path_in_repo is present the revision must be explicit so the
    parser can unambiguously separate revision from sub-path.

    Examples:
      # Models (type segment optional — omitting it implies MODEL)
      hf:///mistralai/Mistral-7B-Instruct-v0.3          # implicit MODEL, default host
      hf:///models/mistralai/Mistral-7B-Instruct-v0.3   # explicit MODEL, default host
      hf://huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3  # explicit host
      hf://ibm.com/models/mistralai/Mistral-7B-Instruct-v0.3         # custom host
      hf:///ibm-granite/granite-3.0-8b-instruct/v1.0    # explicit revision
      hf:///ibm-granite/granite-3.0-8b-instruct/main/config.json     # path_in_repo

      # Datasets
      hf://huggingface.co/datasets/wikitext/wikitext-103-v1          # explicit host
      hf:///datasets/org/my-dataset                                  # default host
      hf:///datasets/org/my-dataset/v2/data/train.csv  # revision + path_in_repo

      # Spaces
      hf://huggingface.co/spaces/huggingface/diffusers-gallery

      # Buckets
      hf://huggingface.co/buckets/org/test-bucket1

      # Path within repo (revision must be explicit)
      hf://huggingface.co/models/org/model/main/checkpoints/best
    """

    _thread_local = threading.local()

    def __init__(
        self: Self,
        uri: Optional[urllib.parse.ParseResult] = None,
        context: Optional[str] = None,
        secrets: Optional[dict] = None,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        self.secrets = secrets or {}
        self.config = config or {}
        self.parts: Optional[_HfParts] = None
        super().__init__(uri, context, secrets, **kwargs)

    @staticmethod
    def get_supported_schemes() -> List[str]:
        return [HF_URI_SCHEME]

    def get_metadata(self) -> Any:
        p = self._parts()
        metadata = {
            "uri": self.get_uristr(self),
            "host": (self.uri.netloc if self.uri else HF_HOST),
            "owner": p.owner,
            "repo": p.repo,
            "revision": p.revision,
            "hf_type": p.hf_type,
        }
        # Include optional config values if present
        if self.config:
            if "organization" in self.config:
                metadata["organization"] = self.config["organization"]
            if "resource_group_id" in self.config:
                metadata["resource_group_id"] = self.config["resource_group_id"]
        return metadata

    def _parts(self) -> _HfParts:
        if self.parts:
            return self.parts
        assert self.uri is not None, "self.uri is None"
        parts = self.uri.path.strip("/").split("/")
        host = self.uri.netloc or HF_HOST  # default to huggingface.co if not provided

        # If the first path segment is a known type keyword, consume it.
        # Otherwise the type defaults to MODEL (the "/models/" segment is optional).
        if parts and parts[0] in _HF_SEGMENT_TO_TYPE:
            hf_type = _HF_SEGMENT_TO_TYPE[parts[0]]
            parts = parts[1:]
        else:
            hf_type = HfType.MODEL

        if not parts or len(parts) < 2:
            raise ValueError(f"Malformed HF URI: {self.uri.geturl()}")

        owner = parts[0]
        repo = parts[1]
        if hf_type == HfType.BUCKET:
            revision = ""
            path_in_repo = "/".join(parts[2:]) if len(parts) > 2 else ""
        else:
            revision = parts[2] if len(parts) > 2 else DEFAULT_REVISION
            path_in_repo = "/".join(parts[3:]) if len(parts) > 3 else ""

        self.parts = _HfParts(
            host=host,
            owner=owner,
            repo=repo,
            revision=revision,
            hf_type=hf_type,
            path_in_repo=path_in_repo,
        )
        return self.parts

    def exists(self: Self, force: bool = False) -> bool:
        """Check whether the HF repo/bucket (and revision) actually exists on the Hub."""
        if is_hf_mocked():
            return True
        try:
            p = self._parts()
            repo_id = f"{p.owner}/{p.repo}"
            token = self._resolve_token()

            if p.hf_type == HfType.BUCKET:
                endpoint = f"https://{p.host}" if p.host != HF_HOST else None
                api = HfApi(endpoint=endpoint, token=token)
                api.bucket_info(bucket_id=repo_id)
                return True

            repo_type = None
            if p.hf_type == HfType.DATASET:
                repo_type = "dataset"
            elif p.hf_type == HfType.SPACE:
                repo_type = "space"

            if p.revision and p.revision != DEFAULT_REVISION:
                return revision_exists(
                    repo_id=repo_id,
                    revision=p.revision,
                    repo_type=repo_type,
                    token=token,
                )
            return repo_exists(repo_id=repo_id, repo_type=repo_type, token=token)
        except Exception:
            return False

    def is_accessible(self: Self) -> bool:
        return self.exists()

    def get_revision(self) -> str:
        """Return the branch/tag/ref of the repo."""
        return self._parts().revision

    def get_owner(self) -> str:
        """Return the owner of the repo."""
        return self._parts().owner

    def get_repo(self) -> str:
        """Return the repo name."""
        return self._parts().repo

    def get_host(self) -> str:
        """Return the host name."""
        return self._parts().host

    def get_hf_type(self) -> Optional[HfType]:
        """Return the HF resource type if encoded in the URI."""
        return self._parts().hf_type

    @staticmethod
    def space_name_to_resource_group_name(space_name: Optional[str]) -> str:
        """Return the HF resource group name derived from a GB space name.

        By convention, GB space names are prefixed with ``_GB_RG_SPACE_NAME_PREFIX``
        to form the resource group name used in HuggingFace Enterprise.

        Args:
            space_name: The GB space name to convert.

        Returns:
            The corresponding HF resource group name.
        """
        if not space_name:
            return ""
        return f"{_GB_RG_SPACE_NAME_PREFIX}{space_name}"

    def get_artifact_type(self) -> ArtifactType:
        """Return the artifact type based on the HF resource type encoded in the URI.

        HfType.MODEL and an unspecified type (None, which defaults to MODEL) both
        map to ArtifactType.MODEL. HfType.DATASET maps to ArtifactType.DATASET.
        HfType.BUCKET maps to ArtifactType.BUCKET. All other types (SPACE)
        return UNDEFINED.

        Returns:
            ArtifactType: The artifact type.
        """
        match self._parts().hf_type:
            case HfType.MODEL | None:
                return ArtifactType.MODEL
            case HfType.DATASET:
                return ArtifactType.DATASET
            case HfType.BUCKET:
                return ArtifactType.BUCKET
            case _:
                return ArtifactType.UNDEFINED

    def get_path_in_repo(self) -> str:
        """Return the sub-path within the repo encoded in the URI, or empty string."""
        return self._parts().path_in_repo

    def custom_str(self) -> str:
        """Return the canonical URI string, always including the type segment."""
        p = self._parts()

        # Always include the type segment so the URI round-trips through _parts()
        path_parts = []
        if p.hf_type:
            path_parts.append(_HF_TYPE_TO_SEGMENT[p.hf_type])

        path_parts.extend([p.owner, p.repo])

        # Include revision if it is non-default and non-empty, OR a path_in_repo follows it
        if p.revision and (p.revision != DEFAULT_REVISION or p.path_in_repo):
            path_parts.append(p.revision)

        if p.path_in_repo:
            path_parts.append(p.path_in_repo)

        path = "/" + "/".join(path_parts)

        # Always include host in the output
        return f"{HF_URI_SCHEME}://{p.host}{path}"

    def is_prod(self) -> bool:
        """Return True if this URI points to the canonical HuggingFace host."""
        # TODO: set correct env variable and update this check as needed
        return self.get_owner() == "ibm-granite"

    @staticmethod
    def parse(uri_str: str) -> "HfURI":
        """Factory method from a raw URI string"""
        return HfURI(urllib.parse.urlparse(uri_str))

    @classmethod
    def from_parts(
        cls,
        owner: str,
        repo: str,
        hf_type: Optional[HfType] = None,
        revision: str = DEFAULT_REVISION,
        host: str = HF_HOST,
        path_in_repo: str = "",
    ) -> "HfURI":
        """
        Create an HfURI from individual components.

        Args:
            owner: Repository owner/organization (e.g., "mistralai")
            repo: Repository name (e.g., "Mistral-7B-Instruct-v0.3")
            hf_type: Type of resource (MODEL, DATASET, SPACE) - encoded in URI if provided
            revision: Git reference (branch/tag/commit), defaults to "main"
            host: Hub host, defaults to "huggingface.co"

        Returns:
            HfURI instance

        Examples:
            # Create model URI
            uri = HfURI.from_parts(
                owner="mistralai",
                repo="Mistral-7B-Instruct-v0.3",
                hf_type=HfType.MODEL
            )
            # Result: hf://huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3

            # Create dataset URI with custom revision
            uri = HfURI.from_parts(
                owner="wikitext",
                repo="wikitext-103-v1",
                hf_type=HfType.DATASET,
                revision="v2.0"
            )
            # Result: hf://huggingface.co/datasets/wikitext/wikitext-103-v1/v2.0

            # Create bucket URI
            uri = HfURI.from_parts(
                owner="org",
                repo="test-bucket1",
                hf_type=HfType.BUCKET
            )
            # Result: hf://huggingface.co/buckets/org/test-bucket1

            # Create with custom host
            uri = HfURI.from_parts(
                owner="google",
                repo="bert-base-uncased",
                hf_type=HfType.SPACE,
                host="internal-hub.company.com"
            )
            # Result: hf://internal-hub.company.com/spaces/google/bert-base-uncased
        """
        type_segment = f"/{_HF_TYPE_TO_SEGMENT[hf_type]}" if hf_type else ""
        path = f"{type_segment}/{owner}/{repo}"
        if hf_type != HfType.BUCKET:
            if revision != DEFAULT_REVISION or path_in_repo:
                path += f"/{revision}"
        if path_in_repo:
            path += f"/{path_in_repo}"

        uri_str = f"{HF_URI_SCHEME}://{host}{path}"
        return cls.parse(uri_str)

    # Note: currently not using this (comments can be deleted after PR approval)

    # def get_repo_url_https(self, token: Optional[str] = None) -> str:
    #     """
    #     Construct the HTTPS URL for the Hugging Face repo.
    #     Example: https://huggingface.co/owner/repo
    #     """
    #     p = self._parts()
    #     base_path = f"{p.owner}/{p.repo}"

    #     url = f"https://{HF_HOST}/{base_path}"
    #     if token:
    #         url = f"https://{token}:x-oauth-basic@{HF_HOST}/{base_path}"

    #     return url

    # def get_repo_from_cache(self: Self, token: Optional[str] = None, force: bool = False) -> Path:
    #     """Shallow clone into a temp cache dir, reuse if available."""
    #     if not hasattr(self._thread_local, "repo_cache"):
    #         self._thread_local.repo_cache = Path(tempfile.mkdtemp())

    #     https_url = self.get_repo_url_https(token=token)
    #     p = self._parts()
    #     repo_cache_path = self._thread_local.repo_cache / f"{p.owner}-{p.repo}-{p.revision}"

    #     if repo_cache_path.exists() and not force:
    #         return repo_cache_path

    #     if repo_cache_path.exists():
    #         import shutil
    #         shutil.rmtree(repo_cache_path, ignore_errors=True)

    #     # Repo.clone_from(https_url, repo_cache_path, branch=p.revision, single_branch=True, depth=1)
    #     return repo_cache_path

    def _resolve_token(self) -> Optional[str]:
        """Resolve HF auth token from secrets dict or HF_TOKEN env var.

        Returns:
            Token string if found, or None if absent/blank.
        """
        token = self.secrets.get("HF_TOKEN") if self.secrets else None
        if token is None:
            token = os.getenv("HF_TOKEN")
        return token if token and token.strip() else None

    def pull(self: Self, dest: Path, force: bool = False) -> bool:
        """Download the HuggingFace repo or bucket referenced by this URI.

        For repos, uses ``huggingface_hub.snapshot_download`` with ``local_dir``
        so all repo files land directly in *dest*.  For buckets, uses
        ``HfApi.sync_bucket`` to download bucket contents to *dest*.

        Returns ``True`` immediately without network calls when
        ``GBTEST_MOCK_HF_CALLS=true``.

        Token is resolved from ``self.secrets['HF_TOKEN']`` or the ``HF_TOKEN``
        environment variable.  For non-default hosts the ``endpoint`` kwarg is
        forwarded to the Hub client.

        Args:
            dest: Local directory to download files into.
            force: Re-download even if files already exist locally.

        Returns:
            True if the download succeeded, False on any error.
        """
        if is_hf_mocked():
            return True
        try:
            p = self._parts()
            repo_id = f"{p.owner}/{p.repo}"
            endpoint = f"https://{p.host}" if p.host != HF_HOST else None
            token = self._resolve_token()

            if p.hf_type == HfType.BUCKET:
                bucket_hf_path = f"hf://buckets/{repo_id}"
                if p.path_in_repo:
                    bucket_hf_path += f"/{p.path_in_repo}"
                logger.info("Downloading HF bucket %s to %s", repo_id, dest)
                api = HfApi(endpoint=endpoint, token=token)
                api.sync_bucket(source=bucket_hf_path, dest=str(dest))
                return True

            hf_type = p.hf_type
            repo_type = (
                _HF_TYPE_TO_REPO_TYPE.get(hf_type, "model")
                if hf_type is not None
                else "model"
            )

            logger.info(
                "Downloading HF repo %s (type=%s, rev=%s) to %s",
                repo_id,
                repo_type,
                p.revision,
                dest,
            )
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=p.revision,
                local_dir=str(dest),
                token=token,
                force_download=force,
                endpoint=endpoint,
            )
            return True
        except Exception as e:
            logger.error("HF pull failed for %s: %s", self, e)
            return False

    @staticmethod
    def hfpush_step(
        uri_str: str,
        source_path: str,
        private: bool = True,
        resource_group_id: Optional[str] = None,
        resource_group_name: Optional[str] = None,
        space_name: Optional[str] = None,
        timeout_seconds: int = 3600,
    ) -> int:
        """Parse a HF URI string, push source_path to it, and return an exit code.

        Handles the push timeout via ``signal.SIGALRM`` so the process does not
        hang indefinitely if the HF API becomes unresponsive.  Intended as the
        single entry point called from the hfpush builtin step script.  Accepts
        any combination of ``resource_group_id``, ``resource_group_name``, and
        ``space_name`` for backward compatibility with older helm templates;
        resolution (and consistency verification when more than one is
        provided) happens inside :meth:`HfURI.push`.

        Args:
            uri_str: HF URI string to push to (e.g. ``hf:///owner/repo``).
            source_path: Local file or directory path to upload.
            private: Whether to create a private repo/bucket if it does not exist.
            resource_group_id: Pre-resolved HF Enterprise resource group id.
            resource_group_name: HF Enterprise resource group name.  Resolved
                to an id via the HF API when ``resource_group_id`` is not
                provided.
            space_name: GB space name.  Converted to a resource group name
                (``"gbspace-<space>"``) and then resolved to an id.
            timeout_seconds: Seconds before the push is aborted. Defaults to 3600.

        Returns:
            0 on success, 1 on failure or timeout.
        """
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError(f"HF push timed out after {timeout_seconds} seconds")

        # Helm templates pass unset values as empty strings via `default ""`.
        resource_group_id = resource_group_id or None
        resource_group_name = resource_group_name or None
        space_name = space_name or None

        uri = HfURI.parse(uri_str)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
        try:
            uri.push(
                Path(source_path),
                private=private,
                resource_group_id=resource_group_id,
                resource_group_name=resource_group_name,
                space_name=space_name,
            )
            signal.alarm(0)  # cancel alarm on success
            return 0
        except Exception as e:
            print(f"HF push failed: {e}", flush=True)
            return 1

    @staticmethod
    def hfpull_step(uri_str: str, dest: str) -> int:
        """Parse a HF URI string, pull its contents to dest, and return an exit code.

        Intended as the single entry point called from the hfpull builtin step
        script so the helm template stays minimal.

        Args:
            uri_str: HF URI string to download (e.g. ``hf:///owner/repo``).
            dest: Local filesystem path to download files into.

        Returns:
            0 on success, 1 on failure.
        """
        uri = HfURI.parse(uri_str)
        return 0 if uri.pull(Path(dest)) else 1

    def delete(self: Self) -> bool:
        """Delete the resource referenced by this URI from the HuggingFace Hub.

        For repos: if ``path_in_repo`` is set, deletes only that file;
        otherwise deletes the entire repository and cleans the local cache.
        For buckets: if ``path_in_repo`` is set, deletes that file from the
        bucket; otherwise deletes the entire bucket.

        Returns:
            True if deletion succeeded, False on any error.
        """
        if is_hf_mocked():
            return True
        p = self._parts()
        try:
            repo_id = f"{p.owner}/{p.repo}"
            endpoint = f"https://{p.host}" if p.host != HF_HOST else None
            api = HfApi(endpoint=endpoint, token=self._resolve_token())

            if p.hf_type == HfType.BUCKET:
                if p.path_in_repo:
                    logger.info(
                        "Deleting file %s from bucket %s", p.path_in_repo, repo_id
                    )
                    api.batch_bucket_files(bucket_id=repo_id, delete=[p.path_in_repo])
                else:
                    logger.info("Deleting bucket %s", repo_id)
                    api.delete_bucket(bucket_id=repo_id)
                return True

            hf_type = p.hf_type
            repo_type = (
                _HF_TYPE_TO_REPO_TYPE.get(hf_type, "model")
                if hf_type is not None
                else "model"
            )
            if p.path_in_repo:
                logger.info(
                    "Deleting file %s from %s (type=%s, rev=%s)",
                    p.path_in_repo,
                    repo_id,
                    repo_type,
                    p.revision,
                )
                api.delete_file(
                    path_in_repo=p.path_in_repo,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=p.revision,
                )
            else:
                logger.info("Deleting repo %s (type=%s)", repo_id, repo_type)
                api.delete_repo(repo_id=repo_id, repo_type=repo_type)
            self._delete_from_cache(repo_id, repo_type, p.revision)
            return True
        except Exception as e:
            logger.error("HF delete failed for %s: %s", self, e)
            return False

    def _delete_from_cache(self, repo_id: str, repo_type: str, revision: str) -> None:
        """Remove the cached entry for a specific revision from the HF hub cache.

        Only the revision matching the URI's ``revision`` ref is deleted; other
        cached revisions of the same repo are left intact.  Cache cleanup failures
        are logged as warnings and do not propagate.

        Args:
            repo_id: The repo ID in ``owner/repo`` form.
            repo_type: The repo type string (``"model"``, ``"dataset"``, ``"space"``).
            revision: The ref (branch/tag/commit) to remove from the cache.
        """
        try:
            cache_info = scan_cache_dir()
            for cached_repo in cache_info.repos:
                if (
                    cached_repo.repo_id == repo_id
                    and cached_repo.repo_type == repo_type
                ):
                    # Match revisions where the ref set contains our revision, or
                    # where the commit hash starts with the revision string (for
                    # short commit hashes).
                    hashes = [
                        rev.commit_hash
                        for rev in cached_repo.revisions
                        if revision in rev.refs or rev.commit_hash.startswith(revision)
                    ]
                    if hashes:
                        cache_info.delete_revisions(*hashes).execute()
                        logger.info(
                            "Deleted cached revision %s for %s (type=%s)",
                            revision,
                            repo_id,
                            repo_type,
                        )
                    break
        except Exception as e:
            logger.warning("Could not clean HF cache for %s: %s", repo_id, e)

    def resolve_resource_group_id(
        self: Self,
        token: Optional[str],
        resource_group_id: Optional[str] = None,
        resource_group_name: Optional[str] = None,
        space_name: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve an HF Enterprise resource group id from id, name, or space.

        Handles the full ``space_name -> resource_group_name -> resource_group_id``
        flow in one place.  Any combination of inputs is accepted as long as
        they agree:

        * If ``resource_group_id`` is given with no name or space, it is
          returned as-is (no API call).
        * If ``resource_group_name`` and/or ``space_name`` are given (without
          ``resource_group_id``), the name is resolved via the HF API.
        * If ``resource_group_id`` is given *and* a name/space is also given,
          the name/space is resolved and the result is verified to match the
          explicit id; a ``ValueError`` is raised on mismatch.
        * If both ``resource_group_name`` and ``space_name`` are given, the
          name derived from ``space_name`` must equal the explicit name.

        Args:
            token: HF auth token used to build the temporary ``HfApi``.  May be
                ``None`` for anonymous lookups, though Enterprise resource
                groups typically require authentication.
            resource_group_id: Explicit resource group id (e.g. from
                build.yaml ``store_push.config.hf.resource_group_id``).
            resource_group_name: Explicit resource group name.
            space_name: GB space name; converted to a resource group name via
                :meth:`space_name_to_resource_group_name`.

        Returns:
            The resolved resource group id, or ``None`` when no inputs were
            supplied.

        Raises:
            ValueError: If any of the provided inputs disagree, or if name/space
                resolution fails.
        """
        if not resource_group_id and not resource_group_name and not space_name:
            return None
        derived_name = (
            self.space_name_to_resource_group_name(space_name) if space_name else None
        )
        if resource_group_name and derived_name and derived_name != resource_group_name:
            raise ValueError(
                f"space-derived resource group name '{derived_name}' does not "
                f"match resource group name '{resource_group_name}'"
            )
        effective_name = resource_group_name or derived_name
        if effective_name is None:
            # Only resource_group_id was supplied; trust it.
            return resource_group_id
        p = self._parts()
        endpoint = f"https://{p.host}" if p.host != HF_HOST else None
        api = HfApi(endpoint=endpoint, token=token)
        resolved_id = self._resolve_resource_group_id(api, p.owner, effective_name)
        if not resolved_id:
            raise ValueError(
                f"Could not resolve resource group id for '{effective_name}' "
                f"in organization '{p.owner}'"
            )
        if resource_group_id and resource_group_id != resolved_id:
            raise ValueError(
                f"resource group id '{resource_group_id}' does not match the "
                f"id '{resolved_id}' resolved from '{effective_name}' in "
                f"organization '{p.owner}'"
            )
        return resolved_id

    def _resolve_resource_group_id(
        self, api: HfApi, organization: str, name: str
    ) -> Optional[str]:
        """Look up a resource group ID by name within the given organization.

        Uses the HF Hub enterprise REST API directly since ``huggingface_hub``
        does not expose a ``list_resource_groups`` helper.

        Args:
            api: Authenticated HfApi instance.
            organization: Organization namespace to search.
            name: Resource group name to look up.

        Returns:
            The resource group ID string if found, or None.
        """
        from huggingface_hub.utils._http import get_session, hf_raise_for_status

        try:
            r = get_session().get(
                f"{api.endpoint}/api/organizations/{organization}/resource-groups",
                headers=api._build_hf_headers(),
            )
            hf_raise_for_status(r)
            for group in r.json():
                if group.get("name") == name:
                    return group.get("id") or group.get("resourceGroupId")
            logger.warning(
                "Resource group '%s' not found in organization '%s'", name, organization
            )
        except Exception as e:
            logger.warning("Could not list resource groups for %s: %s", organization, e)
        return None

    def push(
        self: Self,
        src: Path,
        commit_message: str = "Upload via gbserver",
        private: Optional[bool] = None,
        resource_group_id: Optional[str] = None,
        resource_group_name: Optional[str] = None,
        space_name: Optional[str] = None,
    ) -> None:
        """Upload a local file or directory to a HuggingFace repo or bucket.

        For repos the destination path is derived from the URI's
        ``path_in_repo`` segment.  Uses ``HfApi.upload_file`` for a single
        file and ``HfApi.upload_folder`` for a directory.

        For buckets, uses ``HfApi.create_bucket`` to ensure the bucket exists,
        then ``HfApi.batch_bucket_files`` for a single file or
        ``HfApi.sync_bucket`` for a directory.  The ``commit_message`` arg is
        ignored for buckets (no Git commits).

        Args:
            src: Local path to a file or directory to upload.
            commit_message: Commit message attached to the upload (repos only).
            private: Whether to create a private repo/bucket (if creating).
            resource_group_id: Optional resource group ID for Enterprise
                access control.  May be combined with ``resource_group_name``
                or ``space_name`` as long as they agree — see
                :meth:`resolve_resource_group_id`.
            resource_group_name: Optional resource group name; resolved to an
                ID via the HF API.
            space_name: GB space name used to derive the resource group name.

        Raises:
            ValueError: If ``src`` does not exist.
            Exception: Any error from the HuggingFace Hub API is re-raised.
        """
        if is_hf_mocked():
            return
        p = self._parts()
        repo_id = f"{p.owner}/{p.repo}"
        endpoint = f"https://{p.host}" if p.host != HF_HOST else None

        api = HfApi(endpoint=endpoint, token=self._resolve_token())
        src = Path(src)
        if not src.exists():
            raise ValueError(f"{src} does not exist")

        resource_group_id = self.resolve_resource_group_id(
            token=self._resolve_token(),
            resource_group_id=resource_group_id,
            resource_group_name=resource_group_name,
            space_name=space_name,
        )

        if p.hf_type == HfType.BUCKET:
            bucket_id = repo_id
            api.create_bucket(
                bucket_id=bucket_id,
                private=private,
                resource_group_id=resource_group_id,
                exist_ok=True,
            )
            if src.is_file():
                dest_path = p.path_in_repo or src.name
                logger.info(
                    "Uploading file %s to bucket %s/%s", src, bucket_id, dest_path
                )
                api.batch_bucket_files(bucket_id=bucket_id, add=[(src, dest_path)])
            else:
                bucket_hf_path = f"hf://buckets/{bucket_id}"
                if p.path_in_repo:
                    bucket_hf_path += f"/{p.path_in_repo}"
                logger.info("Uploading folder %s to bucket %s", src, bucket_id)
                api.sync_bucket(source=str(src), dest=bucket_hf_path)
            return

        hf_type = p.hf_type
        repo_type = (
            _HF_TYPE_TO_REPO_TYPE.get(hf_type, "model")
            if hf_type is not None
            else "model"
        )

        # Create repository if it doesn't exist
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            resource_group_id=resource_group_id,
            exist_ok=True,
        )

        if src.is_file():
            dest_path = p.path_in_repo or src.name
            logger.info(
                "Uploading file %s → %s/%s (type=%s, rev=%s)",
                src,
                repo_id,
                dest_path,
                repo_type,
                p.revision,
            )
            api.upload_file(
                path_or_fileobj=src,
                path_in_repo=dest_path,
                repo_id=repo_id,
                repo_type=repo_type,
                revision=p.revision,
                commit_message=commit_message,
            )
        else:
            logger.info(
                "Uploading folder %s → %s/%s (type=%s, rev=%s)",
                src,
                repo_id,
                p.path_in_repo,
                repo_type,
                p.revision,
            )
            api.upload_folder(
                folder_path=str(src),
                path_in_repo=p.path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                revision=p.revision,
                commit_message=commit_message,
            )
