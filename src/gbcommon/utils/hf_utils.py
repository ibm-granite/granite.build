"""General utility functions."""

import logging
from typing import Literal, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_ARTIFACT_TYPE_TO_SEGMENT: dict[str, str] = {
    "model": "models",
    "dataset": "datasets",
    "bucket": "buckets",
}


def get_hf_artifact_uri(
    repo_id: str,
    artifact_type: Literal["model", "dataset", "bucket"],
) -> str:
    segment = _ARTIFACT_TYPE_TO_SEGMENT[artifact_type]
    return f"hf:///{segment}/{repo_id}"


def get_huggingface_hub_url(artifact_type: str, repo_id: str) -> str:
    """Build a HuggingFace Hub URL for a model or dataset.

    Args:
        artifact_type: Type of artifact ('model' or 'dataset').
        repo_id: HuggingFace repository ID in format 'organization/repo-name'.

    Returns:
        Full URL to the artifact on HuggingFace Hub.

    Raises:
        ValueError: If artifact_type is not 'model' or 'dataset'.

    Examples:
        >>> get_huggingface_hub_url('model', 'bert-base-uncased')
        'https://huggingface.co/bert-base-uncased'

        >>> get_huggingface_hub_url('dataset', 'username/my-dataset')
        'https://huggingface.co/datasets/username/my-dataset'
    """
    if artifact_type == "model":
        return f"https://huggingface.co/{repo_id}"
    elif artifact_type == "dataset":
        return f"https://huggingface.co/datasets/{repo_id}"
    elif artifact_type == "bucket":
        return f"https://huggingface.co/buckets/{repo_id}"
    else:
        raise ValueError(
            f"Invalid artifact_type: '{artifact_type}'. Use 'model', 'dataset', or 'bucket'"
        )


def parse_hf_url(url: str) -> tuple[str, str, str]:
    """
    Convert 'https://huggingface.co/datasets/username/my-dataset'into (org, artifact_name, artifact_type)
    """
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")

    if parts[0] == "datasets":
        artifact_type = "dataset"
        org, artifact_name = parts[1:3]

    elif parts[0] == "spaces":
        artifact_type = "space"
        org, artifact_name = parts[1:3]

    else:
        # model URLs do not include "models"
        artifact_type = "model"
        org, artifact_name = parts[0:2]

    return org, artifact_name, artifact_type


def parse_hf_uri(uri: str) -> tuple[str, str, str, str | None]:
    """Parse HuggingFace URI into (org, name, artifact_type, domain).

    Handles the following URI formats:
    - hf:///mistralai/Mistral-7B-Instruct-v0.3 (implied model)
    - hf:///models/mistralai/Mistral-7B-Instruct-v0.3
    - hf:///datasets/org/name
    - hf:///spaces/org/name
    - hf://huggingface.co/datasets/org/name
    - hf://huggingface.co/spaces/org/name
    - hf://huggingface.co/models/org/name (or just org/name for models)
    - hf://ibm.com/models/org/name (or other domains)

    Args:
        uri: HuggingFace URI in one of the supported formats.

    Returns:
        Tuple of (org, name, artifact_type, domain) where:
        - org: Organization/owner name
        - name: Repository name
        - artifact_type: One of 'model', 'dataset', 'space', or 'bucket'
        - domain: Domain name if present (e.g., 'huggingface.co', 'ibm.com'), or None for implicit formats

    Raises:
        ValueError: If the URI format is not recognized or cannot be parsed.

    Examples:
        >>> parse_hf_uri("hf:///mistralai/Mistral-7B")
        ('mistralai', 'Mistral-7B', 'model', None)

        >>> parse_hf_uri("hf:///datasets/wikitext/wikitext-103-v1")
        ('wikitext', 'wikitext-103-v1', 'dataset', None)

        >>> parse_hf_uri("hf://huggingface.co/spaces/huggingface/diffusers-gallery")
        ('huggingface', 'diffusers-gallery', 'space', 'huggingface.co')

        >>> parse_hf_uri("hf://ibm.com/models/org/name")
        ('org', 'name', 'model', 'ibm.com')
    """
    if not uri.startswith("hf://"):
        raise ValueError(f"Invalid HuggingFace URI: {uri}. Must start with 'hf://'")

    # Remove hf:// prefix
    remainder = uri[5:]

    # Parse based on format pattern
    if remainder.startswith("/"):
        # Format: hf:///[type/]org/name (no domain)
        parts = remainder.lstrip("/").split("/")

        if len(parts) == 2:
            # hf:///org/name (assumed model)
            org, name = parts
            return org, name, "model", None
        elif len(parts) == 3:
            # hf:///type/org/name
            artifact_type, org, name = parts
            if artifact_type == "models":
                return org, name, "model", None
            elif artifact_type == "datasets":
                return org, name, "dataset", None
            elif artifact_type == "spaces":
                return org, name, "space", None
            elif artifact_type == "buckets":
                return org, name, "bucket", None
            else:
                raise ValueError(f"Unknown artifact type: {artifact_type}")
        else:
            raise ValueError(f"Invalid URI format: {uri}")

    elif remainder.startswith("huggingface.co/"):
        # Format: hf://huggingface.co/[type/]org/name
        path = remainder.replace("huggingface.co/", "")
        parts = path.split("/")

        if len(parts) == 2:
            # hf://huggingface.co/org/name (assumed model)
            org, name = parts
            return org, name, "model", "huggingface.co"
        elif len(parts) == 3:
            # hf://huggingface.co/type/org/name
            artifact_type, org, name = parts
            if artifact_type == "models":
                return org, name, "model", "huggingface.co"
            elif artifact_type == "datasets":
                return org, name, "dataset", "huggingface.co"
            elif artifact_type == "spaces":
                return org, name, "space", "huggingface.co"
            elif artifact_type == "buckets":
                return org, name, "bucket", "huggingface.co"
            else:
                raise ValueError(f"Unknown artifact type: {artifact_type}")
        else:
            raise ValueError(f"Invalid URI format: {uri}")

    elif "/" in remainder:
        # Format: hf://domain/[type/]org/name (custom domains like ibm.com)
        parts = remainder.split("/")

        if len(parts) == 3:
            # hf://domain/org/name (assumed model)
            domain, org, name = parts
            return org, name, "model", domain
        elif len(parts) == 4:
            # hf://domain/type/org/name
            domain, artifact_type, org, name = parts
            if artifact_type == "models":
                return org, name, "model", domain
            elif artifact_type == "datasets":
                return org, name, "dataset", domain
            elif artifact_type == "spaces":
                return org, name, "space", domain
            else:
                raise ValueError(f"Unknown artifact type: {artifact_type}")
        else:
            raise ValueError(f"Invalid URI format: {uri}")

    else:
        raise ValueError(f"Invalid HuggingFace URI format: {uri}")


def convert_hf_uri_to_url(uri: str) -> str:
    """Convert various HuggingFace URI formats to HuggingFace Hub URLs.

    Handles the following URI formats:
    - hf:///mistralai/Mistral-7B-Instruct-v0.3 (implied model)
    - hf:///models/mistralai/Mistral-7B-Instruct-v0.3
    - hf:///datasets/org/name
    - hf:///spaces/org/name
    - hf://huggingface.co/datasets/org/name
    - hf://huggingface.co/spaces/org/name
    - hf://huggingface.co/models/org/name (or just org/name for models)
    - hf://ibm.com/models/org/name (or other domains)

    Args:
        uri: HuggingFace URI in one of the supported formats.

    Returns:
        Full HuggingFace Hub URL.

    Raises:
        ValueError: If the URI format is not recognized or cannot be parsed.

    Examples:
        >>> convert_hf_uri_to_url("hf:///mistralai/Mistral-7B")
        'https://huggingface.co/mistralai/Mistral-7B'

        >>> convert_hf_uri_to_url("hf:///datasets/wikitext/wikitext-103-v1")
        'https://huggingface.co/datasets/wikitext/wikitext-103-v1'

        >>> convert_hf_uri_to_url("hf://huggingface.co/spaces/huggingface/diffusers-gallery")
        'https://huggingface.co/spaces/huggingface/diffusers-gallery'
    """
    if not uri.startswith("hf://"):
        raise ValueError(f"Invalid HuggingFace URI: {uri}. Must start with 'hf://'")

    # Remove hf:// prefix
    remainder = uri[5:]

    # Parse based on format pattern
    if remainder.startswith("/"):
        # Format: hf:///[type/]org/name
        parts = remainder.lstrip("/").split("/")

        if len(parts) == 2:
            # hf:///org/name (assumed model)
            org, name = parts
            return f"https://huggingface.co/{org}/{name}"
        elif len(parts) == 3:
            # hf:///type/org/name
            artifact_type, org, name = parts
            if artifact_type == "models":
                return f"https://huggingface.co/{org}/{name}"
            elif artifact_type == "datasets":
                return f"https://huggingface.co/datasets/{org}/{name}"
            elif artifact_type == "spaces":
                return f"https://huggingface.co/spaces/{org}/{name}"
            elif artifact_type == "buckets":
                return f"https://huggingface.co/buckets/{org}/{name}"
            else:
                raise ValueError(f"Unknown artifact type: {artifact_type}")
        else:
            raise ValueError(f"Invalid URI format: {uri}")

    elif remainder.startswith("huggingface.co/"):
        # Format: hf://huggingface.co/[type/]org/name
        path = remainder.replace("huggingface.co/", "")
        parts = path.split("/")

        if len(parts) == 2:
            # hf://huggingface.co/org/name (assumed model)
            org, name = parts
            return f"https://huggingface.co/{org}/{name}"
        elif len(parts) == 3:
            # hf://huggingface.co/type/org/name
            artifact_type, org, name = parts
            if artifact_type == "models":
                return f"https://huggingface.co/{org}/{name}"
            elif artifact_type == "datasets":
                return f"https://huggingface.co/datasets/{org}/{name}"
            elif artifact_type == "spaces":
                return f"https://huggingface.co/spaces/{org}/{name}"
            elif artifact_type == "buckets":
                return f"https://huggingface.co/buckets/{org}/{name}"
            else:
                raise ValueError(f"Unknown artifact type: {artifact_type}")
        else:
            raise ValueError(f"Invalid URI format: {uri}")

    elif "/" in remainder:
        # Format: hf://domain/[type/]org/name (custom domains like ibm.com)
        parts = remainder.split("/")

        if len(parts) == 3:
            # hf://domain/org/name (assumed model)
            org, name = parts[1:]
            return f"https://huggingface.co/{org}/{name}"
        elif len(parts) == 4:
            # hf://domain/type/org/name
            artifact_type, org, name = parts[1:]
            if artifact_type == "models":
                return f"https://huggingface.co/{org}/{name}"
            elif artifact_type == "datasets":
                return f"https://huggingface.co/datasets/{org}/{name}"
            elif artifact_type == "spaces":
                return f"https://huggingface.co/spaces/{org}/{name}"
            elif artifact_type == "buckets":
                return f"https://huggingface.co/buckets/{org}/{name}"
            else:
                raise ValueError(f"Unknown artifact type: {artifact_type}")
        else:
            raise ValueError(f"Invalid URI format: {uri}")

    else:
        raise ValueError(f"Invalid HuggingFace URI format: {uri}")


def lookup_hf_resource_group_id(
    organization: str, resource_group_name: str, token: Optional[str]
) -> Optional[str]:
    """Look up an HF Enterprise resource group id by name within an organization."""
    if not organization or not resource_group_name:
        return None
    from huggingface_hub import HfApi
    from huggingface_hub.utils._http import get_session, hf_raise_for_status

    api = HfApi(token=token)
    try:
        r = get_session().get(
            f"{api.endpoint}/api/organizations/{organization}/resource-groups",
            headers=api._build_hf_headers(),
        )
        hf_raise_for_status(r)
        for group in r.json():
            if group.get("name") == resource_group_name:
                return group.get("id") or group.get("resourceGroupId")
    except Exception as e:
        logger.warning("Could not list resource groups for %s: %s", organization, e)
    return None
