import os
import re

import requests

_HF_CACHE_RE = re.compile(r"^/gb-read-write/hfcache/([^/]+)/([^/]+)/[^/]+/[^/]+$")


def lakehouse_path_to_uri(path: str, strip_last_n: int = 1) -> tuple[str, str]:
    """Convert a local lakehouse mount path or HF cache path to an lh:// / hf:// URI.

    Args:
        path: Absolute path like
            /gb-lakehouse-prod-read-only/filesets/granite_dot_build/public/shared/climate/20250906T064534/climate_train.jsonl
            or an HF datasets cache path like
            /gb-read-write/hfcache/ibm-research/finance-test/<hash>/finance_train.jsonl
        strip_last_n: Number of trailing path segments to remove (default 1 removes the filename).
            Only applies to the lakehouse path format; ignored for HF cache paths, which have a
            fixed depth of org/dataset/hash/filename.

    Returns:
        Tuple of (uri, name).
        Lakehouse example:
            ("lh://prod/granite_dot_build.public/filesets/fileset_shared/climate/20250906T064534", "climate")
        HF cache example:
            ("hf:///datasets/ibm-research/finance-test", "finance-test")
    """
    path = path.strip().rstrip("/")

    match = re.match(
        r"^/gb-lakehouse-(\w+)-read-only/filesets/([^/]+)/([^/]+)/shared/(.+)$",
        path,
    )
    if match:
        env, namespace, scope, rest = match.groups()

        parts = rest.split("/")
        if strip_last_n > 0:
            parts = parts[:-strip_last_n] if strip_last_n < len(parts) else []
        if not parts:
            raise ValueError(f"Nothing left after stripping {strip_last_n} segment(s) from: {rest}")

        name = parts[0]
        remaining = "/".join(parts)
        return f"lh://{env}/{namespace}.{scope}/filesets/fileset_shared/{remaining}", name

    hf_match = _HF_CACHE_RE.match(path)
    if hf_match:
        org, dataset = hf_match.groups()
        return f"hf:///datasets/{org}/{dataset}", dataset

    raise ValueError(f"Path does not match expected lakehouse or HF cache format: {path}")


def stem_from_path(path: str) -> str:
    """Return the filename without extension from a path."""
    return os.path.splitext(os.path.basename(path))[0]


def get_github_auth_user(token: str) -> dict:
    """Fetch the authenticated GitHub Enterprise user profile."""
    url = "https://github.ibm.com/api/v3/user"
    response = requests.get(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    response.raise_for_status()
    return response.json()
