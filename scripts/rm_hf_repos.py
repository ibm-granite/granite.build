#!/usr/bin/env python3
"""
Delete Hugging Face repositories in an organization whose names match a regex.

Usage:
    # Dry run (default) — lists repos that would be deleted
    python scripts/delete_hf_test_repos.py

    # Actually delete
    python scripts/delete_hf_test_repos.py --execute

    # Target datasets instead of models
    python scripts/delete_hf_test_repos.py --repo-type dataset --execute

    # Custom regex pattern
    python scripts/delete_hf_test_repos.py --pattern '^test_' --execute

    # Use a specific token
    python scripts/delete_hf_test_repos.py --token hf_xxx --execute

Requires:
    HF_TOKEN env var or --token argument with write access to the org.
"""

import argparse
import os
import re
import sys

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

ORG = "ibm-research"
DEFAULT_PATTERN = "^test_dl"
REPO_TYPES = ("model", "dataset", "space")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments with `org`, `pattern`, `token`, `repo_type`,
        `execute`, and `yes` fields.

    Raises:
        SystemExit: If --pattern is not a valid regular expression.
    """
    parser = argparse.ArgumentParser(
        description="Delete HF org repos whose names match a regex pattern."
    )
    parser.add_argument(
        "--org",
        default=ORG,
        help=f"Hugging Face organization (default: {ORG})",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Regex matched against the repo name (default: '{DEFAULT_PATTERN}')",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face API token (default: $HF_TOKEN)",
    )
    parser.add_argument(
        "--repo-type",
        dest="repo_type",
        choices=REPO_TYPES,
        default="dataset",
        help="Repository type to target (default: model)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete repos (default is dry-run)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt when --execute is set",
    )

    args = parser.parse_args()

    try:
        re.compile(args.pattern)
    except re.error as e:
        parser.error(f"Invalid --pattern regex: {e}")

    return args


def list_matching_repos(api: HfApi, repo_type: str, org: str, pattern: str) -> list[str]:
    """Return repo IDs in org of the given type whose name matches pattern.

    Args:
        api: Authenticated HfApi instance.
        repo_type: One of 'model', 'dataset', or 'space'.
        org: Hugging Face organization name.
        pattern: Regex pattern matched against the bare repo name (not the full ID).

    Returns:
        List of full repo IDs (e.g. 'ibm-research/test_dl_foo').
    """
    list_fn = {
        "model": lambda: api.list_models(author=org),
        "dataset": lambda: api.list_datasets(author=org),
        "space": lambda: api.list_spaces(author=org),
    }[repo_type]
    compiled = re.compile(pattern)
    return [r.id for r in list_fn() if compiled.search(r.id.split("/")[-1])]


def delete_repos(api: HfApi, repo_ids: list[str], repo_type: str) -> None:
    """Delete the given repos, printing success or failure for each.

    Args:
        api: Authenticated HfApi instance.
        repo_ids: List of full repo IDs to delete.
        repo_type: One of 'model', 'dataset', or 'space'.
    """
    for repo_id in repo_ids:
        try:
            api.delete_repo(repo_id=repo_id, repo_type=repo_type)
            print(f"  Deleted: {repo_id}")
        except HfHubHTTPError as e:
            print(f"  ERROR deleting {repo_id}: {e}", file=sys.stderr)


def main() -> None:
    args = parse_args()

    if not args.token:
        print("Error: provide --token or set HF_TOKEN env var.", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=args.token)

    print(f"Fetching {args.repo_type} repos in '{args.org}' matching '{args.pattern}' ...")
    matching = list_matching_repos(api, args.repo_type, args.org, args.pattern)

    if not matching:
        print("No matching repositories found.")
        return

    print(f"\nFound {len(matching)} matching repo(s):")
    for repo_id in matching:
        print(f"  {repo_id}")

    if not args.execute:
        print("\nDry run — no repos deleted. Pass --execute to delete.")
        return

    if not args.yes:
        answer = input(f"\nDelete all {len(matching)} repo(s)? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    print("\nDeleting ...")
    delete_repos(api, matching, args.repo_type)
    print("Done.")


if __name__ == "__main__":
    main()
