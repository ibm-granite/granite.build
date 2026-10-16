# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Github service for api-bridge."""

import logging
import os
import re
from typing import Any

import requests

from api_bridge.utils import get_gb_token, is_gb_enabled

logger = logging.getLogger(__name__)


def get_github_api_url() -> str:
    """Return the GitHub REST API base URL derived from GITHUB_HOST.

    Public GitHub uses the ``api.github.com`` host; GitHub Enterprise Server
    serves the v3 API under ``<host>/api/v3``. Defaults to public GitHub.
    """
    host = os.getenv("GITHUB_HOST", "github.com").strip().rstrip("/")
    if host == "github.com":
        return "https://api.github.com"
    return f"https://{host}/api/v3"


class GitHubService:
    def __init__(self):
        pass

    async def get_gb_user_details(self, build_id) -> dict[str, Any]:
        if not is_gb_enabled():
            return {"message": "GB Token not found"}
        try:
            return self.get_user_email_by_build_id(build_id)
        except Exception as e:
            logger.error(f"Exception occured in get_gb_user_details: {e}", exc_info=True)
            return {"error": str(e)}

    def find_pr_by_build_id(
        self,
        build_id: str,
        repo: str = "granite-dot-build/gbspace-public",
        github_token: str | None = None,
        base_api_url: str | None = None,
    ) -> dict | None:
        """
        Search for a PR in the given repo whose title contains the build_id.

        Uses the GitHub Search API to find the PR efficiently among 24k+ PRs.

        Returns:
            The matched PR dict from the search results, or None if not found.
        """
        token = github_token or get_gb_token()
        api = (base_api_url or get_github_api_url()).rstrip("/")

        response = requests.get(
            f"{api}/search/issues",
            params={
                "q": f"{build_id} repo:{repo} type:pr",
            },
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=15,
        )

        if not response.ok:
            raise Exception(f"GitHub Search API error: {response.status_code} {response.text}")

        data = response.json()
        if data.get("total_count", 0) == 0:
            return None

        for item in data.get("items", []):
            if build_id in item.get("title", ""):
                return item

        return None

    def get_user_email_by_build_id(self, build_id: str) -> dict:
        """
        Given a build_id, find the corresponding PR, extract the username
        from the title, and look up the user's email.

        Returns:
            Dict with user_email and username keys.
        """
        token = get_gb_token()

        pr = self.find_pr_by_build_id(build_id, github_token=token)
        if not pr:
            return {"error": f"No PR found for build_id {build_id}"}

        username = self.extract_username(pr["title"])
        if not username:
            return {
                "error": "Could not extract username from PR title",
                "pr_title": pr["title"],
            }

        user_details = self.get_user_details(token=token, user_name=username)
        return {"user_email": user_details.get("email"), "username": username}

    def extract_username(self, text: str) -> str | None:
        """
        Extracts the GitHub username from a PR title string.

        Example inputs:
            "run: user example-user build d7cc51ba-efd0-4ef0-bab6-19aa826745ec autotunex"
            "run: user `example-user` build `ff76a26e-e18c-44d6-9d37-15c09c0293ab` `autotunex`"

        Returns:
            The username string (e.g. "example-user"), or None if not found.
        """
        match = re.search(r"user\s+`?([^`\s]+)`?", text)
        return match.group(1) if match else None

    def get_user_details(self, token: str, user_name: str) -> dict:
        """
        Fetches user details from GitHub Enterprise Server.

        Args:
            token:     GitHub personal access token.
            user_name: GitHub username to look up.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            requests.HTTPError: If the API call fails.
        """
        url = f"{get_github_api_url()}/users/{user_name}"

        response = requests.get(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=10,
        )

        if not response.ok:
            raise Exception(
                f"GitHub user lookup failed for '{user_name}': "
                f"{response.status_code} {response.text}"
            )

        return response.json()
