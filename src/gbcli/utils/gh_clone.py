import logging
import os
import shutil
from pathlib import Path
from typing import Any, List, Optional

import requests
from fastapi import HTTPException
from git import GitCommandError, Repo
from requests.exceptions import ConnectionError

from gbcli.utils.utils import CloneProgress, remove_prefix
from gbcommon.types.constants import get_gh_api_base

logger = logging.getLogger(__name__)

_GH_API_BASE = get_gh_api_base()

# Short timeout for the unauthenticated public tag fetch used by the CLI version
# check, which runs at the top of most commands. Keep it small so a blackholed
# network can't hang the CLI on a courtesy check (see get_public_repo_tags).
PUBLIC_REPO_TAGS_TIMEOUT_S = 5


def clone_github_repo(
    token: str,
    repo_url: str,
    repo_branch: str,
    cache_path: Path,
    clone_folder: Optional[str] = None,
    from_template: Optional[str] = None,
    update_bar=None,
):
    """
    Clone repository from a given branch
    checks out a particular path if provided clone_folder + from_template (i.e. /templates/hello-gb)
    """
    clone_url_no_prefix = remove_prefix("https://", repo_url)
    clone_url_with_creds = f"https://{token}@{clone_url_no_prefix}"

    shutil.rmtree(cache_path, ignore_errors=True)
    cache_path.mkdir(mode=0o777, parents=True, exist_ok=False)

    exception_message = None
    clone_progress = CloneProgress(update_bar)

    try:
        # Clones the repository without a HEAD checkout:
        # https://www.baeldung.com/ops/git-clone-subdirectory#using-git-checkout
        repo = Repo.clone_from(
            url=clone_url_with_creds,
            to_path=cache_path,
            progress=clone_progress,
            multi_options=["--no-checkout"],
            depth=1,
        )

        if not repo.active_branch or (repo.active_branch.name != repo_branch):
            logger.debug(
                f"Repo default branch '{repo.active_branch.name}' different from clone branch '{repo_branch}'. Using git fetch..."
            )
            repo.git.fetch("origin", f"{repo_branch}:{repo_branch}")

        if clone_folder:
            # just clone specific asset if provided
            checkout_path = (
                f"{clone_folder}/{from_template}"
                if from_template
                else f"{clone_folder}"
            )
            repo.git.checkout(repo_branch, "--", checkout_path)
        else:
            repo.git.checkout(repo_branch)
    except GitCommandError:
        shutil.rmtree(cache_path, ignore_errors=True)

        exception_message = generate_exception_message(
            clone_folder,
            from_template,
            clone_url_no_prefix,
            repo_branch,
            clone_progress.error_lines,
        )
        return None
    finally:
        # Using git checkout to clone a single subdirectory from a Git repository
        # will delete the other files and subdirectories if a commit is performed afterward.
        if clone_folder and os.path.exists(os.path.join(cache_path, ".git")):
            shutil.rmtree(os.path.join(cache_path, ".git"))
        if exception_message:
            raise Exception(exception_message)


def list_repo_tree(
    token: str, assets_org: str, assets_name: str, branch_name: str
) -> bool:
    tree_url = (
        f"{_GH_API_BASE}/repos/{assets_org}/{assets_name}/git/trees/{branch_name}"
    )

    params = {"recursive": 1}

    headers = {
        "Accept": "application/vnd.github.raw+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.get(tree_url, headers=headers, params=params)
    response.raise_for_status()
    data_obj = response.json()

    return data_obj


# TODO: cleanup after confirmed no longer needed
# def get_prs(
#     token: str,
#     space_org: str,
#     space_name: str,
#     state: str = "all",
#     sort: str = "created",
#     direction: str = "desc",
#     next_page_url: Optional[str] = None,
# ) -> Any | str:
#     """
#     Get all pull requests.
#     https://docs.github.com/en/rest/pulls/pulls?apiVersion=2022-11-28#list-pull-requests
#     """

#     if next_page_url:
#         prs_url = next_page_url
#         params = {}
#     else:
#         prs_url = f"{_GH_API_BASE}/repos/{space_org}/{space_name}/pulls"
#         params = {
#             "state": state,
#             "sort": sort,
#             "direction": direction,
#         }

#     headers = {
#         "Accept": "application/vnd.github.raw+json",
#         "Authorization": f"Bearer {token}",
#         "X-GitHub-Api-Version": "2022-11-28",
#     }

#     response = requests.get(prs_url, headers=headers, params=params)
#     response.raise_for_status()
#     data_obj = response.json()

#     return data_obj, response.links.get("next")


def get_pr_comments(
    token: str,
    space_org: str,
    space_name: str,
    pull_number: int,
    next_page_url: Optional[str] = None,
) -> List[Any] | str:
    if next_page_url:
        comments_url = next_page_url
    else:
        comments_url = f"{_GH_API_BASE}/repos/{space_org}/{space_name}/issues/{pull_number}/comments"

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.get(comments_url, headers=headers)
    response.raise_for_status()
    data_obj = response.json()

    return data_obj, response.links.get("next")


def get_forks(
    token: str,
    space_org: str,
    space_name: str,
    sort: str = "newest",
    next_page_url: Optional[str] = None,
) -> Any:

    if next_page_url:
        forks_url = next_page_url
    else:
        forks_url = f"{_GH_API_BASE}/repos/{space_org}/{space_name}/forks"

    params = {
        "sort": sort,
    }

    headers = {
        "Accept": "application/vnd.github.raw+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.get(forks_url, headers=headers, params=params)
    response.raise_for_status()
    data_obj = response.json()

    return data_obj, response.links.get("next")


def define_pr_status(state: str, merged_at: str) -> str:
    if state == "open":
        return "Under review"
    elif merged_at != None:
        return "Queued"
    else:
        return "Closed"


def generate_exception_message(
    clone_folder: str,
    from_template: str,
    clone_url_no_prefix: str,
    repo_branch: str,
    progress_error_lines: List[str] = [],
) -> str:
    folder = ""
    if clone_folder:
        checkout_folder = f"/{clone_folder}" if clone_folder else ""
        checkout_template = f"/{from_template}" if from_template else ""
        folder = f" in folder: {checkout_folder + checkout_template}"
    exception_message = f"Error: Can't access repository: {clone_url_no_prefix}{folder} in branch: {repo_branch}."

    for error in progress_error_lines:
        error = error.split(":", 1)[1].strip()
        exception_message = f"{exception_message} {error[0].upper() + error[1:]}."
    return exception_message


def download_repo_file(token: str, space_org: str, space_name: str, path: str):
    file_url = f"{_GH_API_BASE}/repos/{space_org}/{space_name}/contents/{path}"

    headers = {
        "Accept": "application/vnd.github.raw+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.get(file_url, headers=headers)
    response.raise_for_status()

    return response.text


def get_repo_tags(token: str, space_org: str, space_name: str) -> Any:
    tags_url = f"{_GH_API_BASE}/repos/{space_org}/{space_name}/git/refs/tags"

    headers = {
        "Accept": "application/vnd.github.raw+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.get(tags_url, headers=headers)
    response.raise_for_status()
    data_obj = response.json()

    return data_obj


def get_public_repo_tags(space_org: str, space_name: str) -> Any:
    """Fetch a repo's tags over unauthenticated, public HTTPS.

    Always targets public github.com (api.github.com) regardless of the configured
    enterprise domain, and sends no Authorization header. This lets callers query a
    public repo without GitHub credentials or SSH keys.
    """
    tags_url = f"https://api.github.com/repos/{space_org}/{space_name}/git/refs/tags"

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    # Request the max page size so the caller sees up to 100 tags without paginating.
    # This runs at the top of most commands, so cap the wait: a blackholed network
    # must not hang the CLI on a courtesy version check.
    params = {"per_page": 100}

    logger.debug("Fetching public repo tags (unauthenticated) from %s", tags_url)
    response = requests.get(
        tags_url, headers=headers, params=params, timeout=PUBLIC_REPO_TAGS_TIMEOUT_S
    )
    response.raise_for_status()
    data_obj = response.json()
    logger.debug("Public repo tags response: %d tag(s)", len(data_obj))

    return data_obj


def run_github_command(command, callback=None, final_command=None):
    try:
        result = command()
    except HTTPException as e:
        raise Exception(f"github returned '{e.status_code} {e.detail}'")
    except ConnectionError:
        raise Exception(
            f"Error: Unable to connect to network. Please check network connection"
        )
    finally:
        if final_command is not None:
            final_command()
    return result


def get_repo_subscription(token: str, space_org: str, space_name: str) -> Any:
    subscription_url = f"{_GH_API_BASE}/repos/{space_org}/{space_name}/subscription"

    headers = {
        "Accept": "application/vnd.github.raw+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.get(subscription_url, headers=headers)
    response.raise_for_status()
    data_obj = response.json()

    return data_obj


def ignore_repo_subscription(token: str, space_org: str, space_name: str) -> Any:
    subscription_url = f"{_GH_API_BASE}/repos/{space_org}/{space_name}/subscription"

    data = {
        "ignored": True,
    }

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    response = requests.put(subscription_url, headers=headers, json=data)
    response.raise_for_status()
    data = response.json()
    return data


def delete_repo_subscription(token: str, space_org: str, space_name: str) -> Any:
    subscription_url = f"{_GH_API_BASE}/repos/{space_org}/{space_name}/subscription"

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    response = requests.delete(subscription_url, headers=headers)
    response.raise_for_status()

    return response.text
