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

"""
A custom implementation of the GitHub API.
In some cases the library implementation seems to be failing.
"""

import json
import shutil
import traceback
from base64 import b64decode
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Self

import requests
from pydantic import BaseModel

from gbserver.types.constants import (
    DEFAULT_GH_API_ENDPOINT,
    DEFAULT_GH_REQUEST_TIMEOUT,
    SPACE_REPO_BUILD_BRANCH_NAME,
)
from gbserver.types.pullrequest import (
    CreatedPullRequestResponse,
    MergedPullRequestResponse,
    PullRequest,
    PullRequestComment,
)
from gbserver.utils.git_retry import github_api_retry
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import download_file

logger = get_logger(__name__)


class RepoContentsPathGHResponse(BaseModel):
    """
    Contents of a particular path in the repo.
    """

    name: str
    path: Path
    sha: str
    size: int
    url: str
    html_url: str
    git_url: str
    download_url: Optional[str] = None
    type: str
    _links: dict
    # content is base64 but contains newlines for some reason
    content: Optional[str] = None
    # e.g. encoding="base64"
    encoding: Optional[str] = None


class RepoContentsGHResponse(BaseModel):
    """
    Response from GitHub about the repo contents.
    """

    paths: List[RepoContentsPathGHResponse]


class PullRequestFile(BaseModel):
    """Changes/Diffs to a file inside a pull request."""

    sha: str
    filename: str  # "file1.txt"
    status: str  # "added"
    additions: int = 0  # 103
    deletions: int = 0  # 21
    changes: int = 0  # 124
    blob_url: str = ""
    raw_url: str = ""
    contents_url: str = ""
    patch: str = ""


class MyGHApi:
    """GitHub API client."""

    # class attributes
    # owner/repo -> exists
    cache_repo_exists: Dict[str, bool] = {}
    # instance attributes
    token: str
    owner: str
    repo: str
    gh_api_endpoint: str
    timeout: int

    def __init__(
        self: Self,
        token: str,
        owner: str,
        repo: str,
        gh_api_endpoint: str = DEFAULT_GH_API_ENDPOINT,
        domain: str = "",
        timeout: int = DEFAULT_GH_REQUEST_TIMEOUT,
    ) -> None:
        assert isinstance(token, str), f"invalid GH token: {token}"
        if token == "":
            raise ValueError("empty GH token")
        self.token = token
        self.owner = owner
        self.repo = repo
        self.gh_api_endpoint = gh_api_endpoint
        self.timeout = timeout
        if domain != "":
            self.gh_api_endpoint = f"https://api.{domain}"
        logger.info("using gh_api_endpoint: %s", self.gh_api_endpoint)
        if not self.is_repo_present():
            raise ValueError(f"the repo {self.owner}/{self.repo} doesn't exist")

    @github_api_retry
    def update_issue_comment(self: Self, body: str, pr_id: str = "", comment_id: str = "") -> None:
        """
        Update a comment in an issue/pull request.
        https://docs.github.com/en/rest/issues/comments?apiVersion=2022-11-28#update-an-issue-comment
        If comment_id is empty we will create a new comment on the pull request pr_id.
        https://docs.github.com/en/rest/issues/comments?apiVersion=2022-11-28#create-an-issue-comment
        """
        logger.debug("start pr_id: %s comment_id: %s", pr_id, comment_id)
        api_url = (
            f"{self.gh_api_endpoint}/repos/{self.owner}/{self.repo}"
            + f"/issues/comments/{comment_id}"
        )
        if comment_id == "":
            assert pr_id != "", "either 'pr_id' or 'comment_id' must be provided"
            api_url = (
                f"{self.gh_api_endpoint}/repos/{self.owner}/{self.repo}"
                + f"/issues/{pr_id}/comments"
            )
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = json.dumps({"body": body})
        f0 = requests.post if comment_id == "" else requests.patch
        # logger.debug("api_url: %s", api_url)
        # logger.debug("params: %s", params)
        # logger.debug("headers: %s", headers)
        # send a blocking response to avoid weird race conditions
        response = f0(
            url=api_url,
            headers=headers,
            data=data,
            timeout=self.timeout,
        )
        response.raise_for_status()
        curr_data = response.json()
        logger.debug("curr_data: %s", curr_data)
        logger.debug("end pr_id: %s comment_id: %s", pr_id, comment_id)

    def get_pr(self: Self, pr_id: str) -> PullRequest:
        """
        Get a pullrequest.
        https://docs.github.com/en/rest/pulls/pulls?apiVersion=2022-11-28#get-a-pull-request
        """
        logger.debug("MyGHApi.get_pr start pr_id: %s", pr_id)
        api_url = f"{self.gh_api_endpoint}/repos/{self.owner}/{self.repo}/pulls/{pr_id}"
        data = self.get_all_pages(api_url=api_url, per_page=-1)
        assert len(data) == 1
        pr = PullRequest.model_validate(data[0])
        logger.debug("MyGHApi.get_pr end pr_id: %s", pr_id)
        return pr

    def get_pr_files(self: Self, pr_id: str) -> List[PullRequestFile]:
        """
        Get the files in a pullrequest.
        https://docs.github.com/en/rest/pulls/pulls?apiVersion=2022-11-28#list-pull-requests-files
        """
        logger.debug("start pr_id: %s", pr_id)
        api_url = f"{self.gh_api_endpoint}/repos/{self.owner}/{self.repo}/pulls/{pr_id}/files"
        data = self.get_all_pages(api_url=api_url)
        pr_files = [PullRequestFile.model_validate(x) for x in data]
        logger.debug("end pr_id: %s pr_files: %d", pr_id, len(pr_files))
        return pr_files

    def get_prs(
        self: Self,
        state: str = "all",
        sort: str = "created",
        direction: str = "asc",
        created_after: Optional[datetime] = None,
        base: str = SPACE_REPO_BUILD_BRANCH_NAME,
    ) -> List[PullRequest]:
        """
        Get all pull requests.
        https://docs.github.com/en/rest/pulls/pulls?apiVersion=2022-11-28#list-pull-requests
        """
        logger.debug("MyGHApi.get_prs start")
        api_url = f"{self.gh_api_endpoint}/repos/{self.owner}/{self.repo}/pulls"
        params = {
            "state": state,
            "sort": sort,
            "direction": direction,
        }
        if base is not None:
            params["base"] = base

        if created_after is not None:
            assert (
                direction == "desc"
            ), "sort direction must be set to 'desc' for 'created_after' to work properly"

        def stop_if_pr_created_after(x: Any) -> bool:
            if created_after is None:
                return False
            pr = PullRequest.model_validate(x)
            return pr.created_at < created_after

        data = self.get_all_pages(
            api_url=api_url, params=params, fn_stop_fetching=stop_if_pr_created_after
        )
        prs = [PullRequest.model_validate(x) for x in data]
        logger.debug("MyGHApi.get_prs end %s prs: %d", state, len(prs))
        return prs

    def get_all_comments(
        self: Self,
        pr_id: str = "",
        since: str = "",
        sort: str = "created",
        direction: str = "asc",
    ) -> List[PullRequestComment]:
        """
        Get all issue/pull request comments.
        https://docs.github.com/en/rest/issues/comments?apiVersion=2022-11-28#list-issue-comments-for-a-repository
        https://docs.github.com/en/rest/issues/comments?apiVersion=2022-11-28#list-issue-comments
        """
        logger.debug("MyGHApi.get_all_comments start")
        api_url = (
            f"{self.gh_api_endpoint}/repos/{self.owner}/{self.repo}/issues/comments"
            if pr_id == ""
            else f"{self.gh_api_endpoint}/repos/{self.owner}/{self.repo}/issues/{pr_id}/comments"
        )
        params = {
            "sort": sort,
            "direction": direction,
        }
        if since != "":
            params["since"] = since
        data = self.get_all_pages(api_url=api_url, params=params)
        comments = [PullRequestComment.model_validate(x) for x in data]
        logger.debug("MyGHApi.get_all_comments end comments: %d", len(comments))
        return comments

    @github_api_retry
    def _get_single_page(
        self: Self,
        api_url: str,
        params: dict,
        headers: dict,
    ) -> Any:
        """
        Fetch a single page from a GitHub API endpoint.
        This method is decorated with retry logic.
        """
        response = requests.get(
            api_url,
            params=params,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_all_pages(
        self: Self,
        api_url: str,
        params: Optional[dict] = None,
        per_page: int = 100,
        fn_stop_fetching: Optional[Callable[[Any], bool]] = None,
    ) -> list:
        """
        Get all pages of a paginated API endpoint.
        https://docs.github.com/en/rest/using-the-rest-api/using-pagination-in-the-rest-api?apiVersion=2022-11-28#about-pagination
        """
        logger.debug("MyGHApi.get_all_pages start")
        params = {} if params is None else params
        if per_page <= 0:
            logger.debug("'per_page' is set to disable pagination")
        data = []
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        page = 0
        if per_page > 0:
            params["per_page"] = per_page
        while True:
            if per_page > 0:
                page += 1
                params["page"] = page
            # logger.debug("api_url: %s", api_url)
            # logger.debug("params: %s", params)
            # logger.debug("headers: %s", headers)
            curr_data = self._get_single_page(api_url, params, headers)
            if per_page <= 0:
                data = [curr_data]
                break
            assert isinstance(curr_data, list)
            data.extend(curr_data)
            if len(curr_data) < per_page:
                break
            if fn_stop_fetching is not None:
                stop = False
                for x in curr_data:
                    if fn_stop_fetching(x):
                        stop = True
                        break
                if stop:
                    break
        logger.debug("MyGHApi.get_all_pages end pages: %d data: %d", page, len(data))
        return data

    @github_api_retry
    def create_pr(
        self: Self,
        src_branch: str,
        target_branch: str = "main",
        title: str = "A pull request created by G.B via API",
        body: str = "A pull request created by G.B via API",
    ) -> CreatedPullRequestResponse:
        """
        Create a pull request.
        https://docs.github.com/en/rest/pulls/pulls?apiVersion=2022-11-28#create-a-pull-request
        """
        logger.info(
            "MyGHApi.create_pr start src_branch: %s target_branch: %s title: %s body: %s",
            src_branch,
            target_branch,
            title,
            body,
        )
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        data = {
            "title": title,
            "body": body,
            "head": src_branch,
            "base": target_branch,
            "maintainer_can_modify": True,
        }
        api_url = f"{self.gh_api_endpoint}/repos/{self.owner}/{self.repo}/pulls"
        response = requests.post(url=api_url, headers=headers, json=data, timeout=self.timeout)
        response.raise_for_status()
        result = CreatedPullRequestResponse.model_validate(response.json())
        logger.debug("MyGHApi.create_pr end: %s", result)
        return result

    @github_api_retry
    def _do_merge_pr(
        self: Self,
        api_url: str,
        headers: dict,
        body: dict,
    ) -> MergedPullRequestResponse:
        """
        Execute the merge request. This method is decorated with retry logic.
        """
        response = requests.put(
            api_url,
            headers=headers,
            json=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return MergedPullRequestResponse.model_validate(response.json())

    def merge_pr(self: Self, pr_id: str) -> MergedPullRequestResponse:
        """
        Merge a pull request.
        https://docs.github.com/en/rest/pulls/pulls?apiVersion=2022-11-28#merge-a-pull-request
        The merge of a recently created pull request almost always fails.
        So we will try multiple times sleeping a few seconds in between.
        """
        logger.debug("MyGHApi.merge_pr start pr_id: %s", pr_id)
        api_url = f"{self.gh_api_endpoint}/repos/{self.owner}/{self.repo}/pulls/{pr_id}/merge"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        body = {"merge_method": "squash"}

        try:
            result = self._do_merge_pr(api_url, headers, body)
            logger.debug("MyGHApi.merge_pr end pr_id: %s", pr_id)
            return result
        except requests.HTTPError as e:
            logger.warning("got an error merging PR %s : %s", api_url, e)
            if e.response.status_code == 401:
                raise RuntimeError(
                    f"failed to merge PR {api_url}: token is invalid (401 Unauthorized)"
                ) from e
            raise RuntimeError(f"failed to merge PR {api_url}") from e
        except Exception as e:
            raise RuntimeError(f"failed to merge PR {api_url}") from e

    @github_api_retry
    def _do_update_issue(
        self: Self,
        api_url: str,
        headers: dict,
        data: dict,
    ) -> requests.Response:
        """
        Execute the issue update request. This method is decorated with retry logic.
        """
        response = requests.patch(url=api_url, headers=headers, json=data, timeout=self.timeout)
        response.raise_for_status()
        return response

    def update_issue(
        self: Self,
        issue_id: str,
        assignees: List[str],
        ignore_errors: bool = False,
    ) -> None:
        """
        Update an issue/pull request (mainly to assign it to people).
        https://docs.github.com/en/rest/issues/issues?apiVersion=2022-11-28#update-an-issue
        """
        logger.info(
            "MyGHApi.update_issue start issue_id: %s assignees: %s",
            issue_id,
            assignees,
        )
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        data = {"assignees": assignees}
        api_url = f"{self.gh_api_endpoint}/repos/{self.owner}/{self.repo}/issues/{issue_id}"
        try:
            self._do_update_issue(api_url, headers, data)
        except requests.HTTPError as e:
            if ignore_errors:
                logger.error("status_code: %s", e.response.status_code)
            else:
                raise
        logger.info("MyGHApi.update_issue end")

    def is_repo_present(self: Self, use_cache: bool = True) -> bool:
        """
        See if the repo exists.
        """
        logger.debug("MyGHApi.is_repo_present start")
        key = f"{self.owner}/{self.repo}"
        if use_cache and (key in self.cache_repo_exists):
            return self.cache_repo_exists[key]

        api_url = f"{self.gh_api_endpoint}/repos/{self.owner}/{self.repo}"

        exists = False
        try:
            self.get_all_pages(api_url=api_url, per_page=-1)
            exists = True
        except requests.HTTPError as e:
            logger.warning("got an error checking for the repo %s : %s", self.repo, e)
            if e.response.status_code == 404:
                exists = False
            elif e.response.status_code == 401:
                raise ValueError("the token is invalid (401 Unauthorized)") from e
            else:
                raise RuntimeError(f"failed to check if repo exists {self.repo}") from e
        except Exception as e:
            raise RuntimeError(f"failed to check if the repo exists {self.repo}") from e

        logger.info(
            "MyGHApi.is_repo_present end repo: %s, exists=%s",
            self.repo,
            exists,
        )
        self.cache_repo_exists[key] = exists
        return exists

    def is_branch_present(self: Self, branch_name: str) -> bool:
        """
        See if a given branch exists.
        """
        logger.debug("MyGHApi.is_branch_present start branch_name: %s", branch_name)
        assert branch_name != "", "branch_name cannot be empty"
        api_url = f"{self.gh_api_endpoint}/repos/{self.owner}/{self.repo}/branches/{branch_name}"

        exists = False
        try:
            self.get_all_pages(api_url=api_url, per_page=-1)
            exists = True
        except requests.HTTPError as e:
            logger.info("got an error checking for the branch %s : %s", branch_name, e)
            if e.response.status_code == 404:
                # TODO: what if the repo doesn't exist?
                exists = False
            elif e.response.status_code == 401:
                raise ValueError("the token is invalid (401 Unauthorized)") from e
            else:
                raise RuntimeError(f"failed to check if branch exists {branch_name}") from e
        except Exception as e:
            raise RuntimeError(f"failed to check if the branch exists {branch_name}") from e

        logger.info(
            "MyGHApi.is_branch_present end branch_name: %s, exists=%s",
            branch_name,
            exists,
        )
        return exists

    @github_api_retry
    def _fetch_repo_contents_api(
        self: Self,
        url: str,
        headers: dict,
    ) -> Any:
        """
        Fetch repo contents from GitHub API. This method is decorated with retry logic.
        """
        resp = requests.get(url=url, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def fetch_repo_contents(
        self: Self,
        repo_path: Path,
        output_dir: Path,
        raise_exceptions: bool = True,
    ) -> None:
        """
        Fetches the contents of the file/folder specified by repo_path
        https://docs.github.com/en/rest/repos/contents?apiVersion=2022-11-28#get-repository-content
        """
        logger.info(
            "fetch_repo_contents called with repo_path %s output_dir %s",
            repo_path,
            output_dir,
        )
        url = f"{self.gh_api_endpoint}/repos/{self.owner}/{self.repo}/contents/{repo_path}"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        data = self._fetch_repo_contents_api(url, headers)
        _data = data if isinstance(data, list) else [data]
        logger.debug(
            "GH fetch contents response: %s", _data
        )  # If left at info() it blows through the travis log limit
        contents = RepoContentsGHResponse.model_validate({"paths": _data})
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(exist_ok=True, parents=True)
        for content in contents.paths:
            try:
                logger.info("fetching %s from %s", content.path, content.download_url)
                if content.type == "dir":
                    sub_repo_path = content.path
                    sub_output_dir = output_dir / content.path
                    self.fetch_repo_contents(
                        repo_path=sub_repo_path,
                        output_dir=sub_output_dir,
                    )
                    continue
                if content.type != "file":
                    continue
                output_path = output_dir / content.path
                if content.content is not None and content.encoding == "base64":
                    logger.debug(
                        "content was provided as base64: %s", content.content
                    )  # If left at info() it blows through the travis log limit
                    content_bytes = b64decode(content.content.replace("\n", ""))
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    with open(output_path, "wb") as f:
                        f.write(content_bytes)
                    continue
                if content.download_url is not None:
                    logger.info("downloading content from uri: %s", content.download_url)
                    download_file(
                        url=content.download_url,
                        output_path=output_path,
                        timeout=self.timeout,
                    )
                    continue
                logger.error("the file doesn't have a download URI: %s", content)
            except Exception as e:
                if raise_exceptions:
                    raise RuntimeError(f"failed to fetch contents of {content}") from e
                logger.error(
                    "failed to fetch contents of %s , error: %s",
                    content,
                    e,
                )
                logger.error("%s", traceback.format_exc())
