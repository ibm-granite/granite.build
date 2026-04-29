#!/usr/bin/env python3

"""
Types of pull request.
"""

from datetime import datetime
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, List, Optional, Self
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from gbserver.types.constants import PR_TITLE_DRYRUN, PR_TITLE_IGNORE


def extract_pr_id_from_url(issue_url: str) -> str:
    """Extract the pull request number from the url."""
    issue_url_obj = urlparse(issue_url)
    pr_id = issue_url_obj.path.split("/")[-1]
    return pr_id


class PullRequestRobotCommentSection(StrEnum):
    """The section in the robot comment to update."""

    VALIDATION = auto()
    EXECUTION = auto()


class PullRequestRobotCommentDataBuilderVal(BaseModel):
    """Stores the state of the builds from a single builder."""

    build_file: Optional[Path] = None  # relative path to the build file
    comment: str


class PullRequestRobotCommentDataBuilder(BaseModel):
    """Stores the state of the builds from a single builder."""

    version: str = ""
    name: Optional[str] = None  # deprecated in favor of repo_name
    repo_name: str = ""
    # commit hash -> validation data
    validation: dict[str, PullRequestRobotCommentDataBuilderVal] = Field(
        default_factory=dict
    )
    # commit hash -> execution data
    execution: dict[str, PullRequestRobotCommentDataBuilderVal] = Field(
        default_factory=dict
    )

    def model_post_init(self: Self, __context: Any) -> None:
        if self.repo_name == "" and self.name is not None:
            self.repo_name = self.name


class PullRequestRobotCommentData(BaseModel):
    """Stores the state of the builds from multiple builders."""

    builders: dict[str, PullRequestRobotCommentDataBuilder]


class PullRequestCommentUser(BaseModel):
    """Data about a single pull request comment user."""

    login: str
    id: int
    url: str
    html_url: str
    repos_url: str
    events_url: str
    received_events_url: str
    type: str  # "User"
    site_admin: bool


class PullRequestComment(BaseModel):
    """Data about a single pull request comment."""

    url: str
    html_url: str
    issue_url: str
    id: int
    user: PullRequestCommentUser
    created_at: datetime
    updated_at: datetime
    author_association: str  # "NONE"
    body: str
    # additional fields
    pr_id: str = ""

    def model_post_init(self: Self, __context: Any) -> None:
        self.pr_id = extract_pr_id_from_url(self.issue_url)


class PullRequestLicense(BaseModel):
    """Pull request license."""

    key: str = ""
    name: str = ""
    spdx_id: str = ""
    url: str = ""


class PullRequestUser(BaseModel):
    """Pull request user."""

    login: str
    id: int
    avatar_url: str = ""
    url: str
    html_url: str
    gists_url: str
    subscriptions_url: str
    organizations_url: str
    repos_url: str
    events_url: str
    received_events_url: str
    type: str  # "User"
    site_admin: bool


class PullRequestRepo(BaseModel):
    """Pull request repo."""

    id: int
    name: str
    full_name: str
    private: bool
    owner: PullRequestUser
    html_url: str
    description: Optional[str] = None
    fork: bool
    url: str
    forks_url: str
    issue_events_url: str
    events_url: str
    branches_url: str
    tags_url: str
    blobs_url: str
    commits_url: str
    git_commits_url: str
    comments_url: str
    issue_comment_url: str
    contents_url: str
    compare_url: str
    merges_url: str
    archive_url: str
    downloads_url: str
    issues_url: str
    pulls_url: str
    labels_url: str
    releases_url: str
    created_at: datetime
    updated_at: datetime
    pushed_at: datetime
    git_url: str
    ssh_url: str
    clone_url: str
    svn_url: Optional[str] = None
    homepage: Optional[str] = None
    size: Optional[int] = None
    stargazers_count: Optional[int] = None
    watchers_count: Optional[int] = None
    language: Optional[str] = None
    has_issues: bool
    has_pages: bool
    forks_count: int
    mirror_url: Optional[str] = None
    archived: bool
    disabled: bool
    open_issues_count: int
    license: Optional[PullRequestLicense] = None
    allow_forking: bool
    is_template: bool
    visibility: str
    forks: int
    open_issues: int
    watchers: int
    default_branch: str


class PullRequestLinksLinkAgain(BaseModel):
    """Pull request links link."""

    href: str = ""


class PullRequestLinksAgain(BaseModel):
    """Pull request links."""

    self: PullRequestLinksLinkAgain
    html: PullRequestLinksLinkAgain
    issue: PullRequestLinksLinkAgain
    comments: PullRequestLinksLinkAgain
    review_comments: PullRequestLinksLinkAgain
    review_comment: PullRequestLinksLinkAgain
    commits: PullRequestLinksLinkAgain
    statuses: PullRequestLinksLinkAgain


class PullRequestBase(BaseModel):
    """Pull request base."""

    label: str
    ref: str
    sha: str
    user: PullRequestUser
    repo: Optional[PullRequestRepo] = None


class PullRequest(BaseModel):
    """Pull request."""

    url: str
    id: int
    html_url: str
    diff_url: str
    patch_url: str
    issue_url: str
    number: int
    state: str
    title: str
    user: PullRequestUser
    body: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime] = None
    merged_at: Optional[datetime] = None
    merge_commit_sha: Optional[str] = None
    draft: bool
    commits_url: str
    comments_url: str
    statuses_url: str
    head: PullRequestBase
    base: PullRequestBase
    _links: Optional[PullRequestLinksAgain] = None
    comments: Optional[int] = None
    # add some new fields
    files: Optional[List[Path]] = None
    downloaded_comments: Optional[List[PullRequestComment]] = None
    should_ignore: bool = False
    should_dryrun: bool = False
    my_pr_id: str = ""

    def model_post_init(self: Self, __context: Any) -> None:
        pr_title = self.title.lower()
        self.should_ignore = pr_title.startswith(PR_TITLE_IGNORE)
        self.should_dryrun = pr_title.startswith(PR_TITLE_DRYRUN)
        self.my_pr_id = str(self.number)


class InvalidPullRequestComments(Exception):
    """The robot comments in the pull request are invalid."""

    def __init__(self: Self, pr_id: str) -> None:
        self.pr_id = pr_id
        message = f"""
Found multiple robot comments that contain metadata.
Ignoring the pull request {self.pr_id} as there
should only be one robot comment with metadata.
If you really want this pull request to be considered
then please delete all the robot comments in the pull request
and then restart the LLM.Build pipeline.
"""
        super().__init__(message)


class MergedPullRequestResponse(BaseModel):
    """The response after a PR is merged."""

    sha: str
    merged: bool
    message: str


class CreatedPullRequestResponse(BaseModel):
    """The response after a PR is created."""

    number: int
    html_url: str
