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
Watch the admin metadata tables for new builds, cancellations, etc.
"""

import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional, Self, Tuple

from git import Repo
from tqdm import tqdm

from gbcommon.uri.git import GitURI
from gbserver.build.space import Space
from gbserver.buildrunner.buildlogger import get_message_logger
from gbserver.buildrunner.validation import BuildValidation
from gbserver.github.myghapi import MyGHApi
from gbserver.github.utils import post_validated_build_message
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.storage.space_storage import IStoredSpaceStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.constants import (
    DEFAULT_BUILDWATCHER_COMMITTER_EMAIL,
    DEFAULT_BUILDWATCHER_COMMITTER_NAME,
    DEFAULT_DIR_PERMS,
    GBSERVER_GITHUB_TOKEN,
    MAX_PR_CREATION_TRIES,
    WORKSPACE_REPOS_DIR,
)
from gbserver.types.prwatcherconfig import get_uri_parts
from gbserver.types.status import Status
from gbserver.utils.archive import extract_archive
from gbserver.utils.filesystem import create_temp_subdir
from gbserver.utils.git_retry import git_clone_retry
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import normalize_to_filename

logger = get_logger(__name__)

BUILD_ONLY_THIS_NAME = ""


def get_progress_callback():
    """For use with git library during clone and push for indicating the progress."""
    pbar = tqdm()

    # pylint: disable=unused-argument
    def progress_callback(op_code, cur_count, max_count=None, message=""):
        pbar.total = max_count
        pbar.n = cur_count
        pbar.refresh()

    return progress_callback


class BuildSetup:
    """Provides function to prep the build for running"""

    # Constructor parameters
    gh_token: str = GBSERVER_GITHUB_TOKEN

    # Lakehouse storage
    storage: SingletonAdminStorage

    # Event to stop all threads
    stop_event: threading.Event

    # workspace dir used by the build watcher
    workspace_dir: Path

    space: Optional[Space]
    create_pr: bool

    def __init__(
        self: Self,
        workspace_dir: Path,
        event_source: str,
        gh_token: str = GBSERVER_GITHUB_TOKEN,
        stop_event: Optional[threading.Event] = None,
        create_pr: bool = True,
        space: Optional[Space] = None,
    ) -> None:
        self.gh_token = gh_token
        if stop_event is None:
            stop_event = threading.Event()
        self.stop_event = stop_event
        self.create_pr = create_pr
        self.space = space
        self.storage = get_admin_storage()
        # Allows multiple watchers or others using the default dir to run in the same file system at the same time.
        # This is good at least for parallel test runs.
        work_dir = create_temp_subdir(workspace_dir)
        self.workspace_dir = work_dir
        self.event_source = event_source

    def __get_my_gh_api(self: Self, uri: str) -> MyGHApi:
        """Get a GitHub API client from a PR url or a repo URL."""
        _, domain, owner, repo, _ = get_uri_parts(uri=uri)
        myapi = MyGHApi(
            token=self.gh_token,
            owner=owner,
            repo=repo,
            domain=domain,
        )
        return myapi

    def __create_pr_for_build_helper(
        self: Self, stored_build: StoredBuild, try_count: int = 0
    ) -> str:
        build_id = stored_build.uuid
        logger.info("creating a pull request for build id : %s", stored_build.uuid)
        space_storage: IStoredSpaceStorage = self.storage.space_storage
        stored_space = space_storage.get_by_name(stored_build.space_name)
        if stored_space is None:
            raise ValueError(f"failed to find a space for the build: {build_id}")
        repo_url = stored_space.git_repo_uri
        repo_dir, repo = self.__clone_repo_for_pr(repo_url=repo_url)
        repo_exp_dir = repo_dir / "experiments"
        if not repo_exp_dir.is_dir():
            logger.warning(
                "the experiments directory '%s' is missing, it will be created...",
                repo_exp_dir,
            )
        myapi = self.__get_my_gh_api(uri=repo_url)

        # Don't recreate the PR if already available. This can be triggered by the outer retry.
        if not stored_build.source_uri:
            build_dir = repo_exp_dir / f"build-{build_id}"
            if try_count > 0:
                build_dir = repo_exp_dir / f"build-{build_id}-try-{try_count}"
            build_archive_bytes = stored_build.load_from_build_archive()
            if not extract_archive(build_archive_bytes, build_dir):
                raise ValueError(
                    f"failed to extract archive for the build: {stored_build}"
                )
            branch_name = f"run/build-{build_id}"
            if try_count > 0:
                branch_name += f"-try-{try_count}"
            logger.info("create a new branch named: %s", branch_name)
            repo.git.checkout("-b", branch_name)
            logger.info(
                "adding the following files to commit: %s", repo.untracked_files
            )
            repo.index.add(repo.untracked_files)
            repo.config_writer().set_value(
                "user", "name", DEFAULT_BUILDWATCHER_COMMITTER_NAME
            ).release()
            repo.config_writer().set_value(
                "user", "email", DEFAULT_BUILDWATCHER_COMMITTER_EMAIL
            ).release()
            # ----------------------
            build_user = stored_build.username or "<unknown>"
            pr_title = f"run: user `{build_user}` build `{build_id}`"
            if stored_build.name:
                pr_title += f" `{stored_build.name}`"
            pr_body = f"{pr_title}\n{stored_build.description}"
            commit_msg = pr_body
            # ----------------------
            # https://stackoverflow.com/questions/39274739/how-to-disable-git-gpg-signing
            logger.info("create a commit with message: %s", commit_msg)
            subprocess.run(
                ["git", "-c", "commit.gpgsign=false", "commit", "-m", commit_msg],
                check=True,
                cwd=repo_dir,
            )
            # repo.git.commit("-m", commit_msg, "-c", "commit.gpgsign=false")
            repo.git.push("origin", branch_name)
            create_result = myapi.create_pr(
                src_branch=branch_name, title=pr_title, body=pr_body
            )
            logger.info(
                "build_id %s: created a pull request: %s", build_id, create_result
            )
            pr_id = str(create_result.number)
            # Update the build storage immediately after the build PR was created.
            # The merge GitHub API may fail due to transient error
            stored_build.source_uri = create_result.html_url
            # self.storage.build_storage.update(stored_build, create_if_not_exist=False)
            # NOTE: Unlike PRs, issues can be created directly with assignees
            # https://docs.github.com/en/rest/issues/issues?apiVersion=2022-11-28#create-an-issue
            if stored_build.username:
                logger.info("assigning to %s", stored_build.username)
                myapi.update_issue(
                    issue_id=pr_id,
                    assignees=[stored_build.username],
                    ignore_errors=True,
                )
            else:
                logger.warning("no user to assign the PR to: %s", stored_build)

        pr_id = self.__get_pr_id_from_uri(stored_build.source_uri)
        merge_result = myapi.merge_pr(pr_id=pr_id)
        logger.info("build_id %s: merged a pull request: %s", build_id, merge_result)

        return stored_build.source_uri

    def __get_pr_id_from_uri(self: Self, pr_uri: str) -> str:
        index = pr_uri.rfind("/") + 1
        if index < 0:
            raise ValueError(f"No integers found in the pr uri {pr_uri}")
        substr = pr_uri[index:]
        return substr

    def __create_pr_for_build(self: Self, stored_build: StoredBuild) -> Optional[str]:
        try_count = 0
        sleep_time = 5
        while not self.stop_event.is_set() and try_count < MAX_PR_CREATION_TRIES:
            try:
                pr_uri = self.__create_pr_for_build_helper(
                    stored_build=stored_build, try_count=try_count
                )
                return pr_uri
            except Exception as e:
                logger.error(
                    "failed to create the PR, sleep and retry, try_count: %d error: %s",
                    try_count,
                    e,
                )
                time.sleep(sleep_time)
                try_count += 1
        if self.stop_event.is_set():
            logger.warning("stop event has been set, stopping PR creation...")
        if try_count >= MAX_PR_CREATION_TRIES:
            logger.warning(
                "max retries %d exceeded, stopping PR creation...",
                MAX_PR_CREATION_TRIES,
            )
        return None

    def run(self: Self, stored_build: StoredBuild) -> Tuple[bool, dict[str, Any]]:
        """Validate the build and create its PR and log a message on success..
        Successful if the build is valid and we were able to create the PR and set it in the StoredBuild
        If the build is not valid, then its status is set to INVALID.
        NOTE: that gb_builds table is not updated to show the PR uri.  Caller must do that.

        Return: True on success and the build is ready to run or false if some error, and in both cases,
                a dictionary of updates to be made to the StoredBuild using update_fields().
                Updates include status and failure_reason (on non-success) and source_uri.
        """
        logger.info(
            f"Enter perform_build_prereqs({stored_build.uuid}, status={stored_build.status})"
        )
        updates = {}  # type: ignore[var-annotated]
        if stored_build.source_uri:  # Just in case
            logger.info(f"Exit early perform_build_prereqs({stored_build.uuid})")
            return True, updates

        # 1. Validate build and get new status
        errors = BuildValidation.validate_stored_build(
            stored_build=stored_build, space=self.space
        )
        # 2. Create a PR/Issue to inform the user (if configured)
        pr_uri = None
        if self.create_pr:
            pr_uri = self.__create_pr_for_build(stored_build=stored_build)
        # 3. Inform the user of the build status
        build_message_logger = get_message_logger(stored_build, self.event_source)
        if (
            self.create_pr and not stored_build.source_uri
        ):  # Failed to create the PR, retry later
            # We don't change the build status in case the caller wants to retry.
            build_id = stored_build.uuid
            msg = f"PR creation failed on build {build_id}"
            build_message_logger.error(msg)
            success = False  # Should reattempt
            updates["status"] = Status.FAILED
            updates["falure_reason"] = msg  # type: ignore[assignment]
            stored_build.status = Status.FAILED
            stored_build.failure_reason = msg
        elif not errors.is_valid() or (self.create_pr and not pr_uri):
            updates["status"] = Status.INVALID
            build_id = stored_build.uuid
            msg = f"Validation errors on build {build_id}:\n```\n{errors}\n```\n"
            build_message_logger.error(msg)
            stored_build.status = Status.INVALID
            success = False
        else:
            post_validated_build_message(
                build_message_logger=build_message_logger,
                stored_build=stored_build,
            )
            success = True

        if pr_uri:
            updates["source_uri"] = pr_uri  # type: ignore[assignment]

        logger.info(
            f"Exit perform_build_prereqs({stored_build.uuid} success={success})"
        )
        return success, updates

    def __clone_repo_for_pr(
        self: Self,
        repo_url: str,
        repo_branch: str = "main",
        force: bool = False,
        shallow: bool = True,
    ) -> Tuple[Path, Repo]:
        """Clone the repo fork and branch used in the PR."""
        logger.info("BuildWatcher.clone_repo_for_pr start repo_url: %s", repo_url)
        assert self.gh_token != "", "The GitHub token is empty"
        pr_html_url_parts = get_uri_parts(uri=repo_url)
        pr_owner = pr_html_url_parts[2]
        pr_repo = pr_html_url_parts[3]
        pr_clone_path = (
            self.workspace_dir
            / WORKSPACE_REPOS_DIR
            / str(threading.get_ident())
            / normalize_to_filename(pr_owner)
            / normalize_to_filename(pr_repo)
            / repo_branch
        )
        logger.info("pr_clone_path: %s", pr_clone_path)
        repo_url_no_prefix = repo_url.removeprefix("https://")
        clone_url_with_creds = "https://" + self.gh_token + "@" + repo_url_no_prefix
        clone_url_with_creds_redacted = "https://****@" + repo_url_no_prefix
        logger.info("clone url: %s", clone_url_with_creds_redacted)
        if pr_clone_path.is_dir():
            logger.info("the repo directory already exists at path: %s", pr_clone_path)
            if force:
                logger.warning("force is specified, delete and reclone")
            else:
                try:
                    repo = Repo(path=pr_clone_path)
                    logger.info(
                        "repo already cloned at path %s , pulling latest changes...",
                        pr_clone_path,
                    )
                    repo.git.reset("--hard")
                    repo.git.clean("-xdf")
                    repo.git.checkout(repo_branch)
                    actual_remote_url = repo.git.remote("get-url", "origin")
                    if actual_remote_url != clone_url_with_creds:
                        logger.warning(
                            "setting the remote to '%s' (maybe the creds changed)",
                            clone_url_with_creds_redacted,
                        )
                        repo.git.remote("set-url", "origin", clone_url_with_creds)
                    repo.git.pull()
                    return pr_clone_path, repo
                except Exception as e:
                    logger.error(
                        "delete and reclone because we failed to use the existing repo, error: %s",
                        e,
                    )
        logger.info("delete path %s", pr_clone_path)
        shutil.rmtree(pr_clone_path, ignore_errors=True)
        logger.info("create directory at path %s", pr_clone_path)
        pr_clone_path.mkdir(mode=DEFAULT_DIR_PERMS, parents=True, exist_ok=False)
        clone_kwargs = {
            "url": clone_url_with_creds,
            "to_path": pr_clone_path,
            "progress": get_progress_callback(),
            "branch": repo_branch,
            "no_checkout": True,
        }
        if shallow:
            clone_kwargs["depth"] = 1
        logger.info("cloning the repo")
        repo = self._clone_repo_with_retry(**clone_kwargs)
        repo.git.clean("-xdf")  # Not necessary, just to be sure of git status clean
        repo.git.checkout(repo_branch)
        logger.info("cloned pull request repo %s", repo)
        logger.info("BuildWatcher.clone_repo_for_pr end")
        return pr_clone_path, repo

    @git_clone_retry
    def _clone_repo_with_retry(self, url: str, to_path: Path, **kwargs) -> Repo:
        """Clone repository with retry logic."""
        return Repo.clone_from(url, to_path, **kwargs)
